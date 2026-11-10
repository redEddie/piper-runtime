"""Asynchronous scene memory and grasp/release inference. No robot I/O.

Raw camera times are immutable. A predicted current state is never represented
as a fresh camera measurement, and inferred bowl contact is not task success.
"""
from collections import Counter, deque
import numpy as np

from .control100 import ControlFault
from .kinematics import _matrix
from .show100 import vector


def yaw_quaternion(yaw):
    yaw = (yaw+np.pi/4) % (np.pi/2)-np.pi/4
    return np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])


class SceneState:
    schema = 'show100_scene_state_v1'
    max_measurement_delay = 2.0
    heartbeat_timeout = 1.5
    episode_horizon = 12.0

    def __init__(self, model, environment, *, table_normal=(0., 0., 1.), table_offset=-.015):
        self.model, self.environment = model, environment
        self.normal, self.offset = np.asarray(table_normal, float), float(table_offset)
        self.history = {oid:deque(maxlen=90) for oid in ('green_block', 'bowl')}
        self.latest, self.rest, self.rest_time, self.quaternion = {}, {}, {}, np.array([1., 0, 0, 0])
        self.half = float(environment['block_half'])
        self.bowl_size = dict(rim_radius=environment['bowl_r_rim'], bottom_radius=environment['bowl_r_bottom'],
                              height=environment['bowl_height'])
        self.heartbeat = self.started = None
        self.phase, self.transition = 'stationary', None
        self.grasp_since = self.previous_width = self.previous_time = self.previous_tcp = None
        self.previous_velocity = np.zeros(3)
        self.release_position = self.release_velocity = self.release_time = None
        self.estimated_bowl_contact = False
        self.counts, self.rejected = Counter(), Counter()
        self.updated_objects = set()

    @property
    def ready(self):
        return len(self.rest) == 2

    def _stable(self, oid):
        rows = list(self.history[oid])
        if not rows:
            return None
        rows = [r for r in rows if r[0] >= rows[-1][0]-1.2]
        if len(rows) < 6 or rows[-1][0]-rows[0][0] < .8:
            return None
        points = np.array([r[1] for r in rows])
        middle = np.median(points, axis=0)
        third = max(1, len(points)//3)
        if (np.percentile(np.linalg.norm(points-middle, axis=1), 80) > .008
                or np.linalg.norm(np.median(points[:third], axis=0)-np.median(points[-third:], axis=0)) > .005):
            return None
        return middle

    def ingest(self, document, now):
        if document.get('offline_replay') or document.get('frame') != 'piper_base_link' or document.get('units') != 'm':
            raise ControlFault('Scene input must be live base-frame metre measurements')
        heartbeat = float(document['sample_time_s'])
        if not np.isfinite(heartbeat) or heartbeat > now+.01:
            raise ControlFault('Invalid perception heartbeat timestamp')
        self.heartbeat = max(heartbeat, self.heartbeat or heartbeat)
        objects = document.get('objects', [])
        if len({o['id'] for o in objects}) != len(objects):
            raise ControlFault('Duplicate scene object ID')
        for item in objects:
            oid = item['id']
            if oid not in self.history:
                continue
            stamp, frame = float(item['source_time_s']), item['source_frame']
            if not np.isfinite(stamp) or stamp > now+.01:
                raise ControlFault('Invalid object measurement timestamp')
            previous = self.latest.get(oid)
            if previous is not None and (stamp <= previous['time'] or frame == previous['frame']):
                continue
            if now-stamp > self.max_measurement_delay:
                self.rejected['measurement_too_delayed'] += 1
                continue
            position = vector(item['pos'], 3, 'scene position').copy()
            quaternion = vector(item.get('quat_wxyz', [1., 0, 0, 0]), 4, 'scene quaternion')
            if abs(np.linalg.norm(quaternion)-1) > .001:
                raise ControlFault('Invalid scene quaternion')
            if oid == 'green_block':
                half = vector(item['size']['half_extents'], 3, 'cube half size')
                if item['size']['type'] != 'box' or not np.allclose(half, self.half, atol=.001):
                    raise ControlFault('Scene cube size differs from policy')
                if self.ready and self.phase != 'stationary':
                    # A table-conditioned image estimate cannot observe a held
                    # cube's height. Log receipt but do not drag it back to table.
                    self.latest[oid] = dict(time=stamp, frame=frame, pos=position, quat=quaternion,
                                            ignored_for_state='held_or_release_prediction')
                    if (self.phase == 'released_rest' and stamp > self.release_time
                            and np.linalg.norm(position-self.rest[oid]) < .06):
                        self.history[oid].append((stamp, position, quaternion))
                        stable = self._stable(oid)
                        if stable is not None:
                            self.rest[oid], self.rest_time[oid] = stable, stamp
                            self.phase, self.transition = 'stationary', now
                            self.counts['visual_reacquisitions'] += 1
                    continue
                expected_z = (self.offset-self.normal[:2] @ position[:2])/self.normal[2]+self.half
                if abs(position[2]-expected_z) > .03:
                    self.rejected['cube_height'] += 1
                    continue
                if oid in self.rest and np.linalg.norm(position[:2]-self.rest[oid][:2]) > .12:
                    self.rejected['cube_identity_jump'] += 1
                    continue
            elif item['size']['type'] != 'bowl':
                raise ControlFault('Scene target must be a bowl')
            self.latest[oid] = dict(time=stamp, frame=frame, pos=position, quat=quaternion)
            self.history[oid].append((stamp, position, quaternion))
            # Stability is an INITIALIZATION requirement. Requiring a fresh
            # one-second stationary window for every correction kept the cube
            # at its initial location after a real 26mm movement in the run.
            initialized = oid in self.rest
            stable = position if initialized else self._stable(oid)
            if stable is not None:
                self.rest[oid] = stable.copy()
                self.rest_time[oid] = stamp
                self.updated_objects.add(oid)
                if oid == 'green_block':
                    rows = [(stamp, position, quaternion)] if initialized else [r for r in self.history[oid] if r[0] >= stamp-1.2]
                    angles = np.array([np.arctan2(_matrix(r[2])[1, 0], _matrix(r[2])[0, 0]) for r in rows])
                    self.quaternion = yaw_quaternion(np.arctan2(np.sin(4*angles).mean(), np.cos(4*angles).mean())/4)
            if oid == 'bowl':
                for key in self.bowl_size:
                    if key in item['size']:
                        value = float(item['size'][key])
                        if not np.isfinite(value) or value <= 0:
                            raise ControlFault('Invalid physical bowl dimensions')
                        self.bowl_size[key] = value
                checks = item.get('checks', {})
                if ('table_normal_base' in checks and self.started is None
                        and not checks.get('table_plane_sparse_validated', False)):
                    normal = vector(checks['table_normal_base'], 3, 'scene table normal')
                    offset = float(checks['table_plane_offset_m'])
                    if (abs(np.linalg.norm(normal)-1) > 1e-4 or normal[2] < np.cos(np.deg2rad(5.))
                            or not np.isfinite(offset) or abs(offset/normal[2]-self.offset/self.normal[2]) > .015):
                        raise ControlFault('Invalid scene support plane')
                    self.normal, self.offset = normal, offset

    def start_episode(self, now):
        if not self.ready:
            raise ControlFault('Stable cube/bowl scene not initialized')
        if any(now-self.latest[o]['time'] > 60. for o in self.rest):
            raise ControlFault('Initial scene older than 60 seconds; observe objects again')
        self.started = now
        self.counts.clear()
        self.updated_objects.clear()

    def estimate(self, robot, now, gripper_goal):
        if not self.ready:
            raise ControlFault('Waiting for one second of stable cube/bowl observations')
        if self.heartbeat is None or now-self.heartbeat > self.heartbeat_timeout:
            raise ControlFault('Perception heartbeat lost (>1.5s), not ordinary object occlusion')
        if self.started is not None and now-self.started > self.episode_horizon:
            raise ControlFault('Scene prediction episode horizon exceeded')
        feedback_time = float(robot['sample_time_s'])
        if not -.01 <= now-feedback_time <= .1:
            raise ControlFault('Scene estimator requires current robot feedback')
        fk = self.model.forward(robot['q_rad'])
        tcp = np.asarray(fk['tcp_position'], float)
        width = float(robot['gripper_full_width_m'])
        if not np.isfinite(tcp).all() or not np.isfinite(width):
            raise ControlFault('Invalid robot state for scene prediction')
        dt = now-self.previous_time if self.previous_time is not None else 0.
        velocity = (tcp-self.previous_tcp)/dt if dt > 1e-6 else self.previous_velocity
        velocity = np.clip(velocity, -.5, .5)
        side = 2*self.half
        block, bowl = self.rest['green_block'].copy(), self.rest['bowl'].copy()
        near = np.linalg.norm(tcp-block) < .04
        loose = np.linalg.norm(tcp[:2]-block[:2]) < .05 and abs(tcp[2]-block[2]) < .06
        stalled = self.previous_width is not None and abs(width-self.previous_width) < .004
        band = abs(width-side) <= .006
        if self.phase == 'stationary':
            candidate = band and gripper_goal < side-.006 and (near or (stalled and loose))
            if candidate and stalled:
                if self.grasp_since is None:
                    self.grasp_since = now
                if now-self.grasp_since >= .1:
                    self.phase, self.transition = 'held_tcp', now
            else:
                self.grasp_since = None
        if self.phase == 'held_tcp':
            if not band:
                self.phase, self.transition = 'released_prediction', now
                self.release_position, self.release_velocity, self.release_time = tcp.copy(), velocity.copy(), now
                self.history['green_block'].clear()
                self.estimated_bowl_contact = False
            else:
                block = tcp.copy()
                rotation = _matrix(fk['link6_quaternion_wxyz'])
                self.quaternion = yaw_quaternion(np.arctan2(rotation[1, 1], rotation[0, 1]))
        if self.phase == 'released_prediction':
            elapsed = max(0., now-self.release_time)
            block = self.release_position+self.release_velocity*elapsed-np.array([0., 0., 4.905*elapsed**2])
            # Predict a drop, not success. Capture only if the whole cube fits
            # within the physical rim; do not enlarge the bowl to force placement.
            radial = np.linalg.norm(block[:2]-bowl[:2])
            clear_rim = max(0., self.bowl_size['rim_radius']-np.sqrt(2)*self.half)
            plane_z = (self.offset-self.normal[:2] @ block[:2])/self.normal[2]
            inside = radial < clear_rim
            support_z = bowl[2]+.004 if inside else plane_z
            if block[2]-self.half <= support_z:
                if inside:
                    delta = block[:2]-bowl[:2]
                    radius = max(0., self.bowl_size['bottom_radius']-np.sqrt(2)*self.half)
                    block[:2] = bowl[:2]+delta*min(1., radius/max(radial, 1e-9))
                block[2] = support_z+self.half
                self.rest['green_block'] = block.copy()
                self.phase, self.transition = 'released_rest', now
                self.estimated_bowl_contact = bool(inside)
        if self.phase == 'released_rest':
            block = self.rest['green_block'].copy()
        ages = {oid:max(0., now-self.latest[oid]['time']) for oid in self.rest}
        source = ('held_tcp' if self.phase == 'held_tcp' else self.phase if self.phase.startswith('released')
                  else 'vision_update' if 'green_block' in self.updated_objects else 'stationary_memory')
        bowl_source = 'vision_update' if 'bowl' in self.updated_objects else 'stationary_memory'
        self.counts[source] += 1
        self.previous_time, self.previous_tcp, self.previous_width = now, tcp.copy(), width
        self.previous_velocity = velocity
        self.updated_objects.clear()
        return dict(block_pos_m=block, block_quat_wxyz=self.quaternion.copy(), bowl_pos_m=bowl,
                    block_half_size_m=self.half, object_age_s=max(ages.values()),
                    object_frames=tuple((oid,self.latest[oid]['frame']) for oid in self.rest),
                    support_plane=dict(normal=self.normal.copy(), offset=self.offset),
                    state_estimation=dict(schema=self.schema, state_time_s=now, cube_source=source,
                        bowl_source=bowl_source, raw_observation_age_s=ages,
                        raw_observation_time_s={o:self.latest[o]['time'] for o in self.rest},
                        stationary_reference_time_s=dict(self.rest_time),
                        ignored_observations={o:r['ignored_for_state'] for o,r in self.latest.items() if 'ignored_for_state' in r},
                        grasp_candidate=self.grasp_since is not None and self.phase == 'stationary',
                        estimated_bowl_contact=self.estimated_bowl_contact, success_verified=False,
                        phase=self.phase, rejected_measurements=dict(self.rejected)))
