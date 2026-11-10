"""SHOW-100 transports: camera subprocess, fresh feedback, explicit SDK writes."""
from pathlib import Path
import json
import hashlib
import os
import signal
import subprocess
import sys
import time
import numpy as np

from piper_policy_runtime.control100 import ControlFault
from piper_policy_runtime.show100 import task_objects


class PerceptionFailed(ControlFault):
    pass


def cpu_partition(available=None):
    cpus = sorted(os.sched_getaffinity(0) if available is None else available)
    if len(cpus) < 3:
        raise ControlFault('At least three available CPUs required for separate control/perception affinity')
    return cpus[-2:], cpus[:-2]


class PerceptionProcess:
    def __init__(self, directory, calibration, cpus, *, view=False, seconds=180., attached_directory=None):
        self.directory = Path(attached_directory) if attached_directory else Path(directory)
        self.attached = attached_directory is not None
        self.process = self.log = None
        if self.attached:
            manifest = json.loads((self.directory/'manifest.json').read_text())
            service_calib = Path(manifest.get('calibration', {}).get('path', ''))
            service_bytes = service_calib.read_bytes()
            matching_file = hashlib.sha256(service_bytes).hexdigest() == manifest.get('calibration', {}).get('sha256')
            matching_data = json.loads(service_bytes) == json.loads(Path(calibration).read_bytes())
            if manifest.get('mode') != 'live_camera_only' or not matching_file or not matching_data:
                raise PerceptionFailed('Attached perception calibration or live mode does not match this run')
            return
        raise PerceptionFailed('Start a live perception service separately and provide attached_directory; automatic camera ownership is disabled')

    def observations(self, now):
        """Nonblocking delayed/partial measurements for the scene estimator."""
        if self.process is not None and self.process.poll() is not None:
            raise PerceptionFailed('Perception exited; see perception-console.log')
        try:
            snapshot = json.loads((self.directory/'tracked_poses.json').read_text())
            if self.attached and now-float(snapshot.get('sample_time_s', 0)) > 1.0:
                raise PerceptionFailed('Attached perception heartbeat is older than 1 s')
            return snapshot
        except (OSError, ValueError):
            try:
                phase = json.loads((self.directory/'startup.json').read_text()).get('phase', 'starting')
            except (OSError, ValueError):
                phase = 'perception_process_starting'
            raise ControlFault('Waiting for perception measurements; stage='+str(phase))

    def read(self, now, max_age=.2):
        if self.process is not None and self.process.poll() is not None:
            raise PerceptionFailed('Perception exited; see perception-console.log')
        try:
            document = json.loads((self.directory/'object_poses.json').read_text())
        except (OSError, ValueError):
            try:
                phase = json.loads((self.directory/'startup.json').read_text()).get('phase', 'starting')
            except (OSError, ValueError):
                phase = 'perception_process_starting'
            raise ControlFault('Waiting for perception snapshot; stage='+str(phase))
        if document.get('offline_replay') or not document.get('complete'):
            try:
                reasons = json.loads((self.directory/'diagnostics.json').read_text()).get('object_status', {})
            except (OSError, ValueError):
                reasons = {}
            raise ControlFault('Perception incomplete (cube/bowl missing or stale): '+str(reasons))
        data = task_objects(document)
        times = np.array([o['source_time_s'] for o in document['objects']], float)
        if not np.isfinite(times).all() or np.any(now-times < -.01) or np.any(now-times > max_age):
            ages = ', '.join(f"{o['id']}={now-stamp:.3f}s" for o, stamp in zip(document['objects'], times))
            raise ControlFault(f'Perception timestamp expired or in the future: {ages}; allowed -0.010..{max_age:.3f}s')
        data['object_age_s'] = float(np.max(now-times))
        data['object_frames'] = tuple((o['id'], o['source_frame']) for o in document['objects'])
        bowl = next(o for o in document['objects'] if o['id'] == 'bowl')
        checks = bowl.get('checks', {})
        if 'table_normal_base' in checks and 'table_plane_offset_m' in checks:
            data['support_plane'] = dict(normal=checks['table_normal_base'], offset=checks['table_plane_offset_m'])
        return data

    def close(self):
        if self.attached:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                # This session contains only our camera/SAM child processes.
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3.)
        self.log.close()


class SDKRobot:
    def __init__(self, channel='can_piper'):
        from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config
        from .passive_can import PassiveCAN
        self.reader = PassiveCAN(channel)
        self.reader.start()
        if self.reader.errors:
            self.reader.stop()
            raise ControlFault('; '.join(self.reader.errors))
        config = create_agx_arm_config(
            robot=ArmModel.PIPER, firmeware_version=PiperFW.V189,
            interface='socketcan', channel=channel, bitrate=1000000,
            auto_connect=False, log_level='ERROR')
        self.joint_limits = np.array(list(config['joint_limits'].values()), float)
        self.arm = None
        self.sent_joints, self.sent_gripper, self.stops = 0, 0, 0
        try:
            self.arm = AgxArmFactory.create_arm(config)
            self.gripper = self.arm.init_effector('agx_gripper')
            self.arm.connect()
        except Exception:
            self.reader.stop()
            if self.arm is not None:
                self.arm.disconnect()
            raise

    def read(self, *, require_enabled=True, require_can=True):
        now = time.time()
        feedback = self.reader.feedback(time.monotonic_ns(), max_age_s=.1)
        if not feedback['telemetry_fresh_at_frame']:
            raise ControlFault('Missing/stale per-message joint, velocity or gripper feedback')
        status = self.arm.get_arm_status()
        drivers = [self.arm.get_driver_states(i) for i in range(1, 7)]
        if status is None or not -.01 <= now-status.timestamp <= .25:
            raise ControlFault('Arm status is missing/stale')
        if int(status.msg.arm_status) != 0 or int(status.msg.err_code) != 0:
            raise ControlFault(f'Arm status error: {status.msg.arm_status}, {status.msg.err_code}')
        if require_can and int(status.msg.ctrl_mode) != 1:
            raise ControlFault('CAN control required; exit teaching mode and check SDK read')
        enabled = self.arm.get_joints_enable_status_list()
        if any(d is None or not -.01 <= now-d.timestamp <= .5 for d in drivers):
            raise ControlFault('Driver enable status missing/stale')
        if require_enabled and not all(enabled):
            raise ControlFault('Motors disabled: use show78_sdk_demo.py enable first')
        if int(feedback['gripper_status_code']) & 0x3f:
            raise ControlFault('Gripper feedback reports a driver fault')
        grip = self.gripper.get_gripper_status()
        if grip is None or grip.msg.mode != 'width' or not -.01 <= now-grip.timestamp <= .25:
            raise ControlFault('Fresh width-mode gripper feedback required')
        components = feedback['feedback_components']
        stamps = [components[f'0x{i:03X}']['host_wall_time_ns']/1e9 for i in (0x2A5, 0x2A6, 0x2A7)]
        if max(stamps)-min(stamps) > .025:
            raise ControlFault('Joint-pair timestamps are not aligned')
        return dict(frame='piper_base_link', units='SI', q_rad=np.array(feedback['q_rad']),
                    qvel_rad_s=np.array(feedback['qvel_rad_s']),
                    gripper_full_width_m=feedback['gripper_full_width_m'],
                    joint_timestamps=np.repeat(stamps, 2), enabled=list(map(bool, enabled)),
                    velocity_timestamps_s=np.array([components[f'0x{i:03X}']['host_wall_time_ns']/1e9 for i in range(0x251,0x257)]),
                    ctrl_mode=int(status.msg.ctrl_mode), sample_time_s=now)

    def speed(self, percent):
        self.arm.set_speed_percent(percent)

    def acceleration_limits(self):
        values = []
        for index in range(1, 7):
            feedback = self.arm.get_joint_acc_limits(index, timeout=.5, min_interval=0.)
            if feedback is None:
                raise ControlFault(f'Joint{index} firmware acceleration query timed out')
            value = float(feedback.msg.max_joint_acc)
            if not np.isfinite(value) or value <= 0:
                raise ControlFault(f'Invalid firmware acceleration limit for joint{index}: {value}')
            values.append(value)
        return np.array(values)

    def send_joints(self, target):
        target = np.asarray(target, float)
        if target.shape != (6,) or not np.isfinite(target).all() or np.any(target < self.joint_limits[:, 0]-1e-4) or np.any(target > self.joint_limits[:, 1]+1e-4):
            raise ControlFault('Target outside SDK joint limits; refusing silent SDK clamping')
        # Count before I/O: a failed multi-frame transmission may be partial.
        # A failed first call may already have transmitted partial frames.
        self.sent_joints += 1
        self.arm.move_j(target.tolist())

    def send_gripper(self, width, force):
        if not np.isfinite([width, force]).all() or not 0 <= width <= .1 or not .5 <= force <= 3.:
            raise ControlFault('Invalid gripper width/force command')
        self.sent_gripper += 1
        self.gripper.move_gripper_m(value=float(width), force=float(force))

    def close(self):
        try:
            self.arm.disconnect()
        finally:
            self.reader.stop()


class MockRobot:
    """Explicit CPU test plant; never imports SDK or opens CAN/camera."""
    def __init__(self, nominal):
        self.q = np.array(nominal, float)
        self.target, self.width = self.q.copy(), .1
        self.qvel = np.zeros(6)
        self.sent_joints = self.sent_gripper = self.stops = 0

    def read(self, **kwargs):
        now = time.time()
        return dict(frame='piper_base_link', units='SI', q_rad=self.q.copy(),
                    qvel_rad_s=self.qvel.copy(), gripper_full_width_m=self.width,
                    joint_timestamps=np.full(6, now), velocity_timestamps_s=np.full(6, now),
                    enabled=[True]*6, ctrl_mode=1, sample_time_s=now)

    def advance(self, dt):
        delta = np.clip((self.target-self.q)*dt/.04, -1.5*dt, 1.5*dt)
        self.qvel = delta/dt
        self.q += delta

    def send_joints(self, target):
        self.target = np.array(target, float)
        self.sent_joints += 1

    def send_gripper(self, width, force):
        self.width = float(width)
        self.sent_gripper += 1

    def speed(self, percent):
        pass

    def acceleration_limits(self):
        return np.full(6, 5.)

    def stop(self):
        self.target = self.q.copy()
        self.stops += 1

    def close(self):
        pass


class MockVision:
    def observations(self, now):
        data = self.read(now)
        stamp = now-data['object_age_s']
        return dict(frame='piper_base_link', units='m', offline_replay=False, sample_time_s=now,
                    objects=[dict(id='green_block', pos=data['block_pos_m'].tolist(),
                        quat_wxyz=data['block_quat_wxyz'].tolist(), size=dict(type='box', half_extents=[.025]*3),
                        source_time_s=stamp, source_frame=round(stamp*30)),
                        dict(id='bowl', pos=data['bowl_pos_m'].tolist(), quat_wxyz=[1.,0,0,0],
                        size=dict(type='bowl', rim_radius=.075, bottom_radius=.045, height=.071),
                        source_time_s=stamp, source_frame=round(stamp*30))])

    def read(self, now, max_age=.2):
        return dict(block_pos_m=np.array([.30, 0., .010]), block_quat_wxyz=np.array([1., 0, 0, 0]),
                    bowl_pos_m=np.array([.3, .20, -.015]), block_half_size_m=.025,
                    object_age_s=0., object_frames=(('green_block', round(now*30)), ('bowl', round(now*30))))

    def close(self):
        pass
