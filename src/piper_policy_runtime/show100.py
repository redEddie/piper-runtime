"""SHOW-100 observation/inference adapter. No device access or motion commands.

The caller supplies the joint reference and gripper FINAL goal explicitly.
Measured aperture cannot substitute for that goal. A motion runner must supply
the matching DLS/recovering-QP controller; this adapter does not implement it.
"""
import hashlib
from pathlib import Path

import numpy as np

from .kinematics import PiperKinematics, quaternion_to_rotvec
from .policy import Policy


def observation_layout(environment):
    fields = [('joint_position_rad', 6), ('joint_velocity_rad_s', 6),
              ('servo_target_minus_joint_rad', 6), ('tcp_position_base_m', 3),
              ('tcp_rotation_vector_base_rad', 3), ('gripper_full_width_m', 1),
              ('block_position_base_m', 3), ('block_rotation_vector_base_rad', 3),
              ('bowl_position_base_m', 3), ('block_minus_tcp_m', 3),
              ('block_minus_bowl_m', 3)]
    if environment.get('stage_obs', True):
        fields.append(('stage_one_hot', 4))
    if environment.get('size_obs', False):
        fields.append(('perceived_block_half_size_m', 1))
    if environment.get('gripper_goal_obs', False):
        fields.append(('gripper_final_goal_normalized', 1))
    fields.append(('previous_normalized_policy_action', 7))
    result, offset = [], 0
    for name, width in fields:
        result.append(dict(name=name, start=offset, stop=offset + width))
        offset += width
    return result


def vector(value, length, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name}: expected finite vector of length {length}')
    return result


def task_objects(document, block_id='green_block', bowl_id='bowl'):
    """Read measured base-frame object poses without relocating them to sim.

    Origins: cube centre, bowl outer-bottom centre at the support plane.
    Geometric perception assumes upright objects resting on that plane.
    """
    if document.get('frame') != 'piper_base_link' or document.get('units') != 'm':
        raise ValueError('Object poses must use piper_base_link and metres')
    if block_id == bowl_id:
        raise ValueError('Block and bowl must be distinct')
    selected = []
    for oid in (block_id, bowl_id):
        matches = [o for o in document['objects'] if o['id'] == oid]
        if len(matches) != 1:
            raise ValueError(f'Missing or duplicate object: {oid}')
        selected.append(matches[0])
    block, bowl = selected
    if block['size']['type'] != 'box' or bowl['size']['type'] != 'bowl':
        raise ValueError('Expected a cube and a bowl')
    half = vector(block['size']['half_extents'], 3, 'half_extents')
    if (half <= 0).any() or not np.allclose(half, half[0], atol=1e-6):
        raise ValueError('SHOW-100 expects a cube')
    q = vector(block['quat_wxyz'], 4, 'block quaternion')
    if abs(np.linalg.norm(q) - 1.) > .001:
        raise ValueError('Block quaternion must have unit norm')
    return dict(block_pos_m=vector(block['pos'], 3, 'block position'),
                block_quat_wxyz=q / np.linalg.norm(q),
                bowl_pos_m=vector(bowl['pos'], 3, 'bowl position'),
                block_half_size_m=float(half[0]))


def assemble_observation(sensors, model, environment, *, joint_reference_rad,
                         gripper_final_goal_width_m, previous_raw_action):
    """Assemble the 49 actor values from valid, time-aligned input.

    Sensor timestamp/freshness validation belongs to the acquisition layer.
    joint_reference_rad must correspond to the time of the measured joints.
    No additional synthetic noise, clipping, yaw alignment or task assistance.
    """
    e = environment
    if e.get('stage_obs', True) or not e.get('size_obs') or not e.get('gripper_goal_obs'):
        raise ValueError('SHOW-100 requires stage_obs=false, size_obs and gripper_goal_obs')
    if sensors.get('frame') != 'piper_base_link' or sensors.get('units') != 'SI':
        raise ValueError('Sensors must use piper_base_link and SI')
    q = vector(sensors['q_rad'], 6, 'q')
    qvel = vector(sensors['qvel_rad_s'], 6, 'qvel')
    reference = vector(joint_reference_rad, 6, 'joint reference')
    previous = vector(previous_raw_action, 7, 'previous raw action')
    if (np.abs(previous) > 1.).any():
        raise ValueError('Previous raw action must be normalized to [-1,1]')
    width = float(sensors['gripper_full_width_m'])
    goal = float(gripper_final_goal_width_m)
    maximum = 2. * float(e['gripper_max'])
    half = float(sensors['block_half_size_m'])
    if (not np.isfinite([width, goal, maximum, half]).all() or maximum <= 0
            or not 0 <= width <= maximum + .001 or not 0 <= goal <= maximum or half <= 0):
        raise ValueError('Invalid aperture, final goal or cube size')
    block = vector(sensors['block_pos_m'], 3, 'block position')
    bowl = vector(sensors['bowl_pos_m'], 3, 'bowl position')
    quat = vector(sensors['block_quat_wxyz'], 4, 'block quaternion')
    if abs(np.linalg.norm(quat) - 1.) > .001:
        raise ValueError('Block quaternion must have unit norm')
    fk = model.forward(q)
    tcp = fk['tcp_position']
    result = np.concatenate([
        q, qvel, reference-q, tcp, quaternion_to_rotvec(fk['link6_quaternion_wxyz']),
        [width], block, quaternion_to_rotvec(quat / np.linalg.norm(quat)), bowl,
        block-tcp, block-bowl, [half, goal/maximum], previous,
    ]).astype(np.float32)
    if result.shape != (49,) or not np.isfinite(result).all():
        raise ValueError('Invalid SHOW-100 observation')
    return result


class Show100Adapter:
    """Portable actor inference with explicit controller-state inputs."""

    def __init__(self, policy_dir, model_xml):
        self.policy = Policy(policy_dir)
        meta = self.policy.metadata
        self.environment = meta['environment']
        if meta['observation_size'] != 49 or meta['action_size'] != 7:
            raise ValueError('Expected SHOW-100 49-to-7 policy')
        if meta['observation_layout'] != observation_layout(self.environment):
            raise ValueError('Observation layout mismatch')
        kin = meta['kinematics']
        if hashlib.sha256(Path(model_xml).read_bytes()).hexdigest() != kin['piper_xml_sha256']:
            raise ValueError('Policy and kinematics XML checksum mismatch')
        self.model = PiperKinematics(model_xml)
        if self.model.robot_model != self.environment['robot_model']:
            raise ValueError('Robot model mismatch')
        self.model.tcp_local_position = vector(kin['tcp_local_position_m'], 3, 'TCP')
        expected = [0., 0., self.environment['tcp_local_z']]
        if not np.allclose(self.model.tcp_local_position, expected, atol=1e-8):
            raise ValueError('Exported TCP disagrees with training configuration')

    def step(self, sensors, *, joint_reference_rad, gripper_final_goal_width_m,
             previous_raw_action, reset=False):
        obs = assemble_observation(
            sensors, self.model, self.environment, joint_reference_rad=joint_reference_rad,
            gripper_final_goal_width_m=gripper_final_goal_width_m,
            previous_raw_action=previous_raw_action)
        return dict(observation=obs, raw_action=self.policy.step(obs, reset=reset))
