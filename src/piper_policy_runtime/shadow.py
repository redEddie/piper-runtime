"""File-only vision/telemetry -> observation -> policy -> joint-target proposals.

There is deliberately no SDK, CAN socket, or command transport in this module.
Untrusted calibration and unmeasured geometry are never promoted to valid control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .kinematics import PiperKinematics, quaternion_to_rotvec
from .policy import Policy


def read_json(path):
    return json.loads(Path(path).read_text())


def file_record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def vector(value, size, name):
    if value is None or any(x is None for x in value):
        raise ValueError(f'{name}: missing feedback')
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected {size} finite values')
    return a


def assemble(vision, robot, calibration, model, environment, virtual_target=False):
    """Return a complete real snapshot observation and explicit assumptions.

    Missing physical measurements are fatal. The only optional controller value
    is an explicitly requested shadow target initialized at the measured joints.
    """
    warnings = []
    if vision.get('frame') != 'piper_base_link' or vision.get('units') != 'm':
        raise ValueError('Vision must use piper_base_link and meters')
    transform = np.asarray(calibration['T_base_cam'], dtype=float)
    exported = np.asarray(vision['camera_to_base']['T'], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform, exported, atol=1e-6, rtol=0):
        raise ValueError('Calibration and exported vision transform disagree')
    trusted = calibration.get('trusted', calibration.get('board_measured', False)) is True
    if not trusted:
        warnings.append('camera_to_base_calibration_untrusted')
    if robot.get('telemetry_fresh_at_frame') is not True:
        raise ValueError('Robot telemetry is missing or stale at the camera frame')
    warnings.append('joint_velocity_feedback_scale_requires_hardware_validation')
    q = vector(robot.get('q_rad'), 6, 'q_rad')
    qvel = vector(robot.get('qvel_rad_s'), 6, 'qvel_rad_s')
    width = robot.get('gripper_full_width_m')
    if width is None or not np.isfinite(width) or width < 0:
        raise ValueError('Missing/invalid measured full gripper width')
    if width > 2 * environment['gripper_max'] + 1e-4:
        warnings.append('measured_gripper_width_exceeds_training_range')
    if not virtual_target:
        raise ValueError('Controller target unavailable; explicitly select a virtual shadow target')
    target = q.copy()
    warnings.append('virtual_joint_target_initialized_at_measured_q_not_actual_servo_target')
    objects = {o['id']: o for o in vision['objects']}
    for role in ['block', 'bowl']:
        if role not in objects:
            raise ValueError(f'Missing task object role: {role}')
    block, bowl = objects['block'], objects['bowl']
    block_pos = vector(block.get('pos'), 3, 'block.pos')
    bowl_pos = vector(bowl.get('pos'), 3, 'bowl.pos')
    block_quat = vector(block.get('quat_wxyz'), 4, 'block.quat_wxyz')
    norm = np.linalg.norm(block_quat)
    if abs(norm - 1) > .01:
        raise ValueError('Block quaternion is not unit length')
    # Existing export rounds to five decimals. Preserve quaternion sign.
    block_quat /= norm
    size = block['size']
    if size.get('type') != 'box':
        raise ValueError('Block requires box half_extents; pointcloud size is insufficient')
    half = vector(size.get('half_extents'), 3, 'block.half_extents')
    if np.any(half <= 0) or not np.allclose(half, half[0], atol=1e-6):
        raise ValueError('This policy expects a cube with positive half size')
    if size.get('physical_dimensions_verified') is not True or bowl['size'].get('physical_dimensions_verified') is not True:
        warnings.append('physical_object_dimensions_not_verified')
    for role, position in [('block', block_pos), ('bowl', bowl_pos)]:
        for axis, value in zip(['x', 'y'], position[:2]):
            low, high = environment[f'{role}_{axis}']
            if not low <= value <= high:
                warnings.append(f'{role}_{axis}_outside_training_reset_region')
        if position[2] < -environment.get('base_height', 0.) - .005:
            warnings.append(f'{role}_below_training_table_z')
    fk = model.forward(q)
    tcp = fk['tcp_position']
    warnings.append('training_kinematics_used_real_robot_fk_tcp_not_yet_validated')
    observation = np.concatenate([
        q, qvel, target - q, tcp,
        quaternion_to_rotvec(fk['link6_quaternion_wxyz']), [width],
        block_pos, quaternion_to_rotvec(block_quat), bowl_pos,
        block_pos - tcp, block_pos - bowl_pos, [half[0]],
        np.zeros(7),  # Declared first step of a new shadow episode.
    ]).astype(np.float32)
    if observation.shape != (48,) or not np.isfinite(observation).all():
        raise ValueError('Incomplete/nonfinite policy observation')
    provenance = {
        'joint_position': 'passively received CAN feedback, rad',
        'joint_velocity': 'passively received CAN high-speed feedback, rad/s',
        'gripper_width': 'passively received full opening, m; not half opening',
        'joint_target': 'virtual shadow target initialized at measured q; never sent',
        'tcp': f'training {model.robot_model} FK, link6 local {model.tcp_local_position.tolist()}m',
        'block_bowl_pose': 'new RGB-D snapshot -> existing SAM3/geometric pose scripts',
        'block_half_size': 'task configuration; physical measurement validity recorded separately',
        'previous_action': 'zero at explicitly reset first shadow step',
        'lstm_memory': 'cell/hidden reset for a new single-snapshot shadow episode',
    }
    return observation, target, warnings, provenance


def run(vision_path, robot_path, calibration_path, model_xml, policies, output,
        block_id, bowl_id, diagnostic_only=False, virtual_target=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {'vision': vision_path, 'robot_feedback': robot_path,
             'calibration': calibration_path, 'training_model_xml': model_xml}
    camera = Path(vision_path).parent
    paths.update(camera_color=camera / 'color/0.png', camera_depth=camera / 'depth/0.png',
                 camera_metadata=camera / 'meta.json')
    report = {
        'schema_version': 1, 'mode': 'diagnostic_snapshot' if diagnostic_only else 'strict_snapshot',
        'input_source': 'new_camera_snapshot_and_passive_robot_feedback',
        'loop': 'single_snapshot_no_plant_feedback', 'hardware_commands_sent': False,
        'command_transport': 'absent; JSON files only', 'control_ready': False,
        'sources': {k: file_record(p) for k, p in paths.items()},
        'snapshot_age_at_evaluation_s': None,
    }
    try:
        vision, robot, calibration = [read_json(paths[k]) for k in ['vision', 'robot_feedback', 'calibration']]
        if block_id == bowl_id:
            raise ValueError('Block and bowl must be distinct objects')
        # Roles are selected explicitly; no automatic nearest-object remapping.
        selected = []
        for actual_id, role in [(block_id, 'block'), (bowl_id, 'bowl')]:
            matches = [o for o in vision['objects'] if o['id'] == actual_id]
            if len(matches) != 1:
                raise ValueError(f'Required object missing or duplicated: {actual_id}')
            selected.append(dict(matches[0], id=role))
        vision = dict(vision, objects=selected)
        names = sorted(p.name for p in Path(policies).iterdir() if (p / 'policy.json').is_file())
        if not names:
            raise ValueError('No exported policies found')
        reference = Policy(Path(policies) / names[0])
        model = PiperKinematics(model_xml)
        if model.robot_model != reference.metadata['environment'].get('robot_model', 'menagerie'):
            raise ValueError('Policy and kinematics robot models disagree')
        expected_model = reference.metadata.get('kinematics', {}).get('piper_xml_sha256')
        if expected_model and file_record(model_xml)['sha256'] != expected_model:
            raise ValueError('Policy and kinematics XML checksums disagree')
        observation, target, warnings, provenance = assemble(
            vision, robot, calibration, model, reference.metadata['environment'], virtual_target)
        report.update(warnings=warnings, observation_provenance=provenance,
                      object_roles={'block': block_id, 'bowl': bowl_id})
        frame_ns = robot.get('camera_frame_host_wall_time_ns')
        if frame_ns:
            report['snapshot_age_at_evaluation_s'] = (time.time_ns() - frame_ns) / 1e9
        report['warnings'].append('offline_snapshot_replay_not_live_control')
        if warnings and not diagnostic_only:
            report.update(status='blocked', policy_evaluated=False,
                          rejection_reason='Unverified calibration/geometry/controller state; diagnostic mode required')
        else:
            records = []
            for name in names:
                policy = Policy(Path(policies) / name)
                for field, default in [('robot_model', 'menagerie'), ('base_height', 0.),
                                       ('gripper_max', .035), ('stage_obs', True), ('size_obs', False)]:
                    if policy.metadata['environment'].get(field, default) != reference.metadata['environment'].get(field, default):
                        raise ValueError(f'Policies have incompatible {field}')
                if policy.metadata['observation_size'] != 48 or policy.metadata['action_size'] != 7:
                    raise ValueError('Unexpected policy I/O contract')
                policy.reset(1)
                start = time.perf_counter_ns()
                action = policy.step(observation, reset=True)
                latency = (time.perf_counter_ns() - start) / 1e6
                c = policy.metadata['environment']
                command = model.propose(
                    observation[:6], target, action,
                    **{k: c[k] for k in ['ctrl_dt', 'max_lin_vel', 'max_ang_vel', 'ik_damping',
                                        'max_joint_vel', 'max_target_lag', 'gripper_max',
                                        'gripper_rate', 'gripper_binary']},
                    current_gripper_finger_target=min(max(float(observation[24]) / 2, 0.), float(c['gripper_max'])),
                )
                def serializable(x):
                    if isinstance(x, np.ndarray): return x.tolist()
                    if isinstance(x, np.generic): return x.item()
                    if isinstance(x, dict): return {k: serializable(v) for k, v in x.items()}
                    return x
                records.append({
                    'policy': name, 'weights_sha256': policy.metadata['weights_sha256'],
                    'normalized_action': action.tolist(), 'single_call_latency_ms': latency,
                    'proposed_simulation_command': serializable(command),
                    'real_gripper_effort_command': None,
                    'hardware_commands_sent': False, 'control_ready': False,
                })
            (output / 'observation.json').write_text(json.dumps({
                'values': observation.tolist(), 'layout': reference.metadata['observation_layout'],
                'provenance': provenance, 'warnings': warnings,
            }, indent=2) + '\n')
            (output / 'proposed_commands.json').write_text(json.dumps(records, indent=2) + '\n')
            report.update(status='diagnostic_completed', policy_evaluated=True, policies=len(records))
    except (ValueError, KeyError, TypeError) as error:
        report.update(status='blocked', policy_evaluated=False, rejection_reason=str(error))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['vision', 'robot-state', 'calibration', 'model-xml', 'policies', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--block-id', required=True)
    ap.add_argument('--bowl-id', required=True)
    ap.add_argument('--diagnostic-only', action='store_true')
    ap.add_argument('--virtual-target-at-measured', action='store_true')
    args = ap.parse_args()
    report = run(args.vision, args.robot_state, args.calibration, args.model_xml,
                 args.policies, args.output, args.block_id, args.bowl_id,
                 args.diagnostic_only, args.virtual_target_at_measured)
    print(json.dumps(report, indent=2))
    if not report['policy_evaluated']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
