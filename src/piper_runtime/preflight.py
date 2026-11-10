"""Shared feedback and perception validation, no implicit SDK connection."""
import time
import numpy as np
from piper_policy_runtime.control100 import ControlFault
from .transports.robot_io import PerceptionFailed

def validate_state(state, adapter, *, start=False, recovery_limits=None):
    q, qdot = state['q_rad'], state['qvel_rad_s']
    if not np.isfinite(q).all() or not np.isfinite(qdot).all() or np.max(np.abs(qdot)) > 2.5:
        raise ControlFault('Invalid/overspeed measured joints')
    lower, upper = (adapter.model.lower, adapter.model.upper) if recovery_limits is None else (
        np.asarray(recovery_limits)[:, 0], np.asarray(recovery_limits)[:, 1])
    if np.any(q < lower-.05) or np.any(q > upper+.05):
        offending = [f'J{i+1}={np.rad2deg(q[i]):.2f}° outside {np.rad2deg(lower[i]):.2f}..{np.rad2deg(upper[i]):.2f}°'
                     for i in range(6) if q[i] < lower[i]-.05 or q[i] > upper[i]+.05]
        raise ControlFault(('Measured joints outside SDK recovery limits: ' if recovery_limits is not None
                            else 'Measured joints outside model limits: ')+', '.join(offending))
    if not np.isfinite(state['gripper_full_width_m']) or not 0 <= state['gripper_full_width_m'] <= .101:
        raise ControlFault('Invalid gripper aperture')
    if start:
        if np.max(np.abs(q-np.array(adapter.environment['nominal_qpos']))) > np.deg2rad(3.):
            raise ControlFault('Start pose not reached: run show100_robot.py home first')
        if state['gripper_full_width_m'] < .09 or np.max(np.abs(qdot)) > .05:
            raise ControlFault('Start requires open gripper >=90mm and stationary arm')

def validate_objects(objects, environment):
    for role in ('block', 'bowl'):
        pos = objects[role+'_pos_m']
        for i, axis in enumerate(('x', 'y')):
            low, high = environment[role+'_'+axis]
            if not low <= pos[i] <= high:
                raise ControlFault(f'{role} {axis} outside learned reset range {low}..{high}m')
        if np.linalg.norm(pos[:2]) > environment.get('spawn_max_radius', np.inf):
            raise ControlFault(f'{role} outside learned spawning radius')
    if np.linalg.norm(objects['block_pos_m'][:2]-objects['bowl_pos_m'][:2]) < environment.get('spawn_min_distance', 0.):
        raise ControlFault('Cube and bowl are closer than learned spawning separation')
    if not np.isclose(objects['block_half_size_m'], environment['block_half'], atol=.001):
        raise ControlFault('Perceived cube size differs from checkpoint task')

def wait_feedback(robot, adapter, timeout, *, moving=False, recovery_limits=None):
    deadline, last = time.monotonic()+timeout, None
    while time.monotonic() < deadline:
        try:
            state = robot.read(require_enabled=moving, require_can=moving)
            validate_state(state, adapter, recovery_limits=recovery_limits)
            return state
        except PerceptionFailed:
            raise
        except ControlFault as exc:
            last = str(exc)
            time.sleep(.02)
    raise ControlFault(last or 'Feedback startup timeout')

def wait_vision(robot, vision, adapter, timeout, start, *, estimator=None):
    deadline, last, accepted, previous = time.monotonic()+timeout, None, 0, None
    last_print = 0.
    while time.monotonic() < deadline:
        try:
            state = robot.read(require_enabled=start, require_can=start)
            validate_state(state, adapter, start=start if estimator is None else False)
            if estimator is not None:
                now = time.time()
                estimator.ingest(vision.observations(now), now)
                if not estimator.ready:
                    raise ControlFault('약 1초간 안정된 큐브·그릇 위치 수집 중')
                objects = estimator.estimate(state, now, .1)
                validate_objects(objects, adapter.environment)
                if not previous:
                    print('초기 물체 위치 저장 완료. 팔이 시작 자세가 아니면 다른 터미널에서 show100_robot.py home을 실행하세요. 카메라를 기다리며 자동으로 팔을 움직이지 않습니다.', flush=True)
                    previous = True
                validate_state(state, adapter, start=start)
                return state, objects
            objects = vision.read(time.time())
            validate_objects(objects, adapter.environment)
            if objects['object_frames'] != previous:
                accepted += 1
                previous = objects['object_frames']
            if accepted >= 5:
                return state, objects
        except PerceptionFailed:
            raise
        except ControlFault as exc:
            last, accepted = str(exc), 0
            if time.monotonic()-last_print > 2.:
                print('Preflight waiting:', last, flush=True)
                last_print = time.monotonic()
        time.sleep(.025)
    raise ControlFault('Preflight: '+str(last or 'Five fresh input snapshots not received'))
