"""NumPy SHOW-100 DLS + recovering tracking QP, with explicit latched faults.

Equations match tracking_env.py/tracking_qp.py. No JAX, SDK, device I/O or
task action overrides. Constraints bound references, not the physical plant.
"""
from collections import deque
import numpy as np
from .show100 import vector


class ControlFault(RuntimeError):
    pass


def tracking_problem(q, qdot, ref, velocity, target, lower, upper, *, dt, speed,
                     acceleration, lag, extra, margin, horizon, weight,
                     recovering, timer, smooth=.001, exit_margin=.01):
    error = ref-q
    active = (recovering | (np.abs(error) > lag)) & (np.abs(error) > lag-exit_margin)
    elapsed = np.where(active, timer+dt, 0.)
    band = lag-margin
    recovery = qdot-(error-np.clip(error, -band, band))/horizon
    preferred = (np.where(active, recovery, target)+smooth*velocity)/(1+smooth)
    adt = acceleration*dt
    lo = np.maximum(np.maximum(-speed, (lower-ref)/dt), velocity-adt)
    hi = np.minimum(np.minimum(speed, (upper-ref)/dt), velocity+adt)
    lo = np.maximum(lo, adt-np.sqrt(adt**2+2*acceleration*np.maximum(0., ref-lower)))
    hi = np.minimum(hi, -adt+np.sqrt(adt**2+2*acceleration*np.maximum(0., upper-ref)))
    qnext = q+dt*qdot
    cap = lag+extra
    lo = np.maximum(lo, (np.maximum(q, qnext)-cap-ref)/dt)
    hi = np.minimum(hi, (np.minimum(q, qnext)+cap-ref)/dt)
    predicted = q+horizon*qdot
    return (preferred, lo, hi, (predicted-band-ref)/horizon,
            (predicted+band-ref)/horizon, active, elapsed)


def project_tracking(target, lo, hi, soft_lo, soft_hi, weight, normal, minimum):
    def value(multiplier):
        z = target+multiplier*normal
        v = np.where(z < soft_lo, (z+weight*soft_lo)/(1+weight),
                     np.where(z > soft_hi, (z+weight*soft_hi)/(1+weight), z))
        return np.clip(v, lo, hi)
    maximum = normal @ np.where(normal >= 0, hi, lo)
    if np.any(lo > hi) or maximum < minimum-1e-6:
        raise ControlFault('QP infeasible')
    base = value(0.)
    if normal @ base >= minimum:
        return base
    right = 1.
    for _ in range(40):
        if normal @ value(right) >= minimum:
            break
        right *= 2
    left = 0.
    for _ in range(32):
        middle = (left+right)*.5
        if normal @ value(middle) < minimum:
            left = middle
        else:
            right = middle
    return value(right)


class Controller100:
    def __init__(self, model, environment, *, floor_normal=(0., 0., 1.), floor_offset=None):
        self.model, self.e = model, environment
        e = environment
        self.dt, self.policy_dt = float(e['sim_dt']), float(e['ctrl_dt'])
        self.speed, self.acceleration = float(e['max_joint_vel']), float(e['control_acceleration'])
        self.lag, self.extra = float(e['max_target_lag']), float(e.get('tracking_extra', .02))
        self.margin, self.horizon = float(e.get('tracking_margin', .02)), float(e.get('tracking_horizon', .05))
        self.weight, self.timeout = float(e.get('tracking_weight', 100.)), float(e.get('tracking_timeout', .5))
        self.normal = vector(floor_normal, 3, 'floor normal')
        if not np.isclose(np.linalg.norm(self.normal), 1., atol=1e-6):
            raise ValueError('Floor normal must have unit length')
        self.floor = float(floor_offset if floor_offset is not None else -e['base_height']+e.get('control_floor', 0.))
        limits = [self.dt, self.policy_dt, self.speed, self.acceleration, self.lag,
                  self.margin, self.horizon, self.weight, self.timeout, e['gripper_rate'],
                  e['gripper_max'], e['ik_damping'], e['max_lin_vel'], e['max_ang_vel']]
        if not np.isfinite(limits+[self.floor, self.extra]).all() or min(limits) <= 0 or not 0 < self.margin < self.lag or self.extra < 0 or not 0 < e['gripper_threshold'] < 1:
            raise ValueError('Invalid SHOW-100 control periods, limits or gripper parameters')
        if e.get('control_mode') != 'qp' or e.get('gripper_mode') != 'ternary_latched':
            raise ValueError('SHOW-100 requires qp and ternary_latched gripper')
        self.started, self.latched = False, None

    def reset(self, q, gripper_reference_width, final_goal_width):
        self.reference = vector(q, 6, 'reset q').copy()
        if np.any(self.reference < self.model.lower) or np.any(self.reference > self.model.upper):
            raise ControlFault('Reset outside joint limits')
        self.velocity, self.target = np.zeros(6), np.zeros(6)
        self.recovering, self.timer = np.zeros(6, bool), np.zeros(6)
        self.previous_action = np.zeros(7, np.float32)
        self.gripper_reference = float(gripper_reference_width)
        self.gripper_goal = float(final_goal_width)
        maximum = 2*self.e['gripper_max']
        if not np.isfinite([self.gripper_reference, self.gripper_goal]).all() or not 0 <= self.gripper_reference <= maximum or self.gripper_goal not in (0., maximum):
            raise ControlFault('Explicit valid gripper reference and final goal required')
        self.latched, self.started = None, True

    def fail(self, reason):
        self.latched = str(reason)
        raise ControlFault(self.latched)

    def set_action(self, q, action):
        if not self.started or self.latched:
            raise ControlFault(self.latched or 'Controller not reset')
        try:
            action = vector(action, 7, 'action')
            q = vector(q, 6, 'q')
        except ValueError as exc:
            self.fail(exc)
        if np.any(np.abs(action) > 1.):
            self.fail('Policy action outside [-1,1]')
        j = self.model.jacobian(q)
        twist = action[:6]*np.array([self.e['max_lin_vel']]*3+[self.e['max_ang_vel']]*3)
        target = j.T @ np.linalg.solve(j @ j.T+self.e['ik_damping']**2*np.eye(6), twist)
        self.target = np.clip(target, -self.speed, self.speed)
        threshold = self.e['gripper_threshold']
        if action[6] <= -threshold:
            self.gripper_goal = 0.
        elif action[6] >= threshold:
            self.gripper_goal = 2*self.e['gripper_max']
        rate = 2*self.e['gripper_rate']
        self.gripper_reference = float(np.clip(self.gripper_goal, self.gripper_reference-rate, self.gripper_reference+rate))
        self.previous_action = action.astype(np.float32)

    def step(self, q, qdot):
        if not self.started or self.latched:
            raise ControlFault(self.latched or 'Controller not reset')
        try:
            q, qdot = vector(q, 6, 'q'), vector(qdot, 6, 'qdot')
            if np.max(np.abs(self.reference-q)) > self.lag+self.extra+2e-5:
                self.fail('Actual tracking outer cap exceeded')
            preferred, lo, hi, soft_lo, soft_hi, active, timer = tracking_problem(
                q, qdot, self.reference, self.velocity, self.target, self.model.lower, self.model.upper,
                dt=self.dt, speed=self.speed, acceleration=self.acceleration, lag=self.lag,
                extra=self.extra, margin=self.margin, horizon=self.horizon, weight=self.weight,
                recovering=self.recovering, timer=self.timer)
            z = self.normal @ self.model.forward(self.reference)['tcp_position']
            jac = self.normal @ self.model.jacobian(self.reference)[:3]
            distance = z-self.floor
            minimum = max(-distance/self.dt, -5.*(distance-.001))
            v = project_tracking(preferred, lo, hi, soft_lo, soft_hi, self.weight, jac, minimum)
            candidate = self.reference+self.dt*v
            if np.any(timer > self.timeout):
                self.fail('Tracking recovery timeout')
            if not np.isfinite(candidate).all() or self.normal @ self.model.forward(candidate)['tcp_position']-self.floor < -2e-5:
                self.fail('Nonlinear floor constraint failed')
            self.reference, self.velocity = candidate, v
            self.recovering, self.timer = active, timer
            return dict(joint_target_rad=candidate.copy(), reference_velocity=v.copy(),
                        gripper_width_target_m=self.gripper_reference,
                        gripper_final_goal_width_m=self.gripper_goal,
                        tracking_gap_rad=float(np.max(np.abs(candidate-q))))
        except (ValueError, np.linalg.LinAlgError, ControlFault) as exc:
            self.fail(exc)


class ReferenceHistory:
    """Reference at each joint feedback receipt time; never align to a future target."""
    def __init__(self, timestamp, reference):
        self.rows = deque([(float(timestamp), vector(reference, 6, 'reference').copy())], maxlen=2000)

    def append(self, timestamp, reference):
        if timestamp < self.rows[-1][0]:
            raise ControlFault('Reference clock moved backwards')
        self.rows.append((float(timestamp), vector(reference, 6, 'reference').copy()))

    def at(self, joint_timestamps):
        stamps = vector(joint_timestamps, 6, 'joint timestamps')
        if np.min(stamps) < self.rows[0][0]:
            raise ControlFault('Feedback predates retained reference history')
        rows = list(self.rows)
        times, values = np.array([r[0] for r in rows]), np.stack([r[1] for r in rows])
        # References are held until the next transmission, not interpolated using
        # a command that had not yet been sent at the feedback timestamp.
        indices = np.maximum(0, np.searchsorted(times, stamps, side='right')-1)
        return values[indices, np.arange(6)]
