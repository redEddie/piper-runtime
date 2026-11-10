"""SHOW-92: a shared 10 Hz DLS followed by joint-space postprocessors.

QP is a joint-velocity projection, not SHOW-91's TCP-objective QP. Ruckig
uses velocity control and never inserts a zero terminal velocity per packet.
Experimental limits concern references only, not certified hardware safety.
"""
import numpy as np
import osqp
from scipy import sparse
from ruckig import Ruckig, InputParameter, Trajectory, ControlInterface, Result
from .controllers import Settings, resolve_settings, floor_distance, floor_kinematics

FLOOR_GAIN = 5.  # 1/s, slow the normal approach before reaching the floor
FLOOR_MARGIN = .001  # m, reference-only margin for nonlinear curvature


def floor_minimum(z, s, xp=np):
    distance = floor_distance(z,s)
    return xp.maximum(-distance/s.dt, -FLOOR_GAIN*(distance-FLOOR_MARGIN))


def dls_velocity(kin, q, action, s):
    j = kin.jacobian(q)
    twist = np.asarray(action[:6]) * np.array([s.linear_scale]*3+[s.angular_scale]*3)
    return np.clip(j.T @ np.linalg.solve(j @ j.T+s.damping**2*np.eye(6), twist),
                   -s.velocity, s.velocity)


def bounds(q, ref, v, lower, upper, s, xp=np):
    """Position/lag, velocity, acceleration and static-bound braking envelope."""
    low = xp.maximum(lower, q-s.lag)
    high = xp.minimum(upper, q+s.lag)
    adt = s.acceleration*s.dt
    lo = xp.maximum(xp.maximum(-s.velocity, (low-ref)/s.dt), v-adt)
    hi = xp.minimum(xp.minimum(s.velocity, (high-ref)/s.dt), v+adt)
    lo = xp.maximum(lo, adt-xp.sqrt(adt**2+2*s.acceleration*xp.maximum(0., ref-low)))
    hi = xp.minimum(hi, -adt+xp.sqrt(adt**2+2*s.acceleration*xp.maximum(0., high-ref)))
    return lo, hi


def project_numpy(target, lo, hi, normal, minimum):
    """Euclidean projection onto a box intersected with one halfspace."""
    if np.any(lo > hi) or np.dot(normal, np.where(normal>=0, hi, lo)) < minimum-1e-9:
        raise ValueError('infeasible joint QP')
    v = np.clip(target, lo, hi)
    if normal @ v >= minimum:
        return v
    left, right = 0., 1.
    for _ in range(60):
        if normal @ np.clip(target+right*normal, lo, hi) >= minimum:
            break
        right *= 2
    for _ in range(60):
        mid = .5*(left+right)
        if normal @ np.clip(target+mid*normal, lo, hi) < minimum:
            left = mid
        else:
            right = mid
    return np.clip(target+right*normal, lo, hi)


def project_osqp(target, lo, hi, normal, minimum):
    solver = osqp.OSQP()
    solver.setup(P=sparse.eye(6, format='csc'), q=-target,
        A=sparse.csc_matrix(np.vstack([np.eye(6), normal])),
        l=np.r_[lo, minimum], u=np.r_[hi, np.inf], verbose=False,
        eps_abs=1e-8, eps_rel=1e-8, max_iter=10000, polishing=True)
    result = solver.solve(raise_error=False)
    if result.info.status_val != 1:
        raise ValueError(result.info.status)
    return result.x


def ruckig_segment(start, target, steps, s):
    """Plan once, sample at 200 Hz, carrying p/v/a between 10 Hz packets."""
    inp = InputParameter(6)
    inp.control_interface = ControlInterface.Velocity
    inp.current_position = start[:6].tolist()
    inp.current_velocity = start[6:12].tolist()
    inp.current_acceleration = start[12:18].tolist()
    inp.target_velocity = np.asarray(target).tolist()
    inp.target_acceleration = [0.]*6
    inp.max_velocity = [s.velocity]*6
    inp.max_acceleration = [s.acceleration]*6
    inp.max_jerk = [s.jerk]*6
    trajectory = Trajectory(6)
    result = Ruckig(6, s.dt).calculate(inp, trajectory)
    if result not in (Result.Working, Result.Finished):
        raise ValueError(f'ruckig:{result}')
    return np.array([np.concatenate(trajectory.at_time((i+1)*s.dt)) for i in range(steps)])


def ruckig_packet(start, old_target, target, hold, s):
    """Existing action-execution delay is retained separately from OTG response."""
    count = round(s.policy_dt/s.dt)
    hold = int(np.clip(hold, 0, count))
    before = ruckig_segment(start, old_target, hold, s) if hold else np.empty((0,18))
    after = ruckig_segment(before[-1] if hold else start, target, count-hold, s) if hold<count else np.empty((0,18))
    return np.concatenate([before, after])


class HierarchicalController:
    def __init__(self, kind, kin, settings=None, delay_ticks=0, tau=.04):
        if kind not in ('dls','qp','ruckig','delay','surrogate'):
            raise ValueError(kind)
        self.kind, self.kin, self.s = kind, kin, resolve_settings(kin,settings)
        self.ratio = round(self.s.policy_dt/self.s.dt)
        self.delay_ticks, self.tau = int(delay_ticks), float(tau)
        if self.delay_ticks < 0 or self.tau <= 0:
            raise ValueError('nonnegative delay and positive tau required')
        self.reset(np.zeros(6))

    def reset(self, q):
        self.reference = np.array(q, float)
        self.goal = self.reference.copy()
        self.velocity = np.zeros(6)
        self.acceleration = np.zeros(6)
        self.target = np.zeros(6)
        self.action = np.zeros(7)
        self.tick = 0
        self.latched = False
        self.queue = [self.reference.copy() for _ in range(self.delay_ticks)]
        self.plan = None

    def update(self, q, action=None):
        s, kin = self.s, self.kin
        info = dict(ok=not self.latched, active=0, status='ok', iterations=0, residual=0.)
        if self.latched:
            return self.reference.copy(), dict(info, status='latched')
        try:
            if not np.isfinite(q).all() or (action is not None and not np.isfinite(action).all()):
                raise ValueError('nonfinite input')
            if action is not None:
                self.action = np.clip(action,-1,1)
                self.target = dls_velocity(kin,q,self.action,s)
                self.goal = np.clip(np.clip(self.goal+s.policy_dt*self.target,q-s.lag,q+s.lag),kin.lower,kin.upper)
                self.tick = 0
                if self.kind == 'ruckig':
                    self.plan = ruckig_packet(np.r_[self.reference,self.velocity,self.acceleration],
                                               self.target,self.target,0,s)
            if self.kind in ('dls','delay'):
                self.queue.append(self.goal.copy())
                candidate = self.queue.pop(0)
                v = (candidate-self.reference)/s.dt
            elif self.kind == 'ruckig':
                if self.plan is None or self.tick>=len(self.plan):
                    raise ValueError('missing 10 Hz packet')
                candidate, v, a = np.split(self.plan[self.tick],3)
                # No hidden clipping: violations are counted in the benchmark.
                self.acceleration = a
            else:
                lo, hi = bounds(q,self.reference,self.velocity,kin.lower,kin.upper,s)
                z,normal = floor_kinematics(kin,self.reference,s)
                minimum = floor_minimum(z,s)
                if self.kind == 'qp':
                    target = (self.target+s.smooth_weight*self.velocity)/(1+s.smooth_weight)
                else:
                    desired_a = np.clip((self.target-self.velocity)/self.tau,-s.acceleration,s.acceleration)
                    a = np.clip(desired_a,self.acceleration-s.jerk*s.dt,self.acceleration+s.jerk*s.dt)
                    target = self.velocity+s.dt*a
                v = project_numpy(target,lo,hi,normal,minimum)
                info['active'] = int(np.count_nonzero(np.abs(v-target)>s.tolerance))
                candidate = self.reference+s.dt*v
                z=np.asarray(s.floor_normal_base)@kin.forward(candidate)['tcp_position']
                if floor_distance(z,s) < -s.tolerance:
                    raise ValueError('nonlinear floor verification')
            if self.kind != 'ruckig':
                self.acceleration = (v-self.velocity)/s.dt
            self.velocity, self.reference = np.array(v), np.array(candidate)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as ex:
            self.latched = True
            info.update(ok=False,status=str(ex))
        self.tick += 1
        return self.reference.copy(), info
