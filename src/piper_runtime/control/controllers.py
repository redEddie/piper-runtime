"""Stateful 10 Hz DLS, 200 Hz QP and DLS+Ruckig reference generators.

Limits apply to generated reference motion, not a guarantee about the plant.
On invalid input/infeasibility latch the last reference; this diagnostic hold
is NOT a jerk-limited emergency brake. Only reset clears a latch.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
import osqp
from scipy import sparse
from ruckig import Ruckig, InputParameter, Result, RuckigError, Trajectory


@dataclass(frozen=True)
class Settings:
    dt: float = .005
    policy_dt: float = .1
    linear_scale: float = .1
    angular_scale: float = 1.
    damping: float = .05
    velocity: float = 1.5
    lag: float = .15
    acceleration: float = 8.  # experimental, NOT a hardware-certified limit
    jerk: float = 100.       # experimental, NOT a hardware-certified limit
    tcp_floor: float = .0    # table frame; point constraint, NOT collision safety
    base_height: float = .015
    # Resolved table plane in robot-base coordinates: normal @ TCP >= offset.
    # None keeps legacy world-z tcp_floor/base_height settings for old callers.
    floor_normal_base: tuple = (0.,0.,1.)
    floor_offset_base: float | None = None
    smooth_weight: float = .001
    qp_jerk: bool = False   # opt-in reference jerk bounds; previous acceleration required
    qp_braking: bool = False  # experimental static-bound stopping envelope
    tolerance: float = 2e-5

    def with_scene_frames(self, frames, clearance=None):
        normal,offset = frames.plane_in_base(self.tcp_floor if clearance is None else clearance)
        return replace(self, floor_normal_base=normal, floor_offset_base=offset)


def resolve_settings(kin, settings=None):
    s = settings or Settings()
    if s.floor_offset_base is None and hasattr(kin, 'scene_frames'):
        s = s.with_scene_frames(kin.scene_frames)
    return s


def floor_distance(projected_position, s):
    """Signed distance to the resolved table plane, in metres."""
    if s.floor_offset_base is None:
        return projected_position+s.base_height-s.tcp_floor
    return projected_position-s.floor_offset_base


def floor_kinematics(kin, q, s):
    normal = np.asarray(s.floor_normal_base)
    return normal@kin.forward(q)['tcp_position'], normal@kin.jacobian(q)[:3]


def qp_problem(kin, q, reference, previous_velocity, previous_acceleration, action, s):
    """QP variable v; next reference = reference + dt*v.

    Task Jacobian at measured q, floor linearization at reference q. No
    post-solve clipping (which could invalidate coupled constraints).
    """
    J = kin.jacobian(q)
    twist = np.asarray(action[:6]) * np.array([s.linear_scale]*3+[s.angular_scale]*3)
    H = J.T @ J + (s.damping**2+s.smooth_weight)*np.eye(6)
    g = -J.T @ twist - s.smooth_weight*previous_velocity
    lo = np.maximum.reduce([
        np.full(6, -s.velocity), (kin.lower-reference)/s.dt,
        (q-s.lag-reference)/s.dt, previous_velocity-s.acceleration*s.dt])
    hi = np.minimum.reduce([
        np.full(6, s.velocity), (kin.upper-reference)/s.dt,
        (q+s.lag-reference)/s.dt, previous_velocity+s.acceleration*s.dt])
    if s.qp_braking:
        # Require distance >= dt*|v| + v²/(2*a). This anticipates a
        # stationary position/lag bound; measured-q movement can invalidate
        # that assumption, so infeasibility remains an explicit failure.
        distance_lo=np.maximum(0.,reference-np.maximum(kin.lower,q-s.lag))
        distance_hi=np.maximum(0.,np.minimum(kin.upper,q+s.lag)-reference)
        adt=s.acceleration*s.dt
        lo=np.maximum(lo,adt-np.sqrt(adt**2+2*s.acceleration*distance_lo))
        hi=np.minimum(hi,-adt+np.sqrt(adt**2+2*s.acceleration*distance_hi))
    if s.qp_jerk:
        lo = np.maximum(lo, previous_velocity+s.dt*previous_acceleration-s.jerk*s.dt**2)
        hi = np.minimum(hi, previous_velocity+s.dt*previous_acceleration+s.jerk*s.dt**2)
    z,jf = floor_kinematics(kin,reference,s)
    A = np.vstack([np.eye(6), jf])
    lower = np.r_[lo, -floor_distance(z,s)/s.dt]
    upper = np.r_[hi, np.inf]
    return H, g, A, lower, upper


class Controller:
    def __init__(self, kind, kin, settings=None):
        if kind not in ('dls', 'qp', 'ruckig'):
            raise ValueError(kind)
        self.kind, self.kin, self.s = kind, kin, resolve_settings(kin,settings)
        positive = [self.s.dt,self.s.policy_dt,self.s.linear_scale,self.s.angular_scale,
                    self.s.damping,self.s.velocity,self.s.lag,self.s.acceleration,
                    self.s.jerk,self.s.tolerance]
        if not np.isfinite(positive).all() or np.any(np.asarray(positive)<=0):
            raise ValueError('periods, scales and limits must be finite and positive')
        if (not np.isfinite([self.s.tcp_floor,self.s.base_height,self.s.smooth_weight]).all()
                or self.s.smooth_weight<0):
            raise ValueError('finite floor/base and nonnegative smoothing weight required')
        ratio = self.s.policy_dt/self.s.dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError('policy_dt must be an integer multiple of dt')
        self.ratio = int(round(ratio))
        self.otg = Ruckig(6, self.s.dt)
        self.inp = InputParameter(6)
        self.solver = None
        self.reset(np.zeros(6))

    def reset(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('finite q[6] required')
        self.reference, self.dls_goal = q.copy(), q.copy()
        self.velocity, self.acceleration = np.zeros(6), np.zeros(6)
        self.trajectory_start=np.r_[q,np.zeros(12)]
        self.trajectory=None
        self.action, self.tick, self.latched = np.zeros(7), 0, False
        self.solver = None
        self.otg.reset()
        self.inp.current_position = q.tolist()
        self.inp.current_velocity = [0.]*6
        self.inp.current_acceleration = [0.]*6
        self.inp.target_position = q.tolist()
        self.inp.target_velocity = [0.]*6
        self.inp.target_acceleration = [0.]*6
        self.inp.max_velocity = [self.s.velocity]*6
        self.inp.max_acceleration = [self.s.acceleration]*6
        self.inp.max_jerk = [self.s.jerk]*6

    def update(self, q, action=None, *, input_valid=True):
        q = np.asarray(q, float)
        info = {'ok': True, 'status': 'ok', 'iterations': 0, 'residual': 0.,
                'active': 0, 'updated': self.kind != 'dls' or self.tick % self.ratio == 0}
        if self.latched:
            return self.reference.copy(), dict(info, ok=False, status='latched')
        if (not input_valid or q.shape != (6,) or not np.isfinite(q).all()
                or (action is not None and (np.shape(action) != (7,) or not np.isfinite(action).all()))):
            self.latched = True
            return self.reference.copy(), dict(info, ok=False, status='invalid_input')
        if action is not None:
            self.action = np.clip(np.asarray(action, float), -1, 1)
            # A fresh 10Hz packet may arrive between nominal 200Hz ticks.
            # Rephase the DLS schedule instead of adding an extra update.
            self.tick = 0
            info['updated'] = True
        try:
            if self.kind in ('dls', 'ruckig') and self.tick % self.ratio == 0:
                p = self.kin.propose(q, self.dls_goal, self.action,
                    ctrl_dt=self.s.policy_dt, max_lin_vel=self.s.linear_scale,
                    max_ang_vel=self.s.angular_scale, ik_damping=self.s.damping,
                    max_joint_vel=self.s.velocity, max_target_lag=self.s.lag)
                self.dls_goal = p['joint_target_rad']
                if self.kind == 'dls':
                    self.reference = self.dls_goal.copy()
                else:
                    self.inp.target_position = self.dls_goal.tolist()
                    self.trajectory_start=np.r_[self.reference,self.velocity,self.acceleration]
                    self.trajectory=None
            if self.kind == 'ruckig':
                # Native runtime caches the trajectory; the pure JAX callback
                # reconstructs it from the SAME start state. Never replan from
                # independently rounded intermediate position/derivatives.
                if self.trajectory is None:
                    self.inp.current_position=self.trajectory_start[:6].tolist()
                    self.inp.current_velocity=self.trajectory_start[6:12].tolist()
                    self.inp.current_acceleration=self.trajectory_start[12:18].tolist()
                    self.trajectory=Trajectory(6)
                    result=self.otg.calculate(self.inp,self.trajectory)
                    if result not in (Result.Working,Result.Finished):
                        raise ValueError(f'ruckig:{result}')
                elapsed=(self.tick+1)*self.s.dt
                if elapsed>=self.trajectory.duration:
                    self.reference=self.dls_goal.copy()
                    self.velocity=np.zeros(6);self.acceleration=np.zeros(6)
                else:
                    p,v,a=self.trajectory.at_time(elapsed)
                    self.reference=np.array(p);self.velocity=np.array(v);self.acceleration=np.array(a)
            elif self.kind == 'qp':
                H,g,A,lo,hi = qp_problem(self.kin,q,self.reference,self.velocity,
                                         self.acceleration,self.action,self.s)
                if np.any(lo > hi):
                    raise ValueError('infeasible_bounds')
                # Fixed CSC pattern allows numeric updates without re-setup.
                rr,cc = np.triu_indices(6)
                P = sparse.csc_matrix((H[rr,cc],(rr,cc)),shape=(6,6))
                rows = np.tile(np.arange(7),6); cols = np.repeat(np.arange(6),7)
                Ac = sparse.csc_matrix((A.T.ravel(),(rows,cols)),shape=(7,6))
                if self.solver is None:
                    self.solver = osqp.OSQP()
                    self.solver.setup(P=P,q=g,A=Ac,l=lo,u=hi,verbose=False,
                        eps_abs=1e-7,eps_rel=1e-7,max_iter=1000,polishing=False,
                        check_termination=10,adaptive_rho=True)
                else:
                    self.solver.update(Px=P.data,Ax=Ac.data,q=g,l=lo,u=hi)
                result = self.solver.solve(raise_error=False)
                info.update(status=result.info.status, iterations=int(result.info.iter))
                if result.info.status_val != 1 or not np.isfinite(result.x).all():
                    raise ValueError('qp:'+result.info.status)
                v = result.x
                residual = float(max(0.,np.max(lo-A@v),np.max(A@v-hi)))
                info.update(residual=residual,active=int(np.sum(np.minimum(A@v-lo,hi-A@v)<1e-4)))
                candidate = self.reference+self.s.dt*v
                z = np.asarray(self.s.floor_normal_base)@self.kin.forward(candidate)['tcp_position']
                if residual > self.s.tolerance or floor_distance(z,self.s) < -self.s.tolerance:
                    raise ValueError('constraint_verification')
                self.acceleration = (v-self.velocity)/self.s.dt
                self.velocity = v.copy()
                self.reference = candidate
        except (ValueError, RuntimeError, RuckigError, osqp.OSQPException, np.linalg.LinAlgError) as ex:
            self.latched = True
            lines=str(ex).strip().splitlines()
            info.update(ok=False,status=type(ex).__name__+': '+(lines[0] if lines else 'unspecified'))
        self.tick += 1
        return self.reference.copy(), info

    def describe(self):
        return {'kind':self.kind,'settings':asdict(self.s),
            'ruckig_terminal_velocity':0.,'ruckig_state':'previous planned output',
            'failure':'latch last position reference; requires reset; not smooth braking',
            'limits_scope':'reference motion, not measured robot safety'}
