"""DLS postprocessor: moving tracking band, bounded slack, recovery.

The task's reward and action guards are outside this module. Physical joint
reference bounds, speed, acceleration, optional jerk and TCP plane remain hard. Only the
measured/reference tracking band is soft, with an explicit outer cap/timeout.
"""
from dataclasses import dataclass
import jax
import jax.numpy as jp
import numpy as np


@dataclass(frozen=True)
class TrackingSettings:
    extra: float = .02          # outer tracking cap = .15 + .02 rad
    margin: float = .02         # begin intervention around .13 rad
    horizon: float = .05        # constant measured-velocity lookahead, seconds
    weight: float = 100.        # QP internal cost, NOT an RL reward
    timeout: float = .5         # maximum uninterrupted recovery, seconds
    exit_margin: float = .01    # recovery exits below .14 rad


def tracking_problem(q, qdot, ref, velocity, target, lower, upper, s, p,
                     recovering, timer, xp=np, previous_acceleration=None):
    """Opt-in jerk is a one-step bound; it does not ensure future feasibility."""
    error = ref-q
    active = (recovering | (xp.abs(error)>s.lag)) & (xp.abs(error)>s.lag-p.exit_margin)
    elapsed = xp.where(active, timer+s.dt, 0.)
    # Chase the measured joint at its velocity, reducing only excess error.
    band = s.lag-p.margin
    recovery_target = qdot-(error-xp.clip(error,-band,band))/p.horizon
    preferred = xp.where(active,recovery_target,target)
    preferred = (preferred+s.smooth_weight*velocity)/(1+s.smooth_weight)
    adt = s.acceleration*s.dt
    # Braking envelopes apply ONLY to fixed mechanical reference bounds.
    lo = xp.maximum(xp.maximum(-s.velocity,(lower-ref)/s.dt),velocity-adt)
    hi = xp.minimum(xp.minimum(s.velocity,(upper-ref)/s.dt),velocity+adt)
    lo = xp.maximum(lo,adt-xp.sqrt(adt**2+2*s.acceleration*xp.maximum(0.,ref-lower)))
    hi = xp.minimum(hi,-adt+xp.sqrt(adt**2+2*s.acceleration*xp.maximum(0.,upper-ref)))
    if s.qp_jerk:
        if previous_acceleration is None:
            raise ValueError('Jerk-limited tracking requires previous reference acceleration')
        # a_next=(v_next-v_prev)/dt; |a_next-a_prev| <= jerk*dt.
        center=velocity+s.dt*previous_acceleration
        radius=s.jerk*s.dt**2
        lo=xp.maximum(lo,center-radius)
        hi=xp.minimum(hi,center+radius)
    qnext = q+s.dt*qdot
    cap = s.lag+p.extra
    # Both current and next measured pose constrain the outer tracking cap.
    lo = xp.maximum(lo,(xp.maximum(q,qnext)-cap-ref)/s.dt)
    hi = xp.minimum(hi,(xp.minimum(q,qnext)+cap-ref)/s.dt)
    # Separable soft constraint on predicted relative displacement. Velocity
    # differences, rather than reference velocity alone, determine approach.
    predicted = q+p.horizon*qdot
    soft_lo = (predicted-band-ref)/p.horizon
    soft_hi = (predicted+band-ref)/p.horizon
    return preferred,lo,hi,soft_lo,soft_hi,active,elapsed


def project_tracking(target,lo,hi,soft_lo,soft_hi,weight,normal,minimum):
    """Exact separable piecewise quadratic QP + one hard plane constraint.

min .5||v-target||² + .5*weight*dist(v,[soft_lo,soft_hi])²
s.t. lo<=v<=hi, normal@v>=minimum. Scalar dual bisection stays on GPU.
"""
    def value(multiplier):
        z=target+multiplier*normal
        v=jp.where(z<soft_lo,(z+weight*soft_lo)/(1+weight),
                   jp.where(z>soft_hi,(z+weight*soft_hi)/(1+weight),z))
        return jp.clip(v,lo,hi)
    maximum=jp.dot(normal,jp.where(normal>=0,hi,lo))
    feasible=jp.all(lo<=hi)&(maximum>=minimum-1e-6)
    upper=jax.lax.fori_loop(0,40,lambda _,r:jp.where(jp.dot(normal,value(r))<minimum,2*r,r),jp.array(1.,target.dtype))
    def bisect(_,pair):
        left,right=pair;mid=(left+right)*.5
        below=jp.dot(normal,value(mid))<minimum
        return jp.where(below,mid,left),jp.where(below,right,mid)
    _,multiplier=jax.lax.fori_loop(0,32,bisect,(jp.array(0.,target.dtype),upper))
    base=value(0.)
    return jp.where(jp.dot(normal,base)>=minimum,base,value(multiplier)),feasible
