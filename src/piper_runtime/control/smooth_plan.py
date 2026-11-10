"""GPU jerk-aware velocity shaping and a certified reference stop backup.

Certificate covers sampled reference joint limits and TCP plane, not robot-body
collision or arbitrary plant disturbances. Rejected proposals execute a
previously checked stop sequence; no hard motion limit is relaxed.
"""
import math
import jax
import jax.numpy as jp
from .tracking_qp import tracking_problem,project_tracking
from .jax_kinematics import fk_floor
from .hierarchy import floor_minimum
from .controllers import floor_distance


def terminal_velocity_bounds(v,target,s):
    """Exact discrete acceleration-to-zero tail, for acceleration <= A.

    Peak(v_next)=v_next+sum(max(v_next-v-k*J*dt²,0)). Invert its
    piecewise-linear inequalities without an iterative optimization loop.
    """
    upper=target;lower=target
    h=s.jerk*s.dt*s.dt
    count=math.ceil(s.acceleration/(s.jerk*s.dt))
    for m in range(1,count):
        center=(target+m*v)/(m+1)
        upper=jp.minimum(upper,center+h*m/2)
        lower=jp.maximum(lower,center-h*m/2)
    return lower,upper


def speed_bounds(v,a,s):
    lo=jp.maximum(-s.velocity,jp.maximum(v-s.acceleration*s.dt,v+s.dt*a-s.jerk*s.dt**2))
    hi=jp.minimum(s.velocity,jp.minimum(v+s.acceleration*s.dt,v+s.dt*a+s.jerk*s.dt**2))
    lower,_=terminal_velocity_bounds(v,jp.full_like(v,-s.velocity),s)
    _,upper=terminal_velocity_bounds(v,jp.full_like(v,s.velocity),s)
    return jp.maximum(lo,lower),jp.minimum(hi,upper)


def position_bounds(ref,v,lower,upper,s):
    """Conservative jerk-aware stop room, separate from one-step positions."""
    duration=s.velocity/s.acceleration+3*s.acceleration/s.jerk+4*s.dt
    h=s.jerk*s.dt*s.dt
    lo=(lower-ref)/(s.dt+duration);hi=(upper-ref)/(s.dt+duration)
    for m in range(1,math.ceil(s.acceleration/(s.jerk*s.dt))):
        denom=s.dt+duration*(m+1)
        lo=jp.maximum(lo,((lower-ref)+duration*m*v-duration*h*m*(m+1)/2)/denom)
        hi=jp.minimum(hi,((upper-ref)+duration*m*v+duration*h*m*(m+1)/2)/denom)
    return lo,hi


def velocity_step(v,a,target,s):
    lo,hi=speed_bounds(v,a,s)
    lower,upper=terminal_velocity_bounds(v,target,s)
    desired=jp.where(target>=v,upper,lower)
    vn=jp.clip(desired,lo,hi)
    return vn,(vn-v)/s.dt


def stop_sequence(state,s,length):
    def step(st,_):
        ref,v,a=jp.split(st,3)
        vn,an=velocity_step(v,a,jp.zeros_like(v),s)
        st=jp.concatenate([ref+s.dt*vn,vn,an])
        return st,st
    return jax.lax.scan(step,state,None,length=length)[1]


def checked_plan(kin,q,qdot,state,target,old_target,hold,lower,upper,s,p,steps=20,stop_steps=64,diagnostics=False):
    """Plan one policy packet, append a full stop, check every reference."""
    def step(carry,i):
        st,ok=carry;ref,v,a=jp.split(st,3)
        desired=jp.where(i<hold,old_target,target)
        # Begin braking before a mechanical bound, allowing the remaining
        # joints to move instead of rejecting every packet near one wrist limit.
        desired=jp.clip(desired,(lower-ref)/.5,(upper-ref)/.5)
        z,normal=fk_floor(kin,ref,s.floor_normal_base)
        approach=-jp.maximum(floor_distance(z,s)-.001,0.)/.5
        correction=jp.maximum(approach-jp.dot(normal,desired),0.)
        desired=desired+normal*correction/jp.maximum(jp.dot(normal,normal),1e-8)
        shaped,_=velocity_step(v,a,desired,s)
        measured=q+jp.minimum(i*s.dt,.05)*qdot
        preferred,lo,hi,slo,shi,_,_=tracking_problem(measured,qdot,ref,v,shaped,lower,upper,s,p,
            jp.zeros(6,bool),jp.zeros(6),xp=jp,previous_acceleration=a)
        viable_lo,viable_hi=speed_bounds(v,a,s)
        lo=jp.maximum(lo,viable_lo);hi=jp.minimum(hi,viable_hi)
        position_lo,position_hi=position_bounds(ref,v,lower,upper,s)
        lo=jp.maximum(lo,position_lo);hi=jp.minimum(hi,position_hi)
        vn,feasible=project_tracking(preferred,lo,hi,slo,shi,p.weight,normal,floor_minimum(z,s,xp=jp))
        # An invalid proposal is discarded wholesale; keep arithmetic finite.
        fallback_v,fallback_a=velocity_step(v,a,jp.zeros(6),s)
        vn=jp.where(feasible,vn,fallback_v);an=(vn-v)/s.dt
        st=jp.concatenate([ref+s.dt*vn,vn,an])
        return (st,ok&feasible),st
    (last,ok),motion=jax.lax.scan(step,(state,jp.array(True)),jp.arange(steps))
    stop=stop_sequence(last,s,stop_steps)
    plan=jp.concatenate([motion,stop],axis=0)
    positions=plan[:,:6]
    z=jax.vmap(lambda ref:fk_floor(kin,ref,s.floor_normal_base)[0])(positions)
    distance=floor_distance(z,s)
    within=jp.all((positions>=lower-s.tolerance)&(positions<=upper+s.tolerance))
    terminal=jp.max(jp.abs(stop[-1,6:12]))<1e-5
    terminal=terminal&(jp.max(jp.abs(stop[-1,12:]))<1e-3)
    if diagnostics:
        detail=jp.array([ok,within,jp.min(distance),jp.max(jp.abs(stop[-1,6:12])),jp.max(jp.abs(stop[-1,12:]))])
    ok=ok&within&jp.all(distance>=-s.tolerance)&terminal&jp.all(jp.isfinite(plan))
    if diagnostics:return plan,ok,detail
    return plan,ok
