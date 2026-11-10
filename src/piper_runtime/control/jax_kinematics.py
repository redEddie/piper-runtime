"""JAX reference TCP kinematics; no MJX environment dependency."""
import jax.numpy as jp
from piper_policy_runtime.kinematics import _matrix

def fk_floor(kin, q, plane_normal=(0.,0.,1.)):
    """Reference TCP projection and Jacobian along a base-frame plane normal."""
    position, rotation = jp.zeros(3,q.dtype),jp.eye(3,dtype=q.dtype)
    origins,axes = [],[]
    for angle,(offset,quat),(pivot,axis,reference,_) in zip(q,kin._bodies,kin._joints):
        position = position+rotation@jp.asarray(offset,dtype=q.dtype)
        rotation = rotation@jp.asarray(_matrix(quat),dtype=q.dtype)
        pivot,axis = jp.asarray(pivot,dtype=q.dtype),jp.asarray(axis,dtype=q.dtype)
        origin = position+rotation@pivot
        origins.append(origin); axes.append(rotation@axis)
        x,y,z=axis
        skew=jp.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]],dtype=q.dtype)
        a=angle-reference
        rotation=rotation@(jp.eye(3,dtype=q.dtype)+jp.sin(a)*skew+(1-jp.cos(a))*(skew@skew))
        position=origin-rotation@pivot
    tcp=position+rotation@jp.asarray(kin.tcp_local_position,dtype=q.dtype)
    n=jp.asarray(plane_normal,dtype=q.dtype)
    normal=jp.cross(jp.stack(axes),tcp-jp.stack(origins))@n
    return n@tcp,normal
