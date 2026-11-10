"""NumPy workspace/ground checks using exported physical-model convex hulls.

Convex hull vertices preserve extrema of every original mesh vertex under rigid
transforms. This checks fixed workspace planes and sampled paths, not complete
self/object collision detection or a certified physical safety system.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .show100 import vector
from .control100 import ControlFault


def rotation(axis, angle):
    x, y, z = axis/np.linalg.norm(axis)
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    result = np.eye(4)
    result[:3, :3] = np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew @ skew)
    return result


class GeometryGuard:
    def __init__(self, archive, *, table_z, x_max=None, y_min=None,
                 grip_floor=0., link_floor=.05):
        archive = Path(archive)
        metadata = json.loads(archive.with_suffix('.json').read_text())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != metadata['sha256']:
            raise ControlFault('Physical geometry archive checksum mismatch')
        with np.load(archive, allow_pickle=False) as data:
            self.points = [data[f'part_{i}'].copy() for i in range(9)]
            self.origins, self.axes = data['joint_origins'].copy(), data['joint_axes'].copy()
            self.finger_origins, self.finger_axes = data['finger_origins'].copy(), data['finger_axes'].copy()
        self.table_z, self.x_max, self.y_min = float(table_z), x_max, y_min
        self.grip_floor, self.link_floor = grip_floor, link_floor
        limits = [self.table_z, grip_floor, link_floor]
        limits += [value for value in (x_max, y_min) if value is not None]
        if not np.isfinite(limits).all() or abs(self.table_z) > .1:
            raise ControlFault('Invalid workspace/table geometry limits')
        self.normal, self.floor_offset = np.array([0., 0., 1.]), self.table_z

    def set_plane(self, normal, offset):
        normal = vector(normal, 3, 'table normal')
        if not np.isfinite(offset) or not np.isclose(np.linalg.norm(normal), 1., atol=1e-5) or normal[2] < np.cos(np.deg2rad(5.)) or abs(offset/normal[2]-self.table_z) > .015:
            raise ControlFault('Perceived table plane disagrees with calibrated support')
        self.normal, self.floor_offset = normal.copy(), float(offset)

    def transforms(self, q, full_width):
        q = vector(q, 6, 'geometry q')
        if not np.isfinite(full_width) or not 0 <= full_width <= .101:
            raise ControlFault('Geometry gripper width invalid')
        current = np.eye(4)
        result = [current.copy()]
        for origin, axis, angle in zip(self.origins, self.axes, q):
            current = current @ origin @ rotation(axis, angle)
            result.append(current.copy())
        for index, sign in enumerate((1., -1.)):
            slide = np.eye(4)
            slide[:3, 3] = self.finger_axes[index]*sign*full_width/2
            result.append(current @ self.finger_origins[index] @ slide)
        return result

    def extremes(self, q, width):
        transforms = self.transforms(q, width)
        points = {i:self.points[i] @ transforms[i][:3, :3].T+transforms[i][:3, 3] for i in range(2, 9)}
        grip = np.concatenate([points[i] for i in (6, 7, 8)])
        arm = np.concatenate([points[i] for i in (2, 3, 4, 5)])
        return np.array([grip[:, 0].max(), grip[:, 1].min(), (grip @ self.normal).min(), (arm @ self.normal).min()])

    def check(self, q, target, width, target_width, samples=2):
        extents = []
        for fraction in np.linspace(0., 1., samples+1):
            ext = self.extremes(np.asarray(q)+fraction*(np.asarray(target)-q),
                                width+fraction*(target_width-width))
            extents.append(ext)
            bad = []
            if self.x_max is not None and ext[0] > self.x_max:
                bad.append(f'grip_x={ext[0]:.4f}m > {self.x_max:.4f}m')
            if self.y_min is not None and ext[1] < self.y_min:
                bad.append(f'grip_y={ext[1]:.4f}m < {self.y_min:.4f}m')
            if ext[2] < self.floor_offset+self.grip_floor:
                bad.append(f'grip_plane={ext[2]:.4f}m < {self.floor_offset+self.grip_floor:.4f}m')
            if ext[3] < self.floor_offset+self.link_floor:
                bad.append(f'arm_plane={ext[3]:.4f}m < {self.floor_offset+self.link_floor:.4f}m')
            if bad:
                raise ControlFault(f'Workspace geometry violation at path fraction {fraction:.3f}: '+', '.join(bad))
        return np.array(extents)
