"""NumPy-only Piper kinematics and simulation-equivalent command proposals.

This module never connects to a robot.  Proposals are radians/metres and are
deliberately not Piper SDK or CAN messages.  The model must be the explicit
fixed-base, six-hinge Piper chain used during training; unknown transforms fail
at load time rather than silently changing the observation/action convention.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return result


def _attribute(element, name, default, size):
    # Split/float rejects trailing garbage instead of np.fromstring's partial
    # parse, which could otherwise turn a malformed transform into a valid one.
    return _vector([float(value) for value in element.get(name, default).split()], size, name)


def _multiply(left, right):
    w, x, y, z = left
    a, b, c, d = right
    return np.array([w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                     w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a])


def _matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def quaternion_to_rotvec(quaternion):
    """Training's wxyz convention, including its noncanonical quaternion sign.

    In particular, do not flip a negative w: training's observation conversion
    can produce angles above pi.  This is intentional parity, not a principal
    rotation-vector conversion.
    """
    quaternion = _vector(quaternion, 4, "quaternion")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6):
        raise ValueError("quaternion must have unit norm")
    axis = quaternion[1:]
    sine = np.linalg.norm(axis)
    scale = (2*np.arctan2(sine, np.clip(quaternion[0], -1., 1.)) / sine
             if sine > 1e-8 else 2.)
    return axis * scale


class PiperKinematics:
    """Load body/joint transforms from an explicit training ``piper.xml``.

    ``forward`` and ``jacobian`` use the model's world frame.  The supported
    deployment model requires an identity fixed base, making this exactly the
    robot base frame used by the trained actor.  Body quaternions are normalized
    as in MuJoCo, and hinge position, axis and reference angle are respected.
    """
    tcp_local_position = np.array([0., 0., .12])

    def __init__(self, xml_path):
        self.xml_path = Path(xml_path)
        root = ET.parse(self.xml_path).getroot()
        self.robot_model = 'piper_v100' if root.get('model') == 'piper_v100' else 'menagerie'
        self.tcp_local_position = np.array([0., 0., .1358 if self.robot_model == 'piper_v100' else .12])
        if root.tag != "mujoco" or root.find("include") is not None:
            raise ValueError("Expected a self-contained MuJoCo Piper XML")
        compiler = root.find("compiler")
        if compiler is None or compiler.get("angle", "degree") != "radian":
            raise ValueError("Only explicit radian XML is supported")
        if compiler.get("coordinate", "local") != "local":
            raise ValueError("Only local body coordinates are supported")
        defaults = {"": {}}

        def read_defaults(element, inherited):
            values = dict(inherited)
            joint = element.find("joint")
            if joint is not None:
                values.update(joint.attrib)
            defaults[element.get("class", "")] = values
            for child in element.findall("default"):
                read_defaults(child, values)

        for element in root.findall("default"):
            read_defaults(element, {})

        def transform(body):
            if any(name in body.attrib for name in ("euler", "axisangle", "xyaxes", "zaxis")):
                raise ValueError("Only explicit body quaternion orientations are supported")
            if body.get("mocap", "false") != "false" or body.find("freejoint") is not None:
                raise ValueError("The Piper chain must be fixed-base with hinge joints")
            position = _attribute(body, "pos", "0 0 0", 3)
            quaternion = _attribute(body, "quat", "1 0 0 0", 4)
            norm = np.linalg.norm(quaternion)
            if norm < 1e-12:
                raise ValueError("Body quaternion must be nonzero")
            return position, quaternion / norm

        worldbody = root.find("worldbody")
        # The vendor v100 XML wraps base_link in an identity mounting body.
        if worldbody is not None and self.robot_model == 'piper_v100':
            mount = worldbody.find("body[@name='mount']")
            if mount is None or mount.findall('joint'):
                raise ValueError('Expected the fixed vendor mount')
            mount_position, mount_quaternion = transform(mount)
            if not np.array_equal(mount_position, np.zeros(3)) or not np.array_equal(mount_quaternion, [1., 0., 0., 0.]):
                raise ValueError('Vendor mount must have an identity transform')
            worldbody = mount
        base = None if worldbody is None else worldbody.find("body[@name='base_link']")
        if base is None or base.findall("joint"):
            raise ValueError("Expected a fixed base_link directly under worldbody")
        base_position, base_quaternion = transform(base)
        if not np.array_equal(base_position, np.zeros(3)) or not np.array_equal(base_quaternion, [1., 0., 0., 0.]):
            raise ValueError("Deployment model base_link must have an identity transform")
        self._bodies = []
        self._joints = []
        body = base
        inherited_class = base.get("childclass", "")
        for index in range(1, 7):
            body = body.find(f"body[@name='link{index}']")
            if body is None:
                raise ValueError(f"Missing direct-chain link{index}")
            inherited_class = body.get("childclass", inherited_class)
            self._bodies.append(transform(body))
            joints = body.findall("joint")
            if len(joints) != 1 or joints[0].get("name") != f"joint{index}":
                raise ValueError(f"Expected only joint{index} on link{index}")
            joint = joints[0]
            class_name = joint.get("class", inherited_class)
            if class_name not in defaults:
                raise ValueError(f"Unknown joint default class: {class_name}")
            attributes = dict(defaults[class_name], **joint.attrib)
            if attributes.get("type", "hinge") != "hinge":
                raise ValueError("Only hinge joints are supported")
            if attributes.get("limited", "auto") not in ("true", "auto"):
                raise ValueError("All six arm joints must have finite limits")
            resolved = ET.Element("joint", attributes)
            position = _attribute(resolved, "pos", "0 0 0", 3)
            axis = _attribute(resolved, "axis", "0 0 1", 3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-12:
                raise ValueError("Joint axis must be nonzero")
            bounds = _attribute(resolved, "range", "", 2)
            if bounds[0] >= bounds[1]:
                raise ValueError("Joint range must be increasing")
            reference = float(resolved.get("ref", "0"))
            if not np.isfinite(reference):
                raise ValueError("Joint reference must be finite")
            self._joints.append((position, axis / axis_norm, reference, bounds))
        self.lower = np.array([joint[3][0] for joint in self._joints])
        self.upper = np.array([joint[3][1] for joint in self._joints])

    def _chain(self, q):
        q = _vector(q, 6, "q")
        position, quaternion = np.zeros(3), np.array([1., 0., 0., 0.])
        origins, axes = [], []
        for angle, (offset, body_quaternion), (pivot, axis, reference, _) in zip(q, self._bodies, self._joints):
            position = position + _matrix(quaternion) @ offset
            quaternion = _multiply(quaternion, body_quaternion)
            rotation = _matrix(quaternion)
            origin = position + rotation @ pivot
            origins.append(origin)
            axes.append(rotation @ axis)
            half = .5 * (angle - reference)
            quaternion = _multiply(quaternion, np.r_[np.cos(half), axis*np.sin(half)])
            # Hinge rotation is about the joint pivot, not necessarily body origin.
            position = origin - _matrix(quaternion) @ pivot
        tcp = position + _matrix(quaternion) @ self.tcp_local_position
        return tcp, quaternion, np.asarray(origins), np.asarray(axes)

    def forward(self, q):
        """Return TCP xyz in metres and link6 quaternion in wxyz order."""
        position, quaternion, _, _ = self._chain(q)
        return {"tcp_position": position, "link6_quaternion_wxyz": quaternion}

    def jacobian(self, q):
        """Return a (6, 6) geometric Jacobian: linear xyz then angular xyz."""
        position, _, origins, axes = self._chain(q)
        return np.concatenate((np.cross(axes, position-origins).T, axes.T), axis=0)

    def propose(self, q, current_joint_target, action, *, ctrl_dt=.1,
                max_lin_vel=.10, max_ang_vel=1., ik_damping=.05,
                max_joint_vel=1.5, max_target_lag=.15,
                current_gripper_finger_target=0., gripper_max=.035,
                gripper_rate=.01, gripper_binary=True):
        """Apply the trained environment's velocity IK and gripper target rule.

        ``current_joint_target`` must be the servo's current position target,
        not an assumed zero or measured q.  The gripper target is a single
        simulated finger's travel; full jaw opening is twice that value.  No
        physical effort conversion is implied by these geometric proposals.
        """
        q = _vector(q, 6, "q")
        target = _vector(current_joint_target, 6, "current_joint_target")
        action = _vector(action, 7, "action")
        parameters = np.asarray([ctrl_dt, max_lin_vel, max_ang_vel, ik_damping,
                                 max_joint_vel, max_target_lag, gripper_max, gripper_rate], float)
        if not np.isfinite(parameters).all() or np.any(parameters <= 0):
            raise ValueError("Command scale, time, damping and bounds must be finite and positive")
        previous = float(current_gripper_finger_target)
        if not np.isfinite(previous) or not 0 <= previous <= gripper_max:
            raise ValueError("Current gripper finger target must be within simulated travel limits")
        action = np.clip(action, -1., 1.)
        twist = np.r_[action[:3]*max_lin_vel, action[3:6]*max_ang_vel]
        jacobian = self.jacobian(q)
        velocity = jacobian.T @ np.linalg.solve(jacobian@jacobian.T + ik_damping**2*np.eye(6), twist)
        velocity = np.clip(velocity, -max_joint_vel, max_joint_vel)
        proposed = target + velocity*ctrl_dt
        proposed = np.clip(proposed, q-max_target_lag, q+max_target_lag)
        proposed = np.clip(proposed, self.lower, self.upper)
        desired = (gripper_max if action[6] > 0 else 0.) if gripper_binary else gripper_max*(action[6]+1.)/2.
        gripper = float(np.clip(desired, previous-gripper_rate, previous+gripper_rate))
        return {
            "normalized_action": action,
            "tcp_twist_mps_radps": twist,
            "joint_velocity_radps": velocity,
            "joint_target_rad": proposed,
            "gripper_finger_target_m": gripper,
            "gripper_width_target_m": 2*gripper,
            "hardware_commands_sent": False,
            "physical_effort_mapping": "not_calibrated",
        }
