"""SHOW-103 bounded plans and serialized gripping, without device access.

TCP remains the exported training TCP (138.5mm). Policy weights/actions are
unchanged. Reference constraints do not bound unmeasured physical disturbances.
"""
from collections import deque
import math
import numpy as np
from .control100 import Controller100, ControlFault
from .show100 import vector
from .native103 import Planner103

MOVE, BRAKE, GRIP = 0, 1, 2
PHASE_NAMES = ('MOVE', 'BRAKE', 'GRIP')
DEFAULTS = dict(max_joint_vel=.75, control_acceleration=4., control_jerk=300.,
                max_lin_vel=.05, max_ang_vel=.5, smooth_stop_steps=64,
                grip_arm_stable_speed=.02, grip_arm_stable_seconds=.05,
                grip_stable_speed=.005, grip_stable_seconds=.1,
                grip_timeout=3., brake_timeout=2., release_width_margin=.01)


class Controller103(Controller100):
    def __init__(self, model, environment, *, posture_mode=False, posture_jerk=80., posture_acceleration=5.,
                 posture_shaper_hz=None, **kwargs):
        # A 49D SHOW-100 actor can use this controller without changing weights.
        # A 50D SHOW-103 actor supplies its own resolved controller settings.
        env = dict(environment)
        for key, value in DEFAULTS.items():
            env[key] = environment.get(key, value) if environment.get('task') in (
                'show100_smooth_grip', 'gentle_placement', 'controlled_release', 'quadratic_release',
                'scripted_control') else value
        self.posture_mode = bool(posture_mode)
        if self.posture_mode:
            env['max_joint_vel'], env['control_acceleration'], env['control_jerk'] = 1., float(posture_acceleration), float(posture_jerk)
            if not np.isfinite([env['control_acceleration'], env['control_jerk']]).all() or min(env['control_acceleration'], env['control_jerk']) <= 0:
                raise ValueError('Posture acceleration and jerk must be finite and positive')
            # Longer jerk ramps need a longer checked stop than the training
            # controller's fixed320ms. Cover reachable velocity/acceleration.
            stop_seconds = env['max_joint_vel']/env['control_acceleration']+2*env['control_acceleration']/env['control_jerk']+4*env['sim_dt']
            env['smooth_stop_steps'] = max(64, math.ceil(stop_seconds/env['sim_dt']))
        super().__init__(model, env, **kwargs)
        if not np.allclose(model.tcp_local_position, [0, 0, .1385], atol=1e-9):
            raise ValueError('SHOW-103 deployment requires the unchanged 138.5mm TCP')
        for key in DEFAULTS:
            if not np.isfinite(env[key]) or env[key] <= 0:
                raise ValueError('Invalid SHOW-103 setting: '+key)
        self.jerk = float(env['control_jerk'])
        self.steps = int(round(self.policy_dt/self.dt))
        self.stop_steps = int(env['smooth_stop_steps'])
        if (not np.isclose(self.steps*self.dt, self.policy_dt) or self.stop_steps != env['smooth_stop_steps']
                or not np.isclose(self.dt, .005) or not np.isclose(self.policy_dt, .1)):
            raise ValueError('SHOW-103 deployment requires 200Hz control / 10Hz policy')
        self.planner = Planner103(model, [self.dt, self.speed, self.acceleration, self.jerk,
            self.lag, self.extra, self.margin, self.horizon, self.weight, .001,
            self.floor, *self.normal, 2e-5, .01], self.steps, self.stop_steps)
        self.goal_planner = None
        self.input_shaper = None
        if self.posture_mode:
            from .posture_goal import PostureGoal, ZVDInputShaper
            self.goal_planner = PostureGoal(self.dt, self.speed, self.acceleration, self.jerk, self.steps)
            if posture_shaper_hz is not None:
                self.input_shaper = ZVDInputShaper(self.dt, posture_shaper_hz)
        elif posture_shaper_hz is not None:
            raise ValueError('Input shaper is restricted to home/fold posture mode')

    @property
    def post_state(self):
        return np.r_[self.reference, self.velocity, self.reference_acceleration]

    @property
    def stopped(self):
        return np.max(np.abs(self.velocity)) < 1e-5 and np.max(np.abs(self.reference_acceleration)) < 1e-3

    def reset(self, q, gripper_reference_width, final_goal_width):
        super().reset(q, gripper_reference_width, final_goal_width)
        self.reference_acceleration = np.zeros(6)
        self.phase, self.timer, self.arm_stable = MOVE, 0., 0.
        self.actuated_goal = float(final_goal_width)
        self.start_width = self.measured_width = float(gripper_reference_width)
        self.window = deque([self.measured_width]*int(round(self.e['grip_stable_seconds']/self.dt)),
                            maxlen=int(round(self.e['grip_stable_seconds']/self.dt)))
        self.plan = np.tile(self.post_state, (self.stop_steps, 1))
        self.cursor = 0
        self.last_qdot = np.zeros(6)
        self.previous_target = np.zeros(6)
        self.finishing, self.opening_unlocked = False, False
        self.operations_completed, self.candidate_rejections = 0, 0
        self.plan_detail = np.array([1, 1, np.nan, 0, 0])
        self.nominal_goal_duration_s = None
        self.shaper_settle_bypass = False
        if self.input_shaper is not None:
            self.input_shaper.reset()

    def _backup(self):
        remaining = self.plan[self.cursor:]
        if not len(remaining):
            remaining = self.plan[-1:]
        count = self.steps+self.stop_steps
        return np.concatenate([remaining[:count], np.repeat(remaining[-1:], max(0, count-len(remaining)), axis=0)])

    def set_joint_goal(self, q, goal, *, qdot=None, width=None, speed_cap=None, gain=2., gripper_action=1.):
        """Position-goal interface; retain the same QP, stop plans and interlocks."""
        try:
            goal = vector(goal, 6, 'joint goal')
            if speed_cap is None:
                speed_cap = self.speed if self.posture_mode else .15
            if np.any(goal < self.model.lower) or np.any(goal > self.model.upper):
                self.fail('Joint goal outside model limits')
            if not np.isfinite([speed_cap, gain]).all() or not 0 < speed_cap <= self.speed or gain <= 0:
                self.fail('Invalid position-goal servo parameters')
            desired = np.clip(gain*(goal-self.reference), -speed_cap, speed_cap)
            samples = None
            if self.goal_planner is not None:
                samples, self.nominal_goal_duration_s = self.goal_planner.plan(self.post_state, goal, speed=speed_cap)
                if self.input_shaper is not None and np.max(np.abs(goal-self.reference)) < .05:
                    # At a joint boundary, delayed velocity pulses can make
                    # the checked stop infeasible. Finish with the unshaped
                    # Ruckig reference and keep that choice until reset.
                    self.shaper_settle_bypass = True
                    self.input_shaper.reset()
                if self.input_shaper is not None and not self.shaper_settle_bypass:
                    samples = self.input_shaper.shape(samples)
                # The nominal continuous-time plan can briefly recover from
                # a discrete reference state near a velocity bound. Only the
                # QP output is sent; its hard limits are never relaxed.
                desired = np.clip(samples[0], -self.speed, self.speed)
            self.set_action(q, [0, 0, 0, 0, 0, 0, gripper_action], qdot=qdot, width=width,
                            _joint_velocity=desired, _velocity_samples=samples)
        except (ValueError, ControlFault) as exc:
            self.fail(exc)

    def set_action(self, q, action, *, qdot=None, width=None, _joint_velocity=None, _velocity_samples=None):
        if not self.started or self.latched or self.finishing:
            raise ControlFault(self.latched or 'Controller not ready for a new action')
        try:
            q, action = vector(q, 6, 'q'), vector(action, 7, 'action')
            if np.any(np.abs(action) > 1):
                self.fail('Policy action outside [-1,1]')
            qdot = self.last_qdot if qdot is None else vector(qdot, 6, 'qdot')
            width = self.measured_width if width is None else float(width)
            if not np.isfinite(width) or not 0 <= width <= .101:
                self.fail('Invalid measured gripper width')
            self.measured_width = width
            if self.phase == MOVE:
                if action[6] <= -self.e['gripper_threshold']:
                    self.gripper_goal = 0.
                elif action[6] >= self.e['gripper_threshold']:
                    self.gripper_goal = 2*self.e['gripper_max']
            request = self.phase == MOVE and self.gripper_goal != self.actuated_goal
            allowed = self.phase == MOVE or (self.phase == GRIP and self.actuated_goal > 0
                and width >= self.start_width+self.e['release_width_margin'])
            normal = not request and allowed
            self.opening_unlocked = normal and self.phase == GRIP
            if _joint_velocity is None:
                jac = self.model.jacobian(q)
                twist = action[:6]*np.array([self.e['max_lin_vel']]*3+[self.e['max_ang_vel']]*3)
                desired = np.clip(jac.T @ np.linalg.solve(jac @ jac.T+self.e['ik_damping']**2*np.eye(6), twist),
                                  -self.speed, self.speed)
            else:
                desired = vector(_joint_velocity, 6, 'position servo velocity')
                if np.any(np.abs(desired) > self.speed):
                    self.fail('Position servo velocity exceeds QP limit')
            backup = self._backup()
            if normal:
                if _velocity_samples is None:
                    candidate, accepted, self.plan_detail = self.planner.plan(
                        q, qdot, self.post_state, desired, self.previous_target)
                else:
                    candidate, accepted, self.plan_detail = self.planner.plan_samples(q, qdot, self.post_state, _velocity_samples)
                self.plan = candidate if accepted else backup
                self.candidate_rejections += int(not accepted)
            else:
                self.plan = backup
            self.cursor = 0
            if request:
                self.phase, self.timer = BRAKE, 0.
            self.previous_target = desired.copy() if normal else np.zeros(6)
            self.target = self.previous_target.copy()
            self.previous_action = action.astype(np.float32)
        except (ValueError, np.linalg.LinAlgError, ControlFault) as exc:
            self.fail(exc)

    def begin_brake(self):
        # Continue the already-checked remaining trajectory and its full stop.
        # No new gripper stroke or reference ramp after the requested run ends.
        self.finishing = True

    def step(self, q, qdot, *, width=None):
        if not self.started or self.latched:
            raise ControlFault(self.latched or 'Controller not reset')
        try:
            q, qdot = vector(q, 6, 'q'), vector(qdot, 6, 'qdot')
            width = self.measured_width if width is None else float(width)
            if not np.isfinite(width) or not 0 <= width <= .101:
                self.fail('Invalid measured gripper width')
            self.measured_width, self.last_qdot = width, qdot.copy()
            if np.max(np.abs(self.reference-q)) > self.lag+self.extra+2e-5:
                self.fail('Actual tracking outer cap exceeded')
            st = self.plan[min(self.cursor, len(self.plan)-1)]
            jerk = (st[12:]-self.reference_acceleration)/self.dt
            distance = self.normal @ self.model.forward(st[:6])['tcp_position']-self.floor
            if (not np.isfinite(st).all() or np.any(st[:6] < self.model.lower-2e-5)
                    or np.any(st[:6] > self.model.upper+2e-5) or distance < -2e-5
                    or np.max(np.abs(st[6:12])) > self.speed+1e-5
                    or np.max(np.abs(st[12:])) > self.acceleration+1e-3
                    or np.max(np.abs(jerk)) > self.jerk+.2):
                self.fail('SHOW-103 reference constraint failed')
            if np.max(np.abs(st[:6]-q)) > self.lag+self.extra+2e-5:
                self.fail('Actual tracking outer cap exceeded by planned reference')
            self.reference, self.velocity, self.reference_acceleration = (part.copy() for part in np.split(st, 3))
            self.cursor += 1
            if not self.finishing:
                settled = np.max(np.abs(st[6:])) < 1e-4 and np.max(np.abs(qdot)) <= self.e['grip_arm_stable_speed']
                self.arm_stable = self.arm_stable+self.dt if self.phase == BRAKE and settled else 0.
                if self.phase == BRAKE and self.arm_stable >= self.e['grip_arm_stable_seconds']-1e-6:
                    self.phase, self.timer = GRIP, 0.
                    self.actuated_goal, self.start_width = self.gripper_goal, width
                if self.phase == GRIP:
                    rate = 2*self.e['gripper_rate']*self.dt/self.policy_dt
                    self.gripper_reference = float(np.clip(self.actuated_goal,
                        self.gripper_reference-rate, self.gripper_reference+rate))
                    self.window.append(width)
                    stable = np.ptp(self.window) <= self.e['grip_stable_speed']*self.e['grip_stable_seconds']
                    maximum = 2*self.e['gripper_max']
                    progressed = (width >= self.start_width+.002 or abs(width-maximum) < .002) if self.actuated_goal > maximum/2 else width < maximum-.002
                    done = (stable and self.timer+self.dt >= self.e['grip_stable_seconds']-1e-6
                            and abs(self.gripper_reference-self.actuated_goal) < 2e-6 and progressed)
                    if done:
                        self.phase = MOVE
                        self.operations_completed += 1
                else:
                    self.window = deque([width]*self.window.maxlen, maxlen=self.window.maxlen)
                self.timer = 0. if self.phase == MOVE else self.timer+self.dt
                if ((self.phase == GRIP and self.timer > self.e['grip_timeout'])
                        or (self.phase == BRAKE and self.timer > self.e['brake_timeout'])):
                    self.fail('SHOW-103 gripper/brake timeout')
            return dict(joint_target_rad=self.reference.copy(), reference_velocity=self.velocity.copy(),
                        reference_acceleration=self.reference_acceleration.copy(), reference_jerk=jerk,
                        gripper_width_target_m=self.gripper_reference, gripper_final_goal_width_m=self.gripper_goal,
                        gripper_actuated_goal_width_m=self.actuated_goal,
                        coord_phase=self.phase, coord_phase_name=PHASE_NAMES[self.phase],
                        coord_timer_s=self.timer, opening_motion_unlocked=self.opening_unlocked,
                        operations_completed=self.operations_completed, candidate_rejections=self.candidate_rejections,
                        tracking_gap_rad=float(np.max(np.abs(self.reference-q))))
        except (ValueError, np.linalg.LinAlgError, ControlFault) as exc:
            self.fail(exc)
