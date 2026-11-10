"""Synchronized time-optimal joint-goal reference, before QP validation.

Optimality covers independent joint velocity/acceleration/jerk bounds and a
resting goal. Robot geometry, tracking and actual SDK dynamics can add delay.
"""
import numpy as np
from .control100 import ControlFault


class ZVDInputShaper:
    """Causal fractional-sample ZVD velocity shaper for posture trials only."""

    def __init__(self, dt, frequency_hz):
        if not np.isfinite(frequency_hz) or not 0 < frequency_hz < .5/dt:
            raise ValueError('Invalid input shaper frequency')
        # Undamped ZVD impulses: (0, 1/4), (T/2, 1/2), (T, 1/4).
        # Split each fractional delay between the two adjacent 200Hz ticks.
        delays = (0., .5/frequency_hz/dt, 1./frequency_hz/dt)
        self.taps = np.zeros(int(np.ceil(delays[-1]))+1)
        for delay, weight in zip(delays, (.25, .5, .25)):
            before = int(np.floor(delay))
            fraction = delay-before
            self.taps[before] += weight*(1.-fraction)
            if fraction:
                self.taps[before+1] += weight*fraction
        self.history = np.zeros((len(self.taps)-1, 6))
        self.frequency_hz = float(frequency_hz)
        self.delay_s = float(np.dot(np.arange(len(self.taps)), self.taps)*dt)

    def reset(self):
        self.history[:] = 0.

    def shape(self, samples):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 6 or not np.isfinite(samples).all():
            raise ValueError('Invalid velocity samples for input shaper')
        prior = self.history.shape[0]
        stream = np.concatenate((self.history, samples))
        result = sum(weight*stream[prior-index:prior-index+len(samples)]
                     for index, weight in enumerate(self.taps) if weight)
        self.history = stream[-prior:].copy()
        return result


class PostureGoal:
    def __init__(self, dt, speed, acceleration, jerk, steps):
        try:
            from ruckig import Ruckig, InputParameter, Trajectory, DurationDiscretization
        except ImportError as exc:
            raise ControlFault('home/fold require ruckig==0.19.4 in the SDK environment') from exc
        self.generator = Ruckig(6, dt)
        self.input = InputParameter(6)
        self.trajectory = Trajectory(6)
        self.input.max_velocity = [speed]*6
        self.input.max_acceleration = [acceleration]*6
        self.input.max_jerk = [jerk]*6
        self.input.duration_discretization = DurationDiscretization.Discrete
        self.input.target_velocity = [0.]*6
        self.input.target_acceleration = [0.]*6
        self.dt, self.steps = dt, steps

    def plan(self, state, goal, *, speed):
        state = np.asarray(state, float)
        goal = np.asarray(goal, float)
        if state.shape != (18,) or goal.shape != (6,) or not np.isfinite(state).all() or not np.isfinite(goal).all():
            raise ControlFault('Invalid posture reference or joint goal')
        self.input.current_position = state[:6].tolist()
        self.input.current_velocity = state[6:12].tolist()
        self.input.current_acceleration = state[12:].tolist()
        self.input.target_position = goal.tolist()
        self.input.max_velocity = [speed]*6
        result = self.generator.calculate(self.input, self.trajectory)
        if int(result) < 0:
            raise ControlFault('Time-optimal posture planning failed: '+str(result))
        samples = np.array([self.trajectory.at_time((i+1)*self.dt)[1] for i in range(self.steps)])
        if samples.shape != (self.steps, 6) or not np.isfinite(samples).all():
            raise ControlFault('Invalid time-optimal velocity samples')
        return samples, float(self.trajectory.duration)
