"""SHOW-108 checked arm plans and independent, policy-rate latched gripper.

Matches StageRewardPickPlace -> TernaryControlledPickPlace -> advance_arm.
The jaw reference changes by 2*gripper_rate once per policy interval. There
is no MOVE/BRAKE/GRIP coordinator, close-time arm freeze, or task override.
"""
import numpy as np
from .control100 import Controller100, ControlFault
from .control103 import Controller103
from .native103 import Planner103
from .show100 import vector


class Controller108(Controller100):
    post_state = Controller103.post_state
    stopped = Controller103.stopped
    _backup = Controller103._backup
    begin_brake = Controller103.begin_brake

    def __init__(self, model, environment, **kwargs):
        super().__init__(model, dict(environment), **kwargs)
        self.jerk = float(environment['control_jerk'])
        self.steps = int(round(self.policy_dt/self.dt))
        self.stop_steps = int(environment['smooth_stop_steps'])
        if (not np.allclose(model.tcp_local_position,[0.,0.,.1385],atol=1e-9)
                or not np.allclose([self.dt,self.policy_dt,self.speed,self.acceleration,self.jerk],
                                   [.005,.1,1.,5.,100.]) or self.stop_steps!=64):
            raise ValueError('SHOW-108 requires its exported TCP, periods and v1/a5/j100 checked-plan settings')
        self.planner=Planner103(model,[self.dt,self.speed,self.acceleration,self.jerk,
            self.lag,self.extra,self.margin,self.horizon,self.weight,.001,
            self.floor,*self.normal,2e-5,.01],self.steps,self.stop_steps)

    def reset(self,q,gripper_reference_width,final_goal_width):
        super().reset(q,gripper_reference_width,final_goal_width)
        self.reference_acceleration=np.zeros(6)
        self.plan=np.tile(self.post_state,(self.stop_steps,1))
        self.cursor=0;self.previous_target=np.zeros(6);self.last_qdot=np.zeros(6)
        self.measured_width=float(gripper_reference_width)
        self.finishing=False;self.candidate_rejections=0
        # Compatibility with observation logging only: this profile has no phase observation.
        self.phase=0
        self.plan_detail=np.zeros(5)

    def set_action(self,q,action,*,qdot=None,width=None):
        if not self.started or self.latched or self.finishing:
            raise ControlFault(self.latched or 'Controller not ready for a new action')
        try:
            q=vector(q,6,'q');action=vector(action,7,'action')
            qdot=self.last_qdot if qdot is None else vector(qdot,6,'qdot')
            if np.any(np.abs(action)>1):self.fail('Policy action outside [-1,1]')
            jac=self.model.jacobian(q)
            twist=action[:6]*np.array([self.e['max_lin_vel']]*3+[self.e['max_ang_vel']]*3)
            desired=np.clip(jac.T@np.linalg.solve(jac@jac.T+self.e['ik_damping']**2*np.eye(6),twist),-self.speed,self.speed)
            candidate,accepted,self.plan_detail=self.planner.plan(q,qdot,self.post_state,desired,self.previous_target)
            self.plan=candidate if accepted else self._backup()
            self.cursor=0;self.candidate_rejections+=int(not accepted)
            self.previous_target=self.target=desired.copy()
            if action[6]<=-self.e['gripper_threshold']:self.gripper_goal=0.
            elif action[6]>=self.e['gripper_threshold']:self.gripper_goal=2*self.e['gripper_max']
            rate=2*self.e['gripper_rate']
            self.gripper_reference=float(np.clip(self.gripper_goal,self.gripper_reference-rate,self.gripper_reference+rate))
            self.previous_action=action.astype(np.float32)
        except (ValueError,np.linalg.LinAlgError,ControlFault) as exc:self.fail(exc)

    def step(self,q,qdot,*,width=None):
        if not self.started or self.latched:raise ControlFault(self.latched or 'Controller not reset')
        try:
            q=vector(q,6,'q');qdot=vector(qdot,6,'qdot')
            if width is not None and (not np.isfinite(width) or not 0<=width<=.101):self.fail('Invalid measured gripper width')
            self.last_qdot=qdot.copy()
            st=self.plan[min(self.cursor,len(self.plan)-1)]
            jerk=(st[12:]-self.reference_acceleration)/self.dt
            distance=self.normal@self.model.forward(st[:6])['tcp_position']-self.floor
            if (not np.isfinite(st).all() or np.any(st[:6]<self.model.lower-2e-5)
                    or np.any(st[:6]>self.model.upper+2e-5) or distance < -2e-5
                    or np.max(np.abs(st[6:12]))>self.speed+1e-5
                    or np.max(np.abs(st[12:]))>self.acceleration+1e-3
                    or np.max(np.abs(jerk))>self.jerk+.2):self.fail('SHOW-108 reference constraint failed')
            if max(np.max(np.abs(self.reference-q)),np.max(np.abs(st[:6]-q)))>self.lag+self.extra+2e-5:
                self.fail('Actual tracking outer cap exceeded')
            self.reference,self.velocity,self.reference_acceleration=(part.copy()for part in np.split(st,3))
            self.cursor+=1
            return dict(joint_target_rad=self.reference.copy(),reference_velocity=self.velocity.copy(),
                reference_acceleration=self.reference_acceleration.copy(),reference_jerk=jerk,
                gripper_width_target_m=self.gripper_reference,gripper_final_goal_width_m=self.gripper_goal,
                gripper_actuated_goal_width_m=self.gripper_reference,coord_phase=0,coord_phase_name='INDEPENDENT',
                candidate_rejections=self.candidate_rejections,tracking_gap_rad=float(np.max(np.abs(st[:6]-q))))
        except (ValueError,ControlFault) as exc:self.fail(exc)
