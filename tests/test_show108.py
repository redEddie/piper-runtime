"""Original Flax/JAX fixtures; no training dependencies or devices at test time."""
from pathlib import Path
import json
import numpy as np
import pytest
from piper_policy_runtime.check import check
from piper_policy_runtime.kinematics import PiperKinematics
from piper_policy_runtime.native103 import Planner103
from piper_policy_runtime.control108 import Controller108
from piper_runtime.resources import MODEL_XML
from piper_runtime.contracts import contract,SHOW108_PROFILE_ID

DATA=Path(__file__).parent/'data'

def test_original_flax_encoder_then_lstm_actions_logits_and_memory():
    report=check(DATA/'mlp-lstm-flax')
    assert report['max_action_abs_error']<1e-5
    assert max(report['state_and_logit_max_abs_error'].values())<1e-5

def test_original_jax_checked_plans_and_accept_reject_decisions():
    model=PiperKinematics(MODEL_XML);model.tcp_local_position=np.array([0.,0.,.1385])
    with np.load(DATA/'show108-control-golden.npz')as data:
        planner=Planner103(model,data['settings'])
        for i in range(len(data['q'])):
            actual,accepted,_=planner.plan(*(data[key][i]for key in ('q','qdot','state','target','old_target')),int(data['hold'][i]))
            assert accepted==bool(data['accepted'][i])
            np.testing.assert_allclose(actual[:,:6],data['plans'][i,:,:6],atol=2e-6,rtol=0)
            np.testing.assert_allclose(actual,data['plans'][i],atol=2e-4,rtol=0)

def environment():
    return dict(task='stage_reward_pick_place_five_starts_v2',sim_dt=.005,ctrl_dt=.1,robot_model='piper_v100',
        max_joint_vel=1.,control_acceleration=5.,control_jerk=100.,max_lin_vel=.1,max_ang_vel=.5,ik_damping=.05,
        max_target_lag=.15,tracking_extra=.02,tracking_margin=.02,tracking_horizon=.05,tracking_weight=100.,
        tracking_timeout=.5,smooth_stop_steps=64,gripper_rate=.01,gripper_max=.05,gripper_threshold=.5,
        gripper_mode='ternary_latched',control_mode='qp',control_floor=0.,base_height=.015,tcp_local_z=.1385,
        stage_obs=False,size_obs=True,gripper_goal_obs=True)

def test_close_and_hold_do_not_gate_arm_or_replace_latched_goal():
    model=PiperKinematics(MODEL_XML);model.tcp_local_position=np.array([0.,0.,.1385])
    c=Controller108(model,environment());q=np.array([0.,.523,-.45,0.,0.,0.]);c.reset(q,.1,.1)
    for goal,width in [(-1.,.08),(0.,.06),(-1.,.04),(0.,.02),(0.,0.)]:
        c.set_action(q,[.2,0.,0.,0.,0.,0.,goal],qdot=np.zeros(6),width=.1)
        assert c.gripper_goal==0.
        assert c.gripper_reference==pytest.approx(width)
        for _ in range(20):
            result=c.step(q,np.zeros(6),width=.1)
            assert result['coord_phase_name']=='INDEPENDENT'
            assert result['gripper_width_target_m']==pytest.approx(width)
            q=result['joint_target_rad'].copy()
    assert np.max(np.abs(c.velocity))>0
    c.begin_brake()
    for _ in range(64):q=c.step(q,np.zeros(6),width=.1)['joint_target_rad']
    assert c.stopped

def test_profile_cannot_silently_use_legacy_controller():
    from piper_runtime.contracts import layout,ACTION_CONTRACT
    meta=dict(schema_version=1,actor='mlp_lstm',runtime_profile_id=SHOW108_PROFILE_ID,
        encoder_sizes=[256,64],recurrent_width=128,observation_size=49,action_size=7,
        observation_layout=layout(49),action_contract=ACTION_CONTRACT,environment=environment(),
        kinematics=dict(observation_frame='base_link',tcp_local_position_m=[0.,0.,.1385]))
    assert contract(meta)['arm_gripper_coordinator']is False
    meta['environment']['gripper_rate']=.02
    with pytest.raises(ValueError,match='gripper_rate'):contract(meta)
    meta['runtime_profile_id']='studio-legacy-serialized-v1'
    with pytest.raises(ValueError):contract(meta)
