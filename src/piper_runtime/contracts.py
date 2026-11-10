"""Versioned observation/action and explicit legacy execution profile."""
import math

PROFILE={'controller':'show103_serialized','max_joint_vel':1.,'control_acceleration':5.,'control_jerk':100.,
    'tcp_local_position_m':[0.,0.,.1385],'tcp_floor_clearance_m':.01,'gripper_force':.5,'control_dt_s':.005,'policy_dt_s':.1}
ACTION_CONTRACT='normalized [-1,1]: base-frame linear/angular twist commands (6) and gripper action (1); NOT direct joint/CAN commands'
SHOW108_PROFILE_ID='show108-mlp-lstm-v1'
SHOW108_PROFILE=dict(PROFILE,controller='show108_checked_plan_independent',arm_hold_while_gripping=False,
    gripper_reference_rule='latched final goal; full-width reference changes by <=20mm once per 100ms policy packet',
    max_lin_vel=.1,max_ang_vel=.5)

def layout(size):
    fields=[('joint_position_rad',6),('joint_velocity_rad_s',6),('servo_target_minus_joint_rad',6),
        ('tcp_position_base_m',3),('tcp_rotation_vector_base_rad',3),('gripper_full_width_m',1),
        ('block_position_base_m',3),('block_rotation_vector_base_rad',3),('bowl_position_base_m',3),
        ('block_minus_tcp_m',3),('block_minus_bowl_m',3),('perceived_block_half_size_m',1),('gripper_final_goal_normalized',1)]
    if size==50:fields.append(('controller_phase_normalized',1))
    elif size!=49:raise ValueError('지원하는 관측은 SHOW-100 49차원 또는 SHOW-103 phase 포함 50차원입니다.')
    fields.append(('previous_normalized_policy_action',7));result=[];offset=0
    for name,width in fields:result.append({'name':name,'start':offset,'stop':offset+width});offset+=width
    return result

def contract(meta):
    profile=meta.get('runtime_profile_id','studio-legacy-serialized-v1')
    if profile not in ('studio-legacy-serialized-v1',SHOW108_PROFILE_ID):raise ValueError('이 정책의 제어 프로필에는 별도 실기 어댑터가 필요합니다. 이전 순차 제어기로 대체하지 않습니다.')
    modern=profile==SHOW108_PROFILE_ID
    if meta.get('schema_version')!=1 or meta.get('actor')not in (('mlp_lstm',)if modern else ('mlp','lstm')):raise ValueError('지원하지 않는 정책 형식입니다.')
    size=meta.get('observation_size')
    if meta.get('action_size')!=7 or meta.get('observation_layout')!=layout(size):raise ValueError('정책의 관측 필드 순서 또는 행동 차원이 어댑터와 다릅니다.')
    if meta.get('action_contract')!=ACTION_CONTRACT:raise ValueError('정책 행동의 좌표계·단위·의미가 다릅니다.')
    env=meta['environment'];kin=meta['kinematics']
    if kin.get('observation_frame')!='base_link'or env.get('robot_model')!='piper_v100':raise ValueError('Piper 베이스 좌표계 정책이 필요합니다.')
    if kin.get('tcp_local_position_m')!=PROFILE['tcp_local_position_m']or not math.isclose(float(env.get('tcp_local_z',0)),.1385):raise ValueError('TCP 138.5mm와 일치하는 정책이 필요합니다.')
    if env.get('stage_obs',True)or not env.get('size_obs')or not env.get('gripper_goal_obs'):raise ValueError('정책 관측 옵션이 어댑터와 다릅니다.')
    if env.get('gripper_mode')!='ternary_latched'or (not modern and ('direct_gripper'in env.get('task','')or env.get('independent_gripper'))):
        raise ValueError('독립/직접 그리퍼 정책에는 별도 어댑터가 필요합니다.')
    if not math.isclose(float(env['sim_dt']),.005)or not math.isclose(float(env['ctrl_dt']),.1):raise ValueError('200Hz 제어 / 10Hz 정책 계약과 다릅니다.')
    if modern:
        expected=dict(max_joint_vel=1.,control_acceleration=5.,control_jerk=100.,max_lin_vel=.1,max_ang_vel=.5,
            ik_damping=.05,max_target_lag=.15,tracking_extra=.02,tracking_margin=.02,tracking_horizon=.05,
            tracking_weight=100.,smooth_stop_steps=64,gripper_rate=.01,gripper_max=.05,gripper_threshold=.5,
            control_floor=0.,base_height=.015)
        for key,value in expected.items():
            if not math.isclose(float(env.get(key,float('nan'))),value,rel_tol=0,abs_tol=1e-9):raise ValueError('SHOW-108 학습 제어 계약 불일치: '+key)
        if (size!=49 or meta.get('encoder_sizes')!=[256,64]or meta.get('recurrent_width')!=128
                or env.get('task')!='stage_reward_pick_place_five_starts_v2'):
            raise ValueError('SHOW-108 구조 또는 과제 계약이 다릅니다.')
        return {'adapter_id':'piper-show108-49-v1','observation_size':49,'action_size':7,
            'runtime_profile':dict(SHOW108_PROFILE),'training_runtime_limit_differences':{},'phase_definition':None,
            'deployment_overrides':{'tcp_floor_clearance_m':{'training':0.,'runtime':.01},
                'gripper_force_sdk':{'training':'MuJoCo actuator dynamics','runtime':.5}},
            'arm_gripper_coordinator':False}
    differences={key:{'training':env.get(key),'runtime':PROFILE[key]}for key in ('max_joint_vel','control_acceleration','control_jerk')if env.get(key)!=PROFILE[key]}
    return {'adapter_id':'piper-show103-phase50-v1'if size==50 else 'piper-show100-49-v1','observation_size':size,
        'action_size':7,'runtime_profile':PROFILE,'training_runtime_limit_differences':differences,
        'phase_definition':{'MOVE':0.,'BRAKE':.5,'GRIP':1.}if size==50 else None}


def observation_layout(env):
    fields=[('joint_position_rad',6),('joint_velocity_rad_s',6),('servo_target_minus_joint_rad',6),
        ('tcp_position_base_m',3),('tcp_rotation_vector_base_rad',3),('gripper_full_width_m',1),
        ('block_position_base_m',3),('block_rotation_vector_base_rad',3),('bowl_position_base_m',3),
        ('block_minus_tcp_m',3),('block_minus_bowl_m',3)]
    if env.get('stage_obs',True):fields.append(('stage_one_hot',4))
    if env.get('size_obs',False):fields.append(('perceived_block_half_size_m',1))
    if env.get('gripper_goal_obs',False):fields.append(('gripper_final_goal_normalized',1))
    fields.append(('previous_normalized_policy_action',7))
    layout=[];offset=0
    for name,width in fields:
        layout.append({'name':name,'start':offset,'stop':offset+width});offset+=width
    return layout

def tcp_local_position(model_position,env):
    tcp=list(model_position)
    if 'tcp_local_z' in env:tcp[2]=float(env['tcp_local_z'])
    return tcp

PROFILES = {
 SHOW108_PROFILE_ID:dict(SHOW108_PROFILE),
 "studio-legacy-serialized-v1": dict(PROFILE),
 "show103-stage-reward-v1": {"controller":"jax_checked_plan_v1", "actor_observation_size":49,"critic_observation_size":66,"action_size":7,"max_joint_vel":1.,"control_acceleration":5.,"control_jerk":100.,"tcp_local_position_m":[0.,0.,.1385],"control_dt_s":.005,"policy_dt_s":.1,"max_lin_vel":.1,"max_ang_vel":.5,"arm_hold_while_gripping":False,"status":"reward formula implemented; new MJX task/physics validation pending"}
}
