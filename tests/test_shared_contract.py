import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from piper_runtime.contracts import layout,observation_layout,PROFILES,contract
from piper_runtime.identity import identity,verify
from piper_runtime.resources import MODEL_XML,GEOMETRY
from piper_policy_runtime.kinematics import PiperKinematics

def test_training_layout_and_legacy_adapter_agree():
    assert observation_layout(dict(stage_obs=False,size_obs=True,gripper_goal_obs=True))==layout(49)
    assert layout(49)[-1]['start']==42 and layout(50)[-1]['start']==43
    assert PROFILES['show103-stage-reward-v1']['arm_hold_while_gripping']is False

def test_profile_is_never_silently_substituted():
    with pytest.raises(ValueError,match='프로필'):contract({'runtime_profile_id':'show103-stage-reward-v1'})

def test_resources_and_content_lock(tmp_path):
    assert MODEL_XML.is_file()and GEOMETRY.is_file();assert PiperKinematics(MODEL_XML).lower.shape==(6,)
    path=tmp_path/'lock.json';value=identity();path.write_text(json.dumps(value));assert verify(path)==value
    value['sha256']='0'*64;path.write_text(json.dumps(value))
    with pytest.raises(ValueError):verify(path)

def test_core_import_does_not_load_gpu_or_connect_a_device():
    code="import sys;from piper_runtime import contracts;from piper_policy_runtime import Policy;from piper_runtime.transports.robot_io import MockRobot;assert not any(n in sys.modules for n in ('jax','torch','pyAgxArm','can','pyrealsense2'))"
    subprocess.run([sys.executable,'-c',code],check=True)
