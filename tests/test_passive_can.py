"""Feedback decoding and camera-time selection, with no hardware access."""
import struct

import pytest


from piper_runtime.transports import passive_can
capture = vars(passive_can)


def record(stamp, data):
    return {"host_monotonic_ns": stamp, "host_wall_time_ns": stamp + 1_000_000, "data": data}


def test_feedback_uses_only_past_samples_and_marks_missing_and_stale():
    now = 1_000_000_000
    records = {
        0x2A5: [record(now - 20_000_000, struct.pack(">ii", 90000, -45000)),
                record(now + 1, struct.pack(">ii", 0, 0))],
        0x251: [record(now - 200_000_000, struct.pack(">h", -1234))],
        0x2A8: [record(now - 10_000_000, struct.pack(">ihBB", 96300, 100, 64, 0))],
    }
    result = capture["select_feedback"](records, {0x2A5: 2}, now)
    assert result["q_rad"][:2] == pytest.approx([1.5707963267948966, -0.7853981633974483])
    assert result["q_rad"][2:] == [None] * 4
    assert result["qvel_rad_s"][0] == -1.234
    assert result["gripper_full_width_m"] == 0.0963
    assert result["feedback_components"]["0x251"]["stale"] is True
    assert result["feedback_components"]["0x2A6"]["missing"] is True
    assert result["telemetry_fresh_at_frame"] is False
    assert result["hardware_commands_sent"] is False


def test_complete_fresh_feedback_and_no_frame_reference():
    now = 1_000_000_000
    records = {can_id: [record(now, b"\x00" * 8)] for can_id in capture["FEEDBACK_IDS"]}
    result = capture["select_feedback"](records, {}, now)
    assert result["telemetry_fresh_at_frame"] is True
    missing = capture["select_feedback"](records, {}, None)
    assert missing["q_rad"] == [None] * 6
    assert missing["gripper_full_width_m"] is None
    assert missing["telemetry_fresh_at_frame"] is False

def test_passive_reader_feedback_uses_decoder_without_opening_a_socket():
    reader=passive_can.PassiveCAN('never-opened')
    now=1_000_000_000
    for can_id in passive_can.FEEDBACK_IDS:
        reader.records[can_id].append(record(now,b'\x00'*8))
    reader.records[passive_can.GRIPPER_ID].clear()
    reader.records[passive_can.GRIPPER_ID].append(record(now,struct.pack('>ihBB',-2300,0,64,0)))
    result=reader.feedback(now)
    assert result['telemetry_fresh_at_frame']
    assert result['gripper_full_width_m']==-.0023
    assert reader.sock is None
    assert not result['hardware_commands_sent']

def test_sdk_read_exercises_real_decoder_with_synthetic_can_frames():
    import time
    from types import SimpleNamespace
    from piper_runtime.transports.robot_io import SDKRobot
    reader=passive_can.PassiveCAN('never-opened');now=time.monotonic_ns();wall=time.time_ns()
    for can_id in passive_can.FEEDBACK_IDS:
        reader.records[can_id].append({'host_monotonic_ns':now,'host_wall_time_ns':wall,'data':b'\x00'*8})
    status=SimpleNamespace(timestamp=time.time(),msg=SimpleNamespace(arm_status=0,err_code=0,ctrl_mode=1))
    class Arm:
        def get_arm_status(self):return status
        def get_driver_states(self,index):return SimpleNamespace(timestamp=time.time())
        def get_joints_enable_status_list(self):return [True]*6
    robot=SDKRobot.__new__(SDKRobot);robot.reader=reader;robot.arm=Arm()
    robot.gripper=SimpleNamespace(get_gripper_status=lambda:SimpleNamespace(timestamp=time.time(),msg=SimpleNamespace(mode='width')))
    state=robot.read()
    assert state['q_rad'].shape==(6,)and state['qvel_rad_s'].shape==(6,)
    assert state['gripper_full_width_m']==0
    assert reader.sock is None
