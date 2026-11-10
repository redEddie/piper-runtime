"""Capture RGB-D plus passive Piper CAN feedback; never transmit robot commands.

Run in the existing perception environment (pyrealsense2, OpenCV, NumPy).
The output camera directory is compatible with SHOW-33's --run argument.
Protocol units/byte order were checked against the installed Piper SDK v2
parser and feedback classes on Thor; the SDK is deliberately not imported.
"""

from __future__ import annotations

import argparse

from collections import Counter, defaultdict, deque

import datetime

import json

from pathlib import Path

import socket

import struct

import sys

import threading

import time

JOINT_IDS = (0x2A5, 0x2A6, 0x2A7)

SPEED_IDS = tuple(range(0x251, 0x257))

GRIPPER_ID = 0x2A8

FEEDBACK_IDS = JOINT_IDS + SPEED_IDS + (GRIPPER_ID,)

CAN_FRAME = struct.Struct("=IB3x8s")

def tx_packets(interface: str) -> int | None:
    try:
        return int((Path("/sys/class/net") / interface / "statistics/tx_packets").read_text())
    except (OSError, ValueError):
        return None

class PassiveCAN:
    """Only socket creation, bind and receive. No SDK, enable or query calls."""

    def __init__(self, interface: str):
        self.interface = interface
        self.records = defaultdict(lambda: deque(maxlen=2000))
        self.counts = Counter()
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sock = None
        self.thread = None

    def start(self) -> None:
        try:
            self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            # Exact standard data-frame IDs; reject extended and RTR frames.
            filters = b"".join(struct.pack("=II", can_id, 0x800007FF | 0x40000000)
                               for can_id in FEEDBACK_IDS)
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            self.sock.settimeout(0.1)
            self.sock.bind((self.interface,))
            self.thread = threading.Thread(target=self._receive, daemon=True)
            self.thread.start()
        except (OSError, AttributeError) as error:
            self.errors.append(f"CAN receive unavailable: {error}")
            self.stop()

    def _receive(self) -> None:
        while not self.stop_event.is_set():
            try:
                raw = self.sock.recv(CAN_FRAME.size)
                host_monotonic_ns = time.monotonic_ns()
                host_wall_time_ns = time.time_ns()
                if len(raw) != CAN_FRAME.size:
                    continue
                can_id, size, data = CAN_FRAME.unpack(raw)
                if can_id not in FEEDBACK_IDS or size < (8 if can_id in JOINT_IDS else 2):
                    continue
                if can_id == GRIPPER_ID and size < 7:
                    continue
                record = {"host_monotonic_ns": host_monotonic_ns,
                          "host_wall_time_ns": host_wall_time_ns,
                          "data": data[:size]}
                with self.lock:
                    self.records[can_id].append(record)
                    self.counts[can_id] += 1
            except socket.timeout:
                continue
            except OSError as error:
                if not self.stop_event.is_set():
                    self.errors.append(f"CAN receive failed: {error}")
                return

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.sock:
            self.sock.close()
            self.sock = None

    def feedback(self, frame_monotonic_ns: int | None, max_age_s: float = 0.1) -> dict:
        with self.lock:
            records = {key: list(value) for key, value in self.records.items()}
            counts = dict(self.counts)
        return select_feedback(records, counts, frame_monotonic_ns, max_age_s, self.errors)


def select_feedback(records: dict, counts: dict, frame_monotonic_ns: int | None,
                    max_age_s: float = 0.1, errors=()) -> dict:
    """Use last host-received sample at/before the frame, never a future sample.

    Stale values remain visible with explicit freshness flags. Missing values
    remain null, and must not be replaced with a measured zero downstream.
    """
    selected = {}
    components = {}
    for can_id in FEEDBACK_IDS:
        candidates = records.get(can_id, ())
        record = next((r for r in reversed(candidates)
                       if frame_monotonic_ns is not None
                       and r["host_monotonic_ns"] <= frame_monotonic_ns), None)
        selected[can_id] = record
        age = None if record is None else (frame_monotonic_ns - record["host_monotonic_ns"]) / 1e9
        components[f"0x{can_id:03X}"] = {
            "received_count": counts.get(can_id, 0), "missing": record is None,
            "host_wall_time_ns": None if record is None else record["host_wall_time_ns"],
            "host_monotonic_ns": None if record is None else record["host_monotonic_ns"],
            "age_at_frame_s": age, "stale": None if age is None else age > max_age_s,
            "fresh": age is not None and 0 <= age <= max_age_s,
        }
    q = []
    for can_id in JOINT_IDS:
        record = selected[can_id]
        q.extend([None, None] if record is None else
                 [value / 1000 * (3.141592653589793 / 180)
                  for value in struct.unpack(">ii", record["data"][:8])])
    qvel = [None if selected[can_id] is None else
            struct.unpack(">h", selected[can_id]["data"][:2])[0] * 0.001
            for can_id in SPEED_IDS]
    grip = selected[GRIPPER_ID]
    return {
        "schema_version": 1, "source": "passive_socketcan_receive_only",
        "reference_frame_host_monotonic_ns": frame_monotonic_ns,
        "q_rad": q, "qvel_rad_s": qvel,
        "gripper_full_width_m": None if grip is None else struct.unpack(">i", grip["data"][:4])[0] / 1e6,
        "gripper_status_code": None if grip is None else grip["data"][6],
        "freshness_limit_s": max_age_s,
        "telemetry_fresh_at_frame": all(c["fresh"] for c in components.values()),
        "feedback_components": components, "errors": list(errors),
        "time_alignment": "Host receipt before final color frame host receipt; not hardware timestamp synchronization.",
        "joint_speed_note": "0x251..0x256 motor-speed feedback, SDK unit 0.001 rad/s; hardware scaling validation remains pending.",
        "hardware_commands_sent": False,
    }
