"""Bounded ownership of RGB-D frames while asynchronous detection is running.

A camera ring may evict a frame before its segmentation result is returned.
Keep the exact source frame with the in-flight request and published anchors,
instead of pairing an old mask with new depth or enlarging the entire ring.
"""
from collections import deque
import threading


class AnchorFrames:
    def __init__(self):
        self._owners = {}
        self._lock = threading.Lock()

    def retain(self, owner, frame):
        with self._lock:
            history = self._owners.setdefault(owner, deque(maxlen=2))
            if not history or history[-1]['id'] != frame['id']:
                history.append(frame)

    def release(self, owner):
        with self._lock:
            self._owners.pop(owner, None)

    def get(self, frame_id):
        with self._lock:
            for history in self._owners.values():
                for frame in reversed(history):
                    if frame['id'] == frame_id:
                        return frame
        return None

    def count(self):
        with self._lock:
            return len({frame['id'] for history in self._owners.values() for frame in history})
