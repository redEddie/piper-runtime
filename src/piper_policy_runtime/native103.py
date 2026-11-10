"""Load the CPU-only SHOW-103 planner. Compilation precedes hardware use."""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy as np
from .kinematics import _matrix
from .control100 import ControlFault

_libraries = {}


def load_library():
    source = Path(__file__).with_name('plan103.cpp')
    digest = hashlib.sha256(source.read_bytes()+platform.machine().encode()).hexdigest()[:20]
    cache = Path.home()/'.cache/piper-policy-runtime'
    library = cache/f'plan103-{digest}.so'
    if digest not in _libraries:
        cache.mkdir(parents=True, exist_ok=True)
        if not library.exists():
            descriptor, temp = tempfile.mkstemp(prefix='plan103-', suffix='.so', dir=cache)
            os.close(descriptor)
            try:
                result = subprocess.run(['c++', '-O3', '-shared', '-fPIC', '-std=c++17',
                                         str(source), '-o', temp], capture_output=True, text=True, timeout=60.)
                if result.returncode:
                    raise ControlFault('SHOW-103 planner build failed: '+result.stderr[-2000:])
                os.replace(temp, library)
            finally:
                Path(temp).unlink(missing_ok=True)
        native = ctypes.CDLL(str(library))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        for name in ('plan103', 'plan103_samples'):
            function = getattr(native, name)
            function.argtypes = [array]*6+[ctypes.c_int]+[array]*3+[ctypes.c_int]*2+[array]*2
            function.restype = ctypes.c_int
        _libraries[digest] = native
    return _libraries[digest]


class Planner103:
    def __init__(self, model, settings, steps=20, stop_steps=64):
        self.native = load_library()
        self.model = model
        self.kin = np.concatenate([
            np.concatenate([offset, _matrix(quat).ravel(), pivot, axis, [reference]])
            for (offset, quat), (pivot, axis, reference, _) in zip(model._bodies, model._joints)
        ]+[model.tcp_local_position]).astype(np.float64)
        self.settings = np.ascontiguousarray(settings, dtype=np.float64)
        self.lower, self.upper = (np.ascontiguousarray(v, dtype=np.float64) for v in (model.lower, model.upper))
        self.steps, self.stop_steps = int(steps), int(stop_steps)
        if (self.kin.shape != (117,) or self.settings.shape != (16,)
                or not np.isfinite(self.kin).all() or not np.isfinite(self.settings).all()
                or np.any(self.settings[:9] <= 0) or self.settings[9] < 0
                or self.settings[14] <= 0 or not np.isclose(np.linalg.norm(self.settings[11:14]), 1.)
                or not 1 <= self.steps <= 100 or not 1 <= self.stop_steps <= 1000):
            raise ValueError('Invalid native planner dimensions/settings')

    def plan(self, q, qdot, state, target, old_target, hold=0):
        return self._call(q, qdot, state, target, old_target, hold, sampled=False)

    def plan_samples(self, q, qdot, state, samples):
        return self._call(q, qdot, state, samples, np.zeros(6), 0, sampled=True)

    def _call(self, q, qdot, state, target, old_target, hold, *, sampled):
        values = []
        for v, shape in zip((q, qdot, state, target, old_target),
                            ((6,), (6,), (18,), (self.steps, 6) if sampled else (6,), (6,))):
            x = np.ascontiguousarray(v, dtype=np.float64)
            if x.shape != shape or not np.isfinite(x).all():
                raise ControlFault('Nonfinite/invalid planner input')
            values.append(x)
        if not 0 <= hold <= self.steps:
            raise ValueError('Invalid hold ticks')
        output = np.empty((self.steps+self.stop_steps, 18), np.float64)
        detail = np.empty(5, np.float64)
        function = self.native.plan103_samples if sampled else self.native.plan103
        code = function(self.kin, *values, int(hold), self.lower, self.upper,
                        self.settings, self.steps, self.stop_steps, output, detail)
        if code < 0:
            raise ControlFault('Native planner rejected dimensions')
        return output, bool(code), detail
