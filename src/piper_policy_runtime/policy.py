"""Deterministic FP32 actor inference. Output is normalized action, not CAN commands."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


class Policy:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "policy.json").read_text())
        if self.metadata["schema_version"] != 1:
            raise ValueError("Unsupported policy schema")
        payload = directory / "weights.npz"
        if hashlib.sha256(payload.read_bytes()).hexdigest() != self.metadata["weights_sha256"]:
            raise ValueError("Policy weight checksum mismatch")
        with np.load(payload, allow_pickle=False) as data:
            self.weights = {k: data[k].astype(np.float32) for k in data.files}
        if not all(np.isfinite(x).all() for x in self.weights.values()):
            raise ValueError("Nonfinite weights")
        if np.any(self.weights['normalizer/std'] <= 0):
            raise ValueError("Invalid normalizer standard deviation")
        if self.metadata['actor'] not in ['mlp', 'lstm', 'mlp_lstm']:
            raise ValueError("Unsupported actor")
        self.reset()

    def reset(self, batch_size=1):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        width = self.metadata['recurrent_width']
        self.c = np.zeros((batch_size, width), np.float32)
        self.h = self.c.copy()

    def step(self, observation, reset=False):
        x = np.asarray(observation, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None]
        if x.shape != (self.c.shape[0], self.metadata['observation_size']):
            raise ValueError("Observation shape/batch mismatch; call reset(batch_size) before changing batch")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observation")
        mask = np.broadcast_to(np.asarray(reset, dtype=bool), (len(x),))
        c = np.where(mask[:, None], 0., self.c)
        h = np.where(mask[:, None], 0., self.h)
        w = self.weights
        x = (x - w['normalizer/mean']) / w['normalizer/std']
        kind = self.metadata['actor']
        if kind == 'mlp_lstm':
            for index, _ in enumerate(self.metadata['encoder_sizes']):
                x = x @ w[f'params/encoder_{index}/kernel'] + w[f'params/encoder_{index}/bias']
                x = x * sigmoid(x)
            mean = x.mean(-1, keepdims=True)
            var = np.maximum(np.square(x).mean(-1, keepdims=True)-np.square(mean), 0.)
            x = (x-mean)/np.sqrt(var+np.float32(1e-6))
            x = x*w['params/encoder_norm/scale']+w['params/encoder_norm/bias']
        if kind in ('lstm', 'mlp_lstm'):
            dense = x @ w['runtime/lstm_input_kernel'] + h @ w['runtime/lstm_hidden_kernel'] + w['runtime/lstm_bias']
            i, f, g, o = np.split(dense, 4, axis=-1)
            c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
            h = sigmoid(o) * np.tanh(c)
            x = h
            if kind == 'lstm':
                mean = h.mean(-1, keepdims=True)
                var = np.maximum(np.square(h).mean(-1, keepdims=True) - np.square(mean), 0.)
                x = (h - mean) / np.sqrt(var + np.float32(1e-6))
                x = x * w['params/recurrent_norm/scale'] + w['params/recurrent_norm/bias']
                names = [f'head_{i}' for i in range(len(self.metadata['head_sizes']))] + ['distribution']
            else:
                names = ['distribution']
        else:
            names = [f'hidden_{i}' for i in range(len(self.metadata['head_sizes']) + 1)]
        for index, name in enumerate(names):
            x = x @ w[f'params/{name}/kernel'] + w[f'params/{name}/bias']
            if index != len(names) - 1:
                x = x * sigmoid(x)
        action = np.tanh(x[..., :self.metadata['action_size']])
        if not np.isfinite(x).all() or not np.isfinite(action).all() or not np.isfinite(c).all() or not np.isfinite(h).all():
            raise FloatingPointError("Nonfinite policy state/output; memory was not updated")
        self.c, self.h = c, h
        self.logits = x.copy()
        return action[0] if single else action
