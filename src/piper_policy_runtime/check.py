"""Replay fixed export vectors without connecting to robot hardware."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .policy import Policy


def check(directory, tolerance=1e-4):
    directory = Path(directory)
    policy = Policy(directory)
    with np.load(directory / 'golden.npz', allow_pickle=False) as d:
        observations, resets, expected = d['observations'], d['resets'], d['actions']
        extra = {key:d[key] for key in ('logits','memory_c','memory_h') if key in d}
    policy.reset(observations.shape[1])
    actual=[];errors={key:0. for key in extra}
    for index,(obs,reset) in enumerate(zip(observations,resets)):
        actual.append(policy.step(obs,reset))
        for key,value in extra.items():
            found={'logits':policy.logits,'memory_c':policy.c,'memory_h':policy.h}[key]
            errors[key]=max(errors[key],float(np.max(np.abs(found-value[index]))))
    actual=np.stack(actual)
    error = float(np.max(np.abs(actual - expected)))
    if error > tolerance:
        raise AssertionError(f'Export replay error {error} exceeds {tolerance}')
    if any(value>tolerance for value in errors.values()):
        raise AssertionError(f'Export recurrent/logit replay errors exceed {tolerance}: {errors}')
    policy.reset(1)
    # This is a policy-only CPU latency test, not perception/control-loop latency.
    obs = observations[0, 0]
    for _ in range(20):policy.step(obs)
    times = []
    for _ in range(200):
        start = time.perf_counter(); policy.step(obs);times.append((time.perf_counter()-start)*1e3)
    return {'status':'passed','actor':policy.metadata['actor'], 'steps':len(observations),
        'batch':observations.shape[1],'max_action_abs_error':error,'state_and_logit_max_abs_error':errors,
        'policy_only_cpu_latency_ms':{'p50':float(np.median(times)),'p95':float(np.quantile(times,.95))},
        'robot_connection':'none','scope':'synthetic export parity vectors; not a task success evaluation'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--policy',type=Path,required=True)
    args=ap.parse_args();print(json.dumps(check(args.policy),indent=2))


if __name__=='__main__':main()
