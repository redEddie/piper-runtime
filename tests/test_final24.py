"""Parity against the selected learner's JAX vectors; no simulator or SDK."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
from piper_policy_runtime import Policy
from piper_runtime.contracts import contract
from piper_runtime.identity import identity

SPEC=json.loads((Path(__file__).parents[1]/'compatibility/show108-24u-final.json').read_text())

def test_final24_keeps_the_verified_execution_implementation():
    assert identity()['version']=='0.5.0'
    assert identity()['sha256']==SPEC['runtime_sha256']

@pytest.fixture
def policy_dir():
    path=os.environ.get('PIPER_FINAL24_POLICY_DIR')
    if not path:pytest.skip('Download the final inference archive and set PIPER_FINAL24_POLICY_DIR')
    return Path(path)

def test_selected_learner_normalizer_and_execution_contract(policy_dir):
    meta=json.loads((policy_dir/'policy.json').read_text())
    assert meta['checkpoint_sha256']==SPEC['checkpoint_sha256']
    assert hashlib.sha256((policy_dir/'weights.npz').read_bytes()).hexdigest()==SPEC['weights_sha256']
    assert hashlib.sha256((policy_dir/'training-config.json').read_bytes()).hexdigest()==SPEC['config_sha256']
    assert meta['cumulative_training_steps']==201326592
    assert contract(meta)['runtime_profile']['controller']=='show108_checked_plan_independent'

@pytest.mark.parametrize('vectors',['golden.npz','recorded-golden.npz'])
def test_final24_actions_logits_and_both_memory_states(policy_dir,vectors):
    policy=Policy(policy_dir)
    with np.load(policy_dir/vectors,allow_pickle=False) as d:
        policy.reset(d['observations'].shape[1])
        for i,(obs,reset) in enumerate(zip(d['observations'],d['resets'])):
            np.testing.assert_allclose(policy.step(obs,reset),d['actions'][i],atol=1e-4,rtol=0)
            for key,value in [('logits',policy.logits),('memory_c',policy.c),('memory_h',policy.h)]:
                np.testing.assert_allclose(value,d[key][i],atol=1e-4,rtol=0)
