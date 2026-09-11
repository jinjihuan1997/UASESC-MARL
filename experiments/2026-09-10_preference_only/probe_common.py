"""Versioned preference-only configuration over an immutable SC runtime."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
PARENT = PROJECT/'experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909'
sys.path.insert(0, str(PARENT/'source'))
from common import write, stamp
from tensor_env import TensorSCEnv
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from training_checkpoint import digest
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('prior_closed_eval', PROJECT/'experiments/2026-09-10_instruction_closed_loop/evaluate.py')
closed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closed)
check_tensor = closed.check_tensor
explicit_columns = closed.explicit_columns
load_actors = closed.load_actors
actions_for = closed.actions_for
arr = closed.arr
RULES = [f'm{m}_{allocation}' for m in (0, 5, 10) for allocation in ('equal', 'backlog', 'urgency')]
CAL_SEEDS = list(range(20261401, 20261411))
GATE_SEEDS = list(range(20261501, 20261521))


def read(path):
    return json.loads(Path(path).read_text())


def config(hidden=False, steps=1_000_000):
    c = read(PARENT/'configs/seed_85/IC_HAPPO.json')
    e = c['env_args']
    e.update(Q_req_by_instruction_snr_bucket=[list(e['Q_req_by_instruction_snr_bucket'][1]) for _ in range(3)],
             A_limit_by_instruction=[6., 6., 6.], constraint_penalty_A_by_instruction=[0., 0., 0.],
             eta_recv_aoi_bonus=0., use_task_auxiliary_reward=False, actor_observe_instruction=not hidden)
    c['algo_args']['train'].update(num_env_steps=steps)
    c['algo_args']['device'].update(cuda=False, cuda_deterministic=False)
    c['main_args']['exp_name'] = 'HAPPO_hidden_instruction' if hidden else 'IC_HAPPO'
    return c


def verify_parent():
    manifest = read(PARENT/'manifest.json')
    for path, expected in manifest['input_hashes'].items():
        assert digest(PARENT/path) == expected, path
    return manifest


def make_env(seeds, schedule=None, hidden=False):
    args = config(hidden)['env_args']
    args.update(instruction_mode_strategy='explicit_evaluation', explicit_instruction_schedule=schedule or [[0, 0]])
    env = TensorSCEnv(args, count=len(seeds), seed=seeds[0], device='cpu')
    for source, seed in zip(env.source.envs, seeds):
        source.seed(seed)
    obs, state, masks = env.reset()
    return env, obs, state, masks


def rule_actions(env, name):
    mode_text, allocation = name.split('_')
    mode = int(mode_text[1:])
    if allocation == 'equal':
        score = torch.ones_like(env.beta)
    elif allocation == 'backlog':
        score = env.q.sum(-1).to(env.dtype)
    elif allocation == 'urgency':
        score = (env.aoi*env.q).sum(-1)
    else:
        raise ValueError(allocation)
    share = torch.where(score.sum(-1, keepdim=True)>0, score/score.sum(-1, keepdim=True).clamp_min(1e-12), torch.full_like(score, 1/env.U))
    mu = torch.zeros((env.count, env.M), dtype=env.dtype)
    mu[:, mode] = 1
    return [2*share-1] + [mu.clone() for _ in range(env.U)]


def external_hashes(env):
    fields = ['pos_uav', 'pos_ds', 'q', 'tau', 'aoi']
    tape_fields = ['noise_us', 'noise_sat', 'noise_du', 'potential_content', 'instructions']
    result=[]
    for i in range(env.count):
        h=hashlib.sha256()
        for name in fields:
            h.update(arr(getattr(env,name))[i].tobytes())
        for name in tape_fields:
            h.update(arr(getattr(env,name))[:,i].tobytes())
        result.append(h.hexdigest())
    return result


def snapshot(env):
    return {k:arr(getattr(env,k)).copy() for k in ('aoi','q','tau')}


def checked_step(env, actions):
    before=snapshot(env);slot=env.step_index
    obs,state,reward,done,info,masks=env.step(actions,auto_reset=False)
    metrics=check_tensor(env,info,slot,before)
    np.testing.assert_allclose(arr(reward[:,0,0]),metrics['common_reward'],atol=1e-6,rtol=1e-6)
    return obs,state,masks,info,metrics
