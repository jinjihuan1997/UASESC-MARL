"""Read frozen project inputs; confine every Python write to this experiment."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
FROZEN = HERE.parent / '2026-09-11_alternating_training'
GREEDY = HERE.parent / '2026-09-11_observation_matched_greedy'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(FROZEN))
import helpers as frozen
import copy
import hashlib
import json
import random
import contextlib
import datetime
import numpy as np
import torch
from tensor_train import TensorRollout, TensorValueNorm
from tensor_env import TensorSCEnv
from training_checkpoint import capture, restore, save_checkpoint, load_checkpoint, runtime_signature, reconcile_logs
from common import network_hash
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.ratio_tools import aggregate_action_ratio
torch.set_num_threads(1)

def write_guard(event, args):
    if event == 'open':
        path, mode, flags = args
        writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
        writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        paths = [path] if writing else []
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate'):
        paths = [args[0]]
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        paths = list(args[:2])
    else:
        return
    for path in paths:
        if isinstance(path, (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(path)).resolve()
            if p != Path('/dev/null') and not p.is_relative_to(HERE):
                raise PermissionError(f'Experiment write outside isolated directory: {p}')

def guard():
    # Torch lazy optimizer imports happen before installing the write audit hook.
    import torch._dynamo
    sys.addaudithook(write_guard)

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p, obj):
    p = Path(p); assert p.resolve().is_relative_to(HERE)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)
def append(p, obj):
    with Path(p).open('a') as f: f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
def arr(x): return frozen.arr(x)
def model_dir(seed): return FROZEN / 'jobs' / f'seed_{seed}' / 'joint/milestones/steps_1000000'
def cfg_for(seed):
    cfg = frozen.config('joint', seed)
    cfg['env_args']['semantic_registry_path'] = str(FROZEN / 'source/reference/inputs/mode_registry.json')
    cfg['env_args']['semantic_profile_path'] = str(FROZEN / 'source/reference/inputs/profile.npz')
    return cfg
def manifest(): return read(HERE / 'manifest.json')
def check_seeds(seeds, purpose):
    m = manifest()
    assert not set(seeds) & set(m['reserved_final_test'])
    allowed = m['validation'] if purpose == 'evaluation' else m['preflight_seeds'] + sum([v['environment'] for v in m['new_seeds'].values()], [])
    assert set(seeds) <= set(allowed), (purpose, seeds)

def make_env(cfg, seeds, device='cpu', schedule=None, purpose='training'):
    check_seeds(seeds, purpose)
    assert not {seeds[0]+1000*i for i in range(len(seeds))} & set(manifest()['reserved_final_test'])
    args = copy.deepcopy(cfg['env_args'])
    if schedule is not None:
        args.update(instruction_mode_strategy='explicit_evaluation', explicit_instruction_schedule=schedule)
    env = TensorSCEnv(args, count=len(seeds), seed=seeds[0], device=device)
    for src, seed in zip(env.source.envs, seeds): src.seed(seed)
    env.reset()
    return env

class FrozenOptimizer:
    param_groups = []
    def state_dict(self): return dict(state={},param_groups=[])
    def load_state_dict(self,state): assert state==self.state_dict()
    def step(self,*args,**kwargs): raise RuntimeError('Frozen SUT cannot be optimized')
    def zero_grad(self,*args,**kwargs): pass

def bases(cfg, env, seed, device):
    args = {**cfg['algo_args']['model'], **cfg['algo_args']['algo']}
    actors = [HAPPO(args, o, a, torch.device(device)) for o, a in zip(env.observation_space, env.action_space)]
    status = read(model_dir(seed) / 'status.json')
    assert status['state'] == 'complete' and status['completed_steps'] == 1000000
    assert status['identity']['config_sha256'] == sha(FROZEN / 'configs' / f'seed_{seed}/joint.json')
    assert status['identity']['run_manifest_sha256'] == sha(FROZEN / 'manifest.json')
    for name, h in status['checkpoint_hashes'].items(): assert sha(model_dir(seed) / name) == h
    for i, actor in enumerate(actors):
        actor.actor.load_state_dict(torch.load(model_dir(seed) / f'actor_agent{i}.pt', map_location=device, weights_only=True), strict=True)
        actor.actor.requires_grad_(False); actor.prep_rollout()
        assert network_hash(actor.actor) == status['actor_hashes'][i]
    actors[0].actor_optimizer = FrozenOptimizer()
    return actors

def verify_inputs(include_code=True):
    m = manifest()
    for path, h in m['protected_input_sha256'].items(): assert sha(WORKSPACE / path) == h, path
    assert sha(HERE / 'PROTOCOL.md') == m['protocol_sha256']
    for path,h in m['config_sha256'].items(): assert sha(HERE/path)==h,path
    if include_code:
        for path, h in m.get('execution_code_sha256', {}).items(): assert sha(HERE / path) == h, path
    return m

class ActionStreams:
    """Independent global Torch RNG contexts for original Categorical.sample()."""
    def __init__(self, seeds, device):
        self.device = torch.device(device)
        self.states = [torch.Generator(device=self.device).manual_seed(s).get_state() for s in seeds]
    @contextlib.contextmanager
    def use(self, i):
        cuda = self.device.type == 'cuda'
        old = torch.cuda.get_rng_state(self.device) if cuda else torch.get_rng_state()
        if cuda: torch.cuda.set_rng_state(self.states[i], self.device)
        else: torch.set_rng_state(self.states[i])
        try: yield
        finally:
            self.states[i] = torch.cuda.get_rng_state(self.device) if cuda else torch.get_rng_state()
            if cuda: torch.cuda.set_rng_state(old, self.device)
            else: torch.set_rng_state(old)

def exogenous(env): return frozen.external_hashes(env)

FIELDS = ['common_reward','base_reward','training_reward','mean_aoi','max_aoi','p95_aoi',
          'predicted_quality_sum','deliveries','channel_uses','instruction_id',
          'quality_violations','budget_violations','cache_violations','fraction_above4','fraction_above6',
          'quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost',
          'service_violation_cost','recv_aoi_bonus']

def checked_step(env, actions):
    old = env.step_index
    result = frozen.checked_step(env, actions)
    assert env.step_index == old + 1
    assert not env.allocation_pending
    return result
