"""Isolated diagnostic support. Frozen code is imported, never copied or edited."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
FROZEN = WORKSPACE / 'experiments/2026-09-11_alternating_training'
REFERENCE_GIT = WORKSPACE.parent / '.github-sync/UASESC-MARL_20260911/checkout'
REFERENCE_COMMIT = '7cc2372d19f36ed7d91ed289f83dcc79fe0f3e9d'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TMPDIR'] = str(HERE / 'tmp')
os.environ['GIT_OPTIONAL_LOCKS'] = '0'
sys.dont_write_bytecode = True
import tempfile
tempfile.tempdir = str(HERE / 'tmp')


def install_write_guard():
    def allowed(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(path)).resolve()
            if not p.is_relative_to(HERE):
                raise PermissionError(f'Diagnostic write outside isolated directory: {p}')

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                allowed(path)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'):
            allowed(args[0])
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            allowed(args[0]); allowed(args[1])
    sys.addaudithook(audit)


install_write_guard()
import copy
import datetime
import hashlib
import json
import time
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(FROZEN))
import helpers as frozen
from evaluation import FIELDS as ORIGINAL_FIELDS

FIELDS = ORIGINAL_FIELDS + ['p95_aoi', 'base_reward', 'aoi_exceedance_fraction', 'infeasible_uav_fraction']
COMPONENTS = ['quality_credit', 'age_mean_cost', 'age_max_cost', 'age_tail_cost',
              'resource_cost', 'service_violation_cost', 'recv_aoi_bonus']


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def array(value):
    return value.detach().cpu().numpy().copy()


def digest_arrays(values):
    h = hashlib.sha256()
    for key in sorted(values):
        x = np.ascontiguousarray(values[key])
        h.update(key.encode()); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def save_npz(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)
    return sha(path)


def model_dir(seed):
    return FROZEN / 'jobs' / f'seed_{seed}' / 'joint/milestones/steps_1000000'


def adapted_config(seed):
    cfg = copy.deepcopy(frozen.config('joint', seed))
    cfg['env_args']['semantic_registry_path'] = str(FROZEN / 'source/reference/inputs/mode_registry.json')
    cfg['env_args']['semantic_profile_path'] = str(FROZEN / 'source/reference/inputs/profile.npz')
    return cfg


def verify_inputs(full=False):
    frozen.verify()
    m = read(HERE / 'manifest.json')
    for path, expected in m['protected_input_sha256'].items():
        if full or path in m['execution_input_paths']:
            assert sha(WORKSPACE / path) == expected, f'Frozen input changed: {path}'
    amendment = m.get('analysis_amendment', {})
    overrides = amendment.get('new_sha256', {})
    assert set(overrides) <= {'aggregate.py', 'support.py'}, 'Only the documented reporting fix is allowed'
    for path, expected in m.get('diagnostic_code_sha256', {}).items():
        if path in overrides:
            assert sha(HERE / 'logs' / ('before_reporting_fix_' + path)) == expected
            expected = overrides[path]
        assert sha(HERE / path) == expected, f'Diagnostic implementation changed: {path}'
    assert sha(HERE / 'PROTOCOL.md') == m['protocol_sha256']
    return m


def make_env(seed, env_seeds, scenario, device):
    m = read(HERE / 'manifest.json')
    allowed = set(m['validation_seeds'] + m['probe_fit_seeds'] + m['probe_holdout_seeds'] + m['preflight_seeds'])
    assert set(env_seeds) <= allowed
    assert not (set(env_seeds) & set(m['reserved_final_test_seeds']))
    cfg = adapted_config(seed)
    if scenario.startswith('fixed_'):
        args = copy.deepcopy(cfg['env_args'])
        args.update(instruction_mode_strategy='explicit_evaluation', explicit_instruction_schedule=[[0, int(scenario[-1])]])
    else:
        assert scenario == 'random_switch_once'
        args = copy.deepcopy(cfg['env_args'])
        assert args['instruction_mode_strategy'] == scenario and args['explicit_instruction_schedule'] is None
    env = frozen.TensorSCEnv(args, count=len(env_seeds), seed=env_seeds[0], device=device)
    for source, s in zip(env.source.envs, env_seeds):
        source.seed(s)
    return env, env.reset(), cfg


class ActionStreams:
    """Four explicit Torch states; batch layout is part of the frozen protocol."""
    def __init__(self, root_seed, device):
        self.device = torch.device(device)
        self.seeds = [int(np.random.SeedSequence([root_seed, i, 12092026]).generate_state(1)[0]) for i in range(4)]
        self.states = [torch.Generator(device=self.device).manual_seed(s).get_state() for s in self.seeds]
        self.initial = [x.clone() for x in self.states]

    def call(self, agent, fn):
        cuda = self.device.type == 'cuda'
        # The ambient RNG is restored even if actor execution fails.
        old = torch.cuda.get_rng_state(self.device) if cuda else torch.get_rng_state()
        setter = (lambda x: torch.cuda.set_rng_state(x, self.device)) if cuda else torch.set_rng_state
        setter(self.states[agent])
        try:
            value = fn()
            self.states[agent] = (torch.cuda.get_rng_state(self.device) if cuda else torch.get_rng_state()).clone()
        finally:
            setter(old)
        return value


class FrozenActors:
    def __init__(self, cfg, env, seed):
        directory = model_dir(seed); status = read(directory / 'status.json')
        assert status['state'] == 'complete' and status['completed_steps'] == 1000000
        assert sha(FROZEN / 'configs' / f'seed_{seed}' / 'joint.json') == status['identity']['config_sha256']
        for name, h in status['checkpoint_hashes'].items():
            assert sha(directory / name) == h
        self.actors = frozen.load_actors(cfg, env, directory)
        self.device = env.device
        self.hooks = []; self.capture = {}
        for i, actor in enumerate(self.actors):
            actor.actor.to(self.device)
            actor.device = self.device
            actor.actor.tpdv['device'] = self.device
            actor.actor.requires_grad_(False); actor.prep_rollout()
            # Even accidental calls through the legacy wrapper cannot update an actor.
            def forbidden(*a, **kw):
                raise RuntimeError('Original actor optimizer.step is prohibited')
            actor.actor_optimizer.step = forbidden
            def capture(module, inputs, output, agent=i):
                _, logp, distribution = output
                comp = distribution.components
                assert len(comp) == 1
                d = comp[0]['dist']
                self.capture[agent] = dict(logp=array(logp.sum(-1)), entropy=array(d.entropy()),
                    concentration=array(d.concentration) if agent == 0 else None,
                    probabilities=array(d.probs[:, 0]) if agent else None)
            self.hooks.append(actor.actor.act.action_out.register_forward_hook(capture))
        self.initial_hashes = [frozen.network_hash(a.actor) for a in self.actors]
        assert self.initial_hashes == status['actor_hashes']

    def act(self, agent, obs, available, deterministic, streams):
        rnn = torch.zeros((obs.shape[0], 1, 256), device=self.device)
        masks = torch.ones((obs.shape[0], 1), device=self.device)
        fn = lambda: self.actors[agent].act(obs, rnn, masks, available, deterministic=deterministic)[0]
        action = fn() if deterministic else streams.call(agent, fn)
        return action, self.capture[agent]

    def verify(self):
        assert [frozen.network_hash(a.actor) for a in self.actors] == self.initial_hashes
        assert all(not p.requires_grad for a in self.actors for p in a.actor.parameters())


def exogenous(env):
    records = []
    values = {key: array(getattr(env, key)) for key in ('pos_uav', 'pos_ds', 'q', 'tau', 'aoi', 'psi',
                'noise_us', 'noise_sat', 'noise_du', 'potential_content', 'instructions')}
    future = {'noise_us', 'noise_sat', 'noise_du', 'potential_content', 'instructions'}
    for i in range(env.count):
        records.append(digest_arrays({k: v[:, i] if k in future else v[i] for k, v in values.items()}))
    return records


def save_status(phase, **kwargs):
    write(HERE / 'status.json', dict(state='running', phase=phase, updated_utc=stamp(), **kwargs))
