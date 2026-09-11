"""Safe, atomic full-state checkpoints at completed PPO update boundaries."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import uuid

import numpy as np
import torch
from common import stamp, write


def runtime_signature():
    return dict(python=sys.version, torch=str(torch.__version__), numpy=str(np.__version__),
                torch_cuda=torch.version.cuda)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported checkpoint value: {type(value)}')


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [device_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(device_tree(v, device) for v in value)
    return value


def capture(trainer, update, identity):
    trainer.synchronize()
    assert not trainer.env.allocation_pending, 'Checkpoint requires a completed physical slot'
    numpy_state = np.random.get_state()
    environment = {k: cpu_tree(v) for k, v in vars(trainer.env).items()
                   if isinstance(v, torch.Tensor) or k == 'last'}
    buffer = {k: cpu_tree(v) for k, v in vars(trainer.buffer).items()
              if isinstance(v, torch.Tensor) or k in ('actions', 'log_probs')}
    return dict(schema=1, identity=identity, runtime=runtime_signature(), update=update, total_updates=trainer.total_updates,
        device=str(trainer.device), config=trainer.config, control_phase='pre_allocation',
        initial_actor_hashes=trainer.initial_actors, initial_critic_hash=trainer.initial_critic,
        actors=[cpu_tree(a.actor.state_dict()) for a in trainer.actors],
        actor_optimizers=[cpu_tree(a.actor_optimizer.state_dict()) for a in trainer.actors],
        critic=cpu_tree(trainer.critic.critic.state_dict()),
        critic_optimizer=cpu_tree(trainer.critic.critic_optimizer.state_dict()),
        normalizer=cpu_tree(trainer.normalizer.state_dict()), buffer=buffer,
        environment=environment, step_index=trainer.env.step_index,
        source_episodes=[dict(seed=e._base_seed, episode=e._episode_index) for e in trainer.env.source.envs],
        episode_indices=trainer.env.episode_indices,
        rng=dict(python=random.getstate(), numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                 torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state(trainer.device) if trainer.device.type == 'cuda' else None))


def restore(trainer, state, identity):
    if (state['schema'] != 1 or state['identity'] != identity or state['config'] != trainer.config
        or state['device'] != str(trainer.device) or state['total_updates'] != trainer.total_updates):
        raise ValueError('Checkpoint does not match the frozen run, budget, seed, method or device')
    if state['runtime'] != runtime_signature():
        raise ValueError('Software runtime changed; exact resume is not validated across versions')
    if len(state['actors']) != len(trainer.actors):
        raise ValueError('Checkpoint actor count differs')
    for actor, weights, optimizer in zip(trainer.actors, state['actors'], state['actor_optimizers']):
        actor.actor.load_state_dict(weights)
        actor.actor_optimizer.load_state_dict(optimizer)
    trainer.critic.critic.load_state_dict(state['critic'])
    trainer.critic.critic_optimizer.load_state_dict(state['critic_optimizer'])
    trainer.normalizer.load_state_dict(state['normalizer'])
    # EpisodeSource never steps its CPU reference instances. Recreating their
    # current reset reproduces their complete setup and next-episode RNG state.
    for env, source in zip(trainer.env.source.envs, state['source_episodes']):
        env.seed(source['seed'])
        env._episode_index = source['episode']-1
    trainer.env.reset()
    for name, value in state['environment'].items():
        setattr(trainer.env, name, device_tree(value, trainer.device))
    trainer.env.step_index = state['step_index']
    trainer.env.episode_indices = state['episode_indices']
    assert state['control_phase']=='pre_allocation'
    trainer.env.allocation_pending=False
    trainer.set_training_update(int(state['update'])+1)
    for name, value in state['buffer'].items():
        setattr(trainer.buffer, name, device_tree(value, trainer.device))
    trainer.initial_actors = state['initial_actor_hashes']
    trainer.initial_critic = state['initial_critic_hash']
    random.setstate(state['rng']['python'])
    rng = state['rng']['numpy']
    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
    torch.set_rng_state(state['rng']['torch'])
    if trainer.device.type == 'cuda':
        torch.cuda.set_rng_state(state['rng']['cuda'], trainer.device)
    return int(state['update'])


def save_checkpoint(folder, state):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder/'index.json'
    prior = json.loads(index_path.read_text()) if index_path.exists() else {}
    name = f"update_{state['update']:06d}_{uuid.uuid4().hex[:12]}.pt"
    target, temporary = folder/name, folder/(name+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    current = dict(file=name, sha256=digest(target), update=state['update'], size_bytes=target.stat().st_size)
    # The index commits the new checkpoint only after the complete file exists.
    write(index_path, dict(schema=1, utc=stamp(), current=current, previous=prior.get('current')))
    keep = {current['file']}
    if prior.get('current'):
        keep.add(prior['current']['file'])
    # Only checkpoints created and superseded by this job are rotated.
    if prior.get('previous') and prior['previous']['file'] not in keep:
        old = folder/prior['previous']['file']
        old.unlink(missing_ok=True)
    return current


def load_checkpoint(folder):
    folder = Path(folder)
    index = json.loads((folder/'index.json').read_text())
    errors = []
    for key in ('current', 'previous'):
        entry = index.get(key)
        if not entry:
            continue
        path = folder/entry['file']
        try:
            if digest(path) != entry['sha256']:
                raise ValueError('Checkpoint hash mismatch')
            state = torch.load(path, map_location='cpu', weights_only=True)
            if state['update'] != entry['update']:
                raise ValueError('Checkpoint index disagrees with payload')
            return state, dict(**entry, selected=key, rejected=errors)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f'{key}: {exc}')
    raise RuntimeError(f'No valid complete-state checkpoint: {errors}')


def reconcile_logs(folder, update):
    """Preserve interrupted logs, then discard only uncommitted update rows."""
    folder = Path(folder)
    for name in ('timing.jsonl', 'training_metrics.jsonl'):
        path = folder/name
        if not path.exists():
            continue
        data = path.read_text()
        rows = []
        for line in data.splitlines():
            try:
                value = json.loads(line)
                if value['update'] <= update:
                    rows.append(line)
            except (ValueError, KeyError):
                pass
        retained = ''.join(line+'\n' for line in rows)
        if retained != data:
            archive = folder/'recovery'/f'{uuid.uuid4().hex}_{name}'
            archive.parent.mkdir(exist_ok=True)
            archive.write_text(data)
            temporary = path.with_suffix('.jsonl.tmp')
            temporary.write_text(retained)
            temporary.replace(path)
