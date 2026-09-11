"""Full-budget training using the validated tensor kernels, with exact resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import torch
from common import network_hash, stamp, write, verify_reference
from tensor_train import TensorTrainer
from training_checkpoint import capture, digest, load_checkpoint, reconcile_logs, restore, save_checkpoint


def train(config_path, output, device, run_manifest, checkpoint_every=50, resume=False, stop_after_update=None):
    output = Path(output)
    if (output/'status.json').exists() and not resume:
        raise FileExistsError('Job already exists; preserve it and use --resume explicitly')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = []
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda signum, frame: stopped.append(signum))
    identity = dict(config_sha256=digest(config_path), run_manifest_sha256=digest(run_manifest), device=device)
    config = json.loads(Path(config_path).read_text())
    update, trainer = 0, None
    started = time.monotonic()
    try:
        verify_reference()
        status_path = output/'status.json'
        trainer = TensorTrainer(config, device)
        if resume:
            state, checkpoint = load_checkpoint(output/'checkpoints')
            update = restore(trainer, state, identity)
            reconcile_logs(output, update)
            with (output/'resume_events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=stamp(), checkpoint=checkpoint))+'\n')
        else:
            write(output/'config.json', trainer.config)
            write(output/'manifest.json', dict(schema=1, created_utc=stamp(), identity=identity,
                purpose=json.loads(Path(run_manifest).read_text())['purpose'],
                seed=config['algo_args']['seed']['seed'], method=config['main_args']['exp_name'],
                device=device, environment_device=str(trainer.env.device),
                initial_actor_hashes=trainer.initial_actors, initial_critic_hash=trainer.initial_critic,
                batch=trainer.batch, steps=config['algo_args']['train']['num_env_steps'],
                full_state_checkpoint=True, checkpoint_every_updates=checkpoint_every,
                source_snapshot=str(Path(run_manifest).parent/'source')))
            checkpoint = save_checkpoint(output/'checkpoints', capture(trainer, update, identity))
        while update < trainer.total_updates and not stopped:
            next_update = update+1
            if trainer.algo['train']['use_linear_lr_decay']:
                for actor in trainer.actors:
                    actor.lr_decay(next_update, trainer.total_updates)
                trainer.critic.lr_decay(next_update, trainer.total_updates)
            trainer.synchronize()
            t0 = time.monotonic()
            trainer.collect()
            trainer.synchronize()
            t1 = time.monotonic()
            metrics = trainer.update()
            trainer.synchronize()
            t2 = time.monotonic()
            if not bool(torch.stack([torch.isfinite(x).all() for x in metrics.values()]).all()):
                raise FloatingPointError('Non-finite PPO metrics or returns')
            mean_reward = float(trainer.buffer.rewards.mean())
            trainer.buffer.after_update()
            update = next_update
            timing = dict(update=update, steps=update*trainer.batch, collect_seconds=t1-t0,
                          update_seconds=t2-t1, total_seconds=t2-t0)
            values = dict(update=update, steps=update*trainer.batch, mean_training_reward=mean_reward,
                mean_return=float(metrics['returns'].mean()),
                actor_loss_entropy_grad_ratio=metrics['actor'].detach().cpu().tolist(),
                critic_loss_grad=metrics['critic'].detach().cpu().tolist())
            for name, record in (('timing.jsonl', timing), ('training_metrics.jsonl', values)):
                with (output/name).open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
            if stop_after_update is not None and update >= stop_after_update:
                stopped.append('requested_update_boundary')
            if update % checkpoint_every == 0 or stopped or update == trainer.total_updates:
                checkpoint = save_checkpoint(output/'checkpoints', capture(trainer, update, identity))
            write(status_path, dict(state='training', updated_utc=stamp(), pid=os.getpid(),
                completed_steps=update*trainer.batch, target_steps=trainer.total_updates*trainer.batch,
                device=device, last_timing=timing, recoverable_update=checkpoint['update']))
        if update != trainer.total_updates:
            if checkpoint['update'] != update:
                checkpoint = save_checkpoint(output/'checkpoints', capture(trainer, update, identity))
            write(status_path, dict(state='paused', updated_utc=stamp(), pid=os.getpid(),
                completed_steps=update*trainer.batch, target_steps=trainer.total_updates*trainer.batch,
                device=device, recoverable_update=checkpoint['update']))
            return 75
        for i, actor in enumerate(trainer.actors):
            torch.save(actor.actor.state_dict(), output/f'actor_agent{i}.pt')
        torch.save(trainer.critic.critic.state_dict(), output/'critic_agent.pt')
        torch.save(trainer.normalizer.state_dict(), output/'value_normalizer.pt')
        final = [network_hash(a.actor) for a in trainer.actors]
        if not all(a != b for a, b in zip(final, trainer.initial_actors)):
            raise AssertionError('An actor did not update')
        if network_hash(trainer.critic.critic) == trainer.initial_critic:
            raise AssertionError('Critic did not update')
        for path in output.glob('*.pt'):
            if not all(torch.isfinite(v).all() for v in torch.load(path, map_location='cpu', weights_only=True).values()):
                raise FloatingPointError('Invalid exported model')
        timings = [json.loads(line) for line in (output/'timing.jsonl').read_text().splitlines()]
        write(status_path, dict(state='complete', updated_utc=stamp(), pid=os.getpid(),
            completed_steps=update*trainer.batch, target_steps=trainer.total_updates*trainer.batch,
            device=device, environment_device=str(trainer.env.device), recoverable_update=update,
            invocation_seconds=time.monotonic()-started,
            compute_seconds=sum(row['total_seconds'] for row in timings),
            final_actor_hashes=final, final_critic_hash=network_hash(trainer.critic.critic),
            checkpoint_hashes={p.name: digest(p) for p in output.glob('*.pt')}))
        return 0
    except BaseException as exc:
        write(output/'status.json', dict(state='failed', updated_utc=stamp(), pid=os.getpid(),
            completed_steps=update*(trainer.batch if trainer else 0), error=repr(exc)))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
    parser.add_argument('--checkpoint-every', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-update', type=int)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error('checkpoint-every must be positive')
    raise SystemExit(train(args.config, args.output, args.device, args.run_manifest,
        args.checkpoint_every, args.resume, args.stop_after_update))
