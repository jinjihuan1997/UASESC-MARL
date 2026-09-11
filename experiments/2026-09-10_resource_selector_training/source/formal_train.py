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
from selector_train import ContinuationTrainer as TensorTrainer, capture_fork as capture, restore_fork as restore
from training_checkpoint import digest, load_checkpoint, reconcile_logs, save_checkpoint
from model_snapshot import export_snapshot


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
        milestones=config['training_design']['evaluation_steps']
        assert milestones==sorted(set(milestones)) and all(0<s<=trainer.total_updates*trainer.batch and s%trainer.batch==0 for s in milestones)
        if resume:
            state, checkpoint = load_checkpoint(output/'checkpoints')
            update = restore(trainer, state, identity)
            reconcile_logs(output, update)
            with (output/'resume_events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=stamp(), checkpoint=checkpoint))+'\n')
        else:
            trainer.initialize_transfer(identity)
            trainer.set_training_update(1)
            write(output/'config.json', trainer.config)
            write(output/'manifest.json', dict(schema=1, created_utc=stamp(), identity=identity,
                purpose=json.loads(Path(run_manifest).read_text())['purpose'],
                seed=config['algo_args']['seed']['seed'], method=config['main_args']['exp_name'],
                device=device, environment_device=str(trainer.env.device),
                initial_actor_hashes=trainer.initial_actors, initial_critic_hash=trainer.initial_critic,
                initial_selector_hash=getattr(trainer,'initial_selector',None),
                transfer=config['continuation'],
                batch=trainer.batch, steps=config['algo_args']['train']['num_env_steps'],
                full_state_checkpoint=True, checkpoint_every_updates=checkpoint_every,
                source_snapshot=str(Path(run_manifest).parent/'source')))
            checkpoint = save_checkpoint(output/'checkpoints', capture(trainer, update, identity))
        if update*trainer.batch in milestones:
            export_snapshot(trainer,output,update,identity)
        while update < trainer.total_updates and not stopped:
            next_update = update+1
            trainer.set_training_update(next_update)
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
            if trainer.fixed_resources:
                assert network_hash(trainer.actors[0].actor)==trainer.initial_actors[0], 'Frozen SUT changed'
                assert not trainer.actors[0].actor_optimizer.state, 'Frozen SUT optimizer advanced'
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
                stage=trainer.stage,trained_actor_ids=trainer.trainable_agent_ids,
                actor_learning_rates=[a.actor_optimizer.param_groups[0]['lr'] for a in ([trainer.selector] if trainer.selector else trainer.actors)],
                entropy_coef=trainer.selector.entropy_coef if trainer.selector else trainer.actors[0].entropy_coef,
                selector_counts_by_instruction=trainer.selector_counts.cpu().tolist() if trainer.selector else None,
                critic_learning_rate=trainer.critic.critic_optimizer.param_groups[0]['lr'],
                instruction_counts=trainer.instruction_counts.cpu().tolist(),
                instruction_reward_sums=trainer.instruction_rewards.cpu().tolist(),
                mean_return=float(metrics['returns'].mean()),
                actor_loss_entropy_grad_ratio=metrics['actor'].detach().cpu().tolist(),
                critic_loss_grad=metrics['critic'].detach().cpu().tolist(),
                reward_components={k:float(v/trainer.buffer.length) for k,v in trainer.reward_component_sums.items()})
            for name, record in (('timing.jsonl', timing), ('training_metrics.jsonl', values)):
                with (output/name).open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
            if stop_after_update is not None and update >= stop_after_update:
                stopped.append('requested_update_boundary')
            if update % checkpoint_every == 0 or stopped or update == trainer.total_updates or update*trainer.batch in milestones:
                checkpoint = save_checkpoint(output/'checkpoints', capture(trainer, update, identity))
            if update*trainer.batch in milestones:
                export_snapshot(trainer,output,update,identity)
            write(status_path, dict(state='training', updated_utc=stamp(), pid=os.getpid(),
                completed_steps=update*trainer.batch, target_steps=trainer.total_updates*trainer.batch,
                device=device, stage=trainer.stage, last_timing=timing, recoverable_update=checkpoint['update']))
        if update != trainer.total_updates:
            if checkpoint['update'] != update:
                checkpoint = save_checkpoint(output/'checkpoints', capture(trainer, update, identity))
            write(status_path, dict(state='paused', updated_utc=stamp(), pid=os.getpid(),
                completed_steps=update*trainer.batch, target_steps=trainer.total_updates*trainer.batch,
                device=device, recoverable_update=checkpoint['update']))
            return 75
        if trainer.selector:
            torch.save(trainer.selector.actor.state_dict(),output/'selector.pt')
        for i, actor in enumerate(trainer.actors):
            torch.save(actor.actor.state_dict(), output/f'actor_agent{i}.pt')
        torch.save(trainer.critic.critic.state_dict(), output/'critic_agent.pt')
        torch.save(trainer.normalizer.state_dict(), output/'value_normalizer.pt')
        final = [network_hash(a.actor) for a in trainer.actors]
        if trainer.selector:
            assert final==trainer.initial_actors,'Frozen base controllers changed'
            assert network_hash(trainer.selector.actor)!=trainer.initial_selector,'Selector did not learn'
        else:
            assert all(x!=y for x,y in zip(final,trainer.initial_actors)),'Continuation actor did not update'
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
            final_selector_hash=network_hash(trainer.selector.actor) if trainer.selector else None,
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
