"""Resumable paired evaluation using the unchanged CPU reference and metrics."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from common import stamp, write
from multiseed_protocol import read, verify_model, verify_run
from training_checkpoint import digest
from evaluate import aggregate, external_metrics, load_actors
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
from harl.envs.uav_escs.SC.rules import rule_actions


def evaluate_one(run, item_id):
    run = Path(run).resolve()
    manifest = verify_run(run)
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopped.append(signum))
    torch.set_num_threads(1)
    learned = not item_id.startswith('rules/')
    if learned:
        job = next(job for job in manifest['jobs'] if job['id'] == item_id)
        config = read(run/job['config'])
        name = job['method']
        model_dir = run/job['output']
        model = verify_model(model_dir, manifest['steps_per_method'])
        model_hashes = model['checkpoint_hashes']
    else:
        name = item_id.split('/', 1)[1]
        if name not in manifest['rules']:
            raise ValueError(name)
        config = read(run/manifest['jobs'][0]['config'])
        job, model_hashes = None, {}
    output = run/'evaluation'/item_id
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(run_sha256=digest(run/'manifest.json'), item_id=item_id, model_hashes=model_hashes)
    if (output/'identity.json').exists() and read(output/'identity.json') != identity:
        raise ValueError('Evaluation inputs changed')
    write(output/'identity.json', identity)
    args = copy.deepcopy(config['env_args'])
    args['scheduler'] = 'round_robin' if name == 'RoundRobin' else 'aoi'
    args['instruction_mode_strategy'] = 'explicit_evaluation'
    args['explicit_instruction_schedule'] = [[0, 0]]
    env = SCUAVEnv(args)
    actors = load_actors(config, env, model_dir) if learned else None
    summaries = []
    total = len(manifest['scenarios'])*len(manifest['evaluation_seeds'])
    started = time.monotonic()
    try:
        for scenario, schedule in manifest['scenarios'].items():
            env.explicit_instruction_schedule = schedule
            for seed in manifest['evaluation_seeds']:
                key = f'{scenario}_seed{seed}'
                summary_file = output/'episodes'/f'{key}.json'
                trace_file = output/'traces'/f'{key}.csv'
                if summary_file.exists():
                    summary = read(summary_file)
                    if digest(trace_file) != summary['trace_sha256']:
                        raise ValueError('Committed evaluation trace changed')
                    summaries.append(summary)
                    continue
                if stopped:
                    write(output/'status.json', dict(state='paused', updated_utc=stamp(),
                        completed_episodes=len(summaries), total_episodes=total))
                    return 75
                env.seed(seed)
                obs, shared, available = env.reset()
                trace = hashlib.sha256()
                for array in (env.pos_ds, env.pos_uav, env.owner_uav, env.q_cache, env.tau_cache):
                    trace.update(array.tobytes())
                rnn = np.zeros((env.n_agents, 1, 1, 256), dtype=np.float32)
                masks = np.ones((env.n_agents, 1, 1), dtype=np.float32)
                rng = np.random.default_rng(np.random.SeedSequence([seed, 719]))
                rows = []
                for slot in range(env.max_steps):
                    trace.update(env.gamma_uav_sut.tobytes())
                    trace.update(np.asarray([env.gamma_sut_sat, env.current_instruction_id], dtype=np.float64).tobytes())
                    if actors is None:
                        actions = rule_actions(env, obs, available, name, rng)
                    else:
                        actions = []
                        with torch.no_grad():
                            for i, actor in enumerate(actors):
                                action, state = actor.act(obs[i:i+1], rnn[i], masks[i],
                                    available[i:i+1], deterministic=True)
                                actions.append(action.cpu().numpy().reshape(-1))
                                rnn[i] = state.cpu().numpy()
                    obs, shared, reward, done, infos, available = env.step(actions)
                    if bool(np.all(done)) != (slot == env.max_steps-1):
                        raise AssertionError('Episode did not terminate at 600 slots')
                    expected = next(g for t, g in reversed(schedule) if t <= slot)
                    if infos[0]['instruction_id'] != expected:
                        raise AssertionError('Instruction changed at the wrong slot')
                    row = dict(slot=slot, instruction_id=expected, **external_metrics(env, infos[0]))
                    row.update(proposed_sut_logits=json.dumps(env.last_sut_raw_action.tolist()),
                        proposed_modes=json.dumps(np.argmax(env.last_uav_raw_actions, axis=1).tolist()),
                        executed_modes=json.dumps(env.last_selected_modes.tolist()),
                        selected_ds=json.dumps(infos[0]['selected_ds']),
                        resource_fractions=json.dumps(env.beta_sut_sat.tolist()))
                    rows.append(row)
                trace_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = trace_file.with_suffix('.csv.tmp')
                with temporary.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                temporary.replace(trace_file)
                summary = dict(method=name, training_seed=job['seed'] if learned else None,
                    scenario=scenario, seed=seed, steps=len(rows),
                    external_trajectory_sha256=trace.hexdigest(), trace_sha256=digest(trace_file),
                    **aggregate(rows))
                summary['by_instruction'] = {str(g): dict(steps=len(part), **aggregate(part))
                    for g in range(3) if (part := [r for r in rows if r['instruction_id'] == g])}
                write(summary_file, summary)
                summaries.append(summary)
                write(output/'status.json', dict(state='evaluating', updated_utc=stamp(),
                    completed_episodes=len(summaries), total_episodes=total,
                    current_scenario=scenario, elapsed_seconds=time.monotonic()-started))
        result = dict(item_id=item_id, method=name, training_seed=job['seed'] if learned else None,
            overall=aggregate(summaries),
            by_scenario={s: aggregate([r for r in summaries if r['scenario'] == s]) for s in manifest['scenarios']},
            pairing={f"{r['scenario']}/{r['seed']}": r['external_trajectory_sha256'] for r in summaries})
        write(output/'summary.json', result)
        write(output/'status.json', dict(state='complete', updated_utc=stamp(),
            completed_episodes=len(summaries), total_episodes=total, slots=sum(r['steps'] for r in summaries),
            summary_sha256=digest(output/'summary.json'), model_hashes=model_hashes,
            all_executed_constraints_passed=True))
        return 0
    except BaseException as exc:
        write(output/'status.json', dict(state='failed', updated_utc=stamp(), error=repr(exc),
            completed_episodes=len(summaries), total_episodes=total))
        raise
    finally:
        env.close()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--item', required=True)
    args = parser.parse_args()
    raise SystemExit(evaluate_one(args.run, args.item))
