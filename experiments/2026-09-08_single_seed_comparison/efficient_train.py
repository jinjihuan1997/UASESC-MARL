#!/usr/bin/env python3
"""Measure whole-comparison throughput, then run the frozen queue on that resource plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import queue_supervisor as q
import resource_control

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_ticks():
    row = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(row), row[3] + row[4]


def gpu_sample():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
    return list(map(float, result.stdout.splitlines()[0].split(',')))


def weights(out):
    manifest = q.read(out / 'manifest.json')
    if any(sha(out/p) != h for p,h in manifest['input_hashes'].items()):
        raise RuntimeError('Frozen inputs changed')
    result = {}
    for method in manifest['methods']:
        state = q.read(out / 'jobs' / method / 'status.json')
        if state['state'] != 'complete':
            raise RuntimeError(f'Incomplete benchmark method: {method}')
        qhashes = state['checkpoint_hashes']
        if any(sha(p) != h for p,h in qhashes.items()):
            raise RuntimeError('Checkpoint hash mismatch')
        result[method] = dict(initial_actor=state['initial_actor_hashes'],
            initial_critic=state['initial_critic_hash'], final_actor=state['final_actor_hashes'],
            final_critic=state['final_critic_hash'],
            files={Path(p).name:h for p,h in qhashes.items()})
    return result


def resume_training(journal, snapshot, out):
    """Resume workers first so the watchdog sees fresh progress, not paused time."""
    supervisor_pid = snapshot['supervisor_pid']
    resource_control.resume(journal, defer_pids=[supervisor_pid])
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        ready = True
        for method,pid in snapshot['active_pids'].items():
            state = q.read(out / 'jobs' / method / 'status.json')
            if q.process_info(pid) and state.get('state') != 'complete':
                ready &= state.get('completed_steps', 0) > snapshot['progress'][method]['completed_steps']
        if ready:
            break
        time.sleep(.5)
    resource_control.resume(journal)


def benchmark(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))
    if args.reserve_cpus < 0 or len(cpus) <= args.reserve_cpus:
        raise ValueError('No training CPUs remain')
    selected = cpus[:len(cpus)-args.reserve_cpus] if args.reserve_cpus else cpus
    # Inherited by every coordinator, trainer, and environment worker in these trials.
    os.sched_setaffinity(0, selected)
    driver = out/'execution_driver'
    driver.mkdir()
    for name in ('queue_supervisor.py','optimized_queue.py','spawn_train.py','resource_control.py','efficient_train.py'):
        shutil.copyfile(BASE/name,driver/name)
    old_snapshot, pause_journal = None, None
    results, reference = [], None
    q.write(out/'protocol.json', dict(created_utc=q.stamp(), candidate_jobs=args.candidates,
        training_cpus=selected, reserved_cpus=sorted(set(cpus)-set(selected)),
        seed=args.seed, steps_per_method=args.steps, device='hybrid', logical_envs=10,
        driver_hashes={p.name:sha(p) for p in driver.glob('*.py')},
        criterion='Among trials without failures/retries, lowest whole-comparison wall time; fewer jobs within 5% of fastest',
        note='Engineering benchmark only; no reward or evaluation result used in selection'))
    try:
        if args.pause_output:
            current = args.pause_output.resolve()
            old_snapshot = q.read(current/'status.json')
            if old_snapshot['state'] != 'training':
                raise RuntimeError('Only an active training queue can be temporarily paused')
            pause_journal = out/'long_training_pause.json'
            resource_control.pause([old_snapshot['supervisor_pid'], *old_snapshot['active_pids'].values()], pause_journal)
            q.write(out/'paused_queue_snapshot.json', old_snapshot)
        for jobs in args.candidates:
            trial = out / f'jobs_{jobs}'
            command = [sys.executable, str(BASE/'run.py'), 'prepare', '--seed', str(args.seed),
                '--steps', str(args.steps), '--device', 'hybrid', '--jobs', str(jobs),
                '--train-only', '--output', str(trial)]
            with (out/f'prepare_{jobs}.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            start = time.monotonic(); c0 = cpu_ticks(); samples = []
            with (out/f'queue_{jobs}.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-u', str(driver/'optimized_queue.py'),
                    '--output', str(trial), '--jobs', str(jobs), '--cpu-set', ','.join(map(str,selected)), '--train-only'],
                    stdout=log, stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        samples.append(dict(utc=q.stamp(), gpu=gpu_sample()))
                        time.sleep(1)
                except BaseException:
                    if proc.poll() is None:
                        identity = q.process_info(proc.pid)
                        aborted = resource_control.pause([proc.pid],out/f'aborted_trial_{jobs}.json')
                        q.replace_supervisor(identity,[])
                        for process in aborted['processes']:
                            if process['pid'] != proc.pid and process['pgid'] == process['pid']:
                                q.stop_group(process,grace=.1)
                        proc.wait(timeout=5)
                    raise
                if proc.returncode:
                    raise RuntimeError(f'Benchmark failed; see {out / f"queue_{jobs}.log"}')
            elapsed = time.monotonic()-start; c1 = cpu_ticks()
            current_weights = weights(trial)
            if reference is None:
                reference = current_weights
            if current_weights != reference:
                raise RuntimeError('Parallel scheduling changed matched benchmark weights')
            retries = sum(q.read(p)['attempt']-1 for p in (trial/'jobs').glob('*/status.json'))
            item = dict(jobs=jobs, seconds=elapsed, steps_per_second=7*args.steps/elapsed,
                cpu_busy_percent=100*(1-(c1[1]-c0[1])/(c1[0]-c0[0])),
                gpu_utilization_mean=sum(s['gpu'][0] for s in samples)/len(samples),
                gpu_memory_max_mib=max(s['gpu'][1] for s in samples),
                all_seven_checkpoint_hashes_equal=True, retry_count=retries,
                clean_completion=retries==0, output=str(trial))
            q.write(out/f'resources_{jobs}.json', dict(result=item, samples=samples))
            results.append(item)
            q.write(out/'results.json', results)
            print(json.dumps(item), flush=True)
        eligible = [r for r in results if r['clean_completion']]
        if not eligible:
            raise RuntimeError('All trials needed retries; no clean resource plan can be selected')
        fastest = min(r['seconds'] for r in eligible)
        chosen = min((r for r in eligible if r['seconds'] <= fastest*1.05), key=lambda r:r['jobs'])
        plan = dict(created_utc=q.stamp(), parallel_jobs=chosen['jobs'], training_cpus=selected,
            reserved_cpus=sorted(set(cpus)-set(selected)), device='hybrid', library_threads=1,
            benchmark_steps_per_method=args.steps, all_seven_weights_equal=True, results=results,
            selection_rule='Among trials without retries, fewest jobs within 5% of fastest complete-comparison wall time',
            final_driver='optimized_queue.py', driver_hashes=q.read(out/'protocol.json')['driver_hashes'],
            benchmark_only=True, limitation='Short fixed-seed benchmark; no guarantee of long-run speed or stability')
        q.write(out/'resource_plan.json', plan)
        print('Selected resource plan: '+str(out/'resource_plan.json'), flush=True)
    finally:
        if pause_journal is not None and pause_journal.exists():
            resume_training(pause_journal, old_snapshot, args.pause_output.resolve())


def run(args):
    plan = q.read(args.resource_plan)
    for name in ('queue_supervisor.py','optimized_queue.py','spawn_train.py'):
        if plan.get('driver_hashes',{}).get(name) != sha(BASE/name):
            raise RuntimeError(f'Execution source differs from measured plan: {name}; benchmark this version first')
    cpus = plan['training_cpus']
    if not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError('Resource plan CPUs unavailable on this host')
    gpu_sample()  # Fail explicitly if the planned GPU is inaccessible.
    out = args.output.resolve()
    if not out.exists():
        subprocess.run([sys.executable, str(BASE/'run.py'), 'prepare', '--seed', str(args.seed),
            '--steps', str(args.steps), '--device', 'hybrid', '--jobs', str(plan['parallel_jobs']),
            '--output', str(out)], check=True)
    manifest = q.read(out/'manifest.json')
    if manifest['device'] != 'hybrid':
        raise RuntimeError('This measured plan requires a frozen hybrid run')
    record = out/'execution'/('resource_plan_'+str(time.time_ns())+'.json')
    q.write(record, dict(plan=plan, plan_file=str(args.resource_plan.resolve()),
        plan_sha256=sha(args.resource_plan), entrypoint_sha256=sha(__file__)))
    shutil.copyfile(Path(__file__), record.with_suffix('.py'))
    command = [sys.executable, '-u', str(BASE/'optimized_queue.py'), '--output', str(out),
        '--jobs', str(plan['parallel_jobs']), '--cpu-set', ','.join(map(str,cpus))]
    if args.takeover_pid:
        command += ['--takeover-pid', str(args.takeover_pid)]
    if args.train_only:
        command += ['--train-only']
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    bench = sub.add_parser('benchmark')
    bench.add_argument('--output', type=Path, required=True)
    bench.add_argument('--candidates', type=int, nargs='+', default=[4,5,7])
    bench.add_argument('--reserve-cpus', type=int, default=2)
    bench.add_argument('--pause-output', type=Path)
    bench.add_argument('--steps', type=int, default=32000)
    bench.add_argument('--seed', type=int, default=1)
    start = sub.add_parser('run')
    start.add_argument('--output', type=Path, required=True)
    start.add_argument('--resource-plan', type=Path, required=True)
    start.add_argument('--seed', type=int, default=1)
    start.add_argument('--steps', type=int, default=10000000)
    start.add_argument('--takeover-pid', type=int)
    start.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'benchmark':
        if len(set(args.candidates)) != len(args.candidates) or not all(1 <= n <= 7 for n in args.candidates):
            parser.error('Candidate job counts must be unique and between 1 and 7')
        benchmark(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
