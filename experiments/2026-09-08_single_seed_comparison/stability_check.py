"""Run a new, bounded seven-method check with CPU affinity and thermal stops."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import queue_supervisor as q
import resource_control

BASE = Path(__file__).resolve().parent


def cpu_temperature():
    values = []
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        if (hwmon/'name').read_text().strip() == 'coretemp':
            values.extend(int(path.read_text())/1000 for path in hwmon.glob('temp*_input'))
    if not values:
        raise RuntimeError('CPU temperature unavailable; refusing an unmonitored check')
    return max(values)


def monitor(command, root, phase, options):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONHASHSEED='0', PYTHONFAULTHANDLER='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    started = time.monotonic()
    hot_since = None
    with (root/f'{phase}.log').open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        identity = q.process_info(proc.pid)
        try:
            while proc.poll() is None:
                now = time.monotonic()
                temperature = cpu_temperature()
                with (root/'temperatures.jsonl').open('a') as samples:
                    samples.write(json.dumps(dict(utc=q.stamp(), phase=phase,
                        cpu_max_c=temperature, elapsed_seconds=now-started))+'\n')
                hot_since = (now if hot_since is None else hot_since) if temperature >= options.max_cpu_c else None
                if temperature >= options.max_cpu_c+10 or (
                        hot_since is not None and now-hot_since >= options.hot_seconds):
                    raise RuntimeError(f'Thermal guard triggered: {temperature} C')
                if now-started >= options.max_seconds:
                    raise TimeoutError(f'{phase} exceeded {options.max_seconds} seconds')
                for event_file in (root/'run/execution').glob('*/events.jsonl'):
                    if any(json.loads(line).get('event') == 'isolated_failure'
                           for line in event_file.read_text().splitlines()):
                        raise RuntimeError('Worker failure recorded; validation must not pass after retries')
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f'{phase} exited {proc.returncode}; see {root/f"{phase}.log"}')
        except BaseException as exc:
            journal = root/f'{phase}_pause.json'
            if identity and q.live(identity):
                resource_control.pause([proc.pid], journal)
            q.write(root/'status.json', dict(state='stopped_for_review', phase=phase,
                reason=repr(exc), pause_journal=str(journal) if journal.exists() else None,
                updated_utc=q.stamp()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-set', default='0,2,8,10')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=32000)
    parser.add_argument('--max-cpu-c', type=float, default=85)
    parser.add_argument('--hot-seconds', type=float, default=5)
    parser.add_argument('--max-seconds', type=float, default=600)
    options = parser.parse_args()
    cpus = {int(value) for value in options.cpu_set.split(',')}
    if not cpus or not cpus <= os.sched_getaffinity(0) or 15 in cpus:
        parser.error('Choose available CPUs excluding CPU 15 for this mitigation check')
    if not 1 <= options.jobs <= 6:
        parser.error('The guarded resource check permits one to six concurrent methods')
    if (options.steps < 8000 or options.steps % 4000 or
            not 0 < options.max_cpu_c <= 85 or options.hot_seconds < 0 or options.max_seconds <= 0):
        parser.error('Invalid step budget, temperature threshold, or deadline')
    root = options.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, cpus)
    initial = cpu_temperature()
    q.write(root/'protocol.json', dict(created_utc=q.stamp(), seed=1, steps_per_method=options.steps,
        methods=7, jobs=options.jobs, cpu_set=sorted(cpus), excluded_cpu=15,
        temperature_limit_c=options.max_cpu_c, hot_seconds=options.hot_seconds,
        phase_deadline_seconds=options.max_seconds, initial_cpu_c=initial,
        guard_source_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Engineering validation; no reward-based selection; not long-run stability certification'))
    if initial >= options.max_cpu_c:
        q.write(root/'status.json', dict(state='not_started', reason='Initial CPU temperature too high'))
        raise RuntimeError('CPU already above the selected limit; no process launched')
    monitor([sys.executable, str(BASE/'run.py'), 'prepare', '--output', str(root/'run'),
             '--steps', str(options.steps), '--seed', '1', '--jobs', str(options.jobs),
             '--device', 'hybrid', '--train-only'], root, 'prepare', options)
    monitor([sys.executable, '-u', str(BASE/'optimized_queue.py'), '--output', str(root/'run'),
             '--jobs', str(options.jobs), '--cpu-set', ','.join(map(str, sorted(cpus))),
             '--train-only'], root, 'train', options)
    states = [q.read(path) for path in (root/'run/jobs').glob('*/status.json')]
    if len(states) != 7 or any(s['state'] != 'complete' or s['attempt'] != 1 for s in states):
        raise RuntimeError('Require seven completed methods without retries')
    if any(json.loads(line).get('event') == 'isolated_failure'
           for path in (root/'run/execution').glob('*/events.jsonl') for line in path.read_text().splitlines()):
        raise RuntimeError('Failure occurred; this validation cannot pass')
    q.write(root/'status.json', dict(state='passed', completed_methods=7, steps_per_method=options.steps,
        cpu_set=sorted(cpus), jobs=options.jobs, updated_utc=q.stamp(),
        limitation='Bounded check only; hardware root cause remains unconfirmed'))
    print(json.dumps(q.read(root/'status.json')), flush=True)


if __name__ == '__main__':
    main()
