"""Prepare, run, resume and inspect 21 frozen SC training jobs."""
import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import stamp, write
from multiseed_protocol import prepare, read, verify_model, verify_run
from run_suite import temperature
from training_checkpoint import digest


def process_info(pid):
    try:
        path = Path('/proc')/str(pid)
        stat = (path/'stat').read_text().split(') ', 1)[1].split()
        command = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        return dict(pid=pid, uid=path.stat().st_uid, state=stat[0], start_ticks=int(stat[19]), command=command)
    except (OSError, IndexError, ValueError):
        return None


def reference_active(manifest):
    path = Path(manifest['reference_status'])
    if not path.exists():
        return []
    status = read(path)
    found = []
    for method, pid in status.get('active_pids', {}).items():
        item = process_info(pid)
        if item and item['state'] != 'Z' and method in item['command']:
            found.append(dict(method=method, pid=pid))
    supervisor = process_info(status.get('supervisor_pid', -1))
    if supervisor and supervisor['state'] != 'Z' and str(path.parent) in supervisor['command']:
        found.append(dict(method='reference_supervisor', pid=supervisor['pid']))
    return found


def recorded_process_alive(folder):
    path = Path(folder)/'process.json'
    if not path.exists():
        return False
    record = read(path)
    current = process_info(record['pid'])
    return bool(current and current['state'] != 'Z' and current['uid'] == record['uid']
                and current['start_ticks'] == record['start_ticks'])


class ThermalControl:
    def __init__(self):
        self.cooling, self.hot_since, self.cool_since, self.paused_since = False, None, None, None

    def tick(self, cpu, gpu, now):
        if cpu is None or gpu is None or not (0 <= cpu <= 150 and 0 <= gpu <= 150):
            raise RuntimeError('A required temperature sensor is unavailable or invalid')
        if self.cooling:
            self.cool_since = (now if self.cool_since is None else self.cool_since) if cpu < 75 and gpu < 75 else None
            if now-self.paused_since >= 180 and self.cool_since is not None and now-self.cool_since >= 60:
                self.cooling, self.hot_since, self.cool_since = False, None, None
                return 'resume'
            return None
        self.hot_since = (now if self.hot_since is None else self.hot_since) if cpu >= 85 else None
        if cpu >= 95 or gpu >= 83 or (self.hot_since is not None and now-self.hot_since >= 5):
            self.cooling, self.paused_since, self.cool_since = True, now, None
            return 'pause'
        return None


def gpu_temperature():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return max(float(value) for value in result.stdout.splitlines())


def stop_own_processes(jobs, grace=60):
    for job in jobs:
        if job['process'].poll() is None:
            os.killpg(job['process'].pid, signal.SIGTERM)
    deadline = time.monotonic()+grace
    while any(job['process'].poll() is None for job in jobs) and time.monotonic() < deadline:
        time.sleep(.2)
    for job in jobs:
        if job['process'].poll() is None:
            os.killpg(job['process'].pid, signal.SIGKILL)
        job['process'].wait()
        job['log'].close()


def run_phase(run, manifest, phase, resume, stopping):
    resources = manifest['resources']
    if phase == 'training':
        items = manifest['jobs']
        slots = [dict(device='cpu', core=core) for core in resources['cpu_cores']]
        slots += [dict(device='cuda:0', core=core) for core in resources['gpu_host_cores']]
    else:
        items = [dict(id=job['id'], output='evaluation/'+job['id'], device='cpu') for job in manifest['jobs']]
        items += [dict(id='rules/'+name, output='evaluation/rules/'+name, device='cpu') for name in manifest['rules']]
        slots = [dict(device='cpu', core=core) for core in resources['evaluation_cpu_cores']]
    pending = {'cpu': deque(), 'cuda:0': deque()}
    completed, failures = [], {}
    for item in items:
        path = run/item['output']/'status.json'
        if recorded_process_alive(path.parent):
            raise RuntimeError(f"Job from an earlier supervisor is still alive: {item['id']}")
        status = read(path) if path.exists() else {}
        if status.get('state') == 'complete':
            if phase == 'training':
                verify_model(path.parent, manifest['steps_per_method'])
            elif digest(path.parent/'summary.json') != status['summary_sha256']:
                raise RuntimeError('An evaluation summary changed')
            completed.append(item['id'])
            continue
        if status and not resume:
            raise RuntimeError(f"Existing unfinished job {item['id']}; use run --resume")
        pid = status.get('pid')
        prior = process_info(pid) if pid else None
        if prior and prior['state'] != 'Z' and str(run/item['output']) in prior['command']:
            raise RuntimeError(f"Job still alive without this supervisor: {item['id']} PID {pid}")
        pending[item['device']].append(dict(**item, resume=bool(status)))
    active, launched = {}, []
    thermal = ThermalControl()
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    (run/'logs').mkdir(exist_ok=True)
    source = run/'source'
    try:
        while active or any(pending.values()):
            now, temp, gpu_temp = time.monotonic(), temperature(), gpu_temperature()
            event = thermal.tick(temp, gpu_temp, now)
            if event == 'pause':
                stop_own_processes(list(active.values()))
                for job in active.values():
                    item = job['item']
                    status = read(run/item['output']/'status.json')
                    if job['process'].returncode == 0 and status.get('state') == 'complete':
                        completed.append(item['id'])
                    elif job['process'].returncode == 75 and status.get('state') == 'paused':
                        pending[item['device']].appendleft(dict(item, resume=True))
                    else:
                        failures[item['id']] = dict(returncode=job['process'].returncode, status=status)
                active.clear()
                thermal.paused_since, thermal.cool_since = time.monotonic(), None
            if event:
                with (run/'thermal_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(utc=stamp(), phase=phase, event=event,
                        cpu_c=temp, gpu_c=gpu_temp))+'\n')
            for slot_index, job in list(active.items()):
                process = job['process']
                status_path = run/job['item']['output']/'status.json'
                status = read(status_path) if status_path.exists() else {}
                progress = status.get('completed_steps' if phase == 'training' else 'completed_episodes', 0)
                if progress != job['progress']:
                    job['progress'], job['last_progress_time'] = progress, now
                if process.poll() is not None:
                    if process.returncode == 0 and status.get('state') == 'complete':
                        completed.append(job['item']['id'])
                        print(f"{phase}: {len(completed)}/{len(items)} complete: {job['item']['id']}", flush=True)
                    elif not stopping:
                        failures[job['item']['id']] = dict(returncode=process.returncode, status=status)
                    job['log'].close()
                    del active[slot_index]
                    continue
                if now-job['last_progress_time'] > 600:
                    # This timeout measures lack of progress, not total runtime.
                    stop_own_processes([job], grace=15)
                    failures[job['item']['id']] = dict(error='No progress for 600 seconds; last committed checkpoint retained')
                    del active[slot_index]
            if stopping or failures:
                break
            for index, slot in enumerate(slots):
                if thermal.cooling or index in active or not pending[slot['device']]:
                    continue
                item = pending[slot['device']].popleft()
                core_set = [slot['core']]
                if phase == 'training':
                    command = [sys.executable, str(source/'formal_train.py'), '--config', str(run/item['config']),
                        '--output', str(run/item['output']), '--run-manifest', str(run/'manifest.json'),
                        '--device', item['device'], '--checkpoint-every', str(manifest['checkpoint_every_updates'])]
                    if item['resume']:
                        command.append('--resume')
                else:
                    command = [sys.executable, str(source/'formal_eval.py'), '--run', str(run), '--item', item['id']]
                log = (run/'logs'/f"{phase}_{item['id'].replace('/', '_')}.log").open('a')
                process = subprocess.Popen(['taskset', '-c', ','.join(map(str, core_set)), *command],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
                identity = process_info(process.pid)
                if identity is not None:
                    write(run/item['output']/'process.json', identity)
                job = dict(item=item, process=process, log=log, device=slot['device'], core=slot['core'],
                    progress=0, last_progress_time=now)
                active[index] = job
                launched.append(job)
            progress_rows = {}
            total_steps = 0
            for item in manifest['jobs']:
                path = run/item['output']/'status.json'
                record = read(path) if path.exists() else {}
                total_steps += record.get('completed_steps', 0)
                progress_rows[item['id']] = dict(state=record.get('state', 'queued'),
                    steps=record.get('completed_steps', 0), target=manifest['steps_per_method'])
            report = dict(state='cooling' if thermal.cooling else phase, phase=phase,
                updated_utc=stamp(), supervisor_pid=os.getpid(), seeds=manifest['seeds'],
                completed_phase_jobs=len(completed), total_phase_jobs=len(items),
                completed_training_steps=total_steps, total_training_steps=manifest['total_training_steps'],
                active={j['item']['id']: dict(pid=j['process'].pid, device=j['device'],
                    cpu_cores=[j['core']]) for j in active.values()},
                training_progress=progress_rows, failed_jobs=failures,
                thermal_cooling=thermal.cooling, cpu_max_c=temp, gpu_max_c=gpu_temp)
            write(run/'status.json', report)
            with (run/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=stamp(), phase=phase, cpu_max_c=temp,
                    gpu_max_c=gpu_temp, cooling=thermal.cooling, active_jobs=len(active)))+'\n')
            if active or any(pending.values()):
                time.sleep(1)
        return failures, bool(stopping)
    finally:
        stop_own_processes(launched)


def run_all(run, resume=False, allow_reference_active=False, train_only=False):
    run = Path(run).resolve()
    manifest = verify_run(run)
    lock = Path(manifest['resource_lock']).open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = reference_active(manifest)
    if reference and not (allow_reference_active and manifest['purpose'].startswith('smoke')):
        raise RuntimeError(f'Original formal training is still active: {reference}. Stop or finish it before the formal switch; no original process was modified.')
    required = set(manifest['resources']['cpu_cores']+manifest['resources']['gpu_host_cores']+manifest['resources']['evaluation_cpu_cores'])
    if 15 in required or not required <= os.sched_getaffinity(0):
        raise RuntimeError('The measured CPU allocation is unavailable')
    if not __import__('torch').cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run on the host with GPU access')
    if temperature() is None:
        raise RuntimeError('CPU temperature monitoring is required on this host')
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    try:
        for phase in (['training'] if train_only else ['training', 'evaluating']):
            failures, paused = run_phase(run, manifest, phase, resume, stopping)
            if paused or failures:
                write(run/'status.json', dict(state='paused' if paused else 'attention_required',
                    updated_utc=stamp(), phase=phase, failed_jobs=failures, seeds=manifest['seeds']))
                return 75 if paused else 1
        if train_only:
            write(run/'status.json', dict(state='training_complete_evaluation_pending', updated_utc=stamp(),
                seeds=manifest['seeds'], completed_training_jobs=21, completed_training_steps=manifest['total_training_steps']))
        else:
            subprocess.run(['taskset', '-c', '16', sys.executable, str(run/'source/aggregate_seeds.py'),
                '--run', str(run)], check=True)
            verification = read(run/'report/verification.json')
            write(run/'status.json', dict(state='complete', updated_utc=stamp(), seeds=manifest['seeds'],
                completed_training_jobs=21, completed_training_steps=manifest['total_training_steps'], report=verification))
        return 0
    except BaseException as exc:
        write(run/'status.json', dict(state='attention_required', updated_utc=stamp(), error=repr(exc), seeds=manifest['seeds']))
        raise
    finally:
        lock.close()


def snapshot(run):
    manifest = read(run/'manifest.json')
    result = read(run/'status.json')
    rows = {}
    for item in manifest['jobs']:
        path = run/item['output']/'status.json'
        status = read(path) if path.exists() else {}
        rows[item['id']] = dict(state=status.get('state', 'queued'),
            steps=status.get('completed_steps', 0), device=item['device'])
    result.update(completed_training_steps=sum(r['steps'] for r in rows.values()),
        total_training_steps=manifest['total_training_steps'],
        completed_training_jobs=sum(r['state'] == 'complete' for r in rows.values()),
        total_training_jobs=len(rows), training_progress=rows)
    return result


def live_supervisor(run):
    for path in [run/'current_execution.json', run/'status.json']:
        if not path.exists():
            continue
        record = read(path)
        pid = record.get('supervisor_pid', record.get('pid'))
        current = process_info(pid) if pid else None
        if (current and current['state'] != 'Z' and current['uid'] == os.getuid()
                and str(run) in current['command'] and 'three_seed_train.py' in current['command']
                and ('start_ticks' not in record or current['start_ticks'] == record['start_ticks'])):
            return current
    return None


def start_detached(run, resume=False):
    manifest = verify_run(run)
    with (run/'.launcher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live_supervisor(run):
            raise RuntimeError('This queue is already running; use status')
        if read(run/'status.json')['state'] == 'complete':
            raise RuntimeError('This experiment is already complete')
        if not resume and any((run/j['output']/'status.json').exists() for j in manifest['jobs']):
            raise RuntimeError('Existing training state requires start --resume')
        if any(recorded_process_alive(run/j['output']) for j in manifest['jobs']):
            raise RuntimeError('A worker from an earlier supervisor is still alive')
        with Path(manifest['resource_lock']).open('a+') as resource_lock:
            fcntl.flock(resource_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, str(run/'source/three_seed_train.py'), 'run', '--output', str(run)]
        if resume:
            command.append('--resume')
        (run/'logs').mkdir(exist_ok=True)
        with (run/'logs/supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=run, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        identity = process_info(process.pid)
        if identity is None:
            raise RuntimeError('Supervisor exited during launch; inspect logs/supervisor.log')
        record = dict(identity, supervisor_pid=process.pid, utc=stamp(), resume=resume,
                      manifest_sha256=digest(run/'manifest.json'), argv=command)
        write(run/'current_execution.json', record)
        return dict(state='started', supervisor_pid=process.pid, run=str(run),
                    log=str(run/'logs/supervisor.log'))


def pause(run):
    current = live_supervisor(run)
    if current is None:
        return dict(state='no_live_supervisor', run=str(run))
    os.kill(current['pid'], signal.SIGTERM)
    return dict(state='checkpoint_and_pause_requested', supervisor_pid=current['pid'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs=3)
    p.add_argument('--steps', type=int)
    p.add_argument('--smoke', action='store_true')
    p = commands.add_parser('run')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-reference-active', action='store_true', help='Independent smoke tests only')
    p.add_argument('--train-only', action='store_true')
    p = commands.add_parser('status')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('start')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p = commands.add_parser('pause')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output, args.seeds, args.steps, args.smoke)
        print(json.dumps(dict(output=str(args.output.resolve()), seeds=result['seeds'],
            jobs=result['total_training_jobs'], steps=result['total_training_steps'], purpose=result['purpose']), indent=2))
    elif args.command == 'status':
        print(json.dumps(snapshot(args.output.resolve()), indent=2, ensure_ascii=False))
    elif args.command == 'pause':
        print(json.dumps(pause(args.output.resolve()), indent=2))
    elif args.command == 'start':
        frozen = args.output.resolve()/'source'/Path(__file__).name
        verify_run(args.output)
        if Path(__file__).resolve() != frozen:
            os.execv(sys.executable, [sys.executable, str(frozen), *sys.argv[1:]])
        print(json.dumps(start_detached(args.output.resolve(), args.resume), indent=2))
    else:
        # Always execute the frozen supervisor as well as the frozen workers.
        frozen = args.output.resolve()/'source'/Path(__file__).name
        verify_run(args.output)
        if Path(__file__).resolve() != frozen:
            os.execv(sys.executable, [sys.executable, str(frozen), *sys.argv[1:]])
        raise SystemExit(run_all(args.output, args.resume, args.allow_reference_active, args.train_only))
