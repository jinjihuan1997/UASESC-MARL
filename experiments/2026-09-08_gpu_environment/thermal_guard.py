"""Identity-checked thermal pause/cooldown/resume around an immutable experiment.

The guard does not change any frozen training source, seed, budget or device.
Its own SIGTERM disables monitoring without stopping or restarting the workload.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def identity(pid):
    try:
        path = Path('/proc')/str(pid)
        raw = (path/'stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        return dict(pid=int(pid), uid=path.stat().st_uid, state=fields[0],
            start_ticks=int(fields[19]),
            argv=[x.decode(errors='replace') for x in (path/'cmdline').read_bytes().split(b'\0') if x])
    except (OSError, ValueError, IndexError):
        return None


def same_live(record):
    if not record:
        return False
    current = identity(record.get('pid', record.get('supervisor_pid', -1)))
    return bool(current and current['state'] != 'Z' and current['uid'] == record.get('uid')
                and current['start_ticks'] == record.get('start_ticks'))


def cpu_temperature():
    values = []
    try:
        for folder in Path('/sys/class/hwmon').glob('hwmon*'):
            if (folder/'name').read_text().strip() != 'coretemp':
                continue
            values.extend(float(p.read_text())/1000 for p in folder.glob('temp*_input'))
    except (OSError, ValueError):
        return None
    return max(values) if values and all(math.isfinite(x) and 0 < x < 150 for x in values) else None


class ThermalPolicy:
    def __init__(self, hot_c=85, emergency_c=95, hot_seconds=5,
                 cool_c=75, cool_seconds=60, minimum_pause=180):
        self.hot_c, self.emergency_c, self.hot_seconds = hot_c, emergency_c, hot_seconds
        self.cool_c, self.cool_seconds, self.minimum_pause = cool_c, cool_seconds, minimum_pause
        self.hot_since = self.cool_since = self.paused_since = None

    def stop_reason(self, temperature, now):
        if temperature is None or not math.isfinite(temperature):
            self.hot_since = None
            return 'temperature_sensor_unavailable'
        if temperature >= self.emergency_c:
            return 'cpu_emergency_temperature'
        if temperature >= self.hot_c:
            if self.hot_since is None:
                self.hot_since = now
            if now-self.hot_since >= self.hot_seconds:
                return 'cpu_sustained_high_temperature'
        else:
            self.hot_since = None
        return None

    def begin_cooling(self, now):
        self.paused_since, self.cool_since, self.hot_since = now, None, None

    def can_resume(self, temperature, now):
        if self.paused_since is None:
            self.begin_cooling(now)
        if temperature is not None and math.isfinite(temperature) and temperature < self.cool_c:
            if self.cool_since is None:
                self.cool_since = now
        else:
            self.cool_since = None
        return (now-self.paused_since >= self.minimum_pause and self.cool_since is not None
                and now-self.cool_since >= self.cool_seconds)


def thermal_exit(status):
    error = status.get('error', '')
    return status.get('state') == 'attention_required' and any(
        marker in error for marker in ('95C emergency', 'CPU temperature sensor unavailable'))


class Guard:
    def __init__(self, run, policy=None):
        self.run = Path(run).resolve()
        self.folder = self.run/'thermal_guard'
        self.folder.mkdir(exist_ok=True)
        self.lock = (self.folder/'.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.manifest = read(self.run/'manifest.json')
        self.policy = policy or ThermalPolicy()
        self.control = read(self.folder/'control.json')
        self.stopping_since = None
        self.children = []
        self.observation = 0
        self.guard_identity = identity(os.getpid())
        write(self.folder/'process.json', self.guard_identity)
        self.event('guard_started', recovered_control=self.control.get('mode'))

    def event(self, name, **data):
        with (self.folder/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(utc=stamp(), event=name, **data), ensure_ascii=False)+'\n')

    def set_control(self, **value):
        self.control = value
        write(self.folder/'control.json', value)

    def supervisor(self):
        record = read(self.run/'current_execution.json') or read(self.run/'launch_record.json')
        if not same_live(record):
            return None
        current = identity(record.get('pid', record.get('supervisor_pid')))
        if str(self.run/'source/three_seed_train.py') not in current['argv']:
            raise RuntimeError('Live supervisor identity has an unexpected executable')
        if '--output' not in current['argv'] or str(self.run) not in current['argv']:
            raise RuntimeError('Live supervisor command does not identify this run')
        return current

    def workers(self):
        found = []
        for parent in ('jobs', 'evaluation'):
            for path in (self.run/parent).glob('*/*/process.json'):
                record = read(path)
                if same_live(record):
                    current = identity(record['pid'])
                    expected = self.run/'source'/('formal_train.py' if parent == 'jobs' else 'formal_eval.py')
                    if str(expected) not in current['argv']:
                        raise RuntimeError(f'Worker identity has unexpected executable: {path}')
                    found.append(current)
        return found

    def signal_term(self, record):
        if same_live(record):
            # Recheck the complete identity immediately before this scoped signal.
            os.kill(record['pid'], signal.SIGTERM)
            self.event('sigterm_sent', pid=record['pid'], start_ticks=record['start_ticks'])

    def request_pause(self, supervisor, reason, temperature, now):
        self.set_control(mode='stopping', reason=reason, requested_utc=stamp(), temperature_c=temperature)
        self.stopping_since = now
        self.event('thermal_pause_requested', reason=reason, temperature_c=temperature)
        if supervisor:
            self.signal_term(supervisor)

    def progress(self):
        steps, complete = 0, 0
        for job in self.manifest['jobs']:
            value = read(self.run/job['output']/'status.json')
            steps += value.get('completed_steps', 0)
            complete += value.get('state') == 'complete'
        return dict(completed_training_steps=steps, completed_training_jobs=complete,
                    total_training_steps=self.manifest['total_training_steps'])

    def verify_saved_state(self, execution):
        command = ['/home/king/miniconda3/envs/harl_sionna/bin/python', str(Path(__file__).resolve()),
                   'verify-resume', '--run', str(self.run), '--report', str(execution/'checkpoint_verification.json')]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (execution/'verification.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=120, check=True)
        return read(execution/'checkpoint_verification.json')

    def resume(self):
        if self.supervisor() or self.workers():
            raise RuntimeError('Refuse resume: earlier processes still alive')
        execution = self.run/'execution'/datetime.now(timezone.utc).strftime('thermal_resume_%Y%m%d_%H%M%S_%f')
        execution.mkdir(parents=True)
        # Checkpoint/source verification occurs with training stopped.
        verified = self.verify_saved_state(execution)
        # Do not launch if temperature rose during verification.
        temp = cpu_temperature()
        if temp is None or temp >= self.policy.cool_c:
            self.policy.cool_since = None
            self.event('restart_deferred_after_verification', temperature_c=temp)
            return False
        with Path(self.manifest['resource_lock']).open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = ['/home/king/miniconda3/envs/harl_sionna/bin/python',
                   str(self.run/'source/three_seed_train.py'), 'run', '--resume', '--output', str(self.run)]
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with (execution/'supervisor.log').open('x') as log:
            process = subprocess.Popen(command, cwd=self.run, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env, close_fds=True)
        self.children.append(process)
        record = identity(process.pid)
        if record is None:
            raise RuntimeError('Resumed supervisor exited before its identity could be recorded')
        record.update(supervisor_pid=process.pid, started_utc=stamp(), command=command,
            detached=True, resume=True, fresh_start=False, resumed_from_steps=verified['completed_training_steps'],
            execution_directory=str(execution), log=str(execution/'supervisor.log'),
            checkpoint_verification=str(execution/'checkpoint_verification.json'),
            source_manifest_changed=False, hyperparameters_changed=False,
            resource_allocation=self.manifest['resources'], reason='automatic_thermal_recovery')
        write(execution/'launch_record.json', record)
        write(self.run/'current_execution.json', record)
        self.set_control(mode='restarting', supervisor_pid=process.pid, started_utc=stamp())
        self.event('thermal_resume_started', supervisor_pid=process.pid,
                   completed_training_steps=verified['completed_training_steps'])
        self.policy.hot_since = self.policy.cool_since = self.policy.paused_since = None
        return True

    def tick(self, temperature, now):
        for child in list(self.children):
            if child.poll() is not None:
                self.children.remove(child)
        status = read(self.run/'status.json')
        supervisor = self.supervisor()
        mode = self.control.get('mode')
        if status.get('state') == 'complete':
            return self.publish('complete', temperature)
        if mode == 'attention_required':
            raise RuntimeError('Guard failure is latched for review; do not automatically retry it')
        if supervisor:
            if mode == 'restarting' and status.get('supervisor_pid') == supervisor['pid']:
                self.set_control(mode='watching')
                self.event('thermal_resume_confirmed', supervisor_pid=supervisor['pid'])
                mode = 'watching'
            if mode == 'stopping':
                if self.stopping_since is None:
                    self.stopping_since = now
                    self.signal_term(supervisor)
                if now-self.stopping_since > 120:
                    raise RuntimeError('Orderly shutdown has not finished after 120 seconds')
                return self.publish('saving_and_stopping', temperature)
            reason = self.policy.stop_reason(temperature, now)
            if reason:
                self.request_pause(supervisor, reason, temperature, now)
                return self.publish('saving_and_stopping', temperature)
            return self.publish('monitoring', temperature)
        # Never interpret an unrelated crash or an explicit manual pause as a thermal restart.
        if mode == 'restarting':
            raise RuntimeError('The automatically resumed supervisor exited before a healthy status was observed')
        if mode == 'stopping' or mode == 'cooling' or thermal_exit(status):
            workers = self.workers()
            if workers:
                if self.stopping_since is None:
                    self.stopping_since = now
                for worker in workers:
                    self.signal_term(worker)
                if now-self.stopping_since > 120:
                    raise RuntimeError('Workers remain alive after orderly-stop grace period')
                return self.publish('saving_and_stopping', temperature)
            if status.get('failed_jobs'):
                raise RuntimeError('Non-thermal worker failure requires review before resume')
            if self.policy.paused_since is None:
                self.policy.begin_cooling(now)
                self.set_control(mode='cooling', cooling_started_utc=stamp(),
                                 reason=self.control.get('reason', status.get('error', 'thermal_exit')))
                self.event('cooldown_started', **self.progress())
            if self.policy.can_resume(temperature, now):
                if self.resume():
                    return self.publish('restarting', temperature)
            return self.publish('cooling', temperature, elapsed_cooldown_seconds=now-self.policy.paused_since,
                continuous_cool_seconds=0 if self.policy.cool_since is None else now-self.policy.cool_since)
        if status.get('state') == 'paused':
            return self.publish('manual_pause_preserved', temperature)
        if status.get('state') == 'training_complete_evaluation_pending':
            return self.publish('evaluation_pending', temperature)
        raise RuntimeError(f"Supervisor is absent with non-thermal status: {status.get('state')}, {status.get('error')}")

    def publish(self, state, temperature, **extra):
        self.observation += 1
        report = dict(state=state, updated_utc=stamp(), guard_pid=os.getpid(),
            cpu_max_c=temperature, policy=dict(hot_c=self.policy.hot_c, emergency_c=self.policy.emergency_c,
                hot_seconds=self.policy.hot_seconds, cool_below_c=self.policy.cool_c,
                cool_seconds=self.policy.cool_seconds, minimum_pause_seconds=self.policy.minimum_pause),
            **self.progress(), **extra)
        write(self.folder/'status.json', report)
        if self.observation % 5 == 0:
            with (self.folder/'temperatures.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=stamp(), state=state, cpu_max_c=temperature))+'\n')
        return state

    def close(self):
        self.lock.close()


def verify_resume(run, report):
    run = Path(run).resolve()
    sys.path.insert(0, str(run/'source'))
    from multiseed_protocol import verify_run, verify_model
    from training_checkpoint import load_checkpoint, digest
    manifest = verify_run(run)
    checks = []
    for job in manifest['jobs']:
        folder = run/job['output']
        status = read(folder/'status.json')
        if not status:
            if (folder/'checkpoints/index.json').exists():
                raise RuntimeError(f"Checkpoint without a committed job status: {job['id']}")
            continue
        if status.get('state') not in ('paused', 'complete'):
            raise RuntimeError(f"Job is not safely paused or complete: {job['id']}: {status.get('state')}")
        state, checkpoint = load_checkpoint(folder/'checkpoints')
        if checkpoint['selected'] != 'current' or checkpoint['rejected']:
            raise RuntimeError('Automatic thermal resume requires the current checkpoint to verify')
        expected = dict(config_sha256=digest(run/job['config']), run_manifest_sha256=digest(run/'manifest.json'), device=job['device'])
        if state['identity'] != expected or state['update']*manifest['batch'] != status['completed_steps']:
            raise RuntimeError('Saved state identity or step counter does not match the formal experiment')
        if state['total_updates']*manifest['batch'] != manifest['steps_per_method']:
            raise RuntimeError('Saved training budget changed')
        if status['state'] == 'complete':
            verify_model(folder, manifest['steps_per_method'])
        checks.append(dict(job=job['id'], steps=status['completed_steps'], checkpoint_sha256=checkpoint['sha256']))
    result = dict(passed=True, utc=stamp(), completed_training_steps=sum(x['steps'] for x in checks), checks=checks)
    write(report, result)
    return result


def monitor(run):
    guard = Guard(run)
    quitting = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: quitting.append(signum))
    try:
        while not quitting:
            if guard.tick(cpu_temperature(), time.monotonic()) == 'complete':
                guard.event('experiment_complete')
                return 0
            time.sleep(1)
        guard.event('guard_disabled_by_signal')
        guard.publish('disabled', cpu_temperature())
        return 0
    except BaseException as exc:
        guard.event('guard_requires_attention', error=repr(exc))
        guard.set_control(mode='attention_required', error=repr(exc))
        write(guard.folder/'status.json', dict(state='attention_required', updated_utc=stamp(),
            guard_pid=os.getpid(), error=repr(exc), **guard.progress()))
        raise
    finally:
        guard.close()


def start_monitor(run):
    run = Path(run).resolve()
    folder = run/'thermal_guard'
    source = folder/'source/thermal_guard.py'
    manifest = read(folder/'manifest.json')
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise RuntimeError('Frozen thermal guard source changed')
    with (folder/'.start.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = read(folder/'process.json')
        if same_live(existing):
            return dict(state='already_running', process=existing)
        if read(folder/'control.json').get('mode') == 'attention_required':
            raise RuntimeError('Review the latched guard failure before restarting it')
        command = ['/usr/bin/python3', str(source), 'monitor', '--run', str(run)]
        with (folder/'guard.log').open('a') as log:
            process = subprocess.Popen(command, cwd=run, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Thermal guard exited {process.returncode}; see {folder}/guard.log')
            status = read(folder/'status.json')
            record = read(folder/'process.json')
            if status.get('guard_pid') == process.pid and same_live(record):
                result = dict(state='started', utc=stamp(), process=record, command=command,
                              status_file=str(folder/'status.json'))
                write(folder/'launch_record.json', result)
                return result
            time.sleep(.1)
        raise RuntimeError('Guard launch was not acknowledged within 10 seconds; inspect before retrying')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'monitor', 'verify-resume'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.command == 'start':
        print(json.dumps(start_monitor(args.run), indent=2))
    elif args.command == 'verify-resume':
        if args.report is None:
            parser.error('verify-resume requires --report')
        print(json.dumps(verify_resume(args.run, args.report), indent=2))
    else:
        raise SystemExit(monitor(args.run))
