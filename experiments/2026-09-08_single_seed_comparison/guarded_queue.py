"""Fresh frozen training with a measured CPU pool and automatic thermal derating."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import time

import optimized_queue as optimized
import queue_supervisor as q
import resource_control
from stability_check import cpu_temperature


class GuardedSupervisor(optimized.OptimizedSupervisor):
    def __init__(self, out, jobs, cpus, **kwargs):
        self.fallback_cpus = [0, 2, 8, 10]
        self.hot_since = None
        self.cool_since = None
        self.cooling = None
        self.derated = False
        super().__init__(out, jobs, cpus, max_attempts=1, **kwargs)
        shutil.copyfile(Path(__file__), self.record/'guarded_queue.py')
        manifest = q.read(self.record/'manifest.json')
        manifest.update(thermal_limit_c=85, thermal_hot_seconds=5, immediate_limit_c=95,
            fallback_cpus=self.fallback_cpus, fallback_jobs=2,
            cooling_cpus=[8], cooling_mode='continue_with_one_e_core',
            restore_below_c=70, restore_cool_seconds=15, fresh_random_initialization=True)
        manifest['sources']['guarded_queue.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for name in ('resource_control.py', 'stability_check.py', 'optimized_queue.py'):
            source = Path(__file__).parent/name
            shutil.copyfile(source, self.record/name)
            manifest['sources'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        q.write(self.record/'manifest.json', manifest)

    def prepare(self, snapshot):
        if len(snapshot.get('active_pids', {})) > self.capacity or snapshot.get('suspended_jobs'):
            raise RuntimeError('Adopt only running jobs within capacity; queued jobs must be unstarted')
        super().prepare(snapshot)

    def assign_resources(self, cpus, capacity):
        self.cpus, self.capacity = list(cpus), capacity
        os.sched_setaffinity(0, self.cpus)
        for job in self.active.values():
            if q.live(job['identity']):
                optimized.set_group_affinity(job['identity'], self.cpus)

    def cycle_ready(self):
        now = time.monotonic()
        temperature = cpu_temperature()
        with (self.record/'temperatures.jsonl').open('a') as stream:
            stream.write(__import__('json').dumps(dict(utc=q.stamp(), cpu_max_c=temperature,
                jobs=self.capacity, cpus=self.cpus, cooling=self.cooling is not None))+'\n')
        if self.cooling is not None:
            self.cool_since = (now if self.cool_since is None else self.cool_since) if temperature <= 70 else None
            if self.cool_since is not None and now-self.cool_since >= 15:
                self.assign_resources(self.fallback_cpus, self.cooling['restore_jobs'])
                self.cooling = None
                self.cool_since = None
                self.hot_since = None
                self.event('thermal_resources_restored', cpu_max_c=temperature, cpus=self.cpus)
            # Keep IPC requests and timeout clocks moving during cooling.
            return True
        self.hot_since = (now if self.hot_since is None else self.hot_since) if temperature >= 85 else None
        hot = temperature >= 95 or (self.hot_since is not None and now-self.hot_since >= 5)
        if not hot:
            return True
        if not self.derated:
            self.assign_resources(self.fallback_cpus, min(self.capacity, 2))
            self.derated = True
            self.hot_since = None
            self.event('thermal_derated', cpu_max_c=temperature, cpus=self.cpus,
                       future_parallel_jobs=self.capacity)
            # Existing jobs retain their state and share the reduced CPU pool.
            if temperature < 95:
                return True
        self.cooling = dict(mode='continue_with_one_e_core', restore_jobs=self.capacity)
        self.assign_resources([8], 1)
        self.cool_since = None
        self.event('thermal_cpu_cooling', cpu_max_c=temperature, cpus=self.cpus)
        return True

    def report(self, state='training'):
        super().report(state)
        root = q.read(self.out/'status.json')
        root.update(thermal_derated=self.derated, thermal_cooling=self.cooling,
                    fresh_random_initialization=True, suspended_jobs={})
        q.write(self.out/'status.json', root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, required=True)
    parser.add_argument('--cpu-set', required=True)
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--adopt', action='store_true', help='Adopt this fresh run after its old coordinator has been retired')
    args = parser.parse_args()
    cpus = {int(value) for value in args.cpu_set.split(',')}
    if (not 1 <= args.jobs <= 6 or 15 in cpus or not cpus <= os.sched_getaffinity(0)
            or not {0, 2, 8, 10} <= cpus):
        parser.error('Use a measured CPU set excluding CPU 15 and containing fallback CPUs 0,2,8,10')
    out = args.output.resolve()
    locks = [(out/name).open('a+') for name in ('.transition.lock', '.lock', '.recovery.lock')]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = q.read(out/'status.json') if args.adopt else {}
    if args.adopt:
        restart = q.read(out/'restart_record.json')
        old = q.process_info(snapshot.get('supervisor_pid', -1))
        if restart.get('old_results_used') is not False or (old and q.live(old)):
            raise RuntimeError('Adoption requires this clean restart and an already retired coordinator')
        if snapshot.get('suspended_jobs'):
            raise RuntimeError('Existing paused slots require their recorded resume procedure')
    elif any((out/'jobs').glob('*/attempt_*')):
        raise RuntimeError('This entrypoint starts a fresh prepared run; existing attempts are preserved')
    supervisor = GuardedSupervisor(out, args.jobs, cpus, train_only=args.train_only)
    for name in supervisor.manifest['methods']:
        config = q.read(out/'configs'/f'{name}.json')
        if config['algo_args']['train'].get('model_dir') is not None:
            raise RuntimeError('Fresh training must not load a prior model directory')
    supervisor.prepare(snapshot)
    supervisor.report('prepared')
    try:
        supervisor.run()
    except BaseException as exc:
        roots = [j['identity']['pid'] for j in supervisor.active.values() if q.live(j['identity'])]
        if roots:
            resource_control.pause(roots, supervisor.record/'unexpected_exit_pause.json')
        supervisor.report('attention_required')
        supervisor.event('supervisor_failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
