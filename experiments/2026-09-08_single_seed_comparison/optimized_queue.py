"""Resource-bounded supervisor with safe live adoption and rolling throughput."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import queue_supervisor as q


def resources():
    cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    mem = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    result = dict(cpu_total=sum(cpu), cpu_idle=cpu[3]+cpu[4], memory_available_mib=mem['MemAvailable']/1024)
    try:
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free',
                                      '--format=csv,noheader,nounits'], text=True, timeout=5)
        gpu = list(map(float,raw.splitlines()[0].split(',')))
        result.update(gpu_utilization_percent=gpu[0], gpu_memory_used_mib=gpu[1], gpu_memory_free_mib=gpu[2])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        result.update(gpu_utilization_percent=None, gpu_memory_used_mib=None,
                      gpu_memory_free_mib=None, gpu_query_error=repr(exc))
    return result


def set_group_affinity(identity, cpus):
    if not q.live(identity) or identity['pgid'] != identity['pid']:
        raise RuntimeError('Cannot assign resources to an unverified training group')
    count = 0
    for member in q.group_members(identity['pgid']):
        for thread in (Path('/proc')/str(member['pid'])/'task').glob('*'):
            try:
                os.sched_setaffinity(int(thread.name), cpus)
                count += 1
            except ProcessLookupError:
                pass
    return count


class OptimizedSupervisor(q.Supervisor):
    def __init__(self, out, jobs, cpus, **kwargs):
        self.cpus = sorted(cpus)
        self.last_sample = None
        self.last_sample_time = 0
        self.last_steps = {}
        self.last_admission = None
        self.last_admission_time = 0
        self.resource_wait = None
        os.sched_setaffinity(0, self.cpus)
        super().__init__(out, jobs, **kwargs)
        destination = self.record/'optimized_queue.py'
        shutil.copyfile(Path(__file__), destination)
        manifest = q.read(self.record/'manifest.json')
        manifest.update(training_cpus=self.cpus, library_threads=1,
            ram_headroom_mib=4096, gpu_headroom_mib=1024,
            admission='Require 2 GiB RAM and 768 MiB GPU for each new job, in addition to headroom',
            sampling_seconds=10)
        manifest['sources']['optimized_queue.py'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        q.write(self.record/'manifest.json', manifest)

    def prepare(self, snapshot):
        super().prepare(snapshot)
        for method,job in self.active.items():
            threads = set_group_affinity(job['identity'], self.cpus)
            self.event('resource_assignment', method=method, training_cpus=self.cpus, threads=threads)

    def can_launch(self):
        now = time.monotonic()
        if self.last_admission is None or now-self.last_admission_time >= 5:
            self.last_admission = resources()
            self.last_admission_time = now
        sample = self.last_admission
        return (sample['gpu_memory_free_mib'] is not None and
                sample['memory_available_mib'] >= 4096+2048 and sample['gpu_memory_free_mib'] >= 1024+768)

    def launch(self, method):
        super().launch(method)
        if method in self.active:
            # Reserve initialization demand immediately; nvidia-smi may lag startup.
            self.last_admission['memory_available_mib'] -= 2048
            self.last_admission['gpu_memory_free_mib'] -= 768

    def cycle_ready(self):
        return True

    def job_health(self, job, path):
        return q.health(job, path, self.stale_seconds)

    def report(self, state='training'):
        super().report(state)
        root = q.read(self.out/'status.json')
        root.update(training_cpus=self.cpus, resource_wait=self.resource_wait,
                    resource_metrics_file=str(self.record/'resources.jsonl'))
        q.write(self.out/'status.json', root)
        now = time.monotonic()
        if now-self.last_sample_time < 10:
            return
        sample = resources()
        steps = {m:v['completed_steps'] for m,v in root['progress'].items()}
        entry = dict(utc=q.stamp(), active_methods=list(self.active), progress=steps,
                     training_cpus=self.cpus, **sample)
        if self.last_sample:
            total = sample['cpu_total']-self.last_sample['cpu_total']
            idle = sample['cpu_idle']-self.last_sample['cpu_idle']
            entry['whole_machine_cpu_busy_percent'] = 100*(1-idle/max(total,1))
            deltas = {m:max(0,steps[m]-self.last_steps.get(m,steps[m])) for m in steps}
            entry['method_steps_per_second'] = {m:v/(now-self.last_sample_time) for m,v in deltas.items()}
            entry['aggregate_steps_per_second'] = sum(deltas.values())/(now-self.last_sample_time)
        with (self.record/'resources.jsonl').open('a') as stream:
            stream.write(__import__('json').dumps(entry)+'\n')
        self.last_sample, self.last_steps, self.last_sample_time = sample, steps, now

    def run(self):
        while self.active or self.pending:
            if not self.cycle_ready():
                self.report('cooling')
                time.sleep(2)
                continue
            for method,job in list(self.active.items()):
                if job['proc'] is not None:
                    job['proc'].poll()
                path = self.job_path(method)
                status = q.read(path) if path.exists() else {}
                if not q.live(job['identity']) and status.get('state') == 'complete':
                    self.release(method)
                elif reason := self.job_health(job,path):
                    self.release(method,reason)
            while self.pending and len(self.active) < self.capacity:
                if not self.can_launch():
                    if self.resource_wait is None:
                        self.event('waiting_for_memory', resources=self.last_admission)
                    self.resource_wait = 'Insufficient RAM/GPU startup headroom'
                    break
                self.resource_wait = None
                self.launch(self.pending.pop(0))
            self.report()
            time.sleep(2)
        super().run()  # Shared immutable completion/checkpoint/evaluation checks.


def retire_coordinator(supervisor, snapshot, pid):
    if snapshot.get('supervisor_pid') != pid:
        raise RuntimeError('Exact coordinator PID does not match this run')
    identity = q.process_info(pid)
    if not identity or identity['uid'] != os.getuid():
        raise RuntimeError('Coordinator is not owned/live')
    argv = (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
    if not any(Path(v).name in ('run.py','queue_supervisor.py','optimized_queue.py') for v in argv):
        raise RuntimeError('Unexpected coordinator executable')
    if '--output' not in argv:
        raise RuntimeError('Coordinator has no explicit run directory')
    old_out = Path(argv[argv.index('--output')+1])
    if not old_out.is_absolute():
        old_out = (Path('/proc')/str(pid)/'cwd').resolve()/old_out
    if old_out.resolve() != supervisor.out:
        raise RuntimeError('Coordinator belongs to another run')
    children = []
    for method,child_pid in snapshot.get('active_pids',{}).items():
        state = q.read(supervisor.job_path(method))
        child = q.process_info(child_pid)
        if state.get('pid') != child_pid or state.get('method') != method:
            raise RuntimeError('Active training status identity mismatch')
        if child and (child['uid'] != os.getuid() or child['pgid'] != child_pid):
            raise RuntimeError('Active training group is not isolated/owned')
        if child:
            children.append(child)
    q.write(supervisor.record/'takeover_identities.json', dict(parent=identity,children=children))
    # Some jobs were adopted by the old coordinator and are not its OS children.
    # Their identities were checked above; stop only the coordinating PID.
    q.replace_supervisor(identity, [])
    supervisor.event('coordinator_replaced',old_pid=pid,retained_child_pids=[p['pid'] for p in children])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,required=True)
    parser.add_argument('--cpu-set',required=True)
    parser.add_argument('--takeover-pid',type=int)
    parser.add_argument('--train-only',action='store_true')
    args = parser.parse_args()
    cpus = {int(v) for v in args.cpu_set.split(',')}
    if not cpus or not cpus <= os.sched_getaffinity(0) or not 1 <= args.jobs <= 7:
        parser.error('Invalid CPU set or job count')
    out = args.output.resolve()
    with (out/'.transition.lock').open('a+') as transition:
        fcntl.flock(transition,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks = [(out/name).open('a+') for name in ('.lock','.recovery.lock')]
        try:
            blocked = False
            for lock in locks:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    blocked = True
            if blocked and args.takeover_pid is None:
                raise RuntimeError('Queue already managed; use its exact --takeover-pid only for an intentional handover')
            supervisor = OptimizedSupervisor(out,args.jobs,cpus,train_only=args.train_only)
            snapshot = q.read(out/'status.json')
            q.write(supervisor.record/'status_before.json',snapshot)
            if blocked:
                retire_coordinator(supervisor,snapshot,args.takeover_pid)
                deadline = time.monotonic()+10
                for lock in locks:
                    while True:
                        try:
                            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic()>deadline:
                                raise RuntimeError('Coordinator lock did not release; jobs preserved')
                            time.sleep(.1)
            supervisor.prepare(snapshot)
            supervisor.report()
            fcntl.flock(transition,fcntl.LOCK_UN)
            try:
                supervisor.run()
            except BaseException as exc:
                supervisor.event('supervision_failed_children_preserved',error=repr(exc))
                supervisor.report('supervision_failed')
                raise
        finally:
            for lock in locks:
                lock.close()


if __name__ == '__main__':
    main()
