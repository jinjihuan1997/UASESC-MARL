"""Bounded, matched seven-method resource probe; never controls formal jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import ROOT, stamp, verify_reference, write
from run_suite import temperature


CPU_METHODS = [
    'IC_HAPPO', 'HAPPO_hidden_instruction', 'HAPPO_no_task_aux_reward',
    'HAPPO_fixed_mode_rule', 'HAPPO_equal_resources',
]
GPU_METHODS = ['IC_MAPPO', 'MAPPO_hidden_instruction']
PROFILES = {
    'cpu3_gpu1': {'cpu': [16, 17, 19], 'cuda:0': [18]},
    'cpu3_gpu2': {'cpu': [16, 17, 19], 'cuda:0': [18, 9]},
    'cpu5_gpu2': {'cpu': [16, 17, 19, 11, 13], 'cuda:0': [18, 9]},
    'cpu6_gpu1': {'cpu': [16, 17, 19, 11, 13, 9], 'cuda:0': [18]},
}
METHOD_ALLOCATION = {
    'cpu6_gpu1': {
        'cpu': ['IC_HAPPO', 'HAPPO_hidden_instruction', 'HAPPO_no_task_aux_reward',
                'HAPPO_fixed_mode_rule', 'IC_MAPPO', 'MAPPO_hidden_instruction'],
        'cuda:0': ['HAPPO_equal_resources'],
    },
}
FORMAL_STATUS = ROOT.parent/'2026-09-08_single_seed_comparison/runs/seed1_clean_restart_20260908/status.json'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def resources(include_gpu=False):
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            mem[key] = int(value.split()[0])/1024
    sample = {'utc': stamp(), 'cpu_max_c': temperature(), 'memory_mib': mem}
    if include_gpu:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, timeout=10, check=True)
        name, total, used, utilization, temp, power = [x.strip() for x in result.stdout.strip().split(',')]
        sample['gpu'] = dict(name=name, total_mib=float(total), used_mib=float(used),
            utilization_percent=float(utilization), temperature_c=float(temp), power_w=float(power))
    return sample


def stop_children(jobs):
    for job in jobs:
        process = job['process']
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        job['log'].close()


def run_profile(output, name, steps):
    allocation = PROFILES[name]
    assigned = sum(allocation.values(), [])
    if len(set(assigned)) != len(assigned) or 15 in assigned or not set(assigned) <= os.sched_getaffinity(0):
        raise ValueError('Invalid or unavailable CPU allocation')
    output.mkdir(parents=True, exist_ok=False)
    method_allocation = METHOD_ALLOCATION.get(name, {'cpu': CPU_METHODS, 'cuda:0': GPU_METHODS})
    queues = {device: list(methods) for device, methods in method_allocation.items()}
    slots = [dict(device=device, core=core, job=None)
             for device, cores in allocation.items() for core in cores]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1')
    jobs, samples, completed = [], [], {}
    started = time.monotonic()
    write(output/'launch.json', dict(utc=stamp(), profile=name, allocation=allocation,
        cpu_methods=method_allocation['cpu'], gpu_methods=method_allocation['cuda:0'], steps_per_method=steps,
        seed=1, formal_before=read_json(FORMAL_STATUS), deadline_seconds=600,
        cpu_temperature_stop_c=85, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        while len(completed) < len(CPU_METHODS)+len(GPU_METHODS):
            for slot in slots:
                job = slot['job']
                if job is not None and job['process'].poll() is not None:
                    status = read_json(output/job['method']/'status.json')
                    if job['process'].returncode != 0 or status.get('state') != 'complete':
                        raise RuntimeError(f"{job['method']} failed: {status}")
                    if status['completed_steps'] != steps:
                        raise RuntimeError('Incomplete training budget')
                    completed[job['method']] = dict(device=slot['device'], cpu_core=slot['core'],
                        start_seconds=job['start_seconds'], end_seconds=time.monotonic()-started,
                        status=status)
                    slot['job'] = None
                    print(name, job['method'], 'complete', round(status['steady_steps_per_second']), 'steps/s', flush=True)
                if slot['job'] is None and queues[slot['device']]:
                    method = queues[slot['device']].pop(0)
                    log = (output/f'{method}.log').open('x')
                    command = ['taskset', '-c', str(slot['core']), sys.executable, str(ROOT/'tensor_train.py'),
                        '--method', method, '--device', slot['device'], '--seed', '1',
                        '--steps', str(steps), '--output', str(output/method)]
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env=env)
                    job = dict(method=method, process=process, log=log, start_seconds=time.monotonic()-started)
                    jobs.append(job)
                    slot['job'] = job
            sample = resources(include_gpu=len(samples) % 5 == 0)
            sample['elapsed_seconds'] = time.monotonic()-started
            sample['running'] = {j['method']: read_json(output/j['method']/'status.json').get('completed_steps', 0)
                for j in jobs if j['process'].poll() is None}
            samples.append(sample)
            write(output/'status.json', dict(state='running', updated_utc=stamp(), profile=name,
                completed=list(completed), running=sample['running']))
            if sample['cpu_max_c'] is not None and sample['cpu_max_c'] >= 85:
                raise RuntimeError('85C prototype guard; stopping only allocation probe children')
            if time.monotonic()-started > 600:
                raise TimeoutError('Bounded allocation probe exceeded 600 seconds')
            if len(completed) < len(CPU_METHODS)+len(GPU_METHODS):
                time.sleep(1)
        wall = time.monotonic()-started
        result = dict(state='complete', updated_utc=stamp(), profile=name, allocation=allocation,
            steps_per_method=steps, total_steps=7*steps, wall_seconds=wall,
            effective_steps_per_second=7*steps/wall,
            linear_70m_hours=wall*(10_000_000/steps)/3600,
            cpu_temperature_peak_c=max((s['cpu_max_c'] for s in samples if s['cpu_max_c'] is not None), default=None),
            ram_available_min_mib=min(s['memory_mib']['MemAvailable'] for s in samples),
            gpu_memory_peak_mib=max(s['gpu']['used_mib'] for s in samples if 'gpu' in s),
            gpu_utilization_mean_percent=sum(s['gpu']['utilization_percent'] for s in samples if 'gpu' in s)/sum('gpu' in s for s in samples),
            methods=completed, formal_after=read_json(FORMAL_STATUS),
            limitation='Short engineering probe with original formal jobs active; linear time is not a long-run guarantee.')
        write(output/'status.json', result)
        return result
    except BaseException as exc:
        write(output/'status.json', dict(state='failed', updated_utc=stamp(), profile=name,
            error=repr(exc), completed=completed, formal_after=read_json(FORMAL_STATUS)))
        raise
    finally:
        stop_children(jobs)
        write(output/'resource_samples.json', samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=80_000)
    parser.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=['cpu3_gpu1', 'cpu3_gpu2'])
    args = parser.parse_args()
    if args.steps < 8_000 or args.steps % 4_000:
        parser.error('steps must be at least 8,000 and a multiple of 4,000')
    verify_reference()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for profile in args.profiles:
        result = run_profile(args.output/profile, profile, args.steps)
        results[profile] = result
        write(args.output/'summary.json', results)
        print(json.dumps({key: result[key] for key in ('profile', 'wall_seconds', 'effective_steps_per_second',
            'linear_70m_hours', 'cpu_temperature_peak_c', 'gpu_memory_peak_mib')}), flush=True)
