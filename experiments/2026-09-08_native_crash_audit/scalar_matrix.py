"""Bounded, paired CPU/build controls; every GDB transcript is preserved."""
import concurrent.futures
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = '/home/king/miniconda3/envs/harl_sionna/bin/python'
ALTERNATE = '/home/king/miniconda3/pkgs/python-3.10.19-h3c07f61_2_cpython/bin/python3.10'


def run(build, cpu, trial, iterations=2000000):
    name = f'scalar_{build}_cpu{cpu}_trial{trial}'
    output = ROOT / (name + '.json')
    log = ROOT / (name + '_gdb.log')
    if output.exists() or log.exists():
        raise FileExistsError(name)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONMALLOC='debug')
    executable = ORIGINAL if build == 'original' else ALTERNATE
    commands = ['set pagination off', 'set confirm off',
                'set debuginfod enabled off',
                'handle SIGSEGV stop print nopass',
                'handle SIGABRT stop print nopass']
    if build == 'alternate':
        # Only the inferior uses this prefix; GDB embeds system Python 3.12.
        commands += ['set environment PYTHONHOME /home/king/miniconda3/envs/harl_sionna']
    commands += ['run',
                'thread apply all bt 16', 'info registers', 'x/8i $pc',
                'p $_siginfo']
    args = ['gdb', '-q', '-nx', '-batch']
    for command in commands:
        args += ['-ex', command]
    args += ['--args', executable, '-u', str(ROOT / 'scalar_probe.py'),
             '--output', str(output), '--iterations', str(iterations), '--cpu', str(cpu)]
    start = time.monotonic()
    with log.open('x') as stream:
        proc = subprocess.run(args, env=env, stdout=stream, stderr=subprocess.STDOUT,
                              timeout=180)
    transcript = log.read_text()
    result = dict(name=name, cpu=cpu, build=build, gdb_returncode=proc.returncode,
                  elapsed_seconds=time.monotonic()-start, result_exists=output.exists(),
                  signal=next((line for line in transcript.splitlines()
                               if 'received signal' in line), None), log=str(log))
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', default='original:1,original:2,alternate:1')
    parser.add_argument('--iterations', type=int, default=2000000)
    parser.add_argument('--summary', default='scalar_matrix_results.json')
    options = parser.parse_args()
    results = []
    # Pair on distinct cores, changing only the selected build in the next phase.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for pair in options.pairs.split(','):
            build, trial = pair.split(':')
            futures = [pool.submit(run, build, cpu, int(trial), options.iterations)
                       for cpu in [15, 18]]
            results += [future.result() for future in futures]
            (ROOT / options.summary).write_text(json.dumps(results, indent=2)+'\n')
