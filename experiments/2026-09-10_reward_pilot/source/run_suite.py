"""Sequential bounded smoke training for all seven migrated SC conditions."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from common import ROOT, configurations, verify_reference, write, stamp


def temperature():
    values=[]
    for hw in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (hw/'name').read_text().strip()!='coretemp': continue
            values.extend(float(p.read_text())/1000 for p in hw.glob('temp*_input'))
        except OSError: continue
    return max(values,default=None)


def run(output,steps,device,cpus):
    verify_reference()
    output.mkdir(parents=True,exist_ok=False)
    cpu_set={int(x) for x in cpus.split(',')}
    if not cpu_set or 15 in cpu_set or not cpu_set <= os.sched_getaffinity(0): raise ValueError('Invalid CPU set')
    os.sched_setaffinity(0,cpu_set)
    methods=list(configurations(steps=steps,device='cuda'))
    statuses={};peaks=[]
    for method in methods:
        target=output/method
        command=[sys.executable,str(ROOT/'tensor_train.py'),'--method',method,'--device',device,'--steps',str(steps),'--output',str(target)]
        with (output/f'{method}.log').open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            try:
                while process.poll() is None:
                    temp=temperature();peaks.append(temp)
                    if temp is not None and temp>=85: raise RuntimeError('Prototype temperature guard: stop this smoke job at 85C')
                    if time.monotonic()-started>600: raise TimeoutError('Prototype job exceeded 600 seconds')
                    time.sleep(1)
                if process.returncode: raise RuntimeError(f'{method} exited {process.returncode}; see log')
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
                write(output/'status.json',dict(state='failed',method=method,completed=list(statuses),updated_utc=stamp()))
                raise
        status=json.loads((target/'status.json').read_text())
        assert status['state']=='complete' and status['completed_steps']==steps
        statuses[method]=status
        write(output/'status.json',dict(state='running',completed=list(statuses),updated_utc=stamp()))
        print(method,status['steady_steps_per_second'],'steps/s',flush=True)
    write(output/'status.json',dict(state='complete',updated_utc=stamp(),steps_per_method=steps,device=device,
        cpu_set=sorted(cpu_set),parallel_methods=1,temperature_peak_c=max(x for x in peaks if x is not None),methods=statuses))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=16000)
    p.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    p.add_argument('--cpu-set',default='18')
    a=p.parse_args();run(a.output.resolve(),a.steps,a.device,a.cpu_set)
