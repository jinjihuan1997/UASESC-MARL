"""Bounded two-method CPU/GPU concurrency probe; does not replace formal jobs."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from common import ROOT, write, stamp, verify_reference
from run_suite import temperature


def main(output,steps):
    verify_reference();output.mkdir(parents=True,exist_ok=False)
    assignments=[('IC_HAPPO','cpu',16),('IC_MAPPO','cuda:0',18)]
    jobs=[];samples=[];started=time.monotonic()
    try:
        for method,device,core in assignments:
            log=(output/f'{method}.log').open('x')
            command=['taskset','-c',str(core),sys.executable,str(ROOT/'tensor_train.py'),
                '--method',method,'--device',device,'--steps',str(steps),'--output',str(output/method)]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append(dict(method=method,device=device,cpu_core=core,process=process,log=log))
        write(output/'launch.json',dict(utc=stamp(),steps_per_method=steps,
            assignments=[{k:j[k] for k in ('method','device','cpu_core')}|{'pid':j['process'].pid} for j in jobs]))
        while any(j['process'].poll() is None for j in jobs):
            temp=temperature()
            sample=dict(utc=stamp(),cpu_max_c=temp,progress={})
            for job in jobs:
                p=output/job['method']/'status.json'
                if p.exists(): sample['progress'][job['method']]=json.loads(p.read_text()).get('completed_steps',0)
                if job['process'].poll() not in (None,0): raise RuntimeError(f"{job['method']} failed")
            samples.append(sample)
            if temp is not None and temp>=85: raise RuntimeError('Prototype thermal guard reached 85C; stopping probe only')
            if time.monotonic()-started>600: raise TimeoutError('Mixed probe deadline')
            time.sleep(1)
        results={j['method']:json.loads((output/j['method']/'status.json').read_text()) for j in jobs}
        assert all(r['state']=='complete' and r['completed_steps']==steps for r in results.values())
        write(output/'status.json',dict(state='complete',utc=stamp(),wall_seconds=time.monotonic()-started,
            assignments=[{k:j[k] for k in ('method','device','cpu_core')} for j in jobs],
            steps_per_method=steps,results=results,temperature_peak_c=max(s['cpu_max_c'] for s in samples if s['cpu_max_c'] is not None)))
    except BaseException as exc:
        write(output/'status.json',dict(state='failed',utc=stamp(),error=repr(exc)))
        raise
    finally:
        for job in jobs:
            p=job['process']
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
            job['log'].close()
        write(output/'resource_samples.json',samples)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=80000)
    a=p.parse_args();main(a.output.resolve(),a.steps)
