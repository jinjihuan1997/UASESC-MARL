"""Compare actual nine-job wall time; idle GPU capacity alone is not a speed metric."""
import subprocess,time
from helpers import *

def main():
    results={}
    for label,devices in [('mixed_6cpu_3gpu',['cpu']*6+['cuda:0']*3),('cpu_9',['cpu']*9)]:
        active=[];start=time.monotonic()
        for core,device in enumerate(devices):
            name=f'{label}_{core}';path=HERE/'preflight'/f'{name}.log';path.parent.mkdir(exist_ok=True)
            log=path.open('w');cmd=[sys.executable,'-u',str(HERE/'benchmark.py'),'--device',device,'--name',name]
            p=subprocess.Popen(['taskset','-c',str(core),*cmd],stdout=log,stderr=subprocess.STDOUT)
            active.append((name,p,log))
        for name,p,log in active:
            assert p.wait()==0,name;log.close()
        rows=[read(HERE/'preflight'/f'benchmark_{name}.json') for name,_,_ in active]
        results[label]=dict(wall_seconds=time.monotonic()-start,devices=devices,
            per_job_seconds_per_update=[r['mean_seconds_per_update'] for r in rows],
            estimated_training_seconds=250*max(r['mean_seconds_per_update'] for r in rows))
        print(label,results[label],flush=True)
    best=min(results,key=lambda k:results[k]['estimated_training_seconds'])
    write(HERE/'resource_probe.json',dict(state='PASS',layouts=results,selected=best,
        rule='Minimum measured makespan, each seed uses the same device for all three arms; estimates exclude evaluation and thermal pauses'))

if __name__=='__main__': main()
