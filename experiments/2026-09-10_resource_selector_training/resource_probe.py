"""Choose training layout by measured makespan, not GPU utilization alone."""
import argparse,concurrent.futures,subprocess,time
from helpers import *
from selector_train import ContinuationTrainer
from runtime_control import temperature,gpu_temperature

def worker(arm,seed,device,output):
    t=ContinuationTrainer(config(arm,seed),device);t.initialize_transfer({'test':'resource_probe'})
    durations=[]
    for update in range(1,5):
        t.set_training_update(update);t.synchronize();start=time.monotonic()
        t.collect();t.update();t.buffer.after_update();t.synchronize()
        durations.append(time.monotonic()-start)
    write(Path(output),dict(arm=arm,seed=seed,device=device,seconds_per_update=float(np.mean(durations[1:])),samples=durations))

def main():
    folder=HERE/'resource_probe';folder.mkdir(exist_ok=False);results={}
    for layout in ['six_cpu','four_cpu_two_gpu']:
        pairs=[(seed,arm) for seed in SEEDS for arm in METHODS];processes=[]
        for core,(seed,arm) in enumerate(pairs):
            device='cuda:0' if layout=='four_cpu_two_gpu' and seed==966 else 'cpu'
            out=folder/f'{layout}_{seed}_{arm}.json';log=out.with_suffix('.log').open('w')
            cmd=['taskset','-c',str(core),sys.executable,'-u',__file__,'--arm',arm,'--seed',str(seed),'--device',device,'--output',str(out)]
            processes.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log,out))
        try:
            while any(p.poll() is None for p,_,_ in processes):
                cpu,gpu=temperature(),gpu_temperature()
                if cpu is None or gpu is None or cpu>=85 or gpu>=83:raise RuntimeError('Probe temperature guard')
                time.sleep(1)
            rows=[]
            for p,log,out in processes:
                log.close();assert p.returncode==0,out;rows.append(read(out))
            results[layout]=dict(jobs=rows,seconds_per_update=max(r['seconds_per_update'] for r in rows),
                                estimated_training_seconds=500*max(r['seconds_per_update'] for r in rows))
            print(layout,results[layout]['estimated_training_seconds'],flush=True)
        finally:
            for p,log,_ in processes:
                if p.poll() is None:p.terminate()
                log.close()
    selected=min(results,key=lambda k:results[k]['estimated_training_seconds'])
    write(HERE/'resource_probe.json',dict(state='PASS',layouts=results,selected=selected,
        rule='Minimum measured time to finish all six equal-budget jobs; same device within each seed pair; exclude evaluation and thermal pauses.'))
    print('Selected',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm');p.add_argument('--seed',type=int);p.add_argument('--device');p.add_argument('--output');a=p.parse_args()
    if a.arm:worker(a.arm,a.seed,a.device,a.output)
    else:main()
