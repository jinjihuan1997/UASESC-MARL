"""Choose resources by runtime only, using disposable 40k-step jobs."""
import subprocess
import time
from helpers import *
from run_suite import temperature


def gpu_temperature():
    r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return max(float(v) for v in r.stdout.splitlines())


if __name__=='__main__':
    assert torch.cuda.is_available()
    assert not (HERE/'benchmark_results.json').exists()
    output={}
    for layout in ['cpu6','cpu4_gpu2']:
        folder=HERE/'benchmarks'/layout;folder.mkdir(parents=True,exist_ok=False)
        manifest=folder/'manifest.json';write(manifest,dict(purpose='resource_timing_only_no_science_selection',layout=layout))
        processes=[];logs=[];started=time.monotonic();peak_cpu=peak_gpu=0.
        try:
            i=0
            for seed in [85,218,966]:
                for method in ['IC_HAPPO','HAPPO_hidden_instruction']:
                    device='cuda:0' if layout=='cpu4_gpu2' and seed==966 else 'cpu'
                    c=read(HERE/'configs'/f'seed_{seed}'/f'{method}.json')
                    c['algo_args']['train']['num_env_steps']=40000
                    c['algo_args']['device'].update(cuda=device!='cpu',cuda_deterministic=device!='cpu')
                    job=f'seed_{seed}/{method}';cfg=folder/'configs'/f'{job}.json';write(cfg,c)
                    log=(folder/f'{seed}_{method}.log').open('x');logs.append(log)
                    cmd=['taskset','-c',str(i),sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(cfg),'--output',str(folder/'jobs'/job),'--run-manifest',str(manifest),'--device',device,'--checkpoint-every','25']
                    processes.append((job,device,subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)))
                    i+=1
            while any(p.poll() is None for _,_,p in processes):
                cpu,gpu=temperature(),gpu_temperature()
                if cpu is None:raise RuntimeError('CPU temperature sensor unavailable')
                peak_cpu=max(peak_cpu,cpu);peak_gpu=max(peak_gpu,gpu)
                if cpu>=85 or gpu>=83:raise RuntimeError('Benchmark temperature limit reached')
                for name,_,p in processes:
                    if p.poll() is not None and p.returncode!=0:raise RuntimeError(f'{name} failed')
                time.sleep(1)
            jobs={}
            for name,device,p in processes:
                assert p.returncode==0
                status=read(folder/'jobs'/name/'status.json');assert status['state']=='complete'
                times=[json.loads(row)['total_seconds'] for row in (folder/'jobs'/name/'timing.jsonl').read_text().splitlines()][1:]
                jobs[name]=dict(device=device,median_update_seconds=float(np.median(times)))
            output[layout]=dict(jobs=jobs,wall_seconds=time.monotonic()-started,cpu_peak_c=peak_cpu,gpu_peak_c=peak_gpu,
                predicted_training_seconds=max(j['median_update_seconds'] for j in jobs.values())*2500)
            print(layout,json.dumps(output[layout]),flush=True)
        finally:
            for _,_,p in processes:
                if p.poll() is None:p.terminate()
            for _,_,p in processes:
                if p.poll() is None:p.wait(timeout=60)
            for log in logs:log.close()
    selected=min(output,key=lambda k:output[k]['predicted_training_seconds'])
    write(HERE/'benchmark_results.json',dict(state='PASS',selected=selected,layouts=output,selection_basis='lowest_estimated_slowest_job_completion_time',training_rewards_not_read=True))
    print('SELECTED',selected,flush=True)
