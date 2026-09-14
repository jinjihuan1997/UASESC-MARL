"""At most two owned workers; fixed queues and temperature-only pause/resume."""
from repair_support import *
import subprocess
import signal
import time

def resource():
    temperatures=[]
    for hw in Path('/sys/class/hwmon').glob('*'):
        if (hw/'name').exists() and (hw/'name').read_text().strip()=='coretemp':
            for f in hw.glob('temp*_input'):
                try:temperatures.append(float(f.read_text())/1000)
                except (OSError,ValueError):pass
    out=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
    gpu=None
    if out.returncode==0:
        try:gpu=float(out.stdout.split(',')[0])
        except ValueError:pass
    return dict(utc=stamp(),load=os.getloadavg(),cpu_max_c=max(temperatures) if temperatures else None,
                gpu_c=gpu,gpu_query=out.stdout.strip())

def jobs():
    m=manifest();training=[];evaluation=[]
    for seed in m['parents']:
        for arm in m['arms']:
            training.append(dict(kind='training',name=f'seed_{seed}/{arm}',
                command=[sys.executable,str(HERE/'repair_training.py'),'--seed',str(seed),'--arm',arm,'--device','cpu','--resume'],
                status=str(HERE/f'jobs/seed_{seed}/{arm}/status.json')))
            for steps in m['milestones']:
                evaluation.append(dict(kind='evaluation',name=f'seed_{seed}/{arm}_at_{steps}',
                    command=[sys.executable,str(HERE/'repair_evaluation.py'),'--seed',str(seed),'--method',arm,'--steps',str(steps)],
                    status=str(HERE/f'evaluation/seed_{seed}/{arm}_at_{steps}/status.json')))
        for method in ['original_rl','quality_rule_hybrid']:
            evaluation.append(dict(kind='evaluation',name=f'seed_{seed}/{method}',
                command=[sys.executable,str(HERE/'repair_evaluation.py'),'--seed',str(seed),'--method',method],
                status=str(HERE/f'evaluation/seed_{seed}/{method}/status.json')))
    for method in m['rules']:
        evaluation.append(dict(kind='evaluation',name=f'rules/{method}',
            command=[sys.executable,str(HERE/'repair_evaluation.py'),'--method',method],
            status=str(HERE/f'evaluation/rules/{method}/status.json')))
    return training,evaluation

def main():
    guard();verify_inputs();assert read(HERE/'preflight.json')['state']=='PASS'
    logdir=HERE/'logs';logdir.mkdir(exist_ok=True)
    queues=jobs();write(HERE/'run_commands.json',dict(training=queues[0],evaluation=queues[1],max_workers=2,environment={k:os.environ[k] for k in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    stopped=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda n,f:stopped.append(n))
    for stage,queue in zip(['training','evaluation'],queues):
        pending=[j for j in queue if not (Path(j['status']).exists() and read(j['status'])['state']=='complete')]
        active=[];paused=False
        while pending or active:
            if stopped:
                for p,j,log in active:
                    if paused:os.kill(p.pid,signal.SIGCONT)
                    p.terminate()
                write(HERE/'status.json',dict(state='paused_by_signal',stage=stage,signal=stopped[0]))
                return 75
            usage=resource();append(logdir/'resources.jsonl',usage)
            hot=(usage['cpu_max_c'] is not None and usage['cpu_max_c']>=90) or (usage['gpu_c'] is not None and usage['gpu_c']>=83)
            cool=(usage['cpu_max_c'] is None or usage['cpu_max_c']<=80) and (usage['gpu_c'] is None or usage['gpu_c']<=75)
            if active and hot and not paused:
                for p,j,log in active:
                    if p.poll() is None:os.kill(p.pid,signal.SIGSTOP)
                paused=True;append(logdir/'temperature_events.jsonl',dict(event='pause_owned_workers',**usage))
            if paused and cool:
                for p,j,log in active:
                    if p.poll() is None:os.kill(p.pid,signal.SIGCONT)
                paused=False;append(logdir/'temperature_events.jsonl',dict(event='resume_owned_workers',**usage))
            retained=[]
            for p,j,log in active:
                rc=p.poll()
                if rc is None:retained.append((p,j,log));continue
                log.close();append(logdir/'processes.jsonl',dict(event='exit',pid=p.pid,returncode=rc,job=j,utc=stamp()))
                if rc!=0:
                    # Stop only owned sibling jobs cleanly. Do not hide a failed run.
                    for other,k,l in active:
                        if other.poll() is None:
                            if paused:os.kill(other.pid,signal.SIGCONT)
                            other.terminate()
                    write(HERE/'status.json',dict(state='blocked_execution_error',job=j,returncode=rc,stage=stage))
                    return rc
            active=retained
            while pending and len(active)<2 and not paused and not hot:
                j=pending.pop(0);name=j['kind']+'_'+j['name'].replace('/','_')
                log=(logdir/f'{name}.log').open('a')
                p=subprocess.Popen(j['command'],stdout=log,stderr=subprocess.STDOUT,cwd=HERE,env=os.environ.copy())
                active.append((p,j,log));append(logdir/'processes.jsonl',dict(event='start',pid=p.pid,job=j,utc=stamp()))
            progress=[]
            for j in queues[0]:
                if Path(j['status']).exists():progress.append(read(j['status']).get('completed_steps',0))
            write(HERE/'status.json',dict(state='temperature_paused' if paused else stage,completed_training_steps=sum(progress),target_training_steps=6000000,
                active=[dict(pid=p.pid,name=j['name']) for p,j,l in active],pending_stage_jobs=len(pending),resource=usage,updated_utc=stamp(),final_test_used=False))
            time.sleep(5)
    write(HERE/'status.json',dict(state='execution_complete_pending_aggregation',completed_training_steps=6000000,final_test_used=False,updated_utc=stamp()))
    return 0

if __name__=='__main__':raise SystemExit(main())
