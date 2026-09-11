"""Detached, checkpoint-aware nine-job queue with bounded thermal pauses."""
import argparse
import fcntl
import signal
import subprocess
import time
from collections import deque
from helpers import *
from runtime_control import ThermalControl,process_info,recorded_process_alive,gpu_temperature,temperature


def verify():
    m=read(HERE/'manifest.json')
    for path,h in m['input_hashes'].items():assert digest(HERE/path)==h,path
    return m


def live_record(path):
    if not path.exists():return None
    old=read(path);current=process_info(old['pid'])
    return current if current and current['state']!='Z' and current['uid']==old['uid'] and current['start_ticks']==old['start_ticks'] else None


def stop_workers(active):
    for job in active.values():
        if job['p'].poll() is None:os.killpg(job['p'].pid,signal.SIGTERM)
    deadline=time.monotonic()+60
    while any(j['p'].poll() is None for j in active.values()) and time.monotonic()<deadline:time.sleep(.2)
    for job in active.values():
        if job['p'].poll() is None:
            os.killpg(job['p'].pid,signal.SIGKILL)
            raise RuntimeError('Worker did not stop at its checkpoint boundary within 60 seconds')


def training_progress(m):
    rows={};remaining={core:0. for core in m['resources']['cores']}
    fallback=m['resources']['seconds_per_update_estimate']
    for item in m['jobs']:
        path=HERE/item['output']/'status.json';r=read(path) if path.exists() else {}
        steps=r.get('completed_steps',0)
        rows[item['id']]=dict(state=r.get('state','queued'),steps=steps,target=m['steps_per_method'],device=item['device'],core=item['core'],recoverable_update=r.get('recoverable_update',0))
        seconds=r.get('last_timing',{}).get('total_seconds',fallback[item['core']])
        remaining[item['core']]+=max(0,m['steps_per_method']-steps)/4000*seconds
    return rows,max(remaining.values())


def run_phase(m,phase,stopping,resume):
    items=m['jobs'] if phase=='training' else [dict(id=i,output='evaluation/'+i) for i in m['evaluation_items']]
    pending=deque();completed=[];active={};failures={};thermal=ThermalControl()
    for item in items:
        folder=HERE/item['output'];path=folder/'status.json';status=read(path) if path.exists() else {}
        if recorded_process_alive(folder):raise RuntimeError('A prior worker is still alive: '+item['id'])
        if status.get('state')=='complete':
            if phase=='training':
                assert status['completed_steps']==m['steps_per_method']
                for f,h in status['checkpoint_hashes'].items():assert digest(folder/f)==h
            else:assert digest(folder/'summary.json')==status['summary_sha256']
            completed.append(item['id']);continue
        if status and not resume:raise RuntimeError('Existing unfinished state requires --resume: '+item['id'])
        pending.append(dict(item,resume=bool(status)))
    try:
        while pending or active:
            now=time.monotonic();cpu,gpu=temperature(),gpu_temperature();event=thermal.tick(cpu,gpu,now)
            if event=='pause':
                stop_workers(active)
                for slot,job in list(active.items()):
                    s=read(HERE/job['item']['output']/'status.json')
                    if job['p'].returncode==0 and s['state']=='complete':completed.append(job['item']['id'])
                    elif job['p'].returncode==75 and s['state']=='paused':pending.appendleft(dict(job['item'],resume=True))
                    else:failures[job['item']['id']]=dict(code=job['p'].returncode,status=s)
                    job['log'].close();del active[slot]
                thermal.paused_since=time.monotonic()
            if event:
                with (HERE/'thermal_events.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=stamp(),phase=phase,event=event,cpu_c=cpu,gpu_c=gpu))+'\n')
            for slot,job in list(active.items()):
                status_path=HERE/job['item']['output']/'status.json'
                s=read(status_path) if status_path.exists() else {}
                progress=s.get('completed_steps' if phase=='training' else 'completed_episodes',0)
                if progress!=job['progress']:job['progress']=progress;job['last_progress']=now
                code=job['p'].poll()
                if code is not None:
                    if code==0 and s.get('state')=='complete':completed.append(job['item']['id'])
                    else:failures[job['item']['id']]=dict(code=code,status=s)
                    job['log'].close();del active[slot]
                elif now-job['last_progress']>600:failures[job['item']['id']]=dict(error='No progress for 600 seconds')
            if stopping or failures:break
            for slot in m['resources']['cores']:
                if thermal.cooling or slot in active or not pending:continue
                # Training items have fixed cores/devices. Evaluation uses free slots.
                if phase=='training':
                    match=next((v for v in pending if v['core']==slot),None)
                    if match is None:continue
                    item=match;pending.remove(item)
                    cmd=[sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(HERE/item['config']),
                        '--output',str(HERE/item['output']),'--run-manifest',str(HERE/'manifest.json'),'--device',item['device'],'--checkpoint-every','25']
                    if item['resume']:cmd.append('--resume')
                else:
                    item=pending.popleft();cmd=[sys.executable,'-u',str(HERE/'evaluation.py'),'--item',item['id']]
                log_path=HERE/'logs'/f'{phase}_{item["id"].replace("/","_")}.log';log_path.parent.mkdir(parents=True,exist_ok=True)
                log=log_path.open('a')
                p=subprocess.Popen(['taskset','-c',str(slot),*cmd],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_info(p.pid);assert identity is not None
                write(HERE/item['output']/'process.json',identity)
                active[slot]=dict(item=item,p=p,log=log,progress=0,last_progress=now)
            rows,eta=training_progress(m)
            write(HERE/'status.json',dict(state='cooling' if thermal.cooling else phase,updated_utc=stamp(),supervisor_pid=os.getpid(),
                completed_training_steps=sum(r['steps'] for r in rows.values()),total_training_steps=m['total_training_steps'],
                completed_training_jobs=sum(r['state']=='complete' for r in rows.values()),total_training_jobs=len(m['jobs']),
                training_progress=rows,phase_completed_jobs=len(completed),phase_total_jobs=len(items),
                active={j['item']['id']:dict(pid=j['p'].pid,core=slot,device=j['item'].get('device','cpu')) for slot,j in active.items()},
                cpu_max_c=cpu,gpu_max_c=gpu,estimated_training_seconds_remaining=eta,failed_jobs=failures))
            with (HERE/'resources.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=stamp(),cpu_c=cpu,gpu_c=gpu,phase=phase,active_jobs=len(active),cooling=thermal.cooling))+'\n')
            if active or pending:time.sleep(1)
        return failures
    finally:
        stop_workers(active)
        for job in active.values():job['log'].close()


def run(resume):
    m=verify();lock=(HERE/'.queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda signum,frame:stopping.append(signum))
    try:
        for phase in ['training','evaluating']:
            failures=run_phase(m,phase,stopping,resume)
            if stopping or failures:
                rows,_=training_progress(m)
                write(HERE/'status.json',dict(state='paused' if stopping else 'attention_required',phase=phase,updated_utc=stamp(),failed_jobs=failures,training_progress=rows))
                return 75 if stopping else 1
        subprocess.run([sys.executable,'-u',str(HERE/'aggregate.py')],check=True)
        verify();audit=read(HERE/'report/audit.json');assert audit['state']=='PASS' and audit['episodes']==m['evaluation_episodes']
        write(HERE/'status.json',dict(state='complete',updated_utc=stamp(),completed_training_steps=m['total_training_steps'],completed_training_jobs=len(m['jobs']),
            evaluation_episodes=m['evaluation_episodes'],report=str(HERE/'REPORT.md'),science_status='results_ready_for_interpretation'))
        return 0
    except BaseException as exc:
        write(HERE/'status.json',dict(state='attention_required',updated_utc=stamp(),error=repr(exc)))
        raise
    finally:lock.close()


def start(resume):
    m=verify()
    with (HERE/'.launcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if live_record(HERE/'current_execution.json'):raise RuntimeError('Queue already running')
        if any(recorded_process_alive(HERE/j['output']) for j in m['jobs']):raise RuntimeError('Prior worker still running')
        if not resume and any((HERE/j['output']/'status.json').exists() for j in m['jobs']):raise RuntimeError('Use --resume for existing jobs')
        (HERE/'logs').mkdir(exist_ok=True)
        cmd=[sys.executable,'-u',str(HERE/'supervisor.py'),'run']+(['--resume'] if resume else [])
        with (HERE/'logs/supervisor.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        identity=process_info(p.pid);assert identity is not None
        write(HERE/'current_execution.json',dict(identity,utc=stamp(),manifest_sha256=digest(HERE/'manifest.json')))
        print(json.dumps(dict(state='started',pid=p.pid,root=str(HERE)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['start','run','status','pause']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.command=='start':start(args.resume)
    elif args.command=='run':raise SystemExit(run(args.resume))
    elif args.command=='pause':
        live=live_record(HERE/'current_execution.json')
        if live:os.kill(live['pid'],signal.SIGTERM);print('Requested checkpoint-boundary pause')
        else:print('No live queue')
    else:print(json.dumps(read(HERE/'status.json'),ensure_ascii=False,indent=2))
