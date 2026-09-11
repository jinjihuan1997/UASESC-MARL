"""Six training workers plus concurrent milestone evaluation, with exact resume."""
import argparse,fcntl,signal,subprocess,time
from helpers import *
from runtime_control import ThermalControl,process_info,recorded_process_alive,gpu_temperature,temperature

def live_record(path):
    if not path.exists(): return None
    old=read(path);now=process_info(old['pid'])
    return now if now and now['state']!='Z' and now['uid']==old['uid'] and now['start_ticks']==old['start_ticks'] else None

def stop_workers(active):
    for w in active.values():
        if w['p'].poll() is None: os.killpg(w['p'].pid,signal.SIGTERM)
    end=time.monotonic()+60
    while any(w['p'].poll() is None for w in active.values()) and time.monotonic()<end: time.sleep(.2)
    for w in active.values():
        if w['p'].poll() is None:
            os.killpg(w['p'].pid,signal.SIGKILL)
            raise RuntimeError('Owned worker failed to checkpoint within 60 seconds')

def ready(item,root=HERE):
    return item['kind']=='training' or not item.get('model_dir') or (root/item['model_dir']/'status.json').exists()

def progress(m):
    rows={};remaining=[]
    for j in m['jobs']:
        p=HERE/j['output']/'status.json';s=read(p) if p.exists() else {};steps=s.get('completed_steps',0)
        rows[j['id']]=dict(state=s.get('state','queued'),steps=steps,target=m['steps_per_method'],core=j['core'],device=j['device'],stage=s.get('stage'))
        seconds=s.get('last_timing',{}).get('total_seconds',m['resources']['seconds_per_update_estimate'])
        remaining.append((m['steps_per_method']-steps)/m['batch']*seconds)
    return rows,max(remaining,default=0)

def run(resume):
    m=verify();lock=(HERE/'.queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=[]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda n,f:stopped.append(n))
    pending=[];complete=[];active={};failures={};thermal=ThermalControl()
    try:
        for item in m['jobs']+m['evaluation_jobs']:
            folder=HERE/item['output'];path=folder/'status.json';s=read(path) if path.exists() else {}
            assert not recorded_process_alive(folder),'Prior worker still alive'
            if s.get('state')=='complete':
                if item['kind']=='training':
                    assert s['completed_steps']==m['steps_per_method']
                    for f,h in s['checkpoint_hashes'].items(): assert digest(folder/f)==h
                else: assert digest(folder/'summary.json')==s['summary_sha256']
                complete.append((item['kind'],item['id']));continue
            if s and not resume: raise RuntimeError('Existing work requires --resume: '+item['id'])
            pending.append(dict(item,resume=bool(s)))
        while pending or active:
            now=time.monotonic();cpu,gpu=temperature(),gpu_temperature();event=thermal.tick(cpu,gpu,now)
            if event=='pause': stop_workers(active);thermal.paused_since=time.monotonic()
            if event:
                with (HERE/'thermal_events.jsonl').open('a') as f: f.write(json.dumps(dict(utc=stamp(),event=event,cpu_c=cpu,gpu_c=gpu))+'\n')
            for core,w in list(active.items()):
                item=w['item'];path=HERE/item['output']/'status.json';s=read(path) if path.exists() else {}
                number=s.get('completed_steps' if item['kind']=='training' else 'completed_episodes',0)
                if number!=w['progress']: w['progress']=number;w['last_progress']=now
                code=w['p'].poll()
                if code is not None:
                    if code==0 and s.get('state')=='complete': complete.append((item['kind'],item['id']))
                    elif code==75 and s.get('state')=='paused' and (thermal.cooling or stopped): pending.append(dict(item,resume=True))
                    else: failures[item['id']]=dict(code=code,status=s)
                    w['log'].close();del active[core]
                elif now-w['last_progress']>600: failures[item['id']]=dict(error='No progress for 600 seconds')
            if stopped or failures: break
            if not thermal.cooling:
                for core in m['resources']['cores']:
                    if core in active: continue
                    item=next((j for j in pending if j['kind']=='training' and j['core']==core),None)
                    if item is None: item=next((j for j in pending if j['kind']=='evaluating' and ready(j)),None)
                    if item is None: continue
                    pending.remove(item)
                    if item['kind']=='training':
                        cmd=[sys.executable,'-u',str(HERE/'source/formal_train.py'),'--config',str(HERE/item['config']),
                            '--output',str(HERE/item['output']),'--run-manifest',str(HERE/'manifest.json'),'--device',item['device'],'--checkpoint-every','25']
                        if item['resume']: cmd.append('--resume')
                    else: cmd=[sys.executable,'-u',str(HERE/'evaluation.py'),'--item',item['id']]
                    logpath=HERE/'logs'/f'{item["kind"]}_{item["id"].replace("/","_")}.log';logpath.parent.mkdir(exist_ok=True)
                    log=logpath.open('a');p=subprocess.Popen(['taskset','-c',str(core),*cmd],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    record=process_info(p.pid);assert record is not None;write(HERE/item['output']/'process.json',record)
                    active[core]=dict(item=item,p=p,log=log,progress=0,last_progress=now)
            rows,eta=progress(m)
            write(HERE/'status.json',dict(state='cooling' if thermal.cooling else 'training_and_evaluating',updated_utc=stamp(),supervisor_pid=os.getpid(),
                completed_training_steps=sum(v['steps'] for v in rows.values()),total_training_steps=m['total_training_steps'],
                completed_training_jobs=sum(v['state']=='complete' for v in rows.values()),total_training_jobs=len(m['jobs']),
                completed_evaluation_jobs=sum(kind=='evaluating' for kind,_ in complete),total_evaluation_jobs=len(m['evaluation_jobs']),
                training_progress=rows,active={w['item']['id']:dict(pid=w['p'].pid,core=c,kind=w['item']['kind'],device=w['item'].get('device','cpu')) for c,w in active.items()},
                cpu_max_c=cpu,gpu_max_c=gpu,estimated_training_seconds_remaining=eta,failed_jobs=failures))
            with (HERE/'resources.jsonl').open('a') as f: f.write(json.dumps(dict(utc=stamp(),cpu_c=cpu,gpu_c=gpu,active_jobs=len(active),cooling=thermal.cooling))+'\n')
            if pending and not active and not thermal.cooling:
                raise RuntimeError('Remaining evaluations have no completed milestone snapshots')
            if pending or active: time.sleep(1)
        if stopped or failures:
            write(HERE/'status.json',dict(state='paused' if stopped else 'attention_required',updated_utc=stamp(),failed_jobs=failures,training_progress=progress(m)[0]))
            return 75 if stopped else 1
        subprocess.run([sys.executable,'-u',str(HERE/'aggregate.py')],check=True)
        verify();audit=read(HERE/'report/audit.json');assert audit['state']=='PASS' and audit['episodes']==m['evaluation_episodes']
        write(HERE/'status.json',dict(state='complete',updated_utc=stamp(),completed_training_steps=m['total_training_steps'],completed_training_jobs=len(m['jobs']),
            evaluation_episodes=m['evaluation_episodes'],report=str(HERE/'REPORT.md'),science_status='results_ready_for_interpretation'))
        return 0
    except BaseException as exc:
        write(HERE/'status.json',dict(state='attention_required',updated_utc=stamp(),error=repr(exc)))
        raise
    finally:
        stop_workers(active)
        for w in active.values(): w['log'].close()
        lock.close()

def start(resume):
    m=verify()
    with (HERE/'.launcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not live_record(HERE/'current_execution.json'),'Queue already running'
        assert not any(recorded_process_alive(HERE/j['output']) for j in m['jobs']+m['evaluation_jobs'])
        if not resume and any((HERE/j['output']/'status.json').exists() for j in m['jobs']): raise RuntimeError('Use --resume for existing jobs')
        (HERE/'logs').mkdir(exist_ok=True)
        with (HERE/'logs/supervisor.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(HERE/'supervisor.py'),'run']+(['--resume'] if resume else []),cwd=HERE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_info(p.pid);assert identity is not None
        write(HERE/'current_execution.json',dict(identity,utc=stamp(),manifest_sha256=digest(HERE/'manifest.json')))
        print(json.dumps(dict(state='started',pid=p.pid,root=str(HERE)),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start','run','status','pause']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.command=='start': start(a.resume)
    elif a.command=='run': raise SystemExit(run(a.resume))
    elif a.command=='pause':
        live=live_record(HERE/'current_execution.json')
        if live: os.kill(live['pid'],signal.SIGTERM);print('Checkpoint-boundary pause requested')
        else: print('No live queue')
    else: print(json.dumps(read(HERE/'status.json'),ensure_ascii=False,indent=2))
