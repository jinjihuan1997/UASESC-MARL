"""Two CPU jobs; smoke first, then fresh fixed-budget pilot, then evaluation."""
import subprocess
import time
from probe_common import *

METHODS=['IC_HAPPO','HAPPO_hidden_instruction']


def run_pair(stage,manifest):
    cores=sorted(os.sched_getaffinity(0))
    assert len(cores)>=2
    selected=[cores[0],cores[2] if len(cores)>2 else cores[1]]
    children=[];streams=[]
    try:
        for method,core in zip(METHODS,selected):
            folder=HERE/stage/method;cfg=HERE/stage/'configs'/f'{method}.json'
            c=config(method=='HAPPO_hidden_instruction',8000 if stage=='smoke' else 1_000_000)
            write(cfg,c)
            log=HERE/stage/f'{method}.log';log.parent.mkdir(parents=True,exist_ok=True)
            stream=log.open('x');streams.append(stream)
            cmd=['taskset','-c',str(core),sys.executable,'-u',str(PARENT/'source/formal_train.py'),
                '--config',str(cfg),'--output',str(folder),'--run-manifest',str(manifest),
                '--device','cpu','--checkpoint-every','25']
            children.append((method,subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)))
        while True:
            active=[]
            for method,child in children:
                code=child.poll()
                if code is not None and code!=0: raise RuntimeError(f'{stage}/{method} exited {code}')
                path=HERE/stage/method/'status.json'
                item=read(path) if path.exists() else {'state':'starting'}
                active.append(dict(method=method,pid=child.pid,**{k:item[k] for k in ['state','completed_steps','target_steps','last_timing'] if k in item}))
            write(HERE/'PILOT_STATUS.json',dict(state=stage,updated_utc=stamp(),jobs=active))
            if all(child.poll() is not None for _,child in children): break
            time.sleep(1)
        for method in METHODS:
            status=read(HERE/stage/method/'status.json');assert status['state']=='complete'
            for f,h in status['checkpoint_hashes'].items(): assert digest(HERE/stage/method/f)==h
        print(stage+' complete',flush=True)
    finally:
        for _,child in children:
            if child.poll() is None: child.terminate()
        for _,child in children:
            if child.poll() is None: child.wait(timeout=60)
        for stream in streams: stream.close()


if __name__=='__main__':
    gate=read(HERE/'gate_results.json');assert gate['state']=='PASS' and all(gate['conditions'].values())
    verify_parent()
    assert not (HERE/'pilot_manifest.json').exists(),'Preserve prior pilot'
    manifest=HERE/'pilot_manifest.json'
    write(manifest,dict(purpose='single_seed_1m_preference_only_pilot',created_utc=stamp(),
        protocol_sha256=digest(HERE/'PROTOCOL.md'),gate_sha256=digest(HERE/'gate_results.json'),
        source_hashes={p.name:digest(p) for p in HERE.glob('*.py')},
        parent_manifest_sha256=digest(PARENT/'manifest.json'),seed=85,steps_per_method=1_000_000,
        methods=METHODS,device='cpu',evaluation_seeds=list(range(20261601,20261621)),
        scenarios=read(PARENT/'manifest.json')['scenarios']))
    try:
        run_pair('smoke',manifest)
        run_pair('pilot',manifest)
        subprocess.run([sys.executable,'-u',str(HERE/'pilot_evaluate.py')],check=True)
        verify_parent()
        write(HERE/'PILOT_STATUS.json',dict(state='complete',updated_utc=stamp(),training_steps=2_000_000,
            evaluation_report=str(HERE/'PILOT_REPORT.md')))
    except BaseException as exc:
        write(HERE/'PILOT_STATUS.json',dict(state='failed',updated_utc=stamp(),error=repr(exc)))
        raise
