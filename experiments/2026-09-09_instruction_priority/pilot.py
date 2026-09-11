"""Freeze and run two one-seed pilot jobs with a local thermal pause guard."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time
from common import ROOT,configuration,write,stamp,verify_reference
from protocol import SCENARIOS
from run_suite import temperature

METHODS=['IC_HAPPO','HAPPO_hidden_instruction']
def read(path):return json.loads(Path(path).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(run):
    verify_reference()
    gates=['validation_cpu_v2.json','validation_gpu.json','model_checks.json','parameter_sweep.json','resume_checks.json']
    for name in gates:assert read(ROOT/name)['passed'],name
    for method in METHODS:assert read(ROOT/'smoke_gpu'/method/'status.json')['state']=='complete'
    run.mkdir(parents=True,exist_ok=False);source=run/'source';source.mkdir()
    for path in ROOT.glob('*.py'):shutil.copy2(path,source/path.name)
    shutil.copy2(ROOT/'reference_manifest.json',source/'reference_manifest.json')
    shutil.copytree(ROOT/'reference',source/'reference',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for method in METHODS:
        cfg=configuration(method,85,1_000_000)
        cfg['env_args'].update(semantic_profile_path=str(source/'reference/inputs/profile.npz'),
                              semantic_registry_path=str(source/'reference/inputs/mode_registry.json'))
        write(run/'configs'/f'{method}.json',cfg)
    hashes={str(p.relative_to(run)):sha(p) for base in [source,run/'configs'] for p in sorted(base.rglob('*')) if p.is_file()}
    write(run/'manifest.json',dict(purpose='instruction_priority_single_seed_short_budget_pilot',seed=85,steps=1_000_000,
        methods=METHODS,scenarios=SCENARIOS,evaluation_seeds=list(range(20260921,20260926)),
        resources={'devices':['cuda:0','cuda:0'],'host_cores':[16,18],'parallel_jobs':2},
        gates={name:sha(ROOT/name) for name in gates},input_hashes=hashes,created_utc=stamp()))
    write(run/'status.json',dict(state='prepared',utc=stamp()))

def gpu_temperature():
    result=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return max(float(x) for x in result.stdout.splitlines())

def run_jobs(run):
    manifest=read(run/'manifest.json')
    for name,digest in manifest['input_hashes'].items():assert sha(run/name)==digest,name
    stopped=[]
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *x:stopped.append(True))
    active={};hot_since=None;cool_since=None;cool_start=None;peaks=[0.,0.]
    (run/'logs').mkdir(exist_ok=True)
    def launch(method,core):
        out=run/'jobs'/method
        command=[sys.executable,str(run/'source/formal_train.py'),'--config',str(run/'configs'/f'{method}.json'),
            '--output',str(out),'--run-manifest',str(run/'manifest.json'),'--device','cuda:0','--checkpoint-every','25']
        if (out/'status.json').exists():command.append('--resume')
        log=(run/'logs'/f'{method}.log').open('a')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        process=subprocess.Popen(['taskset','-c',str(core),*command],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        active[method]=(process,log)
    def stop_children():
        for process,_ in active.values():
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        for process,log in active.values():
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            log.close()
        active.clear()
    try:
        while True:
            done=[m for m in METHODS if (run/'jobs'/m/'status.json').exists() and read(run/'jobs'/m/'status.json')['state']=='complete']
            if len(done)==2:break
            if stopped:stop_children();write(run/'status.json',dict(state='paused',utc=stamp()));return
            cpu,gpu=temperature(),gpu_temperature()
            if cpu is None:raise RuntimeError('CPU temperature sensor missing')
            peaks=[max(peaks[0],cpu),max(peaks[1],gpu)];now=time.monotonic()
            hot_since=(hot_since or now) if cpu>=85 else None
            if gpu>=83 or (hot_since and now-hot_since>=5):
                stop_children();cool_start=now;cool_since=None
            if cool_start is not None:
                cool_since=(cool_since or now) if cpu<75 and gpu<75 else None
                if now-cool_start>=180 and cool_since and now-cool_since>=60:cool_start=None;hot_since=None
            for method,(process,log) in list(active.items()):
                if process.poll() is not None:
                    log.close();del active[method]
                    if process.returncode!=0:raise RuntimeError(f'{method} failed: {process.returncode}')
            if cool_start is None:
                for method,core in zip(METHODS,[16,18]):
                    if method not in done and method not in active:launch(method,core)
            progress={m:read(run/'jobs'/m/'status.json').get('completed_steps',0)
                      if (run/'jobs'/m/'status.json').exists() else 0 for m in METHODS}
            write(run/'status.json',dict(state='cooling' if cool_start else 'training',utc=stamp(),
                progress=progress,target_per_method=manifest['steps'],cpu_temperature=cpu,gpu_temperature=gpu,
                temperature_peaks=peaks,pids={m:p.pid for m,(p,_) in active.items()}))
            time.sleep(2)
        # Reap completed training workers before evaluating final weights.
        for process,log in active.values():process.wait();log.close()
        active.clear();write(run/'status.json',dict(state='evaluating',utc=stamp(),temperature_peaks=peaks))
        evaluations=[]
        for method,core in zip(METHODS,[16,18]):
            log=(run/'logs'/f'eval_{method}.log').open('a')
            p=subprocess.Popen(['taskset','-c',str(core),sys.executable,str(run/'source/evaluate_pilot.py'),
                '--run',str(run),'--method',method],stdout=log,stderr=subprocess.STDOUT)
            evaluations.append((p,log))
        for p,log in evaluations:
            code=p.wait();log.close()
            if code:raise RuntimeError('Pilot evaluation failed')
        a,b=[read(run/'evaluation'/m/'summary.json')['episodes'] for m in METHODS]
        assert len(a)==len(b)==65
        for x,y in zip(a,b):assert (x['scenario'],x['seed'],x['external_sha256'])==(y['scenario'],y['seed'],y['external_sha256'])
        write(run/'status.json',dict(state='complete',utc=stamp(),steps_per_method=manifest['steps'],
            paired_episodes=65,temperature_peaks=peaks,training_seeds=1))
    except BaseException as exc:
        stop_children();write(run/'status.json',dict(state='failed',error=repr(exc),utc=stamp()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();prepare(a.run.resolve()) if a.mode=='prepare' else run_jobs(a.run.resolve())
