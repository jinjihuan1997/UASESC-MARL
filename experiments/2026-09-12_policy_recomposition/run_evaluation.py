"""Bounded inference-only process launcher. No training or baseline fitting."""
import os,sys,json,subprocess,time,concurrent.futures
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def read(p):return json.loads(p.read_text())
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def job(seed,c):
 cmd=[sys.executable,str(HERE/'evaluate.py'),'--seed',str(seed),'--controller',c]
 log=HERE/f'logs/evaluation_{seed}_{c}.log'
 with log.open('a') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  write(HERE/f'logs/process_{seed}_{c}.json',dict(pid=p.pid,command=cmd,started=time.time(),state='running'))
  rc=p.wait()
 write(HERE/f'logs/process_{seed}_{c}.json',dict(pid=p.pid,command=cmd,returncode=rc,state='complete' if rc==0 else 'failed'))
 return dict(parent=seed,controller=c,returncode=rc)
def main():
 m=read(HERE/'manifest.json');assert read(HERE/'preflight.json')['state']=='PASS'
 assert read(HERE/'resource_settings.json')['workers']==3
 jobs=[(s,c) for c in m['new_controllers'] for s in m['parents']]
 write(HERE/'status.json',dict(state='evaluating',new_training_steps=0,new_optimizer_updates=0,target_new_episodes=2340,completed_jobs=[]))
 done=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(job,*j) for j in jobs]
  for f in concurrent.futures.as_completed(futures):
   done.append(f.result());write(HERE/'status.json',dict(state='evaluating',new_training_steps=0,new_optimizer_updates=0,target_new_episodes=2340,completed_jobs=done));print(done[-1],flush=True)
 failed=[d for d in done if d['returncode']]
 write(HERE/'status.json',dict(state='evaluation_failed' if failed else 'evaluation_complete_aggregation_pending',completed_jobs=done,new_training_steps=0,new_optimizer_updates=0,target_new_episodes=2340))
 if failed:sys.exit(1)
if __name__=='__main__':main()
