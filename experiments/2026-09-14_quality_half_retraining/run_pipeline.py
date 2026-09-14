"""Run only the sealed three jobs, then the fixed evaluations and report."""
from train_support import *
import signal,fcntl

def main():
 lock=(HERE/'.pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';plan=read(HERE/'resource_plan.json');active=[];stopping=[]
 def stop(sig,frame):
  stopping.append(sig)
  for p in active:
   if p.poll() is None:p.send_signal(signal.SIGTERM)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
 def run_stage(stage,script):
  active.clear();logs=[];commands=[]
  for n,seed in enumerate(m['seeds']):
   command=[sys.executable,str(HERE/script),'--seed',str(seed),'--cpu',str(plan['cpu_ids'][n])]
   if stage=='training':
    job=HERE/f'jobs/seed_{seed}/status.json'
    if job.exists():
     if read(job)['state']=='complete':continue
     raise RuntimeError('Existing incomplete training requires explicit checked resume; automatic budget expansion prohibited')
   stream=(HERE/f'{stage}_{seed}.log').open('a');logs.append(stream);p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));active.append(p);commands.append(dict(seed=seed,pid=p.pid,command=command))
  append(HERE/'execution_commands.jsonl',dict(stage=stage,utc=stamp(),jobs=commands))
  while any(p.poll() is None for p in active):
   progress={str(s):read(HERE/f'jobs/seed_{s}/status.json') if (HERE/f'jobs/seed_{s}/status.json').exists() else {'state':'initializing'} for s in m['seeds']}
   write(HERE/'status.json',dict(state=stage,pid=os.getpid(),updated_utc=stamp(),jobs=progress,target_training_steps=3000000,completed_training_steps=sum(x.get('completed_steps',0) for x in progress.values()),active_processes=commands))
   failed=[p.returncode for p in active if p.poll() not in (None,0)]
   if failed:
    stop(signal.SIGTERM,None);raise RuntimeError(f'{stage} failed: {failed}; own workers asked to checkpoint and stop')
   time.sleep(3)
  for s in logs:s.close()
  if stopping:raise RuntimeError('User or process supervisor requested stop')
  if any(p.returncode!=0 for p in active):raise RuntimeError(stage+' failed')
 run_stage('training','train.py')
 verify();run_stage('evaluation','evaluate.py')
 write(HERE/'status.json',dict(state='aggregating',pid=os.getpid(),completed_training_steps=3000000,updated_utc=stamp()))
 with (HERE/'aggregate.log').open('a') as log:
  subprocess.run([sys.executable,str(HERE/'finalize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(HERE/'pipeline_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=stamp()));write(HERE/'status.json',dict(state='stopped_with_error',error=repr(e),utc=stamp()));raise
