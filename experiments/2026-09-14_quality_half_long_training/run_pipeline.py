from train_support import *
import fcntl,signal
def main():
 lock=(HERE/'.pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=verify();assert read(HERE/'preflight.json')['state']=='PASS';plan=read(HERE/'resource_plan.json');active=[];stopped=[]
 def stop(sig,frame):
  stopped.append(sig)
  for p in active:
   if p.poll() is None:p.send_signal(signal.SIGTERM)
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
 for phase,script in [('training','train.py'),('evaluation','evaluate.py')]:
  active.clear();commands=[];streams=[]
  for i,seed in enumerate(m['seeds']):
   status=read(HERE/f'jobs/seed_{seed}/status.json')
   if phase=='training' and status['state']=='complete':continue
   command=[sys.executable,str(HERE/script),'--seed',str(seed),'--cpu',str(plan['cpu_ids'][i])];f=(HERE/f'{phase}_{seed}.log').open('a');streams.append(f);p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));active.append(p);commands.append(dict(seed=seed,pid=p.pid,command=command))
  append(HERE/'execution_commands.jsonl',dict(utc=stamp(),phase=phase,jobs=commands))
  while any(p.poll() is None for p in active):
   jobs={str(s):read(HERE/f'jobs/seed_{s}/status.json') for s in m['seeds']};cumulative=sum(r.get('completed_steps',1000000) for r in jobs.values())
   write(HERE/'status.json',dict(state=phase,pid=os.getpid(),updated_utc=stamp(),jobs=jobs,completed_additional_steps=cumulative-3000000,target_additional_steps=27000000,completed_cumulative_steps=cumulative,target_cumulative_steps=30000000,processes=commands))
   failed=[p.returncode for p in active if p.poll() not in (None,0)]
   if failed:
    stop(signal.SIGTERM,None)
    for p in active:p.wait()
    raise RuntimeError(f'{phase} error {failed}; own workers stopped')
   time.sleep(3)
  for f in streams:f.close()
  if stopped or any(p.returncode!=0 for p in active):raise RuntimeError('Stopped or failed')
  verify()
 write(HERE/'status.json',dict(state='aggregating',completed_additional_steps=27000000,pid=os.getpid()))
 with (HERE/'aggregate.log').open('a') as f:subprocess.run([sys.executable,str(HERE/'finalize.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  d=dict(state='stopped_with_error',error=repr(e),traceback=traceback.format_exc(),utc=stamp());write(HERE/'pipeline_failure.json',d);write(HERE/'status.json',d);raise
