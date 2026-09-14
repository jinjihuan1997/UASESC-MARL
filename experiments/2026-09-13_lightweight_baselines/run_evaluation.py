"""Bounded inference-only queue; benchmark runs after evaluation workers exit."""
from light_support import *
import subprocess,concurrent.futures,argparse

def run(script,args,label):
 command=[sys.executable,str(HERE/script),*map(str,args)]
 with (HERE/f'logs/{label}.log').open('a') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});write(HERE/f'logs/process_{label}.json',dict(pid=p.pid,command=command,state='running'));rc=p.wait()
 write(HERE/f'logs/process_{label}.json',dict(pid=p.pid,command=command,state='complete' if rc==0 else 'failed',returncode=rc))
 if rc:raise RuntimeError(f'{label} failed; log retained')

def main(evaluate_only=False):
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';(HERE/'logs').mkdir(exist_ok=True)
 write(HERE/'status.json',dict(state='evaluation_running',new_training_steps=0,new_optimizer_updates=0))
 jobs=[('evaluate.py',['--method',method],method) for method in m['methods']]+[('evaluate.py',['--method',method,'--supplement'],method) for method in ['R_equal_single','R_single']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for f in concurrent.futures.as_completed([pool.submit(run,*job) for job in jobs]):f.result()
 write(HERE/'status.json',dict(state='evaluation_complete_pending_benchmark',primary_episodes=1040,supplement_episodes=520,new_training_steps=0,new_optimizer_updates=0))
 if evaluate_only:return
 run('benchmark_latency.py',[],'latency');run('finalize.py',[],'finalize')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evaluate-only',action='store_true');a=p.parse_args();guard();main(a.evaluate_only)
