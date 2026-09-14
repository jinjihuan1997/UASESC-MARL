from weight_support import *
import concurrent.futures
def job(item):
 method=item['method'];parent=item['parent'];key=f'{parent or "rules"}_{method}';cmd=[sys.executable,str(HERE/'evaluate.py'),'--method',method]
 if parent:cmd+=['--parent',str(parent)]
 start=stamp();t=time.perf_counter()
 with (HERE/f'logs/{key}.log').open('a') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);write(HERE/f'logs/{key}_process.json',dict(state='running',pid=p.pid,command=cmd,start=start));rc=p.wait()
 write(HERE/f'logs/{key}_process.json',dict(state='complete' if rc==0 else 'failed',pid=p.pid,command=cmd,start=start,seconds=time.perf_counter()-t,returncode=rc))
 assert rc==0,key
def main():
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';(HERE/'logs').mkdir(exist_ok=True);write(HERE/'status.json',dict(state='running',planned_complete_episodes=5980,planned_physical_steps=3588000,new_training_steps=0,new_optimizer_updates=0))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for future in concurrent.futures.as_completed([pool.submit(job,item) for item in m['instances']]):future.result()
 write(HERE/'status.json',dict(state='evaluation_complete_pending_analysis',complete_episodes=5980,physical_steps=3588000,new_training_steps=0,new_optimizer_updates=0))
if __name__=='__main__':guard();main()
