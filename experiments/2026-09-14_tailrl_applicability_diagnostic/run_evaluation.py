from diag_support import *
from concurrent.futures import ThreadPoolExecutor
import queue
def main():
 m=verify();assert read(HERE/'preflight.json')['state']=='PASS';cores=queue.Queue()
 for c in [0,1]:cores.put(c)
 def worker(seed):
  cpu=cores.get()
  try:
   with (HERE/f'evaluation_{seed}.log').open('a') as log:subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--seed',str(seed),'--cpu',str(cpu)],stdout=log,stderr=subprocess.STDOUT,check=True)
  finally:cores.put(cpu)
 with ThreadPoolExecutor(max_workers=2) as ex:
  futures=[ex.submit(worker,seed) for seed in m['parents']]
  while not all(f.done() for f in futures):
   write(HERE/'status.json',dict(state='controlled_evaluation',completed_batches=len(list((HERE/'evaluation').rglob('*.npz'))),target_batches=153,new_training_steps=0,new_optimizer_updates=0,utc=stamp()));time.sleep(5)
  for f in futures:f.result()
 write(HERE/'evaluation_complete.json',dict(state='complete',batches=153,episodes=3060,physical_steps=1836000,new_training_steps=0,new_optimizer_updates=0));print('EVALUATION COMPLETE',flush=True)
if __name__=='__main__':main()
