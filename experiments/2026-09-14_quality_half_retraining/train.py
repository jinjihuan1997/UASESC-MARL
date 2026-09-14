"""Controlled entry: fresh joint HAPPO, constant budget, isolated reward change."""
from train_support import *
import argparse

def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--device',default='cpu',choices=['cpu','cuda:0']);a.add_argument('--resume',action='store_true');a.add_argument('--cpu',type=int);x=a.parse_args()
 if x.cpu is not None:os.sched_setaffinity(0,{x.cpu})
 guard();m=verify();assert x.seed in m['seeds'];assert read(HERE/'preflight.json')['state']=='PASS'
 formal=base.load_module('quality_half_formal_trainer',FROZEN/'source/formal_train.py');formal.TensorTrainer=Trainer
 def corrected_write(path,data):
  if Path(path).name=='manifest.json':data.update(source_snapshot=str(FROZEN/'source'),wrapper=str(HERE/'train_support.py'),reward_multiplier=.5,from_scratch=True,loaded_parent_weights=False)
  write(path,data)
 formal.write=corrected_write
 return formal.train(HERE/f'configs/seed_{x.seed}.json',HERE/f'jobs/seed_{x.seed}',x.device,HERE/'manifest.json',checkpoint_every=25,resume=x.resume)
if __name__=='__main__':raise SystemExit(main())
