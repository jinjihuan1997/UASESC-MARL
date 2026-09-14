from continuation import *
import argparse
def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--cpu',type=int,required=True);x=a.parse_args();os.sched_setaffinity(0,{x.cpu});guard();m=verify();assert x.seed in m['seeds'];assert read(HERE/'preflight.json')['state']=='PASS';assert read(HERE/'migration_results.json')['state']=='PASS'
 for name,h in read(HERE/'execution_seal.json')['sources'].items():assert sha(HERE/name)==h
 formal=base.load_module('quality_long_formal_trainer',FROZEN/'source/formal_train.py');formal.TensorTrainer=Trainer;formal.write=write
 return formal.train(HERE/f'configs/seed_{x.seed}.json',HERE/f'jobs/seed_{x.seed}','cpu',HERE/'manifest.json',checkpoint_every=25,resume=True)
if __name__=='__main__':raise SystemExit(main())
