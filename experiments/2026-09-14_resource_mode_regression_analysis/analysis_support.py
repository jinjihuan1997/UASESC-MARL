import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
LONG=HERE.parent/'2026-09-14_quality_half_long_training';SHORT=HERE.parent/'2026-09-14_quality_half_retraining'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(LONG))
import train_support as prior
import json,hashlib,time,datetime
import numpy as np
import torch
torch.set_num_threads(1)
ws=prior.ws;base=prior.base;FIELDS=prior.FIELDS;read=prior.read;sha=prior.sha;arrays=prior.arrays
EVAL=base.load_module('resource_mode_readonly_models',LONG/'evaluate.py')
MAN=read(LONG/'manifest.json');SEEDS=MAN['seeds'];STEPS=[1000000,2000000,4000000,6000000,8000000,10000000]
GROUPS=ws.manifest()['equivalence_groups'];LOW=next(g for g in GROUPS if 0 in g);MID=next(g for g in GROUPS if 5 in g)
def dump(p,x):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def text(p,s):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def save(p,**x):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**x)
 p.with_suffix('.tmp').replace(p)
def trace(seed,step,scene='fixed_2'):
 root=SHORT if step==1000000 else LONG
 return root/f'evaluation/seed_{seed}/steps_{step}/{scene}.npz'
def guard():
 def deny(*a,**k):raise RuntimeError('Read-only analysis: no gradients, optimizer updates or physical steps')
 torch.Tensor.backward=deny;torch.autograd.backward=deny;torch.autograd.grad=deny
 for c in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:c.step=deny
 prior.env_kernel.TensorSCEnv.step=deny;prior.env_kernel.TensorSCEnv.commit_modes=deny
 torch.set_grad_enabled(False)
 def audit(event,args):
  paths=[]
  if event=='open':
   p,mode,flags=args
   if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):paths=[p]
  elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.truncate']:paths=[args[0]]
  elif event in ['os.rename','os.link','os.symlink']:paths=list(args[:2])
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and not Path(os.fsdecode(p)).resolve().is_relative_to(HERE) and Path(os.fsdecode(p)).resolve()!=Path('/dev/null'):raise PermissionError(str(p))
 sys.addaudithook(audit)
def stats(x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 return dict(n=len(x),mean=float(x.mean()),p05=float(np.quantile(x,.05)),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(x.max())) if len(x) else dict(n=0,mean=None,p05=None,p50=None,p95=None,maximum=None)
