"""Read-only frozen runtime; inference guard; development-only environment factory."""
import os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent.parent
OLD=HERE.parent/'2026-09-12_quality_mode_repair'
FROZEN=HERE.parent/'2026-09-11_alternating_training'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(OLD))
import repair_support as old
import copy, hashlib, json, datetime, time, importlib.util, contextlib
import numpy as np
import torch
from episode_source import EpisodeSource, SCUAVEnv
import tensor_env
from common import network_hash
frozen=old.frozen;FIELDS=old.FIELDS;arr=old.arr;sha=old.sha;read=old.read;stamp=old.stamp
ENV_CREATIONS=[]
FORBIDDEN_CALLS=[]
def write(p,obj):
 p=Path(p);assert p.resolve().is_relative_to(HERE)
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def guard():
 def audit(event,args):
  paths=[]
  if event=='open':
   p,mode,flags=args
   if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):paths=[p]
  elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate'):paths=[args[0]]
  elif event in ('os.rename','os.link','os.symlink'):paths=list(args[:2])
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)):
    q=Path(os.fsdecode(p)).resolve()
    if q!=Path('/dev/null') and not q.is_relative_to(HERE):raise PermissionError(f'Frozen-input write prohibited: {q}')
 def deny(*a,**kw):
  FORBIDDEN_CALLS.append('gradient_or_optimizer');raise RuntimeError('This experiment prohibits gradient training')
 torch.Tensor.backward=deny;torch.autograd.backward=deny;torch.autograd.grad=deny
 for cls in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=deny
 torch.set_grad_enabled(False);torch.set_num_threads(1)
 sys.addaudithook(audit)
def manifest():return read(HERE/'manifest.json')
def cfg_for(seed):return read(HERE/f'configs/seed_{seed}.json')
def make_env(seed,schedule,seeds=None,device='cpu'):
 m=manifest();seeds=list(m['validation'] if seeds is None else seeds)
 assert seeds and set(seeds)<=set(m['validation']) and not set(seeds)&set(m['reserved_final_test'])
 args=copy.deepcopy(cfg_for(seed)['env_args']);args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
 class ApprovedSC(SCUAVEnv):
  def __init__(self,args,approved):self.approved=approved;super().__init__(args)
  def seed(self,seed=None):
   assert seed is None or seed==self.approved
   return super().seed(self.approved)
 class ApprovedSource(EpisodeSource):
  def __init__(self,args,count,seed):
   assert count==len(seeds) and seed==seeds[0]
   self.envs=[ApprovedSC(copy.deepcopy(args),s) for s in seeds]
 # Only replace construction-time seed assignment. Inherit frozen next() unchanged.
 original=tensor_env.EpisodeSource;tensor_env.EpisodeSource=ApprovedSource
 try:env=tensor_env.TensorSCEnv(args,count=len(seeds),seed=seeds[0],device=device)
 finally:tensor_env.EpisodeSource=original
 env.reset();assert env.p.instruction_names==['balance','aoi','quality']
 assert env.p.actor_observe_instruction and env.obs_dim_common==76
 ENV_CREATIONS.append(dict(seeds=seeds,schedule=schedule,device=device,episode_indices=env.episode_indices))
 assert env.episode_indices==[0]*len(seeds)
 return env

def verify_inputs(full=True):
 m=manifest();assert sha(HERE/'PROTOCOL.md')==m['protocol_sha256']
 for p,h in m['config_sha256'].items():assert sha(HERE/p)==h,p
 if full:
  for p,h in m['protected_input_sha256'].items():assert sha(WORKSPACE/p)==h,p
 seal=HERE/'execution_seal.json'
 if seal.exists():
  for p,h in read(seal)['code_sha256'].items():assert sha(HERE/p)==h,p
 return m

def prior_audit_module():
 spec=importlib.util.spec_from_file_location('prior_numeric_aggregation',OLD/'aggregate.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def tables(scenario):
 m=manifest();env=make_env(m['parents'][0],m['scenarios'][scenario]);external=frozen.external_hashes(env)
 vals={k:[] for k in ('quality','load','req')}
 for slot in range(600):
  env.step_index=slot;env.update_channels(slot);env.update_tables()
  for k,v in [('quality',env.quality),('load',env.load),('req',env.context()[2])]:vals[k].append(arr(v).copy())
 return {**{k:np.stack(v) for k,v in vals.items()},'external':external}

def save_npz(file,records):
 file=Path(file);file.parent.mkdir(parents=True,exist_ok=True)
 with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**records)
 file.with_suffix('.tmp').replace(file)
