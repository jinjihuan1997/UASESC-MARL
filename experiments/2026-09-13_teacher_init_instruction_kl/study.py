"""Isolated experiment IO, seed partitions, frozen environment and cost accounting."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
FROZEN=HERE.parent/'2026-09-11_alternating_training';REPAIR=HERE.parent/'2026-09-12_quality_mode_repair';GREEDY=HERE.parent/'2026-09-11_observation_matched_greedy'
RECOMPOSE=sorted(p for p in HERE.parent.glob('*_policy_recomposition*') if (p/'status.json').exists())[-1]
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(REPAIR))
import repair_support as frozen_support
import copy,json,hashlib,datetime,time,random,contextlib,importlib.util,traceback
import numpy as np
import torch
import torch._dynamo
import tensor_env
from episode_source import EpisodeSource,SCUAVEnv
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from common import network_hash
FIELDS=frozen_support.FIELDS;arr=frozen_support.arr;sha=frozen_support.sha;read=frozen_support.read;frozen=frozen_support.frozen
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
ENV_LEDGER=[]
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard():
 def audit(event,args):
  paths=[]
  if event=='open':
   p,mode,flags=args
   if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):paths=[p]
  elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate'):paths=[args[0]]
  elif event in ('os.rename','os.link','os.symlink'):paths=list(args[:2])
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and Path(os.fsdecode(p)).resolve()!=Path('/dev/null') and not Path(os.fsdecode(p)).resolve().is_relative_to(HERE):raise PermissionError(f'Protected write: {p}')
 sys.addaudithook(audit)
def write(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def save(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');torch.save(d,t);t.replace(p)
def npz(p,**d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('wb') as f:np.savez_compressed(f,**d)
 t.replace(p)
def append(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')
def manifest():return read(HERE/'manifest.json')
def cfg():return read(HERE/'config.json')
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def prior_auditor():return load_module('teacher_study_prior_auditor',REPAIR/'aggregate.py')
def verify(full=True):
 m=manifest();assert sha(HERE/'PROTOCOL.md')==m['protocol_sha256'] and sha(HERE/'config.json')==m['config_sha256']
 if full:
  for p,h in m['protected_sha256'].items():assert sha(ROOT/p)==h,p
 if (HERE/'execution_seal.json').exists():
  for p,h in read(HERE/'execution_seal.json')['source_sha256'].items():assert sha(HERE/p)==h,p
 return m

def make_env(seeds,purpose,schedule=None,device='cpu'):
 m=manifest();seeds=list(seeds);assert len(seeds)==len(set(seeds))
 assert set(seeds)<=set(m['allowed_environment_seeds'][purpose]) and not set(seeds)&set(m['reserved'])
 if purpose in ('A0','dagger','warmup','B','preflight'):assert not set(seeds)&set(m['known_development_seeds']+m['gate_dev'])
 if purpose=='gate':assert set(seeds)<=set(m['gate_dev'])
 args=copy.deepcopy(cfg()['env_args'])
 if schedule is not None:args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
 else:assert args['instruction_mode_strategy']=='random_switch_once'
 class ApprovedSC(SCUAVEnv):
  def __init__(self,args,s):self.approved=s;super().__init__(args)
  def seed(self,seed=None):
   assert seed is None or seed==self.approved;return super().seed(self.approved)
 class ApprovedSource(EpisodeSource):
  def __init__(self,args,count,seed):
   assert count==len(seeds) and seed==seeds[0];self.envs=[ApprovedSC(copy.deepcopy(args),s) for s in seeds]
 original=tensor_env.EpisodeSource;tensor_env.EpisodeSource=ApprovedSource
 try:env=tensor_env.TensorSCEnv(args,len(seeds),seeds[0],device)
 finally:tensor_env.EpisodeSource=original
 env.reset();assert env.p.instruction_names==['balance','aoi','quality'] and env.p.actor_observe_instruction
 ENV_LEDGER.append(dict(seeds=seeds,purpose=purpose,schedule=schedule,device=device))
 return env

class ActionStreams(frozen_support.ActionStreams):pass

def record_cost(kind,identity,**kwargs):
 d=dict(kind=kind,identity=identity,utc=stamp(),pid=os.getpid(),**kwargs)
 append(HERE/f'costs/{os.getpid()}.jsonl',d)

def require_A_gate():
 gate=read(HERE/'gate_report.json');assert gate['state']=='PASS_ALL_THREE' and len(gate['students'])==3 and all(x['passed'] for x in gate['students'].values())
 assert gate['manifest_sha256']==sha(HERE/'manifest.json')
 for s,d in gate['students'].items():assert sha(HERE/f'students/{s}/A3/checkpoint.pt')==d['checkpoint_sha256']
 return gate

def state_equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
 return a==b
