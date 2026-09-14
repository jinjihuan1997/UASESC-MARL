"""Isolated inference-only experiment; immutable old inputs and approved seeds."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
FROZEN=HERE.parent/'2026-09-11_alternating_training';REPAIR=HERE.parent/'2026-09-12_quality_mode_repair';GREEDY=HERE.parent/'2026-09-11_observation_matched_greedy'
RECOMPOSE=sorted(p for p in HERE.parent.glob('*_policy_recomposition*') if (p/'status.json').exists())[-1]
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(RECOMPOSE))
import support as composition_support
import repair_support as old
from composition_policy import Composition
import copy,json,hashlib,time,datetime,contextlib,traceback,importlib.util,types
import numpy as np
import torch
import torch._dynamo
from episode_source import EpisodeSource,SCUAVEnv
import tensor_env
from common import network_hash
from harl.models.policy_models.stochastic_policy import StochasticPolicy
frozen=old.frozen;FIELDS=old.FIELDS;arr=old.arr;sha=old.sha;read=old.read
FORBIDDEN=[];ENV_CREATIONS=[]
torch.set_num_threads(1)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def textfile(p,s):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(s);t.replace(p)
def npz(p,**d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert p.resolve().is_relative_to(HERE)
 with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**d)
 p.with_suffix('.tmp').replace(p)
def append(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')
def cost(kind,**d):append(HERE/f'costs/{os.getpid()}.jsonl',dict(kind=kind,utc=stamp(),pid=os.getpid(),**d))
def guard():
 def audit(event,args):
  paths=[]
  if event=='open':
   p,mode,flags=args
   if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):paths=[p]
  elif event in ['os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate']:paths=[args[0]]
  elif event in ['os.rename','os.link','os.symlink']:paths=list(args[:2])
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and Path(os.fsdecode(p)).resolve()!=Path('/dev/null') and not Path(os.fsdecode(p)).resolve().is_relative_to(HERE):raise PermissionError(f'Protected write: {p}')
 def deny(*a,**kw):FORBIDDEN.append('gradient_or_optimizer');raise RuntimeError('This experiment forbids training and fitting')
 torch.Tensor.backward=deny;torch.autograd.backward=deny;torch.autograd.grad=deny
 for cls in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=deny
 torch.set_grad_enabled(False);sys.addaudithook(audit)
def manifest():return read(HERE/'manifest.json')
def cfg(seed=None):
 seed=seed or manifest()['parents'][0];d=read(HERE/f'configs/seed_{seed}.json')
 d['env_args']['semantic_registry_path']=str(FROZEN/'source/reference/inputs/mode_registry.json');d['env_args']['semantic_profile_path']=str(FROZEN/'source/reference/inputs/profile.npz');return d
def verify(full=True):
 m=manifest();assert sha(HERE/'PROTOCOL.md')==m['protocol_sha256']
 for p,h in m['config_sha256'].items():assert sha(HERE/p)==h
 if full:
  for p,h in m['protected_sha256'].items():assert sha(ROOT/p)==h,p
 if (HERE/'execution_seal.json').exists():
  for p,h in read(HERE/'execution_seal.json')['source_sha256'].items():assert sha(HERE/p)==h,p
 return m
def make_env(schedule,seeds=None):
 m=manifest();seeds=list(m['validation'] if seeds is None else seeds);assert len(seeds)==len(set(seeds)) and set(seeds)<=set(m['validation']) and not set(seeds)&set(m['reserved'])
 args=copy.deepcopy(cfg()['env_args']);args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
 class ApprovedSC(SCUAVEnv):
  def __init__(self,a,s):self.approved=s;super().__init__(a)
  def seed(self,seed=None):assert seed is None or seed==self.approved;return super().seed(self.approved)
 class ApprovedSource(EpisodeSource):
  def __init__(self,a,count,seed):
   assert count==len(seeds) and seed==seeds[0];self.envs=[ApprovedSC(copy.deepcopy(a),s) for s in seeds]
 original=tensor_env.EpisodeSource;tensor_env.EpisodeSource=ApprovedSource
 try:env=tensor_env.TensorSCEnv(args,len(seeds),seeds[0],'cpu')
 finally:tensor_env.EpisodeSource=original
 env.reset();assert env.p.instruction_names==['balance','aoi','quality'] and env.p.actor_observe_instruction and env.episode_indices==[0]*len(seeds)
 ENV_CREATIONS.append(dict(seeds=seeds,schedule=schedule,episode_indices=env.episode_indices));return env
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def auditor():return load_module('lightweight_old_physics_audit',REPAIR/'aggregate.py')
class Arrays(dict):
 @property
 def files(self):return list(self)
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return Arrays({k:z[k] for k in z.files})

def state_before(env,z,t,indices=None):
 ix=np.arange(env.count) if indices is None else indices
 for k in ['q','tau','aoi']:
  src=z['initial_'+k] if t==0 else z[{'q':'cache_after','tau':'tau_after','aoi':'aoi_after'}[k]][t-1]
  setattr(env,k,env.tensor(src[ix],getattr(env,k).dtype))
 env.step_index=t;env.allocation_pending=False;env.beta=env.tensor(np.full((env.count,3),1/3) if t==0 else z['resource_fractions'][t-1,ix]);env.update_channels(t);env.update_tables()
 return env.observe()
