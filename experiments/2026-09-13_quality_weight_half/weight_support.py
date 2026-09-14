"""Read-only original imports and isolated quality-weight intervention."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;LIGHT=HERE.parent/'2026-09-13_lightweight_baselines'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
sys.path.insert(0,str(LIGHT))
import light_support as base
import copy,json,hashlib,time,datetime,traceback,types,subprocess
import numpy as np
import torch
from dataclasses import replace
torch.set_num_threads(1)
arr=base.arr;sha=base.sha;read=base.read;FIELDS=base.FIELDS;Arrays=base.Arrays;arrays=base.arrays
REF=base.load_module('weight_original_reference',LIGHT/'reference_policies.py')
PHYS=base.load_module('weight_original_profile',LIGHT/'independent_physics.py')
LIGHTPOL=__import__('lightweight_policies');FORBIDDEN=[];CREATIONS=[]

def write(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def textfile(p,s):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(s);tmp.replace(p)
def npz(p,**d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**d)
 p.with_suffix('.tmp').replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def cost(kind,**d):
 p=HERE/f'costs/{os.getpid()}.jsonl';p.parent.mkdir(exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(dict(kind=kind,utc=stamp(),**d),allow_nan=False)+'\n')
def guard():
 def audit(event,args):
  paths=[]
  if event=='open':
   p,mode,flags=args
   if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):paths=[p]
  elif event in ['os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate']:paths=[args[0]]
  elif event in ['os.rename','os.link','os.symlink']:paths=list(args[:2])
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and Path(os.fsdecode(p)).resolve()!=Path('/dev/null') and not Path(os.fsdecode(p)).resolve().is_relative_to(HERE):raise PermissionError(str(p))
 def deny(*a,**k):FORBIDDEN.append('gradient_or_optimizer');raise RuntimeError('Inference only')
 torch.Tensor.backward=deny;torch.autograd.backward=deny;torch.autograd.grad=deny
 for cls in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=deny
 torch.set_grad_enabled(False);sys.addaudithook(audit)
def manifest():return read(HERE/'manifest.json')
def effective_cfg(parent=None):
 d=base.cfg(parent);d['env_args']['reward_weights_by_instruction']=manifest()['effective_weights'];return d
def verify():
 m=manifest();assert sha(HERE/'PROTOCOL.md')==m['protocol_sha256']
 for p,h in m['protected_sha256'].items():assert sha(ROOT/p)==h,p
 if (HERE/'execution_seal.json').exists():
  for p,h in read(HERE/'execution_seal.json')['sources'].items():assert sha(HERE/p)==h,p
 return m
def make_env(scene,seeds=None,intervene=True):
 m=manifest();seeds=list(m['validation'] if seeds is None else seeds);assert set(seeds)<=set(m['validation']) and not set(seeds)&set(m['reserved'])
 env=base.make_env(m['scenarios'][scene],seeds)
 # Old constructor requires unit-sum rows. Keep it unchanged; install only the
 # declared multiplicative quality intervention in this private runtime copy.
 # Renormalizing rows would also change the AoI and resource coefficients.
 original=np.asarray(env.p.reward_weights_by_instruction).copy();np.testing.assert_array_equal(original,m['original_weights'])
 if intervene:
  target=np.asarray(m['effective_weights'],dtype=np.float64)
  for private in env.source.envs:private.reward_weights_by_instruction=target.copy()
  env.reward_weights=env.tensor(target)
 np.testing.assert_array_equal(np.asarray(env.p.reward_weights_by_instruction)[:,1:],original[:,1:]);CREATIONS.append(dict(seeds=seeds,scene=scene,episode_indices=env.episode_indices,intervention=intervene))
 return env
def legacy_path(method,parent,scene):
 index=read(LIGHT/'report/results.json')['episode_results'];return ROOT/index[f'{parent or "rules"}/{method}/{scene}']['trace_path']

class Controller:
 def __init__(self,method,env,parent=None):
  self.method=method;self.parent=parent;self.updated=method.endswith('_Qhalf_local');self.origin=method.removesuffix('_Qhalf_local');self.ctrl=REF.policy(self.origin,env,parent)
  if self.updated:
   assert self.origin in manifest()['greedy_methods'];spec=replace(LIGHTPOL.ONLINE.PublicSpec(),weights=tuple(tuple(r) for r in manifest()['effective_weights']))
   candidates=list(range(16)) if self.origin!='greedy_modes_3' else [0,5,10]
   self.ctrl.greedy=LIGHTPOL.ONLINE.UAVGreedy(candidates,spec=spec)
   np.testing.assert_array_equal(arr(self.ctrl.greedy.weights),manifest()['effective_weights'])
  self.initial_hash=REF.policy_hash(self.ctrl)
 def resource(self,obs,mask):return self.ctrl.resource(obs,mask)
 def mode(self,u,obs,mask):return self.ctrl.mode(u,obs,mask)
 def assert_frozen(self):
  self.ctrl.assert_frozen();assert REF.policy_hash(self.ctrl)==self.initial_hash

def independent_check(z,tab):return base.auditor().physical_audit(z,tab,effective_cfg())
def old_physical_equal(z,method,parent,scene,indices=None):
 old=arrays(legacy_path(method,parent,scene));ix=np.arange(20) if indices is None else np.asarray(indices)
 for k in ['modes','requested_modes','raw_resource_action','resource_fractions','budgets','usage','unused_budgets','predicted_quality','quality_requirement','uav_masks','aoi_after','cache_after','tau_after','served']:
  np.testing.assert_array_equal(z[k],old[k][:,ix],err_msg=method+':'+k)
 for f in FIELDS:
  if f in ['common_reward','base_reward','training_reward','quality_credit']:continue
  np.testing.assert_array_equal(z['trace'][:,:,FIELDS.index(f)],old['trace'][:,ix,FIELDS.index(f)],err_msg=f)
 orig=old['trace'][:,ix,0];q=old['trace'][:,ix,FIELDS.index('quality_credit')]
 np.testing.assert_allclose(z['trace'][:,:,0],orig-.5*q,atol=1e-9,rtol=0)
 np.testing.assert_allclose(z['trace'][:,:,FIELDS.index('quality_credit')],.5*q,atol=1e-12,rtol=0)
 return dict(state='EXACT_ACTION_AND_PHYSICAL_PARITY',score_identity_max_error=float(abs(z['trace'][:,:,0]-(orig-.5*q)).max()),old_trace_sha256=sha(legacy_path(method,parent,scene)))
