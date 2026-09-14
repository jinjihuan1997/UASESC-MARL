"""Isolated retraining with frozen HAPPO kernels and one reward intervention."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;PREVIOUS=HERE.parent/'2026-09-13_quality_weight_half'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
sys.path.insert(0,str(PREVIOUS))
import weight_support as ws
import tensor_train as kernel
import tensor_env as env_kernel
import training_checkpoint as checkpoint
import model_snapshot
import copy,json,time,datetime,hashlib,random,subprocess,traceback,types
import numpy as np
import torch
torch.set_num_threads(1)
base=ws.base;FROZEN=base.FROZEN;FIELDS=base.FIELDS;arr=base.arr;sha=base.sha;read=base.read;arrays=base.arrays
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def textfile(p,s):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(s);t.replace(p)
def npz(p,**d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**d)
 p.with_suffix('.tmp').replace(p)
def append(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n')
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
 sys.addaudithook(audit)
def manifest():return read(HERE/'manifest.json')
def config(seed):return read(HERE/f'configs/seed_{seed}.json')
def verify():
 m=manifest();assert sha(HERE/'PROTOCOL.md')==m['protocol_sha256']
 for p,h in m['protected_sha256'].items():assert sha(ROOT/p)==h,p
 for p,h in m['config_sha256'].items():assert sha(HERE/p)==h
 if (HERE/'execution_seal.json').exists():
  for p,h in read(HERE/'execution_seal.json')['sources'].items():assert sha(HERE/p)==h,p
 return m

class TrainingEnv(env_kernel.TensorSCEnv):
 def __init__(self,args,count,seed,device,quality_factor=.5,audit_directory=None):
  m=manifest();derived=[int(seed)+1000*i for i in range(count)];assert count==10 and derived==m['training_environment_seeds'][str(seed)]
  assert not set(derived)&set(m['validation']+m['reserved']);self.quality_factor=quality_factor;self.audit_directory=Path(audit_directory) if audit_directory else None
  class ApprovedSC(base.SCUAVEnv):
   def __init__(self,a,s):self.approved=s;super().__init__(a)
   def seed(self,seed=None):assert seed is None or int(seed)==self.approved;return super().seed(self.approved)
  class Source(base.EpisodeSource):
   def __init__(self,a,count,seed):self.envs=[ApprovedSC(copy.deepcopy(a),s) for s in derived]
  original=env_kernel.EpisodeSource;env_kernel.EpisodeSource=Source
  try:super().__init__(args,count,seed,device)
  finally:env_kernel.EpisodeSource=original
  weights=np.asarray(self.p.reward_weights_by_instruction).copy();weights[:,0]*=quality_factor
  for p in self.source.envs:p.reward_weights_by_instruction=weights.copy()
  self.reward_weights=self.tensor(weights);self.audited_physical_steps=torch.zeros((),device=self.device,dtype=torch.int64)
  self.audit_keys=dict(trace=(len(FIELDS),),modes=(3,),requested_modes=(3,),raw_resource_action=(3,),resource_fractions=(3,),budgets=(3,),usage=(3,),unused_budgets=(3,),predicted_quality=(3,),quality_requirement=(3,),uav_masks=(3,16),aoi_after=(3,10),cache_after=(3,10),tau_after=(3,10),served=(3,10))
  for key,shape in self.audit_keys.items():setattr(self,'audit_'+key,torch.zeros((600,count,*shape),device=self.device,dtype=torch.float64))
  for key in ['q','tau','aoi']:setattr(self,'audit_initial_'+key,torch.zeros((count,3,10),device=self.device,dtype=torch.float64))
 def selected_episode(self):return self.episode_indices[0]%25==0
 @torch.no_grad()
 def reset(self):
  result=super().reset()
  for key in ['q','tau','aoi']:getattr(self,'audit_initial_'+key).copy_(getattr(self,key))
  if self.audit_directory:
   append(self.audit_directory/'environment_ledger.jsonl',dict(event='episode_setup',episode_indices=self.episode_indices,seeds=[p._base_seed for p in self.source.envs],external_hashes=base.frozen.external_hashes(self),physical_counter=int(self.audited_physical_steps)))
  return result
 @torch.no_grad()
 def allocate_resources(self,action):
  result=super().allocate_resources(action)
  if self.selected_episode():
   self.audit_raw_resource_action[self.step_index].copy_(action);self.audit_uav_masks[self.step_index].copy_(result[2][:,1:])
  return result
 @torch.no_grad()
 def commit_modes(self,actions,auto_reset=True):
  slot=self.step_index;before={k:arr(getattr(self,k)).copy() for k in ['aoi','q','tau']};selected=self.selected_episode()
  obs,state,reward,done,info,masks=super().commit_modes(actions,auto_reset=False)
  values=base.frozen.check_tensor(self,info,slot,before);values['instruction_id']=arr(info['gid']);np.testing.assert_allclose(arr(reward[:,0,0]),values['common_reward'],atol=1e-6,rtol=1e-6)
  self.audited_physical_steps+=self.count
  if selected:
   data=dict(trace=np.column_stack([values[f] for f in FIELDS]),modes=info['mode'],requested_modes=torch.stack([x.argmax(-1) for x in actions],-1),resource_fractions=self.beta,budgets=info['budget'],usage=info['usage'],unused_budgets=info['budget']-info['usage'],predicted_quality=info['quality'],quality_requirement=info['req'],aoi_after=self.aoi,cache_after=self.q,tau_after=self.tau,served=info['served'])
   for key,value in data.items():getattr(self,'audit_'+key)[slot].copy_(self.tensor(value))
  if bool(done[0,0]) and selected and self.audit_directory:self.save_audit_episode()
  if bool(done[0,0]) and auto_reset:
   info.update(terminal_observation=obs,terminal_state=state,terminal_available=masks);obs,state,masks=self.reset()
  return obs,state,reward,done,info,masks
 def save_audit_episode(self):
  z=base.Arrays({key:arr(getattr(self,'audit_'+key)).copy() for key in self.audit_keys})
  for key in ['cache_after','served','uav_masks']:z[key]=z[key].astype(bool)
  for key in ['modes','requested_modes']:z[key]=z[key].astype(np.int8)
  z.update(fields=np.array(FIELDS),seeds=np.array([p._base_seed for p in self.source.envs]),initial_q=arr(self.audit_initial_q).astype(bool),initial_aoi=arr(self.audit_initial_aoi),initial_tau=arr(self.audit_initial_tau))
  c=base.cfg();c['env_args']['reward_weights_by_instruction']=np.asarray(self.p.reward_weights_by_instruction).tolist();check=base.auditor().physical_audit(z,ws.PHYS.tables(self),c)
  target=self.audit_directory/f'episode_{self.episode_indices[0]:06d}.npz'
  if target.exists():
   existing=arrays(target)
   for k in z:np.testing.assert_array_equal(z[k],existing[k])
  else:npz(target,**z)
  write(target.with_suffix('.json'),dict(state='PASS',episode=self.episode_indices[0],trace_sha256=sha(target),independent_audit=check,seeds=z['seeds'].tolist(),source='real sampled training trajectory',physical_steps=600*self.count))

class Trainer(kernel.TensorTrainer):
 def __init__(self,configuration,device='cpu'):
  factory=kernel.TensorSCEnv
  def env(args,count,seed,device):return TrainingEnv(args,count,seed,device,configuration.get('quality_multiplier',.5),configuration.get('audit_directory'))
  kernel.TensorSCEnv=env
  try:super().__init__(configuration,device)
  finally:kernel.TensorSCEnv=factory
  if device=='cpu':
   prior=read(FROZEN/f'jobs/seed_{configuration["algo_args"]["seed"]["seed"]}/joint/manifest.json')
   assert self.initial_actors==prior['initial_actor_hashes'] and self.initial_critic==prior['initial_critic_hash'], 'Fresh initialization must match the paired original joint experiment'
 def update(self,*a,**kw):
  # Preserve the original finite kernels. Stop rather than silently replacing
  # nonfinite advantages/probability factors in the legacy nan_to_num calls.
  original=torch.nan_to_num
  def finite(value,*args,**kwargs):
   if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite tensor before legacy nan_to_num')
   return original(value,*args,**kwargs)
  torch.nan_to_num=finite
  try:result=super().update(*a,**kw)
  finally:torch.nan_to_num=original
  for actor in self.actors:
   if not all(torch.isfinite(p).all() for p in actor.actor.parameters()):raise FloatingPointError('Nonfinite actor weights')
  return result
