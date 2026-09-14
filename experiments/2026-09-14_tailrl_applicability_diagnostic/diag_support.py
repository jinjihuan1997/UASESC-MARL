import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;SHORT=HERE.parent/'2026-09-14_quality_half_retraining';LONG=HERE.parent/'2026-09-14_quality_half_long_training'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(SHORT))
import train_support as prior
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.models.value_function_models.v_net import VNet
import json,hashlib,time,datetime,copy,traceback,subprocess,math
import numpy as np
import torch
torch.set_num_threads(1)
base=prior.base;ws=prior.ws;arr=prior.arr;arrays=prior.arrays;read=prior.read;sha=prior.sha;FIELDS=prior.FIELDS;FROZEN=prior.FROZEN;FORBIDDEN=[]
def write(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def textfile(p,s):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def npz(p,**d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**d)
 p.with_suffix('.tmp').replace(p)
def append(p,d):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def manifest():return read(HERE/'manifest.json')
def guard():
 def audit(event,args):
  paths=[]
  if event=='open':
   p,mode,flags=args
   if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):paths=[p]
  elif event in ['os.mkdir','os.remove','os.rmdir','os.chmod','os.truncate']:paths=[args[0]]
  elif event in ['os.rename','os.link','os.symlink']:paths=list(args[:2])
  for p in paths:
   if isinstance(p,(str,bytes,os.PathLike)) and Path(os.fsdecode(p)).resolve()!=Path('/dev/null') and not Path(os.fsdecode(p)).resolve().is_relative_to(HERE):raise PermissionError(str(p))
 def deny(*a,**kw):FORBIDDEN.append('training_update');raise RuntimeError('No backward/optimizer.step; diagnostic autograd.grad only')
 torch.Tensor.backward=deny;torch.autograd.backward=deny
 for cls in [torch.optim.Optimizer,torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD]:cls.step=deny
 sys.addaudithook(audit)
def verify():
 m=manifest();assert sha(HERE/'PROTOCOL.md')==m['protocol_sha256']
 for p,h in m['protected_sha256'].items():assert sha(ROOT/p)==h,p
 if (HERE/'execution_seal.json').exists():
  for p,h in read(HERE/'execution_seal.json')['sources'].items():assert sha(HERE/p)==h,p
 return m
def model_dir(seed,steps):return ROOT/manifest()['models'][f'{seed}/{steps}']['path']

class Policy:
 def __init__(self,seed,steps,env):
  c=prior.config(seed);args={**c['algo_args']['model'],**c['algo_args']['algo']};p=model_dir(seed,steps);status=read(p/'status.json');assert status['completed_steps']==steps
  self.nets=[];self.device=env.device
  for i in range(4):
   n=StochasticPolicy(args,env.observation_space[i],env.action_space[i],env.device);n.load_state_dict(torch.load(p/f'actor_agent{i}.pt',weights_only=True,map_location='cpu'));n.eval().requires_grad_(False);self.nets.append(n)
  assert [prior.kernel.network_hash(n) for n in self.nets]==status['actor_hashes']
  self.critic=VNet(args,env.share_observation_space[0],env.device);self.critic.load_state_dict(torch.load(p/'critic_agent.pt',weights_only=True,map_location='cpu'));self.critic.eval().requires_grad_(False)
  self.normalizer=prior.kernel.TensorValueNorm(1,device=env.device);self.normalizer.load_state_dict(torch.load(p/'value_normalizer.pt',weights_only=True,map_location='cpu'));self.initial_hashes=self.hashes()
 def hashes(self):return [prior.kernel.network_hash(n) for n in self.nets+[self.critic,self.normalizer]]
 def assert_frozen(self):assert self.hashes()==self.initial_hashes;assert not any(n.training for n in self.nets+[self.critic]);assert all(p.grad is None for n in self.nets for p in n.parameters())
 def call(self,i,obs,mask,det):return self.nets[i](obs,torch.zeros((len(obs),1,256)),torch.ones((len(obs),1)),mask,deterministic=det)[:2]
 def value(self,state):return self.normalizer.denormalize(self.critic(state,torch.zeros((len(state),1,256)),torch.ones((len(state),1)))[0]).squeeze(-1)

def tail_advantage(reward):
 r=np.asarray(reward,dtype=np.float64);assert r.ndim==1 and np.isfinite(r).all();N=len(r);order=np.argsort(r,kind='stable');x=r[order];gaps=np.diff(x,prepend=x[0]);w=N*np.cumsum(gaps/np.arange(N,0,-1));w-=w.mean();a=np.empty_like(w);a[order]=w;return a
def distribution(x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 if not len(x):return dict(n=0,mean=None,median=None,std=None,p50=None,p75=None,p90=None,p95=None,p99=None,maximum=None)
 qs=np.quantile(x,[.5,.75,.9,.95,.99]);return dict(n=len(x),mean=float(x.mean()),median=float(qs[0]),std=float(x.std()),**dict(zip(['p50','p75','p90','p95','p99'],map(float,qs))),maximum=float(x.max()))
def corr(a,b):
 a=np.asarray(a);b=np.asarray(b);take=np.isfinite(a)&np.isfinite(b)
 if take.sum()<3 or a[take].std()<1e-14 or b[take].std()<1e-14:return None
 return float(np.corrcoef(a[take],b[take])[0,1])

def episode_metrics(z,gid=2):
 fields=z['fields'].tolist();v=z['trace'];ids=v[:,:,fields.index('instruction_id')].astype(int);take=np.ones(ids.shape,bool) if gid is None else ids==gid;T=take.sum(0);den=np.maximum(T,1)
 def field(k):return v[:,:,fields.index(k)]
 delivered=field('deliveries');ps=field('predicted_quality_sum');D=(delivered*take).sum(0);pssum=(ps*take).sum(0)
 q=(ps-21*delivered)/12/30
 def avg(x):return (x*take).sum(0)/den
 def mean_array(k):return (z[k]*take[:,:,None]).sum(0)/den[:,None]
 out=dict(quality=avg(q),psnr=np.divide(pssum,D,out=np.full_like(D,np.nan,dtype=float),where=D>0),deliveries=avg(delivered),mean_aoi=avg(field('mean_aoi')),max_aoi=avg(field('max_aoi')),p95_aoi=avg(field('p95_aoi')),max_aoi_ever=np.where(take,field('max_aoi'),-np.inf).max(0),fraction_above6=avg(field('fraction_above6')),resource=avg(field('channel_uses')),native_reward=avg(field('common_reward')),Q_slots=T,valid=(T>0),no_delivery=(D==0))
 weights=np.asarray(manifest()['effective_weights']);reward=weights[ids,0]*q-field('age_mean_cost')-field('age_max_cost')-field('age_tail_cost')-field('resource_cost')-field('service_violation_cost')+field('recv_aoi_bonus');out['reward']=avg(reward)
 for k in ['age_mean_cost','age_max_cost','age_tail_cost','resource_cost','quality_violations','budget_violations','cache_violations']:out[k]=avg(field(k))
 out['violation']=sum(out[k] for k in ['quality_violations','budget_violations','cache_violations'])
 for k in ['resource_fractions','budgets','usage','unused_budgets']:out[k]=mean_array(k)
 out['requested_mode_counts']=np.stack([np.stack([((z['requested_modes'][:,:,u]==mode)&take).sum(0)/den for mode in range(16)],-1) for u in range(3)],1)
 if 'snr_db' in z:out['snr_db']=mean_array('snr_db')
 for k in out:
  if k!='valid':out[k]=np.asarray(out[k])
 return out
