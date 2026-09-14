"""Fixed per-observation routing, never online search or trainable routing."""
from support import *
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from adapter_policy import ResidualPolicy

# Policy source IDs: 0 original; 1 full effective all adapter; 2 full quality adapter.
class Composition:
 def __init__(self,seed,env,controller):
  assert controller in ('C0','C1','C2','C3');self.controller=controller
  self.B=controller in ('C1','C3');self.Q=controller in ('C2','C3');self.seed=seed
  cfg=cfg_for(seed);args={**cfg['algo_args']['model'],**cfg['algo_args']['algo']}
  assert not args['use_recurrent_policy'] and not args['use_naive_recurrent_policy']
  self.device=env.device;parent=old.model_dir(seed);ps=read(parent/'status.json')
  self.base=[]
  for i in range(4):
   net=StochasticPolicy(args,env.observation_space[i],env.action_space[i],env.device)
   net.load_state_dict(torch.load(parent/f'actor_agent{i}.pt',map_location=env.device,weights_only=True),strict=True)
   net.eval().requires_grad_(False);assert network_hash(net)==ps['actor_hashes'][i]
   self.base.append(net)
  self.uav={0:self.base[1:]}
  for sid,arm in [(1,'residual_all'),(2,'residual_quality')]:
   folder=OLD/f'jobs/seed_{seed}/{arm}/milestones/steps_1000000';status=read(folder/'status.json')
   assert status['completed_steps']==1000000 and status['state']=='complete' and status['base_hashes']==ps['actor_hashes']
   self.uav[sid]=[]
   for i in range(1,4):
    net=ResidualPolicy(copy.deepcopy(self.base[i]),76,arm,env.device)
    net.adapter.load_state_dict(torch.load(folder/f'adapter_agent{i}.pt',map_location=env.device,weights_only=True),strict=True)
    net.eval().requires_grad_(False);assert network_hash(net.adapter)==status['adapter_hashes'][i-1]
    self.uav[sid].append(net)
  self.initial_hashes=self.hashes();self.assert_frozen()
 def hashes(self):
  return {'sut':network_hash(self.base[0]),**{f'uav_{s}_{i}':network_hash(n) for s,ns in self.uav.items() for i,n in enumerate(ns)}}
 def assert_frozen(self):
  for n in [self.base[0]]+sum(list(self.uav.values()),[]):
   assert not any(p.requires_grad for p in n.parameters()) and not any(q.training for q in n.modules())
  assert self.hashes()==self.initial_hashes
 @staticmethod
 def gid(obs,start):
  one=obs[:,start:start+3];assert torch.all((one==0)|(one==1)) and torch.all(one.sum(-1)==1)
  return one.argmax(-1)
 def call(self,net,obs,mask):
  return net(obs,torch.zeros((len(obs),1,256),device=self.device),torch.ones((len(obs),1),device=self.device),mask,deterministic=True)[0]
 @torch.inference_mode()
 def resource(self,sut_obs,sut_mask,equal_action):
  gid=self.gid(sut_obs,23);original=self.call(self.base[0],sut_obs,sut_mask)
  source=((gid==2)&self.Q).to(torch.int8)
  # Equal helper uses float64; native allocator casts BEFORE adding one.
  action=torch.where(source[:,None].bool(),equal_action,original) if self.Q else original
  return action,source
 @torch.inference_mode()
 def mode(self,uav_index,local_obs,local_mask):
  gid=self.gid(local_obs,67)
  source=torch.where(gid==2,2,torch.where((gid==0)&self.B,1,0))
  action=torch.empty((len(local_obs),16),device=self.device)
  for s in (0,1,2):
   take=source==s
   if take.any():
    # Evaluate a complete batch, preserving frozen CPU GEMM evaluation shape.
    candidate=self.call(self.uav[s][uav_index],local_obs,local_mask)
    action[take]=candidate[take]
  return action,source.to(torch.int8)
