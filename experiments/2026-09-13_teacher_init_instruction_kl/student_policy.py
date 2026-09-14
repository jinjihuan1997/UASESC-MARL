"""Fresh original four-actor controller. No teacher or adapter in inference."""
from study import *

class Student:
 def __init__(self,seed,env,device='cpu'):
  self.seed=int(seed);self.device=torch.device(device);self.args={**cfg()['algo_args']['model'],**cfg()['algo_args']['algo']}
  assert not self.args['use_recurrent_policy'] and not self.args['use_naive_recurrent_policy']
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(self.seed)
   self.actors=[StochasticPolicy(self.args,o,a,torch.device('cpu')).to(self.device) for o,a in zip(env.observation_space,env.action_space)]
  for net in self.actors:net.tpdv["device"]=self.device
  self.set_mode(False)
 def set_mode(self,training):
  for net in self.actors:net.train(training);net.requires_grad_(training)
 def hashes(self):return [network_hash(x) for x in self.actors]
 def state(self):return [copy.deepcopy(n.state_dict()) for n in self.actors]
 def load(self,p):
  d=torch.load(p,map_location=self.device,weights_only=False);assert d['student_seed']==self.seed and d['manifest_sha256']==sha(HERE/'manifest.json')
  for n,s in zip(self.actors,d['actors']):n.load_state_dict(s,strict=True)
  assert self.hashes()==d['actor_hashes'];return d
 def forward(self,i,obs,available,deterministic=True):
  net=self.actors[i];obs=obs.to(self.device);available=available.to(self.device)
  return net(obs,torch.zeros(len(obs),1,256,device=self.device),torch.ones(len(obs),1,device=self.device),available,deterministic=deterministic)[:2]
 def distribution(self,i,obs,available):
  n=self.actors[i];obs=obs.to(self.device,dtype=torch.float32);available=available.to(self.device)
  feat=n.base(obs);head=n.act.action_out
  if i==0:
   assert len(head.continuous_heads)==1 and len(head.categorical_heads)==0
   return head.continuous_heads[0](feat,available[:,:3])
  assert len(head.categorical_heads)==1 and len(head.continuous_heads)==0
  return head.categorical_heads[0](feat,available[:,:16,None].transpose(1,2))
 def evaluate_actions(self,i,obs,action,available):
  return self.actors[i].evaluate_actions(obs.to(self.device),torch.zeros(len(obs),1,256,device=self.device),action.to(self.device),torch.ones(len(obs),1,device=self.device),available.to(self.device),torch.ones(len(obs),1,device=self.device))

def simplex(action):
 raw=((action.to(torch.float64)+1)/2).clamp_min(0);total=raw.sum(-1,keepdim=True)
 return torch.where(total>0,raw/total.clamp_min(1e-300),torch.full_like(raw,1/3))
def physical_shares(action):
 p=simplex(action);s=.05+.85*p;return s/s.sum(-1,keepdim=True)
def target_alpha(teacher_action):return 1000*(simplex(teacher_action)+1e-4)/(1+3e-4)
def sut_loss(dist,teacher_action):
 target=target_alpha(teacher_action).to(dist.concentration.device)
 assert dist.safe_mask.all()
 kl=torch.distributions.kl_divergence(torch.distributions.Dirichlet(target),torch.distributions.Dirichlet(dist.concentration.double()))
 assert torch.isfinite(kl).all();return kl
