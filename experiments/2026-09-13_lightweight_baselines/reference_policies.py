"""Unmodified fitted greedy, original simple helpers and fixed old RL weights."""
from light_support import *
from lightweight_policies import Lightweight,METHODS,ONLINE,equal_resource

class Reference:
 def __init__(self,method,env,parent=None):
  self.method=method;self.parent=parent;self.nets=[];self.greedy=None;self.composition=None
  if method.startswith('greedy_'):
   variant=method.removeprefix('greedy_');assert variant in ['modes_16','modes_3'];p=GREEDY/variant/'predictor.npz';assert sha(p)==read(GREEDY/variant/'fit.json')['predictor_sha256']
   self.sut=ONLINE.SUTGreedy(ONLINE.TreeScorePredictor(p));self.greedy=ONLINE.UAVGreedy(list(range(16)) if variant=='modes_16' else [0,5,10])
  elif method in ['C1','C3']:
   assert parent in manifest()['parents'];self.composition=Composition(parent,env,method)
  elif method=='original_rl':
   assert parent in manifest()['parents'];params={**cfg(parent)['algo_args']['model'],**cfg(parent)['algo_args']['algo']};folder=old.model_dir(parent);status=read(folder/'status.json')
   assert status['completed_steps']==1000000 and status['state']=='complete'
   for i in range(4):
    net=StochasticPolicy(params,env.observation_space[i],env.action_space[i],torch.device('cpu'));net.load_state_dict(torch.load(folder/f'actor_agent{i}.pt',weights_only=True,map_location='cpu'),strict=True);net.eval().requires_grad_(False);assert network_hash(net)==status['actor_hashes'][i];self.nets.append(net)
   self.initial_hashes=[network_hash(n) for n in self.nets]
  else:assert method in frozen.RULE_METHODS
 @torch.inference_mode()
 def resource(self,sut_obs,sut_mask):
  if self.greedy:return self.sut.act(sut_obs)[0]
  if self.composition:return self.composition.resource(sut_obs,sut_mask,equal_resource(sut_obs))[0]
  if self.nets:return self.call(self.nets[0],sut_obs,sut_mask)
  # Exact original helper, called once per slot; reuse its original mode outputs.
  spec=types.SimpleNamespace(dtype=torch.float64,U=3,M=16)
  acts=frozen.simple_rule_actions(spec,self.method,sut_obs[:,None]);self.rule_modes=acts[1:];return acts[0]
 @torch.inference_mode()
 def mode(self,uav_index,local_obs,local_mask):
  if self.greedy:return self.greedy.act(local_obs,local_mask)
  if self.composition:return self.composition.mode(uav_index,local_obs,local_mask)[0]
  if self.nets:return self.call(self.nets[uav_index+1],local_obs,local_mask)
  return self.rule_modes[uav_index]
 def call(self,net,obs,mask):return net(obs,torch.zeros(len(obs),1,256),torch.ones(len(obs),1),mask,deterministic=True)[0]
 def assert_frozen(self):
  if self.nets:
   assert self.initial_hashes==[network_hash(n) for n in self.nets]
   assert all(not n.training and not any(p.requires_grad for p in n.parameters()) for n in self.nets)
  if self.composition:self.composition.assert_frozen()

def policy(method,env,parent=None):return Lightweight(method) if method in METHODS else Reference(method,env,parent)

def tensors(obj):
 seen=set();out=[]
 def visit(x):
  if id(x) in seen:return
  seen.add(id(x))
  if isinstance(x,(torch.Tensor,np.ndarray)):out.append(x)
  elif isinstance(x,dict):
   for k,v in x.items():
    if k!='rule_modes':visit(v)
  elif isinstance(x,(tuple,list)):
   for v in x:visit(v)
  elif hasattr(x,'__dict__') and not isinstance(x,(types.ModuleType,type)):visit(vars(x))
 visit(obj);return out
def policy_hash(obj):
 h=hashlib.sha256()
 for t in tensors(obj):
  a=arr(t) if isinstance(t,torch.Tensor) else t;h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
 return h.hexdigest()
