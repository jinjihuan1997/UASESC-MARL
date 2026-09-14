"""Four fixed, local-observation, deterministic controls with no fitted object."""
from light_support import torch,load_module,GREEDY
ONLINE=load_module('lightweight_frozen_online_policy',GREEDY/'online_policy.py')
METHODS=['G_equal_local16','G_urgency_local16','R_equal_minload','R_equal_maxquality']

def equal_resource(sut_obs):
 # Same float64 action arithmetic as helpers.simple_rule_actions.
 return 2*torch.full((len(sut_obs),3),1/3,dtype=torch.float64,device=sut_obs.device)-1
def urgency_resource(sut_obs):
 urgency=(sut_obs[:,13:16].to(torch.float64)*80).round().clamp_min(0);total=urgency.sum(-1,keepdim=True)
 p=torch.where(total>0,urgency/total.clamp_min(1),torch.full_like(urgency,1/3));return 2*p-1

def allowed_local(local_obs,local_mask):
 o=local_obs.to(torch.float64);q=o[:,1:11]>.5;load=o[:,31:47]*100000;quality=o[:,47:63]*33;budget=o[:,0]*100000;req=o[:,74]*33
 plausible=(quality>=req[:,None]-1e-5)&(load<=budget[:,None]+2e-3)
 # Mask is authoritative. Public tolerances are inherited from UAVGreedy;
 # they only disambiguate its all-one no-feasible fallback, not hidden truth.
 mask=local_mask[:,:16]>.5;all_open=mask.all(-1,keepdim=True)
 valid=mask&torch.where(all_open,plausible,torch.ones_like(plausible))&q.any(-1,keepdim=True)
 return load,quality,valid

class Lightweight:
 def __init__(self,method):
  assert method in METHODS;self.method=method
  self.greedy=ONLINE.UAVGreedy(list(range(16))) if method.startswith('G_') else None
 @torch.inference_mode()
 def resource(self,sut_obs,sut_mask=None):
  return urgency_resource(sut_obs) if self.method=='G_urgency_local16' else equal_resource(sut_obs)
 @torch.inference_mode()
 def mode(self,uav_index,local_obs,local_mask):
  if self.greedy:return self.greedy.act(local_obs,local_mask)
  load,quality,valid=allowed_local(local_obs,local_mask)
  # Stable sorts start with natural mode-ID order. No utility or future state.
  if self.method=='R_equal_minload':
   order=torch.argsort(-quality,dim=-1,stable=True);order=order.gather(-1,torch.argsort(load.gather(-1,order),dim=-1,stable=True))
  else:
   order=torch.argsort(load,dim=-1,stable=True);order=order.gather(-1,torch.argsort(-quality.gather(-1,order),dim=-1,stable=True))
  flags=valid.gather(-1,order);selected=order.gather(-1,flags.long().argmax(-1,keepdim=True)).squeeze(-1)
  selected=torch.where(valid.any(-1),selected,torch.zeros_like(selected))
  return torch.nn.functional.one_hot(selected,16).to(torch.float32)
 def assert_frozen(self):
  assert not hasattr(self,'predictor') and not any(isinstance(x,torch.nn.Module) for x in vars(self).values())
