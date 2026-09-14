"""Frozen observation-only teacher. Supervision adapter does not control student."""
from study import *

class Teacher:
 def __init__(self):
  m=manifest();assert m['teacher']['method']=='greedy_modes_16'
  assert sha(GREEDY/'online_policy.py')==m['teacher']['source_sha256']
  path=ROOT/m['teacher']['predictor'];assert sha(path)==m['teacher']['predictor_sha256']
  module=load_module('frozen_teacher_online_policy',GREEDY/'online_policy.py')
  self.sut=module.SUTGreedy(module.TreeScorePredictor(path));self.uav=module.UAVGreedy(list(range(16)))
  self.queries=[0]*4;self.initial_hash=self.state_hash()
 def state_hash(self):
  h=hashlib.sha256()
  for obj in [self.sut.predictor,self.uav]:
   for k,v in sorted(vars(obj).items()):
    if isinstance(v,torch.Tensor):h.update(k.encode());h.update(arr(v).tobytes())
  return h.hexdigest()
 def assert_frozen(self):assert self.state_hash()==self.initial_hash
 @torch.no_grad()
 def resource(self,sut_obs):
  assert sut_obs.ndim==2 and sut_obs.shape[-1]==76 and sut_obs.device.type=='cpu'
  before=sut_obs.clone();act,_=self.sut.act(sut_obs);assert torch.equal(before,sut_obs)
  self.queries[0]+=len(sut_obs);return act
 @torch.no_grad()
 def mode_label(self,uav,local_obs,local_mask):
  assert local_obs.shape[-1]==76 and local_obs.ndim==2 and local_mask.shape==(len(local_obs),16)
  before=local_obs.clone();bm=local_mask.clone();requested=self.uav.act(local_obs,local_mask).argmax(-1)
  assert torch.equal(before,local_obs) and torch.equal(bm,local_mask)
  self.queries[uav+1]+=len(local_obs)
  mask=local_mask>.5;q=local_obs[:,1:11]>.5
  # The original mask is authoritative. An all-one fallback is disambiguated
  # conservatively using only the actor's existing float32 public fields.
  o=local_obs.double();load=o[:,31:47]*100000;quality=o[:,47:63]*33
  budget=o[:,0]*100000;req=o[:,74]*33
  clear=(load<budget[:,None]-2e-3)&(quality>req[:,None]+1e-5)
  all_open=mask.all(-1);has_cache=q.any(-1)
  active=has_cache&((~all_open)|clear.any(-1))
  plausible=(load<=budget[:,None]+2e-3)&(quality>=req[:,None]-1e-5)
  reason=torch.zeros(len(o),dtype=torch.int8)
  reason[~has_cache]=1
  reason[has_cache&all_open&~plausible.any(-1)]=2
  reason[has_cache&all_open&plausible.any(-1)&~clear.any(-1)]=3
  effective=torch.where(mask.gather(-1,requested[:,None]).squeeze(-1),requested,mask.long().argmax(-1))
  assert mask.gather(-1,effective[:,None]).squeeze(-1).all()
  return requested,effective,active,reason
