"""Analytic rollout-old KL and atomic parameter+Adam candidate rollback."""
from study import *

def distribution_snapshot(student,i,obs,available):
 dist=student.distribution(i,obs,available)
 if i==0:return dict(kind='dirichlet',alpha=dist.concentration.detach().double().clone(),mask=dist.safe_mask.detach().clone())
 return dict(kind='categorical',logits=dist.logits.squeeze(1).detach().double().clone(),mask=(available[:,:16]>.5).detach().clone())

def analytic_kl(old,student,i,obs,available):
 dist=student.distribution(i,obs,available)
 if i==0:
  assert torch.equal(old['mask'],dist.safe_mask);alpha=dist.concentration.double();mask=old['mask']
  # Every SUT sample has three active resource coordinates in this experiment.
  assert mask.all();v=torch.distributions.kl_divergence(torch.distributions.Dirichlet(old['alpha']),torch.distributions.Dirichlet(alpha))
 else:
  assert torch.equal(old['mask'],available[:,:16]>.5)
  v=torch.distributions.kl_divergence(torch.distributions.Categorical(logits=old['logits']),torch.distributions.Categorical(logits=dist.logits.squeeze(1).double()))
 assert torch.isfinite(v).all();assert v.min()>=-1e-9
 return v

def summarize_kl(values,gids,limit=.01):
 def stat(v):return dict(count=len(v),mean=float(v.mean()) if len(v) else None,p95=float(torch.quantile(v,.95)) if len(v) else None,p99=float(torch.quantile(v,.99)) if len(v) else None)
 d=dict(overall=stat(values),by_instruction={str(g):stat(values[gids==g]) for g in range(3)})
 d['exceeds']=d['overall']['mean']>limit or any(x['count']>=64 and x['mean']>limit for x in d['by_instruction'].values())
 return d

class Candidate:
 def __init__(self,net,optimizer):self.net=net;self.optimizer=optimizer;self.parameters=copy.deepcopy(net.state_dict());self.adam=copy.deepcopy(optimizer.state_dict())
 def rollback(self):
  self.net.load_state_dict(self.parameters,strict=True);self.optimizer.load_state_dict(self.adam)
  assert state_equal(self.parameters,self.net.state_dict()) and state_equal(self.adam,self.optimizer.state_dict())
