"""Finite-input native HAPPO objective; fail loudly instead of hiding NaNs."""
from study import *
from harl.utils.ratio_tools import aggregate_action_ratio

def strict_actor_step(actor,sample):
 obs,rnn,action,mask,active,oldlog,adv,available,factor=sample
 for x in sample:assert torch.isfinite(torch.as_tensor(x)).all()
 oldlog=oldlog.to(**actor.tpdv);adv=adv.to(**actor.tpdv);active=active.to(**actor.tpdv);factor=factor.to(**actor.tpdv)
 lp,entropy,_=actor.evaluate_actions(obs,rnn,action,mask,available,active)
 assert torch.isfinite(lp).all() and torch.isfinite(entropy)
 ratio=aggregate_action_ratio(lp-oldlog,actor.action_aggregation,clip=20.0).clamp(0,1e3)
 assert torch.isfinite(ratio).all()
 s1=ratio*adv;s2=torch.clamp(ratio,1-actor.clip_param,1+actor.clip_param)*adv
 if actor.use_policy_active_masks:loss=(-torch.sum(factor*torch.min(s1,s2),dim=-1,keepdim=True)*active).sum()/active.sum().clamp(min=1e-6)
 else:loss=-torch.sum(factor*torch.min(s1,s2),dim=-1,keepdim=True).mean()
 total=loss-entropy*actor.entropy_coef;assert torch.isfinite(total)
 actor.actor_optimizer.zero_grad();total.backward()
 grad=torch.nn.utils.clip_grad_norm_(actor.actor.parameters(),actor.max_grad_norm,error_if_nonfinite=True)
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.actor.parameters())
 actor.actor_optimizer.step()
 assert all(torch.isfinite(p).all() for p in actor.actor.parameters())
 return dict(policy_loss=float(loss.detach()),entropy=float(entropy.detach()),gradient_norm=float(grad),ratio_mean=float(ratio.detach().mean()),clip_fraction=float(((ratio.detach()-1).abs()>actor.clip_param).float().mean()))
