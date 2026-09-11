from helpers import *
FIELDS=['common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses','instruction_id','quality_violations','budget_violations','cache_violations']
def predict_reward(env,actions):
    """One-slot current-state calculation; no future channel or task tape access."""
    p=env.p;gid,limit,req=env.context()
    shares=((actions[0]+1)/2).clamp_min(0)
    shares=torch.where(shares.sum(-1,keepdim=True)>0,shares/shares.sum(-1,keepdim=True).clamp_min(1e-12),torch.full_like(shares,1/env.U))
    beta=(p.beta_sat_lower_bound+(1-p.beta_sat_lower_bound*env.U)*shares).clamp(p.beta_sat_lower_bound,1.)
    beta=beta/beta.sum(-1,keepdim=True)
    budget=p.delta_T*p.backhaul_availability*torch.minimum(torch.full_like(beta,p.B_uav_sut),beta*p.B_sut_sat)
    mu=torch.stack(actions[1:],1)
    valid=(env.quality>=req[:,:,None]-1e-9)&(env.load<=budget[:,:,None]+1e-9)&env.q.any(-1)[:,:,None]
    order=torch.argsort(-mu,dim=-1,stable=True)
    mode=order.gather(-1,valid.gather(-1,order).long().argmax(-1,keepdim=True)).squeeze(-1)
    load=env.load.gather(-1,mode[:,:,None]).squeeze(-1)
    quality=env.quality.gather(-1,mode[:,:,None]).squeeze(-1)
    ranks=torch.argsort(torch.where(env.q,env.aoi,-torch.inf),dim=-1,descending=True,stable=True)
    cached=env.q.gather(-1,ranks);left=budget.clone();takes=[]
    for k in range(env.K):
        take=valid.any(-1)&cached[:,:,k]&(load<=left+1e-9)
        left-=torch.where(take,load,0.);takes.append(take)
    served=torch.zeros_like(env.q).scatter(-1,ranks,torch.stack(takes,-1))
    count=served.sum(-1)
    next_aoi=torch.where(served,(env.step_index-torch.where(env.tau<0,env.step_index,env.tau)+1).to(env.dtype),env.aoi+1).clamp_max(p.A_max)
    quality_term=(((quality-p.Q_min_eval)/(p.Q_max-p.Q_min_eval)).clamp_min(0)*count).sum(-1)/env.D
    aoi_term=p.aoi_mean_weight*(next_aoi/p.aoi_reward_ref).mean((-1,-2))+p.aoi_max_weight*(next_aoi/p.aoi_reward_ref).amax((-1,-2))+p.aoi_tail_weight*((next_aoi-p.aoi_tail_threshold)/p.aoi_reward_ref).clamp_min(0).mean((-1,-2))
    load_term=(load*count/p.Lambda_ref).sum(-1)/env.U
    w=env.reward_weights[gid]
    return w[:,0]*quality_term-w[:,1]*aoi_term-w[:,2]*load_term


def myopic_actions(env):
    candidates=[rule_actions(env,name) for name in RULES]
    rewards=torch.stack([predict_reward(env,action) for action in candidates])
    chosen=rewards.argmax(0);batch=torch.arange(env.count)
    actions=[torch.stack([action[i] for action in candidates])[chosen,batch] for i in range(env.n_agents)]
    return actions,rewards[chosen,batch]


def summarize(trace):
    rows=trace.reshape(-1,len(FIELDS));delivery=rows[:,3].sum()
    return dict(steps=len(rows),common_reward=float(rows[:,0].mean()),mean_aoi=float(rows[:,1].mean()),
        predicted_quality_sum=float(rows[:,2].sum()),deliveries=int(delivery),channel_uses=float(rows[:,4].sum()),
        delivered_predicted_psnr=float(rows[:,2].sum()/delivery) if delivery else None,
        deliveries_per_slot=float(delivery/len(rows)),channel_uses_per_slot=float(rows[:,4].mean()))
