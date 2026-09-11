import numpy as np
import torch
def arr(t): return t.detach().cpu().numpy()


def explicit_columns(p):
    allowed=np.zeros((1+p.n_uav,p.obs_dim_common),bool)
    allowed[0,p.sut_obs_dim-4:p.sut_obs_dim]=True
    allowed[1:,p.uav_obs_dim-9:p.uav_obs_dim-6]=True
    allowed[1:,p.uav_obs_dim-2:p.uav_obs_dim]=True
    assert p.num_instructions==3 and p.include_instruction_id_in_obs and p.include_instruction_constraints_in_obs
    return allowed


@torch.no_grad()
def actions_for(actors,obs,available):
    E=len(obs);rnn=np.zeros((E,1,256),np.float32);mask=np.ones((E,1),np.float32)
    return [actor.act(obs[:,i],rnn,mask,available[:,i],deterministic=True)[0] for i,actor in enumerate(actors)]


def check_tensor(env,info,slot,before):
    p=env.p
    x={k:arr(v) for k,v in info.items()}
    served=x['served'];old=before['aoi'];after=arr(env.aoi);cache=before['q'];tau=before['tau']
    expected=np.where(served,slot-np.where(tau<0,slot,tau)+1,old+1).clip(max=p.A_max)
    np.testing.assert_allclose(after,expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(arr(env.q),(cache & ~served)|(~cache))
    np.testing.assert_array_equal(arr(env.tau),np.where(~cache,slot,np.where(cache & ~served,tau,-1)))
    assert not np.any(served & ~cache)
    mode=x['mode'];selected=np.take_along_axis(x['load_table'],mode.clip(min=0)[:,:,None],-1).squeeze(-1)
    usage=selected*served.sum(-1)
    np.testing.assert_allclose(usage,x['usage'],rtol=0,atol=1e-8)
    assert not np.any(usage>x['budget']+1e-8)
    assert not np.any(served & (x['quality']<x['req']-1e-8)[:,:,None])
    gid=x['gid'].astype(int);limits=np.asarray(p.A_limit_by_instruction)[gid]
    weights=np.asarray(p.reward_weights_by_instruction)[gid]
    qgain=(np.maximum((x['quality']-p.Q_min_eval)/(p.Q_max-p.Q_min_eval),0)*served.sum(-1)).sum(-1)/p.n_ds
    flat=after.reshape(env.count,p.n_ds)
    aoi=p.aoi_mean_weight*np.mean(flat/p.aoi_reward_ref,-1)+p.aoi_max_weight*np.max(flat/p.aoi_reward_ref,-1)+p.aoi_tail_weight*np.mean(np.maximum((flat-p.aoi_tail_threshold)/p.aoi_reward_ref,0),-1)
    base=weights[:,0]*qgain-weights[:,2]*p.reward_load_scale*usage.sum(-1)/p.reward_load_ref-weights[:,1]*aoi
    violation=np.maximum((flat.max(-1)-limits)/limits,0)
    bonus=p.eta_recv_aoi_bonus*np.maximum(old-after,0).mean((-1,-2))/limits
    common=base-np.asarray(p.constraint_penalty_A_by_instruction)[gid]*violation+bonus
    np.testing.assert_allclose(common,x['common_reward'],atol=1e-9,rtol=0)
    zeros=np.zeros(env.count,dtype=int)
    return dict(common_reward=common,base_reward=base,training_reward=x['reward'],recv_aoi_bonus=bonus,
        objective_tail10=common+weights[:,1]*p.aoi_tail_weight*(np.maximum(flat-p.aoi_tail_threshold,0).mean(-1)-np.maximum(flat-10,0).mean(-1))/p.aoi_reward_ref,
        objective_tail4=common+weights[:,1]*p.aoi_tail_weight*(np.maximum(flat-p.aoi_tail_threshold,0).mean(-1)-np.maximum(flat-4,0).mean(-1))/p.aoi_reward_ref,
        fraction_above4=(flat>4).mean(-1),fraction_above6=(flat>6).mean(-1),
        quality_credit=weights[:,0]*qgain,
        age_mean_cost=weights[:,1]*p.aoi_mean_weight*flat.mean(-1)/p.aoi_reward_ref,
        age_max_cost=weights[:,1]*p.aoi_max_weight*flat.max(-1)/p.aoi_reward_ref,
        age_tail_cost=weights[:,1]*p.aoi_tail_weight*np.maximum(flat-p.aoi_tail_threshold,0).mean(-1)/p.aoi_reward_ref,
        resource_cost=weights[:,2]*p.reward_load_scale*usage.sum(-1)/p.reward_load_ref,
        service_violation_cost=np.asarray(p.constraint_penalty_A_by_instruction)[gid]*violation,
        mean_aoi=flat.mean(-1),max_aoi=flat.max(-1),p95_aoi=np.quantile(flat,.95,axis=-1),
        aoi_exceedance_fraction=(flat>limits[:,None]).mean(-1),max_aoi_violation=violation,
        deliveries=served.sum((-1,-2)),predicted_quality_sum=(x['quality']*served.sum(-1)).sum(-1),
        channel_uses=usage.sum(-1),quality_violations=zeros,budget_violations=zeros,cache_violations=zeros,
        infeasible_uav_fraction=(mode<0).mean(-1))

