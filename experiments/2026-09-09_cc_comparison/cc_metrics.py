"""Independent NumPy checks of CC transitions and the shared SC objective."""
import hashlib
import numpy as np
from common import ROOT
from multiseed_protocol import read
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv


def array(tensor): return tensor.detach().cpu().numpy()


def external_reference(config,seed,schedule):
    args=read(ROOT/'inputs/parent_IC_HAPPO.json')['env_args']
    args.update(semantic_registry_path=str(ROOT/'reference/inputs/mode_registry.json'),
                semantic_profile_path=str(ROOT/'reference/inputs/profile.npz'),
                instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
    env=SCUAVEnv(args);env.seed(seed);env.reset()
    trace=hashlib.sha256()
    for value in (env.pos_ds,env.pos_uav,env.owner_uav,env.q_cache,env.tau_cache): trace.update(value.tobytes())
    return env,trace


def metrics(env,info,slot):
    p=env.p
    def a(k): return array(info[k])[0]
    attempted,served=a('attempted'),a('served')
    mode=a('mode');q=a('quality');expected_q=a('expected_quality')
    before,after=a('old_aoi'),a('next_aoi')
    cache,tau=a('cache_before'),a('tau_before')
    gid=int(a('gid'));limit=float(p.A_limit_by_instruction[gid]);req=a('req')
    assert not np.any(served & ~attempted)
    assert not np.any(attempted & ~cache)
    expected_aoi=np.where(served,slot-np.where(tau<0,slot,tau)+1,before+1).clip(max=p.A_max)
    np.testing.assert_allclose(after,expected_aoi,atol=1e-10,rtol=0)
    expected_cache=(cache & ~attempted)|(~cache)
    expected_tau=np.where(~cache,slot,np.where(cache & ~attempted,tau,-1))
    np.testing.assert_array_equal(array(env.q)[0],expected_cache)
    np.testing.assert_array_equal(array(env.tau)[0],expected_tau)
    loads=a('load_table')[np.arange(p.n_uav),mode.clip(min=0)]
    use=loads*attempted.sum(-1)
    np.testing.assert_allclose(use,a('usage'),atol=1e-8,rtol=0)
    assert not np.any(use>a('budget')+1e-8)
    assert not np.any(attempted & (expected_q<req-1e-8)[:,None])
    qgain=float((np.maximum((q-p.Q_min_eval)/(p.Q_max-p.Q_min_eval),0)*served.sum(-1)).sum()/p.n_ds)
    # Every measured success-quality sample is above Q_min_eval (verified separately).
    np.testing.assert_allclose(qgain,float(a('quality_term')),atol=1e-10,rtol=0)
    cost=float(use.sum()/p.Lambda_ref/p.n_uav)
    aoi=p.aoi_mean_weight*np.mean(after/p.aoi_reward_ref)+p.aoi_max_weight*np.max(after/p.aoi_reward_ref)+p.aoi_tail_weight*np.mean(np.maximum((after-p.aoi_tail_threshold)/p.aoi_reward_ref,0))
    wq,wa,wl=p.reward_weights_by_instruction[gid]
    base=wq*qgain-wl*cost-wa*aoi
    excess=max(0.,(float(after.max())-limit)/limit)
    bonus=p.eta_recv_aoi_bonus*np.mean(np.maximum(before-after,0))/limit
    common=base-p.constraint_penalty_A_by_instruction[gid]*excess+bonus
    np.testing.assert_allclose(common,float(a('common_reward')),atol=1e-9,rtol=0)
    return dict(common_reward=float(common),base_reward=float(base),training_reward=float(a('reward')),
        recv_aoi_bonus=float(bonus),mean_aoi=float(after.mean()),max_aoi=float(after.max()),p95_aoi=float(np.quantile(after,.95)),
        aoi_exceedance_fraction=float(np.mean(after>limit)),max_aoi_violation=float(excess),
        deliveries=int(served.sum()),attempts=int(attempted.sum()),failed_packets=int((attempted & ~served).sum()),
        predicted_quality_sum=float((q*served.sum(-1)).sum()),channel_uses=float(use.sum()),
        quality_violations=0,budget_violations=0,cache_violations=0,infeasible_uav_fraction=float(np.mean(mode<0)))
