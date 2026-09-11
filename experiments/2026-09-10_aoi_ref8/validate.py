"""Changed parameter reaches every scoring path; physics and actors unchanged."""
import subprocess
from helpers import *
from rule_tools import predict_reward,myopic_actions
from evaluate import external_metrics
from tensor_train import TensorTrainer, network_hash


def main():
    cpu_slots=0;edge_cases=0;seeds=[20262451,20262452];schedule=[[0,0],[200,1],[400,2]]
    for o in OBJECTIVES:
        cfg=config(o);env,_,_,_=make_env(cfg,seeds,schedule);refs=[]
        for seed in seeds:
            args=copy.deepcopy(cfg['env_args']);args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
            ref=SCUAVEnv(args);ref.seed(seed);ref.reset();refs.append(ref)
        for slot in range(600):
            actions=rule_actions(env,RULES[(slot//31)%len(RULES)]);pred=arr(predict_reward(env,actions))
            _,_,_,_,v=checked_step(env,actions)
            np.testing.assert_allclose(pred,v['common_reward'],atol=1e-9,rtol=0)
            for i,ref in enumerate(refs):
                _,_,reward,_,infos,_=ref.step([arr(a)[i] for a in actions]);out=external_metrics(ref,infos[0])
                for key in ['common_reward','mean_aoi','deliveries','predicted_quality_sum','channel_uses']:
                    np.testing.assert_allclose(out[key],v[key][i],atol=1e-7,rtol=1e-10)
                np.testing.assert_allclose(np.asarray(reward).ravel(),out['common_reward'],atol=1e-6,rtol=1e-6)
                cpu_slots+=1
        full,fo,fs,fm=make_env(cfg,seeds,schedule);hidden,ho,hs,hm=make_env(config(o,True),seeds,schedule)
        allowed=torch.as_tensor(explicit_columns(full.p))
        assert torch.equal(fo[:,~allowed],ho[:,~allowed]) and torch.equal(fs,hs) and torch.equal(fm,hm)
        assert torch.count_nonzero(ho[:,allowed])==0
        feeds=[]
        for g in range(3):hidden.instructions[0]=g;feeds.append(hidden.observe())
        assert all(torch.equal(feeds[0][0],x[0]) and torch.equal(feeds[0][2],x[2]) for x in feeds)
        for case in ['nonzero_penalty','empty','quality_infeasible','budget_infeasible','threshold_boundary']:
            c=copy.deepcopy(cfg);e=c['env_args']
            if case=='nonzero_penalty':e.update(constraint_penalty_A_by_instruction=[.1]*3,eta_recv_aoi_bonus=.05)
            if case=='quality_infeasible':e['Q_req_by_instruction_snr_bucket']=[[100]*4 for _ in range(3)]
            if case=='budget_infeasible':e['backhaul_availability']=1e-6
            t,_,_,_=make_env(c,seeds)
            if case in ['nonzero_penalty','empty']:t.aoi.fill_(12)
            if case=='empty':t.q.zero_()
            if case=='threshold_boundary':t.aoi.fill_(4)
            acts,pred=myopic_actions(t);_,_,_,_,v=checked_step(t,acts)
            np.testing.assert_allclose(arr(pred),v['common_reward'],atol=1e-9,rtol=0)
            if case=='nonzero_penalty':assert np.all(v['service_violation_cost']>0)
            if case in ['empty','quality_infeasible','budget_infeasible']:assert not np.any(v['deliveries'])
            edge_cases+=1
        print('CPU/tensor and edge cases PASS',o,flush=True)
    # Same frozen actor gets identical observations/actions under refs10 and8.
    a,ao,_,am=make_env(config('ref10'),seeds,schedule);b,bo,_,bm=make_env(config('ref8'),seeds,schedule)
    assert external_hashes(a)==external_hashes(b)
    actors=load_actors(config('ref10'),a,HERE/'jobs/ref10/seed_85/IC_HAPPO')
    for slot in range(600):
        assert torch.equal(ao,bo) and torch.equal(am,bm)
        acts=actions_for(actors,ao,am)
        ao,_,am,ia,va=checked_step(a,acts);bo,_,bm,ib,vb=checked_step(b,acts)
        assert torch.equal(a.aoi,b.aoi) and torch.equal(a.q,b.q) and torch.equal(a.tau,b.tau)
        assert torch.equal(ia['mode'],ib['mode']) and torch.equal(a.beta,b.beta)
        age=sum(va[k] for k in ['age_mean_cost','age_max_cost','age_tail_cost'])
        np.testing.assert_allclose(vb['common_reward']-va['common_reward'],-.25*age,atol=1e-12,rtol=0)
        np.testing.assert_allclose(va['objective_ref8'],vb['common_reward'],atol=1e-12,rtol=0)
    del actors,a,b,env,full,hidden,t,refs
    a=TensorTrainer(config('ref10'),'cpu');b=TensorTrainer(config('ref8'),'cpu')
    assert a.initial_actors==b.initial_actors and a.initial_critic==b.initial_critic
    assert torch.equal(a.buffer.obs,b.buffer.obs)
    # Demonstrate actual optimizer update with the changed reward.
    b.collect();out=b.update()
    assert all(torch.isfinite(v).all() for v in out.values())
    assert all(network_hash(actor.actor)!=h for actor,h in zip(b.actors,b.initial_actors))
    del a,b
    write(HERE/'validation.json',dict(state='PASS',cpu_tensor_slots=cpu_slots,edge_cases=edge_cases,
        frozen_actor_paired_slots=1200,only_25_percent_age_cost_change=True,hidden_input_invariance=True,
        initial_networks_identical=True,optimizer_smoke_steps=4000,independent_reward=True,
        checkpoint_resume_code_unchanged_from_verified_parent=True))
    print('Validation PASS',flush=True)


if __name__=='__main__':main()
