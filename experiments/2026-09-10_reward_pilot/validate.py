"""Independent physics/formula, fixed old trajectories and reward routing checks."""
from helpers import *
from rule_tools import myopic_actions,predict_reward
from evaluate import external_metrics


def main():
    cpu_slots=0;replay_slots=0;edge_cases=0
    for objective in OBJECTIVES:
        cfg=config(objective);schedule=[[0,0],[200,1],[400,2]]
        seeds=[20262101,20262102]
        tensor,_,_,_=make_env(cfg,seeds,schedule)
        refs=[]
        for seed in seeds:
            args=copy.deepcopy(cfg['env_args']);args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=schedule)
            ref=SCUAVEnv(args);ref.seed(seed);ref.reset();refs.append(ref)
        for slot in range(600):
            actions=rule_actions(tensor,RULES[(slot//31)%len(RULES)])
            predicted=arr(predict_reward(tensor,actions))
            _,_,_,info,values=checked_step(tensor,actions)
            np.testing.assert_allclose(predicted,values['common_reward'],atol=1e-9,rtol=0)
            for i,ref in enumerate(refs):
                _,_,reward,_,infos,_=ref.step([arr(a)[i] for a in actions])
                actual=external_metrics(ref,infos[0])
                for key in ('common_reward','mean_aoi','deliveries','predicted_quality_sum','channel_uses'):
                    np.testing.assert_allclose(actual[key],values[key][i],atol=1e-7,rtol=1e-10,err_msg=key)
                np.testing.assert_allclose(np.asarray(reward).ravel(),actual['common_reward'],atol=1e-6,rtol=1e-6)
                cpu_slots+=1
        full,fo,fs,fm=make_env(cfg,seeds,schedule)
        hidden,ho,hs,hm=make_env(config(objective,True),seeds,schedule)
        allowed=torch.as_tensor(explicit_columns(full.p))
        assert torch.equal(fo[:,~allowed],ho[:,~allowed]) and torch.equal(fs,hs) and torch.equal(fm,hm)
        assert torch.count_nonzero(ho[:,allowed])==0
        feeds=[];masks=[]
        for g in range(3):
            hidden.instructions[0]=g;o,_,m=hidden.observe();feeds.append(o);masks.append(m)
        assert all(torch.equal(feeds[0],o) and torch.equal(masks[0],m) for o,m in zip(feeds,masks))
        for case in ('nonzero_penalty','empty','quality_infeasible','budget_infeasible','threshold_boundary'):
            args=copy.deepcopy(cfg['env_args']);args.update(instruction_mode_strategy='explicit_evaluation',explicit_instruction_schedule=[[0,0]])
            if case=='nonzero_penalty':args.update(constraint_penalty_A_by_instruction=[.1]*3,eta_recv_aoi_bonus=.05)
            if case=='quality_infeasible':args['Q_req_by_instruction_snr_bucket']=[[100]*4 for _ in range(3)]
            if case=='budget_infeasible':args['backhaul_availability']=1e-6
            env=TensorSCEnv(args,count=2,seed=20262109,device='cpu');env.reset()
            if case in ('nonzero_penalty','empty'):env.aoi.fill_(12)
            if case=='empty':env.q.zero_()
            if case=='threshold_boundary':env.aoi.fill_(env.p.aoi_tail_threshold)
            actions,pred=myopic_actions(env)
            _,_,_,info,v=checked_step(env,actions)
            np.testing.assert_allclose(arr(pred),v['common_reward'],atol=1e-9,rtol=0)
            if case=='nonzero_penalty':assert np.all(v['service_violation_cost']>0)
            if case in ('empty','quality_infeasible','budget_infeasible'):assert not np.any(v['deliveries'])
            edge_cases+=1
        print('CPU/tensor, input and edge checks PASS:',objective,flush=True)
    # Reproduce actual old outcomes with the equivalent original objective.
    for seed,scenario in [(85,'fixed_2'),(218,'fixed_2'),(966,'fixed_2'),(85,'switch300_0_to_1')]:
        item=f'seed_{seed}/IC_HAPPO';folder=PARENT/'evaluation'/item
        meta=read(folder/'summary.json');path=folder/f'{scenario}.npz'
        assert digest(path)==meta['traces'][scenario]
        with np.load(path) as z:data=z['trace'];modes=z['modes'];fractions=z['resource_fractions'];seeds=z['seeds'].tolist()
        env,_,_,_=make_env(config('original_tail10'),seeds,read(PARENT/'manifest.json')['scenarios'][scenario])
        assert external_hashes(env)==meta['pairing'][scenario]
        for slot in range(600):
            share=(fractions[slot]-env.p.beta_sat_lower_bound)/(1-env.p.beta_sat_lower_bound*env.U)
            mu=np.eye(env.M)[modes[slot].clip(min=0)]
            actions=[env.tensor(2*share-1)]+[env.tensor(mu[:,i]) for i in range(env.U)]
            _,_,_,info,v=checked_step(env,actions)
            actual=np.column_stack([v[k] for k in ('common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses')])
            np.testing.assert_allclose(actual,data[slot,:,:5],atol=1e-8,rtol=1e-10)
            np.testing.assert_array_equal(arr(info['mode']),modes[slot]);replay_slots+=len(seeds)
    # Reward-specific monotonic examples: fixed other inputs, more age/load is worse.
    p=make_env(config('candidate_tail4'),[20262111])[0].p
    before=np.full(30,5.);after=np.full(30,4.)
    base=reward_components(p,10.,10000.,before,after,2)['objective_reward']
    assert reward_components(p,11.,10000.,before,after,2)['objective_reward']>base
    assert reward_components(p,10.,11000.,before,after,2)['objective_reward']<base
    assert reward_components(p,10.,10000.,before,after+1,2)['objective_reward']<base
    # Tail change only adds its stated penalty under an otherwise identical transition.
    a=make_env(config('original_tail10'),[20262113])[0]
    b=make_env(config('candidate_tail4'),[20262113])[0]
    a.aoi.fill_(8);b.aoi.fill_(8)
    acts=rule_actions(a,'m5_equal')
    _,_,_,ia,va=checked_step(a,acts);_,_,_,ib,vb=checked_step(b,acts)
    assert torch.equal(a.aoi,b.aoi) and torch.equal(a.q,b.q)
    expected=-.4*.3*(np.maximum(arr(a.aoi)-4,0)-np.maximum(arr(a.aoi)-10,0)).mean((-1,-2))/10
    np.testing.assert_allclose(vb['common_reward']-va['common_reward'],expected,atol=1e-12,rtol=0)
    write(HERE/'validation.json',dict(state='PASS',cpu_tensor_slots=cpu_slots,frozen_old_replay_slots=replay_slots,
        edge_cases=edge_cases,shared_reward_routing_nonzero_penalties=True,independent_recomputation=True,
        hidden_inputs_and_critic_pairing=True,original_objective_equivalence=True,monotonic_examples=3,
        candidate_only_stated_tail_change=True))
    print('PASS all reward/physics checks',flush=True)


if __name__=='__main__':main()
