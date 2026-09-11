"""Held-out paired evaluation, including task-informed rule baselines."""
import csv
from probe_common import *

METHODS=['IC_HAPPO','HAPPO_hidden_instruction','R_single','R_instruction','R_myopic']
FIELDS=['common_reward','mean_aoi','predicted_quality_sum','deliveries','channel_uses','instruction_id',
        'quality_violations','budget_violations','cache_violations']


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


def evaluate():
    manifest=read(HERE/'pilot_manifest.json');selection=read(HERE/'gate_selection.json')
    for f,h in manifest['source_hashes'].items(): assert digest(HERE/f)==h,f
    verify_parent();results={};pairing={};hashes={};all_stats={}
    seeds=manifest['evaluation_seeds']
    for method in METHODS:
        results[method]={};all_stats[method]={};actors=None
        hidden=method=='HAPPO_hidden_instruction'
        if method.startswith(('IC_','HAPPO_')):
            folder=HERE/'pilot'/method;model=read(folder/'status.json')
            assert model['state']=='complete' and model['completed_steps']==1_000_000
            for f,h in model['checkpoint_hashes'].items(): assert digest(folder/f)==h
        for scenario,schedule in manifest['scenarios'].items():
            env,obs,_,masks=make_env(seeds,schedule,hidden)
            external=external_hashes(env)
            if scenario in pairing: assert pairing[scenario]==external
            else: pairing[scenario]=external
            if actors is None and method.startswith(('IC_','HAPPO_')):
                actors=load_actors(read(HERE/'pilot'/'configs'/f'{method}.json'),env,HERE/'pilot'/method)
            records=[];modes=[];betas=[]
            for slot in range(600):
                predicted=None
                if actors is not None:
                    actions=actions_for(actors,obs,masks)
                elif method=='R_single':
                    actions=rule_actions(env,selection['single'])
                elif method=='R_instruction':
                    gid=int(env.context()[0][0]);assert bool((env.context()[0]==gid).all())
                    actions=rule_actions(env,selection['by_instruction'][gid])
                else:
                    actions,predicted=myopic_actions(env)
                obs,_,masks,info,metrics=checked_step(env,actions)
                if predicted is not None:
                    np.testing.assert_allclose(arr(predicted),metrics['common_reward'],atol=1e-9,rtol=0)
                row=dict(metrics,instruction_id=arr(info['gid']))
                records.append(np.column_stack([row[f] for f in FIELDS]))
                modes.append(arr(info['mode']));betas.append(arr(env.beta))
            trace=np.stack(records)
            assert not np.any(trace[:,:,6:])
            file=HERE/'evaluation'/method/f'{scenario}.npz';file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(file,trace=trace,fields=np.asarray(FIELDS),seeds=np.asarray(seeds),
                modes=np.stack(modes),resource_fractions=np.stack(betas))
            hashes[str(file.relative_to(HERE))]=digest(file)
            with np.load(file) as data: np.testing.assert_array_equal(trace,data['trace'])
            all_stats[method][scenario]=trace
            results[method][scenario]=dict(overall=summarize(trace),
                by_evaluation_seed=[summarize(trace[:,i,:]) for i in range(len(seeds))],
                by_instruction={str(g):summarize(trace[trace[:,:,5]==g]) for g in range(3) if np.any(trace[:,:,5]==g)})
            write(HERE/'PILOT_STATUS.json',dict(state='evaluating',method=method,scenario=scenario,updated_utc=stamp()))
            print(f'{method} {scenario} complete',flush=True)
    grouped={}
    for method,scenarios in all_stats.items():
        trace=np.concatenate(list(scenarios.values()))
        grouped[method]=dict(overall=summarize(trace),
            by_instruction={str(g):summarize(trace[trace[:,:,5]==g]) for g in range(3)})
    paired={}
    for other in METHODS[1:]:
        values=[]
        for scenario in manifest['scenarios']:
            values.append(all_stats['IC_HAPPO'][scenario][:,:,0].mean(0)-all_stats[other][scenario][:,:,0].mean(0))
        a=np.stack(values);by_seed=a.mean(0)
        paired[other]=dict(mean=float(a.mean()),positive_evaluation_seeds=int((by_seed>0).sum()),
            per_evaluation_seed=by_seed.tolist(),by_scenario={s:float(v.mean()) for s,v in zip(manifest['scenarios'],a)})
    audit=dict(state='PASS',episodes=1300,slots=780000,paired_external_trajectories=True,
        all_constraints_checked=True,all_rewards_recomputed=True,myopic_predictions_checked=True,
        one_training_seed=85,steps_per_learning_method=1_000_000,trace_hashes=hashes,
        pairing=pairing)
    write(HERE/'pilot_results.json',dict(per_scenario=results,grouped=grouped,paired_ic_minus=paired))
    write(HERE/'pilot_audit.json',audit)
    lines=['偏好任务短训练与评估已完成。两种方法均从头训练 100 万步，训练种子 85，使用最后一次模型。结果只用于下一阶段决策，不宣称跨训练种子的稳定优势。','',
        '本轮与原任务不同：三个指令共享质量底线、动作范围与执行规则，只改变奖励偏好。所有方法配对相同外部环境；知道指令的规则也参与对比。','',
        '| 方法 | 平均奖励 | 平均 AoI | 交付 PSNR | 更新/槽 |','|---|---:|---:|---:|---:|']
    for name in METHODS:
        v=grouped[name]['overall']
        lines.append(f'| {name} | {v["common_reward"]:.6f} | {v["mean_aoi"]:.4f} | {v["delivered_predicted_psnr"]:.4f} | {v["deliveries_per_slot"]:.4f} |')
    lines+=['','R_single 为校准集选出的单一规则；R_instruction 按真实指令选择校准规则；R_myopic 知道真实偏好，在 9 个当前候选动作中选下一槽奖励最高者。它们不读取未来信道。','',
        '| IC_HAPPO 减对照 | 平均奖励差 | 正差评估种子/20 |','|---|---:|---:|']
    for name,v in paired.items(): lines.append(f'| {name} | {v["mean"]:+.6f} | {v["positive_evaluation_seeds"]}/20 |')
    lines+=['','这些是同一训练种子下的外部环境配对结果，20 个评估种子不能当作 20 个独立训练种子。共同奖励只能在相同任务条件下比较。','',
        '| 方法 | 真实偏好 | 平均奖励 | AoI | 交付 PSNR |','|---|---|---:|---:|---:|']
    for name in METHODS:
        for g,label in enumerate(['均衡','AoI','质量']):
            v=grouped[name]['by_instruction'][str(g)]
            lines.append(f'| {name} | {label} | {v["common_reward"]:.6f} | {v["mean_aoi"]:.4f} | {v["delivered_predicted_psnr"]:.4f} |')
    lines+=['','已完成 1300 个评估回合、78 万槽，逐槽核算奖励及资源/缓存/质量约束，所有方法外生条件配对。平均表与原正式版本未修改。',
        '',f'证据：[完整结果]({HERE}/pilot_results.json)、[核验]({HERE}/pilot_audit.json)、[规则预检]({HERE}/GATE_REPORT.md)、[协议]({HERE}/PROTOCOL.md)。']
    (HERE/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')
    verify_parent()


if __name__=='__main__':
    torch.set_num_threads(1)
    evaluate()
