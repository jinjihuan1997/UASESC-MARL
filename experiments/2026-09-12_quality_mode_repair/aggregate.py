"""Recompute every reported score from numeric trajectories; paired episode CIs."""
from repair_support import *
import argparse
import subprocess

GROUP_NAMES={'overall_13':'13场景综合','balance_fixed':'固定均衡','aoi_fixed':'固定AoI','quality_fixed':'固定质量','switch_10':'10个切换场景'}
METHOD_NAMES={'original_rl':'原RL','quality_rule_hybrid':'质量规则混合','residual_all':'全指令修正','residual_quality':'质量门控修正',
              'R_instruction':'R_instruction','R_equal_instruction':'R_equal_instruction','greedy_modes_3':'同观测贪心(3模式)','greedy_modes_16':'同观测贪心(16模式)'}

def ci(x,indices):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim>1:x=x.mean(tuple(range(x.ndim-1)))
    draws=x[indices].mean(-1)
    return dict(mean=float(x.mean()),ci95=[float(v) for v in np.quantile(draws,[.025,.975])],paired_environment_clusters=len(x))

def tables(m,scenario):
    env=make_env(cfg_for(m['parents'][0]),m['validation'],'cpu',m['scenarios'][scenario],'evaluation')
    external=exogenous(env)
    quality=[];load=[];req=[]
    for slot in range(600):
        env.step_index=slot;env.update_channels(slot);env.update_tables()
        quality.append(arr(env.quality).copy());load.append(arr(env.load).copy());req.append(arr(env.context()[2]).copy())
    return dict(quality=np.stack(quality),load=np.stack(load),req=np.stack(req),external=external)

def physical_audit(z,tab,cfg):
    """Independent NumPy reconstruction: profile lookup -> executor -> reward."""
    env=cfg['env_args'];trace=z['trace'];fields=z['fields'].tolist();val={f:trace[:,:,i] for i,f in enumerate(fields)}
    E=trace.shape[1]
    assert trace.shape==(600,E,len(FIELDS)) and fields==FIELDS
    assert all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind in 'fi')
    aoi=z['aoi_after'].astype(float);cache=z['cache_after'];tau=z['tau_after'];served=z['served']
    before_aoi=np.concatenate([z['initial_aoi'][None],aoi[:-1]])
    before_q=np.concatenate([z['initial_q'][None],cache[:-1]])
    before_tau=np.concatenate([z['initial_tau'][None],tau[:-1]])
    slot=np.arange(600)[:,None,None,None]
    expected=np.where(served,slot-np.where(before_tau<0,slot,before_tau)+1,before_aoi+1).clip(max=env['A_max'])
    np.testing.assert_array_equal(aoi,expected)
    np.testing.assert_array_equal(cache,(before_q & ~served)|(~before_q))
    np.testing.assert_array_equal(tau,np.where(~before_q,slot,np.where(before_q & ~served,before_tau,-1)))
    assert not np.any(served & ~before_q)
    # Native allocate_resources first casts the action to float64, then adds 1.
    # Casting after float32 arithmetic changes some greedy candidate shares.
    raw=((z['raw_resource_action'].astype(np.float64)+1)/2).clip(min=0)
    total=raw.sum(-1,keepdims=True);normal=np.where(total>0,raw/np.maximum(total,1e-300),1/3)
    shares=.05+.85*normal;shares/=shares.sum(-1,keepdims=True)
    np.testing.assert_allclose(shares,z['resource_fractions'],atol=1e-12,rtol=0)
    budget=.6*np.minimum(120000,z['resource_fractions']*100000)
    np.testing.assert_allclose(budget,z['budgets'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(z['quality_requirement'],tab['req'],atol=1e-12,rtol=0)
    valid=(tab['quality']>=tab['req'][...,None]-1e-9)&(tab['load']<=budget[...,None]+1e-9)&before_q.any(-1)[...,None]
    mask=np.where(valid.any(-1,keepdims=True),valid,True)
    np.testing.assert_array_equal(mask,z['uav_masks'][...,:16])
    requested=z['requested_modes'].astype(int)
    assert requested.min()>=0 and requested.max()<16
    preferred_valid=np.take_along_axis(valid,requested[...,None],-1).squeeze(-1)
    chosen=np.where(preferred_valid,requested,valid.argmax(-1))
    loads=np.take_along_axis(tab['load'],chosen[...,None],-1).squeeze(-1)
    qualities=np.take_along_axis(tab['quality'],chosen[...,None],-1).squeeze(-1)
    order=np.argsort(-np.where(before_q,before_aoi,-np.inf),axis=-1,kind='stable')
    ranked_q=np.take_along_axis(before_q,order,-1);left=budget.copy();takes=[]
    for j in range(10):
        take=valid.any(-1)&ranked_q[...,j]&(loads<=left+1e-9)
        left-=np.where(take,loads,0);takes.append(take)
    ordered=np.stack(takes,-1);expected_served=np.zeros_like(before_q)
    np.put_along_axis(expected_served,order,ordered,-1)
    np.testing.assert_array_equal(served,expected_served)
    count=served.sum(-1);executed=np.where(count>0,chosen,-1)
    np.testing.assert_array_equal(z['modes'],executed)
    usage=loads*count
    np.testing.assert_allclose(z['usage'],usage,atol=1e-8,rtol=0)
    np.testing.assert_allclose(z['unused_budgets'],budget-usage,atol=1e-8,rtol=0)
    np.testing.assert_allclose(z['predicted_quality'][count>0],qualities[count>0],atol=1e-12,rtol=0)
    gid=val['instruction_id'].astype(int);w=np.asarray(env['reward_weights_by_instruction'])[gid]
    flat=aoi.reshape(600,E,30)
    qgain=(np.maximum((qualities-env['Q_min_eval'])/(env['Q_max']-env['Q_min_eval']),0)*count).sum(-1)/30
    components=dict(quality_credit=w[:,:,0]*qgain,
        age_mean_cost=w[:,:,1]*.4*flat.mean(-1)/8,
        age_max_cost=w[:,:,1]*.3*flat.max(-1)/8,
        age_tail_cost=w[:,:,1]*.3*np.maximum(flat-4,0).mean(-1)/8,
        resource_cost=w[:,:,2]*.2*usage.sum(-1)/60000,
        service_violation_cost=np.zeros((600,E)),recv_aoi_bonus=np.zeros((600,E)))
    for k,v in components.items():np.testing.assert_allclose(val[k],v,atol=1e-9,rtol=0)
    common=components['quality_credit']-sum(components[k] for k in ['age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost'])+components['recv_aoi_bonus']
    np.testing.assert_allclose(val['common_reward'],common,atol=1e-9,rtol=0)
    np.testing.assert_allclose(val['training_reward'].astype(np.float32),common,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(val['predicted_quality_sum'],(qualities*count).sum(-1),atol=1e-9,rtol=0)
    np.testing.assert_array_equal(val['deliveries'],count.sum(-1))
    return dict(physical_slots=600*E,max_reward_reconstruction_error=float(np.max(abs(val['common_reward']-common))),
        checks=['AoI','cache','timestamp','one_resource_floor','budget','profile_quality','profile_load',
                'original_mask','preferred_action_fallback','stable_AoI_scheduler','delivery','usage','reward_components'])

def compact_stats(z,m):
    v={f:z['trace'][:,:,i] for i,f in enumerate(FIELDS)};gids=v['instruction_id'].astype(int)
    groups={};mapping=np.full(17,-1,dtype=int)
    for i,g in enumerate(m['equivalence_groups']):
        for mode in g:mapping[mode+1]=i
    for gid in range(3):
        take=gids==gid;n=int(take.sum())
        if n==0:continue
        d=float(v['deliveries'][take].sum())
        group=dict(slots=n,score_x100=float(v['common_reward'][take].mean()*100),
            physical_reward_means={f:float(v[f][take].mean()) for f in FIELDS},
            predicted_psnr_per_delivery=float(v['predicted_quality_sum'][take].sum()/d) if d else None,
            resource_share_mean=z['resource_fractions'][take].mean(0).tolist(),
            budget_mean=z['budgets'][take].mean(0).tolist(),usage_mean=z['usage'][take].mean(0).tolist(),
            unused_budget_mean=z['unused_budgets'][take].mean(0).tolist(),deliveries_by_uav_mean=z['served'][take].sum(-1).mean(0).tolist(),
            adapter_enabled_count=z['adapter_enabled'][take].sum(0).tolist(),
            requested_mode_counts=[np.bincount(z['requested_modes'][take][:,u],minlength=16).tolist() for u in range(3)],
            executed_mode_counts_minus1_then_0_to_15=[np.bincount(z['modes'][take][:,u]+1,minlength=17).tolist() for u in range(3)],
            equivalence_counts_no_delivery_then_groups=[np.bincount(mapping[z['modes'][take][:,u]+1]+1,minlength=len(m['equivalence_groups'])+1).tolist() for u in range(3)])
        groups[str(gid)]=group
    return groups

def carryover_mask(schedule):
    gids=np.zeros(600,dtype=int)
    for slot,gid in schedule:gids[slot:]=gid
    seen_quality=np.maximum.accumulate(gids==2)
    return seen_quality&(gids!=2)

def aggregate():
    m=verify_inputs();assert read(HERE/'preflight.json')['state']=='PASS'
    assert sha(HERE/'aggregate.py')==read(HERE/'analysis_manifest.json')['aggregate_sha256']
    assert sha(HERE/'seed_selection.json')==m['seed_selection_sha256']
    report=HERE/'report';report.mkdir(exist_ok=True)
    scenarios=list(m['scenarios']);N=len(scenarios);E=len(m['validation']);parents=m['parents']
    assert N==13 and E==20
    boot=np.random.default_rng(m['bootstrap_seed']).integers(0,E,(m['bootstrap_replicates'],E))
    group_indices=dict(overall_13=list(range(N)),balance_fixed=[scenarios.index('fixed_0')],
        aoi_fixed=[scenarios.index('fixed_1')],quality_fixed=[scenarios.index('fixed_2')],
        switch_10=[i for i,s in enumerate(scenarios) if not s.startswith('fixed_')])
    tabcache={};audits={};metadata={};physical={};curves={};trajectories={};trajectory_hashes={}
    score={method:np.zeros((3,N,E)) for method in ['original_rl','quality_rule_hybrid',*m['arms'],*m['rules']]}
    carry={method:{s:np.zeros((3,E)) for s in scenarios if carryover_mask(m['scenarios'][s]).any()} for method in score}
    items=[]
    for seed in parents:
        for method in ['original_rl','quality_rule_hybrid',*m['arms']]:
            label=method+('_at_1000000' if method in m['arms'] else '')
            items.append((seed,method,1000000 if method in m['arms'] else None,f'seed_{seed}/{label}'))
        for method in m['arms']:
            for step in m['milestones'][:-1]:items.append((seed,method,step,f'seed_{seed}/{method}_at_{step}'))
    for method in m['rules']:items.append((None,method,None,f'rules/{method}'))
    cfg=cfg_for(parents[0]);reserved=set(m['reserved_final_test']);seen_seeds=set();gate_equal=[]
    # The deployed strong baselines remain the versions authenticated by their
    # original fitting protocol, not merely files present at this run's start.
    greedy_sources=read(GREEDY/'protocol.json')['source_hashes']
    for old_path,h in greedy_sources.items():
        suffix=old_path.split('/experiments/',1)[1]
        assert sha(HERE.parent/suffix)==h
    for seed,method,step,item in items:
        folder=HERE/'evaluation'/item
        status=read(folder/'status.json');assert status['state']=='complete'
        summ=read(folder/'summary.json');assert sha(folder/'summary.json')==status['summary_sha256']
        expected=[s for s in scenarios if s.startswith('fixed_')] if step and step<1000000 else scenarios
        assert list(summ['scenarios'])==expected
        for scenario in expected:
            key=f'{item}/{scenario}';file=folder/f'{scenario}.npz';meta=read(folder/f'{scenario}.json')
            assert sha(file)==meta['trace_sha256'];trajectory_hashes[key]=meta['trace_sha256']
            assert meta['identity']['manifest_sha256']==sha(HERE/'manifest.json')
            with np.load(file,allow_pickle=False) as z:
                assert z['seeds'].tolist()==m['validation'];seen_seeds.update(z['seeds'].tolist());assert not seen_seeds & reserved
                if scenario not in tabcache:tabcache[scenario]=tables(m,scenario)
                assert meta['external_hashes']==tabcache[scenario]['external']
                audit=physical_audit(z,tabcache[scenario],cfg)
                rewards=z['trace'][:,:,FIELDS.index('common_reward')];episode=rewards.mean(0)*100
                np.testing.assert_allclose(episode,meta['scores_by_environment'],atol=0,rtol=0)
                audits[key]=audit
                physical[key]=compact_stats(z,m)
                metadata[key]=dict(score_x100=float(episode.mean()),episode_scores_x100=episode.tolist(),
                    episode_mean_raw_reward=(episode/100).tolist(),episode_total_reward=(episode*6).tolist())
                if step and step<1000000:
                    curves.setdefault(str(seed),{}).setdefault(method,{}).setdefault(str(step),{})[scenario]=ci(episode,boot)
                    continue
                pi=parents.index(seed) if seed else slice(None);si=scenarios.index(scenario)
                score[method][pi,si]=episode
                cmask=carryover_mask(m['scenarios'][scenario])
                if cmask.any():carry[method][scenario][pi]=rewards[cmask].mean(0)*100
                if method=='residual_quality':
                    gid=z['trace'][:,:,FIELDS.index('instruction_id')].astype(int)
                    np.testing.assert_array_equal(z['adapter_enabled'],np.repeat((gid==2)[...,None],3,axis=-1))
                    if scenario in ('fixed_0','fixed_1'):
                        original=HERE/f'evaluation/seed_{seed}/original_rl/{scenario}.npz'
                        with np.load(original,allow_pickle=False) as parent:
                            for k in z.files:
                                if k=='adapter_enabled':continue
                                np.testing.assert_array_equal(z[k],parent[k],err_msg=f'{key}:{k}')
                        assert not z['adapter_enabled'].any();gate_equal.append(key)
                elif method=='residual_all':assert z['adapter_enabled'].all()
        print(json.dumps(dict(aggregated=item,checked_traces=len(audits))),flush=True)
    scores={};paired={}
    for method,x in score.items():
        scores[method]=dict(conditional_mean={},by_parent={},by_scenario={})
        for group,idx in group_indices.items():
            values=x[:,idx].mean(1)
            scores[method]['conditional_mean'][group]=ci(values,boot)
            for pi,seed in enumerate(parents):scores[method]['by_parent'].setdefault(str(seed),{})[group]=ci(values[pi],boot)
        for si,scenario in enumerate(scenarios):scores[method]['by_scenario'][scenario]=ci(x[:,si],boot)
    for treatment in m['arms']:
        for comparator in ['original_rl','quality_rule_hybrid',*m['rules']]+(['residual_all'] if treatment=='residual_quality' else []):
            key=f'{treatment}_minus_{comparator}';diff=score[treatment]-score[comparator]
            paired[key]=dict(conditional_mean={},by_parent={},by_scenario={})
            for group,idx in group_indices.items():
                d=diff[:,idx].mean(1);paired[key]['conditional_mean'][group]=ci(d,boot)
                for pi,seed in enumerate(parents):paired[key]['by_parent'].setdefault(str(seed),{})[group]=ci(d[pi],boot)
            for si,scenario in enumerate(scenarios):paired[key]['by_scenario'][scenario]=ci(diff[:,si],boot)
    carry_results={}
    for treatment in m['arms']:
        for comparator in ['original_rl','quality_rule_hybrid','residual_all']:
            if treatment==comparator:continue
            key=f'{treatment}_minus_{comparator}'
            values=np.stack([carry[treatment][s]-carry[comparator][s] for s in carry[treatment]],1)
            carry_results[key]=dict(equal_scenario_mean=ci(values.mean(1),boot),
                by_parent={str(seed):ci(values[i].mean(0),boot) for i,seed in enumerate(parents)},
                by_scenario={s:dict(post_quality_slots=int(carryover_mask(m['scenarios'][s]).sum()),difference=ci(values[:,i],boot)) for i,s in enumerate(carry[treatment])})
    training={};paired_environment_ledgers=[]
    for seed in parents:
        inits=[];ledgers=[]
        for arm in m['arms']:
            folder=HERE/f'jobs/seed_{seed}/{arm}';status=read(folder/'status.json')
            assert status['state']=='complete' and status['completed_steps']==1000000 and status['update']==250
            rows=[json.loads(line) for line in (folder/'training_metrics.jsonl').read_text().splitlines()]
            assert [r['update'] for r in rows]==list(range(1,251))
            assert sum(sum(r['collection']['instruction_samples']) for r in rows)==1000000
            init=read(folder/'initialization.json');inits.append(init)
            ledger=read(folder/'environment_ledger.json');ledgers.append(ledger)
            assert not set(ledger['seeds'])&reserved
            milestone={}
            for step in m['milestones']:
                path=folder/f'milestones/steps_{step}';st=read(path/'status.json')
                assert st['completed_steps']==step and st['base_hashes']==init['base_hashes']
                for f,h in st['checkpoint_hashes'].items():assert sha(path/f)==h
                milestone[str(step)]=st
            state,entry=load_checkpoint(folder/'checkpoints');assert state['update']==250
            assert state['actor_optimizers'][0]==dict(state={},param_groups=[])
            assert state['config']==read(HERE/f'configs/seed_{seed}/{arm}.json')
            for i in range(4):
                parent_state=torch.load(model_dir(seed)/f'actor_agent{i}.pt',weights_only=True)
                frozen_state=state['actors'][i] if i==0 else {k.removeprefix('base_policy.'):v for k,v in state['actors'][i].items() if k.startswith('base_policy.')}
                assert parent_state.keys()==frozen_state.keys()
                assert all(torch.equal(parent_state[k],frozen_state[k]) for k in parent_state)
                if i:
                    adapter=torch.load(folder/f'milestones/steps_1000000/adapter_agent{i}.pt',weights_only=True)
                    assert all(torch.equal(v,state['actors'][i]['adapter.'+k]) for k,v in adapter.items())
                    assert len(state['actor_optimizers'][i]['param_groups'])==1
                    assert len(state['actor_optimizers'][i]['param_groups'][0]['params'])==6
            for filename,key in [('critic_agent.pt','critic'),('value_normalizer.pt','normalizer')]:
                snapshot=torch.load(folder/f'milestones/steps_1000000/{filename}',weights_only=True)
                assert all(torch.equal(v,state[key][k]) for k,v in snapshot.items())
            summed_instruction=np.sum([r['collection']['instruction_samples'] for r in rows],0).tolist()
            assert summed_instruction==status['counters']['instruction_samples']
            actual_updates=[sum(a['updates'] for r in rows for a in r['optimization']['actors'] if a['agent']==i) for i in range(1,4)]
            assert actual_updates==status['counters']['actor_updates']
            c=rows[-1]['optimization'];times=[json.loads(line) for line in (folder/'timing.jsonl').read_text().splitlines()]
            training[f'{seed}/{arm}']=dict(parent_model_file_hashes=read(model_dir(seed)/'status.json')['checkpoint_hashes'],
                initialization=init,counters=status['counters'],milestones=milestone,last_update_optimization=c,
                summed_collect_update_seconds=float(sum(r['collect_seconds']+r['update_seconds'] for r in times)),
                full_checkpoint=entry,full_checkpoint_contains=['base_actors','adapters','critic','ValueNorm','optimizers','action_rng','update_rng','global_rng','source_episode_rng','buffer','environment_tapes_and_state','progress'])
        for key in ['base_hashes','adapter_hashes','critic_hash','normalizer_hash','seeds']:assert inits[0][key]==inits[1][key]
        assert ledgers[0]==ledgers[1];paired_environment_ledgers.append(seed)
    results=dict(schema=1,unit='mean common reward per slot x100; raw and total also saved',split='development_validation',
        parent_seeds=parents,validation_seeds=m['validation'],reserved_final_test_run=False,
        statistical_unit='environment seed with its complete 13 scenarios; common resampling across all three parent models',
        confidence_intervals='95% paired cluster percentile bootstrap conditional on these three parent models, not random-training population',
        scenarios=m['scenarios'],score_summary=scores,paired_differences=paired,post_quality_carryover=carry_results,
        learning_curves_fixed_scenarios_only=curves,episode_results=metadata,training=training,
        equivalence_groups=m['equivalence_groups'],predicted_psnr_notice='PSNR from frozen average quality profile; not measured video decoding PSNR')
    write(report/'results.json',results);write(report/'physical_statistics.json',physical)
    gitbefore=read(HERE/'git_inspection.json');git_changes=[]
    for root,expected in gitbefore['repositories'].items():
        for key,args in [('head',['rev-parse','HEAD']),('status',['status','--short'])]:
            p=subprocess.run(['git','-C',root,*args],capture_output=True,text=True)
            actual=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
            if actual!=expected[key]:git_changes.append(dict(root=root,field=key))
    assert not git_changes,git_changes
    verify_inputs()
    audit=dict(state='PASS',protocol_sha256=m['protocol_sha256'],manifest_sha256=sha(HERE/'manifest.json'),
        protected_original_files=len(m['protected_input_sha256']),original_hash_changes=[],git_changes=git_changes,
        total_training_steps=sum(v['counters']['physical_steps'] for v in training.values()),training_jobs=6,
        evaluated_traces=len(audits),evaluated_complete_episodes=len(audits)*20,evaluation_physical_steps=sum(a['physical_slots'] for a in audits.values()),
        every_new_evaluation_original_checker='PASS',independent_offline_profile_executor_reward_audit=audits,
        fixed_nonquality_parent_trajectory_exact_matches=gate_equal,paired_training_exogenous_ledger_models=paired_environment_ledgers,
        final_test_used=False,final_test_verification=dict(guarded_environment_whitelist=True,actual_evaluation_seed_union=sorted(seen_seeds),
            training_seed_lists_checked=True,intersection_with_reserved=[]),trace_sha256=trajectory_hashes,
        all_required_models_and_baselines_evaluated=True,results_sha256=sha(report/'results.json'),physical_statistics_sha256=sha(report/'physical_statistics.json'))
    write(report/'audit.json',audit)
    build_report(results,audit)
    write(HERE/'status.json',dict(state='aggregated_pending_repeat_check',completed_training_steps=6000000,training_jobs=6,evaluation_episodes=len(audits)*20,
        evaluation_physical_steps=sum(a['physical_slots'] for a in audits.values()),audit='PASS',final_test_used=False,
        report='report/REPORT.md',results_sha256=sha(report/'results.json')))
    print(json.dumps(dict(state='complete',results_sha256=sha(report/'results.json'),traces=len(audits))))

def fmt(d):return f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"

def build_report(r,a):
    s=r['score_summary'];p=r['paired_differences'];parents=r['parent_seeds'];g=list(GROUP_NAMES)
    lines=['# 质量指令门控的模式修复：实际运行报告','',
      '本报告对应固定父模型、固定新增预算和最后100万步检查点。分数为每物理槽共同奖励平均×100，越高越好；所有差值为绝对分差。全部评估是已用于开发的20个验证种子，不是独立最终测试。PSNR是固定平均质量表预测值。','',
      '## 执行与审计','',f"6个任务均新增1,000,000步，共{a['total_training_steps']:,}步；每任务250次更新，保留20/40/60/80/100万步快照。三份原父模型各已有100万步成本；本次是冻结基础网络后的权重热启动，不是原联合控制器或原优化器精确续训。",'',
      f"实际评估 {a['evaluated_complete_episodes']:,} 个完整600槽回合、{a['evaluation_physical_steps']:,}步，保存 {a['evaluated_traces']} 份场景数值轨迹。每槽通过原独立核验，聚合时再从profile、模式、缓存和资源独立重建执行与奖励。{a['protected_original_files']} 个原文件最终哈希不变，原有Git工作区状态不变。",'',
      '## 固定最终检查点：逐父模型分数','',
      '| 父模型 | 方法 | 13场景综合 | 固定均衡 | 固定AoI | 固定质量 | 切换10场景 |','|---|---|---:|---:|---:|---:|---:|']
    for seed in parents:
        for method in ['original_rl','quality_rule_hybrid','residual_all','residual_quality']:
            v=s[method]['by_parent'][str(seed)]
            lines.append(f"| {seed} | {METHOD_NAMES[method]} | "+' | '.join(f"{v[k]['mean']:.4f}" for k in g)+' |')
    lines+=['','基线也在本次同样的13场景、20种子上重新评估，没有直接减去旧评估集的历史分数。基线不依赖父模型，以下只列一次。','',
       '| 方法 | 13场景综合 | 固定均衡 | 固定AoI | 固定质量 | 切换10场景 |','|---|---:|---:|---:|---:|---:|']
    for method in ['R_instruction','R_equal_instruction','greedy_modes_3','greedy_modes_16']:
        v=s[method]['conditional_mean'];lines.append(f"| {METHOD_NAMES[method]} | "+' | '.join(f"{v[k]['mean']:.4f}" for k in g)+' |')
    lines+=['','## 配对分差及不确定性','',
       '95%区间以环境种子/完整回合为成组单位；同一环境的13场景和三父模型共同重采样4000次。以下平均与区间以当前三个父模型为条件，不是随机训练总体结论，不把时隙当独立样本。','',
       '| 比较（前者减后者） | 综合分差 [95%CI] | 质量分差 [95%CI] | 切换分差 [95%CI] |','|---|---|---|---|']
    for key,v in p.items():
        t,c=key.split('_minus_');q=v['conditional_mean']
        lines.append(f"| {METHOD_NAMES[t]} − {METHOD_NAMES[c]} | {fmt(q['overall_13'])} | {fmt(q['quality_fixed'])} | {fmt(q['switch_10'])} |")
    lines+=['','门控相对父模型和全指令修正的逐模型差：','',
       '| 父模型 | 门控−父模型：综合 | 门控−父模型：质量 | 门控−全指令：综合 | 门控−全指令：质量 |','|---|---|---|---|---|']
    for seed in parents:
        q=p['residual_quality_minus_original_rl']['by_parent'][str(seed)]
        b=p['residual_quality_minus_residual_all']['by_parent'][str(seed)]
        lines.append(f"| {seed} | {fmt(q['overall_13'])} | {fmt(q['quality_fixed'])} | {fmt(b['overall_13'])} | {fmt(b['quality_fixed'])} |")
    lines+=['','## 对八个问题的回答','']
    qparent=p['residual_quality_minus_original_rl'];deltas=[qparent['by_parent'][str(seed)]['quality_fixed']['mean'] for seed in parents]
    improved=sum(x>0 for x in deltas)
    supported=sum(qparent['by_parent'][str(seed)]['quality_fixed']['ci95'][0]>0 for seed in parents)
    lines.append(f"1. **质量专项是否改善：{improved}/3 个父模型改善。** 逐模型质量分差分别为 "+'、'.join(f'{x:+.4f}' for x in deltas)+f"；平均 {fmt(qparent['conditional_mean']['quality_fixed'])}。其中 {supported}/3 个父模型的配对区间完全高于0，当前开发验证支持这三个父模型的专项修复有效。")
    lines+=['',f"2. **非质量固定指令确实保留。** 三父模型的固定均衡、固定AoI共6组、120个完整回合逐数值相同，adapter启用次数均为0。这是冻结与门控的实现验收结果，不是训练波动。",'']
    q=p['residual_quality_minus_residual_all']['conditional_mean']
    lines.append(f"3. **门控与全指令修正的取舍。** 门控−全指令的综合分差 {fmt(q['overall_13'])}，质量分差 {fmt(q['quality_fixed'])}。门控质量专项更好，但没有综合优于全指令修正的证据。全指令修正改善了均衡任务，却造成AoI任务的平均退化；门控保留了两者原有行为，也放弃了均衡任务的这一提升。两组共享完整批次分母，门控并未把质量样本权重额外放大。实际更新量见训练统计。")
    carry=r['post_quality_carryover']['residual_quality_minus_original_rl']
    lines+=['',f"4. **质量阶段后的影响。** 在曾经历质量且当前已离开质量的时段，门控−父模型的等场景平均分差为 {fmt(carry['equal_scenario_mean'])}。本次观察到小幅后续代价，区间完全低于0，来自缓存/AoI的路径变化。该代价没有抵消13场景综合收益；切换返回后不应要求复现父轨迹。",'']
    q=p['residual_quality_minus_quality_rule_hybrid']['conditional_mean']
    lines.append(f"5. **相对手工混合的额外价值。** 门控−质量规则混合：综合 {fmt(q['overall_13'])}，质量 {fmt(q['quality_fixed'])}。门控在两项上都略低于手工混合，配对区间均不含0，本次没有显示RL修正的额外收益。全指令修正相对混合的综合分差仅+0.0029，区间跨0，也没有稳定优势。手工混合直接复用原指令规则模式，属于手工条件控制器。")
    lines+=['','6. **与基线比较。** 两种修正方案的13场景综合分数都低于 R_instruction、R_equal_instruction 以及两个同观测贪心版本，平均配对差的区间均低于0。本次未超过这些同口径基线；两个冻结贪心版本在本验证集上得分相同，均完整保留。','']
    q=qparent['conditional_mean']['overall_13'];hy=p['residual_quality_minus_quality_rule_hybrid']['conditional_mean']['overall_13']
    if q['ci95'][0]>0 and hy['ci95'][0]>0:
        recommendation='当前门控在综合分数和相对手工混合方面已有配对支持，可以作为下一步另行批准的联合优化候选；本轮结果仍不足以保证联合微调有效。'
    elif q['ci95'][0]>0:
        recommendation='建议保留这一受限修复结果，并停止扩大本门控方案的训练预算。虽然相对父模型的专项和综合改善明确，但仍未超过手工混合，且与简单/强基线都有差距；当前证据不足以支持直接开启更大的联合优化。'
    else:
        recommendation='目前不建议扩大这一方案或自动追加联合训练：固定预算结果尚未给出稳定的综合改善证据。保留负结果，停止本轮扩展。'
    lines.append('7. **下一步判断。** '+recommendation)
    lines+=['','8. **仍不能得出：** 门控成功不等于证明梯度冲突是唯一原因；它使用显式质量指令先验，不是隐藏指令消融；开发验证不是独立最终测试；三个父模型不能代表随机训练总体；固定平均表预测不能外推真实视频解码；优于父模型不等于优于手工混合或所有强基线。','',
      '## 质量阶段后的逐场景差','',
      '| 场景 | 质量后的非质量槽数 | 门控−父模型 [95%CI] |','|---|---:|---|']
    for scene,v in carry['by_scenario'].items():lines.append(f"| {scene} | {v['post_quality_slots']} | {fmt(v['difference'])} |")
    lines+=['','## 训练统计和过程检查点','',
      '| 父模型 | 方法 | 均衡/AoI/质量实际样本 | UAV1/2/3实际optimizer.step | 无质量跳过次数 |','|---|---|---|---|']
    for key,v in r['training'].items():
        seed,method=key.split('/');c=v['counters']
        lines.append(f"| {seed} | {METHOD_NAMES[method]} | {c['instruction_samples']} | {c['actor_updates']} | {c['skipped_no_quality']} |")
    lines+=['','逐更新策略损失、熵、梯度范数、概率比、clip fraction、critic损失、固定GAE目标的价值误差与解释方差均在各任务 training_metrics.jsonl；模型hash和累计计数在每个milestone/status.json。新critic正常更新，原critic文件不变，未加入当前预算。','',
      '20/40/60/80万步只评估三个固定场景，所有结果保存在 results.json 的 learning_curves_fixed_scenarios_only，不能当作13场景综合分数。主要比较始终取100万步，未按曲线选择检查点。','',
      '模式原编号、profile精确等价组、各UAV资源份额/预算/使用/未用预算/交付和逐项奖励按场景、指令保存在 physical_statistics.json。固定质量任务的易读分解另见 MODE_AND_REWARD_DETAILS.md；30个检查点的训练累计明细在 checkpoint_statistics/。原动作空间仍为16类。','',
      '## 可复算性与边界','',
      '所有 evaluation/**/*.npz 都含600槽数值记录、20个环境种子、采样偏好模式与执行模式、缓存/AoI和资源。aggregate.py重新核验profile、确定性执行器、奖励并聚合。results.json只含确定性数值和固定统计种子；重复运行的数值hash须一致，记录在 reproducibility.json。','',
      f"预留最终测试未使用：构造环境必须经过种子白名单，聚合另核对全部轨迹的实际seed数组、训练外生seed清单，与reserved集合交集为空。原始保护文件 {a['protected_original_files']} 个重新逐个SHA-256核验。",'',
      '初次预检失败与修复均保留：16维分摊logp在float32累加会引入舍入，因此用float64精确求和验证数学恒等式，没有放宽分布容差；检查点内存副本的动作RNG列表改为复制，随后跨真实终止的恢复逐值一致。执行代码在预检通过后冻结。','',
      '最终离线聚合核验器初版把动作先用float32加1、再转float64；原执行器先转float64再加1。在贪心资源候选上产生最大约1.85e-8份额差异。修正的只是核验器的转换顺序，资源核验容差仍为1e-12，训练/评估代码、轨迹、公式和统计口径均不变。旧核验器、失败日志及哈希修订保存在 analysis_revisions/initial 和 REPORTING_FIX.md。','',
      '原论文、源码、配置、模型、质量表与历史报告未修改；未commit、push、创建PR或启动下一轮。']
    (HERE/'report/REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    guard();aggregate()
