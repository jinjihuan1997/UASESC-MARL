"""Audit paired milestone results without selecting a favorable stopping point."""
from helpers import *
from evaluation import FIELDS,summarize

def stats(x):
    return dict(mean=float(np.mean(x)),sd=float(np.std(x,ddof=1)) if len(x)>1 else None,values=list(map(float,x)),positive_count=int(np.sum(np.asarray(x)>0)))

def audit_trace(data,age):
    age=age.reshape(*data.shape[:2],-1).astype(float);v={k:data[...,i] for i,k in enumerate(FIELDS)}
    assert np.isfinite(data).all() and not data[...,6:9].any()
    assert not v['service_violation_cost'].any() and not v['recv_aoi_bonus'].any()
    weights=np.asarray(config()['env_args']['reward_weights_by_instruction'])[v['instruction_id'].astype(int)]
    quality=(v['predicted_quality_sum']-21*v['deliveries'])/360;load=.02*v['channel_uses']/60000
    mean=.4*age.mean(-1);maximum=.3*age.max(-1);tail=.3*np.maximum(age-4,0).mean(-1)
    expected=dict(quality_credit=weights[...,0]*quality,resource_cost=load,mean_aoi=age.mean(-1),max_aoi=age.max(-1),
        age_mean_cost=weights[...,1]*mean/8,age_max_cost=weights[...,1]*maximum/8,age_tail_cost=weights[...,1]*tail/8,
        fraction_above4=(age>4).mean(-1),fraction_above6=(age>6).mean(-1))
    for ref in [8,10]: expected[f'objective_ref{ref}']=weights[...,0]*quality-weights[...,1]*(mean+maximum+tail)/ref-load
    expected['common_reward']=expected['objective_ref8']
    for k,value in expected.items(): np.testing.assert_allclose(value,v[k],atol=1e-9,rtol=0)

def main(root=HERE,check_inputs=True):
    m=read(root/'manifest.json')
    if check_inputs: verify()
    seeds=m['seeds'];per={};pairing=None;episodes=0;hashes={};behavior={};recovery={};seed_scores={}
    for seed in seeds:
        initial=[read(root/'jobs'/f'seed_{seed}'/arm/'manifest.json') for arm in METHODS]
        assert all(x['initial_actor_hashes']==initial[0]['initial_actor_hashes'] and x['initial_critic_hash']==initial[0]['initial_critic_hash'] for x in initial)
    for j in m['jobs']:
        job=root/j['output'];status=read(job/'status.json');cfg=read(root/j['config'])
        assert status['state']=='complete' and status['completed_steps']==m['steps_per_method']
        for f,h in status['checkpoint_hashes'].items(): assert digest(job/f)==h
        logs=[json.loads(x) for x in (job/'training_metrics.jsonl').read_text().splitlines()]
        assert len(logs)==m['steps_per_method']//m['batch']
        spec=cfg['continuation']
        for row in logs:
            assert row['trained_actor_ids']==(['selector'] if j['method']=='selector' else [0,1,2,3])
            fraction=(row['update']-1)/max(1,len(logs)-1)
            expected=spec['actor_lr_start']+(spec['actor_lr_end']-spec['actor_lr_start'])*fraction
            np.testing.assert_allclose(row['actor_learning_rates'],expected,rtol=1e-10,atol=1e-14)
            expected_entropy=spec['entropy_start']+(spec['entropy_end']-spec['entropy_start'])*fraction
            np.testing.assert_allclose(row['entropy_coef'],expected_entropy,rtol=1e-10,atol=1e-14)
        initial=read(job/'manifest.json')
        if j['method']=='selector':
            assert status['final_actor_hashes']==initial['initial_actor_hashes']
            assert status['final_selector_hash']!=initial['initial_selector_hash']
    for item in m['evaluation_items']:
        folder=root/'evaluation'/item;status=read(folder/'status.json');assert status['state']=='complete'
        assert digest(folder/'summary.json')==status['summary_sha256'];summary=read(folder/'summary.json')
        if pairing is None: pairing=summary['pairing']
        assert pairing==summary['pairing'];per[item]={};behavior[item]={};recovery[item]={};seed_scores[item]={};parts=[]
        category,method=item.split('/')
        if category.startswith('seed_'):
            arm,step=method.rsplit('_at_',1);model=root/'jobs'/category/arm/'milestones'/f'steps_{step}';s=read(model/'status.json')
            assert s['state']=='complete' and s['completed_steps']==int(step)
            for f,h in s['checkpoint_hashes'].items(): assert digest(model/f)==h
        for scenario,schedule in m['scenarios'].items():
            file=folder/f'{scenario}.npz';h=digest(file);assert h==summary['traces'][scenario];hashes[str(file.relative_to(root))]=h
            with np.load(file) as z:
                data=z['trace'];age=z['aoi_after'];modes=z['modes'];beta=z['resource_fractions']
                np.testing.assert_array_equal(z['fields'],FIELDS);np.testing.assert_array_equal(z['seeds'],m['evaluation_seeds'])
            n=len(m['evaluation_seeds']);assert data.shape==(600,n,len(FIELDS));audit_trace(data,age)
            gids=np.asarray([next(g for t,g in reversed(schedule) if t<=s) for s in range(600)])
            np.testing.assert_array_equal(data[...,5],np.broadcast_to(gids[:,None],data.shape[:2]))
            per[item][scenario]=summarize(data);parts.append(data);episodes+=n
            for k,v in per[item][scenario].items(): np.testing.assert_allclose(v,summary['scenarios'][scenario]['overall'][k],atol=1e-8,rtol=1e-10)
            seed_scores[item][scenario]=data[...,0].mean(0).tolist()
            selector_stats={}
            if category.startswith('seed_') and arm=='selector':
                with np.load(file) as z:choice=z['selector_choices']
                assert np.isin(choice,[0,1]).all()
                selector_stats={str(g):float(choice[data[...,5]==g].mean()) for g in range(3) if (data[...,5]==g).any()}
            triples,counts=np.unique(modes.reshape(-1,3),axis=0,return_counts=True)
            behavior[item][scenario]=dict(selector_equal_fraction_by_instruction=selector_stats,joint_modes=[dict(modes=t.tolist(),fraction=float(c/modes[...,0].size)) for t,c in zip(triples,counts)],
                resource_std_per_uav=beta.reshape(-1,3).std(0).tolist(),resource_mean_per_uav=beta.reshape(-1,3).mean(0).tolist())
            for j,(start,g) in enumerate(schedule):
                end=schedule[j+1][0] if j+1<len(schedule) else 600
                per[item][f'{scenario}/segment_{j}_g{g}']=summarize(data[start:end])
                if j:
                    per[item][f'{scenario}/first20_after_{start}']=summarize(data[start:min(start+20,end)])
                    recovery[item][f'{scenario}/switch_{start}']=dict(fields=FIELDS,offsets=list(range(min(end-start,100))),mean_across_eval_seeds=data[start:min(start+100,end)].mean(1).tolist())
        merged=np.concatenate(parts);per[item]['overall']=summarize(merged)
        for g in range(3):
            selected=merged[merged[...,5]==g]
            if selected.size: per[item][f'true_instruction_{g}']=summarize(selected)
    grouped={};paired={};scopes=per[m['evaluation_items'][0]]
    for step in m['milestones']:
        grouped[step]={};paired[step]={}
        for method in METHODS+RULE_METHODS:
            ids=[f'seed_{s}/{method}_at_{step}' for s in seeds] if method in METHODS else [f'rules/{method}']
            grouped[step][method]={scope:{k:stats([per[i][scope][k] for i in ids]) for k in per[ids[0]][scope] if k!='slots'} for scope in scopes}
        for arm in METHODS:
            for rule in RULE_METHODS:
                paired[step][f'{arm}_minus_{rule}']={scope:stats([per[f'seed_{s}/{arm}_at_{step}'][scope]['common_reward']-per[f'rules/{rule}'][scope]['common_reward'] for s in seeds]) for scope in scopes}
        paired[step]['selector_minus_joint_continue']={scope:stats([per[f'seed_{s}/selector_at_{step}'][scope]['common_reward']-per[f'seed_{s}/joint_continue_at_{step}'][scope]['common_reward'] for s in seeds]) for scope in scopes}
    assert episodes==m['evaluation_episodes']
    if check_inputs:
        for f,h in read(root/'provenance.json')['parent_input_hashes'].items(): assert digest(f)==h
    out=root/'report';write(out/'results.json',dict(per_item=per,grouped_by_steps=grouped,paired_by_steps=paired,per_eval_seed_scores=seed_scores))
    write(out/'behavior.json',behavior);write(out/'switch_recovery.json',recovery);write(out/'trace_hashes.json',hashes)
    write(out/'audit.json',dict(state='PASS',episodes=episodes,slots=episodes*600,training_steps=m['total_training_steps'],paired_exogenous_sequences=True,
        matched_initial_networks=True,stage_actor_updates_verified=True,learning_rate_schedule_verified=True,independent_reward_and_physics=True,old_artifacts_checked=check_inputs))
    lines=['资源选择器与原joint续训的配对结果','',
        f'各从1000万步joint出发，再训练{m["steps_per_method"]}物理步；种子{seeds}。选择器组固定原四个actor，只训练二选一选择器和critic；对照组继续更新原四个actor和critic。',
        '所有分数乘100，越大越好。原开发种子重复评估，不是独立测试；最后检查点是预定主要结果，不选择最有利的中间结果。','',
        '| 新增步数 | 方法 | 综合分数 | AoI | 交付预测PSNR | 质量指令分数 | AoI指令分数 |','|---|---|---:|---:|---:|---:|---:|']
    for step in m['milestones']:
        for method in METHODS+RULE_METHODS:
            r=grouped[step][method];g=r['overall'];q=r.get('fixed_2',g);a=r.get('fixed_1',g)
            lines.append(f'| {step} | {method} | {100*g["common_reward"]["mean"]:.4f} | {g["mean_aoi"]["mean"]:.4f} | {g["delivered_predicted_psnr"]["mean"]:.4f} | {100*q["common_reward"]["mean"]:.4f} | {100*a["common_reward"]["mean"]:.4f} |')
    lines+=['','最终检查点的配对差值：','']
    for name,values in paired[m['milestones'][-1]].items():
        r=values['overall'];lines.append(f'- {name}：均值{100*r["mean"]:+.4f}；各训练种子差值{[round(100*x,4) for x in r["values"]]}。')
    lines+=['','本轮没有训练隐藏指令对照，不能据此确认显式指令输入的独立贡献。三个种子同方向也不等于独立测试统计显著。',
        '原1000万步joint开发集参考分数-3.0248；手工质量均分混合诊断参考分数-2.8801。它们不是本轮新增训练结果。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'artifact_hashes.json',{str(f.relative_to(root)):digest(f) for f in [root/'REPORT.md',*out.glob('*.json')] if f.name!='artifact_hashes.json'})
    print('Aggregation PASS',episodes,flush=True)

if __name__=='__main__': main()
