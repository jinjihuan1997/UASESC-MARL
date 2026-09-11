"""Audit all saved slots and compare paired seeds under each resource capability."""
from helpers import *
from evaluation import FIELDS,summarize

def stats(x):
    return dict(mean=float(np.mean(x)),sd=float(np.std(x,ddof=1)) if len(x)>1 else None,
        values=list(map(float,x)),positive_count=int(np.sum(np.asarray(x)>0)))

def audit_trace(data,age):
    age=age.reshape(*data.shape[:2],-1).astype(float);v={k:data[...,i] for i,k in enumerate(FIELDS)}
    assert np.isfinite(data).all() and not data[...,6:9].any()
    assert not v['service_violation_cost'].any() and not v['recv_aoi_bonus'].any()
    weights=np.asarray(config()['env_args']['reward_weights_by_instruction'])[v['instruction_id'].astype(int)]
    quality=(v['predicted_quality_sum']-21*v['deliveries'])/360
    load=.02*v['channel_uses']/60000
    mean=.4*age.mean(-1);maximum=.3*age.max(-1);tail=.3*np.maximum(age-4,0).mean(-1)
    expected=dict(quality_credit=weights[...,0]*quality,resource_cost=load,mean_aoi=age.mean(-1),max_aoi=age.max(-1),
        age_mean_cost=weights[...,1]*mean/8,age_max_cost=weights[...,1]*maximum/8,age_tail_cost=weights[...,1]*tail/8,
        fraction_above4=(age>4).mean(-1),fraction_above6=(age>6).mean(-1))
    for ref in [8,10]: expected[f'objective_ref{ref}']=weights[...,0]*quality-weights[...,1]*(mean+maximum+tail)/ref-load
    expected['common_reward']=expected['objective_ref8']
    for k,value in expected.items(): np.testing.assert_allclose(value,v[k],atol=1e-9,rtol=0)

def main():
    m=verify();per={};pairing=None;episodes=0;hashes={};behavior={};recovery={};seed_scores={}
    for seed in SEEDS:
        initial=[read(HERE/'jobs'/f'seed_{seed}'/arm/'manifest.json') for arm in METHODS]
        assert all(x['initial_actor_hashes']==initial[0]['initial_actor_hashes'] and x['initial_critic_hash']==initial[0]['initial_critic_hash'] for x in initial)
    for item in m['evaluation_items']:
        folder=HERE/'evaluation'/item;status=read(folder/'status.json');assert status['state']=='complete'
        assert digest(folder/'summary.json')==status['summary_sha256'];summary=read(folder/'summary.json')
        if pairing is None: pairing=summary['pairing']
        assert pairing==summary['pairing'];per[item]={};behavior[item]={};recovery[item]={};seed_scores[item]={};parts=[]
        if item.startswith('seed_'):
            job=HERE/'jobs'/item;j=read(job/'status.json');assert j['state']=='complete' and j['completed_steps']==1000000
            for f,h in j['checkpoint_hashes'].items(): assert digest(job/f)==h
            logs=[json.loads(x) for x in (job/'training_metrics.jsonl').read_text().splitlines()]
            assert len(logs)==250
            for row in logs:
                fixed=item.endswith('mode_only') or (item.endswith('staged') and row['update']<=75)
                assert row['trained_actor_ids']==([1,2,3] if fixed else [0,1,2,3])
        for scenario,schedule in m['scenarios'].items():
            file=folder/f'{scenario}.npz';h=digest(file);assert h==summary['traces'][scenario];hashes[str(file.relative_to(HERE))]=h
            with np.load(file) as z:
                data=z['trace'];age=z['aoi_after'];modes=z['modes'];beta=z['resource_fractions']
                np.testing.assert_array_equal(z['fields'],FIELDS);np.testing.assert_array_equal(z['seeds'],EVAL_SEEDS)
            assert data.shape==(600,20,len(FIELDS));audit_trace(data,age)
            gids=np.asarray([next(g for t,g in reversed(schedule) if t<=s) for s in range(600)])
            np.testing.assert_array_equal(data[...,5],np.broadcast_to(gids[:,None],data.shape[:2]))
            if item.endswith('mode_only') or '/R_equal_' in item: np.testing.assert_allclose(beta,1/3,atol=1e-8)
            per[item][scenario]=summarize(data);parts.append(data);episodes+=20
            for k,v in per[item][scenario].items(): np.testing.assert_allclose(v,summary['scenarios'][scenario]['overall'][k],atol=1e-8,rtol=1e-10)
            seed_scores[item][scenario]=data[...,0].mean(0).tolist()
            # Physical-mode IDs are retained; aliases are only summarized for interpretation.
            triples,counts=np.unique(modes.reshape(-1,3),axis=0,return_counts=True)
            behavior[item][scenario]=dict(joint_modes=[dict(modes=t.tolist(),fraction=float(c/modes[...,0].size)) for t,c in zip(triples,counts)],
                per_uav_mode_counts=[np.bincount(modes[:,:,u].reshape(-1)+1,minlength=17).tolist() for u in range(3)],
                resource_std_per_uav=beta.reshape(-1,3).std(0).tolist(),resource_mean_per_uav=beta.reshape(-1,3).mean(0).tolist())
            for j,(start,g) in enumerate(schedule):
                end=schedule[j+1][0] if j+1<len(schedule) else 600
                per[item][f'{scenario}/segment_{j}_g{g}']=summarize(data[start:end])
                if j:
                    per[item][f'{scenario}/first20_after_{start}']=summarize(data[start:min(start+20,end)])
                    recovery[item][f'{scenario}/switch_{start}']=dict(fields=FIELDS,
                        offsets=list(range(min(end-start,100))),mean_across_eval_seeds=data[start:min(start+100,end)].mean(1).tolist())
        merged=np.concatenate(parts);per[item]['overall']=summarize(merged)
        for g in range(3): per[item][f'true_instruction_{g}']=summarize(merged[merged[...,5]==g])
    grouped={};paired={}
    for method in METHODS+RULE_METHODS:
        ids=[f'seed_{s}/{method}' for s in SEEDS] if method in METHODS else [f'rules/{method}']
        grouped[method]={scope:{k:stats([per[i][scope][k] for i in ids]) for k in per[ids[0]][scope] if k!='slots'} for scope in per[ids[0]]}
    for arm in METHODS:
        rules=RULE_METHODS[3:] if arm=='mode_only' else RULE_METHODS[:3]
        for rule in rules:
            paired[f'{arm}_minus_{rule}']={scope:stats([per[f'seed_{s}/{arm}'][scope]['common_reward']-per[f'rules/{rule}'][scope]['common_reward'] for s in SEEDS]) for scope in per[f'seed_85/{arm}']}
    paired['staged_minus_joint']={scope:stats([per[f'seed_{s}/staged'][scope]['common_reward']-per[f'seed_{s}/joint'][scope]['common_reward'] for s in SEEDS]) for scope in per['seed_85/joint']}
    assert episodes==m['evaluation_episodes']
    for f,h in read(HERE/'provenance.json')['parent_input_hashes'].items(): assert digest(f)==h
    out=HERE/'report';write(out/'results.json',dict(per_item=per,grouped=grouped,paired_same_capability=paired,per_eval_seed_scores=seed_scores))
    write(out/'behavior.json',behavior);write(out/'switch_recovery.json',recovery);write(out/'trace_hashes.json',hashes)
    write(out/'audit.json',dict(state='PASS',episodes=episodes,slots=episodes*600,training_steps=9000000,
        paired_exogenous_sequences=True,matched_initial_networks=True,stage_actor_updates_verified=True,
        fixed_resource_baselines=True,independent_reward_and_physics=True,old_artifacts_unchanged=True))
    lines=['# 新时序与训练方式对比结果','',
        '每组3个训练种子，每个模型100万环境步。分数使用相同原始奖励，显示乘100。仅模式组应与均分资源规则比较；联合/分阶段组与完整资源规则比较。',
        '这是既有诊断评估种子上的探索结果，不是独立新测试集确认，也不是新旧时序的单因素消融。','',
        '| 方法 | 综合分数×100 | AoI | 交付预测PSNR | 质量指令分数×100 | AoI指令分数×100 |',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS+RULE_METHODS:
        r=grouped[method];g=r['overall']
        lines.append(f'| {method} | {100*g["common_reward"]["mean"]:.4f} | {g["mean_aoi"]["mean"]:.4f} | {g["delivered_predicted_psnr"]["mean"]:.4f} | {100*r["fixed_2"]["common_reward"]["mean"]:.4f} | {100*r["fixed_1"]["common_reward"]["mean"]:.4f} |')
    lines+=['','同资源能力下，学习方法减规则方法的综合分数差：','']
    for name,scopes in paired.items():
        r=scopes['overall'];lines.append(f'- {name}：均值 {100*r["mean"]:+.4f}，3个训练种子差值 {[round(100*x,4) for x in r["values"]]}，正差 {r["positive_count"]}/3。')
    lines+=['','三种子均值/标准差及正差数量不等同于统计显著性；本轮没有新接口隐藏指令消融，不能据此宣称显式指令信息优于隐藏指令。',
        '',f'已独立审计 {episodes} 回合、{episodes*600} 时隙。完整逐场景、逐指令、切换分段、模式分布及资源波动见 report/。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'artifact_hashes.json',{str(f.relative_to(HERE)):digest(f) for f in [HERE/'REPORT.md',*out.glob('*.json')] if f.name!='artifact_hashes.json'})
    print('Aggregation PASS',episodes,flush=True)

if __name__=='__main__': main()
