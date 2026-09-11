"""Independent ref10/ref8 scoring and paired inference from every saved slot."""
from helpers import *
from evaluation import FIELDS,summarize


def stats(x):
    return dict(mean=float(np.mean(x)),sd=float(np.std(x,ddof=1)) if len(x)>1 else None,
                values=list(map(float,x)),positive_count=int(np.sum(np.asarray(x)>0)))


def audit_trace(data,age,cfg):
    e=cfg['env_args'];age=age.reshape(*data.shape[:2],-1).astype(float)
    x={k:data[...,i] for i,k in enumerate(FIELDS)}
    assert np.isfinite(data).all() and not np.any(data[...,6:9])
    assert not np.any(x['service_violation_cost']) and not np.any(x['recv_aoi_bonus'])
    w=np.asarray(e['reward_weights_by_instruction'])[x['instruction_id'].astype(int)]
    q=(x['predicted_quality_sum']-21*x['deliveries'])/360
    assert np.all(q>=0)
    load=.02*x['channel_uses']/60000
    age_raw=.4*age.mean(-1)+.3*age.max(-1)+.3*np.maximum(age-4,0).mean(-1)
    for ref in [10,8]:
        u=w[...,0]*q-w[...,1]*age_raw/ref-load
        np.testing.assert_allclose(u,x[f'objective_ref{ref}'],atol=1e-9,rtol=0)
    ref=e['aoi_reward_ref']
    expected=dict(quality_credit=w[...,0]*q,resource_cost=load,
        age_mean_cost=w[...,1]*.4*age.mean(-1)/ref,
        age_max_cost=w[...,1]*.3*age.max(-1)/ref,
        age_tail_cost=w[...,1]*.3*np.maximum(age-4,0).mean(-1)/ref,
        mean_aoi=age.mean(-1),max_aoi=age.max(-1),fraction_above4=(age>4).mean(-1),fraction_above6=(age>6).mean(-1))
    for k,v in expected.items():np.testing.assert_allclose(v,x[k],atol=1e-9,rtol=0)
    np.testing.assert_allclose(x['common_reward'],x[f'objective_ref{int(ref)}'],atol=1e-9,rtol=0)


def main():
    m=verify();per={};traces={};model_hashes={};pairing=None;episodes=0;slots=0
    for item in m['evaluation_items']:
        o,cat,method=item.split('/');folder=HERE/'evaluation'/item
        status=read(folder/'status.json');assert status['state']=='complete'
        assert digest(folder/'summary.json')==status['summary_sha256']
        summary=read(folder/'summary.json');cfg=config(o,method==METHODS[1],int(cat[5:]) if cat.startswith('seed_') else 85)
        if cat.startswith('seed_'):
            job=HERE/'jobs'/item;js=read(job/'status.json')
            assert js['state']=='complete' and js['completed_steps']==1000000
            for f,h in js['checkpoint_hashes'].items():assert digest(job/f)==h;model_hashes[str((job/f).relative_to(HERE))]=h
        if pairing is None:pairing=summary['pairing']
        assert pairing==summary['pairing']
        per[item]={};parts=[];hidden_reference=None
        for scenario,schedule in m['scenarios'].items():
            file=folder/f'{scenario}.npz';h=digest(file);assert h==summary['traces'][scenario]
            traces[str(file.relative_to(HERE))]=h
            with np.load(file) as z:
                data=z['trace'];ages=z['aoi_after'];modes=z['modes'];beta=z['resource_fractions']
                np.testing.assert_array_equal(z['fields'],FIELDS);np.testing.assert_array_equal(z['seeds'],EVAL_SEEDS)
            assert data.shape==(600,20,len(FIELDS))
            gid=np.asarray([next(g for t,g in reversed(schedule) if t<=s) for s in range(600)])
            np.testing.assert_array_equal(data[...,5],np.broadcast_to(gid[:,None],data.shape[:2]))
            audit_trace(data,ages,cfg)
            actual=summarize(data)
            for k,v in actual.items():np.testing.assert_allclose(v,summary['scenarios'][scenario]['overall'][k],atol=1e-8,rtol=1e-10)
            per[item][scenario]=actual;parts.append(data);episodes+=20;slots+=12000
            if method==METHODS[1]:
                now=(data[...,1:5],ages,modes,beta)
                if hidden_reference is None:hidden_reference=tuple(x.copy() for x in now)
                for a,b in zip(now,hidden_reference):np.testing.assert_array_equal(a,b)
            if len(schedule)>1:
                for j,(start,g) in enumerate(schedule):
                    end=schedule[j+1][0] if j+1<len(schedule) else 600
                    per[item][f'{scenario}/segment_{j}_g{g}']=summarize(data[start:end])
                    if j:
                        per[item][f'{scenario}/first20_after_{start}']=summarize(data[start:min(start+20,end)])
                        if end>start+20:per[item][f'{scenario}/later_after_{start}']=summarize(data[start+20:end])
        merged=np.concatenate(parts);per[item]['overall']=summarize(merged)
        for g in range(3):per[item][f'true_instruction_{g}']=summarize(merged[merged[...,5]==g])
        # Equal task weight changes only the aggregation, not per-slot preferences.
        per[item]['equal_instruction_utility']={k:float(np.mean([per[item][f'true_instruction_{g}'][k] for g in range(3)])) for k in ['objective_ref10','objective_ref8']}
    grouped={};paired={};hidden={};rule_gaps={}
    for o in OBJECTIVES:
        grouped[o]={};rule_gaps[o]={};hidden[o]={}
        for method in METHODS+RULE_METHODS:
            ids=[f'{o}/seed_{s}/{method}' for s in SEEDS] if method in METHODS else [f'{o}/rules/{method}']
            grouped[o][method]={scope:{k:stats([per[i][scope][k] for i in ids]) for k in per[ids[0]][scope] if k!='slots'} for scope in per[ids[0]]}
        scopes=per[f'{o}/seed_85/{METHODS[0]}']
        for scope,metrics in scopes.items():
            hidden[o][scope]={k:stats([per[f'{o}/seed_{s}/{METHODS[0]}'][scope][k]-per[f'{o}/seed_{s}/{METHODS[1]}'][scope][k] for s in SEEDS]) for k in metrics if k!='slots'}
        for rule in RULE_METHODS:
            rule_gaps[o][rule]={scope:{k:stats([per[f'{o}/seed_{s}/{METHODS[0]}'][scope][k]-per[f'{o}/rules/{rule}'][scope][k] for s in SEEDS]) for k in metrics if k!='slots'} for scope,metrics in scopes.items()}
    for method in METHODS:
        scopes=per[f'ref8/seed_85/{method}']
        paired[method]={scope:{k:stats([per[f'ref8/seed_{s}/{method}'][scope][k]-per[f'ref10/seed_{s}/{method}'][scope][k] for s in SEEDS]) for k in metrics if k not in ['slots','common_reward']} for scope,metrics in scopes.items()}
    out=HERE/'report';write(out/'results.json',dict(per_item=per,grouped=grouped,
        paired_ref8_minus_ref10=paired,paired_explicit_minus_hidden=hidden,paired_RL_minus_rules=rule_gaps))
    assert episodes==4680 and slots==2808000
    provenance=read(HERE/'provenance.json')
    for f,h in provenance['parent_input_hashes'].items():assert digest(PARENT/f)==h
    for f in (HERE/'source').rglob('*'):
        if f.is_file():assert digest(f)==digest(PARENT/'source'/f.relative_to(HERE/'source'))
    verify()
    audit=dict(state='PASS',episodes=episodes,slots=slots,both_scores_independently_recomputed=True,
        reward_components_verified=True,external_pairing=True,instruction_timing=True,
        hidden_physical_invariance=True,zero_physical_violations=True,input_model_trace_hashes=True,
        production_source_identical=True,old_training_reused=True,new_training_steps=6000000)
    write(out/'audit.json',audit);write(out/'trace_hashes.json',traces);write(out/'model_hashes.json',model_hashes)
    lines=['# AoI归一化参考8的实际训练结果','',
        '新训练6个模型，每个100万步；ref10复用同种子同预算旧末次模型。独立新测试4680回合。两组尾部阈值都为4，仅aoi_reward_ref=10/8不同。',
        '两列分数分别按固定参考值重新计分，显示乘100；必须比较同一列。新参数8使同一状态的AoI成本增加25%，不代表AoI状态本身增加。','',
        '| 训练/执行参考 | 方法 | ref10分数×100 ↑ | ref8分数×100 ↑ | AoI ↓ | 预测交付PSNR ↑ | 更新/槽 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for o in OBJECTIVES:
        for method in METHODS+RULE_METHODS:
            v=grouped[o][method]['overall'];get=lambda k:v[k]['mean']
            lines.append(f'| {o} | {method} | {100*get("objective_ref10"):.4f} | {100*get("objective_ref8"):.4f} | {get("mean_aoi"):.4f} | {get("delivered_predicted_psnr"):.4f} | {get("deliveries_per_slot"):.4f} |')
    lines+=['','## 同评分下的新训练减旧训练','',
        '| 方法 | ref8分数差×100，均值±SD | 正差种子 | AoI差 | PSNR差 |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        v=paired[method]['overall'];d=v['objective_ref8']
        lines.append(f'| {method} | {100*d["mean"]:+.4f} ± {100*d["sd"]:.4f} | {d["positive_count"]}/3 | {v["mean_aoi"]["mean"]:+.4f} | {v["delivered_predicted_psnr"]["mean"]:+.4f} |')
    lines+=['','## 固定指令下的新RL与新规则','',
        '| 指令 | 方法 | ref8分数×100 ↑ | AoI ↓ | 预测交付PSNR ↑ |','|---|---|---:|---:|---:|']
    for g,name in enumerate(['均衡','AoI','质量']):
        for method in ['IC_HAPPO','R_instruction','R_myopic']:
            v=grouped['ref8'][method][f'fixed_{g}'];get=lambda k:v[k]['mean']
            lines.append(f'| {name} | {method} | {100*get("objective_ref8"):.4f} | {get("mean_aoi"):.4f} | {get("delivered_predicted_psnr"):.4f} |')
    lines+=['','## ref8显式RL相对基线','']
    for rule in RULE_METHODS:
        d=rule_gaps['ref8'][rule]['overall']['objective_ref8']
        lines.append(f'- 相对{rule}：ref8分数差×100={100*d["mean"]:+.4f}，正差{d["positive_count"]}/3种子。')
    d=hidden['ref8']['overall']['objective_ref8']
    lines.append(f'- 相对隐藏指令RL：ref8分数差×100={100*d["mean"]:+.4f}，正差{d["positive_count"]}/3种子。')
    lines+=['','这些是三训练种子的探索性结果，不等同于统计显著性。规则的当前信息和模型计算条件与RL并非完全相同。分段/切换窗口/当前指令统计见完整结果；切换收益还受切换前状态影响。',
        '',f'审计：{episodes}回合、{slots}时隙，全部独立复算通过。',
        '',f'[完整结果]({out}/results.json)；[验证审计]({out}/audit.json)；[协议]({HERE}/PROTOCOL.md)。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'artifact_hashes.json',{str(p.relative_to(HERE)):digest(p) for p in [out/'results.json',out/'audit.json',out/'trace_hashes.json',out/'model_hashes.json',HERE/'REPORT.md']})
    print('Aggregation PASS',episodes,slots,flush=True)


if __name__=='__main__':main()
