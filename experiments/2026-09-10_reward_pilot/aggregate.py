"""Reaggregate every saved episode; compare reward designs on both standards."""
from helpers import *
from evaluation import FIELDS,summarize


def stats(v):
    return dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)) if len(v)>1 else None,
                values=list(map(float,v)),positive_count=int(np.sum(np.asarray(v)>0)))


def audit_trace(data,age,cfg):
    x={f:data[...,i] for i,f in enumerate(FIELDS)};e=cfg['env_args'];age=age.reshape(*data.shape[:2],-1).astype(float)
    assert data.shape[:2]==age.shape[:2] and np.isfinite(data).all()
    assert not np.any(data[:,:,6:9])
    w=np.asarray(e['reward_weights_by_instruction'])[x['instruction_id'].astype(int)]
    q=(x['predicted_quality_sum']-e['Q_min_eval']*x['deliveries'])/(e['Q_max']-e['Q_min_eval'])/e['n_ds']
    assert np.all(q>=0)
    cost=w[...,2]*e['reward_load_scale']*x['channel_uses']/e['reward_load_ref']
    for threshold,key in [(10,'objective_tail10'),(4,'objective_tail4')]:
        af=.4*age.mean(-1)/10+.3*age.max(-1)/10+.3*np.maximum(age-threshold,0).mean(-1)/10
        reward=w[...,0]*q-w[...,1]*af-cost
        np.testing.assert_allclose(reward,x[key],atol=1e-9,rtol=0)
    expected=x['objective_tail10' if e['aoi_tail_threshold']==10 else 'objective_tail4']
    np.testing.assert_allclose(expected,x['common_reward'],atol=1e-9,rtol=0)
    np.testing.assert_allclose(age.mean(-1),x['mean_aoi'],atol=1e-12,rtol=0)
    for threshold,key in [(4,'fraction_above4'),(6,'fraction_above6')]:
        np.testing.assert_allclose((age>threshold).mean(-1),x[key],atol=1e-12,rtol=0)


def main():
    m=verify();all_items=[j['id'] for j in m['jobs']]+[f'{o}/rules/{r}' for o in OBJECTIVES for r in RULE_METHODS]
    per_item={};global_pairing=None;slots=episodes=0;all_trace_hashes={}
    for item in all_items:
        objective,category,method=item.split('/');folder=HERE/'evaluation'/item
        s=read(folder/'status.json');assert s['state']=='complete' and digest(folder/'summary.json')==s['summary_sha256']
        summary=read(folder/'summary.json')
        if global_pairing is None:global_pairing=summary['pairing']
        assert summary['pairing']==global_pairing
        cfg=config(objective,method=='HAPPO_hidden_instruction')
        parts=[];per_item[item]={};hidden_reference=None
        for scenario in m['scenarios']:
            path=folder/f'{scenario}.npz';assert digest(path)==summary['traces'][scenario]
            all_trace_hashes[str(path.relative_to(HERE))]=digest(path)
            with np.load(path) as z:
                data=z['trace'];age=z['aoi_after'];mode=z['modes'];beta=z['resource_fractions']
                np.testing.assert_array_equal(z['fields'],FIELDS);np.testing.assert_array_equal(z['seeds'],m['evaluation_seeds'])
            assert data.shape==(600,len(m['evaluation_seeds']),len(FIELDS))
            schedule=m['scenarios'][scenario]
            expected=np.asarray([next(g for t,g in reversed(schedule) if t<=slot) for slot in range(600)])
            np.testing.assert_array_equal(data[:,:,5],np.broadcast_to(expected[:,None],data[:,:,5].shape))
            audit_trace(data,age,cfg)
            actual=summarize(data)
            for k,v in actual.items():np.testing.assert_allclose(v,summary['scenarios'][scenario]['overall'][k],atol=1e-8,rtol=1e-10)
            if method=='HAPPO_hidden_instruction':
                if hidden_reference is None:hidden_reference=(data[:,:,1:5].copy(),age.copy(),mode.copy(),beta.copy())
                for v,ref in zip((data[:,:,1:5],age,mode,beta),hidden_reference):np.testing.assert_array_equal(v,ref)
            per_item[item][scenario]=actual;parts.append(data);episodes+=len(m['evaluation_seeds']);slots+=600*len(m['evaluation_seeds'])
        merged=np.concatenate(parts);per_item[item]['overall']=summarize(merged)
        for g in range(3):per_item[item][f'true_instruction_{g}']=summarize(merged[merged[:,:,5]==g])
    metrics=[k for k in next(iter(per_item.values()))['overall'] if k!='slots']
    grouped={};paired={}
    for objective in OBJECTIVES:
        grouped[objective]={}
        for method in METHODS+RULE_METHODS:
            ids=[f'{objective}/seed_{s}/{method}' for s in SEEDS] if method in METHODS else [f'{objective}/rules/{method}']
            grouped[objective][method]={scope:{k:stats([per_item[i][scope][k] for i in ids]) for k in metrics} for scope in per_item[ids[0]]}
    for method in METHODS:
        paired[method]={scope:{k:stats([per_item[f'candidate_tail4/seed_{s}/{method}'][scope][k]-per_item[f'original_tail10/seed_{s}/{method}'][scope][k] for s in SEEDS]) for k in metrics if k!='common_reward'} for scope in per_item[f'original_tail10/seed_85/{method}']}
    instruction_pairs={o:{scope:{k:stats([per_item[f'{o}/seed_{s}/IC_HAPPO'][scope][k]-per_item[f'{o}/seed_{s}/HAPPO_hidden_instruction'][scope][k] for s in SEEDS]) for k in metrics} for scope in per_item[f'{o}/seed_85/IC_HAPPO']} for o in OBJECTIVES}
    out=HERE/'report';write(out/'results.json',dict(per_item=per_item,grouped=grouped,paired_candidate_minus_original=paired,paired_explicit_minus_hidden=instruction_pairs))
    audit=dict(state='PASS',episodes=episodes,slots=slots,all_traces_reaggregated=True,external_pairing=True,
               both_objectives_independently_recomputed=True,hidden_physical_invariance=True,input_hashes_verified=True)
    assert episodes==m['evaluation_episodes'];write(out/'audit.json',audit);write(out/'trace_hashes.json',all_trace_hashes)
    lines=['本轮奖励对照实验完成。12个模型，各100万步；两套目标分别为尾部阈值10和4。每套目标评估6个模型和3个规则。','',
           '两列效用按同一轨迹分别使用原评分和候选评分计算，可以横向比较同一列；不要直接比较不同目标的各自总分。RL为三种子均值，规则不重复为训练种子。','',
           '| 训练/执行目标 | 方法 | 原效用 tail10 | 候选效用 tail4 | AoI | 交付PSNR | 更新/槽 | A>4比例 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for o in OBJECTIVES:
        for method in METHODS+RULE_METHODS:
            v=grouped[o][method]['overall']
            lines.append(f'| {o} | {method} | {v["objective_tail10"]["mean"]:.6f} | {v["objective_tail4"]["mean"]:.6f} | {v["mean_aoi"]["mean"]:.4f} | {v["delivered_predicted_psnr"]["mean"]:.4f} | {v["deliveries_per_slot"]["mean"]:.4f} | {100*v["fraction_above4"]["mean"]:.4f}% |')
    lines+=['','候选训练减原奖励训练（同训练种子配对，三种子均值±标准差）：','', '| 方法 | 原效用差 | 候选效用差 | 候选效用正差种子 | AoI差 | PSNR差 |','|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        v=paired[method]['overall'];fmt=lambda k:f'{v[k]["mean"]:+.6f} ± {v[k]["sd"]:.6f}'
        lines.append(f'| {method} | {fmt("objective_tail10")} | {fmt("objective_tail4")} | {v["objective_tail4"]["positive_count"]}/3 | {fmt("mean_aoi")} | {fmt("delivered_predicted_psnr")} |')
    lines+=['','这是一轮固定预算的候选验证，3个训练种子不等同于统计显著性，旧10M结果不是1M对照。本轮不自动延长训练。',
            '',f'审计通过：{episodes}回合、{slots}时隙，两套效用独立复算、外生轨迹配对、隐藏策略物理不变性及输入哈希检查通过。',
            '',f'完整数值：[results.json]({out}/results.json)。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'artifact_hashes.json',{str(p.relative_to(HERE)):digest(p) for p in [out/'results.json',out/'audit.json',out/'trace_hashes.json',HERE/'REPORT.md']})
    print('Aggregation PASS',episodes,slots,flush=True)


if __name__=='__main__':main()
