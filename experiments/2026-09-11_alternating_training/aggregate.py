"""Verify completed validation traces and report all seeds without selection."""
from helpers import *
from evaluation import FIELDS,summarize

def main():
    m=verify();results={};pairing=None;episodes=0
    for job in m['evaluation_jobs']:
        folder=HERE/job['output'];status=read(folder/'status.json')
        assert status['state']=='complete' and digest(folder/'summary.json')==status['summary_sha256']
        summary=read(folder/'summary.json')
        if pairing is None:pairing=summary['pairing']
        assert summary['pairing']==pairing,'External episode pairing mismatch'
        all_data=[];scenes={};switches={}
        for name,schedule in m['scenarios'].items():
            file=folder/f'{name}.npz';assert digest(file)==summary['traces'][name]
            with np.load(file,allow_pickle=False) as trace:
                assert list(trace['fields'])==FIELDS
                np.testing.assert_array_equal(trace['seeds'],m['evaluation_seeds'])
                data=trace['trace'];assert data.shape==(600,20,len(FIELDS)) and np.isfinite(data).all()
                for key in ['quality_violations','budget_violations','cache_violations']:
                    assert (data[...,FIELDS.index(key)]==0).all()
                np.testing.assert_allclose(trace['resource_fractions'].sum(-1),1,rtol=0,atol=1e-12)
                assert (trace['resource_fractions']>=.05-1e-12).all()
                assert (trace['unused_budgets']>=-1e-8).all()
                ages=trace['aoi_after'].reshape(600,20,-1)
                np.testing.assert_allclose(ages.mean(-1),data[...,FIELDS.index('mean_aoi')],rtol=0,atol=1e-12)
                components={f:data[...,FIELDS.index(f)] for f in FIELDS}
                expected=(components['quality_credit']-components['age_mean_cost']-components['age_max_cost']
                    -components['age_tail_cost']-components['resource_cost']-components['service_violation_cost']+components['recv_aoi_bonus'])
                np.testing.assert_allclose(expected,components['common_reward'],rtol=0,atol=1e-10)
                scenes[name]=summarize(data);all_data.append(data.copy())
                switches[name]={str(t):summarize(data[t:min(t+20,600)]) for t,g in schedule[1:]}
        joined=np.concatenate(all_data,axis=0)
        gid=joined[...,FIELDS.index('instruction_id')].astype(int)
        results[job['id']]=dict(overall=summarize(joined),scenarios=scenes,
            true_instruction={str(g):summarize(joined[gid==g]) for g in range(3)},
            first20_after_switch=switches,
            per_validation_seed_score=(joined[...,FIELDS.index('common_reward')].mean(0)*100).tolist(),
            diagnostic_equal_resources=summary['diagnostic_equal_resources'])
        episodes+=status['completed_episodes']
    assert episodes==m['evaluation_episodes']
    paired=[]
    for seed in m['seeds']:
        a=results[f'seed_{seed}/alternating_at_1000000'];b=results[f'seed_{seed}/joint_at_1000000']
        paired.append(dict(seed=seed,alternating_minus_joint=100*(a['overall']['common_reward']-b['overall']['common_reward']),
            per_validation_seed_difference=(np.asarray(a['per_validation_seed_score'])-b['per_validation_seed_score']).tolist()))
    write(HERE/'report/results.json',dict(split='new_validation',per_item=results,paired_final_differences=paired))
    lines=['# 新三种子联合训练与交替训练验证结果','',
        '所有模型从头训练100万环境步；三个种子在运行前随机确定，全部报告。13场景×20个新验证种子×600时隙。奖励×100，越大越好。预留最终测试集未使用。','',
        '| 训练种子 | 同预算联合训练 | 交替训练 | 交替减联合 |','|---|---:|---:|---:|']
    for row in paired:
        seed=row['seed'];a=results[f'seed_{seed}/alternating_at_1000000']['overall']['common_reward']*100
        b=results[f'seed_{seed}/joint_at_1000000']['overall']['common_reward']*100
        lines.append(f"| {seed} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")
    lines+=['','| 方法 | 综合分数 | 全程均衡 | 全程AoI | 全程质量 |','|---|---:|---:|---:|---:|']
    for arm in METHODS:
        items=[results[f'seed_{s}/{arm}_at_1000000'] for s in m['seeds']]
        values=[np.mean([v['overall']['common_reward'] for v in items])*100]
        values += [np.mean([v['scenarios'][f'fixed_{g}']['common_reward'] for v in items])*100 for g in range(3)]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    for rule in RULE_METHODS:
        item=results['rules/'+rule]
        values=[item['overall']['common_reward']*100]+[item['scenarios'][f'fixed_{g}']['common_reward']*100 for g in range(3)]
        lines.append('| '+rule+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '交替训练的20万、40万步检查点只用于“均分资源下的选模诊断”，不能声称完整SUT已经训练。其他检查点为网络完整执行，详见report/results.json。',
        '', '当前检验的是训练方式，不是显式指令优于隐藏指令的验证。短训、验证集和三个训练种子的结果不能直接替代最终测试结论；报告负结果和专项取舍。','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    audit=dict(state='PASS',episodes=episodes,all_external_pairings_equal=True,all_physical_and_reward_checks_pass=True,
        final_test_used=False,results_sha256=digest(HERE/'report/results.json'),report_sha256=digest(HERE/'REPORT.md'))
    write(HERE/'report/audit.json',audit);print(json.dumps(audit))

if __name__=='__main__':main()
