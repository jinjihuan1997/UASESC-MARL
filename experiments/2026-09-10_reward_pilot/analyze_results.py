"""Post-run interpretation; no changes to frozen training or scoring inputs."""
import hashlib
from helpers import *
from aggregate import audit_trace
from evaluation import summarize


def main():
    manifest=verify()
    assert read(HERE/'status.json')['state']=='complete'
    for f,h in read(HERE/'report/artifact_hashes.json').items():assert digest(HERE/f)==h
    results=read(HERE/'report/results.json');groups=results['grouped']
    profile=np.load(HERE/'source/reference/inputs/profile.npz')
    q,load=profile['q_hat_mean'],profile['bar_ls_main_mean']
    mode_groups=[]
    for mode in range(len(q)):
        for group in mode_groups:
            if np.array_equal(q[mode],q[group[0]]) and np.array_equal(load[mode],load[group[0]]):
                group.append(mode);break
        else:mode_groups.append([mode])
    assert [0,4,8,12] in mode_groups and [5,9,13] in mode_groups
    mode_counts={};slots=0
    for f,h in read(HERE/'report/trace_hashes.json').items():
        path=HERE/f;assert digest(path)==h
        relative=path.relative_to(HERE/'evaluation');objective,category,method,filename=relative.parts
        scenario=Path(filename).stem
        cfg=config(objective,method=='HAPPO_hidden_instruction')
        with np.load(path) as z:data=z['trace'];ages=z['aoi_after'];modes=z['modes']
        audit_trace(data,ages,cfg)
        computed=summarize(data);expected=results['per_item'][f'{objective}/{category}/{method}'][scenario]
        for k,v in computed.items():np.testing.assert_allclose(v,expected[k],atol=1e-9,rtol=1e-10)
        slots+=data.shape[0]*data.shape[1]
        if method in METHODS and scenario.startswith('fixed_'):
            key=f'{objective}/{method}/{scenario}'
            count=np.bincount(modes[modes>=0],minlength=16)
            mode_counts[key]=mode_counts.get(key,np.zeros(16,dtype=int))+count
    modes={k:{str(g):float(v[g].sum()/v.sum()) for g in mode_groups if v[g].sum()} for k,v in mode_counts.items()}
    responses={};training={};rule_gaps={};same_score_instruction_gaps={}
    for objective in OBJECTIVES:
        responses[objective]={}
        for method in METHODS:
            values=[]
            for seed in SEEDS:
                item=f'{objective}/seed_{seed}/{method}';v=results['per_item'][item]
                values.append(dict(seed=seed,
                    quality_minus_aoi_psnr=v['fixed_2']['delivered_predicted_psnr']-v['fixed_1']['delivered_predicted_psnr'],
                    quality_minus_aoi_age=v['fixed_2']['mean_aoi']-v['fixed_1']['mean_aoi']))
            responses[objective][method]=dict(per_seed=values,
                mean_psnr_difference=float(np.mean([v['quality_minus_aoi_psnr'] for v in values])),
                mean_age_difference=float(np.mean([v['quality_minus_aoi_age'] for v in values])))
            for name,window in [('first100k',slice(0,25)),('last200k',slice(-50,None))]:
                all_rows=[]
                for seed in SEEDS:
                    folder=HERE/'jobs'/objective/f'seed_{seed}'/method
                    status=read(folder/'status.json');assert status['completed_steps']==1000000
                    for f,h in status['checkpoint_hashes'].items():assert digest(folder/f)==h
                    log=folder/'training_metrics.jsonl'
                    rows=[json.loads(line)['reward_components'] for line in log.read_text().splitlines()]
                    assert len(rows)==250;all_rows.extend(rows[window])
                training[f'{objective}/{method}/{name}']={k:float(np.mean([v[k] for v in all_rows])) for k in all_rows[0]}
        full=groups[objective]['IC_HAPPO']['overall'];hidden=groups[objective]['HAPPO_hidden_instruction']['overall']
        same_score_instruction_gaps[objective]={k:full[k]['mean']-hidden[k]['mean'] for k in ['objective_tail10','objective_tail4']}
        rule_gaps[objective]={}
        for rule in RULE_METHODS:
            baseline=groups[objective][rule]['overall']['objective_tail4']['mean']
            diffs=[results['per_item'][f'{objective}/seed_{s}/IC_HAPPO']['overall']['objective_tail4']-baseline for s in SEEDS]
            rule_gaps[objective][rule]=dict(mean=float(np.mean(diffs)),values=diffs,positive_count=sum(v>0 for v in diffs))
    output=dict(state='PASS',verified_slots=slots,verified_episodes=slots//600,mode_equivalence_classes=mode_groups,
                selected_mode_fractions=modes,instruction_response=responses,training_windows=training,
                explicit_minus_hidden_same_score=same_score_instruction_gaps,explicit_minus_rules_tail4=rule_gaps,
                interpretation='Candidate shows limited short-budget improvement; not promoted to a final method.')
    write(HERE/'analysis_results.json',output)
    lines=['# 奖励阈值对照结果分析','',
           '本轮完成12个模型、每模型100万环境步，4680回合、2808000时隙评估。已重新验证冻结输入、模型、报告及234份轨迹的哈希，并从每份存档重新核算两套评分及物理汇总。参数和旧正式结果没有修改。','',
           '结论：阈值4在这次短训中有改善新鲜度和指令响应的迹象，但带指令RL的综合改善仅2/3种子为正，隐藏指令RL改善更大；尚不足以认定阈值4全面优于原奖励，或解决RL低于知指令规则的问题。','',
           '## 同一评分标准下比较','',
           '以下全部使用tail4评分，显示值乘100；原始数值保存在results.json。两组RL都是100万步，不是上一轮1000万步。','',
           '| 训练奖励 | 方法 | 得分×100 ↑ | AoI ↓ | 预测交付PSNR ↑ | 更新/槽 | A>4比例 ↓ |','|---|---|---:|---:|---:|---:|---:|']
    for objective in OBJECTIVES:
        for method in METHODS:
            v=groups[objective][method]['overall']
            lines.append(f'| {objective} | {method} | {100*v["objective_tail4"]["mean"]:.4f} | {v["mean_aoi"]["mean"]:.4f} | {v["delivered_predicted_psnr"]["mean"]:.4f} | {v["deliveries_per_slot"]["mean"]:.4f} | {100*v["fraction_above4"]["mean"]:.4f}% |')
    lines+=['','新奖励的显式RL相对原奖励：得分提高约0.438显示单位，AoI减少0.047槽，交付PSNR平均减少0.210dB，A>4比例约减半。三个配对种子的原始效用变化为-0.001688、+0.004370、+0.010446；不是稳定的三个种子全面改善。','',
            '隐藏RL的得分三种子均提高，平均AoI减少0.441槽、PSNR减少1.880dB。其物理轨迹依旧对指令不变，说明改善来自更好的通用策略，而不是隐藏策略恢复了指令识别。','',
            '同用tail4标准，显式减隐藏的原始效用差由0.008429缩至0.003746；新奖励中显式仍在3/3种子高于隐藏。这与“显式输入没有用”不同，也不意味着整个新设计提高了显式优势。','',
            '## 指令响应有增强，质量任务仍未学充分','',
            '| 奖励 | 固定指令 | AoI | PSNR | 中档模式5/9/13占比 |','|---|---|---:|---:|---:|']
    for objective in OBJECTIVES:
        for g,label in enumerate(['均衡','AoI','质量']):
            v=groups[objective]['IC_HAPPO'][f'fixed_{g}']
            mid=modes[f'{objective}/IC_HAPPO/fixed_{g}'].get('[5, 9, 13]',0)
            lines.append(f'| {objective} | {label} | {v["mean_aoi"]["mean"]:.4f} | {v["delivered_predicted_psnr"]["mean"]:.4f} | {100*mid:.2f}% |')
    lines+=['','显式RL的质量指令减AoI指令：平均PSNR差由0.236dB扩大到0.854dB，平均AoI差由0.033槽扩大到0.102槽。三个种子的PSNR响应幅度均比原奖励更大；因此不能把新策略说成完全不分指令。','',
            '但新奖励的质量指令仍有约67.13%的有效UAV模式执行落在低档0/4/8/12，中档只有32.87%；AoI指令低档占89.73%。新隐藏RL所有任务低档占84.58%，相比旧隐藏的46.18%明显增加。这支持“短训更偏向低档快更新”的行为解释，不能据此断言唯一学习原因。','',
            '整体PSNR稍降不代表质量指令的PSNR下降：质量指令本身由25.928升至26.070dB，主要的平均质量下降发生在AoI/均衡时段；每种指标应按对应任务解释。','',
            '## 相对规则的差距','',
            '| 新奖励下方法 | tail4得分×100 ↑ |','|---|---:|']
    for method in ['IC_HAPPO']+RULE_METHODS:
        lines.append(f'| {method} | {100*groups["candidate_tail4"][method]["overall"]["objective_tail4"]["mean"]:.4f} |')
    lines+=['','新显式RL总体优于单一固定规则的均值，但只有2/3训练种子超过它；相对按指令选档和单步贪心规则，3/3种子仍落后。单步规则依赖当前全局信息和一步模型，仍需说明信息条件。','',
            '新奖励固定质量任务：RL PSNR约26.070dB，选档规则29.808dB；其tail4效用分别0.082693与0.117444。候选奖励自身仍明显偏好规则的质量任务行为，所以不能说新评分把质量目标取消了。','',
            '## 新惩罚是否进入训练','',
            '显式RL的前10万步平均尾部扣分：阈值10约0.0000255，阈值4约0.0103322；最后20万步分别约0.0000002和0.0029660。新惩罚已经进入随机动作训练，并在训练后期仍有作用。确定性最终评估时新显式RL平均尾部扣分只有0.0000409；评估罚项小不代表训练从未受罚。','',
            '当前差距不能归结为“罚项没有接入”。执行正确性已通过，科学效果依然有限。这一试验只验证了尾部阈值10→4，没有验证所有可能奖励设计，也没有进行长期收敛比较。','',
            '## 后续建议','',
            '保留统一评分接口和分项日志；阈值4继续作为候选，不直接替换论文最终设置，也不继续按本测试集加大惩罚。下一步优先定位质量指令下低档占比仍高的原因，可对固定末次模型分别替换模式/资源，以及对比随机采样和确定性执行，评价同一原始目标与物理结果。','',
            '若要判断新奖励的长期效果，应另外预先冻结原/新两套相同长训练预算、从头初始化或明确记录恢复方案，并采用新外部评估种子；不能拿本轮新1M与旧10M直接比较。这次分析没有自动续训。','',
            f'完整分析数值：[analysis_results.json]({HERE}/analysis_results.json)；正式数据：[results.json]({HERE}/report/results.json)；协议：[PROTOCOL.md]({HERE}/PROTOCOL.md)。']
    (HERE/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    # Check that analysis did not mutate any frozen input or official artifact.
    verify()
    for f,h in read(HERE/'report/artifact_hashes.json').items():assert digest(HERE/f)==h
    write(HERE/'analysis_hashes.json',{p.name:digest(p) for p in [HERE/'analyze_results.py',HERE/'ANALYSIS.md',HERE/'analysis_results.json']})
    print(json.dumps(dict(state='PASS',slots=slots,episodes=slots//600,responses=responses,
                         rule_gaps=rule_gaps),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
