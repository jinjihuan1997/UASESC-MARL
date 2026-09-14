"""Deterministic report aggregation. Inputs are completed read-only analyses."""
from analysis_support import *

def pct(x):return f'{100*x:.2f}%'
def num(x):return f'{x:.6f}'
def run():
 guard();r=read(HERE/'report/results.json');o=read(HERE/'report/same_load_opportunity.json');b=read(HERE/'report/budget_capacity.json')
 old=read(LONG/'report/results.json');pairs=read(LONG/'report/paired_differences.json');gaps=read(LONG/'report/gap_decomposition.json')
 draws=np.random.default_rng(MAN['bootstrap_seed']).integers(0,20,(4000,20))
 def ci(x):
  x=np.asarray(x,dtype=float);assert x.shape==(20,)
  return dict(mean=float(x.mean()),ci95=np.quantile(x[draws].mean(-1),[.025,.975]).tolist(),environment_clusters=20)
 summary=dict(score_definition='mean frozen common reward times 100',final_comparisons={},same_load_opportunity={},quality_10M_minus_1M={},regression_components={},limits='Current fixed three models and reused development environments; no causal update attribution or new closed-loop controller results.')
 for seed in SEEDS:
  k=str(seed);model=r['models'][k];start=model['checkpoints']['1000000'];end=model['checkpoints']['10000000']
  summary['quality_10M_minus_1M'][k]=ci(np.array(end['fixed_tasks']['fixed_2']['score_by_environment'])-start['fixed_tasks']['fixed_2']['score_by_environment'])
  f=start['fixed_tasks']['fixed_2']['metrics'];t=end['fixed_tasks']['fixed_2']['metrics'];comp={}
  for name in ['quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost']:
   comp[name]=(t[name]-f[name])*100*(1 if name=='quality_credit' else -1)
  np.testing.assert_allclose(sum(comp.values()),summary['quality_10M_minus_1M'][k]['mean'],atol=1e-9,rtol=0)
  summary['regression_components'][k]=comp
  x=np.mean([s['score_opportunity_per_environment'] for s in o[k]['final_scenes'].values()],axis=0)
  summary['same_load_opportunity'][k]=dict(overall=ci(x),quality=ci(o[k]['final_scenes']['fixed_2']['score_opportunity_per_environment']))
 for method in ['retrained_rl_Qhalf','greedy_modes_16_Qhalf_local','G_equal_local16_Qhalf_local','R_instruction']:
  summary['final_comparisons'][method]=pairs['long_rl_Qhalf_minus_'+method]
 summary['same_load_opportunity']['conditional_mean']=dict(overall=ci(np.mean([np.mean([s['score_opportunity_per_environment'] for s in o[str(seed)]['final_scenes'].values()],0) for seed in SEEDS],0)),quality=ci(np.mean([o[str(seed)]['final_scenes']['fixed_2']['score_opportunity_per_environment'] for seed in SEEDS],0)))
 # Example is the first archived mode-1 choice by UAV1, no search for best reward.
 z=arrays(trace(SEEDS[0],10000000));t,e=np.argwhere((z['modes'][:,:,0]==1)&(z['served'][:,:,0].sum(-1)>0))[0]
 ob=z['post_uav_obs'][t,e,0];q1=float(ob[48]*33);q5=float(ob[52]*33);n=int(z['served'][t,e,0].sum())
 example=dict(seed=SEEDS[0],environment_seed=int(z['seeds'][e]),scene='fixed_2',slot_zero_based=int(t),uav=1,actual_mode=1,same_load_higher_quality_mode=5,actual_budget=float(z['budgets'][t,e,0]),actual_delivery_count=n,chosen_psnr_from_float32_actor_input=q1,alternative_psnr_from_float32_actor_input=q5,chosen_load_from_actor_input=float(ob[32]*100000),alternative_load_from_actor_input=float(ob[36]*100000),action_mask_allows_both=bool(z['uav_masks'][t,e,0,1] and z['uav_masks'][t,e,0,5]),exact_profile_quality_difference=4.92,algebraic_quality_score_opportunity=.35*(4.92/12)*n/30*100,interpretation='Algebraic same-load comparison at a saved state, not a newly executed controller score. Rounded example uses exact fixed-profile difference; aggregate uses float64 independent profile tables.')
 assert example['action_mask_allows_both'] and example['chosen_load_from_actor_input']==example['alternative_load_from_actor_input']
 summary['example']=example;dump(HERE/'report/summary.json',summary)
 lines=['# 长训练后的资源—选模配合与退步诊断','',
 '结论：已经发现明确的选模学习不足和跨训练阶段的策略漂移。质量任务落后不能主要解释为“中档模式被预算掩码禁止”，但“能传一个中档块”也不等于“资源分配已经合理”。当前证据支持先隔离选模与资源的作用，再考虑训练改动；没有证据支持直接再加训练步数就一定解决。','',
 '本轮没有训练、没有运行新的物理回合。读取原1/2/4/6/8/10M固定快照、54份固定任务轨迹，以及10M全部13场景中的探索性等载荷检查。冻结网络只在已保存观测上前向计算。所有分数为共同奖励每槽均值×100，越大越好；PSNR为固定平均表预测值。','',
 '## 1. 完整13场景结论保持不变','',
 '| 方法 | 综合分 | 固定均衡 | 固定AoI | 固定质量 |','|---|---:|---:|---:|---:|']
 for method in ['retrained_rl_Qhalf','long_rl_Qhalf','G_equal_local16_Qhalf_local','greedy_modes_16_Qhalf_local','R_instruction']:
  x=old['scores'][method]['conditional_mean'];lines.append('| '+method+' | '+' | '.join(num(x[k]['mean']) for k in ['overall_13','balance_fixed','aoi_fixed','quality_fixed'])+' |')
 lines+=['','长期训练相对100万步平均综合增加0.057063，95%配对区间[0.018615, 0.094668]。相对按新权重选模的强贪心仍低0.128370，区间[-0.164164, -0.095774]；质量专项低0.500036。强贪心的SUT预测器仍来自旧目标；无需该预测器的均分局部贪心也领先RL，质量专项领先0.475507。','',
 '| 训练种子 | 1M综合 | 10M综合 | 综合变化 | 质量专项变化 |','|---|---:|---:|---:|---:|']
 for seed in SEEDS:
  k=str(seed);x=old['scores']['retrained_rl_Qhalf']['by_parent'][k]['groups'];y=old['scores']['long_rl_Qhalf']['by_parent'][k]['groups'];p=pairs['long_rl_Qhalf_minus_retrained_rl_Qhalf']['by_parent'][k]['groups']
  lines.append(f'| {seed} | {x["overall_13"]["mean"]:.6f} | {y["overall_13"]["mean"]:.6f} | {p["overall_13"]["mean"]:+.6f} | {p["quality_fixed"]["mean"]:+.6f} |')
 lines+=['','## 2. 能选却没有选，与确实选不了分开','',
 '“中档”仅指原精确等价组[5,9,13]；低载荷组为[0,4,8,12]。保留全部16动作。“中档可行”要求有待传缓存、满足质量门槛且预算至少够一块，直接用float64 profile和真实预算重建；没有将全1回退掩码当作所有模式可行。','',
 '| 10M种子/UAV | 中档可行时隙 | 请求低载荷组 | 请求中档组 | 同一输入局部贪心选中档 |','|---|---:|---:|---:|---:|']
 for seed in SEEDS:
  for q in r['models'][str(seed)]['checkpoints']['10000000']['quality_physical']:
   lines.append(f'| {seed}/{q["uav"]} | {pct(q["mid_feasible_fraction"])} | {pct(q["low_selected_fraction"])} | {pct(q["mid_selected_fraction"])} | {pct(q["local_greedy_mid_on_same_inputs_fraction"])} |')
 lines+=['','以上比例分母均为该UAV的12000个固定质量决策时隙。中档可行率最低仍为99.72%。因此“预算/掩码不让选中档”只覆盖极少部分时隙，不能解释大量低载荷选择。','',
 '同一输入比较也有差异：104948945的UAV3，在RL选低载荷且中档可行的时隙中，局部贪心有98.24%选择中档。它读取完全相同的局部观测和掩码。这个对照支持存在动作偏好差异，但局部贪心评分不是实际长期回报，不能将全部分歧叫作错误。低载荷有时确实能增加交付、降低AoI。','',
 '资源仍然影响一次能交付几块。固定质量下，111868397/UAV3在原预算与原缓存状态下平均可传2.915个中档块，静态均分预算下为3.503；但104948945/UAV3会从3.517降到2.422。这里仅替换代数预算，未推进环境，不能据此给均分控制器排实际成绩。它说明不能用“平均份额接近1/3”或“中档能选”直接认定资源合理，也不能认定均分必然更好。','',
 '## 3. 更直接的问题：相同载荷却选了质量更低的模式','',
 '这是读取profile后追加的探索性检查，详见EXPLORATORY_EXTENSION.md。原profile中，模式1与5载荷逐值相同，模式5预测质量高4.92 dB；模式2与10载荷逐值相同，模式10质量高7.01 dB。还存在6→10、14→10等同载荷质量改善。比较未修改动作集或策略，只按原交付数计算缺失的质量收益。','',
 '| 10M种子 | 质量任务选择同载荷低质量模式的比例 | 质量专项代数机会损失 | 13场景代数机会损失 |','|---|---:|---:|---:|---:|']
 for seed in SEEDS:
  k=str(seed);x=o[k]['final_scenes']['fixed_2'];lines.append(f'| {seed} | {pct(x["fraction_all_uav_slots"])} | {x["score_opportunity_x100"]:.6f} | {o[k]["overall_opportunity_x100"]:.6f} |')
 lines+=['','比例分母为每个模型36000个UAV质量任务时隙。104948945的UAV1单独达到24.78%，造成质量专项0.178078分的代数机会损失。与“传更高质量会花更多资源”不同，这部分不需要增加载荷；因此确实反映未充分利用当前模式表。它不是新控制器实测收益，也不能说明修复后一定超过强基线。','',
 f'可复查例子：开发环境种子{example["environment_seed"]}，固定质量第{example["slot_zero_based"]}槽（从0计），104948945/UAV1选择模式1，预算{example["actual_budget"]:.3f}，交付{n}块。模式1和5在可见输入上载荷相同且掩码都允许，预测质量差为4.92 dB。在保持交付和资源不变的条件下，该槽质量分差为0.35×(4.92/12)×{n}/30×100={example["algebraic_quality_score_opportunity"]:.6f}。该式只算一个已保存状态的代数差，没有试探环境step。','',
 '## 4. 104948945是怎样退步的','',
 '| 累计训练步 | 104948945质量分 | 111868397质量分 | 160441552质量分 |','|---|---:|---:|---:|']
 for step in STEPS:lines.append(f'| {step//1000000}M | '+' | '.join(num(r['models'][str(seed)]['checkpoints'][str(step)]['fixed_tasks']['fixed_2']['score_x100']) for seed in SEEDS)+' |')
 lines+=['','104948945从1M到2M已出现下降，4M恢复，6M以后再次下降；它不是达到某个固定训练长度后持续单调退化。快照间隔限制了定位精度，不能确定某一次梯度更新是根因。其他两个种子也有中途回撤，最终方向不同，不能只保留上涨的种子。','',
 '| 104948945质量任务 | 1M预测PSNR | 10M预测PSNR | 1M中档请求 | 10M中档请求 |','|---|---:|---:|---:|---:|']
 x=r['models']['104948945']['checkpoints']
 for u in range(3):
  a=x['1000000']['quality_physical'][u];zq=x['10000000']['quality_physical'][u]
  lines.append(f'| UAV{u+1} | {a["predicted_psnr_delivery_weighted"]:.3f} | {zq["predicted_psnr_delivery_weighted"]:.3f} | {pct(a["mid_selected_fraction"])} | {pct(zq["mid_selected_fraction"])} |')
 cc=summary['regression_components']['104948945'];lines+=['',f'质量专项总下降0.584368分，其中质量收益变化{cc["quality_credit"]:+.6f}，平均AoI贡献{cc["age_mean_cost"]:+.6f}，最大AoI贡献{cc["age_max_cost"]:+.6f}，尾部AoI贡献{cc["age_tail_cost"]:+.6f}，资源贡献{cc["resource_cost"]:+.6f}。加总与原总奖励分差一致。','',
 'UAV2改善，UAV1/3变差，团队没有形成稳定配合。固定10M真实输入，换回1M冻结网络只做前向时，UAV1中档请求率为86.29%，10M网络为19.29%；UAV2为0%→73.53%，UAV3为73.23%→44.43%。输入保持完全相同仍发生大变化，说明差异确实包含网络更新，并非仅因信道或环境种子变化。旧网络在新输入上不是旧网络完整闭环的表现。','',
 '同一10M输入上，104948945的1M→10M SUT解析KL均值为1.0882 nats，三个UAV分别为0.4134、0.7801、0.4133。SUT资源份额逐样本绝对差均值约为3.74、13.76、11.18个百分点。资源和选模都变了；这些跨900万步累计漂移不能直接与单次PPO的KL阈值比较。','',
 '## 5. 已排除哪些简单解释，仍缺哪些证据','',
 '未发现质量样本长期缺失：各模型每100万训练步中，质量样本为31.78万至35.80万。学习率始终为actor 1e-4、critic 4e-4，四actor均更新；原断点恢复及物理核验已通过。现有日志中的损失和梯度摘要均有限，critic损失未出现明显爆炸。梯度范数是裁剪前、多个actor/小批次的混合摘要，不能拿它证明每个actor都稳定。','',
 '104948945的训练随机轨迹质量均分从前100万步窗口的-3.114改善到最后100万步窗口的-1.291，但固定质量确定性验证反而变差。两者的指令过程、内生状态和随机执行不同，因此不是同口径优劣比较；它提示“训练日志上升”不能代替固定验证。没有新的10M配对随机评估，不能量化最终随机—确定性退化，也没有证明这是唯一根因。','',
 '等价模式概率分散确实存在，但不是统一答案。104948945/UAV3的7.23%质量时隙存在“单个编号argmax选低载荷、等价组总概率最高却是中档”；其他多个UAV该比例接近0。反方向也有：111868397/UAV1确定性94.41%选择中档，但其低载荷组平均总概率反而高于中档。直接改为按组取最大概率不保证改进，本轮未修改解码。','',
 '原日志没有逐actor、逐指令的每次更新KL/clip fraction，不能证明一次更新过大或指令梯度冲突是唯一原因。此前TailRL诊断针对1M/6M冻结模型，本报告不把它的梯度结果冒充10M证据。没有改奖励、没有挑换种子；也未证明奖励公式存在执行错误。','',
 '## 6. 下一步应优先做什么','',
 '第一优先级是质量选模：明确纠正同载荷低质量选择，并区分它与有意义的质量—交付—AoI取舍。若要继续实验，应先固定当前SUT，对模式处理作单因素、完整闭环对照；不要把本轮代数机会损失当成已经完成的修复成绩。','',
 '第二优先级是资源配合：另设选模规则保持不变、只替换资源规则的对照，并与前者组成固定对照，以分开两者及其交互。当前静态容量结果已说明资源替换可能帮助一架、损害另一架，不能预设均分成功。','',
 '如果之后再训练，应先补逐actor/逐指令更新诊断，研究如何保留已学会的行为，再决定是否采用更新限制；现有累计KL不能证明某个限制一定有效。当前不建议直接再延长、继续降低质量权重，或直接上TailRL。所有建议均未自动执行。','',
 '## 7. 核验与可复算文件','',
 'analysis_support.py硬性禁止backward、optimizer.step和环境step/commit，并限制写入本目录。原模型固定eval，参数哈希一致；18份质量轨迹的原动作均已重放复现，掩码由独立float64表逐值重建通过；54份固定轨迹奖励分项重算通过。原轨迹的逐槽物理核验状态及SHA同时检查。本轮没有重新调用原物理执行器产生新轨迹。','',
 '5303个主分析保护输入前后SHA一致；额外13场景等载荷检查的78个输入SHA一致。只读取原20个开发环境轨迹，reserved集合交集为空。新增训练步、优化器更新、物理环境步均为0。网络前向和局部贪心只读查询不计为新物理回合。执行次数和耗时单列于EXECUTION.md及report/execution_audit.json。','',
 'results.json包含全部18快照的质量物理统计、概率与同输入KL，training_logs保存30个100万步窗口。summary.json含配对区间和例子；same_load_opportunity.json为探索性代数差；budget_capacity.json为静态预算容量。所有未展示模式组、UAV和负结果均保留在JSON。重复聚合核验见reproducibility.json。区间按20个环境种子成簇，13场景和三个模型共同保留；只适用于当前固定模型与反复使用的开发集，不是独立最终测试或训练总体结论。']
 text(HERE/'report/REPORT.md','\n'.join(lines)+'\n')
if __name__=='__main__':run()
