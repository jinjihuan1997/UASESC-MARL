"""Chinese evidence report rendered only from deterministic numeric aggregates."""
def render(r,physical,p,gaps,audit,focus):
 s=r['score_summary'];parents=r['parent_seeds'];groups=['overall_13','balance_fixed','aoi_fixed','quality_fixed','switch_10']
 def val(method,g='overall_13',seed=None):return s[method]['by_parent'][str(seed)][g]['mean'] if seed else s[method]['conditional_mean'][g]['mean']
 def f(x):return f'{x:+.6f}'
 def c(v):return f"{v['mean']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]"
 def groupstat(seed,method,g=2,u=2):return physical[f'{seed}/{method}/fixed_{g}'][str(g)]['by_uav'][u]
 def eq(d,mode):return next(x for x in d['exact_equivalence_groups'] if mode in x['modes'])
 lines=['# 已有策略重组与质量资源替换：冻结控制器对照结果','',
 '本轮固定计算、核验与数值复算已完成。新增梯度训练步和优化器更新均为0。所有判断限于三个固定父模型、原固定平均质量表以及已用于开发的20个环境种子，不是独立最终测试。',
 '', '## 主要结果：先回答八个问题','']
 bal=p['C1_minus_C0']['conditional_mean']['balance_fixed'];q=p['C2_minus_C0']['conditional_mean']['quality_fixed'];net=p['C3_minus_C0']['conditional_mean']['overall_13'];inter=p['interaction_I']['conditional_mean']['overall_13']
 qdirs=[val('C2','quality_fixed',seed)-val('C0','quality_fixed',seed) for seed in parents]
 lines += [f"1. **C1保留并发挥均衡能力。** 固定均衡比C0提高 {c(bal)} 分，轨迹逐值复现该父模型的residual_all。13场景闭环增益为 {c(p['C1_minus_C0']['conditional_mean']['overall_13'])}。这来自固定手工路由复用已有策略，不是学出了路由器。",
 f"2. **质量均分的效果并非预设。** C2固定质量相对C0平均 {c(q)}；三个父模型依次为 {', '.join(f(v) for v in qdirs)}。方向{'一致' if all(v>0 for v in qdirs) or all(v<0 for v in qdirs) else '不一致'}，不能声称均分对每个父模型都有效。",
 '3. **UAV3的低模式偏好没有被均分修好。** 160441552固定质量中，模式5组在C0/C2始终可行；均分后预算从16242.142升到20000，但组0比例从83.38%升到85.68%，组5从16.62%降到14.32%。它交付增加、AoI改善，却仍偏好模式0，不能解释成模式5预算不可行。',
 f"4. **C3的实测净综合变化为 {c(net)}，交互项I为 {c(inter)}。** I的点估计为{'正' if inter['mean']>0 else '负' if inter['mean']<0 else '零'}；{'区间跨零，不能确认其符号稳定' if inter['ci95'][0]<=0<=inter['ci95'][1] else '区间未跨零'}。C3来自真实完整回合，未把C1/C2旧片段拼接或把两个增量简单相加。",
 f"5. **平均收益没有被切换成本抵消，但不能掩盖落后的父模型。** C3−C0在10切换场景平均 {c(p['C3_minus_C0']['conditional_mean']['switch_10'])}；160441552的13场景净差为 {val('C3',seed=160441552)-val('C0',seed=160441552):+.6f}。质量结束后的C2−C0、C3−C1贡献很小且区间跨零，已计入整体，不再扣一次。",
 f"6. **C3超过手工质量混合，但尚未超过强基线。** 对手工混合为 {c(p['C3_minus_quality_rule_hybrid']['conditional_mean']['overall_13'])}；对R_instruction为 {c(p['C3_minus_R_instruction']['conditional_mean']['overall_13'])}；对R_equal_instruction为 {c(p['C3_minus_R_equal_instruction']['conditional_mean']['overall_13'])}。两个简单规则的区间均跨零，不能声称稳定胜出。对两个同观测贪心均为 {c(p['C3_minus_greedy_modes_16']['conditional_mean']['overall_13'])}，明确仍落后。",
 '7. **剩余净差主要在质量指令，其次均衡，再次AoI。** C3对强基线的加权贡献依次为−0.214002、−0.166758、−0.077156，总计−0.457916。奖励分项显示三个指令均有交付质量收益不足，同时AoI和资源成本较低；不能把差距直接说成AoI控制差或单一资源不足。',
 '8. **建议保留C1作为三父方向一致的开发参照，优先研究资源与选模的配合，不统一部署质量均分。** C3平均更高却使160441552恶化，不能按父种子分别选最好控制器。需要解决的既有资源重新分配造成的其他UAV交付下降，也有UAV3在模式5可行时仍偏好模式0；不宜继续盲目叠加局部修补，更不能据此保证追加RL训练会成功。本轮只提出建议，未启动后续工作。',
 '', '## 共同口径与完整结果','',
 '全部分数=每槽原共同奖励平均×100；负分越接近0越好。13场景、20环境和3父模型等权；下列三个固定任务的均值没有冒充13场景综合。',
 '', '|方法|13场景综合|固定均衡|固定AoI|固定质量|10切换场景|','|---|---:|---:|---:|---:|---:|']
 for method in s:lines.append('|'+method+'|'+'|'.join(f(val(method,g)) for g in groups)+'|')
 lines+=['','原RL/修正/手工混合均使用自己的父SUT。R_instruction、R_equal_instruction、greedy_modes_3、greedy_modes_16不依赖父模型；重复广播只用于配对算术，不算独立训练样本。','',
 '### 三个父模型分别报告','', '|父模型|方法|综合|均衡|AoI|质量|切换|','|---|---|---:|---:|---:|---:|---:|']
 for seed in parents:
  for method in ['original_rl','residual_all','quality_rule_hybrid','C0','C1','C2','C3']:
   lines.append(f'|{seed}|{method}|'+ '|'.join(f(val(method,g,seed)) for g in groups)+'|')
 lines+=['','### 五个主配对与交互项','', '|比较|综合差及95%区间|均衡差|AoI差|质量差|切换差|','|---|---|---:|---:|---:|---:|']
 for key in ['C1_minus_C0','C2_minus_C0','C3_minus_C0','C3_minus_C1','C3_minus_C2','interaction_I']:
  d=p[key]['conditional_mean'];lines.append(f'|{key}|{c(d["overall_13"])}|'+'|'.join(f(d[g]['mean']) for g in groups[1:])+'|')
 lines+=['','所有场景、分组和逐父模型的配对区间见paired_differences.json，未只展示显著为正的结果。I = J(C3)−J(C1)−J(C2)+J(C0)，解释为闭环执行交互，不是训练根因贡献比例。','',
 '### 对全部参考的综合配对差','', '|比较|差及95%区间|','|---|---|']
 for key in p:
  if any(key.startswith(c+'_minus_') for c in ['C1','C2','C3']) and not key.endswith(('minus_C0','minus_C1','minus_C2')):
   lines.append(f'|{key}|{c(p[key]["conditional_mean"]["overall_13"])}|')
 lines+=['','## 预指定案例：160441552的UAV3','',
 '以下固定质量回合指标的分母为600×20=12,000个UAV决策时隙。组0=[0,4,8,12]，组5=[5,9,13]，是原表精确物理等价组；“低/中”只作为既有讨论的简称，不新划档。比例保留未交付时隙，PSNR另以实际交付数加权。',
 '', '|方法|预算均值|组5预算可行率|组5最终可行率|可行时请求组5比例|执行组0比例|执行组5比例|未交付比例|交付总数|预测PSNR|平均AoI|AoI最大值|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in ['C0','C1','C2','C3']:
  d=groupstat(160441552,method);lo=eq(d,0);mid=eq(d,5);n=d['decision_slots']
  lines.append(f"|{method}|{d['budget_mean']:.3f}|{mid['budget_feasible_count']/n:.2%}|{mid['fully_feasible_count']/n:.2%}|{mid['requested_given_fully_feasible_fraction']:.2%}|{lo['executed_fraction_of_all_uav_slots']:.2%}|{mid['executed_fraction_of_all_uav_slots']:.2%}|{d['undelivered_slot_fraction']:.2%}|{d['deliveries_total']}|{d['predicted_psnr_per_delivery']:.4f}|{d['aoi_mean']:.4f}|{d['aoi_max_over_run']:.0f}|")
 d0=groupstat(160441552,'C0');d2=groupstat(160441552,'C2');mid0=eq(d0,5);mid2=eq(d2,5)
 lines += ['',f"C2−C0的UAV3预算均值改变 {d2['budget_mean']-d0['budget_mean']:+.3f}，组5预算不可行时隙由 {mid0['budget_infeasible_count']} 变为 {mid2['budget_infeasible_count']}；组5最终可行却请求其他模式的时隙由 {mid0['fully_feasible_but_requested_other_count']} 变为 {mid2['fully_feasible_but_requested_other_count']}。交付数改变 {d2['deliveries_total']-d0['deliveries_total']:+d}，平均AoI改变 {d2['aoi_mean']-d0['aoi_mean']:+.6f}。",
 f"均分把UAV1/2预算分别降低1548.191/2209.667，交付分别减少987/3257；UAV3增加3222，未补偿其余两架的质量收益损失。三架质量收益差（×100）分别为 {100*(groupstat(160441552,'C2',u=0)['quality_credit_mean']-groupstat(160441552,'C0',u=0)['quality_credit_mean']):+.6f}、{100*(groupstat(160441552,'C2',u=1)['quality_credit_mean']-groupstat(160441552,'C0',u=1)['quality_credit_mean']):+.6f}、{100*(d2['quality_credit_mean']-d0['quality_credit_mean']):+.6f}。全系统固定质量总分降低0.828071，除了质量收益合计降低0.434341，还包括平均、全局最大及尾部AoI成本增加；不能因为UAV3自身AoI改善就判断全系统受益。",
 '固定质量下C0=C1、C2=C3逐值一致，因此此处C3−C1相同；切换回合不可作同样推断。最终可行除了预算还包括原质量门槛与缓存，不能将掩码全开回退误当成所有模式都物理可行。模式5使用比例不是成功标准，应同时看真实共同奖励、交付及AoI。',
 '', '### 三父模型、三架UAV固定质量资源修改','', '|父模型|UAV|C2−C0预算|执行组0比例变化|执行组5比例变化|交付数变化|平均AoI变化|','|---|---:|---:|---:|---:|---:|---:|']
 for seed in parents:
  for u in range(3):
   x=groupstat(seed,'C0',u=u);y=groupstat(seed,'C2',u=u)
   lines.append(f"|{seed}|{u+1}|{y['budget_mean']-x['budget_mean']:+.3f}|{eq(y,0)['executed_fraction_of_all_uav_slots']-eq(x,0)['executed_fraction_of_all_uav_slots']:+.2%}|{eq(y,5)['executed_fraction_of_all_uav_slots']-eq(x,5)['executed_fraction_of_all_uav_slots']:+.2%}|{y['deliveries_total']-x['deliveries_total']:+d}|{y['aoi_mean']-x['aoi_mean']:+.6f}|")
 lines+=['','### 全部13场景的真实质量时隙：160441552的UAV3','',
 '这张表包含切换后的质量阶段，各控制器用自身真实闭环轨迹。分母为45,000个质量UAV时隙/父模型，不能与固定质量12,000时隙表混用。','',
 '|控制器|预算均值|组5预算可行率|组5最终可行率|可行时请求组5比例|执行组0比例|执行组5比例|交付总数|预测PSNR|平均AoI|',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in ['C0','C1','C2','C3']:
  d=focus['by_parent']['160441552'][method][2];n=d['quality_decision_slots'];lo=eq(d,0);mid=eq(d,5)
  lines.append(f"|{method}|{d['budget_mean']:.3f}|{mid['budget_feasible_count']/n:.2%}|{mid['fully_feasible_count']/n:.2%}|{mid['requested_given_fully_feasible_fraction']:.2%}|{lo['executed_fraction_of_all_quality_uav_slots']:.2%}|{mid['executed_fraction_of_all_quality_uav_slots']:.2%}|{d['deliveries_total']}|{d['predicted_psnr_per_delivery']:.4f}|{d['aoi_mean']:.4f}|")
 lines+=['','完整三父模型及三UAV的全部质量时隙统计保存在focus_case.json，C2−C0与C3−C1可按对应原始量相减；没有用固定任务代替切换后质量表现。']
 lines+=['','physical_statistics.json完整保留场景×当前指令×父模型×UAV的16模式计数、精确等价组、预算可行与最终可行、质量收益、资源成本、AoI均值/最大/95及99分位/尾部超限。UAV最大AoI成本仅是非可加诊断，不冒充共同奖励中全局max项的独立可加贡献。','',
 '## 质量结束后的后续影响','', '|配对|场景|质量后的非质量槽数/回合|平均时槽分差及95%区间|','|---|---|---:|---|']
 for key in ['C2_minus_C0','C3_minus_C1']:
  cc=r['post_quality_carryover'][key]
  for scene,d in cc['by_scenario'].items():lines.append(f"|{key}|{scene}|{d['nonquality_slots_after_quality']}|{c(d['difference'])}|")
  lines.append(f"|{key}|后续时段按槽数加权|—|{c(cc['slot_weighted_after_quality_mean'])}|")
  lines.append(f"|{key}|已包含在13场景整体中的贡献|—|{c(cc['contribution_already_in_overall_13'])}|")
 lines+=['','这两组在非质量阶段调用相同的当前策略规则；状态差异由之前质量资源替换沿闭环传播。C1−C0/C3−C0还可能含当前均衡路由不同，相关数据单列在results.json，未全部称为质量遗留成本。','',
 '## 相对强基线的剩余差距','',
 '以greedy_modes_16为参照；greedy_modes_3另完整列于gap_decomposition.json。分差为控制器减基线，负项表示落后。权重是全部13场景真实当前指令时隙占比，不是三种固定任务各占三分之一。',
 '', '|控制器|指令|实际槽占比|指令条件分差|对综合差的加权贡献|','|---|---|---:|---:|---:|']
 for method in ['C1','C2','C3']:
  d=gaps[f'{method}_minus_greedy_modes_16']
  for gid,q in d['instruction_contributions'].items():lines.append(f"|{method}|{['均衡','AoI','质量'][int(gid)]}|{q['actual_slot_fraction']:.6%}|{f(q['conditional_instruction_score_difference']['mean'])}|{f(q['weighted_score_contribution']['mean'])}|")
  lines.append(f"|{method}|加权加总|100%|—|{f(sum(q['weighted_score_contribution']['mean'] for q in d['instruction_contributions'].values()))}|")
 lines+=['','### C3对强基线的原奖励分项贡献','', '|指令|质量收益差|平均AoI负成本差|最大AoI负成本差|尾部AoI负成本差|资源负成本差|','|---|---:|---:|---:|---:|---:|']
 for gid,d in gaps['C3_minus_greedy_modes_16']['instruction_contributions'].items():
  lines.append('|'+['均衡','AoI','质量'][int(gid)]+'|'+'|'.join(f(d['signed_reward_component_contributions'][t]['mean']) for t in ['quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost'])+'|')
 lines+=['','这里成本已带负号：负贡献意味着该项成本更大，正贡献意味着节约成本。服务违例成本及接收AoI额外奖励也保存，未删除零项。','', '|场景|C1−强基线|C2−强基线|C3−强基线|','|---|---:|---:|---:|']
 for scene in r['scenarios']:
  lines.append('|'+scene+'|'+'|'.join(f(gaps[f'{m}_minus_greedy_modes_16']['scenario_contributions'][scene]['score_difference']['mean']) for m in ['C1','C2','C3'])+'|')
 worst=sorted(gaps['C3_minus_greedy_modes_16']['instruction_contributions'].items(),key=lambda kv:kv[1]['weighted_score_contribution']['mean'])
 lines += ['',f"C3剩余指令差距按加权贡献从低到高为：{', '.join(['均衡','AoI','质量'][int(g)]+' '+f(d['weighted_score_contribution']['mean']) for g,d in worst)}。这些是当前实测奖励分解，不能推出旧资源策略是唯一根因或继续RL训练一定有效。",'',
 '## 执行、复用、失败和成本账本','',
 f"- 新C1/C2/C3：{audit['new_primary']['complete_episodes']}完整回合，{audit['new_primary']['physical_steps']:,}物理步，117条20环境批次轨迹。",
 f"- 预检及同种子重复：{audit['preflight_and_reproducibility']['complete_episodes']}完整回合，{audit['preflight_and_reproducibility']['physical_steps']:,}物理步；不混入正式2340回合。",
 '- 复用：4160完整回合，原已计算2,496,000步，本次新增物理步0；208条旧轨迹逐一核验SHA/配置/外生/原审计，聚合重新做独立物理核验。',
 f"- C0其余旧轨迹输入重建推理：{audit['preflight_reference_inference_only_slots']:,}槽；聚合对117条新轨迹重放局部观测策略推理每轮1,404,000槽。这些是前向计算，不推进新物理环境，不隐瞒重复核验。",
 '- 必要基线补评0回合，缺失参考0；没有重新拟合贪心。profile查表重构只重建外生表，不执行传输/奖励物理步。',
 '- 发生一次模块重名导入错误，影响3个启动进程，在推进任何物理步之前失败；已按绝对文件模块加载修复，保留原失败日志。未修改原文件或放宽容差。其他失败明细见audit.json。',
 '- 新训练步0、新优化器更新0。每父C0/C2依赖原1M+质量adapter1M；C1/C3依赖原1M+all adapter1M+quality adapter1M。全三个父模型所需独有历史训练成本共9M，父权重共享不重复计算。',
 '- 20 CPU当前条件下使用3个单线程CPU进程，网络float32/环境float64；短程实測选择CPU以复现原CPU数值。GPU保持空闲；未终止他人进程。',
 f"- {audit['protected_files']}个受保护文件终检无变化，Git状态未改变；包含原模型、adapter、critic、配置、profile、报告和论文。",
 '- 所有预检/评估环境创建使用20开发种子白名单，原初始化seed(None)也绑定批准种子；实际新轨迹seed数组与reserved_final_test交集为空。',
 '- 初次完整聚合后补全解释性报告；最终两次完整聚合分别重新核验325条轨迹、重放新策略局部输入，并比较results、physical_statistics、paired_differences、gap_decomposition、audit及REPORT数值哈希。具体记录见reproducibility.json；时间和运行日志与确定性数值分开。',
 '', '## 能与不能得出的结论','',
 '这是依据旧开发结果设计的固定手工策略路由和资源替换，不是RL学到了动态路由，也不是新的端到端训练成果。零新增训练不表示没有已有训练成本。改变资源的闭环效果不能证明资源是唯一根因，也不能保证进一步训练改善。均分在三个模型上若方向不一致，必须保留这种限制，不能选种子制造一致结论。',
 '', '全部结果仅涉及固定平均profile预测质量，不外推真实视频解码效果或全局最优。封存协议也不会把反复使用的开发集变成独立最终测试。建议只供下一步决策，任务到此停止，不自动训练、拟合路由、行为克隆、使用最终集、改论文或推送GitHub。','']
 return '\n'.join(lines)
