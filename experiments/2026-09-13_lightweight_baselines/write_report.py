"""Render Chinese findings directly from deterministic, audited numeric outputs."""
from light_support import *
GROUPS={'overall_13':'13场景综合','balance_fixed':'固定均衡','aoi_fixed':'固定AoI','quality_fixed':'固定质量','switch_10':'10个切换场景'}
REWARD_PARTS=['quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost','service_violation_cost','recv_aoi_bonus']

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def fmt(x,n=6):return f'{x:.{n}f}'
def cistr(d):return f"{d['mean']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]"

def main():
 m=manifest();r=read(HERE/'report/results.json');ph=read(HERE/'report/physical_statistics.json');paired=read(HERE/'report/paired_differences.json');audit=read(HERE/'report/independent_audit.json');methods=r['methods'];scores=r['scores'];ts=r['latency_summary'];groups=list(GROUPS)
 def J(method,group='overall_13'):return scores[method]['conditional_mean'][group]['mean']
 lines=['# 无离线拟合、低复杂度贪心与规则基线：实际评估报告','',
 '本轮完成四个新基线的 1,040 个完整回合（624,000 物理步），以及执行前登记的两个旧固定规则补评 520 回合（312,000 步）。新增训练、拟合和优化器更新均为 0。完整保留原强基线及三个固定父模型，未按结果修改算法。','',
 f"最有用的简化是 **G_equal_local16**：综合 {J('G_equal_local16'):.6f}，原 greedy_modes_16 为 {J('greedy_modes_16'):.6f}，配对差 {cistr(paired['G_equal_local16_minus_greedy_modes_16']['conditional_mean']['overall_13'])}。它取消了 SUT 离线预测器和资源候选评分，仍明显高于原 RL、C1、C3 的条件平均。区间不跨零，说明本开发集上存在小幅性能损失；不能称与强贪心统计等价。",'',
 '低载荷和最高单块质量规则损失较大。尤其 R_equal_minload 在所有 13 场景的请求模式及完整物理结果上与原 R_equal_single 完全一致，却多做了在线排序；它是当前平均表上的重复参照，不能包装为算法创新。','',
 '**计算定义与输入保护**','',
 '原仓库工作区位于当前项目根目录；该顶层本身不属于 Git 工作树，嵌套工作树与发布检出位置、HEAD 和已有未提交改动记录在 git_status.json。本实验全部输出位于本目录。原实验、模型、适配分支、profile、奖励、执行器、报告和论文只读。实际解析路径与 SHA-256 见 manifest.json、reference_index.json；冻结参考提交为 7cc2372d19f36ed7d91ed289f83dcc79fe0f3e9d。','',
 '每槽依次运行资源规则、原 allocate_resources、重新生成的各 UAV 局部观测/掩码、模式决策、一次原 commit_modes 与奖励。原资源动作先转换为 float64 再执行加 1 等运算，下限只应用一次。切换指令不重置缓存、AoI 或时钟。局部策略不接收 env、critic、其他 UAV 观测或未来数组。','',
 table(['方法','资源决策','UAV 决策','删除的计算或依赖'],[
  ['G_equal_local16','下限变换前 1/3 均分','原 UAVGreedy 全 16 模式原局部效用','SUT 拟合预测器与 3 组资源候选评分'],
  ['G_urgency_local16','原观测 AoI 紧急度归一化，零和回退均分','与上一行完全相同','同上，改用紧急度算术规则'],
  ['R_equal_minload','原均分','载荷升序、质量降序、ID 升序','不计算完整局部效用，不拟合网络'],
  ['R_equal_maxquality','原均分','质量降序、载荷升序、ID 升序','不计算完整局部效用，不拟合网络']]),
 '原 16 模式和精确等价组均保留；等价组仅在统计时合并。无缓存、无可交付候选和掩码全开回退按原公开接口处理，不能把全开掩码直接解释为全部模式物理可行。新单目标规则用相同公开字段的容差消歧，原执行器仍决定真实交付；量化后的局部观测无法辨认的边界不注入隐藏真值。','',
 '**完整得分：单位为原共同奖励每槽均值 ×100，越大越好**','',
 '每种基线只执行一份 13 场景 ×20 环境的评估。学习方法按三个固定父模型条件平均，随后展示各模型；广播基线只用于配对算术，不增加重复样本。13 场景等权、20 环境等权，不以三个固定场景代替综合。','',
 table(['方法']+[GROUPS[g] for g in groups],[[method]+[fmt(J(method,g)) for g in groups] for method in methods]),'',
 table(['父模型','控制器']+[GROUPS[g] for g in groups],[[str(p),method]+[fmt(scores[method]['by_parent'][str(p)]['groups'][g]['mean']) for g in groups] for p in m['parents'] for method in m['learned_methods']]),'',
 'C1、C3 是已完成的手工策略重组控制器，并非新训练的端到端 RL。三个父模型均使用原 joint 100 万步和此前指定的修正分支最后 100 万步权重，没有替换模型。','',
 '**配对分差：预定比较全部保留**','',
 '下表为综合分差和按环境种子成簇的 4,000 次配对 bootstrap 95% 百分位区间。一次重采样保留同环境的 13 场景、所有方法、三个固定父模型。区间只针对当前固定模型和反复使用的开发验证环境；未做多重比较校正，跨零不表示统计等价或稳定胜出。','',
 table(['新方法','相对原 greedy_modes_16','相对 R_instruction'],[[method,cistr(paired[method+'_minus_greedy_modes_16']['conditional_mean']['overall_13']),cistr(paired[method+'_minus_R_instruction']['conditional_mean']['overall_13'])] for method in m['methods']]),'',
 table(['基线','original_rl − 基线','C1 − 基线','C3 − 基线'],[[base]+[cistr(paired[method+'_minus_'+base]['conditional_mean']['overall_13']) for method in m['learned_methods']] for base in m['methods']+m['old_baselines']]),'',
 '完整的十个切换场景、各固定任务配对区间、逐父模型比较及原始逐环境分数见 DETAILS.md、paired_differences.json 与 results.json；没有只展示学习方法获胜的子集。','',
 '**是否真正减少开销：统一 CPU 实测**','',
 '硬件 Intel Core Ultra 7 265K，Torch/BLAS 各 1 线程，CPU 上使用同一原始推理与环境接口。GPU 在资源检查时空闲，未用于本轮 CPU 对比。正式评估至多 3 个进程，全部退出后才做单进程延迟测试。没有终止其他项目进程。','',
 '输入固定为原 R_instruction 的 13 场景 ×槽 0/300/599，共 39 个保存状态。单环境与 batch=20 分别每实例预热 30 次、测量 300 次，实例顺序预先封存。计时包含必要输入处理、资源决策、原预算/分配后观测与掩码接口、三架 UAV 选模；不含物理推进、日志、审计、磁盘、模型加载或公共初态重建。分段原始纳秒数据、加载时间、输入哈希及逐父模型延迟均已保存。没有隐藏依赖动作的预计算。','',
 table(['方法','单环境 P50 ms','单环境 P95 ms','batch20 决策环境数/s','持久参数/数组字节','推理权重/预测器文件字节'],[[method,fmt(ts[method]['1']['P50_ms'],4),fmt(ts[method]['1']['P95_ms'],4),fmt(ts[method]['20']['sample_decisions_per_second'],0),str(ts[method]['storage']['unique_persistent_tensor_array_bytes']),str(ts[method]['storage']['dependency_file_bytes'])] for method in methods]),'',
 '学习方法的上表延迟为三个固定父模型各 300 次计时等权合并后取分位数；存储栏给出第一父模型实际加载实例，所有父模型数据在 complexity.json。存储只计唯一持久 tensor/array 和推理依赖文件，排除共享 Python/Torch 运行时、输入环境、公共 profile、临时工作区及 Python 对象开销，不是进程 RSS 或峰值内存。','',
 table(['方法','SUT 候选评分/槽','每 UAV 局部效用候选/槽','单目标属性候选/槽','离线依赖'],[[method,str(ts[method]['storage']['resource_candidate_score_evaluations_per_environment_slot']),str(ts[method]['storage']['UAV_utility_candidate_evaluations_each']),str(ts[method]['storage']['UAV_attribute_candidates_each']),'已有网络/预测器' if ts[method]['storage']['offline_training_or_fit_dependency'] else '无网络拟合；旧规则原映射保留'] for method in methods]),
 'RL 的 16 类输出不是 16 次显式局部效用评分，不能把两类计数混作相同运算成本。G_equal/G_urgency 只含原 UAVGreedy 的固定权重数组，均没有 SUT 预测器加载或调用。原强贪心的预测器文件各 31,499 字节，仍按原版本加载，未重拟合；历史拟合耗时本轮未重新测量，不编造其节省分钟数。所有方法共用原平均表，其历史构建成本不能声称消失。','']
 for method in m['methods']:
  p50=ts[method]['1']['P50_ms'];base=ts['greedy_modes_16']['1']['P50_ms'];through=ts[method]['20']['sample_decisions_per_second']/ts['greedy_modes_16']['20']['sample_decisions_per_second']
  lines += [f"{method}：相对原强贪心，单环境 P50 比值 {p50/base:.3f}（实测减少 {(1-p50/base)*100:.1f}%），batch20 吞吐比值 {through:.3f}；综合分差 {J(method)-J('greedy_modes_16'):+.6f}。",'']
 lines += ['这证明当前硬件和输入分布上的计算变化，不提供硬实时保证或跨设备的速度结论。保留共同观测接口，未实施控制信令压缩，不能宣称通信量减少；各方法实际用到的字段与共同接口大小分别记录在 complexity.json。','',
 '**物理表现解释：质量值是平均表预测，不是视频实测解码 PSNR**','',
 table(['方法','质量收益×100','平均 AoI 成本×100','最大 AoI 成本×100','尾部 AoI 成本×100','资源成本×100','交付数/槽','交付加权预测 PSNR','总未用预算/槽'],[[method]+[fmt(ph['summary'][method]['groups']['overall_13']['reward_parts_x100'][k]) for k in REWARD_PARTS[:5]]+[fmt(ph['summary'][method]['groups']['overall_13']['deliveries_per_slot'],4),fmt(ph['summary'][method]['groups']['overall_13']['predicted_psnr_per_delivery'],4),fmt(sum(ph['summary'][method]['groups']['overall_13']['unused_budget_mean']),2)] for method in methods]),'',
 '最低载荷规则每槽交付较多、AoI 较低，但质量收益不足；最高单块质量规则交付加权预测 PSNR 较高，但每槽交付从轻量均分贪心的约 12.84 降至约 6.63，平均 AoI 从约 3.03 上升至约 6.14。原共同奖励同时衡量这些代价，所以“单块更清楚”并不等于“系统总收益更高”。这些是本次实测轨迹关系，不是新的奖励定义。','',
 '原模式 ID、含未交付类的 16 模式分布、精确物理等价组分布、每 UAV 资源份额/预算/交付、平均/最大/95 分位 AoI、尾部超阈值及回退统计均见 physical_statistics.json。未交付以每架 UAV 的物理槽数为分母单列；PSNR 按交付数量汇总，不平均各场景 PSNR 均值。','',
 '**性能—延迟取舍与六个问题的直接回答**','',
 '1. 四种简化删去的计算见方法定义表：两种轻量贪心保留完整 UAV 局部效用，仅删除 SUT 拟合预测器与资源候选搜索；两种单目标规则进一步取消局部效用，改用固定排序。没有删除原 16 模式或降低决策频率。','',
 '2. 新方法均无需网络离线拟合；实测延迟低于原强贪心。相对已有简单规则，它们并不更快：原 R_instruction 约 0.218 ms，轻量均分贪心约 0.721 ms，排序规则约 0.337 ms。R_equal_minload 与原 R_equal_single 行为完全重复，而后者约 0.216 ms，因此额外排序没有本表下的实际收益。','',
 '3. 均分局部贪心仅损失约 0.0626 分，紧急度局部贪心损失约 0.4503 分；两个单目标规则的性能代价大得多。这里使用绝对分差，不用负分的百分比制造优势。','',
 '4. 原 RL、C1、C3 的条件平均均超过最低载荷、最高质量以及两个旧单模式规则；仍低于均分局部贪心和原两个强贪心。C3 相对 R_equal_instruction 的点估计为正，但区间跨零；相对紧急度贪心和 R_instruction 的条件平均略低且区间跨零。个别父模型的得分高于部分规则，不改变全部模型共同报告的要求。','',
 '5. 原 R_instruction、R_equal_instruction 已很简洁，且速度明显快于当前网络重组控制器；无需为了让 RL 排名上升再削弱它们。最高质量规则也被多种得分更高、延迟更低的原规则同时超过。','',
 '6. 结果只支持学习方法相对若干单目标/单模式低复杂度参照的优势；没有超过原同观测强基线的证据，也不能说 RL 的资源—选模学习问题已解决。值得保留 G_equal_local16 作为无需拟合且性能较强的对照，同时保留原强贪心和原指令规则。下一步如研究学习算法，应面对这些参照；本轮到此停止，不自动启动训练。','',
 '以下是按综合分数与单环境 P50 的点估计同时占优关系，不是有不确定性保证的统计支配：','',
 table(['被比较方法','得分不低且 P50 不高的其他方法'],[[method,'、'.join(r['point_estimate_score_latency_dominators'][method]) or '无'] for method in methods]),'',
 '**核验、实际成本与可复算性**','',
 '预检验证原 UAVGreedy 一致性、预测器零调用、排序/并列/不可行/无缓存边界、逐样本混合指令、资源下限只执行一次、同种子重复及反向运行顺序一致，并对 13 个原参考实例完成完整切换场景的请求动作/资源/物理重放。每个正式新时隙都调用原物理和奖励核验；聚合再次用独立 NumPy 信道/profile 查询、预算、请求/执行模式、缓存时间戳及调度重建交付、AoI、全部奖励分项。没有放宽任何原容差。','',
 table(['成本类别','完整回合','实际物理步','额外说明'],[
  ['四个新基线正式评估','1040','624000','每种 260 回合，不按父模型重复'],
  ['两个旧单模式规则补评','520','312000','所有旧可比较数组逐值一致；补全缓存与时间戳日志'],
  ['复用旧轨迹','3380','本轮 0；历史 2028000','169 份同口径轨迹，全部重新独立核验'],
  ['预检','16','10802','9600 完整回合步 +1200 前缀步 +失败时已推进的 2 步'],
  ['原参考推理重放','0','0','含失败后完整重做，共 312000 个样本决策'],
  ['延迟测试','0','0','38 个实例/批大小组合；1140 预热 +11400 计时批；131670 个样本决策'],
  ['新增训练/拟合/优化器','0','0','始终 inference/eval；运行时禁止 backward/optimizer.step']]),'',
 '本轮实际新增物理计算合计 946,802 步。正式与补评无失败；预检发生一次记录器遗漏当前 instruction_id 字段的错误，真实执行了 2 步后停止，正式启动前补记原 info.gid 后通过全套预检。策略、奖励和容差未改，失败日志、成本记录和重复重放均保留。聚合入口曾有一个括号语法错误，在任何聚合或新增计算开始前修正，原日志保留；不影响模型、轨迹或得分。','',
 '保护输入共 3,622 个文件逐一复核 SHA-256，Git HEAD/状态前后对照。原20开发种子白名单用于所有新环境，检查全部新轨迹 seed 数组、实际运行账本、延迟输入及环境构造记录与 reserved_final_test 无交集。没有使用教师初始化阶段的新准入环境。','',
 '数值聚合与报告重复生成两次；确定性结果、物理统计、配对差、独立审计、逐环境分数和报告数值 SHA-256 应一致，执行证明见 reproducibility.json。耗时、进程、时间戳等运行元数据与数值结果分离。两轮聚合各重建 2,964,000 个旧/新轨迹时隙，并各复算 624,000 个新策略请求决策，这些均是离线审计，不再次推进物理环境。','',
 '新增训练为零不抹去历史成本：original_rl 每父模型已有 100 万步；C1/C3 每父模型调用原模型及两套各追加 100 万步的修正分支，历史累计 300 万步，三个父模型对应 900 万步；C1 与 C3 复用同一批权重不能再重复加账。原平均表、固定规则映射及强贪心预测器历史成本同样保留说明。','',
 '本开发证据不能外推随机训练总体、全局最优、真实视频解码性能或实时部署保障。完整文件索引和复算命令见 EXECUTION.md。实验完成后不继续削弱基线、不训练、不改论文、不提交或推送 GitHub。','']
 textfile(HERE/'report/REPORT.md','\n'.join(lines))
 detail=['# 全部场景、配对比较及物理统计附表','','数值均由同一份原始轨迹重算；区间定义与主报告相同。','']
 detail += ['**所有场景：条件平均及三个固定父模型分别展示**','']
 rows=[]
 for method in methods:
  entries=[('条件平均',scores[method])]
  if method in m['learned_methods']:entries += [(p,scores[method]['by_parent'][p]) for p in scores[method]['by_parent']]
  for p,x in entries:
   for scene,s in x['by_scenario'].items():rows.append([method,p,scene,cistr(s)])
 detail += [table(['方法','模型','场景','分数及 95% CI'],rows),'','**全部预定配对比较：总体、固定任务和切换回合**','']
 rows=[]
 for key,d in paired.items():
  for g in groups:rows.append([key,'条件平均',GROUPS[g],cistr(d['conditional_mean'][g])])
  for p,sub in d.get('by_parent',{}).items():
   for g in groups:rows.append([key,p,GROUPS[g],cistr(sub['groups'][g])])
 detail += [table(['比较','模型','范围','分差及 95% CI'],rows),'','**全部模式和 AoI 的综合分布**','']
 rows=[]
 for method in methods:
  d=ph['summary'][method]['groups']['overall_13']
  for u in range(3):rows.append([method,u+1,fmt(d['aoi_by_uav_mean'][u],4),fmt(d['max_aoi_by_uav_mean'][u],4),fmt(d['p95_aoi_by_uav_mean'][u],4),str(d['requested_mode_counts'][u]),str(d['executed_mode_counts_minus1_then_0_to_15'][u]),str(d['equivalence_counts_no_delivery_then_groups'][u])])
 detail += [table(['方法','UAV','平均 AoI','每槽最大 AoI 均值','每槽 P95 AoI 均值','请求 0–15 计数','执行 −1,0–15 计数','未交付及等价组计数'],rows),'',f"精确等价组：`{m['equivalence_groups']}`。计数保留独有执行量，学习方法综合计数合并三父模型但不作为独立训练样本。各场景×当前指令×父模型×UAV 的完整统计保存在 physical_statistics.json。",'']
 textfile(HERE/'report/DETAILS.md','\n'.join(detail))

if __name__=='__main__':guard();main()
