"""Chinese diagnosis; every claim is bounded by the two actual evaluation arms."""
from weight_support import *
GROUPS={'overall_13':'13场景综合','balance_fixed':'固定均衡','aoi_fixed':'固定AoI','quality_fixed':'固定质量','switch_10':'10个切换场景'}
def fmt(x):return f'{x:.6f}'
def ci(x):return f"{x['mean']:+.6f} [{x['ci95'][0]:+.6f}, {x['ci95'][1]:+.6f}]"
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def main():
 m=manifest();r=read(HERE/'report/results.json');p=read(HERE/'report/paired_differences.json');ph=read(HERE/'report/physical_statistics.json');a=read(HERE/'report/independent_audit.json');old=read(LIGHT/'report/results.json');oldph=read(LIGHT/'report/physical_statistics.json');s=r['new_objective_scores'];groups=list(GROUPS)
 def j(method,g='overall_13'):return s[method]['conditional_mean'][g]['mean']
 def gap(method,base):return p[method+'_minus_'+base]['conditional_mean']['overall_13']
 lines=['# 质量收益权重减半：实际运行结果与原因分析','',
 '用户选择：三种指令的质量收益权重均降至50%，AoI与资源系数原值不动，不重新归一化。本轮全部冻结RL、adapter和已有预测器，无梯度训练/拟合；旧源代码、配置、模型、质量表、历史报告只读。','',
 table(['指令','原[质量,AoI,资源]','本轮有效系数'],[[name,str(m['original_weights'][i]),str(m['effective_weights'][i])] for i,name in enumerate(['均衡','AoI','质量'])]),'',
 '**先回答为什么原来会这样**','',
 '基线并未忽略指令。原两种轻量贪心每槽读取当前指令、局部缓存/预算/质量表，直接比较16个模式的即时取舍；原强贪心还用冻结预测器选择资源。当前平均表与有限模式下，这种显式计算已经很有效。RL是对决策映射的学习近似，不能因为使用神经网络就推断必然超过它。','',
 '原结果中，RL更偏向多交付、较低AoI的选择，却损失了质量收益。以原RL相对均分局部贪心为例：质量收益少2.933200分，AoI及资源成本合计省1.858958分，净差−1.074242分。这支持“质量—时效取舍影响排名”，不直接证明实现bug或原质量权重不科学。','',
 table(['原目标下方法','质量收益×100','AoI成本合计×100','资源成本×100','交付数/槽','交付加权预测PSNR','平均AoI'],[[method,fmt(oldph['summary'][method]['groups']['overall_13']['reward_parts_x100']['quality_credit']),fmt(sum(oldph['summary'][method]['groups']['overall_13']['reward_parts_x100'][k] for k in ['age_mean_cost','age_max_cost','age_tail_cost'])),fmt(oldph['summary'][method]['groups']['overall_13']['reward_parts_x100']['resource_cost']),fmt(oldph['summary'][method]['groups']['overall_13']['deliveries_per_slot']),fmt(oldph['summary'][method]['groups']['overall_13']['predicted_psnr_per_delivery']),fmt(oldph['summary'][method]['groups']['overall_13']['field_means']['mean_aoi'])] for method in ['original_rl','C1','C3','G_equal_local16','greedy_modes_16']]),'',
 '**本次运行的两类对照必须分开理解**','',
 'A：全部原13种方法（含三个父模型，共19实例）决策完全冻结，在新奖励下实际重跑4,940个完整回合。逐值验证请求/执行模式、资源、交付、缓存、时间戳和AoI与历史轨迹相同。其分数变化仅是评分变化：J_half = J_old − 0.5×旧质量收益。不是RL发生了任何学习。','',
 'B：四个明确带 _Qhalf_local 后缀的方法，把原UAVGreedy的局部评分系数同步到新奖励，实际重跑1,040回合。资源分配、候选集合、质量可行门槛、回退和局部其他公式不变。新的均分/紧急度贪心完全无需拟合。两个旧强贪心的SUT预测器仍按旧奖励拟合且保持冻结，只同步了UAV局部评分，因此不能称“针对新目标重新拟合的完整强基线”。','',
 '正式合计5,980回合、3,588,000物理步，均为原20个development validation种子、13场景、每回合600槽。学习方法保留父模型104948945、111868397、160441552及既定检查点；没有引入教师初始化学生或其gate_dev环境。','',
 '**主要结果：原目标与质量减半目标**','',
 table(['方法','原目标下历史分数','冻结原决策：新目标分数','局部评分同步新目标：分数'],[[method,fmt(old['scores'][method]['conditional_mean']['overall_13']['mean']),fmt(j(method)),fmt(j(method+'_Qhalf_local')) if method in m['greedy_methods'] else '不适用：决策冻结'] for method in m['static_methods']+m['learned_methods']]),'',
 '不同奖励目标的绝对分数不可直接当作性能退步或进步。降低正质量收益系数后，固定轨迹分数自然降低；应在同一个新目标内比较方法。学习控制器上表为三个固定父模型的条件平均。','',
 table(['方法']+[GROUPS[g] for g in groups],[[method]+[fmt(j(method,g)) for g in groups] for method in r['methods']]),'',
 table(['父模型','学习控制器']+[GROUPS[g] for g in groups],[[parent,method]+[fmt(s[method]['by_parent'][str(parent)]['groups'][g]['mean']) for g in groups] for parent in m['parents'] for method in m['learned_methods']]),'',
 '**配对结论：不能只展示旧贪心继续按旧目标行动的结果**','',
 table(['基线','原RL − 基线','C1 − 基线','C3 − 基线'],[[b]+[ci(gap(method,b)) for method in m['learned_methods']] for b in m['static_methods']+m['adaptive_methods']]),'',
 '区间是以20个环境种子为簇、4,000次配对bootstrap的95%百分位区间；同次重采样保留全部13场景、所有方法和三个固定父模型。未做多重比较校正。区间跨零不表示统计等价，也不支持稳定胜出；区间仅条件于本次固定模型及开发环境。','']
 for method in m['learned_methods']:
  old_f=gap(method,'G_equal_local16');matched=gap(method,'G_equal_local16_Qhalf_local');strong=gap(method,'greedy_modes_16_Qhalf_local')
  lines += [f"{method}：相对冻结旧均分贪心 {ci(old_f)}；相对已同步新权重的均分贪心 {ci(matched)}；相对同步局部评分、但保留旧SUT预测器的强贪心16 {ci(strong)}。",'']
 lines += ['**局部目标变化是否真的改变了动作**','',
 table(['方法','相对冻结原版的新目标分差','请求模式变化/全部UAV槽','执行模式变化/全部UAV槽'],[[method,ci(p[method+'_minus_'+method.removesuffix('_Qhalf_local')]['conditional_mean']['overall_13']),str(sum(v['requested_uav_slots_changed'] for k,v in r['changed_modes'].items() if '/'+method+'/' in k))+'/'+str(13*20*600*3),str(sum(v['executed_uav_slots_changed'] for k,v in r['changed_modes'].items() if '/'+method+'/' in k))+'/'+str(13*20*600*3)] for method in m['adaptive_methods']]),'',
 '这些计数比较完整闭环轨迹，包含动作改变后的后续状态差异；不能解释为每个旧状态上的独立反事实动作差。每个同步权重版本的新请求动作另用保存的本地观测/掩码完全重放核验。','',
 '**新目标下的物理与奖励分项**','',
 table(['方法','质量收益×100','平均AoI成本×100','最大AoI成本×100','尾部AoI成本×100','资源成本×100','交付数/槽','预测PSNR','平均AoI'],[[method]+[fmt(ph['summary'][method]['groups']['overall_13']['reward_parts_x100'][k]) for k in ['quality_credit','age_mean_cost','age_max_cost','age_tail_cost','resource_cost']]+[fmt(ph['summary'][method]['groups']['overall_13']['deliveries_per_slot']),fmt(ph['summary'][method]['groups']['overall_13']['predicted_psnr_per_delivery']),fmt(ph['summary'][method]['groups']['overall_13']['field_means']['mean_aoi'])] for method in r['methods']]),'',
 '预测PSNR来自固定平均质量表，按实际交付数量加权，不是真实视频解码测量。所有原模式16类、精确等价组、无交付比例、每UAV资源/预算/余量、缓存、AoI和奖励项已保留。差距按全部奖励分项重建，见gap_decomposition.json。十个切换场景和逐模型完整区间见DETAILS.md与paired_differences.json。','',
 '**如何判断这次测试**','']
 matched='G_equal_local16_Qhalf_local';wins=[method for method in m['learned_methods'] if gap(method,matched)['ci95'][0]>0];loss=[method for method in m['learned_methods'] if gap(method,matched)['ci95'][1]<0];uncertain=[method for method in m['learned_methods'] if method not in wins+loss]
 lines += [f"相对无需拟合且已同步新权重的均分局部贪心：配对区间全部高于0的学习控制器为 {'、'.join(wins) or '无'}；全部低于0的为 {'、'.join(loss) or '无'}；跨零的为 {'、'.join(uncertain) or '无'}。",'',
 '这轮检验的是目标偏好和冻结控制器选择之间的关系。仅把质量项减半，确实会相对有利于质量收益较低、AoI/资源成本较低的旧策略；这属于指标重新定价。贪心也必须有机会按同一偏好重新选模，才能评估其取舍能力。','',
 '无论排名如何，不能据此把原权重称为代码错误，或把没有更新参数的RL称为训练能力提升。若希望最终采用新目标，下一步应先根据论文任务要求论证质量与时效偏好，再单独设计新目标下的训练对照；本轮不自动训练、不继续调权重直到某方法胜出。','',
 '**实现核验与成本**','',
 '原环境构造器要求权重每行和为1。本次不修改这一旧文件，也不把新行重新归一化，而是在独立脚本创建的私人环境内存中安装有效系数；原reward_components及每槽独立核验器直接读取它。验证观测、掩码、质量门槛及物理参数均不因该系数替换而改变。','',
 '首次预检用2环境批次对照历史20环境批次，在5个usage元素有最大3.637978807091713e−12差异，原逐槽和独立物理核验均已通过，但严格逐值历史重现未通过。按PREFLIGHT_AMENDMENT统一该旧策略对照为20环境，未放宽任何容差；随后通过。保留失败的2完整回合/1200步。实际预检48物理回合/28,800步，其中46回合/27,600步为通过项。正式5,980回合无失败、丢弃或换种子。','',
 table(['项目','物理回合','实际新物理步'],[['冻结控制器正式','4940','2964000'],['同步局部目标正式','1040','624000'],['预检通过','46','27600'],['预检失败并保留','2','1200'],['总计','6028','3616800'],['两轮离线聚合/重放','0','0'],['RL训练/监督拟合/预测器拟合','0','0']]),'',
 '每槽原核验后，聚合重新用独立NumPy信道/profile、资源动作、缓存时间戳与调度重建物理量和所有奖励分项。两次完整聚合各核验299份轨迹、3,588,000时隙，并重放624,000个同步局部评分决策。聚合不推进环境，不重复计入新物理成本。全部新轨迹seed数组、环境元数据和实际成本记录都检查开发白名单与reserved_final_test无交集。','',
 '全部3,953个受保护文件结束时重算SHA-256，Git前后状态核对；模型、adapter、预测器原文件及私人策略参数哈希不变。数值聚合与中文报告各重复生成两次，字节哈希一致证据见reproducibility.json。时间戳、耗时、命令与异常独立记录。','',
 '没有下载数据、改变原质量表、执行旧恢复脚本、训练新模型、使用最终测试、修改论文、commit或push。保留旧13方法和原强基线结果，不以新目标替换旧目标的历史成绩。全部固定范围计算结束后停止。','']
 textfile(HERE/'report/REPORT.md','\n'.join(lines))
 detail=['# 完整场景和配对附表','','所有分数均为本轮新目标下每槽共同奖励均值×100。','']
 rows=[]
 for method in r['methods']:
  for scene,d in s[method]['by_scenario'].items():rows.append([method,'条件平均',scene,ci(d)])
  for parent,d in s[method].get('by_parent',{}).items():
   for scene,x in d['by_scenario'].items():rows.append([method,parent,scene,ci(x)])
 detail += [table(['方法','模型','场景','分数/CI'],rows),'','所有预定配对比较；负分差和跨零区间均保留。','']
 rows=[]
 for key,d in p.items():
  for group,x in d['conditional_mean'].items():rows.append([key,'条件平均',group,ci(x)])
  for parent,e in d.get('by_parent',{}).items():
   for group,x in e['groups'].items():rows.append([key,parent,group,ci(x)])
 detail += [table(['比较','父模型','范围','分差/CI'],rows),'']
 textfile(HERE/'report/DETAILS.md','\n'.join(detail))
if __name__=='__main__':guard();main()
