"""Deterministic report aggregation; contains no environment stepping or optimization."""
from diag_support import *
from controlled_stats import ci

def fmt(x,n=4):return '不可估计' if x is None else f'{x:.{n}f}'
def pc(x):return f'{100*x:.2f}%'
def estimate(x,scale=1):return f"{x['mean']*scale:.4f} [{x['ci95'][0]*scale:.4f}, {x['ci95'][1]*scale:.4f}]"
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def run():
 guard();m=manifest();c=read(HERE/'report/controlled_results.json');a=read(HERE/'report/additional_results.json');g=read(HERE/'report/gradient_results.json');h=read(HERE/'report/history_distributions.json');hi=read(HERE/'report/history_inventory.json');tr=read(HERE/'report/checkpoint_tail_trend.json');logs=read(HERE/'report/training_evidence.json')
 boot=np.random.default_rng(m['bootstrap_seed']).integers(0,20,(4000,20));grand={};individual={}
 for steps in [1000000,6000000]:
  ar=[arrays(HERE/f'analysis_arrays/{seed}_{steps}_fixed_2.npz') for seed in m['parents']]
  grand[str(steps)]={k:ci(np.stack([z[k] for z in ar]),boot) for k in ['deltaQ','Q','success_feasible','success_safe','full_total']}
 for k,v in c.items():
  individual[k]={z:v[z] for z in ['quality','delta_quality','delta_psnr','success','success_vs_fixed_1m','full_reward_difference','environment_vs_policy','reward_correlations','top10_Q_given_top10_R_within_environment']};individual[k]['cost_on_feasible_success']=a[k]['safety_failure'];individual[k]['success_detail']=a[k]['success_detail']
 # Every available historical method/scenario remains visible in the detailed source output.
 hist={}
 for k,v in h.items():
  if not k.startswith(('recomposition/','half_objective/','half_short/')):continue
  family,seed,method,scene=k.split('/')[:4];dest=hist.setdefault(f'{family}/{seed}/{method}',{})
  dest[scene]=dict(reward=v['overall']['reward']['mean'],native_reward=v['overall']['native_reward']['mean'],quality=v['overall']['quality']['mean'],quality_stage=v.get('2'))
 hist_overview={}
 for k,v in hist.items():
  hist_overview[k]=dict(scenes=list(v),full13_mean_reward_times100=float(np.mean([x['reward'] for x in v.values()])*100) if len(v)==13 else None,fixed_quality_Q=v.get('fixed_2',{}).get('quality'),fixed_quality_reward_times100=v['fixed_2']['reward']*100 if 'fixed_2' in v else None)
 summary=dict(verdict='Not supported as a likely no-regression remedy; partial quality-only tails exist in one of three current models, with AoI/QoE regression',scope='Current half-quality reward, immutable 1M and 6M snapshots; original 13-scene history, controlled sampling in fixed Quality and two Quality-first switches',conditioned_means=grand,per_model_scenario=individual,historical_overview=hist_overview,gradient_joint={s:v['joint'] for s,v in g.items()},checkpoint_trend=tr,new_evaluation_episodes=3060,new_evaluation_physical_steps=1836000,new_training_steps=0,new_optimizer_updates=0)
 write(HERE/'report/results.json',summary)
 lines=['# TailRL 适用性诊断：实际执行报告','',
 '结论：当前证据不支持“Quality 失败主要因为少量兼顾其他目标的成功轨迹被普通 RL 忽略”，也不支持 TailRL 有较高概率在基本不损害 AoI、资源和总体 QoE 的情况下解决问题。存在局部质量尾部，但它带有明显的 AoI 代价。此次不启动 TailRL 训练。',
 '',
 '## 1. 本轮实际做了什么',
 f"只读分析 {hi['records']} 条历史轨迹记录索引（{hi['unique_payloads']} 个不同文件载荷，别名/重复不视为独立样本）、18 份训练日志快照和 {len(logs['archived_training_traces'])} 份已保存训练轨迹。已核对 policy_recomposition 的 2340 个正式回合、1280 个预检/复现回合；后者只用于复现证据。历史记录覆盖原 13 场景，详细分布不删失败模型。",
 '新增冻结推理 3060 个完整回合、1,836,000 物理步：100 万/600 万步固定质量任务各 3 模型×20 环境×16 动作重复；600 万步两个质量起始的切换场景各 8 重复；补充 180 个确定性参照回合。新增训练步、优化器更新均为 0。模型、critic、ValueNorm 未更新；反事实计算只调用 autograd.grad。',
 '这里的种子编号标识各实验中的对应网络，不能把同编号旧 adapter 控制器与新权重重新训练的网络当成同一策略。当前研究对象是质量权重减半的新联合 HAPPO，固定共同已存在的 600 万步快照；原有长期训练在另一个任务流程中继续，本轮不根据其随后进展挑模型。',
 '新增随机重复只覆盖固定 Quality、Quality→均衡、Quality→AoI 三场景；其平均不能称为 13 场景综合结果。其余场景利用已有确定性轨迹，不能从那些数据估计同环境随机策略尾概率。',
 '工作区父目录及 HARL 空 .git 目录无法提供有效 Git HEAD，已记录 git_inspection.json；使用父实验 manifest 与逐文件 SHA-256 定位冻结版本，未用其他仓库模型替代。所有新文件只在本实验目录。用户附件末尾止于“如果可能”，本报告执行其可见要求。',
 '',
 '## 2. 指令确实进入策略，并非同一输入下隐藏了目标',
 '指令顺序为 balance=0、aoi=1、quality=2。SUT 局部观测下标 23:26、UAV 局部观测 67:70 明确包含当前指令 one-hot；critic 状态也包含对应字段。实际固定相同物理状态进行切换，SUT 只变 23/25，UAV 只变 67/69，均经预检。actor_observe_instruction=true。',
 '因此，同一物理状态加 AoI 与 Quality 指令会产生不同输入，当前不需要先补指令字段。共享网络参数仍可能发生多目标梯度干扰；“输入不同”不能证明梯度不冲突。本轮没有把不同指令梯度单独做全面冲突归因，不能称其为唯一根因。证据：preflight.json 与冻结 tensor_env.py 的 observe。',
 '',
 '## 3. Quality 成功与不退步是两件事',
 '当前质量来自固定平均 profile 的预测 PSNR，非 LPIPS、非本轮视频解码实测。主指标 Q 是原奖励的未加权质量收益：每槽交付块累加 max((预测 PSNR−21)/12,0)，除以 30 条源流，再在当前指令时段平均。它同时计入质量与交付数量。附报 PSNR 按交付数加权；平均 PSNR 提高而交付变少，Q 仍可下降。',
 '系统不存在可以直接相减的同一帧“前/后解码质量状态”。这里 ΔQ 使用相同模型、场景、初态与外生序列下的采样轨迹减确定性轨迹，避免将不同时槽或不同视频的 PSNR 差冒充改善。跨检查点另固定 100 万步确定性参照。',
 '预先固定弱/中/强门槛 ΔQ>0.001/0.005/0.010。“可行改善”还需 Quality 阶段平均 AoI≤6，且质量、预算、缓存执行违规为 0。“不明显退步改善”还要求平均/最大/尾部 AoI 成本与资源使用均不超过配对参照的 105%；另报零退步与全槽最差 AoI≤6。原 max-AoI 服务上限是软项且罚系数为 0，不能把零罚分当成没有超限。',
 '当前没有独立可学习的能耗、计算资源、路由或父节点选择模型，不能推断这些未建模指标不受损。',
 '',
 '## 4. 固定质量任务：受控尾部分布',
 '以下 ΔQ 单位为规范化质量收益，未乘 100。每行 320 回合；先固定环境再重复动作，置信区间以 20 个环境为簇。',
 table(['模型','累计步数','均值','中位数','P95','P99','最大值','中门槛可行改善','不明显退步改善'],[[seed,f'{steps//1000000}M',*[fmt(c[f'{seed}/{steps}/fixed_2']['delta_quality'][x],6) for x in ['mean','median','p95','p99','maximum']],pc(c[f'{seed}/{steps}/fixed_2']['success']['0.005']['feasible']['mean']),pc(c[f'{seed}/{steps}/fixed_2']['success']['0.005']['safe']['mean'])] for seed in m['parents'] for steps in [1000000,6000000]]),
 '均值、标准差、P50/P75/P90/P95/P99/最大值以及每个环境的条件分布全部在 controlled_results.json；全部历史场景/检查点分布在 history_distributions.json。P99 是样本插值，单环境 16 重复无法准确识别 1% 稀有事件，Top1% 与 Top5% 都仅对应该环境的最好 1 次。',
 table(['6M 模型','弱门槛可行改善','中门槛可行改善及环境簇区间','强门槛可行改善','零退步/严格服务成功'],[[seed,pc(c[f'{seed}/6000000/fixed_2']['success']['0.001']['feasible']['mean']),estimate(c[f'{seed}/6000000/fixed_2']['success']['0.005']['feasible'],100),pc(c[f'{seed}/6000000/fixed_2']['success']['0.01']['feasible']['mean']),pc(c[f'{seed}/6000000/fixed_2']['success']['0.005']['zero_regression']['mean'])+' / '+pc(c[f'{seed}/6000000/fixed_2']['success']['0.005']['strict_service']['mean'])] for seed in m['parents']]),
 '所有受控组在预定门槛和 5% 退步限制下均未观察到成功。这不证明真实概率等于零；全零样本的 bootstrap 区间退化为 [0,0]，不能解释为排除了更稀有的无损轨迹。前两个模型的随机尾部未超过自身确定性策略，也不等于系统完全没有可行质量策略。',
 '',
 '## 5. 高质量是运气还是行为？',
 '同组初始缓存、时间戳、AoI 与外生哈希一致，各动作重复的 SNR 数组逐值相同。缓存负载及空缓存补入由先前动作决定，属于内生状态，未误当作低到达率的外生好运。',
 table(['6M 模型','跨环境均值与 SNR 相关','环境间均值方差','环境内动作方差','前半可行成功率','后半可行成功率'],[[seed,fmt(c[f'{seed}/6000000/fixed_2']['environment_vs_policy']['Q_envmean_vs_SNR_correlation']),fmt(c[f'{seed}/6000000/fixed_2']['environment_vs_policy']['between_environment_mean_Q_variance'],8),fmt(c[f'{seed}/6000000/fixed_2']['environment_vs_policy']['within_environment_Q_variance'],8),pc(a[f'{seed}/6000000/fixed_2']['split_repetition'][0]['feasible_probability']['mean']),pc(a[f'{seed}/6000000/fixed_2']['split_repetition'][1]['feasible_probability']['mean'])] for seed in m['parents']]),
 '原始 Q 的跨环境差异很大，不能将不同信道的确定性结果混合成 TailRL 样本组。控制外生随机性后，160441552 仍有 32/320 次中门槛质量改善，分布在 5 个环境，前后半各 10%；其中一个环境 16/16 均改善，另一些环境几乎不改善。因此它是部分状态下可重复的行为差异，不是统一的低概率成功机制，也不能仅归为好运。',
 '',
 '## 6. 好质量尾部的动作规律及代价',
 'Top1/5/10%、中位、底部50%的原16模式、精确等价组、各UAV资源份额、预算、使用/未用量、交付、PSNR和AoI均已保存。以下是每个环境内 Top10% 相对中位轨迹的平均差：',
 table(['6M 模型','Q 差','预测 PSNR 差(dB)','交付数差/槽','平均 AoI 差','资源使用差/槽'],[[seed,*[fmt(a[f'{seed}/6000000/fixed_2']['top10_minus_median'][x]['mean'],6 if x=='quality' else 4) for x in ['quality','psnr','deliveries','mean_aoi','resource']]] for seed in m['parents']]),
 '较高 Q 的样本主要伴随模式配比的小幅变化和较高单块质量；跨模型并非同一个确定模式或统一资源分配向量。精确等价组差及两半重复的方向记录在 additional_results.json。它们是选择结果后的行为关联，不能直接证明某一个动作模式就是因果解法。',
 '160441552 的 32 条中门槛可行质量改善轨迹，相对同环境确定性参照：',
 table(['指标','配对平均变化'],[['Q',fmt(a['160441552/6000000/fixed_2']['success_detail']['quality_delta_mean'],6)],['预测 PSNR (dB)',fmt(a['160441552/6000000/fixed_2']['success_detail']['psnr_delta'])],['平均 AoI',fmt(a['160441552/6000000/fixed_2']['safety_failure']['mean_aoi']['mean_delta_on_feasible'])+' ('+pc(a['160441552/6000000/fixed_2']['safety_failure']['mean_aoi']['mean_relative_change_on_feasible'])+')'],['每槽最大 AoI 的均值',fmt(a['160441552/6000000/fixed_2']['safety_failure']['max_aoi']['mean_delta_on_feasible'])+' ('+pc(a['160441552/6000000/fixed_2']['safety_failure']['max_aoi']['mean_relative_change_on_feasible'])+')'],['资源使用/槽',fmt(a['160441552/6000000/fixed_2']['safety_failure']['resource']['mean_delta_on_feasible'])],['共同奖励×100',fmt(a['160441552/6000000/fixed_2']['success_detail']['full_reward_difference']*100)]]),
 '这 32 条全部因平均/最大/尾部 AoI 退步超过 5% 而未通过安全性诊断；资源本身没有超过 5% 门槛。不能将它们写成兼顾所有目标的“好轨迹”。',
 '',
 '## 7. 普通 RL 是否正在压掉好轨迹？',
 '100 万→600 万步，在相同环境与动作种子下，三个模型采样 Q 与采样共同奖励均上升：',
 table(['模型','采样 Q 变化 [95%区间]','采样奖励×100变化 [95%区间]'],[[s,estimate(tr[str(s)]['sampled_quality_change']),estimate(tr[str(s)]['sampled_reward_change'],100)] for s in m['parents']]),
 '因此这个固定区间没有出现“质量成功概率下降而奖励上升”的预期 Pattern 1。零无损成功也不能据此解释为曾经被压掉。早期 20/40/60/80/100 万步确定性 Quality 的起伏另在结果中保留；确定性性能下降不是稀有轨迹概率下降的证明。保存的原训练轨迹跨更新边界，不能装成同一冻结策略的 N 个样本。',
 '日志显示 Quality 样本约占三分之一，并非没有训练过 Quality。实际历史逐样本梯度未保存；本轮 GAE 是冻结 critic/ValueNorm 在新轨迹上计算的代理，不能追认某条历史成功轨迹被 PPO 裁剪或丢弃。',
 '',
 '## 8. 总回报尾部不等于质量尾部',
 '当前实际权重 [质量,AoI,资源] 分别为均衡 [.25,.4,.1]、AoI [.10,.7,.1]、Quality [.35,.2,.1]，仅原质量系数减半。真实奖励分项原样保留。',
 table(['6M 模型','环境内 corr(Q,R)','Top10%(R) 中也属 Top10%(Q)','采样相对确定性奖励×100'],[[s,fmt(c[f'{s}/6000000/fixed_2']['environment_vs_policy']['within_environment_Q_vs_total_correlation']),pc(c[f'{s}/6000000/fixed_2']['top10_Q_given_top10_R_within_environment']),estimate(c[f'{s}/6000000/fixed_2']['full_reward_difference'],100)] for s in m['parents']]),
 '两者正相关但排序差异明显。这里每条件16次，Top10%按向上取整取2次（实际12.5%），不能假装精确连续分位。完整原始相关及 AoI/资源分项相关均已保留。若目标是专项质量修复，不应把 TailRL(total) 当成 TailRL(Q) 的替代；使用 Q 也不会自动处理其 AoI 代价。',
 '',
 '## 9. 实际模拟 TailRL 权重',
 '采用 [TailRL 原论文 §4、附录 C](https://arxiv.org/html/2609.02987v1) 的组内排序间隔权重，原实现来源为 [官方仓库](https://github.com/Zanette-Labs/TailRL)。N条同条件轨迹排序后，w_i=N∑_{j≤i}(r_j−r_{j−1})/(N−j+1)，再组内中心化。相同回报权重相同，平移共同常数不改变中心化结果；中心化有限样本估计对应论文的有限阶目标，不是精确无穷阶极尾。',
 '每个条件固定 N=16，切换场景 N=8。比较原尺度与同批归一化尺度；实际动作时隙权重和“先平均整条轨迹优势再取正”的代理分开报告，二者不可混用。以下针对160441552固定质量的32条可行改善轨迹，使用逐时隙正权重口径：',
 table(['估计器','E[A|成功]','成功/失败的平均正权重比','成功样本正权重份额'],[[name,fmt(a['160441552/6000000/fixed_2']['per_slot_positive_weight'][key]['mean_success']),fmt(a['160441552/6000000/fixed_2']['per_slot_positive_weight'][key]['positive_weight_ratio']),pc(a['160441552/6000000/fixed_2']['per_slot_positive_weight'][key]['success_fraction_of_positive_weight'])] for name,key in [('冻结 GAE 代理','frozen_GAE'),('TailRL(Q)','tail_quality')]]),
 '权重确实增强，但增强对象仍是 AoI 退步的质量改善轨迹。对零无损成功组，E[A|成功]和相应比值不可估计，用 NULL 保留，不能填0后声称算法无效或除零得到无穷增强。普通轨迹均值代理、条件中心化 GAE 和 TailRL(total) 的全部数字也已保存。',
 '',
 '## 10. 反事实梯度：更偏质量，同时有明确冲突风险',
 '固定600万步，每模型前8个预定开发环境×16重复×600槽，共76,800样本；4个actor分别用真实局部输入、原请求动作及原掩码。计算11种权重的上升梯度，未创建优化器、未调用 backward、未执行 step。采样/evaluate_actions logp完全复现，初始比率为1，模型/归一化哈希不变。',
 table(['模型','cos(基准,TailQ)','cos(TailQ,平均Q)','cos(TailQ,降低AoI成本)','cos(TailQ,降低资源成本)','cos(TailQ,可行成功)'],[[s,*[fmt(g[str(s)]['joint']['cosine']['tail_quality'][x]) for x in ['baseline_GAE','expected_quality','negative_AoI_cost','negative_resource_cost','feasible_success']]] for s in m['parents']]),
 '负的“降低成本”余弦意味着与成本下降方向相反，是局部冲突证据，不能直接换算为训练后的 AoI 增幅。TailQ 与平均Q梯度高度一致，也提示问题未必需要特殊尾部算法：它首先改变了质量与时效的优化取舍。成功指示在前两模型为常量、无损成功在全部模型为常量，对应零梯度，余弦不可估计。',
 '去掉了各方案共同熵项，只比较初始PPO/HAPPO策略梯度；未模拟Adam、多epoch裁剪、顺序更新后的factor，也没有执行有限步参数扰动。GAE按400槽和末200槽窗口重建，完整600槽终止不bootstrap；公共诊断批次的规范化不等同于原每个4000样本更新批次。三模型的平均Q、总回报尾部、AoI/资源/服务超限梯度和各actor完整余弦矩阵均在 gradient_results.json。',
 '',
 '## 11. 切换场景和历史控制器证据',
 table(['模型','场景','可行质量改善率','不明显退步改善率','全回合奖励×100差'],[[s,scene,pc(c[f'{s}/6000000/{scene}']['success']['0.005']['feasible']['mean']),pc(c[f'{s}/6000000/{scene}']['success']['0.005']['safe']['mean']),estimate(c[f'{s}/6000000/{scene}']['full_reward_difference'],100)] for s in m['parents'] for scene in m['sampling_scenes'] if scene!='fixed_2']),
 '两个切换场景的前300槽，以及固定质量的相同前缀，在相同动作重复下逐值相同，不能算成独立质量成功重复。切换后仍按当时策略采样，因此非质量阶段分差包含继承缓存/AoI状态及当期随机动作差异；不能全部叫作质量阶段遗留代价。全部阶段分项及后续差保存在 controlled_results.json。',
 'C0/C1/C2/C3已完成的13场景结果按当前半质量权重另行统一重算，原native_reward同时保留。这是冻结动作重评分，不代表它们按新权重重新训练；资源均分带来的收益也不等于当前随机策略已经会产生相同行为。',
 table(['旧父模型','C0综合','C1综合','C2综合','C3综合','C0质量Q','C3质量Q'],[[s,*[fmt(hist_overview[f'recomposition/{s}/{z}']['full13_mean_reward_times100']) for z in ['C0','C1','C2','C3']],fmt(hist_overview[f'recomposition/{s}/C0']['fixed_quality_Q'],6),fmt(hist_overview[f'recomposition/{s}/C3']['fixed_quality_Q'],6)] for s in m['parents']]),
 '',
 '## 12. 建议及不能得出的结论',
 '当前不建议直接把长期 HAPPO 替换为 TailRL，也不建议马上进行大预算 TailRL 重训。先保留现有奖励和检查点，围绕“提高已交付质量时如何维持交付频率与AoI”研究资源—模式配合，并区分随机资源执行与随机选模各自的损失。上述为后续建议，本轮不自动追加单侧采样、改熵、改网络或训练。',
 '如果未来仍要检验TailRL，应采用按Quality指令分组、外生条件匹配，并对AoI/QoE退步预设停止条件的独立小规模对照；同时保留普通质量目标加权的对照，避免把提高质量权重本身的作用误称尾部学习优势。本轮没有实施此训练，也不声称这些修改会成功。',
 '不能得出：TailRL一定无效；神经网络无法学会任务；初始化/梯度冲突是唯一原因；存在较高概率的无损改善；TailRL已超过教师/强贪心；验证集等同独立最终测试；平均表质量等同真实视频解码性能。尤其，物理系统不能实际运行多个未来再择真实reward最佳者，Best-of-N尾部诊断不能当成可部署控制得分。',
 '',
 '## 13. 审计与复算',
 '逐槽原核验与离线物理/奖励重建均通过；未放宽原容差。最终受保护哈希、实际seed与reserved交集、成本账本、重复聚合结果见 audit.json 与 reproducibility.json。新样本仅原20个开发验证环境；保存动作随机种子和外生哈希。旧长期任务的自然日志增长不计为本轮训练，且未用其6M后的模型改变诊断。',
 '主要文件：PROTOCOL.md、manifest.json、history_index.json、preflight.json、evaluation/、analysis_arrays/、gradients/、report/results.json、report/controlled_results.json、report/additional_results.json、report/gradient_results.json、report/history_distributions.json、report/training_evidence.json、EXECUTION.md。'
 ]
 lines.insert(lines.index('## 5. 高质量是运气还是行为？'),f'![相同外生条件下的质量改善分布]({HERE}/report/quality_tail_cdf.png)')
 lines.insert(lines.index('## 11. 切换场景和历史控制器证据'),f'![反事实质量梯度与各项目标的方向]({HERE}/report/gradient_directions.png)')
 textfile(HERE/'report/REPORT.md','\n\n'.join(x for x in lines if x)+'\n')
 print('AGGREGATION COMPLETE',flush=True)
if __name__=='__main__':run()
