# UASESC-MARL

指令条件下的多智能体强化学习：SUT分配回传资源，三个UAV基于各自真实预算从16种语义模式中选模，权衡交付质量、AoI和资源消耗。

本次同步更新至2026-09-14，保留此前历史和不利结果，补齐9月12日至14日的代码、配置、报告、数值轨迹及小型策略权重。没有在同步过程中启动训练、修改奖励或修改论文。

## 最新：半质量权重下的长期训练与诊断

三个固定种子104948945、111868397、160441552先在新权重下各训练100万步，再从完整状态分别续训至累计1000万步；长训新增2700万步，三模型累计3000万步，已完成并核验。

新权重仅将质量收益系数乘0.5，AoI和资源系数不变，不重新归一化。综合成绩为原共同奖励每槽均值×100，按13场景、20个开发环境和三个固定训练种子等权；固定任务不冒充13场景综合。

| 方法 | 综合 | 固定均衡 | 固定AoI | 固定质量 |
|---|---:|---:|---:|---:|
| retrained_rl_Qhalf | -8.380924 | -6.970245 | -16.669149 | 0.032502 |
| long_rl_Qhalf | -8.323860 | -7.003098 | -16.621921 | 0.209971 |
| G_equal_local16_Qhalf_local | -8.209130 | -7.022373 | -16.691485 | 0.685478 |
| greedy_modes_16_Qhalf_local | -8.195490 | -7.174723 | -16.527518 | 0.710007 |
| R_instruction | -9.044141 | -9.394658 | -16.577791 | 0.623283 |

长期训练相对100万步起点平均综合增加0.057063，但仍低于同目标均分局部贪心0.114730、同观测强贪心0.128370。三个种子分别变化−0.294647、+0.132866、+0.332972，未筛除退步种子。强贪心的资源预测器仍按旧目标拟合；`_Qhalf_local`表示UAV局部选模使用新权重，不代表重新拟合了资源预测器。

最新只读诊断发现：固定质量下中档模式可行率均在99.72%以上，仍存在大量低载荷偏好；部分动作选择了同载荷但预测质量更低的模式。相同输入下新旧网络的选模也明显变化，支持继续检查选模及策略保持。资源均分的静态容量影响并不一致，不能把静态机会损失写成新控制器的实际胜利。

- [长期训练完整报告](experiments/2026-09-14_quality_half_long_training/report/REPORT.md)、[数值](experiments/2026-09-14_quality_half_long_training/report/results.json)、[独立核验](experiments/2026-09-14_quality_half_long_training/report/audit.json)
- [资源—选模与退步诊断](experiments/2026-09-14_resource_mode_regression_analysis/report/REPORT.md)、[完整统计](experiments/2026-09-14_resource_mode_regression_analysis/report/results.json)、[配对摘要](experiments/2026-09-14_resource_mode_regression_analysis/report/summary.json)
- [诊断复算核验](experiments/2026-09-14_resource_mode_regression_analysis/report/reproducibility.json)

## 实验索引

| 实验 | 实际范围及报告 |
|---|---|
| [9月11日交替训练](experiments/2026-09-11_alternating_training/REPORT.md) | 历史冻结参考，原提交7cc2372保留在Git历史中 |
| [同观测强贪心](experiments/2026-09-11_observation_matched_greedy/REPORT.md) | 资源预测器和局部模式评分，区别于具有全局信息的一步择优 |
| [冻结归因与价值探针](experiments/2026-09-12_credit_assignment_probe/report/REPORT.md) | 已完成，包含报告修正及审计 |
| [质量门控模式修复](experiments/2026-09-12_quality_mode_repair/report/REPORT.md) | 六个固定预算训练任务及评估已完成 |
| [策略重组与资源替换](experiments/2026-09-12_policy_recomposition/report/REPORT.md) | 冻结控制器2×2闭环评估，无新增训练 |
| [教师初始化与条件KL](experiments/2026-09-13_teacher_init_instruction_kl/report/REPORT.md) | 阶段A已完成，准入0/3；按协议停止，critic预热和阶段B未运行 |
| [轻量基线](experiments/2026-09-13_lightweight_baselines/report/REPORT.md) | 四个新规则/局部贪心及实际延迟测量 |
| [质量权重减半](experiments/2026-09-13_quality_weight_half/report/REPORT.md) | 冻结重评分与按新目标重新选模分开报告 |
| [新权重100万步训练](experiments/2026-09-14_quality_half_retraining/report/REPORT.md) | 三模型已完成，为长期续训起点 |
| [新权重1000万步训练](experiments/2026-09-14_quality_half_long_training/report/REPORT.md) | 固定最终检查点比较，未事后挑赢家 |
| [TailRL适用性](experiments/2026-09-14_tailrl_applicability_diagnostic/report/REPORT.md) | 针对1M/6M冻结模型的诊断；没有执行TailRL训练，不冒充10M结论 |
| [最新只读分析](experiments/2026-09-14_resource_mode_regression_analysis/report/REPORT.md) | 资源可行性、模式概率、训练退步及同载荷质量机会损失 |

各实验`PROTOCOL.md`、`manifest.json`、`EXECUTION.md`、`status.json`及`report/`保存协议、执行范围、实际成本和结论边界。旧协议与新权重的成绩不能直接混算。

## 目录与复现边界

保留原本地布局：`HARL/HARL/`为HARL与UAV环境，`CRL-SemCom-VidCI/`为语义编解码与质量表相关代码，`Promptus/`为原组件，`Manuscript/`为原稿及已有修订，`experiments/`为完整实验记录。旧的另一仓库副本`UASESE-MARL/`不再次嵌套上传。

当前在线RL使用固定平均质量—载荷表，不运行Promptus大型视频模型。PSNR是该表给出的交付加权预测质量，不能外推为实测视频解码质量。原20个环境已用于多轮开发；报告的配对区间只描述当前固定模型与开发条件，预留最终测试未使用。

原始实验文件保持原字节及SHA，包括其中记录的旧机器绝对路径。复现时需按相应EXECUTION在新的输出目录处理路径重定位；不要对历史完成目录直接执行恢复脚本。因为完整训练状态和数据集不在本仓库，不能将此快照称为具备全部输入的逐位训练恢复包。模型哈希及源配置仍可审查，已发布的数值报告与仿真评估轨迹保留。

## 同步范围

沿用9月11日已授权范围：代码、配置、报告、日志、CSV/JSON/JSONL数值、NPZ仿真评估轨迹、平均质量表，以及不超过10MiB的小型策略/模型文件。排除大型模型、完整训练状态断点、原始视频/飞行数据、教师/DAgger监督训练数据包、虚拟环境、Git内部目录、本地缓存和凭据。监督数据元信息、生成代码和评估结果保留；没有因负结果而删除实验。

一个超过GitHub单文件限制的历史工作区差异日志以`.diff.gz`无损保存。原始路径、解压后SHA及还原方式见[同步说明](SYNC_INFO.md)。98MB的TailRL历史数值JSON保留原文件，未抽样或截断。

第三方组件许可证保留在对应目录。最新报告优先于历史README中的旧实验结论。
