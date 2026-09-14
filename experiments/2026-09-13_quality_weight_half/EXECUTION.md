# 质量收益减半实验：执行与复算

本目录由实际日期2026-09-13创建，全部旧实验和模型只读。用户选择的单因素干预为三条指令质量收益系数×0.5，AoI及资源系数不变，不归一化。原评价定义、预测器及历史结果不被新目标覆盖。

**实际执行顺序**

从本目录运行以下等价命令；实际shell从项目根目录调用脚本路径，日志与进程命令保存在本目录。

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WEIGHT_PYTHON=/home/king/miniconda3/envs/harl_sionna/bin/python
"$WEIGHT_PYTHON" prepare.py
"$WEIGHT_PYTHON" preflight.py
"$WEIGHT_PYTHON" run_evaluation.py
"$WEIGHT_PYTHON" finalize.py
```

prepare只允许首次创建manifest，不能再次覆盖。预检首次因2环境与历史20环境批量的浮点路径差异未通过严格逐值重现；按PREFLIGHT_AMENDMENT修正该项的批量，第二次预检通过。原奖励、物理容差与正式20环境批量不变。已通过且身份一致的预检复用，不重复运行。首次失败1200步保留在costs和failures中。

**同协议恢复**

```bash
bash run_evaluation.sh
```

要求preflight已通过，至多3个CPU工作者。仅跳过manifest、源码与轨迹哈希完全匹配的完成项，随后完整聚合两次。完成本轮后无需再运行此命令；若再次运行，会重复离线聚合而不新增已完成的正式回合。

**仅从现有轨迹重算**

```bash
"$WEIGHT_PYTHON" aggregate.py
"$WEIGHT_PYTHON" write_report.py
```

使用相同4000次配对bootstrap索引，独立物理重建，不训练或推进真实新回合。完整两次复算证明在report/reproducibility.json。执行时间、命令与错误等元数据独立保存，数值产品不包含时间戳。

| 文件 | 职责 |
| --- | --- |
| PROTOCOL.md / manifest.json | 执行前定义、原/新系数、模型、种子、场景、预算和3953个受保护输入哈希 |
| PREFLIGHT_AMENDMENT.md | 仅预检批量修正、失败原因及额外计算；原协议/manifest保留 |
| configs/ | 原完整配置的只读副本索引语义及明确的新有效系数，不向原配置写回 |
| weight_support.py | 隔离资源/路径、梯度禁止、白名单、私人环境系数替换、原控制器与新局部评分包装 |
| evaluate.py | 完整600槽闭环、每槽原核验、数值轨迹、原物理过程逐值重现 |
| preflight.py / preflight.json | 单因素变化、无归一化、外生配对、冻结、目标匹配及可重现预检 |
| run_evaluation.py / .sh | 固定23个实例、5980完整回合，3个CPU工作者，无训练队列 |
| aggregate.py | 原独立NumPy信道/profile及物理/奖励重建，配对统计及差距分解 |
| write_report.py | 从数值结果生成中文主报告与全部场景、父模型、区间附表 |
| finalize.py | 完整聚合两次、字节哈希一致、原保护哈希与Git状态复核、完成后停止 |
| execution_seal.json / analysis_seal.json | 执行源码及分析源码封存 |
| evaluation/ | 299份正式场景轨迹，每份20×600槽；共5980回合 |
| preflight_traces/ / failures/ / costs/ | 实际预检、失败、额外计算、正式成本与种子账本 |
| logs/ / execution/ | 运行命令、PID、软件资源、耗时和异常 |
| report/results.json / episode_scores.npz | 新目标下得分、同轨迹旧目标得分、逐回合数值与bootstrap索引 |
| report/physical_statistics.json | 每场景×指令×模型×UAV资源、交付、16模式/等价组与AoI |
| report/paired_differences.json / gap_decomposition.json | 全部预定配对比较、区间和质量/AoI/资源分项差距 |
| report/independent_audit.json / audit.json | 299份轨迹逐槽重建、实际成本、最终测试未使用、零训练和保护终检 |
| report/reproducibility.json | 两次完整数值聚合与报告生成的一致性 |
| report/REPORT.md / DETAILS.md | 中文结论及全部结果附表 |
| status.json | 完成状态与无剩余项目 |

模型/参考来源：2026-09-13_lightweight_baselines及其manifest解析的2026-09-11_alternating_training、2026-09-11_observation_matched_greedy、2026-09-12_quality_mode_repair、2026-09-12_policy_recomposition。weight_support从文件位置定位，不依赖旧机器硬编码输出路径。旧预测器SHA-256、三个父模型及adapter来源继续按旧加载器核验。

旧构造器的行和为1检查没有被删除或修改。实验先创建合法原配置的私人环境，再按用户授权仅在该环境内存安装质量减半的系数，避免归一化同时改变AoI/资源。实际生成奖励使用原reward_components，逐槽原检查器和独立离线重建均使用同一有效系数。

正式3,588,000步；通过预检27,600步；保留失败1,200步；实际总计3,616,800步。新梯度训练、监督拟合、预测器拟合、优化器更新、最终测试均为0。两次聚合各核验3,588,000个已有轨迹槽，并重放624,000个同步局部评分决策，没有新物理推进。不重新测延迟，不提出新速度结论。

本任务已经停止，不自动再降低权重、训练、修改论文或推送GitHub。
