# 执行与文件索引

本轮已实际完成：6个新增100万步任务、30个不可变检查点、40个评估任务、280份数值轨迹（5600个600槽回合）。正式训练墙钟1267.11秒，评估367.06秒；实现、预检和最终审计另计。两CPU工作进程、Torch/BLAS各1线程，正式阶段观测CPU最高68°C、GPU最高39°C，无温控暂停，无正式任务失败。软件清单在 software_environment.json，实测与温控记录在 preflight.json、resource_summary.json、logs/resources.jsonl。

## 阅读结果

- report/REPORT.md：逐父模型完整比较、配对区间和八个问题的明确回答。
- report/MODE_AND_REWARD_DETAILS.md：固定质量任务的PSNR预测、交付、AoI、奖励分项及模式频率。
- report/results.json：可机读的最终分数、配对差、中间固定场景曲线、逐环境回合分数和模型来源。
- report/physical_statistics.json：每场景、指令及每UAV的完整模式/等价组/资源/奖励统计。
- report/audit.json、report/reproducibility.json：独立核验、原文件保护、未使用最终测试、重复聚合一致性。
- checkpoint_statistics/：30个检查点的训练日志前缀累计统计，包含实际更新、有效质量样本、梯度/熵/概率比、critic诊断与模式分布；没有改动原模型快照。

## 实现与原始证据

- PROTOCOL.md、manifest.json、seed_selection.json、configs/：固定协议、原输入与执行代码hash、种子和配置。
- adapter_policy.py：冻结父UAV加独立修正分支，原观测字段67:70解析质量门控。
- repair_training.py：独立HAPPO训练入口；SUT确定性执行，UAV采样，critic副本更新。
- repair_evaluation.py：所有方法共用冻结环境和原独立物理/奖励核验。
- preflight.py、preflight.json、preflight_attempts/：预检、吞吐测试和保留的失败尝试。
- jobs/seed_*/residual_*/：新适配器、critic、ValueNorm、最近两份完整恢复检查点、训练指标及外生回合hash记录。
- evaluation/：每场景完整数值轨迹与配对环境/模型身份。
- logs/processes.jsonl、run_commands.json：实际进程和运行命令。
- REPORTING_FIX.md、analysis_revisions/：额外离线核验器的精度顺序修正、未改变数值统计的报告措辞修订及前版哈希。没有更改正式训练/评估或放宽容差。

## 复算命令

在本实验目录中，使用已核验的 harl_sionna Python 环境：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/king/miniconda3/envs/harl_sionna/bin/python finalize.py
/home/king/miniconda3/envs/harl_sionna/bin/python describe_components.py
/home/king/miniconda3/envs/harl_sionna/bin/python summarize_checkpoints.py
```

finalize.py会完整聚合两次，要求results、audit、physical_statistics和REPORT的hash逐字节相同。单次聚合入口为 aggregate.py。辅助统计仅重读已有日志/轨迹，不训练模型。

run_training.sh保留实际运行入口。对本目录再次执行时只允许同协议恢复/跳过已完成任务；总预算固定250次更新，不能追加。它不会触发原实验的恢复脚本。环境路径可通过REPAIR_PYTHON指定，但已有完整检查点的恢复要求软件和设备签名完全一致。

本轮所有正式worker已经退出；不自动开启下一轮。未修改论文、原源码、原配置、原模型、原平均表和历史报告，未commit、push或创建PR。评估均为开发验证，预留最终测试保持未使用。
