本目录是独立的“仅偏好变化”诊断，保留旧正式实验。先读 PROTOCOL.md；门槛结果在 GATE_REPORT.md，短训练结果在 PILOT_REPORT.md，独立复核后的综合解释在 REPORT.md。

从项目根目录执行（已有完成结果时脚本拒绝覆盖）：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TMC_PY=/home/king/miniconda3/envs/harl_sionna/bin/python
TMC_PREF=experiments/2026-09-10_preference_only
"$TMC_PY" "$TMC_PREF/gate.py"
"$TMC_PY" "$TMC_PREF/run_pilot.py"
"$TMC_PY" "$TMC_PREF/audit_and_report.py"
```

训练由冻结的父版本 formal_train.py 执行，仅传入本目录的新配置；无需修改 HARL、冻结环境、平均表或 Manuscript。两个学习方法各一个 CPU 进程、一个数学线程，同时从头训练 1M 步。训练前各 8k 步烟雾模型只做技术检查。每 25 次更新保存可恢复检查点；本启动器若失败会停止另一作业，不自行重置实验或更换训练预算。通过父版本正式训练入口 --resume 可从匹配配置/设备/manifest 的检查点恢复。

规则校准与保留预检使用 20261401—20261410 和 20261501—20261520。RL 与规则共同评估使用另外 20 个种子 20261601—20261620，原 13 个固定/切换场景。种子数字是标识，不是日期。该诊断不能与原来的硬门槛指令实验混用作同一设置下的排名。
