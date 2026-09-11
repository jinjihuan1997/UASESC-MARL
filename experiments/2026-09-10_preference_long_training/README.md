本次从偏好任务的已验证参数开始，训练 IC_HAPPO 与 HAPPO_hidden_instruction，各3个种子、每模型1000万步。六个模型从头开始，不继承短训练权重。规则、平均表与物理参数保持原诊断版本。

CPU/GPU短基准已实际运行：六CPU并行预计约93分钟，四CPU加两GPU预计约150分钟。选择六CPU、绑定核心0—5、每进程一数学线程。加上评估和运行波动，预期约1.5—2小时；温控暂停会延长。GPU可正常使用，此次选择依据是实测完成时间。

正式状态在 status.json；训练进度在 jobs/seed_*/方法/status.json；每25次更新（10万步）保存完整恢复点。完成后自动评估六模型及三种规则，再从保存轨迹独立聚合。最终报告为 REPORT.md，完整数值为 report/results.json。若失败会写 attention_required，不自动改参数重试。

从项目根目录运行以下命令；启动与状态查询需在能访问宿主GPU温度传感器和进程的环境中执行：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TMC_PY=/home/king/miniconda3/envs/harl_sionna/bin/python
TMC_LONG=experiments/2026-09-10_preference_long_training
"$TMC_PY" "$TMC_LONG/supervisor.py" status
"$TMC_PY" "$TMC_LONG/supervisor.py" pause
"$TMC_PY" "$TMC_LONG/supervisor.py" start --resume
```

首次准备流程是 prepare.py → benchmark.py → 启动前接口/评估检查 → freeze.py → supervisor.py start。准备与冻结脚本拒绝覆盖既有正式记录。训练内温控只在本次队列运行期间有效，不创建应用自动监视任务。
