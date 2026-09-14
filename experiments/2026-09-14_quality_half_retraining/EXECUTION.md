# 执行索引

本实验独立目录为 `experiments/2026-09-14_quality_half_retraining`。原工程与所有历史实验只读。

Python：`/home/king/miniconda3/envs/harl_sionna/bin/python`。始终设置 `PYTHONDONTWRITEBYTECODE=1`；Torch、OpenMP、BLAS各1线程。

首次已实际运行：

```bash
export PYTHONDONTWRITEBYTECODE=1
/home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_retraining/prepare.py
/home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_retraining/preflight.py
bash experiments/2026-09-14_quality_half_retraining/run_training.sh
```

`prepare.py`仅用于首次封存，已有manifest时拒绝再次执行。预检训练不计入正式300万步，正式任务全部从零开始。

`run_training.sh`启动独立管理进程；实际PID见launcher.json，各训练/评估子进程的命令和PID见execution_commands.jsonl。仅管理本轮自己的三个任务。每个种子1个CPU进程、10环境、400槽rollout；三个种子并行，最大并发3。实时状态在status.json和jobs/seed_*/status.json。

自动流程：三组100万步训练 → 每组20/40/60/80万步固定三任务及100万步全13任务评估 → 完整物理核验与两次聚合 → 中文报告 → 停止。评估使用同一20个开发种子，最终测试不运行。20万步种子104948945的fixed_0已先行完成作为评估入口验证，后续按完整输入/代码/轨迹哈希跳过，计入预定1500回合，未增加预算。

训练脚本在源码锁定后运行。`execution_seal.json`保存训练代码哈希；评估及聚合源码哈希单独封存于analysis_seal.json，在正式批量评估前建立。协议和配置哈希已在manifest中封存。

如发生中断，禁止从旧实验恢复。只可用本目录同一配置与软件环境的完整断点，例如：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_retraining/train.py --seed 104948945 --device cpu --cpu 0 --resume
```

恢复保留固定100万步上限；其他种子分别使用其自身目录和CPU。管理脚本不会自动重跑已有未完成任务；先按失败日志修复执行问题，再显式恢复。SIGTERM/SIGINT向本轮训练任务请求在更新边界保存完整状态，不终止其他项目。

训练完成后的独立评估入口（已完成项按哈希验证后复用）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_retraining/evaluate.py --seed 104948945 --cpu 0
```

其余两个种子同样执行。重新聚合与报告，不训练、不扩大评估：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_retraining/finalize.py
```

文件职责：PROTOCOL.md/manifest.json/configs用于封存设置；train_support.py隔离半质量奖励、原训练器包装、每槽审计与采样完整轨迹；train.py复用冻结HAPPO有限值内核；preflight.py验证更新、概率、奖励隔离和跨600槽终止精确恢复；evaluate.py加载不可变快照执行真实闭环；aggregate.py重新构建全部评估物理量、训练采样轨迹与配对统计；finalize.py重复聚合并生成中文报告。

CPU/CUDA实测与分配见resource_plan.json。运行成本、异常与重放验证分别保存于preflight_costs.jsonl、evaluation_costs、failures、aggregation_preflight.json和日志，未把脚本生成或任务启动记作完整实验结果。
