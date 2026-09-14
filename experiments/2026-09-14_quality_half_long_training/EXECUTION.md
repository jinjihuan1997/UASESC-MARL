# 长期训练执行索引

本次用户明确授权延长训练。旧目录 `experiments/2026-09-14_quality_half_retraining` 全部只读，其100万步实验仍保持完成状态。新增文件和训练结果全部位于本目录。PROTOCOL.md/manifest.json封存累计1000万步、新增900万步/种子的范围。

已运行准备和预检：

```bash
export PYTHONDONTWRITEBYTECODE=1
/home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_long_training/prepare.py
/home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_long_training/preflight.py
bash experiments/2026-09-14_quality_half_long_training/run_training.sh
```

prepare.py只运行一次。preflight.py核验旧完成状态、元数据白名单迁移、续训第一轮逐值一致、跨600槽终止精确恢复，并为三个种子各写入新的update250起点断点。migration_results.json记录原/新检查点哈希，原文件不修改。

长期预算为各自update251至2500。开始前已训练250次更新，后续2250次，每次4000环境步。总新增2700万步，累计3000万步；status.json分开记录completed_additional_steps和completed_cumulative_steps。

运行资源：3个CPU进程，Torch/BLAS各1线程，分别绑定逻辑核14/15/16。实时检查过其他研究进程；不停止或修改其他任务。resource_plan.json保存续训吞吐估计，实际速度看jobs/seed_*/timing.jsonl，可能随系统负载变化。launcher.json保存管理进程PID，execution_commands.jsonl保存准确子进程命令。

完整恢复命令（仅本目录同配置断点，不能扩大预算）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_long_training/train.py --seed 104948945 --cpu 14
```

该入口严格恢复新目录已匹配协议的断点，并限制累计1000万步。其他两种子使用各自目录和CPU15、16。管理脚本锁阻止多个管理进程，原formal_train文件锁阻止重复训练同一任务。SIGTERM/SIGINT只请求本轮任务在更新边界保存状态后停止。

自动流程：三种子续训 → 200/400/600/800万步固定三任务评估 → 1000万步全部13场景评估 → 独立核验 → 两次完整聚合与中文报告 → 停止。所有评估固定20个开发种子、600槽；不使用最终测试。

源代码锁见execution_seal.json，评估和聚合代码锁见analysis_seal.json。train_support.py沿用上一轮已验证的有限值HAPPO内核与半权奖励包装；continuation.py只转换新旧任务元数据并验证其余完整状态；evaluate.py原确定性闭环不变；aggregate.py分开累计/新增训练成本并比较全部参照；finalize.py生成报告并复算两次。

validate_pipeline.py将迁移的100万步策略对旧固定均衡轨迹保存输入做全部动作重放（每种子12000条观测，三种子共36000；不新增物理步），并重新核验一段已生成的新训练完整回合。它不更新模型。

本轮结束后的重算入口：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-14_quality_half_long_training/finalize.py
```

状态未成为complete、全部模型未达到1000万步、全部1500回合评估和重复聚合未通过前，不能将长期实验称为完成。异常记录保留；不会按得分换种子、延长预算或修改奖励，不推送GitHub或修改论文。
