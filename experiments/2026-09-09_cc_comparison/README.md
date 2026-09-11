CC 已接入独立训练流程，使用与运行中 SC 相同的指令、奖励、物理环境参数、学习算法设置和随机种子。原 SC 训练及所有历史 profile/checkpoint 保留。

| 项目 | 正式设置 |
|---|---|
| 方法 | CC_IC_HAPPO、CC_IC_MAPPO |
| 训练种子 | 85、218、966，与 SC 配对 |
| 每组预算 | 1000 万环境步；2500 次 PPO 更新 |
| 总预算 | 6 组，6000 万环境步 |
| 环境与批次 | 每回合 600 槽，10 个环境×400 步/更新 |
| CC 模式 | 全部 20 个 QP/LDPC 组合 |
| 资源 | CPU 0、4、6，3 组同时训练；GPU 继续运行 SC |
| 后续评估 | 每模型 13 场景×20 评估种子×600 槽，共 936000 槽 |
| 工期预估 | 短测外推训练约 3.5 小时，含评估与降温先预留 4–5 小时 |

核心修改是把“尝试发送”和“成功接收”分开。失败发送照常消耗资源、释放缓存，但接收端 AoI 增长而不会被重置。整个流程采用单次发送、无重传；成功时的质量由逐片段概率加权后的条件质量提供。没有把平均 PSNR 当作每次必然成功的质量，也不重复扣除失败损失。固定载荷保留平均模型定义。

SC 配置的 Q_max=33 是归一化参考值，没有对 33 dB 以上质量截断。CC 沿用同一实际奖励公式，无需为此调整 SC 参数。完整建模选择和科学限制见 [PROTOCOL.md](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/PROTOCOL.md)。

验证记录在 [evidence/verification.json](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/evidence/verification.json)。9 项检查通过，包含全部模式可达、资源/AoI/缓存转移、条件质量插值、策略不能改变外生随机数、完整回合 SC 轨迹哈希一致、跨回合精确恢复、配置防篡改和温控。两套资源分配的 6 组 40k 短训练及 7200 槽评估均完成，最终指标文件完全一致。这些只验证实现，不能证明方法优劣或收敛。

4 路、3 路短测都曾记录单次 85°C，没有连续达到触发条件；不能声称 3 路已实证降低峰值。选择 3 路是因为 6 组任务仍只需两批、外推时间相近，减少同时繁忙核心并保留系统余量。正式队列继续使用温控保存/暂停/冷却/恢复机制，长时间运行仍可能触发降温并延长工期。

控制命令如下。正式 run 为 `runs/three_seed_cc_20260909`，实际状态以 status 命令和进程为准。

```bash
# 状态
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/run_cc_training.sh status
# 首次启动（拒绝重复启动）
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/run_cc_training.sh start
# 保存完整状态并暂停
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/run_cc_training.sh pause
# 确認 status 为 paused 后继续
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/run_cc_training.sh start --resume
```

启动后自动执行训练、最终模型评估和 CC 两算法汇总。若相应 SC 模型也已完成评估，会生成 `sc_comparison/comparison.json`；否则写明待完成的评估列表，差值保持空值，不用短测结果替代正式结果。之后可运行：

```bash
/home/king/miniconda3/envs/harl_sionna/bin/python \
  /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/runs/three_seed_cc_20260909/source/compare_sc_cc.py \
  --cc-run /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/runs/three_seed_cc_20260909
```

验证命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/king/miniconda3/envs/harl_sionna/bin/python -m unittest discover \
  -s /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_comparison/tests -p test_cc.py -v
```

实际执行的环境为本目录 `tensor_env.py` 的 TensorCCEnv；`cc_setup_env.py` 只复用 SC 的场景初始化和观察/动作空间，禁止用它 step。旧 HARL CC 环境的 5 模式映射不参与新流程。各 run 冻结独立 source、inputs、configs 和哈希，恢复使用相同模型、优化器、环境、缓存及随机数状态。

正式训练已于 2026-09-09T14:07:50.884745+00:00 启动，supervisor PID 4003618。启动后确认 3 个任务均有训练步数推进，SC 与 CC 冻结输入哈希不变。该条是启动历史，最新进度请使用上面的 status 命令。
