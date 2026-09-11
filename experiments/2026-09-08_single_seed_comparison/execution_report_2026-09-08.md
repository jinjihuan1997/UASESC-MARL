当前训练执行记录｜2026-09-08

已启动当前修订方案的单种子七方法长训练：每方法 10M 环境步、2,500 次更新，总 70M 步；同时最多四个方法。CPU 执行环境、动作推断和 GAE，GPU 执行大批量 PPO 更新；每次更新后同步推断副本。原 A/B 初筛已经包含此前的环境 bug 修复，但保持本次动作/奖励修订前的冻结设置，并继续独立运行。它不是本轮新方案的主结果。

当前输出：/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_single_seed_comparison/runs/seed1_10m_hybrid_parallel

启动命令（已经执行，不要重复启动）：

```bash
/home/king/miniconda3/envs/harl_sionna/bin/python -u experiments/2026-09-08_single_seed_comparison/run.py run --seed 1 --steps 10000000 --device hybrid --jobs 4 --eval-episodes 20 --output experiments/2026-09-08_single_seed_comparison/runs/seed1_10m_hybrid_parallel
```

工程变更：将已经独立验证的批量 Dirichlet 概率/熵计算接入当前 runtime；修复 GPU ValueNorm 的统计注册；加入 CPU 推断/GPU 更新及同步校验；队列支持 --jobs 并发调度、独立进程/日志/检查点和逐轮性能记录。物理环境、平均表、任务、奖励、动作、网络结构与 PPO 超参数未因性能优化改动。原已冻结代码与文件未修改，101 个旧输入哈希全部通过。

验证：17 项测试通过，包含 CPU/CUDA、float32/float64 概率和梯度、HAPPO/MAPPO 的完整固定 minibatch 更新、Adam 状态、终止/截断 GAE、归一化统计保存/重载；全部七种条件在混合执行中完成短训练并通过推断副本校验。

同一七方法、同种子、每方法 16,000 步的实测：

| 执行方案 | 完成 112,000 步的队列时间 | 含启动的总吞吐 |
|---|---:|---:|
| 混合执行，串行 | 125.18 秒 | 894.7 步/秒 |
| 混合执行，四组并发 | 51.12 秒 | 2190.8 步/秒 |

并发相对同一优化版本串行快 2.45 倍。七个方法的最终 actor、critic 和 ValueNorm 在串行/并发之间全部逐值相同；未使用策略成绩决定执行方式。这是本机短测，不是长程耗时承诺，也不是新算法性能提升证据。

长训练启动后的 15 秒资源采样：整机 CPU 平均约 80.2%，GPU 平均约 11.5%，整卡显存最高 2275 MiB，GPU 进程表确认四个训练进程。CPU 数值包含原队列和其他程序；GPU 更新较短，采样期间呈脉冲占用。

实时状态看运行目录 status.json 和 jobs/<method>/status.json。当前前四种方法执行中，后面三种自动补位。七种训练完成后自动评估 13 个场景，每场景 20 个配对环境种子，并导出 comparison.md / CSV / JSON。四种规则基线只评估，无需训练。详细计时、哈希与设备证据见本目录 execution_benchmark_2026-09-08.json。
