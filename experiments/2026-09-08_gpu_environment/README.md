# GPU / CPU 批量SC环境与三种子正式训练

**当前正式入口：**[21项正式训练脚本与命令](THREE_SEED_TRAINING.md)。种子85、218、966；每项10M步，6个CPU任务与1个GPU任务并行，支持完整状态续训和统一评估。已于2026-09-08 20:52:57 JST从零启动；最新进度见[status.json](runs/three_seed_sc_20260908/status.json)。

已开启[温控监视与自动冷却续训](THERMAL_MONITORING.md)：每秒检查CPU温度，过热保存暂停，至少冷却3分钟且低于75°C连续1分钟后自动恢复。最新守护状态见[thermal_guard/status.json](runs/three_seed_sc_20260908/thermal_guard/status.json)。

同一套批量环境可在CPU或GPU运行，已通过7个SC条件的环境对照、GPU短训练、三种子完整流程短测与CPU/GPU完整状态续训验证。旧单种子训练已停止并移至 `../_archive/2026-09-08_before_three_seed_formal/`；本轮正式输出仅在 `runs/three_seed_sc_20260908/`，`results/`保留独立工程验证证据。

资源分配依据见[并行资源短测方案](parallel_training_plan.md)：4种分配各完成7方法、每方法80,000步；采用的6 CPU + 1 GPU整批72.54秒，最高CPU温度84°C。三种子监督器已启用温控、资源锁、任务身份检查和每200,000步完整状态保存。

## 改了什么

- `tensor_env.py`：10个逻辑环境合并为一个张量批次。信道、平均表插值、模式选择、传输预算、缓存、AoI、奖励、观测及动作掩码在所选设备计算。使用float64进行物理计算，观测保持float32。
- `episode_source.py`：每个600时隙回合开始时，在CPU按原独立随机流准备拓扑、指令和外生随机数，一次上传。GPU逐时隙只读取当前输入；未来随机数不提供给策略。时隙内没有CPU参考环境的step调用。
- `tensor_train.py`：观测、动作、采样缓存、GAE和模型更新都保留为张量。直接复用冻结HARL的HAPPO/MAPPO actor与critic更新函数；保存每次更新的耗时、训练奖励及损失、权重和源代码快照。
- `common.py`及`reference/`：保存并验证87份参考输入。最终平均表SHA-256仍为 `a6f721686b28d6c28ccf811ab8b63e80ae8c135a08b9d7e1b4bbfe7ca13d91cc`。
- `run_suite.py`：7个GPU方案的顺序短训练检查，包含温控与超时。
- `mixed_probe.py`：独立CPU/GPU并发验证入口，CPU16运行IC_HAPPO，GPU与CPU18运行IC_MAPPO。

CPU仍负责Python控制代码、每回合初始化、外生随机数准备、少量更新统计和日志；GPU运行并不意味着完全不使用CPU。复用HARL更新函数中也保留了梯度检查等少量标量同步。

支持当前冻结的前馈网络、EP状态、最终平均表、16模式与7种SC条件。未实现CC、循环网络或其他质量模型。模式优先级、DS并列排序、逐项扣减预算及一时隙缓存补充规则保持原语义。表在同一UAV/模式下对各DS相同，因此使用每UAV/模式的紧凑张量；不同内容质量表不能直接套用此实现。

## 已验证结果

详细独立核查在 [results/verification.json](results/verification.json)。

| 验证 | 结果 |
| --- | --- |
| GPU环境逐步对照 | 9,080个环境时隙、72,640项离散比较通过；覆盖7种SC条件、完整600时隙回合、重置、指令切换，以及5种附加情形 |
| 附加情形 | 部分缓存与轮询调度、质量全部不可行、预算不足、无信道噪声、关闭显式指令约束 |
| 独立边界测试 | 预算刚好容纳/略不足/略超过两个载荷；相同动作与AoI得分按低编号优先；无效缓存时间戳 |
| HAPPO / MAPPO更新 | 同一批数据、同一小批次和agent顺序，与原HARL比较；CUDA参数最大绝对差约1.19e-7 |
| GPU训练覆盖 | 7种条件各完成16,000步；初始化与原相同seed完全一致，所有可训练actor和critic均更新，检查点可安全读取且数值有限 |
| CPU / GPU并发 | 两个不同方案同时各完成80,000步，包含20次PPO更新及多回合重置 |

物理量浮点比较使用有记录的容差，不宣称CPU和GPU逐位相同；离散调度、缓存、AoI与掩码逐项一致。CUDA与CPU的策略随机采样流不同，因此不会要求完整训练后的权重逐位相同。

## 实测速度

单方法IC_HAPPO、同一个CPU18、单线程数值库、同一张GPU、10环境×400步、同样16,000步；剔除第一批预热后：

| 执行方式 | 每秒环境步数 | 每4,000步 |
| --- | ---: | ---: |
| 原CPU多进程环境 + GPU更新 | 193 | 20.68秒 |
| 新GPU批量环境 + GPU更新 | 1,428 | 2.80秒 |
| 新CPU批量环境 + CPU更新 | 1,411 | 2.83秒 |

新GPU版约为原流程的7.38倍；新CPU版也达到接近速度，说明批量化和减少进程通信贡献很大。该比较在正式训练持续运行的背景下顺序测量，属于单核心、单方法的短测，不能据此把正式7组总工期直接除以7.38，也不能把约1%的CPU/GPU速度差解释为稳定优劣。

混合并发实测：IC_HAPPO使用CPU16，稳态约1,362步/秒；IC_MAPPO使用GPU和CPU18，约1,301步/秒。两组各80,000步，共64.17秒（含启动和收尾），最高CPU温度80°C，无错误。七组GPU顺序短测的最高CPU温度82°C。完整1000万步及更高并发的长期稳定性、总体工期仍需另测。

## 运行命令

以下命令在项目根目录运行，解释器为 `/home/king/miniconda3/envs/harl_sionna/bin/python`。输出目录必须不存在；已完成记录不会被覆盖。GPU命令需要可访问宿主机CUDA。

单方法GPU短训练：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 18 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_gpu_environment/tensor_train.py --method IC_HAPPO --device cuda:0 --steps 16000 --output experiments/2026-09-08_gpu_environment/results/manual_gpu_trial
```

纯CPU运行使用相同入口，把 `--device cuda:0` 改为 `--device cpu`，并选择独立输出目录。

同时跑一个CPU方案和一个GPU方案：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_gpu_environment/mixed_probe.py --steps 80000 --output experiments/2026-09-08_gpu_environment/results/manual_mixed_trial
```

七个GPU条件检查：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_gpu_environment/run_suite.py --steps 16000 --device cuda:0 --cpu-set 18 --output experiments/2026-09-08_gpu_environment/results/manual_gpu_suite
```

`mixed_probe.py`和`run_suite.py`为短测工具，达到85°C或单项超过600秒会结束其自身试验，不是正式长训练调度器。单方法入口可设置其他符合4,000步倍数的预算，但不能把80,000步短测权重当作完整训练结果。权重文件不包含optimizer、环境和RNG恢复状态，不能据此声称精确续训。

验证命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_gpu_environment/validate.py --device cpu --steps 620 --output experiments/2026-09-08_gpu_environment/results/manual_cpu_equivalence.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 18 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_gpu_environment/validate.py --device cuda:0 --steps 620 --output experiments/2026-09-08_gpu_environment/results/manual_gpu_equivalence.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/king/miniconda3/envs/harl_sionna/bin/python -m unittest discover -s experiments/2026-09-08_gpu_environment/tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GPU_ENV_TEST_DEVICE=cuda:0 taskset -c 18 /home/king/miniconda3/envs/harl_sionna/bin/python -m unittest discover -s experiments/2026-09-08_gpu_environment/tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_gpu_environment/verify_results.py
```

`verify_results.py`仅检查既有证据，不重训、不重写检查点。首次GPU短测、7条件覆盖短测和后续混合并发试验均保存各自执行时的源代码快照；后续增加了设备配置记录修正与训练指标日志，最新版本在混合并发80,000步试验中执行。
