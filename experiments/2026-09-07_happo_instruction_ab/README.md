本轮用于初筛：在同一套执行约束下，给 HAPPO 策略显式输入指令，是否能带来超出规则投影的收益。正式训练已于 2026-09-07 启动；实时状态以 status.json 和 jobs/*.json 为准。短训练结果只用于检查程序，不能用于论文性能结论。

实验设置

| 方法 | 策略输入与行为 | 训练量 |
| --- | --- | --- |
| A | 屏蔽 actor 的显式指令 ID 和目标字段 | 3 个种子，每个 2,000,000 步 |
| B | actor 接收显式指令 ID 和目标字段 | 同 A |
| R | 均分资源、固定模式偏好，再由公共约束执行 | 无训练 |
| G | 利用本地可观测状态选择模式及分配资源，再由公共约束执行 | 无训练 |

A/B 的 critic 都保留完整指令状态，采用普通单头 critic、相同的全局优势归一化、奖励、动作维度和约束。同一种子的网络初始化一致。A 仍能看到物理 SNR 桶、受任务影响的可用动作掩码和待传负载，因此实验结论应限定为“显式 actor 指令输入的增益”。它不是完全隐藏任务信息的实验，也没有同时检验此前实现的指令多头 critic。

每组使用 10 个 CPU 环境、每次更新收集 400 × 10 步、共 500 次更新。环境每回合 600 时隙，3 架 UAV、30 个 DS。训练种子为 1、2、3；队列顺序为 A1、B1、A2、B2、A3、B3。首次启动时，受限执行环境未暴露 GPU 设备，采用了 CPU 配置。后续主机检查确认 RTX 4070 Ti SUPER 与 CUDA 正常，并实际完成 GPU 短训练；独立的 8,000 步检查耗时为 CPU 18.47 秒、GPU 36.81 秒，因此本轮保持 CPU 配置冻结。该短测期间原 CPU 训练持续运行，不能视为严格的长期速度基准。诊断记录位于 experiments/2026-09-07_gpu_access_check/。按短训练约 425 步/秒估算，训练和评估总计约 8–10 小时；实际取决于运行时负载。

训练完成后，使用 20 个共同环境种子 20261001–20261020，评估固定 balance、固定 AoI、固定 quality，以及第 300 时隙从 balance 切换至 AoI/quality，共 5 个场景。6 个训练模型与 R/G 共执行 800 回合、480,000 时隙。保存每时隙奖励、AoI、AoI 超标比例、预计质量、实际交付数、传输量、策略提议和执行后模式；按场景和种子核对各方法的信道轨迹哈希。G 是采用本地信息的启发式，不代表最优求解器。

本轮修改及证据

- CRL-SemCom-VidCI/experiment_scripts/comm/channel.py、transmission.py：按实际激活符号归一化发射功率，并屏蔽填充位置的信道噪声；避免低码率因填充比例额外获益。
- HARL/HARL/harl/envs/uav_escs/SC/uav_escs_env_sc.py：一个时隙只采样一次执行信道；拆分拓扑、信道、内容、指令和缓存随机流，消除动作对后续外生随机轨迹的影响；同时动作的事前模式掩码不再按上一轮资源分配排除模式；新增 A/B 输入开关和全部 5 个模式的选择。
- HARL/HARL/harl/envs/uav_escs/semantic_models/semantic_registry.py：校验质量表模式 ID 与登记项逐项对应，拒绝重复 SNR 网格点。
- HARL/HARL/scripts/build_harl_semantic_profile.py：严格加载权重；仅补齐经核实为全 1 的固定 shutter 缓冲量，不忽略缺失的学习参数。
- HARL/HARL/examples/evaluate_instruction_constraints.py：缺少所需 actor 权重时直接报错，避免评估退回启发式。
- HARL/HARL/scripts/calibrate_instruction_pilot.py：生成此次实际测量的平均质量/载荷表。
- HARL/HARL/scripts/run_instruction_pilot.py：生成配置、冻结源码、执行训练队列并进行配对评估。
- 相关测试位于 HARL/HARL/tests/test_instruction_pilot.py、test_semantic_registry_profile.py，以及 CRL-SemCom-VidCI/tests/test_active_symbol_channel.py。

原始数据、已有检查点和既有配置保持原样；新训练必须使用本目录 configs/ 中的配置。旧派生质量表不能直接替换本次实测表，旧结果也不应与本次结果混合比较。source_before/ 保存本轮修改前的主要运行文件，source_frozen/ 与 source_frozen_manifest.json 保存正式训练采用的代码。每组训练开始前及正式评估前检查代码、配置和质量表哈希；发生变化则终止，避免混合版本。

质量表范围与限制

只使用本地实际找回的 5 个已登记权重，原登记中的另外 11 个模式明确排除。输入继承现有数据加载流程，为 16 × 256 × 256 灰度张量，未重新训练编解码器。按原验证视频索引固定划分：视频 0–7 校准、8–13 诊断、14–19 保留。共选择 16 个校准片段、6 个诊断片段，在 7 个 SNR 点和 2 次噪声重复下执行 1,540 次重建。

表中数据仅由校准片段生成；诊断片段的 PSNR 预测 MAE 为 3.2989 dB、RMSE 为 3.8411 dB。每个模式和 SNR 使用平均质量，不能据此声称逐视频 QoS、内容感知质量预测或真实硬件闭环。目标阈值按测量前固定的规则从校准均值包络计算，没有按策略收益调参；质量与载荷硬约束之后原本为零的对应罚项在本轮配置中明确置零。

calibration/ 下保留原始逐次测量、视频索引与片段选择、权重哈希、诊断报告、模式表和 profile.npz。本文实验是修正后的系统学习初筛，不能单凭它宣称新的 PPO/HAPPO 算法创新。需要等待正式 A/B 和规则基线结果，判断是否值得继续实现并消融新的算法模块。

运行记录与检查

- status.json：队列阶段、当前组、完成组数。
- jobs/*.json：每组已完成步数、有限数值检查、运行目录、初始参数哈希及完成后的模型哈希。
- queue.log 和 A_*.log、B_*.log：正式日志；training/：正式模型与训练记录。
- tests.log：19 项测试通过。
- training_smoke_check.json：A/B 各 8,000 步均更新参数；同种子初始 actor/critic 一致；A/B 配置只有实验名和 actor 输入开关不同。
- evaluation_smoke_check.json：20 个完整回合、12,000 时隙，涵盖 5 个场景，切换时刻及配对轨迹检查通过；smoke_training/、smoke_evaluation/ 为独立检查数据。
- evaluation/episode_summary.json：全部训练及评估完成后生成的正式结果，运行结束前不会存在。

复现命令（从项目根目录运行）

```bash
PYTHONPATH=HARL/HARL:CRL-SemCom-VidCI /home/king/miniconda3/envs/harl_sionna/bin/python -m pytest HARL/HARL/tests CRL-SemCom-VidCI/tests -q
```

本次正在运行的队列命令如下；不要重复启动同一输出目录。脚本会拒绝覆盖既有校准记录、配置、冻结清单及正式单组日志。需要独立复现时，选择一个新的实验目录，按 calibrate → prepare → freeze → run 顺序运行。

```bash
/home/king/miniconda3/envs/harl_sionna/bin/python -u HARL/HARL/scripts/run_instruction_pilot.py --experiment experiments/2026-09-07_happo_instruction_ab --action run
```

新的独立复现示例（保持源代码版本和已冻结的数据、种子规则）：

```bash
PILOT_DIR=experiments/happo_instruction_ab_replication
PILOT_PY=/home/king/miniconda3/envs/harl_sionna/bin/python
"$PILOT_PY" HARL/HARL/scripts/calibrate_instruction_pilot.py --output "$PILOT_DIR/calibration"
"$PILOT_PY" HARL/HARL/scripts/run_instruction_pilot.py --experiment "$PILOT_DIR" --action prepare
"$PILOT_PY" HARL/HARL/scripts/run_instruction_pilot.py --experiment "$PILOT_DIR" --action freeze
"$PILOT_PY" -u HARL/HARL/scripts/run_instruction_pilot.py --experiment "$PILOT_DIR" --action run
```
