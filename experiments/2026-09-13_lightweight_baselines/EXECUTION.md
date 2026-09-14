# 执行与复算索引

本目录是实际日期 2026-09-13 创建的隔离实验，根目录由脚本文件位置解析。未写回历史目录，未下载数据、拟合预测器、训练 RL 或推送 GitHub。冻结输入位置与逐文件哈希以 manifest.json 为准。

实际解释器：`/home/king/miniconda3/envs/harl_sionna/bin/python`。搬迁环境时须使用匹配版本依赖，并重新核对输入/协议哈希；不要修改共享环境来静默兼容。

**已实际执行的顺序**

在本目录运行以下等价命令。原过程从项目根目录调用绝对脚本路径；`evaluation_queue.log`、各方法 `logs/`、`benchmark.log`、`finalization_execution.jsonl` 保留实际输出与进程命令。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LIGHT_PYTHON=/home/king/miniconda3/envs/harl_sionna/bin/python
"$LIGHT_PYTHON" prepare.py
"$LIGHT_PYTHON" validate.py
"$LIGHT_PYTHON" run_evaluation.py --evaluate-only
"$LIGHT_PYTHON" benchmark_latency.py
"$LIGHT_PYTHON" aggregate.py
"$LIGHT_PYTHON" finalize.py
```

`prepare.py` 仅供新实验首次封存，已有 manifest 时拒绝重写。本次 validate 首次遇到记录器缺字段，修复后重做；失败及额外计算未删除。aggregate 首次遇到括号语法错误，修正后完成第一次数值聚合；finalize 再执行一次完整聚合和两次报告渲染，逐文件比较确定性结果。

**恢复已有同协议任务**

```bash
bash run_evaluation.sh
```

入口要求预检已通过，只跳过协议、源码、模型输入和轨迹哈希完全匹配的完成项。六个评估任务最多三个 CPU 工作者并发，随后延迟测试单进程运行，再聚合核验。不会新增训练或扩大场景、种子、候选集合。已经完成的本轮无需再次启动。该命令可能重复离线聚合核验，其执行元数据会追加保留。

**只重算现有轨迹，不重新评估或计时**

```bash
"$LIGHT_PYTHON" aggregate.py
"$LIGHT_PYTHON" write_report.py
```

同样会独立重建物理量、重放新方法的全部请求动作并重新检查保护哈希。此过程从白名单外生初态构建审计表，不调用真实物理推进；不加载任何新教师学生，不接触预留最终测试。完整两遍复算证明与哈希在 report/reproducibility.json。

**主要文件职责**

| 文件 | 内容 |
| --- | --- |
| PROTOCOL.md / manifest.json | 执行前四方法定义、20种子、13场景、计时预算、bootstrap与输入哈希 |
| light_support.py | 原模块只读导入、路径重定位、白名单环境、禁止梯度/越界写、原子输出 |
| lightweight_policies.py | 四个新方法，仅公开 SUT / 单架 UAV 观测接口 |
| reference_policies.py | 原强基线、原规则、原 RL 和原 C1/C3 的冻结加载与推理 |
| evaluate.py | 原两阶段时序、逐槽原核验、完整闭环轨迹和旧单模式规则逐值补评 |
| validate.py / preflight.json | 边界、资源、信息、原模型动作重放和可重复性预检 |
| independent_physics.py | 独立 NumPy 信道/profile 计算，复用原独立调度与奖励核验职责 |
| benchmark_latency.py | 统一完整决策路径计时，原始纳秒与加载/存储成本 |
| aggregate.py / write_report.py | 配对统计、完整数值物理重建、中文主报告与附表 |
| finalize.py | 重复聚合、Git/保护哈希终检、审计与停止状态 |
| run_evaluation.sh / run_evaluation.py | 固定有限任务队列；只有本实验任务，无自动训练 |
| reference_index.json | 169份已核验旧轨迹索引和26份旧规则补评依据 |
| evaluation/ | 四方法×13文件，每份20完整600槽回合；NPZ与哈希元数据 |
| supplement/ | 两个原单模式规则的260回合/方法补评，匹配所有旧可比较数组 |
| preflight_traces/ / failures/ / preflight_failures/ | 实际预检、失败和重复执行证据 |
| latency/ | batch1/20的38份原始计时数据、输入来源与哈希 |
| costs/ / logs/ | 分种类实际步骤、回合、重放、计时、命令、进程和异常 |
| execution_seal.json / benchmark_execution_seal.json / analysis_execution_seal.json | 正式执行、计时与统计代码封存 |
| report/results.json / episode_scores.npz | 全方法/父模型/场景/环境奖励及未乘100的数值、4000次重采样索引 |
| report/physical_statistics.json | 各场景×指令×父模型×UAV物理与16模式/等价组原始计数 |
| report/paired_differences.json | 所有预定配对分差、区间、固定与切换场景和逐父模型 |
| report/complexity.json | 实测延迟、吞吐、离线依赖、候选次数、参数/数组及文件存储量 |
| report/independent_audit.json / audit.json | 247份完整轨迹独立重建及终检、最终测试未使用、成本与零训练 |
| report/reproducibility.json | 两次数值聚合及报告字节哈希一致证明 |
| report/REPORT.md / DETAILS.md | 中文结论、全部方法总表、全部场景与配对区间附表 |
| status.json | 全部完成、无未完成项、按固定范围停止 |

数值轨迹保存请求与实际执行模式、实际决策指令、资源动作/份额/预算/用量、缓存/时间戳/AoI、交付/预测质量和所有奖励分项；新轨迹额外保存 SUT / post-UAV 观测、掩码及真实可行性日志。真实可行性仅为核验输出，不传给决策函数。PSNR 为固定平均表预测。

**成本和异常**

正式 624,000 步；补评 312,000 步；预检 10,802 步；实际新增共 946,802 步。旧轨迹 3,380 回合只复用；参考推理重放因预检重做共312,000样本决策，物理步为0。计时包括1140预热批与11400测量批，计131,670样本决策，物理步为0。完整独立聚合两次，各核验2,964,000轨迹槽、重算624,000新策略请求动作，无新增物理推进。

预检错误是新记录器未填原核验器不返回的 instruction_id；已从本槽 info.gid 记录，原策略/环境/奖励/容差没有调整。失败的2步及重复参考重放保留。聚合脚本开发时一个多余右括号在启动前报语法错误，0计算，日志 aggregate_first.log 保留；成功日志 aggregate_attempt2.log 和 aggregate_second_complete.log。没有正式回合失败、丢弃或换种子。

首次报告渲染有一次 REWARD_PARTS 常量未定义错误，未修改数值结果也未触发新的环境推进；修复本报告模块常量后重新生成。该失败的日志与 analysis_execution_seal_attempt1.json 保留，最终统计封存见 analysis_execution_seal.json。

资源测量与运行设置在 software_environment.json / benchmark_resource_start.json，运行元数据与确定性结果分离。无本实验温控暂停或重启；没有建立持续监视、后台后续实验或自动推送任务。
