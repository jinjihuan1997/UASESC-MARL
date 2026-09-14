# 执行索引和复现命令

实际工作区：`/home/king/Downloads/Projects/2_th_paper_TMC`。
实验目录：`experiments/2026-09-14_tailrl_applicability_diagnostic`。
运行环境：`/home/king/miniconda3/envs/harl_sionna/bin/python`，Python 3.10.19、PyTorch 2.4.1+cu121、NumPy 1.26.4。

本轮工作区不是有效 Git 根，HARL/.git 为空；源码和模型依据父 manifest 及 SHA-256 冻结。相邻仓库 Git 状态只读记录在 git_inspection.json，未替换任何冻结实现。

## 实际执行顺序

1. 读取附件、冻结源码及历史报告，查阅 TailRL 原论文 §4 和附录 C；生成 PROTOCOL.md、manifest.json、训练日志前缀快照。只使用当时已存在的 1M/6M 不可变检查点。
2. `preflight.py`：指令输入、物理状态固定、logp、TailRL排序/并列/平移测试。没有推进物理时隙。
3. 先执行104948945/1M/fixed_2/动作重复0。该完整20环境批次检查通过后纳入正式计数，未作为额外预检回合重复计费。
4. `run_evaluation.py`：2个CPU进程，分别固定逻辑CPU0/1，每进程Torch/BLAS线程1，全部3模型按固定范围评估。已有同身份、同哈希完整批次直接复用。每批20个完整600槽回合，合计153批、3060回合、1,836,000步。
5. `history.py`、`training_evidence.py`：历史轨迹索引和已有日志/训练轨迹重算。
6. `controlled_stats.py`、`additional_analysis.py`：阈值、配对统计、动作模式、成功成本和权重诊断。
7. `gradient_analysis.py`：固定6M模型，前8个预定开发环境、16动作重复；每actor 76,800真实采样输入。11种权重，actor前向样本合计921,600，向量雅可比计算6,600次。只用autograd.grad，零optimizer对象及更新。logp和概率比复现误差均为0。
8. `aggregate.py`、`plot_results.py`：中文报告、完整JSON和图表。
9. `reproduce.py`：从同一冻结输入重复完整数值聚合，检查所有目标JSON/报告哈希、NPZ逐数组数值、保存梯度的范数/余弦，然后调用`validate.py`复核输入和预算。

示例命令（项目目录下执行，所有输出都留在本实验目录）：

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
diag_dir=/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-14_tailrl_applicability_diagnostic
python_exec=/home/king/miniconda3/envs/harl_sionna/bin/python

# 仅重新聚合、核验已完成结果，不增加物理回合
bash "$diag_dir/run_diagnostic.sh"

# 只有需要恢复本轮缺失批次时使用；完整批次只在哈希一致时跳过
bash "$diag_dir/run_evaluation.sh"

# 单独复核或重算梯度，均不修改模型
"$python_exec" "$diag_dir/validate.py"
taskset -c 0 "$python_exec" "$diag_dir/gradient_analysis.py"
```

不要在已封存实验内重新运行`prepare.py`覆盖manifest；它保留的是运行前模型、动作种子及原长期训练状态快照。这里的恢复不调用任何旧训练脚本，不追加训练预算。

## 成本口径

| 项目 | 本轮新增物理步 | 说明 |
| --- | ---: | --- |
| 冻结随机/确定性评估 | 1,836,000 | 3060个完整回合，含首个通过检查的正式批次 |
| 预检 | 0 | 只检查观察、分配、分布及公式，不commit |
| 反事实梯度 | 0 | 重新前向和求导，不改变参数 |
| 历史报告/训练轨迹分析 | 0 | 已有数据，不计为本轮采集 |
| 重复聚合/离线核验 | 0 | 不调用环境推进 |
| 本轮RL或TailRL训练 | 0 | 优化器更新0 |

153批评估的进程用时合计约439.70秒，不能当作并发墙钟时间；梯度计算约43.61秒。原长期HAPPO作业在独立流程中运行，未计入本诊断成本、未停止或修改它。计时属于机器运行元数据，不纳入确定性数值比较。

## 文件职责

| 文件/目录 | 内容 |
| --- | --- |
| PROTOCOL.md / manifest.json | 执行前定义、阈值、模型与输入哈希、种子、场景、固定预算 |
| diag_support.py | 冻结加载、目录写入保护、禁止backward和optimizer.step、共同质量计量 |
| evaluation/ | 逐时隙物理量、局部观测、真实请求动作、logp、critic值、GAE、外生哈希、独立核验 |
| history_index.json | 924个历史逻辑记录、别名/重复标识、逐文件SHA |
| training_evidence_index.json / log_snapshots/ | 18份日志与120份已保存训练轨迹证据索引，当前日志只截到6M |
| analysis_arrays/ | 每组配对Q、总奖励、成功掩码与Tail/GAE权重，供重算 |
| gradients/ | 各actor反事实梯度向量，不是更新后的模型 |
| report/controlled_results.json | 逐模型/检查点/场景/环境分布、阈值和相关性 |
| report/additional_results.json | 成功失败原因、动作等价组、两半重复、逐时隙正权重 |
| report/gradient_results.json | 整体及每actor梯度范数和完整余弦矩阵 |
| report/history_distributions.json | 所有可用历史场景、方法、检查点和指令分布 |
| report/history_episode_metrics.json | 可重算的逐环境历史指标 |
| report/training_evidence.json | 非平稳训练轨迹与日志曲线，不混作冻结策略重复 |
| report/results.json / REPORT.md | 主要数值与中文结论 |
| report/audit.json / reproducibility.json | 保护输入、实际种子、核验误差、成本、重复聚合 |
| costs/ / *.log | 实际执行账本和命令输出 |
| execution_seal.json | 重复聚合时的实现哈希；不冒称所有分析代码都在采样前写完 |

质量是平均profile预测PSNR及其交付归一化收益，不是视频解码实测。所有新环境是原20个开发验证种子，最终测试未使用。未修改论文、训练任何新模型、推送GitHub或启动后续实验。

## 已解决的复算问题

第一次完整重复聚合因历史日志格式差异在`history.py`汇总时报`KeyError: instruction_counts`：补充索引中加入了旧adapter训练日志，其统计字段位于`collection`内。当前权重日志汇总现仅读取原已封存的short/long前缀；全部18份日志仍由`training_evidence.py`单独解析，没有删除旧证据。保留`reproduce_attempt1.log`、`history_recompute_attempt1.log`和`failures/reaggregation_attempt1.json`。修正后已完整重复聚合通过，未新增物理回合或改变数值定义。

图表视觉检查还修正了梯度图横轴标签重叠。原实现哈希保留在`execution_seal_v1.json`，现行哈希记录在`execution_seal.json`。两次完整数值聚合的JSON、报告及NPZ数值完全一致；所有保存梯度向量重算的范数和余弦也逐值一致。
