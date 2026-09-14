# 执行与复现索引

本目录是独立实验。冻结输入从当前项目相对位置解析；旧Git根实际位于嵌套目录及旁边.github-sync快照，父项目根本身不是Git仓库。读取参考HEAD=7cc2372d19f36ed7d91ed289f83dcc79fe0f3e9d。本轮没有网络Git操作。git_inspection.json保存完整初始状态。

已发现且读取的上轮文件布局：../2026-09-12_quality_mode_repair/{PROTOCOL.md,manifest.json,EXECUTION.md,REPORTING_FIX.md,adapter_policy.py,repair_evaluation.py,aggregate.py,repair_support.py,report/REPORT.md,report/MODE_AND_REWARD_DETAILS.md,report/results.json,report/physical_statistics.json,report/audit.json,report/reproducibility.json}。辅助gap_reanalysis.json不作为原始依据。所有文件及所需原文件SHA在本manifest封存。模型详细状态与相对路径见manifest.models；208条可复用轨迹见reference_index.json。

## 环境及固定指令

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/home/king/miniconda3/envs/harl_sionna/bin/python
EXPERIMENT=/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-12_policy_recomposition
```

上面的解释器已在software_environment.json记录并实时验证；重定位时可指定兼容的已有PYTHON，不升级共享环境。prepare.py只允许在没有manifest的新目录封存一次；不要对完成目录重做prepare。

预检实际执行命令：

```bash
"$PYTHON" "$EXPERIMENT/evaluate.py" --seed 104948945 --controller C0 --scene fixed_0 --preflight
"$PYTHON" "$EXPERIMENT/validate.py" --seed 104948945
"$PYTHON" "$EXPERIMENT/validate.py" --seed 111868397
"$PYTHON" "$EXPERIMENT/validate.py" --seed 160441552
"$PYTHON" "$EXPERIMENT/validate.py" --finish
```

首条为完整20回合吞吐测量；被后续预检按相同哈希复用，无重复计数。三父预检实际并发3个进程。原启动的同名模块导入失败日志和修复说明保留在preflight_failures/import_collision.json；没有因此推进物理步。

正式推理评估：

```bash
PYTHON="$PYTHON" bash "$EXPERIMENT/run_evaluation.sh"
```

仅启动C1/C2/C3三个控制器×三父模型，固定13场景。调度最大3个工作进程；完整回合文件/身份/源码哈希匹配才跳过。不会新增训练、拟合参考或运行reserved_final_test。

独立核验与数值聚合（实际连续执行两次）：

```bash
"$PYTHON" "$EXPERIMENT/aggregate.py"
"$PYTHON" "$EXPERIMENT/aggregate.py"
```

每轮聚合重新查profile、重建物理与奖励、按每次20环境的原批次形状重放新控制器保存的SUT及局部UAV观测，最后重新核对保护输入与Git状态。reproducibility.json记录确定性文件SHA。再次执行聚合不会推进环境或产生训练。

## 产物

- support.py：新目录写保护、禁止梯度/优化器、开发种子构造器；继承冻结源next和物理模型。
- composition_policy.py：直接加载StochasticPolicy和原ResidualPolicy，逐样本固定路由；不实例化critic或优化器。
- evaluate.py：真实完整闭环、原在线独立核验、扩展逐槽记录、原子保存。
- validate.py：1280回合预检、精确轨迹关系、混合批次、C0额外输入重放。
- aggregate.py/report_writer.py：独立物理复算、来源与实际动作重推理、配对bootstrap、中文结果。
- manifest.json/PROTOCOL.md/execution_seal.json：输入及执行定义封存。
- configs/、seed_selection.json：原物理配置和允许环境种子。
- evaluation/：本轮117条新数值轨迹；preflight_traces/、preflight_repeat/为单列的1280个额外计算回合。
- compute_ledger/：每次实际推进计数、完整/失败状态、耗时和命令；logs/：调度和所有工作进程日志。
- reference_index.json：208条复用原轨迹及原元数据/SHA，不复制或覆盖旧数值。
- report/{results,physical_statistics,paired_differences,gap_decomposition,audit,reproducibility}.json和REPORT.md。

新轨迹policy_source的0/1/2分别对应本父原UAV/P_all/P_quality，resource_source的0/1对应本父原SUT/原均分动作；每个轨迹元数据包含对应模型SHA索引。post_uav_obs为原分配后float32局部观测，不能用全局状态路由。原模式16类及无交付-1保留。

统计随机种子固定202609122122、4000次20环境成组bootstrap。原始未乘100的奖励位于npz.trace[:,:,0]，results中保留每完整回合的原量。逐UAV最大AoI诊断不是可加全局max成本；共同奖励精确分解使用原trace全局分项。

## 追加独立核验与完成封存

independent_profile_check.py独立使用NumPy从初态位置与外生噪声重建ATG/SAT信道、SNR耦合、原NPZ插值、边信息载荷和指令SNR门槛；不读取native update_channels/update_tables输出。随后对全部325条轨迹再次逐值核验原物理/奖励，保留原容差，没有新物理推进：

```bash
"$PYTHON" "$EXPERIMENT/independent_profile_check.py"
```

report/independent_profile_audit.json保存其源码SHA和逐轨迹检查。report/focus_case.json将全部13场景真实质量时隙按时隙/交付数聚合，完整保留三个父模型三UAV；不能与fixed_2混用。

实际完成了三次完整聚合：第一次通过后补强中文解释和全13场景质量焦点统计，四份核心数值输出保持原SHA；随后用最终分析代码完整聚合两次，验证全部最终数值与REPORT哈希一致。初次/最终记录在aggregation_runs/，分析修订仅影响报告解释和新增聚合视图，没有修改协议或运行中的任何控制器。

最终封存使用finalize.py，须先确认聚合工作进程已退出：

```bash
"$PYTHON" "$EXPERIMENT/finalize.py"
```

该入口只核验、写交付索引和完成状态，不训练、不增加环境步。resource_summary.json保存实测评估时间和资源设置；无温控暂停、恢复或终止他人进程。delivery_manifest.json为本目录全部交付文件索引（自身除外）。
