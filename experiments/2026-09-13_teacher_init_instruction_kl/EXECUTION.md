# 实际执行索引与复现

本地工作区：`/home/king/Downloads/Projects/2_th_paper_TMC`。该父目录不是Git根目录；其内HARL/HARL等子仓库有既有未提交修改，完整执行前状态见git_status.json。只读发布检出记录的HEAD为`7cc2372d19f36ed7d91ed289f83dcc79fe0f3e9d`。本轮没有依赖远程拉取，没有提交或推送。

输出仅在本目录。输入实际解析为：

- `../2026-09-11_alternating_training/`：冻结配置、TensorEnv/HAPPO/ValueNorm及runtime；适用runtime/AGENTS.md已读。
- `../2026-09-11_observation_matched_greedy/online_policy.py`及`modes_16/predictor.npz`：实际教师，原始模型与局部评分不变。
- `../2026-09-12_quality_mode_repair/`：原执行/核验、REPORTING_FIX、动作分布及加载语义。
- `../2026-09-12_policy_recomposition/`：本机最新已完成重组实验及其保护哈希。其他固定控制器的同口径旧开发轨迹索引保留在reference_index.json，没有用于新gate_dev打分。

manifest.json含输入、协议和配置SHA-256、路径与先验文档索引；seed_selection.json封存全部学生、训练、gate及条件B随机流。原始20开发环境与reserved都排除在采集/拟合/准入训练之外。新gate只在固定A3末尾使用。

## 执行顺序及实际停止点

1. prepare.py封存协议、来源和种子；validate.py的基础接口、损失、KL、回滚、教师260回合重现与学生烟测；checkpointing.py执行跨600终止恢复；benchmark_fit.py实测GPU/CPU拟合吞吐；seal_preflight.py封存A预检PASS。
2. run_dagger.py --train-only实际执行A0和三学生A1—A3。A0共享120回合，后续每学生每轮80回合（40确定性+40随机）。GPU单个拟合worker；CPU最多3个单线程采集worker。
3. A1恢复遇到已记录的设备类型错误，修复shuffle_rng.cpu()后从原A0端点继续，没有增加预算；详见AMENDMENT_001.md及execution_seal.before_amendment_001.json。
4. 固定A3与gate代码先封存到gate_execution_seal.json，再运行evaluate.py --gate（教师）及全部三个学生D/S0/S1/S2。共1690完整回合、1,014,000物理步。
5. audit_trajectories.py逐个重建全部42份采集和169份gate轨迹的信道/profile/物理量、前后局部输入和原策略动作/logp。正式采集标签全部复现。并发核验有17份重复离线重放，成本保留，没有重复物理采集。
6. gate_check.py得出0/3，STOP_AFTER_A。budget_diagnostics.py只对保存的学生输入进行预算门槛诊断，没有新增回合或梯度。critic预热、条件B集成预检、B0/B1训练与其过程/最终评估均按停止条件不运行。
7. aggregate.py读取原始轨迹统计；finalize.py实际运行两次数值聚合、比较全部主要JSON和中文报告哈希、重新核对保护文件和Git状态、核验两个后续入口确实被失败gate阻挡。

本轮新建完整学生，未加载旧父actor/adapter。每个学生监督50epoch、每actor6360个更新，合计76,320个正式监督更新。另有262个隔离原型更新。不能把监督初始化写作“新增训练为0”或原HAPPO无示范成果。

## 准确复现命令

实际使用解释器和版本保存在software_environment.json。当前解释器为`/home/king/miniconda3/envs/harl_sionna/bin/python`，Torch2.4.1+cu121、NumPy1.26.4；没有升级共享环境。

在本目录执行：

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export STUDY_PYTHON=/home/king/miniconda3/envs/harl_sionna/bin/python
bash run_experiment.sh
```

完成状态下该入口只重新核验并聚合，不重训、不重新采集。目录重定位后必须使用已配置兼容环境，通过STUDY_PYTHON指定解释器，不安装新依赖。

分项命令（缓存只在输入/协议/输出哈希匹配时复用）：

```bash
"$STUDY_PYTHON" run_dagger.py --train-only
"$STUDY_PYTHON" evaluate.py --gate
"$STUDY_PYTHON" evaluate.py --gate --student 1032592498
"$STUDY_PYTHON" evaluate.py --gate --student 788864185
"$STUDY_PYTHON" evaluate.py --gate --student 1626362722
"$STUDY_PYTHON" audit_trajectories.py --scope data
"$STUDY_PYTHON" audit_trajectories.py --scope gate
"$STUDY_PYTHON" audit_trajectories.py --scope gate --student 1032592498
"$STUDY_PYTHON" audit_trajectories.py --scope gate --student 788864185
"$STUDY_PYTHON" audit_trajectories.py --scope gate --student 1626362722
"$STUDY_PYTHON" gate_check.py
"$STUDY_PYTHON" finalize.py
```

为避免结果和输入同名模块碰撞，新入口使用study.py；冻结评估脚本与旧aggregate.py通过独立模块名加载。所有资源物理计算保留“动作先转float64再加1”的运算顺序。

## 文件职责

| 文件/目录 | 内容 |
|---|---|
| PROTOCOL.md、manifest.json、seed_selection.json、config.json | 封存定义、配置、路径和全部种子 |
| teacher_interface.py | 冻结同观测16模式教师；纯局部资源/模式标签、有效标签/跳过原因 |
| student_policy.py | 新建四个原结构actor、原分布和监督KL目标 |
| collect_demos.py、trajectory.py | 实际教师/学生闭环采集与逐槽原检查 |
| train_imitation.py、run_dagger.py | 固定20+10+10+10轮监督及独立学生数据聚合 |
| students/<seed>/A0—A3/ | 每阶段固定端点、resume（权重/Adam/RNG/进度）、日志和哈希 |
| data/、gate/ | 压缩原始数值轨迹、局部输入/掩码/动作/logp、标签、物理状态及独立种子 |
| evaluate.py、gate_check.py | 固定A3教师/学生评估和全体准入判定 |
| validate.py、preflight/、preflight.json | 实际小规模预检与结果 |
| checkpointing.py | 原环境/source状态恢复，实际跨600终止一致性验证 |
| audit_trajectories.py、audits/ | 独立信道/profile/物理重算及策略/预算条件标签重放 |
| budget_diagnostics.py | 完成轨迹上的离线容量门槛敏感性，不是控制器 |
| warmup_critic.py、train_joint.py、kl_control.py、happo_kernel.py | 条件后续入口和KL/Adam回滚模块；失败gate禁止运行 |
| B_preflight.json | 明确NOT_RUN_STAGE_A_GATE_FAILED；不得将原型KL检查冒充B集成通过 |
| costs/、logs/ | 实际计算账本、命令、进程退出状态和异常；运行时间元数据不参与数值哈希 |
| report/results.json、physical_statistics.json、gap_decomposition.json | 实际分数、物理统计和按当槽指令加权的奖励分差 |
| report/diagnostics_by_scene.json、budget_threshold_diagnostics.json、post_quality_comparison.json | 逐场景拟合误差、门槛实例和切换后对比 |
| gate_report.json、report/audit.json、report/reproducibility.json | 准入、保护/物理审计、两次聚合一致性 |
| report/REPORT.md、status.json | 中文结论与实际条件停止状态 |

## 未运行项目与限制

未运行的是用户条件明确禁止继续的critic预热、6M RL与相关B评估，并非把缺失结果记为完成。后续训练入口的完整联调没有执行，不能声称已经通过正式B集成。原型已核验解析Dirichlet/Categorical KL、候选参数与Adam逐值回滚、关闭KL时的有限原HAPPO更新等价及环境断点恢复；这些仅是实现预检，不能证明KL带来科学收益。

异构原型测试没有记录完整前向次数，账本该项为null；实际物理步、监督/原型optimizer.step、正式前向、标签查询和重复离线审核已单列。最终测试未使用；论文、Git提交和远程未修改。完成此固定范围后停止，不自动增加示范轮次、种子、浓度或联合训练。
