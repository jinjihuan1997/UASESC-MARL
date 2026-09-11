# 三随机种子正式训练脚本

本次固定随机种子为 **85、218、966**。这三个数在首次流程短测前一次抽取，此后不因训练或评估结果更换。7个SC方法均使用同一组三种子，共 **21项训练任务，每项10,000,000步，总计210,000,000步**。

当前正式训练目录为：

`/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908`

正式长训练已于 **2026-09-08 20:52:57 JST（11:52:57 UTC）** 从零启动，首次监督器PID为3295936；21项任务首次初始化均不加载已有模型。当前状态见[status.json](runs/three_seed_sc_20260908/status.json)，首次启动身份与冻结清单哈希见[launch_record.json](runs/three_seed_sc_20260908/launch_record.json)。运行中的队列不要再次执行首次启动命令。

**2026-09-09 01:17:26 JST恢复记录：** 前次队列在09-08 21:55触发95°C温度保护后有序退出，共保存39,428,000步（18.775%）；发现时已暂停约3小时20分。7份完整恢复点的哈希、安全加载、更新计数、预算和冻结身份均核验通过，宿主机温度连续10秒为40–44°C后，沿用原6 CPU + 1 GPU资源配置及参数恢复。该次监督器PID为3414705，当前执行身份见[current_execution.json](runs/three_seed_sc_20260908/current_execution.json)，日志和核验记录见[本次恢复目录](runs/three_seed_sc_20260908/execution/resume_20260909_0115/)。温度突升的具体原因尚未确定。当时尚未配置退出后的自动恢复；现已由01:35启动的[温控守护](THERMAL_MONITORING.md)处理保存暂停、冷却与自动续训。

按用户要求，原单种子队列的49个相关进程已结束，原训练和两组资源测速输出已移至 `../_archive/2026-09-08_before_three_seed_formal/`。旧进度不计入本轮；迁移短测和验证证据仍单独保留在 `results/`。详细切换记录见[retirement.json](runs/three_seed_sc_20260908/transition/retirement.json)。

20:57 JST启动验收通过：已观察约243秒，7项首发任务均超过200,000步，实际CPU/GPU设备、随机初始化、模型与优化器更新以及完整恢复点哈希和安全加载均通过检查；110份源文件和21份配置保持冻结。观测窗口CPU最高78°C，无降载或失败。验收记录见[deployment_verification.json](runs/three_seed_sc_20260908/deployment_verification.json)；这是启动检查，完整训练和统一评估仍在后续队列中。

## 启动、恢复和查看进度

首次正式启动（自动训练、评估、汇总）：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/run_three_seeds.sh run --output /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908
```

中断后的恢复命令：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/run_three_seeds.sh run --resume --output /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908
```

查看状态：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/run_three_seeds.sh status --output /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908
```

运行中Ctrl+C或向监督器发送SIGTERM，会请求正在运行的训练任务完成当前PPO更新并保存完整状态，再退出。恢复时跳过已完成任务，未完成任务从同一实验的完整恢复点继续；评估在完整600时隙回合边界保存，可跳过已提交回合。

进程意外崩溃时保留最近恢复点。每次调用中不自动重复失败任务，其他健康任务可继续；需要再次使用`--resume`。若尚无有效恢复点，脚本明确报错，不把旧权重热启动当作精确续训。

## CPU/GPU怎么分工

同时最多 **6个CPU任务 + 1个GPU任务**。CPU池使用16、17、19、11、13、9；GPU任务由CPU18负责控制，计算设备为cuda:0。每个进程的Torch、OMP、OpenBLAS、MKL线程均为1。

| 方法 | 三个种子 | 环境、采样和模型更新 |
| --- | --- | --- |
| IC_HAPPO | 85、218、966 | CPU |
| IC_MAPPO | 85、218、966 | CPU |
| HAPPO_hidden_instruction | 85、218、966 | CPU |
| MAPPO_hidden_instruction | 85、218、966 | CPU |
| HAPPO_fixed_mode_rule | 85、218、966 | CPU |
| HAPPO_no_task_aux_reward | 85、218、966 | CPU |
| HAPPO_equal_resources | 85、218、966 | GPU |

某个CPU任务完成后，该位置立即接CPU队列中的下一项，可先开始下一种子，不等待同一种子的所有方法结束。三个GPU任务依次运行。每个任务有独立环境、网络、优化器和随机数状态；不共享梯度，不在运行中迁移设备。

每个任务内部仍为10个逻辑环境，每环境收集400步后更新，一次4,000步；共2,500次更新，物理回合为600时隙。Actor学习率1e-4、Critic学习率4e-4、PPO/critic epoch均5、网络256×256；算法、奖励、观测、最终SC表及其他参数沿用冻结版本。

原单种子模型和短测模型都不加载。两项CC待表口径对齐和后端支持后另补，本次不计入21项。

## 恢复与温控

**2026-09-09更新：** 已启用[秒级温控守护与自动冷却续训](THERMAL_MONITORING.md)。CPU达到85°C持续5秒即请求保存暂停；95°C立即保护；全部工作进程退出后至少冷却3分钟，并低于75°C连续1分钟，再核验完整恢复点自动续训。以下为冻结监督器自身的底层保护规则，新增守护在外层补充暂停、冷却和自动恢复。

- 每50次更新（200,000步）保存一次完整恢复点，启动、正常结束和收到退出请求时也保存。每任务保留当前及前一份已提交恢复点，避免无限占用磁盘。
- 恢复点包含actor、critic、优化器、ValueNorm、采样缓存、环境张量、当前回合/时隙、外生随机输入和Python/NumPy/Torch/CUDA随机数状态。
- 代码、配置、种子、预算、设备或Python/NumPy/PyTorch版本不匹配时拒绝恢复；要求同一软件环境和设备。修改CPU核号不会更改随机数，但本入口固定为已经测速的分配。
- 采用完整文件写入后再提交索引；当前恢复点哈希不符时可回退上一份，并记录原因。恢复时归档中断日志，只保留恢复点已经提交的更新行，防止重复计数。
- 温度≥85°C持续5秒后，CPU任务共享4个E核；持续高温进一步缩至2核。已有任务保留状态并继续计算，降低后续CPU任务入队数量。温度低于75°C持续60秒后逐步恢复。
- 95°C或温度传感器失效会触发有序退出，保存可提交状态；不使用SIGSTOP做日常温控。600秒限制针对“没有进度”，并不是整项训练时长限制。
- 实验资源锁、每任务锁和PID启动身份检查拦截重复队列及遗留进程。脚本不修改GPU功率限制、驱动或BIOS设置。

## 自动评估和汇总

21个模型全部训练完成后，使用冻结CPU参考环境评估，最多2个评估进程，CPU11、13。每个模型覆盖原13场景、20个配对评估种子和600时隙完整回合。四种规则只评估一次，共25项评估、3,900,000时隙。

统一采用最终10M预算检查点，不按评估成绩选择模型。报告使用公共外部指标，不直接用不同奖励定义下的训练曲线排名。输出包含每种子的结果、三种子均值和样本标准差、各场景结果、相对IC_HAPPO的配对差值及完整轨迹。评估回合不冒充额外训练种子，三种子结果不自动构成显著性或收敛保证。

最终文件：

- `report/comparison.md`：可直接阅读的三种子对比表。
- `report/comparison.csv`和`report/comparison.json`：均值、样本标准差及分场景数据。
- `report/per_seed_results.json`：每个训练种子的结果，四种规则单独记录。
- `jobs/seed_<种子>/<方法>/`：训练指标、模型、完整恢复点、恢复日志。
- `evaluation/`：各模型及规则的完整600时隙轨迹和回合汇总。
- `status.json`、`resources.jsonl`、`logs/`：整体进度、温控记录和运行日志。

完成后的只读核查命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/king/miniconda3/envs/harl_sionna/bin/python /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/verify_three_seed_run.py --run /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908
```

该检查器不训练、不重新生成报告；核对冻结输入、完整预算、恢复点与导出权重一致、所有轨迹哈希及物理约束，并从CSV独立重算共同奖励、AoI的种子均值和样本标准差。

## 时间预案

按此前7方法整批实测及本次两进程评估短测，**正式训练预留9–12小时，评估和核查另留4–5小时，合计约13–17小时**。这是工程外推，不包含等待原队列、CC、故障重训或论文写作；实际温控和长期吞吐可能延长时间。

## 新建其他运行

已准备的正式目录不要再次执行prepare。若以后要新建实验，可以显式指定种子，也可省略`--seeds`，让脚本一次随机抽取三个不同种子并写入新清单：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/run_three_seeds.sh prepare --seeds 85 218 966 --output /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/another_three_seed_run
```

`--smoke`只用于开发验证：每项8,000步、2个评估场景、1个评估种子，结果明确标记为smoke。允许旧队列仍运行的`--allow-reference-active`只对smoke有效，不能用于正式预算。

本次验证证据位于`results/three_seed_smoke_20260908/`和`results/three_seed_smoke_v2_20260908/`。前者验证CPU真实进程退出续训，后者验证GPU真实进程退出续训；种子保持85、218、966。训练恢复单元检查同时覆盖下一回合重置后的状态，软件版本检查和进程重复启动保护另外测试。
