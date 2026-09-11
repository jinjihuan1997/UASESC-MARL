# 新参数的三种子长期训练

本目录使用已经验证有指令区分的 instruction_priority_v1 参数，保留原来的 actor 可行性筛选和公共约束调度。正式目录为 `runs/three_seed_sc_20260909`。准备脚本不会自动启动正式训练；实际进度以该目录的 `status.json` 为准。

2026-09-09 准备验收通过：正式21组配置已冻结。两轮完整小预算流程已完成；最终分配下21组训练、25项评估均通过，质量/预算/缓存检查和外生轨迹配对通过。实测人工暂停时5组从完整恢复点续训，另1组已完成并正确跳过；重复启动被拦截。

正式训练已按用户指令于2026-09-09 12:07:54 UTC（21:07:54 JST）启动，初始supervisor PID为3921833。12:12:06 UTC核查时8组均在推进，累计2,496,000/210,000,000步，无失败任务；该数字是历史启动快照，最新进度请执行下面的status命令。`readiness.json`保留启动前的验收状态，不代表当前运行状态。

## 执行命令

后台启动正式训练，退出当前终端后仍继续运行：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_instruction_long_training/run_long_training.sh start
```

查看进度：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_instruction_long_training/run_long_training.sh status
```

请求保存并暂停：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_instruction_long_training/run_long_training.sh pause
```

待状态成为 `paused` 后，从完整恢复点继续：

```bash
bash /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_instruction_long_training/run_long_training.sh start --resume
```

需要前台运行时，把 `start` 换成 `run`；Ctrl+C 会请求保存暂停。已有运行中的队列会拒绝重复启动。GPU入口需在正常主机环境执行；受限沙箱可能看不到实际GPU。

## 本次训练内容

|方法|种子|计算设备|
|---|---|---|
|IC_HAPPO|85、218、966|CPU|
|HAPPO_hidden_instruction|85、218、966|CPU|
|IC_MAPPO|85、218、966|CPU|
|MAPPO_hidden_instruction|85、218、966|CPU|
|HAPPO_no_task_aux_reward|85、218、966|CPU|
|HAPPO_fixed_mode_rule|85、218、966|GPU|
|HAPPO_equal_resources|85、218、966|GPU|

每组1000万环境步，总计21组、2.1亿步。每回合600时隙；每次PPO更新收集10环境×400步，共2500次更新。全部从随机初始化开始，不加载旧正式模型或1M短训模型。

最多6个CPU训练进程和2个GPU训练进程并行，同一个GPU可同时运行两个方法或两个种子。完整方法及其显式/隐藏对照使用相同CPU后端；两种控制消融进入GPU队列。CPU/GPU上的环境与策略都由已验证的张量实现计算，运行中不迁移设备。每个进程1线程，完成一组就补下一组。

eta=.6，AoI奖励参考10，状态上限600，三种AoI软目标6/4/10秒；指令条件权重和画质门限继承已验证配置。最终平均表不变。CC不在此队列，继续等待质量载荷表口径对齐。

## 自动评估

训练完成后自动评估21个末次模型和4个非学习规则：R_fixed、G_local_greedy、Random、RoundRobin。每项13种固定/切换场景×20个新评估种子×600槽，共6500回合。规则共用完整系统配置，不继承消融方法的固定模式或均分资源开关。

比较相同外部共同奖励、AoI、预测PSNR、交付数、开销和违规；保存提议模式、实际模式、选择的DS以及资源份额。整体差值统一为IC_HAPPO减对照，正数表示IC_HAPPO较好。隐藏显式指令的对照仍接收间接任务信息，名称不代表整个系统没有指令。

输出包括 `report/comparison.md`、`comparison.json`、`comparison.csv`、`per_seed_results.json`、`switch_response.json`。后者包含六种第300槽切换场景的前后均值。评估回合不是额外训练种子，不自动宣称显著性或稳定收敛。

## 恢复与温控

每25次更新保存完整状态，包括网络、优化器、归一化器、采样缓存、环境和随机数。收到暂停请求时在更新边界存盘。评估在完整600槽回合边界提交；中断后跳过已完成回合。

CPU达到85°C持续5秒、95°C立即，或GPU达到83°C，自动请求当前工作进程保存并暂停。全部退出后至少冷却3分钟，CPU/GPU都低于75°C连续1分钟才继续。人工暂停、程序错误和传感器故障不会触发自动重试。本机制只在训练队列运行期间有效，不启动应用的持续监视任务。

脚本冻结配置、源码、平均表和设备，恢复时核验；不同预算或源文件不能混用。程序错误需要查看 `logs/supervisor.log` 和对应工作日志；修正后使用新版本或明确恢复，不能用旧权重冒充精确断点。

## 验证与时间

`runs/smoke_40k` 是首轮40k流程验证，包含实际CPU/GPU暂停和恢复。`runs/smoke_40k_v2` 使用调整后的资源分配，仍是21个小预算任务及4个规则的流程验证，不是论文实验。资源选择只依据耗时、温度和队列负载，不依据这些短训的性能排名。

最终准备状态、检查结果及耗时外推记录在本目录 `readiness.json`。长训练工期需要计入温控与系统负载；短测外推不能保证最终完成时间。正式训练是否启动，以正式目录状态和进程为准。

最终分配的40k短测外推：训练约6.5小时，评估约2.1小时，建议预留9–12小时。该短测记录CPU最高76°C、GPU最高43°C，没有触发热暂停；长时间运行仍可能因温度或其他负载而延长。首次4 CPU+2 GPU分配在后段受GPU队列限制，最终6 CPU+2 GPU把完整模型放在更快的CPU后端，并用GPU并行承担两个消融方法；两个版本的原始耗时记录均保留。
