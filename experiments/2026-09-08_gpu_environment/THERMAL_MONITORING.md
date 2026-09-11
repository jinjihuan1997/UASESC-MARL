# 正式训练的温控暂停与自动续训

用户于2026-09-09授权持续监视，过热时暂停、温度正常后自动续训。本地守护于 **2026-09-09 01:35 JST** 启动，首次守护PID为3423062，附着于当前三种子正式实验。

正式目录：`/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908`。

## 自动规则

- 每1秒读取主机CPU最高温度。
- CPU达到85°C并持续5秒：向经过PID、UID和启动时间核实的本实验监督器发送SIGTERM，监督器等待当前更新结束、保存完整状态后退出。
- CPU达到95°C：立即请求同样的有序退出；冻结训练监督器原有95°C保护继续有效。
- 全部训练/评估工作进程退出后，至少冷却180秒；同时要求CPU低于75°C连续60秒。任何回温或温度传感器不可读都会重新计算连续低温时长。
- 冷却条件满足后，核验冻结源文件、配置、预算、每个已开始训练任务的当前完整恢复点哈希及安全加载。恢复点必须与已提交步数一致；完成任务还要验证导出模型。
- 核验通过且温度仍低于75°C，使用同一冻结监督器执行 `run --resume`。沿用6 CPU + 1 GPU、种子85/218/966、每项10M步；不从零重训。
- 训练完成后的统一评估也受温控守护保护；评估从已提交的完整回合恢复。全部训练、评估及最终核查完成后，本地守护退出。
- 非温度错误、恢复点损坏、进程身份不符或有序退出超时进入待处理状态；不会盲目重启。明确的人工暂停予以保留。

这是执行控制的独立版本，位于正式目录的 `thermal_guard/source/`；训练 `source/`、21份配置及原 `manifest.json` 没有改动。旧训练说明中的“温度退出后需要手动恢复”已由本守护补足自动冷却续训。

## 运行记录

| 文件 | 用途 |
| --- | --- |
| `thermal_guard/status.json` | 守护最新状态、温度、冷却计时及总训练进度 |
| `thermal_guard/process.json` | 守护PID、UID、start_ticks身份 |
| `thermal_guard/events.jsonl` | 暂停、保存退出、冷却、恢复及异常事件 |
| `thermal_guard/temperatures.jsonl` | 每5秒记录温度；控制循环仍为每1秒 |
| `thermal_guard/manifest.json` | 守护版本、代码哈希、阈值及验证记录 |
| `thermal_guard/control.json` | 可恢复的控制状态及失败锁定 |
| `thermal_guard/guard.log` | 守护标准输出和错误日志 |
| `current_execution.json` | 当前训练监督器的身份与本次运行日志路径 |
| `execution/thermal_resume_*/` | 每次自动续训的恢复点核验、监督器日志和启动身份 |

不要仅凭 `launch_record.json` 的首次监督器PID判断当前训练。温控自动续训后PID会改变，应使用 `current_execution.json` 和工作进程的 `process.json`，对照宿主机 `/proc` 的UID与start_ticks。

## 每5分钟的任务巡检

巡检用于核对本地守护是否存活、进度是否增长、温控停启是否成功，以及发现非温度异常；快速温控由上述本地循环负责。状态正常且无可操作变化时保持安静，温控事件合并报告，只在有意义变化、失败、需要处理或全部完成时通知。

巡检读取 `thermal_guard/status.json`、`thermal_guard/events.jsonl`、正式 `status.json` 和 `current_execution.json`，并验证对应主机进程身份。若守护的 `process.json` 身份仍存活，禁止重复启动。`disabled` 表示明确停用；`control.json` 的 `attention_required` 表示待核查错误，均不可当作普通进程掉线自动重启。

若守护意外退出、没有人工停用或失败锁定，且实验未完成，可在主机权限下运行下面的幂等启动命令。它验证守护源文件哈希、检查已有进程并通过锁防止重复实例：

```bash
/usr/bin/python3 /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908/thermal_guard/source/thermal_guard.py start --run /home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/runs/three_seed_sc_20260908
```

恢复守护后检查其状态更新时间及原有任务进度，不另外启动第二个训练监督器。冻结训练输入变化、非温度错误或恢复点损坏需要先核查具体原因。用户要求暂停时应同时尊重人工暂停和守护停用意图。

任务完成后暂停对应的巡检自动化，不归档当前任务。[官方定时任务说明](https://learn.chatgpt.com/docs/automations?surface=app)说明，本地项目巡检需要电脑开机且桌面应用保持运行；本地秒级守护是独立后台进程，电脑关机或重启会结束该进程。

## 验证范围

8项控制测试通过，覆盖持续高温、瞬时紧急温度、冷却的最小时长与连续低温条件、传感器缺失、PID复用、重复守护锁、人工暂停、非温度错误及错误锁定。独立合成工作进程实际接收SIGTERM、保存计数、退出，并由守护自动重启且保留计数；该测试缩短冷却计时，未制造真实高温或中断正式训练。

自动续训核验器还安全读取了既有完整流程短测的21份模型和完整恢复点。当前正式训练的110份源文件、21份配置哈希仍与原冻结清单一致。以上验证不声称已确定此前CPU温度突升的硬件或系统原因。
