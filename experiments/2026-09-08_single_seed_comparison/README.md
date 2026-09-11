# 单种子训练：已退役的历史入口

**2026-09-08 20:52 JST更新：本目录的旧训练已全部停止，49个相关进程均已结束。** 原训练和两组资源测速输出已移至 `../_archive/2026-09-08_before_three_seed_formal/`。当前入口为[CPU / GPU三种子正式训练](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-08_gpu_environment/THREE_SEED_TRAINING.md)，种子85、218、966，共21项任务，从零开始。下文保留历史配置和判断，不代表当前运行状态；不要再执行下文的旧启动命令。

2026-09-08：用户要求结束旧训练、排除旧结果，重新随机初始化。旧训练的60个进程已结束，原A/B和旧运行目录移至 ../_archive/2026-09-08_before_clean_restart/。新实验不加载旧权重，不合并旧进度或结果。

用户确认本轮先训练7个SC条件；CC的两项待质量—载荷表口径对齐后另行补训。本轮尚未覆盖CC。

| 方法 | 作用 |
| --- | --- |
| IC_HAPPO | 显式任务指令，HAPPO |
| IC_MAPPO | 显式任务指令，MAPPO |
| HAPPO_hidden_instruction | 隐藏HAPPO actor显式指令 |
| MAPPO_hidden_instruction | 隐藏MAPPO actor显式指令 |
| HAPPO_fixed_mode_rule | 固定选模规则，训练其他控制 |
| HAPPO_equal_resources | 等分SUT资源，训练UAV控制 |
| HAPPO_no_task_aux_reward | 去任务附加奖惩 |

每方法seed=1、10,000,000步；10个逻辑环境×400步/更新，共2,500次更新。七方法共70,000,000步。Actor学习率1e-4、Critic学习率4e-4；PPO/critic epoch均5；两层256单元。继续使用用户最终SC表，SHA-256=a6f721686b28d6c28ccf811ab8b63e80ae8c135a08b9d7e1b4bbfe7ca13d91cc。四种规则基线直接评估，不计入训练数。

## 当前资源方案

4组并行、6核CPU（0,2,8,10,12,14；2个P核和4个E核）；CPU15暂时排除。CPU运行环境、推断和GAE，RTX4070TiSUPER运行PPO更新。其余3组有空位后从零初始化并自动补上。机器共有20核、约64GB内存和16GB显存；本次持续负载受CPU温度限制。

| 并行数 / CPU数 | 同样7×16,000步耗时 | 吞吐 | 采样最高CPU温度 | 70M步线性估算 |
| --- | ---: | ---: | ---: | ---: |
| 2 / 4 | 154.04秒 | 727步/秒 | 72°C | 26.74小时 |
| 4 / 8 | 78.99秒 | 1,418步/秒 | 82°C | 13.71小时 |
| 6 / 12 | 61.51秒 | 1,821步/秒 | 86°C | 10.68小时 |

以上三组短测均完成7个方法各16,000步，同一方法跨资源配置的最终参数和检查点完全一致。证据见 runs/restart_resource_plan.json。6组/12核和4组/8核在后续持续运行中均触及85°C控制阈值，分别采样最高87°C和86°C并自动降载，因此正式运行改为4组/6核；原10.68小时与13.71小时估算已被取代。

18:03 JST核查：6核配置持续采样约192秒，4组总吞吐约1,190步/秒，最高78°C，无降载。按当前4组完成后再运行后3组、并保守沿用当前最慢单方法吞吐估算，训练总工期约19小时，训练预算16–24小时。统一评估另预留2–3小时，训练加评估按20–27小时安排，不含CC与更多训练种子。详见 runs/seed1_clean_restart_20260908/deployment_verification.json。

以上是早期持续吞吐的工程外推，不是置信区间或长期稳定保证；持续降载或故障会延长工期。评估预算依据此前85,800时隙约327秒的短测线性外推。

## 正式新目录与命令

输出为 runs/seed1_clean_restart_20260908。准备和启动仅执行一次；guarded_queue.py拒绝已有attempt，且检查model_dir为null。已有运行不要重复启动。

```bash
/home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_single_seed_comparison/run.py prepare --seed 1 --steps 10000000 --device hybrid --jobs 4 --output experiments/2026-09-08_single_seed_comparison/runs/seed1_clean_restart_20260908
/home/king/miniconda3/envs/harl_sionna/bin/python -u experiments/2026-09-08_single_seed_comparison/guarded_queue.py --output experiments/2026-09-08_single_seed_comparison/runs/seed1_clean_restart_20260908 --jobs 4 --cpu-set 0,2,8,10,12,14
```

总状态为输出目录的status.json，各方法为jobs/<method>/status.json，调度和温度日志为execution/recovery_*/。训练完成后自动评估四种规则和七个学习方法，使用完整预算的最终检查点，不按评估成绩挑选模型。

## 保护机制

环境使用spawn，worker死亡会报错，响应等待和退出清理有时限。完整预检24项通过，执行监督/温度控制10项测试通过。

CPU持续达到85°C时，先缩减至4核、后续并发上限改为2；已有任务保留状态并共享较小CPU池。如仍过热，则暂用1个E核（CPU8）继续运行，后续并发上限为1；温度不高于70°C持续15秒后恢复4核。95°C立即进入最低CPU配置。85°C为本项目保守控制阈值。降载事件保留记录；耗时可能超过以上估算。

排队和自动温控不再使用SIGSTOP：独立测试3次均复现暂停时间超过IPC响应时限后误超时。两个早期暂停的短任务已结束并归档，四个健康训练保留本轮新进度。训练算法、冻结环境和表未改；被排除的短任务不参与比较。

执行控制验证命令：

```bash
/home/king/miniconda3/envs/harl_sionna/bin/python -m unittest discover -s experiments/2026-09-08_single_seed_comparison/recovery_tests -v
```

每方法最多一次尝试；失败明确列出，其他健康方法可继续完成。原生崩溃根因尚未完全确诊，不能把临时排除CPU15表述为硬件已经修好。
