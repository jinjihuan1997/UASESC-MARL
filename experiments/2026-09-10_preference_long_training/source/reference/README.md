# 从零重训：当前入口

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

6组并行、12核CPU（0,1,2,3,4,5,8,9,10,11,16,17）；CPU15暂时排除。CPU运行环境、推断和GAE，RTX4070TiSUPER运行PPO更新。第七组有空位后补上。

| 并行数 / CPU数 | 同样7×16,000步耗时 | 吞吐 | 采样最高CPU温度 | 70M步线性估算 |
| --- | ---: | ---: | ---: | ---: |
| 2 / 4 | 154.04秒 | 727步/秒 | 72°C | 26.74小时 |
| 4 / 8 | 78.99秒 | 1,418步/秒 | 82°C | 13.71小时 |
| 6 / 12 | 61.51秒 | 1,821步/秒 | 86°C | 10.68小时 |

以上三组完整训练的最终参数和检查点完全一致。证据见 runs/restart_resource_plan.json。训练先按9–13小时安排，不含CC、更多种子、评估和持续降载；这是工程预算，不是置信区间或长期稳定保证。统一评估另预留2–3小时，依据此前85,800时隙约327秒的短测线性外推。

## 正式新目录与命令

输出为 runs/seed1_clean_restart_20260908。准备和启动仅执行一次；guarded_queue.py拒绝已有attempt，且检查model_dir为null。已有运行不要重复启动。

```bash
/home/king/miniconda3/envs/harl_sionna/bin/python experiments/2026-09-08_single_seed_comparison/run.py prepare --seed 1 --steps 10000000 --device hybrid --jobs 6 --output experiments/2026-09-08_single_seed_comparison/runs/seed1_clean_restart_20260908
/home/king/miniconda3/envs/harl_sionna/bin/python -u experiments/2026-09-08_single_seed_comparison/guarded_queue.py --output experiments/2026-09-08_single_seed_comparison/runs/seed1_clean_restart_20260908 --jobs 6 --cpu-set 0,1,2,3,4,5,8,9,10,11,16,17
```

总状态为输出目录的status.json，各方法为jobs/<method>/status.json，调度和温度日志为execution/recovery_*/。训练完成后自动评估四种规则和七个学习方法，使用完整预算的最终检查点，不按评估成绩挑选模型。

## 保护机制

环境使用spawn，worker死亡会报错，响应等待和退出清理有时限。完整预检24项通过，执行监督/温度控制9项测试通过。

CPU持续达到85°C时，先缩减至4核、后续并发上限改为2；已有任务保留状态并共享较小CPU池。如仍过热则暂停trainer/worker，冷却至不高于70°C持续15秒后恢复，并为恢复后的进度设置宽限期。85°C为本项目保守控制阈值。降载事件保留记录；耗时可能超过以上估算。

每方法最多一次尝试；失败明确列出，其他健康方法可继续完成。原生崩溃根因尚未完全确诊，不能把临时排除CPU15表述为硬件已经修好。
