偏好任务短训练与评估已完成。两种方法均从头训练 100 万步，训练种子 85，使用最后一次模型。结果只用于下一阶段决策，不宣称跨训练种子的稳定优势。

本轮与原任务不同：三个指令共享质量底线、动作范围与执行规则，只改变奖励偏好。所有方法配对相同外部环境；知道指令的规则也参与对比。

| 方法 | 平均奖励 | 平均 AoI | 交付 PSNR | 更新/槽 |
|---|---:|---:|---:|---:|
| IC_HAPPO | -0.022709 | 2.7859 | 26.8055 | 13.7983 |
| HAPPO_hidden_instruction | -0.028061 | 3.4271 | 29.4770 | 11.4556 |
| R_single | -0.022069 | 3.4356 | 29.8120 | 11.3320 |
| R_instruction | -0.005276 | 3.1069 | 27.7241 | 12.6169 |
| R_myopic | -0.002405 | 3.1267 | 28.6367 | 12.4721 |

R_single 为校准集选出的单一规则；R_instruction 按真实指令选择校准规则；R_myopic 知道真实偏好，在 9 个当前候选动作中选下一槽奖励最高者。它们不读取未来信道。

| IC_HAPPO 减对照 | 平均奖励差 | 正差评估种子/20 |
|---|---:|---:|
| HAPPO_hidden_instruction | +0.005352 | 14/20 |
| R_single | -0.000640 | 8/20 |
| R_instruction | -0.017433 | 0/20 |
| R_myopic | -0.020304 | 0/20 |

这些是同一训练种子下的外部环境配对结果，20 个评估种子不能当作 20 个独立训练种子。共同奖励只能在相同任务条件下比较。

| 方法 | 真实偏好 | 平均奖励 | AoI | 交付 PSNR |
|---|---|---:|---:|---:|
| IC_HAPPO | 均衡 | -0.000655 | 2.7876 | 26.8274 |
| IC_HAPPO | AoI | -0.139950 | 2.7760 | 26.7181 |
| IC_HAPPO | 质量 | 0.093140 | 2.7958 | 26.8858 |
| HAPPO_hidden_instruction | 均衡 | -0.000827 | 3.4244 | 29.4683 |
| HAPPO_hidden_instruction | AoI | -0.170423 | 3.4305 | 29.4866 |
| HAPPO_hidden_instruction | 质量 | 0.112046 | 3.4264 | 29.4761 |
| R_single | 均衡 | 0.005194 | 3.4344 | 29.8116 |
| R_single | AoI | -0.164698 | 3.4371 | 29.8126 |
| R_single | 质量 | 0.118329 | 3.4352 | 29.8119 |
| R_instruction | 均衡 | 0.005236 | 3.4339 | 29.8116 |
| R_instruction | AoI | -0.117129 | 2.5059 | 24.8236 |
| R_instruction | 质量 | 0.118353 | 3.4346 | 29.8119 |
| R_myopic | 均衡 | 0.009396 | 3.3786 | 29.7966 |
| R_myopic | AoI | -0.115998 | 2.6640 | 26.9254 |
| R_myopic | 质量 | 0.121744 | 3.3788 | 29.7983 |

已完成 1300 个评估回合、78 万槽，逐槽核算奖励及资源/缓存/质量约束，所有方法外生条件配对。平均表与原正式版本未修改。

证据：[完整结果](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_preference_only/pilot_results.json)、[核验](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_preference_only/pilot_audit.json)、[规则预检](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_preference_only/GATE_REPORT.md)、[协议](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_preference_only/PROTOCOL.md)。
