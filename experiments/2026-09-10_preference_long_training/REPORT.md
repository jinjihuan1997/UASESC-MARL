三种子长训练与评估完成。两方法各3个模型，每模型从头训练1000万步；固定末次模型评估。

本任务共享质量底线、动作范围与执行规则，仅改变奖励偏好。它与原来的指令硬门槛任务不同，不能混为同一设置的排名。

| 方法 | 平均奖励 | 平均 AoI | 交付预测 PSNR | 更新/槽 |
|---|---:|---:|---:|---:|
| IC_HAPPO | -0.001694 | 2.9583 | 27.4836 | 13.1823 |
| HAPPO_hidden_instruction | -0.019047 | 3.0144 | 28.0414 | 12.9445 |
| R_single | -0.015428 | 3.3725 | 29.9361 | 11.5033 |
| R_instruction | -0.000009 | 3.0646 | 27.8720 | 12.7329 |
| R_myopic | 0.003259 | 3.0738 | 28.7091 | 12.6496 |

| 显式输入 RL 减对照 | 三种子奖励平均差 | 样本标准差 | 正差训练种子 |
|---|---:|---:|---:|
| HAPPO_hidden_instruction | +0.017353 | 0.003284 | 3/3 |
| R_single | +0.013734 | 0.001348 | 3/3 |
| R_instruction | -0.001684 | 0.001348 | 0/3 |
| R_myopic | -0.004952 | 0.001348 | 0/3 |

| 方法 | 固定偏好 | 平均 AoI | 交付预测 PSNR |
|---|---|---:|---:|
| IC_HAPPO | 均衡 | 2.9805 | 28.1612 |
| IC_HAPPO | AoI | 2.5029 | 25.3581 |
| IC_HAPPO | 质量 | 3.4857 | 29.9081 |
| HAPPO_hidden_instruction | 均衡 | 3.0144 | 28.0414 |
| HAPPO_hidden_instruction | AoI | 3.0144 | 28.0414 |
| HAPPO_hidden_instruction | 质量 | 3.0144 | 28.0414 |
| R_single | 均衡 | 3.3725 | 29.9361 |
| R_single | AoI | 3.3725 | 29.9361 |
| R_single | 质量 | 3.3725 | 29.9361 |
| R_instruction | 均衡 | 3.3725 | 29.9361 |
| R_instruction | AoI | 2.5003 | 24.9632 |
| R_instruction | 质量 | 3.3725 | 29.9361 |
| R_myopic | 均衡 | 3.3296 | 29.9258 |
| R_myopic | AoI | 2.5991 | 26.9029 |
| R_myopic | 质量 | 3.3297 | 29.9258 |

三训练种子是独立训练的重复；规则只有一套评估，20个外部评估种子不当作20个训练种子。不据描述性均值自动宣称显著性、创新性或跨设置泛化。

训练布局：cpu6。同种子内的显式/隐藏模型使用相同训练后端。

核验完成：2340回合，1404000槽；全部逐槽统计重新聚合、外生条件配对、输入哈希通过。

完整数值：[results.json](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_preference_long_training/report/results.json)；核验：[audit.json](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_preference_long_training/report/audit.json)。
