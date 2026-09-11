# 参考值8：新训练效果解释

下列分数全部采用参考值8，均为同一组新测试条件。旧模型为参考10训练，新模型为参考8训练，各100万步、3个训练种子。

| 方法 | 旧模型分数×100 | 新模型分数×100 | 新减旧×100 | 正差种子 |
|---|---:|---:|---:|---:|
| IC_HAPPO | -4.3987 | -4.2645 | +0.1342 | 2/3 |
| HAPPO_hidden_instruction | -4.7442 | -4.7878 | -0.0436 | 1/3 |

## 改善或损失来自哪一项

| 方法 | 质量收益变化×100 | AoI节省变化×100 | 资源节省变化×100 |
|---|---:|---:|---:|
| IC_HAPPO | -0.2294 | +0.3564 | +0.0072 |
| HAPPO_hidden_instruction | -0.4978 | +0.4071 | +0.0471 |

## 指令区分度

| 方法 | 训练参考 | 质量指令减AoI指令：PSNR差 | 对应AoI差 |
|---|---|---:|---:|
| IC_HAPPO | ref10 | +0.8735 dB | +0.1048 |
| IC_HAPPO | ref8 | +0.5142 dB | +0.0101 |
| HAPPO_hidden_instruction | ref10 | +0.0000 dB | +0.0000 |
| HAPPO_hidden_instruction | ref8 | +0.0000 dB | +0.0000 |

更大的指令响应不自动等于更高绝对性能。规则也按新目标重新校准；单一规则在新目标下改为m0_urgency。所有结论限于当前三种子100万步，不能替代长期训练或统计显著性。

[正式完整表](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_aoi_ref8/REPORT.md)；[独立审计](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_aoi_ref8/report/audit.json)；[完整解释数据](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-10_aoi_ref8/analysis_results.json)。
