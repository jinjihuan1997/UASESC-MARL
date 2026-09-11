# UASESC-MARL

指令条件下的多智能体强化学习：SUT 分配回传资源，三个 UAV 根据实际预算从 16 种语义模式中分别选择一种，联合权衡交付质量、AoI 和资源消耗。

本次同步对应 2026-09-11 的完整本地项目快照，包含实现、配置、实验记录、数值结果、逐时隙评估轨迹和小型策略权重。大型模型、完整训练状态断点、原始数据集、虚拟环境及本地缓存不包含在仓库中。

## 最新结果：交替训练短试

最新实验：[2026-09-11_alternating_training](experiments/2026-09-11_alternating_training/)。三个训练种子在实验前随机确定，分别为 `104948945`、`111868397`、`160441552`；每个种子运行联合训练与交替训练，各 100 万环境步，共 600 万步。

交替方案为：前 40 万步在不同资源预算下训练 UAV 模式策略，接着 20 万步训练 SUT，最后 40 万步交替更新两侧。两臂使用相同观测、动作空间、平均质量载荷表、奖励与优化器设置。

分数为原始奖励乘 100，越大越好。综合分数覆盖 13 个指令场景；专项列为全程固定相应指令。每个模型使用 20 个新验证环境种子、每回合 600 个时隙。表中 RL 为全部三个训练种子的均值。

| 方法 | 综合分数 | 均衡指令 | AoI 指令 | 质量指令 |
|---|---:|---:|---:|---:|
| 联合训练 | -3.9002 | -2.5525 | -15.0641 | 8.0928 |
| 交替训练 | -4.1695 | -2.3411 | -15.1259 | 6.9626 |
| 均分资源＋固定模式 | -5.0321 | -3.1148 | -15.0831 | 4.8640 |
| 紧急度分配＋固定模式 | -4.9755 | -3.0633 | -14.9955 | 4.8916 |
| 均分资源＋指令选模 | -3.2321 | -2.4705 | -15.0831 | 10.3171 |
| 按指令切换简单规则 | -3.2044 | -2.4705 | -14.9955 | 10.3171 |

这次交替训练未改善综合表现，三个种子均落后于同预算联合训练。均衡专项改善不足以抵消质量专项退步。训练和全部检查点/基线共 8840 回合评估已完成，物理、奖励及外生环境配对核验通过；预留最终测试集尚未使用。这是每模型 100 万步的新验证短试，不与历史 1200 万步模型混作同预算比较。

- [完整报告](experiments/2026-09-11_alternating_training/REPORT.md)
- [机器可读结果](experiments/2026-09-11_alternating_training/report/results.json)
- [核验结果](experiments/2026-09-11_alternating_training/report/audit.json)
- [训练协议](experiments/2026-09-11_alternating_training/PROTOCOL.md)
- [温控后的执行资源调整](experiments/2026-09-11_alternating_training/EXECUTION_AMENDMENT.md)
- [代码、配置与输入哈希](experiments/2026-09-11_alternating_training/manifest.json)

## 项目目录

保留当前工作区的目录层级，包括 `HARL/HARL`，以保持实验之间的相对位置。

| 目录 | 内容 |
|---|---|
| `experiments/` | 新实现的冻结运行环境、CPU/CUDA 训练、阶段训练与选择器、SC/CC 质量表诊断和重建、训练/评估脚本、配置、报告、日志及数值轨迹；包括历史实验和失败诊断 |
| `HARL/HARL/` | 原 HARL 项目及本地 UAV-ESCS、HAPPO/MAPPO、混合动作头、配置、测试与实验脚本 |
| `CRL-SemCom-VidCI/` | 语义通信/视频压缩成像代码、模式质量载荷测量及数据准备脚本 |
| `Manuscript/` | 原稿 `main.tex`、新增修订稿、参考文献及图表；稿件内容不替代最新实验报告 |
| `Promptus/` | 保留的原始组件；当前固定平均表 RL 训练不调用其大型视频模型 |

当前在线训练使用固定平均质量载荷表。报告中的交付 PSNR 是该表对应的预测质量，不是每个时隙实际解码视频的测量值。历史协议、旧参数与新参数的结果保留各自实验目录，不跨协议拼接为同一对比。

## 核验和复现入口

运行依赖见 [HARL requirements](HARL/HARL/requirements.txt) 和 [CRL requirements](CRL-SemCom-VidCI/requirements.txt)。在具备项目 Python 依赖的环境中，从仓库根目录运行以下命令可核验最新冻结源码、表及配置：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import sys
from pathlib import Path
run = Path('experiments/2026-09-11_alternating_training').resolve()
sys.path.insert(0, str(run))
from helpers import verify
verify()
print('Frozen input hashes verified')
PY
```

最新报告可以从已同步的完整评估轨迹重新计算；这不需要下载原视频数据集或大型模型：

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/2026-09-11_alternating_training/aggregate.py
```

历史配置、日志和清单保留当时机器的绝对路径，以维持原始文件哈希。原 `run_training.sh` 及 `execution_override.py` 用于当时机器上的运行/断点恢复；完整训练状态断点未上传，因此不应在新机器上直接对历史完成目录执行恢复。

在新机器上从头复现单个模型时，用现有 `make_config` 生成指向当前检出目录的配置，并使用一个新的输出目录。例如：

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json, sys
from pathlib import Path
run = Path('experiments/2026-09-11_alternating_training').resolve()
sys.path.insert(0, str(run))
from prepare import make_config
out = Path('reproduction/seed_104948945/alternating').resolve()
out.mkdir(parents=True, exist_ok=False)
(out / 'config.json').write_text(json.dumps(make_config(104948945, 'alternating'), indent=2))
PY

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/2026-09-11_alternating_training/source/formal_train.py \
  --config reproduction/seed_104948945/alternating/config.json \
  --output reproduction/seed_104948945/alternating/run \
  --run-manifest experiments/2026-09-11_alternating_training/manifest.json \
  --device cpu
```

将方法改为 `joint`，并对另外两个预先确定的训练种子重复，可复现同预算训练设计。新运行不是原机器断点的逐位恢复；软件和硬件差异可能影响数值结果。

## 同步范围

已包含历史与最新数值结果、CSV/JSON/JSONL、NPZ 评估轨迹、质量载荷表、报告、图表、日志，以及不超过 10 MiB 的小型模型/策略文件。没有因为结果不利而删除实验。

排除原始视频数据集、原始飞行数据、大于 10 MiB 的模型文件及完整训练状态、包含大型模型的归档、虚拟环境、Git 内部目录、缓存、运行锁和本地工具配置。原始数据准备/模型测量脚本仍保留。旧 `UASESE-MARL` 同步副本没有再次嵌套进本仓库。

各第三方组件的许可证保留在对应目录。仓库中的历史结果仅支持各自报告列明的比较条件。
