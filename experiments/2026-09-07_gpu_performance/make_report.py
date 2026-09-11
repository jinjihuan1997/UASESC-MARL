from pathlib import Path
import json

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / "summary.json").read_text())
labels = {"cpu": "原实现 CPU", "gpu": "原实现 GPU", "cpu_vector": "批量概率计算 + CPU",
          "gpu_vector": "批量概率计算 + GPU", "hybrid_vector": "批量概率计算 + CPU 采样 / GPU 更新"}
baseline = summary["cpu"]["pooled_steady_fps"]
best = summary["hybrid_vector"]["pooled_steady_fps"]
lines = [
    f"混合执行方案在本轮测试中最快，汇总吞吐量为 {best:.1f} 步/秒，是原 CPU 实现的 {best/baseline:.2f} 倍。优化和校验均在独立目录中执行，正式 A/B 队列的代码与配置保持冻结。",
    "",
    "| 方案 | 汇总稳态步/秒 | 3 个种子的范围 | 相对原 CPU |",
    "| --- | ---: | ---: | ---: |",
]
for mode, r in summary.items():
    lines.append(f"| {labels[mode]} | {r['pooled_steady_fps']:.1f} | {r['min_fps']:.1f}–{r['max_fps']:.1f} | {r['pooled_steady_fps']/baseline:.2f}× |")
lines += ["", "每次更新的分阶段平均耗时（秒；每次更新收集 4,000 个环境步）：", "",
          "| 方案 | 动作采样与价值推断 | 环境与进程通信 | 回报计算 | 参数更新及数值检查 |",
          "| --- | ---: | ---: | ---: | ---: |"]
for mode, r in summary.items():
    s = r["mean_stage_seconds_per_update"]
    lines.append(f"| {labels[mode]} | {s['collect']:.3f} | {s['environment_and_ipc']:.3f} | {s['compute']:.3f} | {s['train']:.3f} |")
lines += [
    "", "测试范围与解释", "",
    "5 种方案各 3 个种子（1、2、3），每次 16,000 步，共 240,000 步。每次 4 次更新，首更新作为预热从稳态指标中排除；每种方案累计 36,000 个稳态计时步。另有早期混合方案和原 GPU 函数剖析各 8,000 步，用于定位问题，不纳入上表。执行顺序与参数事先记录在 plan.json。正式 CPU 队列全程仍在运行，因此这不是机器独占基准；结果用于本机实现选择，不用于比较策略优劣。",
    "", "瓶颈证据", "",
    "原 GPU 的两次更新在函数剖析中耗时 28.177 秒，其中 FixedSimplex.log_probs 和 entropy 累计约 15.52 秒；创建 Dirichlet 分布 112,000 次，调用 torch.nonzero 112,000 次。逐样本循环既增加 Python 与设备同步开销，也生成庞大的反向计算图。剖析期间有测量开销，该耗时不用于上表吞吐量。",
    "",
    "vectorized_simplex.py 将掩码 Dirichlet 的概率密度和熵改为等价的批量计算。采样、可用维度掩码、单维/空掩码退化处理、动作归一化和确定性动作保持原实现。CPU 与 GPU 都因此提速。混合方案进一步将每步的小批量推断和回报计算放在 CPU，将大批量网络更新放在 GPU，每次更新后同步参数和归一化统计。它未改变网络结构、奖励、物理环境、PPO 损失、更新顺序规则、学习轮数、截断规则或每次更新的数据量。不同设备的随机数流可能不同，不能要求完整训练轨迹逐位一致。",
    "", "检查点缺陷", "",
    "原 ValueNorm 在构造 nn.Parameter 后调用 .to(cuda)，该 Torch 版本会使这些统计量成为未注册的普通张量。已实际发现原 GPU 的 value_normalizer.pt 是空字典，CPU 文件则有 running_mean、running_mean_sq、debiasing_term 三个字段。因此之前的检查点可读取检查不足以证明能完整恢复 critic 的归一化状态。原 GPU 基线保留这一行为用于测量，并在 summary.json 中标为 normalizer_checkpoint_complete=false；优化 GPU 方案修复了注册方式。",
    "",
    "proposed_valuenorm_registration_fix.patch 和 proposed_vectorized_simplex.patch 是可审查的修改补丁；proposals/ 保存对应源码。未将补丁应用到正在运行的正式实验。",
    "", "验证记录", "",
    "vectorization_tests.log：CPU/CUDA、float32/float64 下的混合掩码概率、熵及梯度；空掩码、单维掩码、无效位置零梯度；采样及随机数状态保持一致；GPU 归一化统计保存/加载。",
    "update_equivalence_tests.log：完整 HAPPO 单个固定 minibatch 的损失、梯度、参数与 Adam 状态对照；包含回合终止与时间截断的完整 GAE 回报对照；直接 GPU 构造修复的检查点重载。",
    "每次混合训练结束，还核对了 GPU/CPU 推断副本的参数逐项一致、相同动作的 log-probability、确定性动作、critic 输出和反归一化结果。各组均完成训练且参数/训练指标为有限数值，所有 actor 均更新。这里只证明实现与固定输入计算的一致性及短程训练可运行，不能代替长程收敛和正式论文性能实验。",
    "", "运行与复现", "",
    "probe.py 为隔离测试入口；run_matrix.py 为事先固定顺序的 15 组测试队列；source_frozen_manifest.json 所指向的正式源文件不受这些进程内替换影响。probe_frozen.py.txt 和 vectorized_simplex_frozen.py.txt 保存此次测试源码，plan.json 保存其哈希。每组目录下保留配置、分阶段计时、模型、训练输出与检查点哈希。",
    "",
    "在可访问 GPU 的主机环境中验证测试（项目根目录）：", "", "```bash",
    "/home/king/miniconda3/envs/harl_sionna/bin/python -m pytest experiments/2026-09-07_gpu_performance/test_vectorized_simplex.py experiments/2026-09-07_gpu_performance/test_update_equivalence.py -q",
    "```", "",
    "独立复现速度测试时，先复制本目录脚本到新的 experiments 子目录，然后运行该目录的 run_matrix.py。脚本拒绝覆盖原 plan.json 和已存在的测试结果目录。",
    "", "采用建议", "",
    f"后续新实验优先考虑批量概率计算加 CPU 采样/GPU 更新。按本轮稳态吞吐量，12,000,000 步的纯训练时间粗估为 {12000000/best/3600:.2f} 小时，仍需另计初始化、检查点、评估及运行时负载；这不是长程耗时保证。采用优化时，应重新冻结整套 A/B 实现，并使用完整检查点继续或重新训练，避免将不同版本结果混合。该提速属于工程实现优化，不能作为新的 HAPPO/PPO 算法创新。",
]
(OUT / "README.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"best_fps":best,"speedup_over_original_cpu":best/baseline,"train_only_hours_estimate":12000000/best/3600}))
