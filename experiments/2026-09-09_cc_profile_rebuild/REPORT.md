CC profile 已重建并完成核验（2026-09-09）。这是一份独立的 H.264 编码测量、5G LDPC 信道蒙特卡洛测量和整包成功率解析模型相结合的平均表。没有使用旧表中人为设定的 H.264 bpp/PSNR 曲线，也没有为了得到 SC 优势而缩放 CC 载荷。

正式表：[cc_h264_ldpc_rgb8_nfs_v1.npz](/home/king/Downloads/Projects/2_th_paper_TMC/HARL/HARL/harl/envs/uav_escs/semantic_models/profiles/cc_h264_ldpc_rgb8_nfs_v1.npz)。逐片段数据及冻结协议：[测量目录](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_profile_rebuild/runs/rgb8_nfs_v1)。SC 原表、旧 CC 表、旧生成脚本、SC 正式训练输入均未修改；本次未启动 CC 强化学习训练。

**数据和固定设置**

- 数据来自现有 NFS test 索引。视频 ID 0–7 各取 12 段，共 96 段校准；ID 8–13 各取 8 段，共 48 段验证；ID 14–19 保留未使用。按索引中固定的均匀分位位置选片段，不根据质量或 SC 对比结果筛选。
- 每段取索引中最前面的 8 张连续图像，256×256 RGB；逐帧保存文件 SHA-256。每段独立编码，使用 libx264 medium、yuv420p、QP 42/38/34/30/26、8 帧 GOP、240 fps 元数据。RGB PSNR 包含颜色空间转换误差。
- 没有使用旧的 16 帧灰度模型加载器，也不向接收端免费提供当前或前一个原始视频块。校准的 768 个帧引用对应 739 张不同图像，验证的 384 个帧引用对应 382 张不同图像；视频内部片段可少量重叠，两个分区没有交叉图像。不能把所有帧或片段当作独立统计样本。
- 5 个 QP × 4 个 LDPC 目标码率（1/2、2/3、3/4、5/6），共 20 个模式全部保留。每块 1024 个输入位；实际输出长度分别为 2048、1536、1366、1230 位，实际码率见表内字段。采用 QPSK 和 20 次 LDPC 译码迭代。
- SNR 为复符号 Es/N0，采样点为 0、1、2、3、4、5、6、8、10、15、20 dB。每个码率/SNR 点测试 8192 块，总计 360448 块。随机种子 20260909，逐批次种子和误块数全部保留。运行使用 CPU 0、2；GPU 继续用于 SC 训练。
- 协议固定后执行了 720 次真实 H.264 编码/解码，保存全部 720 个码流；生成 31680 条片段×模式×SNR 统计。其中验证视频不参与最终表求均值。先在视频内平均，再对视频等权平均。

**表的含义**

载荷单位是复信道使用次数，既不是字节也不是码率。先对每个片段统计真实 H.264 字节，再增加明确的 16 字节包头（版本、模式、长度及 CRC32 等），补齐到完整 LDPC 输入块，最后计算实际调制符号数。H.264 自带的参数集等已包含在测得的码流字节中。

```text
J = ceil(8 × (H264_bytes + 16) / 1024)
n = 向上补齐到完整 QPSK 符号的 LDPC 输出位数
L = J × n / 2
p_success = (1 − measured_BLER)^J
Q_expected = p_success × Q_success + (1 − p_success) × Q_outage
```

同一码率使用固定 QPSK，每次发送的 L 因此不随 SNR 改变，不能再除一次 log2(1+SNR)。传输失败时采用接收端已知的 RGB=128 中性画面；它的 PSNR 是对各原始片段实际计算的。该模型假设 AWGN 编码块独立、包错误被检测、无 ARQ；没有把损坏的 H.264 比特流送入视频解码器测试，也没有测量完整 CRC 漏检机制。

`q_hat_mean` 是包含传输失败的“片段 PSNR 的期望”，不是每次收到后的确定质量；`q_success_mean` 和 `q_outage_mean` 分别是对所有片段的无损传输重建质量、中性画面质量求均值，`deliver_prob` 是平均整包成功率。由于片段大小、图像质量和成功率有关联，不能把这些均值直接相乘来代替逐片段混合后求均值。尤其是 `q_success_mean` 不等于跨片段总体按实际成功事件筛选后的条件均值。若需要总体条件质量，应从逐片段记录计算 `E[p × Q_success]/E[p]` 和 `E[(1−p) × Q_outage]/E[1−p]`；分母为零的分支不会发生，条件值未定义。`psnr_from_expected_mse` 单独保存另一种聚合口径，不能与前者混用。`q_hat_std` 是片段与成败混合分布的标准差，不是均值的标准误。`avg_kept_real_symbols_mean = 2 × bar_ls_main_mean`，表示实维度计数。

**测量结果**

下表是校准视频的平均值。成功质量只描述完整收到包后的图像，含丢包的最终质量还随 SNR 和码率变化。载荷范围从 5/6 码率到 1/2 码率。

| H.264 QP | 平均码流字节 | 成功收到时 RGB PSNR | 平均载荷范围（复信道使用） |
|---|---:|---:|---:|
| 42 | 1966.68 | 33.112 dB | 9840.00–16384.00 |
| 38 | 2608.80 | 34.810 dB | 12915.00–21504.00 |
| 34 | 3717.74 | 36.376 dB | 18225.78–30346.67 |
| 30 | 5589.62 | 37.836 dB | 27245.78–45365.33 |
| 26 | 9146.32 | 39.184 dB | 44344.06–73834.67 |

含传输失败后的全表质量范围为 11.392–39.184 dB。以下为每个信道点 8192 块中观测的错误块数：

| LDPC 码率 | 0 dB | 1 dB | 2 dB | 3 dB | 4 dB | 5 dB | 6 dB | 8/10/15/20 dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1/2 | 8192 | 3872 | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 |
| 2/3 | 8192 | 8192 | 8192 | 3878 | 3 | 0 | 0 | 0/0/0/0 |
| 3/4 | 8192 | 8192 | 8192 | 8191 | 4047 | 6 | 0 | 0/0/0/0 |
| 5/6 | 8192 | 8192 | 8192 | 8192 | 8192 | 4909 | 28 | 0/0/0/0 |

零错误不代表完全可靠。每点给出 95% Clopper–Pearson 误块率区间，并传播到包成功率和质量区间。8192 块零错误时误块率的上限仍约为 0.000450；高 SNR 下各模式平均整包成功率的相应区间下限约为 0.9683–0.9928。区间只反映每个信道点的蒙特卡洛不确定性，不是所有点的同时置信区间，也不包含视频总体抽样不确定性。

对 6 个未参与建表的视频，以表均值预测各片段的期望质量，20 个模式和 11 个 SNR 点等权后的平均绝对误差为 **1.585 dB**；平均载荷相对绝对误差为 **17.84%**。各模式分别为 1.150–2.065 dB 和 13.01%–21.63%。这说明平均表有真实内容波动，不能当作任意具体视频的精确预测。验证集的“期望质量”仍由同一信道模型组合得到，这项验证主要检查平均表跨视频的代表性，不能替代独立无线实测。

**完成的核验**

- 6 项测试通过：8 帧 RGB 读取与篡改检查、真实 H.264 往返与确定性、四种码率的无噪声 LDPC/QPSK 逐位恢复、包头/补齐边界、逐片段成功率后平均及验证集隔离、零误块置信区间。
- 修复本次新构建器的 FFmpeg 帧率处理问题：显式输入帧率并使用 passthrough，防止原始 H.264 解码时静默丢帧。不通过补帧或重复帧掩盖错误。
- 用冻结构建器重新聚合，profile 和逐片段统计与首次生成逐项一致，manifest 哈希一致。
- 独立脚本不调用构建器的聚合函数，重新核算 220 个质量/载荷/整包成功率值，最大浮点误差分别为 7.11e-15、7.28e-12、2.22e-16。1152 个源帧引用逐一校验；14 个视频各选一段、抽查两个极端 QP，共 28 个码流重新解码，全部精确为 8 帧且 PSNR 一致。
- 现有 CC 表加载器能读取新表的 20×11 数组；这仅验证文件读取。

profile SHA-256：`1577cd5ee5cfe0f74c325f42d32e59fe5f417c13bc1f51a4539d2b4addad7a96`。详见 [validation.json](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_profile_rebuild/runs/rgb8_nfs_v1/validation.json)、[inspection.json](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_profile_rebuild/runs/rgb8_nfs_v1/inspection.json) 和 [protocol.json](/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_profile_rebuild/runs/rgb8_nfs_v1/protocol.json)。

**接下来接入 CC 训练时的边界**

1. 现有 CC `_build_semantic_mode_lookup` 仍返回 4×4 旧映射。即使把 `n_semantic_modes` 改为 20、设置 `all_modes`，实际只到达 ID 0–4。因此不能只改一个表路径就启动正式对比；需把全部 20 个模式接入与 SC 一致的训练环境。现有 GPU 环境也专用于 SC 表，不能直接替换输入。
2. 旧 CC 加载器只返回质量和载荷，忽略本表的整包成功率。正式环境必须明确发送尝试、成功收到、显示质量及 AoI 更新之间的关系；若抽样包成败，使用同一片段的载荷/成功率/质量联合记录，或先按上一段公式聚合总体条件质量。失败不能按成功接收重置 AoI，也不能一边抽样失败一边再次使用已经扣过失败的平均质量。不能把本表当作每次必然成功更新的可靠链路。
3. 当前 SC 配置下，每 UAV 均分预算为 20000，单 UAV 最大可分到 54000 个复信道使用。高 SNR 下分别有 8/18 个 CC 模式满足平均载荷预算和 21 dB 最低质量；模式 16、17 的平均载荷超过单 UAV 上限。0–1 dB 所有模式都达不到 21 dB。这里只是静态均值检查，不代表实际调度结果；未据此删除模式或调整预算。
4. 更正（2026-09-09 CC 接入时逐行复核）：33 dB 是质量归一化的参考值，不是截断上限。当前冻结的 SC `tensor_env.py` 使用 `max((Q−21)/(33−21), 0)`，没有上界截断；所以 CC 的 33.11–39.18 dB 成功质量仍会产生不同的质量增益。此前本报告关于超过 33 dB 不增加奖励的表述不正确。CC 沿用实际代码中的同一公式，无需因此改变参数或重启 SC。
5. 本次统一的是声明的 8 帧、256×256 RGB 更新块和载荷单位。用户指定保留的 SC 表仍是 derived 平均表，其真实编码器、历史输入和码流测量绑定尚未完全建立。因此可以明确写成“固定 SC 平均模型与独立校准 CC 平均模型的系统比较”，仅凭本次重建不能声称完成真实 SC/CC 编解码器之间的严格公平验证。

**复现命令**

在项目根目录执行；`prepare` 必须使用一个不存在的新输出目录，避免覆盖当前正式表。`run` 可用冻结源文件续跑未完成的测量。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
CC_PY=/home/king/miniconda3/envs/harl_sionna/bin/python
CC_EXP=/home/king/Downloads/Projects/2_th_paper_TMC/experiments/2026-09-09_cc_profile_rebuild
CC_OUT="$CC_EXP/runs/rgb8_nfs_v1"
"$CC_PY" "$CC_OUT/source/rebuild_cc.py" verify --output "$CC_OUT"
"$CC_PY" "$CC_EXP/inspect_profile.py" --output "$CC_OUT"
"$CC_PY" -m unittest discover -s "$CC_EXP" -p test_rebuild_cc.py -v

# 需要独立重跑时，使用一个新的路径：
CC_NEW_OUT="$CC_EXP/runs/rgb8_nfs_reproduction"
"$CC_PY" "$CC_EXP/rebuild_cc.py" prepare --output "$CC_NEW_OUT"
"$CC_PY" "$CC_NEW_OUT/source/rebuild_cc.py" run --output "$CC_NEW_OUT"
```

正式执行使用 ffmpeg version N-122425-g21a3e44fbe Copyright (c) 2000-2026 the FFmpeg developers、TensorFlow 2.15.1、Sionna 0.19.2。文件级输入/代码哈希与批次种子保存在协议和记录中。正式测量约 5 分 18 秒（同机 SC 训练并行时），不包含前期脚本开发与测试；异机耗时不可直接套用。
