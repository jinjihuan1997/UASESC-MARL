#!/bin/bash

# 定义功率限制列表 (100W, 180W, 260W)
POWER_LIMITS=(100 180 260)

echo "========================================"
echo "Starting SC Benchmark (Sweep Mode)"
echo "Target: Calculate quality once, sweep Bitrate for FPS 15-25"
echo "========================================"

# 循环：遍历所有功率限制
# 我们只需要改变功率，然后运行一次生成和一次测量即可
for PL in "${POWER_LIMITS[@]}"; do
    echo "----------------------------------------"
    echo "  [Status] Setting GPU Power Limit to ${PL}W..."

    # 1. 设置显卡功率墙 (需要 sudo 权限)
    sudo nvidia-smi -i 0 -pl "$PL"

    # 2. 运行生成脚本
    # 目的：在当前功耗限制下运行模型，生成 enc_fps.txt (记录硬件编码速度)
    # 注意：生成过程与 FPS 无关，只与 Rank/Interval 有关，所以跑一次就行
    echo "  [Status] Running generation script (to measure encoding speed)..."
    python real_time_generation_scan.py -results_root data/sky/results -batch 10

    # 3. 运行测量脚本 (新版 Sweep 逻辑)
    # 目的：
    #   a. 计算一次画质 (LPIPS/PSNR)
    #   b. 读取刚才生成的 enc_fps
    #   c. 在内部循环生成 15~25 FPS 的码率数据并保存 CSV
    echo "  [Status] Running measurement sweep..."
    python measure_sc.py \
        --frames_dir data/sky \
        --fps_start 15 \
        --fps_end 25 \
        --enable_lpips \
        --max_frames 25 \
        --pwd "${PL}W"
    # 可选：短暂休眠让显卡状态稳定
    sleep 2
done

# 脚本结束后将功率恢复到默认最大值 (假设您的卡是 285W)
echo "========================================"
echo "Benchmark Finished. Resetting Power Limit to default (285W)."
sudo nvidia-smi -i 0 -pl 285