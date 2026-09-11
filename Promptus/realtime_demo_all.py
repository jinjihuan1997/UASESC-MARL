import os
import re
import glob
import time
import csv
import argparse

import cv2 as cv
import numpy as np
import torch
from cuda import cudart
from polygraphy import cuda

from quantization import QParam
from tensorrt_acceleration import Engine

# ========================
# QoE 评价依赖（可选）
# ========================
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import lpips

    HAS_QOE_LIBS = True
except ImportError:
    HAS_QOE_LIBS = False


# 运行指令示例：
# python realtime_demo_all.py --prompts_root "data/ocean/results" -batch 10 --power_tag 285W \
#   --result_root "realtime_result/ocean" --eval_qoe --gt_dir "data/ocean" \
#   --target_fps 20 --fps_thr 20 --lpips_thr 0.35

# ========================
# 实时生成引擎
# ========================
class Generator:
    """
    从 prompt 生成视频序列的实时引擎（UNet + VAE TensorRT）。
    支持 cond 序列长度 N 与 engine batch 不一致：
      - N <= batch  时：自动 padding 到 batch；
      - N >  batch  时：切成多个 block 分批进 engine。
    """

    def __init__(self, batch=10, device="cuda:0"):
        self.cuda_stream = cuda.Stream()
        self.use_cuda_graph = False

        self.batch = int(batch)
        self.device = device

        # 固定噪声 & diffusion 常数
        self.noise = self.seeded_randn(shape=(1, 4, 64, 64), seed=88)
        self.sigma = torch.tensor([0.05], dtype=torch.float32, device=self.device)
        self.prev_frame = None

        self.timesteps = torch.tensor([999], dtype=torch.long, device=self.device)
        self.timesteps = self.timesteps.repeat(self.batch)

        self.c_in = torch.tensor([[[[0.0683]]]], dtype=torch.float32, device=self.device)
        self.c_out = torch.tensor([[[[-14.6146]]]], dtype=torch.float32, device=self.device)

        self._last_times = {
            "denoise_ms": 0.0,
            "decode_ms": 0.0,
            "normalize_ms": 0.0,
        }

        self.denoise_engine = None
        self.decoder_engine = None

        self.denoise_engine_load()
        self.decoder_engine_load()

    # ---------- Engine 构建 & 加载 ----------

    def build_engine(self, input_profile, onnx_path, engine_path):
        _, free_mem, _ = cudart.cudaMemGetInfo()
        GiB = 2 ** 30
        if free_mem > 6 * GiB:
            activation_carveout = 4 * GiB
            max_workspace_size = free_mem - activation_carveout
        else:
            max_workspace_size = 0

        engine = Engine(engine_path)
        engine.build(
            onnx_path,
            fp16=True,
            input_profile=input_profile,
            enable_refit=False,
            enable_all_tactics=False,
            workspace_size=max_workspace_size,
        )
        return engine

    def denoise_engine_build(self):
        print("[Engine] Start building denoise engine...")
        engine_path = f"engine/denoise_batch_{self.batch}.engine"
        onnx_path = f"engine/denoise_batch_{self.batch}.onnx"
        input_profile = {
            "x": [(self.batch, 4, 64, 64)] * 3,
            "timesteps": [(self.batch,)] * 3,
            "context": [(self.batch, 77, 1024)] * 3,
        }
        self.build_engine(input_profile, onnx_path, engine_path)
        print("[Engine] Denoise engine build completed.")

    def denoise_engine_load(self):
        engine_path = f"engine/denoise_batch_{self.batch}.engine"
        if not os.path.exists(engine_path):
            print("[Engine] Denoise engine not found, building...")
            self.denoise_engine_build()
        print(f"[Engine] Loading denoise TensorRT engine: {engine_path}")
        self.denoise_engine = Engine(engine_path)
        self.denoise_engine.load()
        self.denoise_engine.activate()
        self.denoise_engine.allocate_buffers(
            shape_dict={
                "x": [self.batch, 4, 64, 64],
                "timesteps": [self.batch],
                "context": [self.batch, 77, 1024],
            },
            device=self.device,
        )

    def decoder_engine_build(self):
        print("[Engine] Start building decoder engine...")
        engine_path = f"engine/decoder_batch_{self.batch}.engine"
        onnx_path = f"engine/decoder_batch_{self.batch}.onnx"
        input_profile = {"latent": [(self.batch, 4, 64, 64)] * 3}
        self.build_engine(input_profile, onnx_path, engine_path)
        print("[Engine] Decoder engine build completed.")

    def decoder_engine_load(self):
        engine_path = f"engine/decoder_batch_{self.batch}.engine"
        if not os.path.exists(engine_path):
            print("[Engine] Decoder engine not found, building...")
            self.decoder_engine_build()
        print(f"[Engine] Loading decoder TensorRT engine: {engine_path}")
        self.decoder_engine = Engine(engine_path)
        self.decoder_engine.load()
        self.decoder_engine.activate()
        self.decoder_engine.allocate_buffers(
            shape_dict={"latent": [self.batch, 4, 64, 64]},
            device=self.device,
        )

    # ---------- 内部工具 ----------

    def seeded_randn(self, shape, seed):
        rs = np.random.RandomState(seed)
        arr = rs.randn(*shape).astype(np.float32)
        return torch.from_numpy(arr).to(self.device)

    def add_noise(self, prev_frame):
        base = (prev_frame * self.sigma + self.noise * (1 - self.sigma)).detach()
        return base.repeat(self.batch, 1, 1, 1)

    # ---------- Engine 调用 ----------

    def run_denoise_engine(self, x, timesteps, context):
        t0 = time.perf_counter()
        output = self.denoise_engine.infer(
            {"x": x, "timesteps": timesteps, "context": context},
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self._last_times["denoise_ms"] = (t1 - t0) * 1000.0
        return output["out"]

    def run_decoder_engine(self, latent):
        t0 = time.perf_counter()
        output = self.decoder_engine.infer(
            {"latent": latent},
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self._last_times["decode_ms"] = (t1 - t0) * 1000.0
        return output["images"]

    def normalization(self, images_tensor, to_numpy=False):
        t0 = time.perf_counter()
        images_tensor = torch.clamp((images_tensor + 1.0) / 2.0, 0.0, 1.0)
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        images_tensor = (images_tensor * 255).byte()
        if to_numpy:
            out = images_tensor.detach().cpu().numpy()
        else:
            out = images_tensor.detach()
        t1 = time.perf_counter()
        self._last_times["normalize_ms"] = (t1 - t0) * 1000.0
        return out

    # ---------- 兼容 interval ≠ batch 的生成 ----------

    def _generate_block(self, cond_block):
        L = cond_block.shape[0]
        assert 1 <= L <= self.batch

        if L < self.batch:
            pad = self.batch - L
            last = cond_block[-1:].repeat(pad, 1, 1)
            cond_full = torch.cat([cond_block, last], dim=0)
        else:
            cond_full = cond_block

        z = self.add_noise(self.prev_frame)
        timesteps = self.timesteps
        latent_noise = self.run_denoise_engine(z, timesteps, cond_full)
        latent_full = latent_noise * self.c_out + z / self.c_in
        images_full = self.run_decoder_engine(latent_full)
        self.prev_frame = latent_full[L - 1: L, ...]
        return images_full[:L, ...]

    def generate(self, cond_seq):
        N = cond_seq.shape[0]
        outputs = []
        start = 0
        while start < N:
            end = min(start + self.batch, N)
            block = cond_seq[start:end]
            block_imgs = self._generate_block(block)
            outputs.append(block_imgs)
            start = end  # 必须更新 start，防止死循环
        return torch.cat(outputs, dim=0)


# ========================
# QoE 计算与 CSV 记录
# ========================
def compute_qoe_metrics(result_dir, gt_dir, max_frames, device="cuda"):
    if not HAS_QOE_LIBS:
        raise RuntimeError("缺少 QoE 依赖：需要 scikit-image 和 lpips。")

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    loss_fn.eval()

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i in range(max_frames):
        pred_path = os.path.join(result_dir, f"{i:05d}.png")
        gt_path = os.path.join(gt_dir, f"{i:05d}.png")
        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            break

        pred = cv.imread(pred_path, cv.IMREAD_COLOR)
        gt = cv.imread(gt_path, cv.IMREAD_COLOR)
        if pred is None or gt is None:
            continue

        pred = cv.cvtColor(pred, cv.COLOR_BGR2RGB)
        gt = cv.cvtColor(gt, cv.COLOR_BGR2RGB)

        h = min(pred.shape[0], gt.shape[0])
        w = min(pred.shape[1], gt.shape[1])
        pred = pred[:h, :w]
        gt = gt[:h, :w]

        psnr_vals.append(peak_signal_noise_ratio(gt, pred, data_range=255))
        ssim_vals.append(structural_similarity(gt, pred, channel_axis=-1, data_range=255))

        pred_t = (
            torch.from_numpy(pred)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        gt_t = (
            torch.from_numpy(gt)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        pred_t = pred_t / 127.5 - 1.0
        gt_t = gt_t / 127.5 - 1.0

        with torch.no_grad():
            lp = loss_fn(pred_t, gt_t).item()
        lpips_vals.append(lp)

    if len(psnr_vals) == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals)), float(np.mean(lpips_vals))


def append_summary_to_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    # 为了保持 visualize_benchmark.py 的兼容性，字段名不改变
    # theo_kb_per_frame_interval / theo_kb_per_frame_all 存的是“理论语义负载”（U/V + scale/zero_point）
    fieldnames = [
        "power_tag",
        "config",
        "rank",
        "interval",
        "frames",
        "fps",
        "denoise_ms_per_frame",
        "decode_ms_per_frame",
        "e2e_ms_per_frame",
        "png_kb_per_frame",
        "theo_kb_per_frame_interval",
        "theo_kb_per_frame_all",
        "coding_rate_overall_fps_kbps",
        "coding_rate_target_fps_kbps",
        "target_fps",
        "meets_fps_thr",
        "meets_lpips_thr",
        "meets_both_thr",
        "psnr",
        "ssim",
        "lpips",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _bytes_of(x):
    """兼容 tensor / 标量的字节数估计函数。"""
    if torch.is_tensor(x):
        return x.element_size() * x.numel()
    # 对于 Python 标量（float / int），按 4 字节估算
    return 4


def prompt_bytes(p):
    """
    理论语义负载：只统计 U/V 量化张量及其 scale / zero_point 的字节数，
    不包含文件封装开销（pickle header 等）。
    """
    b = 0
    Uq, Vq = p["U"], p["V"]
    b += _bytes_of(Uq)
    b += _bytes_of(Vq)
    for k in ["U", "V"]:
        s = p[f"{k}_scale"]
        zp = p[f"{k}_zero_point"]
        b += _bytes_of(s)
        b += _bytes_of(zp)
    return b


# ========================
# 跑一个配置（一个 rankX_intervalY 目录）
# ========================
def run_one_config(prompt_dir, args):
    batch = args.batch
    device = args.device
    visualize = args.visualize

    Generator_RT = Generator(batch=batch, device=device)

    result_dir = prompt_dir
    result_frames = None
    speed_warm_up = False
    generation_speed = []

    prompts = sorted(glob.glob(os.path.join(prompt_dir, "frame_*.prompt")))
    if len(prompts) < 2:
        raise RuntimeError(f"[{prompt_dir}] 中找不到足够的 frame_*.prompt 文件")

    total_dequant_ms = 0.0
    total_interp_ms = 0.0
    total_denoise_ms = 0.0
    total_decode_ms = 0.0
    total_normalize_ms = 0.0
    total_gen_e2e_ms = 0.0
    total_frames = 0
    total_bytes_saved = 0

    # 理论语义负载总字节数（只算 U/V + scale/zero_point）
    total_theoretical_bytes = 0
    total_interval_frames = 0
    iter_idx = 0

    # ========== 主循环：逐区间生成 ==========
    for prompt_curr, prompt_next in zip(prompts, prompts[1:]):
        id_curr = int(re.search(r"frame_(\d{5})\.prompt", os.path.basename(prompt_curr)).group(1))
        id_next = int(re.search(r"frame_(\d{5})\.prompt", os.path.basename(prompt_next)).group(1))
        interval = id_next - id_curr
        if interval <= 0:
            continue

        total_interval_frames += interval

        # ---------- 2. 反量化 + 统计理论语义载荷 ----------
        _t_deq0 = time.perf_counter()
        # 先加载 prompt 对象
        prompt_curr_obj = torch.load(prompt_curr, weights_only=True)
        prompt_next_obj = torch.load(prompt_next, weights_only=True)

        # 1) 统计“理论语义负载”：只算 U/V 量化张量 + scale/zero_point 的字节数
        send_bytes = prompt_bytes(prompt_next_obj)
        if iter_idx == 0:
            # 第一个区间需要把首帧也一起传过去
            send_bytes += prompt_bytes(prompt_curr_obj)

        total_theoretical_bytes += send_bytes
        avg_bytes_per_frame_this_interval = send_bytes / max(1, interval)
        print(
            f"[{os.path.basename(prompt_dir)}][Iter-{iter_idx}] "
            f"[Theoretical Payload]/frame ≈ {avg_bytes_per_frame_this_interval / 1024:.2f} KB "
            f"(interval={interval}, send_theoretical={send_bytes / 1024:.2f} KB)"
        )

        # 2) 反量化到浮点 U/V
        U_curr, V_curr = prompt_curr_obj["U"], prompt_curr_obj["V"]
        U_next, V_next = prompt_next_obj["U"], prompt_next_obj["V"]

        qp_U_curr = QParam(num_bits=8)
        qp_U_curr.scale = prompt_curr_obj["U_scale"]
        qp_U_curr.zero_point = prompt_curr_obj["U_zero_point"]
        U_curr = qp_U_curr.dequantize_tensor(U_curr)

        qp_V_curr = QParam(num_bits=8)
        qp_V_curr.scale = prompt_curr_obj["V_scale"]
        qp_V_curr.zero_point = prompt_curr_obj["V_zero_point"]
        V_curr = qp_V_curr.dequantize_tensor(V_curr)

        qp_U_next = QParam(num_bits=8)
        qp_U_next.scale = prompt_next_obj["U_scale"]
        qp_U_next.zero_point = prompt_next_obj["U_zero_point"]
        U_next = qp_U_next.dequantize_tensor(U_next)

        qp_V_next = QParam(num_bits=8)
        qp_V_next.scale = prompt_next_obj["V_scale"]
        qp_V_next.zero_point = prompt_next_obj["V_zero_point"]
        V_next = qp_V_next.dequantize_tensor(V_next)

        torch.cuda.synchronize()
        _t_deq1 = time.perf_counter()
        dequant_ms = (_t_deq1 - _t_deq0) * 1000.0
        total_dequant_ms += dequant_ms

        # ---------- 3. 插值 + prompt 组装 ----------
        _t_int0 = time.perf_counter()
        rank = U_curr.shape[1]
        prompt_seq = []
        factor = 1.0 / interval
        for step in range(1, interval + 1):
            alpha = step * factor
            u = (1 - alpha) * U_curr + alpha * U_next
            v = (1 - alpha) * V_curr + alpha * V_next
            c = (u @ v / np.sqrt(rank)).unsqueeze(0)
            prompt_seq.append(c)
        prompt_seq = torch.cat(prompt_seq, dim=0)

        torch.cuda.synchronize()
        _t_int1 = time.perf_counter()
        interp_ms = (_t_int1 - _t_int0) * 1000.0
        total_interp_ms += interp_ms

        # ---------- 4. 首帧 warmup ----------
        if Generator_RT.prev_frame is None:
            init_path = os.path.join(prompt_dir, "init.pth")
            if not os.path.exists(init_path):
                raise RuntimeError(f"找不到初始 latent：{init_path}")
            Generator_RT.prev_frame = torch.load(init_path, weights_only=True).to(device)

            c0 = (U_curr @ V_curr / np.sqrt(rank)).unsqueeze(0)

            _t_wu0 = time.perf_counter()
            images0 = Generator_RT.generate(c0.to(device))
            images0 = Generator_RT.normalization(images0, to_numpy=True)
            torch.cuda.synchronize()
            _t_wu1 = time.perf_counter()
            print(f"[{os.path.basename(prompt_dir)}][Warmup-Init] gen_ms={(_t_wu1 - _t_wu0) * 1000.0:.2f} ms")

            if result_frames is None:
                result_frames = images0[0:1, ...]
            else:
                result_frames = np.append(result_frames, images0[0:1, ...], axis=0)
            total_frames += 1

        # ---------- 5. 正式生成本区间 ----------
        torch.cuda.synchronize()
        t_begin = time.time()

        _t_gen0 = time.perf_counter()
        images = Generator_RT.generate(prompt_seq.to(device))
        images = Generator_RT.normalization(images, to_numpy=True)
        torch.cuda.synchronize()
        _t_gen1 = time.perf_counter()
        gen_ms = (_t_gen1 - _t_gen0) * 1000.0
        total_gen_e2e_ms += gen_ms

        total_denoise_ms += Generator_RT._last_times.get("denoise_ms", 0.0)
        total_decode_ms += Generator_RT._last_times.get("decode_ms", 0.0)
        total_normalize_ms += Generator_RT._last_times.get("normalize_ms", 0.0)

        t_end = time.time()
        if speed_warm_up:
            fps = 1.0 / (t_end - t_begin) * batch
            generation_speed.append(fps)
        else:
            speed_warm_up = True

        num_generated = images.shape[0]
        total_frames += num_generated
        fps_batch = 1000.0 / gen_ms if gen_ms > 0 else 0.0
        fps_frame = fps_batch * num_generated

        print(
            f"[{os.path.basename(prompt_dir)}][Iter-{iter_idx}] interval={interval}  batch={batch} | "
            f"dequant={dequant_ms:.2f} ms  interp={interp_ms:.2f} ms  "
            f"denoise={Generator_RT._last_times['denoise_ms']:.2f} ms  "
            f"decode={Generator_RT._last_times['decode_ms']:.2f} ms  "
            f"normalize={Generator_RT._last_times['normalize_ms']:.2f} ms  "
            f"generate(E2E)={gen_ms:.2f} ms  | "
            f"FPS_batch={fps_batch:.2f}  FPS_frame={fps_frame:.2f} (frames={num_generated})"
        )

        if result_frames is None:
            result_frames = images
        else:
            result_frames = np.append(result_frames, images, axis=0)

        iter_idx += 1

    # ========== 保存 PNG ==========
    avg_speed = int(np.mean(generation_speed)) if len(generation_speed) > 0 else 0
    for i in range(result_frames.shape[0]):
        image = result_frames[i, ...]
        image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)

        out_path = os.path.join(result_dir, f"{i:05d}.png")
        cv.imwrite(out_path, image)

        try:
            total_bytes_saved += os.path.getsize(out_path)
        except OSError:
            pass

        if visualize:
            cv.putText(
                image,
                f"Generation Speed {avg_speed} FPS",
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv.imshow("Real-time Generation", image)
            cv.waitKey(10)

    # ========== SUMMARY ==========
    total_sec = total_gen_e2e_ms / 1000.0 if total_gen_e2e_ms > 0 else 0.0
    overall_fps = (total_frames / total_sec) if total_sec > 0 else 0.0
    mb = total_bytes_saved / (1024.0 * 1024.0)

    if total_frames > 0:
        avg_dequant_per_frame = total_dequant_ms / total_frames
        avg_interp_per_frame = total_interp_ms / total_frames
        avg_denoise_per_frame = total_denoise_ms / total_frames
        avg_decode_per_frame = total_decode_ms / total_frames
        avg_normalize_per_frame = total_normalize_ms / total_frames
        avg_e2e_per_frame = total_gen_e2e_ms / total_frames
        avg_kb_per_frame_saved = (total_bytes_saved / 1024.0) / total_frames
    else:
        avg_dequant_per_frame = avg_interp_per_frame = 0.0
        avg_denoise_per_frame = avg_decode_per_frame = 0.0
        avg_normalize_per_frame = avg_e2e_per_frame = 0.0
        avg_kb_per_frame_saved = 0.0

    # -------------------------------------------------
    # 1) 理论语义负载（仍然用 KiB/frame 方便人读）
    # -------------------------------------------------
    theoretical_kb_per_frame_interval = (total_theoretical_bytes / 1024.0) / max(1, total_interval_frames)
    theoretical_kb_per_frame_all = (total_theoretical_bytes / 1024.0) / max(1, total_frames)

    # -------------------------------------------------
    # 2) 标准 kbps（1 kbps = 1000 bits/s）
    #    直接从 total_theoretical_bytes 按秒算 bits，再除以 1000
    # -------------------------------------------------
    bits_per_frame_interval = (total_theoretical_bytes * 8.0) / max(1, total_interval_frames)  # bits / frame

    coding_rate_overall_fps_kbps = bits_per_frame_interval * overall_fps / 1000.0
    coding_rate_target_fps_kbps = bits_per_frame_interval * args.target_fps / 1000.0

    print(f"\n========== SUMMARY [{os.path.basename(prompt_dir)}] ==========")
    print(f"Total frames generated: {total_frames}")
    print(f"1) Total data size (saved PNG): {mb:.2f} MB ({total_bytes_saved} bytes)")
    print(f"2) Total end-to-end latency: {total_gen_e2e_ms:.2f} ms ({total_sec:.3f} s)")
    print("3) Total latency by stage:")
    print(f"   - dequant:   {total_dequant_ms:.2f} ms")
    print(f"   - interp:    {total_interp_ms:.2f} ms")
    print(f"   - denoise:   {total_denoise_ms:.2f} ms")
    print(f"   - decode:    {total_decode_ms:.2f} ms")
    print(f"   - normalize: {total_normalize_ms:.2f} ms")
    print(f"4) Overall average FPS: {overall_fps:.2f} fps")
    print("5) Average per-frame latency (ms/frame):")
    print(f"   - dequant/frame:   {avg_dequant_per_frame:.2f} ms")
    print(f"   - interp/frame:    {avg_interp_per_frame:.2f} ms")
    print(f"   - denoise/frame:   {avg_denoise_per_frame:.2f} ms")
    print(f"   - decode/frame:    {avg_decode_per_frame:.2f} ms")
    print(f"   - normalize/frame: {avg_normalize_per_frame:.2f} ms")
    print(f"   - E2E/frame:       {avg_e2e_per_frame:.2f} ms")
    print("6) Average data per frame (Theoretical Semantic Payload):")
    print(f"   - per interval frames: {theoretical_kb_per_frame_interval:.2f} KB/frame")
    print(f"   - over ALL frames:     {theoretical_kb_per_frame_all:.2f} KB/frame")
    print("7) Semantic coding rate (kbps) [Based on Theoretical Payload]:")
    print(f"   - at overall FPS ({overall_fps:.2f} fps): {coding_rate_overall_fps_kbps:.2f} kbps")
    print(f"   - at target FPS ({args.target_fps:.2f} fps): {coding_rate_target_fps_kbps:.2f} kbps")
    print("====================================================\n")

    # ========== QoE 评价（可选）==========
    config_name = os.path.basename(prompt_dir.rstrip(os.sep))
    m_cfg = re.search(r"rank(\d+)_interval(\d+)", config_name)
    rank_val = int(m_cfg.group(1)) if m_cfg else -1
    interval_val = int(m_cfg.group(2)) if m_cfg else -1

    psnr_mean = ssim_mean = lpips_mean = 0.0

    if args.eval_qoe and args.gt_dir is not None:
        if not HAS_QOE_LIBS:
            print("[QoE] 缺少依赖：请先 pip install scikit-image lpips")
        else:
            print(f"[QoE][{config_name}] Computing PSNR / SSIM / LPIPS (offline)...")
            psnr_mean, ssim_mean, lpips_mean = compute_qoe_metrics(
                result_dir, args.gt_dir, total_frames, device=device
            )
            print(f"[QoE][{config_name}] Mean PSNR : {psnr_mean:.2f} dB")
            print(f"[QoE][{config_name}] Mean SSIM : {ssim_mean:.4f}")
            print(f"[QoE][{config_name}] Mean LPIPS: {lpips_mean:.4f}")

    meets_fps_thr = 1 if overall_fps >= args.fps_thr else 0
    if args.eval_qoe and HAS_QOE_LIBS:
        meets_lpips_thr = 1 if lpips_mean <= args.lpips_thr else 0
    else:
        meets_lpips_thr = -1
    meets_both_thr = 1 if (meets_fps_thr == 1 and meets_lpips_thr == 1) else 0

    if args.metrics_csv:
        append_summary_to_csv(
            args.metrics_csv,
            dict(
                power_tag=args.power_tag,
                config=config_name,
                rank=rank_val,
                interval=interval_val,
                frames=total_frames,
                fps=overall_fps,
                denoise_ms_per_frame=avg_denoise_per_frame,
                decode_ms_per_frame=avg_decode_per_frame,
                e2e_ms_per_frame=avg_e2e_per_frame,
                png_kb_per_frame=avg_kb_per_frame_saved,
                # theo_* 字段：理论语义负载 (U/V + scale/zero_point)
                theo_kb_per_frame_interval=theoretical_kb_per_frame_interval,
                theo_kb_per_frame_all=theoretical_kb_per_frame_all,
                coding_rate_overall_fps_kbps=coding_rate_overall_fps_kbps,
                coding_rate_target_fps_kbps=coding_rate_target_fps_kbps,
                target_fps=args.target_fps,
                meets_fps_thr=meets_fps_thr,
                meets_lpips_thr=meets_lpips_thr,
                meets_both_thr=meets_both_thr,
                psnr=psnr_mean,
                ssim=ssim_mean,
                lpips=lpips_mean,
            ),
        )
        print(f"[Summary] [{config_name}] appended to CSV: {args.metrics_csv}")

    return {
        "config": config_name,
        "frames": total_frames,
        "fps": overall_fps,
        "psnr": psnr_mean,
        "ssim": ssim_mean,
        "lpips": lpips_mean,
    }


# ========================
# 主入口：批量模式
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_root", type=str, required=True, help="包含 rank*_interval* 子目录的根目录")
    parser.add_argument("-batch", type=int, default=10, help="TensorRT engine 的 batch 尺寸")
    parser.add_argument("-visualize", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gt_dir", type=str, default=None, help="GT 帧目录")
    parser.add_argument("--eval_qoe", action="store_true", help="是否计算 PSNR/SSIM/LPIPS")
    parser.add_argument("--result_root", type=str, default="realtime_result", help="保存 CSV 的根目录")
    parser.add_argument("--power_tag", type=str, required=True, help="功率标签")
    parser.add_argument("--metrics_csv", type=str, default=None, help="CSV 文件名")
    parser.add_argument("--target_fps", type=float, default=20.0, help="计算码率的目标 FPS")
    parser.add_argument("--lpips_thr", type=float, default=0.35, help="LPIPS 阈值")
    parser.add_argument("--fps_thr", type=float, default=15.0, help="FPS 阈值")

    args = parser.parse_args()

    summary_dir = os.path.join(args.result_root, args.power_tag)
    os.makedirs(summary_dir, exist_ok=True)

    if args.metrics_csv is None:
        args.metrics_csv = os.path.join(summary_dir, "benchmark_summary.csv")
    else:
        if not os.path.isabs(args.metrics_csv):
            args.metrics_csv = os.path.join(summary_dir, args.metrics_csv)

    root = args.prompts_root
    subdirs = sorted(glob.glob(os.path.join(root, "rank*_interval*")))
    if len(subdirs) == 0:
        raise RuntimeError(f"在 {root} 下没有找到 rank*_interval* 子目录")
    print(f"[Batch] Found {len(subdirs)} configs under {root}")

    for d in subdirs:
        print(f"\n[Batch] Running config: {d}")
        run_one_config(d, args)
