import time
import torch
import numpy as np
import cv2 as cv
import os
import re
import glob
import argparse
from cuda import cudart
from quantization import QParam
from polygraphy import cuda
from tensorrt_acceleration import Engine


## Real-time generation engine, from prompt to video.
class Generator():
    def __init__(self, batch=1, device='cuda:0'):
        self.cuda_stream = cuda.Stream()
        self.use_cuda_graph = False
        self.batch = batch  # the number of frames generated at once, needs to match the engine file.
        self.device = device
        self.noise = self.seeded_randn(shape=(1, 4, 64, 64),
                                       seed=88)  # the random seed needs to be consistent with the inversion.
        self.sigma = torch.Tensor([0.05]).float().cuda()
        self.prev_frame = None
        self.timesteps = torch.Tensor([999]).long().to(self.device)
        self.timesteps = self.timesteps.repeat(self.batch)
        self.c_in = torch.Tensor([[[[0.0683]]]]).float().to(self.device)
        self.c_out = torch.Tensor([[[[-14.6146]]]]).float().to(self.device)
        self.denoise_engine = None
        self.decoder_engine = None

        # 记录最近一次推理各阶段耗时（毫秒）；不改变任何函数签名
        self._last_times = {'denoise_ms': 0.0, 'decode_ms': 0.0, 'normalize_ms': 0.0}

        self.denoise_engine_load()
        self.decoder_engine_load()

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
        print('Start building the denoise engine for this machine.')
        engine_path = 'engine/denoise_batch_{}.engine'.format(self.batch)
        onnx_path = 'engine/denoise_batch_{}.onnx'.format(self.batch)
        input_profile = {'x': [(self.batch, 4, 64, 64), (self.batch, 4, 64, 64), (self.batch, 4, 64, 64)],
                         'timesteps': [(self.batch,), (self.batch,), (self.batch,)],
                         'context': [(self.batch, 77, 1024), (self.batch, 77, 1024), (self.batch, 77, 1024)]}
        self.build_engine(input_profile, onnx_path, engine_path)
        print('Build completed successfully.')

    def denoise_engine_load(self):
        engine_path = 'engine/denoise_batch_{}.engine'.format(self.batch)
        if not os.path.exists(engine_path):
            print('The denoise engine is not found.')
            self.denoise_engine_build()
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
        print('Start building the decoder engine for this machine.')
        engine_path = 'engine/decoder_batch_{}.engine'.format(self.batch)
        onnx_path = 'engine/decoder_batch_{}.onnx'.format(self.batch)
        input_profile = {'latent': [(self.batch, 4, 64, 64), (self.batch, 4, 64, 64), (self.batch, 4, 64, 64)]}
        self.build_engine(input_profile, onnx_path, engine_path)
        print('Build completed successfully.')

    def decoder_engine_load(self):
        engine_path = 'engine/decoder_batch_{}.engine'.format(self.batch)
        if not os.path.exists(engine_path):
            print('The decoder engine is not found.')
            self.decoder_engine_build()
        self.decoder_engine = Engine(engine_path)
        self.decoder_engine.load()
        self.decoder_engine.activate()
        self.decoder_engine.allocate_buffers(
            shape_dict={
                "latent": [self.batch, 4, 64, 64],
            },
            device=self.device,
        )

    # ========= 增加计时（不改变签名/返回值）=========
    def run_denoise_engine(self, x, timesteps, context):
        t0 = time.perf_counter()
        output = self.denoise_engine.infer(
            {
                "x": x,
                "timesteps": timesteps,
                "context": context,
            },
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self._last_times['denoise_ms'] = (t1 - t0) * 1000.0
        return output['out']

    def run_decoder_engine(self, latent):
        t0 = time.perf_counter()
        output = self.decoder_engine.infer(
            {
                "latent": latent,
            },
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self._last_times['decode_ms'] = (t1 - t0) * 1000.0
        return output['images']

    def normalization(self, images_tensor, to_numpy=False):
        t0 = time.perf_counter()
        images_tensor = torch.clamp((images_tensor + 1.0) / 2.0, min=0.0, max=1.0)
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        images_tensor = (images_tensor * 255).byte()
        if to_numpy:
            out = images_tensor.detach().cpu().numpy()
        else:
            out = images_tensor.detach()
        t1 = time.perf_counter()
        self._last_times['normalize_ms'] = (t1 - t0) * 1000.0
        return out

    def seeded_randn(self, shape, seed):
        randn = np.random.RandomState(seed).randn(*shape)
        randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
        return randn

    def add_noise(self, prev_frame):
        # 使用 self.batch，避免潜在的 NameError
        noised_prev_frame = (prev_frame * self.sigma + self.noise * (1 - self.sigma)).detach()
        noised_prev_frame = noised_prev_frame.repeat(self.batch, 1, 1, 1)
        return noised_prev_frame

    # 用这段修复后的代码替换原有的 generate 函数
    def generate(self, cond):
        """支持生成任意长度序列，自动分批处理"""
        # 重置单次调用的计时
        self._last_times['denoise_ms'] = 0.0
        self._last_times['decode_ms'] = 0.0

        total_frames_needed = cond.shape[0]
        all_generated_images = []

        # 循环：每次处理 self.batch (10) 帧
        for start_idx in range(0, total_frames_needed, self.batch):
            end_idx = min(start_idx + self.batch, total_frames_needed)
            current_cond_chunk = cond[start_idx:end_idx]
            actual_batch_size = current_cond_chunk.shape[0]

            # 填充逻辑：如果不足 10 帧，补齐到 10 帧
            if actual_batch_size < self.batch:
                pad_size = self.batch - actual_batch_size
                padding = current_cond_chunk[-1:].repeat(pad_size, 1, 1)
                batch_cond = torch.cat([current_cond_chunk, padding], dim=0)
            else:
                batch_cond = current_cond_chunk

            # 推理
            z = self.add_noise(self.prev_frame)
            latent_noise = self.run_denoise_engine(z, self.timesteps, batch_cond)
            latent = latent_noise * self.c_out + z / self.c_in
            images = self.run_decoder_engine(latent)

            # 更新上一帧（关键！）
            self.prev_frame = latent[actual_batch_size - 1: actual_batch_size, ...]

            # 收集有效帧
            valid_images = images[:actual_batch_size, ...]
            all_generated_images.append(valid_images)

        # 拼接结果
        if len(all_generated_images) > 0:
            return torch.cat(all_generated_images, dim=0)
        else:
            return torch.Tensor([]).to(self.device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-prompt_dir', type=str, default="data/sky/results/rank8_interval10")
    parser.add_argument('-batch', type=int, default=10)  # 沿用你的默认；类型为 int
    parser.add_argument('-visualize', type=bool, default=False)
    args = parser.parse_args()

    prompt_dir = args.prompt_dir
    batch = args.batch
    Generator_RT = Generator(batch=batch)
    result_dir = prompt_dir
    result_frames = None
    speed_warm_up = False
    generation_speed = []
    prompts = sorted(glob.glob(os.path.join(prompt_dir, 'frame_*.prompt')))

    # ====== 总计累加器 ======
    total_dequant_ms = 0.0
    total_interp_ms = 0.0
    total_denoise_ms = 0.0
    total_decode_ms = 0.0
    total_normalize_ms = 0.0
    total_gen_e2e_ms = 0.0
    total_frames = 0
    total_bytes_saved = 0  # 保存 PNG 的总字节数

    # 理论传输统计：每个 interval 仅传下一关键帧（首区间额外传首关键帧）
    total_theoretical_bytes = 0
    total_interval_frames = 0  # 累加所有 interval（通常=10）
    iter_idx = 0


    def prompt_bytes(p):
        """计算一份关键帧提示（量化 U/V + scale/zero_point）的理论传输字节数"""
        b = 0
        Uq, Vq = p['U'], p['V']
        b += Uq.element_size() * Uq.numel()
        b += Vq.element_size() * Vq.numel()
        for k in ['U', 'V']:
            s = p[f'{k}_scale']
            zp = p[f'{k}_zero_point']
            b += s.element_size() * s.numel()
            b += zp.element_size() * zp.numel()
        return b


    for prompt_pair in zip(prompts[::], prompts[1::]):
        prompt_curr = prompt_pair[0]
        id_curr = int(re.search(r'frame_(\d{5})\.prompt', prompt_curr).group(1))
        prompt_next = prompt_pair[1]
        id_next = int(re.search(r'frame_(\d{5})\.prompt', prompt_next).group(1))
        # interpolation interval
        interval = id_next - id_curr

        # ============ 反量化计时 ============
        _t_deq0 = time.perf_counter()
        prompt_curr_obj = torch.load(prompt_curr, weights_only=True)
        prompt_next_obj = torch.load(prompt_next, weights_only=True)

        # 理论传输：本区间仅需发送下一关键帧；首个区间额外包含当前关键帧
        send_bytes_this_interval = prompt_bytes(prompt_next_obj)
        if iter_idx == 0:
            send_bytes_this_interval += prompt_bytes(prompt_curr_obj)
        total_theoretical_bytes += send_bytes_this_interval
        total_interval_frames += interval
        avg_bytes_per_frame_this_interval = send_bytes_this_interval / max(1, interval)
        print(f"[Iter-{iter_idx}] theoretical payload per frame ≈ {avg_bytes_per_frame_this_interval / 1024:.2f} KB "
              f"(interval={interval}, sent={send_bytes_this_interval / 1024:.2f} KB)")

        # low-rank factors
        U_curr, V_curr = prompt_curr_obj['U'], prompt_curr_obj['V']
        U_next, V_next = prompt_next_obj['U'], prompt_next_obj['V']

        # prompt dequantization
        Quant_Param_U_curr = QParam(num_bits=8)
        Quant_Param_U_curr.scale = prompt_curr_obj['U_scale']
        Quant_Param_U_curr.zero_point = prompt_curr_obj['U_zero_point']
        U_curr = Quant_Param_U_curr.dequantize_tensor(U_curr)

        Quant_Param_V_curr = QParam(num_bits=8)
        Quant_Param_V_curr.scale = prompt_curr_obj['V_scale']
        Quant_Param_V_curr.zero_point = prompt_curr_obj['V_zero_point']
        V_curr = Quant_Param_V_curr.dequantize_tensor(V_curr)

        Quant_Param_U_next = QParam(num_bits=8)
        Quant_Param_U_next.scale = prompt_next_obj['U_scale']
        Quant_Param_U_next.zero_point = prompt_next_obj['U_zero_point']
        U_next = Quant_Param_U_next.dequantize_tensor(U_next)

        Quant_Param_V_next = QParam(num_bits=8)
        Quant_Param_V_next.scale = prompt_next_obj['V_scale']
        Quant_Param_V_next.zero_point = prompt_next_obj['V_zero_point']
        V_next = Quant_Param_V_next.dequantize_tensor(V_next)
        torch.cuda.synchronize()
        _t_deq1 = time.perf_counter()
        dequant_ms = (_t_deq1 - _t_deq0) * 1000.0
        total_dequant_ms += dequant_ms

        # ============ 插值 + 组装 prompt 计时 ============
        _t_int0 = time.perf_counter()
        rank = U_curr.shape[1]
        prompt = []
        # linear interpolation on keyframe prompts, approximating the intermediate prompts.
        for step in range(1, interval + 1):
            factor = 1 / interval
            u = (1 - step * factor) * U_curr + (step * factor) * U_next
            v = (1 - step * factor) * V_curr + (step * factor) * V_next
            # prompt composition
            c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
            prompt.append(c)
        prompt = torch.concatenate(prompt, dim=0)
        torch.cuda.synchronize()
        _t_int1 = time.perf_counter()
        interp_ms = (_t_int1 - _t_int0) * 1000.0
        total_interp_ms += interp_ms

        # generating frames from prompts
        if Generator_RT.prev_frame is None:
            # initialize for the first frame
            Generator_RT.prev_frame = torch.load(os.path.join(prompt_dir, 'init.pth'), weights_only=True)
            c0 = (U_curr @ V_curr / np.sqrt(rank)).unsqueeze(dim=0)

            # ---- Warmup 首帧计时（不影响原 FPS 逻辑）----
            _t_wu0 = time.perf_counter()
            images0 = Generator_RT.generate(c0)
            images0 = Generator_RT.normalization(images0, to_numpy=True)
            torch.cuda.synchronize()
            _t_wu1 = time.perf_counter()
            print(f"[Warmup-Init] gen_ms={(_t_wu1 - _t_wu0) * 1000.0:.2f} ms")

            if result_frames is None:
                result_frames = images0[0:1, ...]
            total_frames += 1  # 首帧计入总帧数

        # ------------------ 原计时起点（保持原逻辑） ------------------
        torch.cuda.synchronize()
        t_begin = time.time()

        # ------------------ 新增端到端生成计时（perf_counter） ------------------
        _t_gen0 = time.perf_counter()
        images = Generator_RT.generate(prompt)
        images = Generator_RT.normalization(images, to_numpy=True)
        torch.cuda.synchronize()
        _t_gen1 = time.perf_counter()
        gen_ms = (_t_gen1 - _t_gen0) * 1000.0
        total_gen_e2e_ms += gen_ms

        # 从 Generator_RT._last_times 取分段耗时并累计
        total_denoise_ms += Generator_RT._last_times.get('denoise_ms', 0.0)
        total_decode_ms += Generator_RT._last_times.get('decode_ms', 0.0)
        total_normalize_ms += Generator_RT._last_times.get('normalize_ms', 0.0)

        # ------------------ 原有 FPS 统计（保持不变） ------------------
        torch.cuda.synchronize()
        t_end = time.time()
        if speed_warm_up:
            # generation speed in FPS
            fps = 1 / (t_end - t_begin) * batch
            generation_speed.append(fps)
            print('Generation Speed: {} FPS'.format(fps))
        else:
            # the first batch is used for warming up
            speed_warm_up = True

        # ------------------ 控制台打印：该批详细时延 ------------------
        num_generated = images.shape[0]
        total_frames += num_generated
        fps_batch = 1000.0 / gen_ms if gen_ms > 0 else 0.0
        fps_frame = fps_batch * num_generated
        print(f"[Iter-{iter_idx}] interval={interval}  batch={batch} | "
              f"dequant={dequant_ms:.2f} ms  interp={interp_ms:.2f} ms  "
              f"denoise={Generator_RT._last_times['denoise_ms']:.2f} ms  "
              f"decode={Generator_RT._last_times['decode_ms']:.2f} ms  "
              f"normalize={Generator_RT._last_times['normalize_ms']:.2f} ms  "
              f"generate(E2E)={gen_ms:.2f} ms  | "
              f"FPS_batch={fps_batch:.2f}  FPS_frame={fps_frame:.2f} (frames={num_generated})")

        # 追加到结果缓存
        if result_frames is None:
            result_frames = images
        else:
            result_frames = np.append(result_frames, images, axis=0)

        # 进入下一个区间
        iter_idx += 1

    # ====== 保存并统计总数据量（PNG 文件总大小）======
    average_speed = int(np.mean(generation_speed)) if len(generation_speed) > 0 else 0
    for i in range(result_frames.shape[0]):
        image = result_frames[i, ...]
        image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)
        # Save the generated frames
        out_path = os.path.join(result_dir, '{:05d}.png'.format(i))
        cv.imwrite(out_path, image)
        # 累加文件大小
        try:
            total_bytes_saved += os.path.getsize(out_path)
        except OSError:
            pass
        if args.visualize:
            # Visualize the generated frames（保持原逻辑）
            cv.putText(image, 'Generation Speed {} FPS'.format(average_speed), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 2)
            cv.imshow('Real-time Generation', image)
            cv.waitKey(10)

    # ====== 汇总统计（控制台打印）======
    total_sec = total_gen_e2e_ms / 1000.0 if total_gen_e2e_ms > 0 else 0.0
    overall_fps = (total_frames / total_sec) if total_sec > 0 else 0.0
    mb = total_bytes_saved / (1024.0 * 1024.0)

    # 平均每帧耗时（将区间成本均摊到帧，便于横向对比）
    if total_frames > 0:
        avg_dequant_per_frame = total_dequant_ms / total_frames
        avg_interp_per_frame = total_interp_ms / total_frames
        avg_denoise_per_frame = total_denoise_ms / total_frames
        avg_decode_per_frame = total_decode_ms / total_frames
        avg_normalize_per_frame = total_normalize_ms / total_frames
        avg_e2e_per_frame = total_gen_e2e_ms / total_frames
        avg_kb_per_frame_saved = (total_bytes_saved / 1024.0) / total_frames
    else:
        avg_dequant_per_frame = avg_interp_per_frame = avg_denoise_per_frame = 0.0
        avg_decode_per_frame = avg_normalize_per_frame = avg_e2e_per_frame = 0.0
        avg_kb_per_frame_saved = 0.0

    # 理论平均每帧数据量（两种口径）
    # A) 仅按区间帧数均摊（Σ interval）
    theoretical_kb_per_frame_interval = (total_theoretical_bytes / 1024.0) / max(1, total_interval_frames)
    # B) 按所有生成帧均摊（Σ interval + warmup 首帧）
    theoretical_kb_per_frame_all = (total_theoretical_bytes / 1024.0) / max(1, total_frames)

    print("\n========== SUMMARY ==========")
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
    print("6) Average data per frame (theoretical):")
    print(f"   - per interval frames (Σ interval): {theoretical_kb_per_frame_interval:.2f} KB/frame")
    print(f"   - over ALL frames (including warmup): {theoretical_kb_per_frame_all:.2f} KB/frame")
    print("================================\n")
