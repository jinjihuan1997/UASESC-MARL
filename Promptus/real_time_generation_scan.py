import time
import torch
import numpy as np
import cv2 as cv
import os
import re
import glob
import argparse
import math
from cuda import cudart
from quantization import QParam
from polygraphy import cuda
from tensorrt_acceleration import Engine

"""
Scheme-A: Steady-state Device Max FPS (denoise+decode only)
python real_time_generation_scan.py -results_root data/sky/results -batch 10
"""


# ==========================================
# 1. 硬件设备检测与打印
# ==========================================
def print_device_info():
    print("\n==========================================")
    if torch.cuda.is_available():
        device_cnt = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"[System] Mode: GPU Acceleration")
        print(f"[System] Device Count: {device_cnt}")
        print(f"[System] Current Device: {device_name}")
        print(f"[System] CUDA Version: {torch.version.cuda}")
    else:
        print(f"[System] Mode: CPU Only (Warning: TensorRT requires NVIDIA GPU)")
    print("==========================================\n")


# ==========================================
# 2. Real-time generation engine
# ==========================================
class Generator:
    def __init__(self, batch=1, device='cuda:0'):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. TensorRT path requires NVIDIA GPU.")

        self.cuda_stream = cuda.Stream()
        self.use_cuda_graph = False
        self.batch = int(batch)
        self.device = device

        # state
        self.prev_frame = None

        # fixed tensors
        self.noise = self.seeded_randn(shape=(1, 4, 64, 64), seed=88).to(self.device)
        self.sigma = torch.tensor([0.05], dtype=torch.float32, device=self.device)

        self.timesteps = torch.tensor([999], dtype=torch.long, device=self.device).repeat(self.batch)
        self.c_in = torch.tensor([[[[0.0683]]]], dtype=torch.float32, device=self.device)
        self.c_out = torch.tensor([[[[-14.6146]]]], dtype=torch.float32, device=self.device)

        self.denoise_engine = None
        self.decoder_engine = None

        # per-call timing
        self._last_times = {'denoise_ms': 0.0, 'decode_ms': 0.0}

        self.denoise_engine_load()
        self.decoder_engine_load()

    def reset_state(self):
        self.prev_frame = None

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
            workspace_size=max_workspace_size
        )
        return engine

    def denoise_engine_build(self):
        print('Start building the denoise engine for this machine.')
        engine_path = f'engine/denoise_batch_{self.batch}.engine'
        onnx_path = f'engine/denoise_batch_{self.batch}.onnx'
        input_profile = {
            'x': [(self.batch, 4, 64, 64), (self.batch, 4, 64, 64), (self.batch, 4, 64, 64)],
            'timesteps': [(self.batch,), (self.batch,), (self.batch,)],
            'context': [(self.batch, 77, 1024), (self.batch, 77, 1024), (self.batch, 77, 1024)]
        }
        self.build_engine(input_profile, onnx_path, engine_path)
        print('Build completed successfully.')

    def denoise_engine_load(self):
        engine_path = f'engine/denoise_batch_{self.batch}.engine'
        if not os.path.exists(engine_path):
            print('The denoise engine is not found.')
            self.denoise_engine_build()

        self.denoise_engine = Engine(engine_path)
        self.denoise_engine.load()
        self.denoise_engine.activate()
        self.denoise_engine.allocate_buffers(
            shape_dict={"x": [self.batch, 4, 64, 64], "timesteps": [self.batch], "context": [self.batch, 77, 1024]},
            device=self.device
        )

    def decoder_engine_build(self):
        print('Start building the decoder engine for this machine.')
        engine_path = f'engine/decoder_batch_{self.batch}.engine'
        onnx_path = f'engine/decoder_batch_{self.batch}.onnx'
        input_profile = {
            'latent': [(self.batch, 4, 64, 64), (self.batch, 4, 64, 64), (self.batch, 4, 64, 64)]
        }
        self.build_engine(input_profile, onnx_path, engine_path)
        print('Build completed successfully.')

    def decoder_engine_load(self):
        engine_path = f'engine/decoder_batch_{self.batch}.engine'
        if not os.path.exists(engine_path):
            print('The decoder engine is not found.')
            self.decoder_engine_build()

        self.decoder_engine = Engine(engine_path)
        self.decoder_engine.load()
        self.decoder_engine.activate()
        self.decoder_engine.allocate_buffers(shape_dict={"latent": [self.batch, 4, 64, 64]}, device=self.device)

    def run_denoise_engine(self, x, timesteps, context):
        # denoise timing (ms) - includes needed sync for stable wall-time measurement
        t0 = time.perf_counter()
        output = self.denoise_engine.infer(
            {"x": x, "timesteps": timesteps, "context": context},
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph
        )
        torch.cuda.synchronize()
        self._last_times['denoise_ms'] = (time.perf_counter() - t0) * 1000.0
        return output['out']

    def run_decoder_engine(self, latent):
        # decode timing (ms)
        t0 = time.perf_counter()
        output = self.decoder_engine.infer(
            {"latent": latent},
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph
        )
        torch.cuda.synchronize()
        self._last_times['decode_ms'] = (time.perf_counter() - t0) * 1000.0
        return output['images']

    @staticmethod
    def seeded_randn(shape, seed):
        return torch.from_numpy(np.random.RandomState(seed).randn(*shape)).to(dtype=torch.float32, device="cuda")

    def add_noise(self, prev_frame):
        noised = (prev_frame * self.sigma + self.noise * (1 - self.sigma)).detach()
        return noised.repeat(self.batch, 1, 1, 1)

    @torch.no_grad()
    def generate_with_timing(self, cond, measure=True):
        """
        Returns:
            images: torch.Tensor (effective frames, C,H,W) on GPU
            infer_ms: sum of denoise_ms + decode_ms across all batches (only if measure else 0)
        Note:
            - Only counts TensorRT infer time (denoise+decode).
            - No normalization/cpu copy here.
        """
        if cond.numel() == 0:
            return torch.empty((0,), device=self.device), 0.0

        if self.prev_frame is None:
            raise RuntimeError("prev_frame is None. Call warmup first to init state.")

        # ensure cond on device (H2D outside if caller wants)
        if cond.device.type != 'cuda':
            cond = cond.to(self.device)

        total_frames_needed = int(cond.shape[0])
        all_generated_images = []
        infer_ms = 0.0

        for start_idx in range(0, total_frames_needed, self.batch):
            end_idx = min(start_idx + self.batch, total_frames_needed)
            current_cond_chunk = cond[start_idx:end_idx]
            actual_batch_size = int(current_cond_chunk.shape[0])

            # pad to full batch (device-side)
            if actual_batch_size < self.batch:
                pad_size = self.batch - actual_batch_size
                padding = current_cond_chunk[-1:].repeat(pad_size, 1, 1)
                batch_cond = torch.cat([current_cond_chunk, padding], dim=0)
            else:
                batch_cond = current_cond_chunk

            # denoise+decode
            z = self.add_noise(self.prev_frame)
            latent_noise = self.run_denoise_engine(z, self.timesteps, batch_cond)
            latent = latent_noise * self.c_out + z / self.c_in
            images = self.run_decoder_engine(latent)

            if measure:
                infer_ms += (self._last_times['denoise_ms'] + self._last_times['decode_ms'])

            # update state using last *effective* sample
            self.prev_frame = latent[actual_batch_size - 1: actual_batch_size, ...]

            # keep only effective images
            all_generated_images.append(images[:actual_batch_size, ...])

        out = torch.cat(all_generated_images, dim=0) if all_generated_images else torch.empty((0,), device=self.device)
        return out, infer_ms

    @torch.no_grad()
    def normalization_to_numpy(self, images_tensor):
        # not used in timing for scheme A
        images_tensor = torch.clamp((images_tensor + 1.0) / 2.0, min=0.0, max=1.0)
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        images_tensor = (images_tensor * 255).byte()
        return images_tensor.detach().cpu().numpy()


# ==========================================
# 3. Prompt utilities
# ==========================================
def prompt_bytes(p):
    b = 0
    for k in ['U', 'V', 'U_scale', 'U_zero_point', 'V_scale', 'V_zero_point']:
        t = p[k]
        b += t.element_size() * t.numel()
    return b


def process_folder(prompt_dir, generator: Generator, args):
    print(f"\n==========================================")
    print(f" Processing: {prompt_dir}")
    print(f"==========================================")

    generator.reset_state()
    result_frames = None

    prompts = sorted(glob.glob(os.path.join(prompt_dir, 'frame_*.prompt')))
    if len(prompts) < 2:
        print(f"Warning: Found {len(prompts)} prompts. Skipping.")
        return

    print(f"[DEBUG] Found {len(prompts)} prompt files.")

    # Scheme-A accumulators (steady-state only)
    total_measured_infer_ms = 0.0
    total_measured_computed_frames = 0

    # optional: stats
    total_effective_frames = 0
    total_theoretical_bytes = 0
    iter_idx = 0

    # helper: dequant
    def deq(t, s, z):
        qp = QParam(num_bits=8)
        qp.scale = s
        qp.zero_point = z
        return qp.dequantize_tensor(t)

    # iterate consecutive keyframes
    for prompt_curr, prompt_next in zip(prompts[::], prompts[1::]):
        match_curr = re.search(r'frame_(\d+)\.prompt', prompt_curr)
        match_next = re.search(r'frame_(\d+)\.prompt', prompt_next)
        if not match_curr or not match_next:
            continue

        id_curr = int(match_curr.group(1))
        id_next = int(match_next.group(1))
        interval = id_next - id_curr
        if interval <= 0:
            continue

        prompt_curr_obj = torch.load(prompt_curr, weights_only=True)
        prompt_next_obj = torch.load(prompt_next, weights_only=True)

        # bytes stats (not affecting scheme A timing)
        send_bytes = prompt_bytes(prompt_next_obj)
        if iter_idx == 0:
            send_bytes += prompt_bytes(prompt_curr_obj)
        total_theoretical_bytes += send_bytes
        avg_bytes = send_bytes / max(1, interval)
        print(f"[Iter-{iter_idx}] interval={interval} | theoretical payload/frame ≈ {avg_bytes / 1024:.2f} KB")

        # Dequant & Interp (outside scheme A timing)
        U_curr = deq(prompt_curr_obj['U'], prompt_curr_obj['U_scale'], prompt_curr_obj['U_zero_point'])
        V_curr = deq(prompt_curr_obj['V'], prompt_curr_obj['V_scale'], prompt_curr_obj['V_zero_point'])
        U_next = deq(prompt_next_obj['U'], prompt_next_obj['U_scale'], prompt_next_obj['U_zero_point'])
        V_next = deq(prompt_next_obj['V'], prompt_next_obj['V_scale'], prompt_next_obj['V_zero_point'])

        rank = int(U_curr.shape[1])

        # build per-frame contexts (CPU)
        prompt_list = []
        for step in range(1, interval + 1):
            alpha = step / interval
            u = (1 - alpha) * U_curr + alpha * U_next
            v = (1 - alpha) * V_curr + alpha * V_next
            c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)  # (1,77,1024)
            prompt_list.append(c)

        prompt = torch.cat(prompt_list, dim=0)  # (interval,77,1024)

        # ==========================
        # Warmup (NOT counted)
        # ==========================
        if generator.prev_frame is None:
            init_pth = os.path.join(prompt_dir, 'init.pth')
            if os.path.exists(init_pth):
                generator.prev_frame = torch.load(init_pth, weights_only=True).to(generator.device)
            else:
                generator.prev_frame = torch.zeros((1, 4, 64, 64), dtype=torch.float32, device=generator.device)

            c0 = (U_curr @ V_curr / np.sqrt(rank)).unsqueeze(dim=0)  # (1,77,1024)
            c0 = c0.to(generator.device)  # H2D outside measurement

            # run warmup once, ignore timing & frames
            _imgs0, _ = generator.generate_with_timing(c0, measure=False)

            # optional: store first frame (outside timing)
            if result_frames is None:
                imgs0_np = generator.normalization_to_numpy(_imgs0[:1, ...])
                result_frames = imgs0_np  # (1,H,W,C)
                total_effective_frames += 1  # user-view only

        # ==========================
        # Measured generation (Scheme A)
        # ==========================
        prompt_gpu = prompt.to(generator.device)  # H2D outside measurement

        imgs, infer_ms = generator.generate_with_timing(prompt_gpu, measure=True)

        # scheme A: computed frames are batch-aligned
        num_generated = int(imgs.shape[0])  # effective frames
        batches_run = math.ceil(num_generated / args.batch)
        computed_frames = batches_run * args.batch

        total_measured_infer_ms += infer_ms
        total_measured_computed_frames += computed_frames
        total_effective_frames += num_generated

        # print scheme A throughput for this iter
        fps_iter = (computed_frames * 1000.0) / infer_ms if infer_ms > 0 else 0.0
        print(f"   [Scheme-A] infer(denoise+decode)={infer_ms:.2f} ms | Effective={num_generated} | Computed={computed_frames}")
        print(f"   => Iter Device Max Throughput: {fps_iter:.2f} FPS")

        # optional: save frames (outside scheme A timing)
        imgs_np = generator.normalization_to_numpy(imgs)
        if result_frames is None:
            result_frames = imgs_np
        else:
            result_frames = np.append(result_frames, imgs_np, axis=0)

        iter_idx += 1

    # Save PNGs (outside scheme A timing)
    if result_frames is not None:
        print(f"[DEBUG] Saving {result_frames.shape[0]} images...")
        for i in range(result_frames.shape[0]):
            out_path = os.path.join(prompt_dir, f'{i:05d}.png')
            cv.imwrite(out_path, result_frames[i, :, :, ::-1])

    # ==========================
    # Summary (Scheme A)
    # ==========================
    print("\n========== SUMMARY (Scheme-A) ==========")
    print(f"Total Effective Frames (User View, incl. warmup first frame): {total_effective_frames}")
    print(f"Total GPU Computed Frames (Measured, batch-aligned): {total_measured_computed_frames}")
    print(f"Total Infer Time (Measured, denoise+decode only): {total_measured_infer_ms:.2f} ms")

    theoretical_kb_per_frame = (total_theoretical_bytes / 1024.0) / max(1, total_effective_frames)
    print(f"Theoretical payload/frame: {theoretical_kb_per_frame:.2f} KB/frame")

    if total_measured_computed_frames > 0 and total_measured_infer_ms > 0:
        max_fps_a = total_measured_computed_frames / (total_measured_infer_ms / 1000.0)
        print(f"[Scheme-A] Device Max Throughput FPS: {max_fps_a:.2f}")

        enc_time_path = os.path.join(prompt_dir, 'enc_fps.txt')
        try:
            with open(enc_time_path, 'w') as f:
                f.write(f"{max_fps_a:.6f}")
            print(f"[SUCCESS] Saved Scheme-A FPS to: {enc_time_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save enc_fps.txt: {e}")
    else:
        print(f"[WARNING] No measured frames for Scheme-A FPS.")

    print("========================================\n")


# ==========================================
# 4. Main
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-results_root', type=str, required=True, help="Path to results root or a single rank*_interval* folder")
    parser.add_argument('-batch', type=int, default=10)
    args = parser.parse_args()

    print_device_info()
    gen = Generator(batch=args.batch, device='cuda:0')

    if not os.path.isdir(args.results_root):
        raise RuntimeError(f"Not found: {args.results_root}")

    subfolders = []
    input_path = os.path.normpath(args.results_root)
    base_name = os.path.basename(input_path)

    # If user points directly to a rank*_interval* folder (or contains frame_*.prompt)
    if len(glob.glob(os.path.join(input_path, "frame_*.prompt"))) > 0 or ('rank' in base_name and 'interval' in base_name):
        print(f"[INFO] Target: {base_name}")
        subfolders.append(input_path)
    else:
        print(f"[INFO] Scanning subdirectories...")
        for d in os.listdir(input_path):
            p = os.path.join(input_path, d)
            if os.path.isdir(p) and ('rank' in d) and ('interval' in d):
                subfolders.append(p)

    subfolders = sorted(subfolders)
    print(f"Found {len(subfolders)} folders.")
    for folder in subfolders:
        process_folder(folder, gen, args)

    print("\nAll folders processed.")
