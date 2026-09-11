#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run examples:
python measure_tc.py --frames_dir data/sky --out_csv results_tc/tc_eval_sky.csv
python measure_tc.py --frames_dir data/sky --encode_mode cbr --codecs h264 h265 --bitrate_start 50 --bitrate_end 500 --bitrate_step 50 --cbr_preset veryfast --rate_safety 1.0 --max_retries 1 --out_csv results_tc/tc_eval_sky.csv

Measure TC and export CSV columns:
sequence, codec, bitrate_kbps_budget, target_bitrate_kbps_used_int, required_kbps_at_tx, lpips_mean.
"""

import os
import csv
import shutil
import argparse
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import cv2 as cv

try:
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_FPS = 16


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def log(msg: str) -> None:
    print(f"[measure_tc] {msg}", flush=True)


def run_cmd(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    try:
        if capture:
            p = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        else:
            p = subprocess.run(cmd, check=check)
        return p
    except FileNotFoundError:
        raise RuntimeError(f"Command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        if capture and hasattr(e, "stdout") and e.stdout:
            raise RuntimeError(f"Command failed (code {e.returncode}): {' '.join(cmd)}\n--- output ---\n{e.stdout}")
        raise RuntimeError(f"Command failed (code {e.returncode}): {' '.join(cmd)}")


def pick_ffmpeg(prefer: Optional[List[str]] = None) -> str:
    if prefer is None:
        prefer = ["/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg", "/bin/ffmpeg"]
    for p in prefer:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("ffmpeg not found. Install ffmpeg or build it.")


def build_lpips_model(device: str = "cuda") -> Tuple[Optional["LearnedPerceptualImagePatchSimilarity"], str]:
    if not HAS_LPIPS:
        log("torchmetrics not installed, skip LPIPS.")
        return None, "cpu"
    if device == "cuda":
        if ("torch" not in globals()) or (not torch.cuda.is_available()):
            log("CUDA not available, fall back to CPU for LPIPS.")
            device = "cpu"
    log(f"Loading LPIPS(vgg, torchmetrics) on {device}...")
    model = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    return model, device


def compute_lpips_mean(
        frames_ref: List[np.ndarray],
        frames_dec: List[np.ndarray],
        model: Optional["LearnedPerceptualImagePatchSimilarity"],
        device: str = "cuda",
) -> Optional[float]:
    if model is None or not HAS_LPIPS:
        return None
    if device == "cuda" and (("torch" not in globals()) or (not torch.cuda.is_available())):
        device = "cpu"

    n = min(len(frames_ref), len(frames_dec))
    if n <= 0:
        return None

    scores: List[float] = []
    for i in range(n):
        ref = frames_ref[i]
        dec = frames_dec[i]
        ref_rgb = cv.cvtColor(ref, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dec_rgb = cv.cvtColor(dec, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        dec_t = torch.from_numpy(dec_rgb).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        dec_t = torch.clamp(dec_t, min=-1.0, max=1.0)
        with torch.no_grad():
            d = float(model(dec_t, ref_t).item())
        scores.append(d)

    return float(np.mean(scores))


def load_frames_from_dir(frames_dir: str, pattern: str = "%05d.png", max_frames: Optional[int] = None) -> List[
    np.ndarray]:
    frames: List[np.ndarray] = []
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break

        fpath = os.path.join(frames_dir, pattern % idx)
        if not os.path.isfile(fpath):
            break
        img = cv.imread(fpath, cv.IMREAD_COLOR)
        if img is None:
            log(f"Warning: failed to read {fpath}, stop.")
            break
        frames.append(img)
        idx += 1
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir} with pattern {pattern}")
    log(f"Loaded {len(frames)} reference frames from {frames_dir}")
    return frames


def decode_video_to_frames(video_path: str) -> List[np.ndarray]:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _ffmpeg_codec_name(codec: str) -> str:
    c = codec.lower()
    if c in ("h264", "libx264"):
        return "libx264"
    if c in ("h265", "hevc", "libx265"):
        return "libx265"
    raise ValueError(f"Unsupported codec: {codec}")


def encode_sequence_with_ffmpeg(
        ffmpeg_path: str,
        frames_dir: str,
        fps_video: float,
        codec: str,
        out_video_path: str,
        bitrate_kbps: Optional[int] = None,
        crf: Optional[int] = None,
        preset: str = "veryfast",
        start_number: int = 0,
        frame_pattern: str = "%05d.png",
        max_frames: Optional[int] = None,
) -> str:
    input_pattern = os.path.join(frames_dir, frame_pattern).replace("\\", "/")
    out_video_path = out_video_path.replace("\\", "/")
    vcodec = _ffmpeg_codec_name(codec)
    if not out_video_path.lower().endswith(".mp4"):
        out_video_path += ".mp4"

    cmd = [
        ffmpeg_path, "-y", "-loglevel", "error",
        "-framerate", str(float(fps_video)),
        "-start_number", str(int(start_number)),
        "-i", input_pattern,
    ]

    if max_frames is not None:
        cmd += ["-frames:v", str(int(max_frames))]

    cmd += [
        "-c:v", vcodec,
        "-preset", preset,
        "-pix_fmt", "yuv420p",
    ]

    if crf is not None:
        cmd += ["-crf", str(int(crf))]
    else:
        if bitrate_kbps is None:
            raise ValueError("bitrate_kbps is required when crf is None (CBR mode).")
        bitrate_kbps = max(1, int(round(bitrate_kbps)))
        buf_kbps = max(2, 2 * bitrate_kbps)
        cmd += [
            "-b:v", f"{bitrate_kbps}k",
            "-minrate", f"{bitrate_kbps}k",
            "-maxrate", f"{bitrate_kbps}k",
            "-bufsize", f"{buf_kbps}k",
        ]
        if vcodec == "libx265":
            cmd += ["-x265-params", f"nal-hrd=cbr:vbv-maxrate={bitrate_kbps}:vbv-bufsize={buf_kbps}"]
        elif vcodec == "libx264":
            cmd += ["-x264-params", f"nal-hrd=cbr:vbv-maxrate={bitrate_kbps}:vbv-bufsize={buf_kbps}"]
    cmd += [out_video_path]

    run_cmd(cmd)
    return out_video_path


def compute_required_kbps_at_tx(file_bytes: int, n_frames: int, tx_fps: float) -> float:
    if n_frames <= 0 or tx_fps <= 0: return float("inf")
    total_bits = float(file_bytes) * 8.0
    duration_tx = float(n_frames) / float(tx_fps)
    return (total_bits / duration_tx) / 1000.0


def try_encode_under_budget(
        ffmpeg_path: str,
        frames_dir: str,
        fps_video: float,
        codec: str,
        out_video_path: str,
        n_frames: int,
        tx_fps: float,
        budget_kbps: float,
        rate_safety: float,
        max_retries: int,
        cbr_preset: str,
        max_frames: Optional[int] = None,
) -> Tuple[str, int, float]:
    budget_kbps = max(1e-6, float(budget_kbps))
    target = max(1, int(np.floor(budget_kbps * float(rate_safety))))
    tries = 0
    while True:
        tries += 1
        outp = encode_sequence_with_ffmpeg(
            ffmpeg_path=ffmpeg_path,
            frames_dir=frames_dir,
            fps_video=fps_video,
            codec=codec,
            out_video_path=out_video_path,
            bitrate_kbps=target,
            preset=cbr_preset,
            max_frames=max_frames
        )
        file_bytes = os.path.getsize(outp)
        required_kbps = compute_required_kbps_at_tx(file_bytes, n_frames, tx_fps)

        if required_kbps <= budget_kbps * 1.0001 or tries >= max_retries:
            return outp, target, required_kbps

        scale = budget_kbps / max(required_kbps, 1e-9)
        target_new = int(np.floor(max(1.0, float(target) * scale * float(rate_safety))))
        if target_new >= target: target_new = max(1, target - 1)
        target = target_new


def main() -> None:
    parser = argparse.ArgumentParser(description="TC measurement with fixed FPS=16 and LPIPS(VGG).")
    parser.add_argument("--frames_dir", type=str, required=True, help="Input frames dir")
    parser.add_argument("--codecs", nargs="+", default=["h264", "h265"], choices=["h264", "h265"])
    parser.add_argument("--encode_mode", type=str, default="cbr", choices=["sc_like", "cbr"],
                        help="sc_like: same CRF profile as measure_sc; cbr: target bitrate sweep")
    parser.add_argument("--bitrate_start", type=int, default=500, help="Start bitrate (kbps), cbr mode only")
    parser.add_argument("--bitrate_end", type=int, default=500, help="End bitrate (kbps, inclusive), cbr mode only")
    parser.add_argument("--bitrate_step", type=int, default=1, help="Step size (kbps)")
    parser.add_argument("--sc_crf", type=int, default=23, help="CRF when encode_mode=sc_like")
    parser.add_argument("--sc_preset", type=str, default="veryfast", help="preset when encode_mode=sc_like")
    parser.add_argument("--cbr_preset", type=str, default="veryfast", help="preset when encode_mode=cbr")

    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--lpips_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--rate_safety", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=1)

    args = parser.parse_args()

    if not os.path.isdir(args.frames_dir):
        raise RuntimeError(f"frames_dir not found: {args.frames_dir}")

    bitrate_kbps_list: List[int] = []
    if args.encode_mode == "cbr":
        if args.bitrate_start is None or args.bitrate_end is None:
            parser.error("--bitrate_start and --bitrate_end are required when --encode_mode cbr")
        curr_b = int(args.bitrate_start)
        while curr_b <= int(args.bitrate_end):
            bitrate_kbps_list.append(curr_b)
            curr_b += int(args.bitrate_step)
        if not bitrate_kbps_list:
            parser.error("No bitrate point generated. Check bitrate_start/end/step.")
    else:
        bitrate_kbps_list = [0]

    current_fps = int(FIXED_FPS)
    current_tx_fps = float(current_fps)
    effective_max_frames = int(FIXED_FPS)
    log(f"Using fixed FPS: {current_fps}, encode_mode={args.encode_mode}")
    log(f"Using max_frames aligned to FPS: {effective_max_frames}")
    if args.encode_mode == "cbr":
        log(f"Generated Bitrate List: {bitrate_kbps_list}")
    else:
        log(f"Using SC-like profile: codec in {args.codecs}, preset={args.sc_preset}, crf={args.sc_crf}")

    seq_name = os.path.basename(os.path.normpath(args.frames_dir))
    ffmpeg_path = pick_ffmpeg()
    log(f"Using ffmpeg: {ffmpeg_path}")

    frames_ref = load_frames_from_dir(args.frames_dir, pattern="%05d.png", max_frames=effective_max_frames)
    n_frames = len(frames_ref)

    out_csv = args.out_csv
    if out_csv is None:
        out_dir = os.path.join(THIS_DIR, "results_tc")
        ensure_dir(out_dir)
        out_csv = os.path.join(out_dir, f"tc_eval_{seq_name}.csv")

    out_video_base_dir = os.path.join(THIS_DIR, "results_tc", "videos", "encoded")
    ensure_dir(out_video_base_dir)

    lpips_device = args.lpips_device
    lpips_model, lpips_device = build_lpips_model(lpips_device)
    if lpips_model is None:
        raise RuntimeError("LPIPS dependency missing. Please install `torchmetrics` in your environment.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f_csv:
        fieldnames = [
            "sequence",
            "codec",
            "bitrate_kbps_budget",
            "target_bitrate_kbps_used_int",
            "required_kbps_at_tx",
            "lpips_mean",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        def fmt(val, precision=4):
            if val is None or val == "" or (isinstance(val, float) and np.isnan(val)): return ""
            return f"{float(val):.{precision}f}"

        for current_codec in args.codecs:
            log(f"=== Processing Codec: {current_codec.upper()} (FPS={current_fps}) ===")
            out_video_dir = os.path.join(out_video_base_dir, args.encode_mode, current_codec, seq_name, f"fps_{current_fps}")
            ensure_dir(out_video_dir)

            for b in bitrate_kbps_list:
                budget_kbps: Optional[float] = None
                target_int: Optional[int] = None
                if args.encode_mode == "cbr":
                    budget_kbps = float(b)
                    log(f"--- fps={current_fps}, codec={current_codec}, budget={budget_kbps:.2f} kbps ---")
                    out_video_path = os.path.join(
                        out_video_dir, f"{seq_name}_{current_codec}_{int(round(budget_kbps))}kbps.mp4"
                    )
                    out_video_path, target_int, req_kbps = try_encode_under_budget(
                        ffmpeg_path=ffmpeg_path,
                        frames_dir=args.frames_dir,
                        fps_video=float(current_fps),
                        codec=current_codec,
                        out_video_path=out_video_path,
                        n_frames=n_frames,
                        tx_fps=current_tx_fps,
                        budget_kbps=budget_kbps,
                        rate_safety=args.rate_safety,
                        max_retries=args.max_retries,
                        cbr_preset=args.cbr_preset,
                        max_frames=effective_max_frames
                    )
                else:
                    log(
                        f"--- fps={current_fps}, codec={current_codec}, "
                        f"sc_profile=preset:{args.sc_preset},crf:{args.sc_crf} ---"
                    )
                    out_video_path = os.path.join(
                        out_video_dir, f"{seq_name}_{current_codec}_crf{int(args.sc_crf)}.mp4"
                    )
                    out_video_path = encode_sequence_with_ffmpeg(
                        ffmpeg_path=ffmpeg_path,
                        frames_dir=args.frames_dir,
                        fps_video=float(current_fps),
                        codec=current_codec,
                        out_video_path=out_video_path,
                        crf=args.sc_crf,
                        preset=args.sc_preset,
                        max_frames=effective_max_frames,
                    )
                    file_bytes_sc = os.path.getsize(out_video_path)
                    req_kbps = compute_required_kbps_at_tx(file_bytes_sc, n_frames, current_tx_fps)

                file_bytes = os.path.getsize(out_video_path)
                frames_dec = decode_video_to_frames(out_video_path)
                n_common = min(len(frames_ref), len(frames_dec))
                if n_common <= 0:
                    continue

                frames_ref_subset = frames_ref[:n_common]
                frames_dec_subset = frames_dec[:n_common]

                lpips_mean = None
                if lpips_model:
                    lpips_mean = compute_lpips_mean(frames_ref_subset, frames_dec_subset, lpips_model, lpips_device)

                writer.writerow(dict(
                    sequence=seq_name,
                    codec=current_codec,
                    bitrate_kbps_budget=fmt(budget_kbps, 2),
                    target_bitrate_kbps_used_int=(int(target_int) if target_int is not None else ""),
                    required_kbps_at_tx=fmt(req_kbps, 4),
                    lpips_mean=fmt(lpips_mean, 4),
                ))

                lp_str = fmt(lpips_mean, 4) if lpips_mean is not None else "N/A"
                log(
                    f"Done | FPS={current_fps}, Codec={current_codec}, RealKbps={fmt(req_kbps, 2)}, "
                    f"LPIPS={lp_str}"
                )

    log(f"Saved Evaluated CSV: {out_csv}")


if __name__ == "__main__":
    main()
