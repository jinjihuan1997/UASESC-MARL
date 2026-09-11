#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run examples:
python measure_sc.py --frames_dir data/sky --out_csv results_sc/sc_eval_sky.csv

Measure SC profiles and export CSV columns:
rank, interval, lpips, sl_bits.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")
DEFAULT_PROMPT_U = 77
DEFAULT_PROMPT_V = 1024

FIXED_RI_SELECTION: List[Tuple[int, int]] = [
    (2, 5),
    (4, 5),
    (8, 5),
    (16, 5),
    (24, 5),
    (2, 10),
    (4, 10),
    (8, 10),
    (16, 10),
    (24, 10),
    (2, 15),
    (4, 15),
    (8, 15),
    (16, 15),
    (24, 15),
]
FIXED_RI_SET = set(FIXED_RI_SELECTION)


def parse_rank_interval(name: str) -> Optional[Tuple[int, int]]:
    """Parse rank and interval from a directory name.

    Supports patterns like:
      rank64_interval4
      rank_64_interval_4
      R64_I4
      r64_i4
    """
    s = name

    m_rank = re.search(r"(?:rank|r)\s*[_\-]?\s*(\d+)", s, flags=re.IGNORECASE)
    m_intv = re.search(r"(?:interval|i)\s*[_\-]?\s*(\d+)", s, flags=re.IGNORECASE)

    if m_rank is None:
        m_rank = re.search(r"\bR\s*[_\-]?\s*(\d+)\b", s, flags=re.IGNORECASE)
    if m_intv is None:
        m_intv = re.search(r"\bI\s*[_\-]?\s*(\d+)\b", s, flags=re.IGNORECASE)

    if (m_rank is None) or (m_intv is None):
        return None

    rank = int(m_rank.group(1))
    interval = int(m_intv.group(1))
    if rank <= 0 or interval <= 0:
        return None
    return rank, interval


def find_video_file(cfg_dir: str) -> Optional[str]:
    """Find a recon video file under cfg_dir (recursive)."""
    for root, _, files in os.walk(cfg_dir):
        for fn in files:
            if fn.lower().endswith(_VIDEO_EXTS):
                return os.path.join(root, fn)
    return None


def find_frames_root(cfg_dir: str) -> Optional[str]:
    """Find a directory under cfg_dir that contains numbered png frames."""
    for root, dirs, files in os.walk(cfg_dir):
        pngs = [f for f in files if f.lower().endswith(".png")]
        if not pngs:
            continue
        has_numbered = any(re.fullmatch(r"\d{4,6}\.png", f) for f in pngs)
        if has_numbered or len(pngs) >= 10:
            return root
    return None


def frames_pattern_from_root(fr_root: str) -> str:
    """Return ffmpeg input pattern for numbered png frames.

    We assume files are like 00000.png, 00001.png ... (5 digits).
    If your digits differ, adjust this function accordingly.
    """
    candidates = [f for f in os.listdir(fr_root) if f.lower().endswith(".png")]
    candidates.sort()
    for f in candidates:
        m = re.fullmatch(r"(\d+)\.png", f)
        if m:
            nd = len(m.group(1))
            return os.path.join(fr_root, f"%0{nd}d.png")
    return os.path.join(fr_root, "%05d.png")


def read_img(path: str, target_hw: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if target_hw is not None:
        h, w = target_hw
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def load_ref_shape(ref_dir: str) -> Tuple[int, int]:
    p0 = os.path.join(ref_dir, "00000.png")
    img0 = cv2.imread(p0, cv2.IMREAD_COLOR)
    if img0 is None:
        pngs = [f for f in os.listdir(ref_dir) if f.lower().endswith(".png")]
        pngs.sort()
        if not pngs:
            raise FileNotFoundError(f"No PNG frames under ref_dir={ref_dir}")
        img0 = cv2.imread(os.path.join(ref_dir, pngs[0]), cv2.IMREAD_COLOR)
        if img0 is None:
            raise RuntimeError(f"Failed to read reference frame: {pngs[0]}")
    return int(img0.shape[0]), int(img0.shape[1])


def read_ref_frames(ref_dir: str, max_frames: int, target_hw: Tuple[int, int]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for i in range(max_frames):
        fp = os.path.join(ref_dir, f"{i:05d}.png")
        img = read_img(fp, target_hw)
        if img is None:
            break
        frames.append(img)
    return frames


def read_frames_from_dir(fr_root: str, max_frames: int, target_hw: Tuple[int, int]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for i in range(max_frames):
        fp = os.path.join(fr_root, f"{i:05d}.png")
        img = read_img(fp, target_hw)
        if img is None:
            break
        frames.append(img)
    return frames


def list_numbered_png_indices(fr_root: str) -> List[int]:
    ids: List[int] = []
    try:
        files = os.listdir(fr_root)
    except Exception:
        return ids
    for f in files:
        m = re.fullmatch(r"(\d+)\.png", f, flags=re.IGNORECASE)
        if m is None:
            continue
        ids.append(int(m.group(1)))
    ids.sort()
    return ids


def read_frames_by_indices(fr_root: str, indices: List[int], target_hw: Tuple[int, int]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for idx in indices:
        fp = os.path.join(fr_root, f"{idx:05d}.png")
        img = read_img(fp, target_hw)
        if img is None:
            continue
        frames.append(img)
    return frames


def read_video_frames(video_path: str, max_frames: int, target_hw: Tuple[int, int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    h, w = target_hw
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def compute_lpips_mean(ref_frames: List[np.ndarray], rec_frames: List[np.ndarray], device: str = "cuda") -> float:
    """Compute mean LPIPS between ref and rec frames.

    Uses the same LPIPS implementation as generation.py:
    torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    """
    try:
        import torch  # type: ignore
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LPIPS requires `torch` and `torchmetrics`. "
            "Install them (e.g., pip install torchmetrics)."
        ) from e

    n = min(len(ref_frames), len(rec_frames))
    if n <= 0:
        raise ValueError("No overlapping frames between ref and rec for LPIPS.")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    loss_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(dev)

    vals = []
    with torch.no_grad():
        for i in range(n):
            gt = ref_frames[i]
            pred = rec_frames[i]

            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(dev) * 2.0 - 1.0
            pred_t = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).to(dev) * 2.0 - 1.0
            pred_t = torch.clamp(pred_t, min=-1.0, max=1.0)

            v = loss_fn(pred_t, gt_t).item()
            vals.append(float(v))

    return float(np.mean(vals))


def compute_sl_bits_per_segment(
    *,
    seg_fps: float,
    seg_sec: float,
    rank: int,
    interval: int,
    prompt_u: int = DEFAULT_PROMPT_U,
    prompt_v: int = DEFAULT_PROMPT_V,
) -> int:
    """Compute encoded semantic payload SL^p in bits/segment."""
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")
    if prompt_u <= 0 or prompt_v <= 0:
        raise ValueError("prompt_u and prompt_v must be positive.")

    F_ct = int(math.ceil(float(seg_fps) * float(seg_sec)))
    if F_ct <= 0:
        raise ValueError("Invalid F_ct computed from seg_fps/seg_sec")

    n_blk = int(math.ceil((float(F_ct)-1) / float(interval)))
    size_blk = int((prompt_u + prompt_v) * rank)
    return int((n_blk + 1) * size_blk)


@dataclass
class Accum:
    lpips_sum: float = 0.0
    slbits_sum: float = 0.0
    count: int = 0


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--frames_dir",
        type=str,
        default="",
        help="single-sequence reference frames dir (e.g., data/sky); if set, this mode is used",
    )
    ap.add_argument(
        "--recon_root",
        type=str,
        default="",
        help="single-sequence recon root (default: <frames_dir>/results)",
    )

    ap.add_argument("--out_csv", type=str, default="results_sc/sc_eval_sky.csv", help="output CSV path")

    ap.add_argument("--seg_fps", type=float, default=16.0, help="segment fps (fps_ct)")
    ap.add_argument("--seg_sec", type=float, default=1.0, help="segment duration in seconds (slot length)")
    ap.add_argument("--prompt_u", type=int, default=DEFAULT_PROMPT_U, help="prompt tensor u dimension")
    ap.add_argument("--prompt_v", type=int, default=DEFAULT_PROMPT_V, help="prompt tensor v dimension")

    ap.add_argument(
        "--eval_frames",
        type=int,
        default=-1,
        help="frames used for LPIPS; -1 means one segment (ceil(seg_fps*seg_sec))",
    )

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="device for LPIPS")

    args = ap.parse_args()

    F_ct = int(math.ceil(float(args.seg_fps) * float(args.seg_sec)))
    eval_frames = int(args.eval_frames)
    if eval_frames <= 0:
        eval_frames = F_ct

    log(f"Segment alignment: seg_fps={args.seg_fps}, seg_sec={args.seg_sec} => F_ct={F_ct} frames")
    log(f"LPIPS eval_frames={eval_frames}")
    log(f"SL^p params: prompt_u={args.prompt_u}, prompt_v={args.prompt_v}")
    if FIXED_RI_SET:
        selected = ", ".join([f"(rank={r}, interval={i})" for r, i in sorted(FIXED_RI_SET)])
        log(f"Profile filter enabled: {selected}")
    else:
        log("Profile filter disabled: evaluate all rank/interval profiles")

    seq_items: List[Tuple[str, str, str]] = []
    if args.frames_dir:
        frames_dir = os.path.normpath(args.frames_dir)
        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

        recon_root = os.path.normpath(args.recon_root) if args.recon_root else os.path.join(frames_dir, "results")
        if not os.path.isdir(recon_root):
            raise FileNotFoundError(
                f"recon_root not found: {recon_root}. "
                f"Use --recon_root or keep default <frames_dir>/results."
            )

        seq_name = os.path.basename(frames_dir)
        seq_items.append((seq_name, recon_root, frames_dir))
        log(f"Single-sequence mode: seq={seq_name}, recon_root={recon_root}, ref_dir={frames_dir}")
    else:
        out_root = args.out_root
        ref_root = args.ref_root

        if not os.path.isdir(out_root):
            raise FileNotFoundError(f"out_root not found: {out_root}")
        if not os.path.isdir(ref_root):
            raise FileNotFoundError(f"ref_root not found: {ref_root}")

        seq_names = [d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))]
        seq_names.sort()
        if not seq_names:
            raise RuntimeError(f"No sequences found under out_root={out_root}")

        for seq_name in seq_names:
            seq_dir = os.path.join(out_root, seq_name)
            ref_dir = os.path.join(ref_root, seq_name)
            seq_items.append((seq_name, seq_dir, ref_dir))

    agg: Dict[Tuple[int, int], Accum] = {}
    for seq_name, seq_dir, ref_dir in seq_items:
        if not os.path.isdir(ref_dir):
            log(f"[WARN] reference missing for seq '{seq_name}', skip.")
            continue

        target_hw = load_ref_shape(ref_dir)
        ref_frames_default = read_ref_frames(ref_dir, eval_frames, target_hw)
        if len(ref_frames_default) == 0:
            log(f"[WARN] no ref frames in '{ref_dir}', skip.")
            continue
        ref_ids_all = list_numbered_png_indices(ref_dir)

        cfg_dirs = [d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
        cfg_dirs.sort()

        for cfg_name in cfg_dirs:
            ri = parse_rank_interval(cfg_name)
            if ri is None:
                continue
            rank, interval = ri
            if FIXED_RI_SET and (rank, interval) not in FIXED_RI_SET:
                continue
            cfg_dir = os.path.join(seq_dir, cfg_name)

            vid = find_video_file(cfg_dir)
            fr_root = None if vid is not None else find_frames_root(cfg_dir)
            if vid is None and fr_root is None:
                log(f"[WARN] no recon output found: {cfg_dir}")
                continue

            try:
                slb = compute_sl_bits_per_segment(
                    seg_fps=args.seg_fps,
                    seg_sec=args.seg_sec,
                    rank=rank,
                    interval=interval,
                    prompt_u=args.prompt_u,
                    prompt_v=args.prompt_v,
                )

                lpips_pair_info = ""
                if vid is not None:
                    ref_frames_lpips = ref_frames_default
                    rec_frames_lpips = read_video_frames(vid, eval_frames, target_hw)
                    n_lpips = min(len(ref_frames_lpips), len(rec_frames_lpips))
                    if n_lpips > 0:
                        lpips_pair_info = (
                            f"first_pair={os.path.join(ref_dir, '00000.png')} <-> {vid}[frame0]"
                        )
                else:
                    rec_ids = list_numbered_png_indices(fr_root)
                    common_ids = sorted(set(ref_ids_all).intersection(rec_ids))
                    if eval_frames > 0:
                        common_ids = common_ids[:eval_frames]
                    if not common_ids:
                        raise RuntimeError("No common numbered PNG frame ids between ref and recon.")
                    ref_frames_lpips = read_frames_by_indices(ref_dir, common_ids, target_hw)
                    rec_frames_lpips = read_frames_by_indices(fr_root, common_ids, target_hw)
                    n_lpips = min(len(ref_frames_lpips), len(rec_frames_lpips))
                    if n_lpips > 0:
                        first_id = common_ids[0]
                        last_id = common_ids[n_lpips - 1]
                        lpips_pair_info = (
                            f"first_pair={os.path.join(ref_dir, f'{first_id:05d}.png')} <-> "
                            f"{os.path.join(fr_root, f'{first_id:05d}.png')} "
                            f"(id_range={first_id}-{last_id})"
                        )

                lp = compute_lpips_mean(ref_frames_lpips, rec_frames_lpips, device=args.device)

            except Exception as e:
                log(f"[WARN] failed on {seq_name}/{cfg_name}: {e}")
                continue

            key = (rank, interval)
            if key not in agg:
                agg[key] = Accum()
            agg[key].lpips_sum += float(lp)
            agg[key].slbits_sum += float(slb)
            agg[key].count += 1

            log(
                f"seq={seq_name} cfg={cfg_name} -> rank={rank} interval={interval} "
                f"lpips={lp:.6f} sl_bits={slb} n_lpips={n_lpips} {lpips_pair_info}"
            )

    if not agg:
        raise RuntimeError("No valid profile results collected.")

    out_list: List[Dict[str, float]] = []
    for (rank, interval), a in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lp_mean = a.lpips_sum / max(a.count, 1)
        slb_mean = a.slbits_sum / max(a.count, 1)
        out_list.append(
            {
                "rank": int(rank),
                "interval": int(interval),
                "lpips": float(lp_mean),
                "sl_bits": float(slb_mean),
            }
        )

    out_csv = args.out_csv
    if out_csv.endswith(os.sep) or (os.path.isdir(out_csv) and not out_csv.lower().endswith(".csv")):
        if not os.path.isdir(out_csv):
            raise FileNotFoundError(f"Output directory not found: {out_csv}")
        out_csv = os.path.join(out_csv, "sc_eval_sky.csv")

    out_dir = os.path.dirname(out_csv) or "."
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"Output directory not found: {out_dir}. "
            f"Use an existing directory (e.g., results_sc)."
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "interval", "lpips", "sl_bits"])
        writer.writeheader()
        for row in out_list:
            writer.writerow(row)

    log(f"\nSaved {len(out_list)} profiles -> {out_csv}")


if __name__ == "__main__":
    main()
