#!/usr/bin/env python3
"""Build a HARL offline H.264+LDPC profile from REAL codec + channel simulation.

This is the conventional-coding counterpart of ``build_harl_semantic_profile.py``.
Instead of a synthetic PSNR/bpp formula, it measures, on the SAME CRL-SemCom
validation clips used by the semantic profile:

  * real H.264 source coding  -> coded bits and reconstruction PSNR per QP
    (ffmpeg / libx264, lossless raw round-trip);
  * real 5G-LDPC + QAM + AWGN -> block error rate (BLER) per code rate and SNR
    (Sionna), giving the digital cliff behaviour;
  * end-to-end quality  Q_hat = P_deliver * PSNR_h264 + (1 - P_deliver) * PSNR_outage
    where P_deliver = (1 - BLER) ** num_ldpc_blocks.

It writes a compact NPZ table indexed by (mode, SNR) using the SAME schema as the
semantic profile so the environment can consume it the same way. The CRL-SemCom
repository is not modified.

Each CC "mode" m is an (H.264 QP, LDPC code rate) operating point.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# CRL data access (same clips as the semantic profile)
# --------------------------------------------------------------------------- #
def import_crl_modules(crl_root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(crl_root))
    from src import dataloading  # pylint: disable=import-outside-toplevel

    return {"dataloading": dataloading}


def make_crl_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=str(args.data_root),
        dataset_name=str(args.dataset_name),
        block_size=[int(args.video_l), int(args.video_h), int(args.video_w)],
        gt=int(args.gt),
        local=True,
        test=False,
        batch_size=1,
        num_workers=int(args.num_workers),
    )


def iter_validation_clips(args: argparse.Namespace, crl_modules: Dict[str, Any], max_batches: int):
    """Yield ground-truth clips as float arrays in [0, 1], shape (T, H, W)."""
    crl_args = make_crl_args(args)
    val_data = crl_modules["dataloading"].loadValDataset(crl_args)
    if val_data is None:
        raise RuntimeError("CRL-SemCom validation dataset is unavailable")
    val_dataloader, _ = val_data
    count = 0
    for batch in val_dataloader:
        # CRL batches: (_, model_input, gt, ref_input, _, _). gt is the clean
        # source used for PSNR. Layout is a frame batch (B, C, H, W) in [0, 1].
        gt = batch[2]
        gt_np = gt.detach().cpu().numpy()
        if gt_np.ndim == 4:  # (B, C, H, W) -> treat batch*channel-1 as frames
            if gt_np.shape[1] in (1, 3):
                frames = gt_np[:, 0, :, :] if gt_np.shape[1] == 1 else gt_np.mean(axis=1)
            else:
                frames = gt_np.reshape(-1, gt_np.shape[-2], gt_np.shape[-1])
        elif gt_np.ndim == 3:
            frames = gt_np
        else:
            frames = gt_np.reshape(-1, gt_np.shape[-2], gt_np.shape[-1])
        frames = np.clip(frames.astype(np.float64), 0.0, 1.0)
        yield frames
        count += 1
        if max_batches > 0 and count >= max_batches:
            break


# --------------------------------------------------------------------------- #
# Real H.264 source coding via ffmpeg (gray, lossless raw round-trip)
# --------------------------------------------------------------------------- #
def h264_encode_decode(frames: np.ndarray, qp: int, ffmpeg: str, fps: int = 25) -> Tuple[int, float]:
    """Return (coded_bits, psnr_db) for a real libx264 encode/decode at the QP."""
    t, h, w = frames.shape
    raw_in = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8).tobytes()
    enc = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-qp", str(int(qp)), "-g", str(int(t)),
            "-pix_fmt", "gray", "-f", "h264", "pipe:1",
        ],
        input=raw_in, capture_output=True,
    )
    stream = enc.stdout
    if not stream:
        raise RuntimeError(f"ffmpeg encode produced no output at qp={qp}: {enc.stderr.decode()[:300]}")
    coded_bits = len(stream) * 8
    dec = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "h264", "-i", "pipe:0",
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        input=stream, capture_output=True,
    )
    out = np.frombuffer(dec.stdout, dtype=np.uint8)
    need = t * h * w
    if out.size < need:
        # decoder dropped frames; pad by repeating last decoded frame
        if out.size == 0:
            raise RuntimeError(f"ffmpeg decode produced no frames at qp={qp}")
        out = np.concatenate([out, np.repeat(out[-h * w:], (need - out.size) // (h * w) + 1)])
    rec = out[:need].reshape(t, h, w).astype(np.float64) / 255.0
    mse = float(np.mean((rec - frames) ** 2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1.0e-12))
    return coded_bits, psnr


def freeze_outage_psnr(frames: np.ndarray) -> float:
    """PSNR proxy when a block is lost: show the previous frame (freeze)."""
    if frames.shape[0] < 2:
        ref = np.full_like(frames, float(np.mean(frames)))
    else:
        ref = np.concatenate([frames[:1], frames[:-1]], axis=0)
    mse = float(np.mean((ref - frames) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1.0e-12))


# --------------------------------------------------------------------------- #
# Real 5G-LDPC + QAM + AWGN BLER table via Sionna
# --------------------------------------------------------------------------- #
def build_bler_table(
    code_rates: List[float],
    snr_grid_db: np.ndarray,
    *,
    info_bits: int,
    num_bits_per_symbol: int,
    mc_blocks: int,
    seed: int,
) -> np.ndarray:
    """Return BLER[len(code_rates), len(snr_grid_db)] over an AWGN channel.

    The grid SNR is interpreted as the received symbol SNR Es/N0 in dB (matching
    the linear backhaul SNR gamma_bh consumed by the environment): no = 10**(-snr/10).
    """
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder  # noqa: E501  pylint: disable=import-outside-toplevel
    from sionna.mapping import Mapper, Demapper  # pylint: disable=import-outside-toplevel
    from sionna.channel import AWGN  # pylint: disable=import-outside-toplevel

    tf.random.set_seed(int(seed))
    mapper = Mapper("qam", int(num_bits_per_symbol))
    demapper = Demapper("app", "qam", int(num_bits_per_symbol))
    awgn = AWGN()

    bler = np.zeros((len(code_rates), int(snr_grid_db.size)), dtype=float)
    for r_idx, rate in enumerate(code_rates):
        k = int(info_bits)
        n = int(round(k / float(rate)))
        # 5G-LDPC needs n a valid length; round n up to a multiple of bits/symbol.
        if n % int(num_bits_per_symbol) != 0:
            n += int(num_bits_per_symbol) - (n % int(num_bits_per_symbol))
        encoder = LDPC5GEncoder(k, n)
        decoder = LDPC5GDecoder(encoder, hard_out=True)
        for s_idx, snr_db in enumerate(snr_grid_db):
            no = float(10.0 ** (-float(snr_db) / 10.0))
            bits = tf.cast(tf.random.uniform([int(mc_blocks), k]) > 0.5, tf.float32)
            codewords = encoder(bits)
            x = mapper(codewords)
            y = awgn([x, no])
            llr = demapper([y, no])
            bits_hat = decoder(llr)
            block_err = tf.reduce_any(tf.not_equal(bits, bits_hat), axis=1)
            bler[r_idx, s_idx] = float(tf.reduce_mean(tf.cast(block_err, tf.float32)))
    return bler


# --------------------------------------------------------------------------- #
# Profile assembly
# --------------------------------------------------------------------------- #
def build_profile(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    qp_modes = [int(x) for x in str(args.qp_modes).split(",") if x.strip()]
    code_rates = [float(x) for x in str(args.ldpc_code_rate_modes).split(",") if x.strip()]
    if len(qp_modes) != len(code_rates):
        raise ValueError("--qp-modes and --ldpc-code-rate-modes must have equal length")
    n_modes = len(qp_modes)
    snr_grid = np.asarray([float(x) for x in str(args.snr_grid_db).split(",") if x.strip()], dtype=float)
    if snr_grid.size == 0:
        raise ValueError("--snr-grid-db must contain at least one value")

    crl_modules = import_crl_modules(Path(args.crl_root).expanduser().resolve())

    # 1) Real H.264 stats per (mode, clip): coded bits, PSNR, outage PSNR.
    h264_bits: List[List[int]] = [[] for _ in range(n_modes)]
    h264_psnr: List[List[float]] = [[] for _ in range(n_modes)]
    outage_psnr_clips: List[float] = []
    n_clips = 0
    for frames in iter_validation_clips(args, crl_modules, int(args.max_batches)):
        n_clips += 1
        outage_psnr_clips.append(freeze_outage_psnr(frames))
        for m, qp in enumerate(qp_modes):
            bits, psnr = h264_encode_decode(frames, qp, args.ffmpeg, fps=int(args.fps))
            h264_bits[m].append(bits)
            h264_psnr[m].append(psnr)
        if args.progress:
            print(f"[h264] clip {n_clips} done", flush=True)
    if n_clips == 0:
        raise RuntimeError("No validation clips were evaluated")

    bits_mean = np.asarray([np.mean(b) for b in h264_bits], dtype=float)          # (n_modes,)
    psnr_h264_mean = np.asarray([np.mean(p) for p in h264_psnr], dtype=float)     # (n_modes,)
    outage_db = float(args.outage_floor_db) if args.outage_mode == "floor" else float(np.mean(outage_psnr_clips))

    # 2) Real LDPC BLER table over the SNR grid.
    bler = build_bler_table(
        code_rates, snr_grid,
        info_bits=int(args.ldpc_info_bits),
        num_bits_per_symbol=int(args.num_bits_per_symbol),
        mc_blocks=int(args.mc_blocks),
        seed=int(args.seed),
    )  # (n_modes, n_snr)

    # 3) Channel-use load and end-to-end quality per (mode, SNR).
    q_hat_mean = np.zeros((n_modes, snr_grid.size), dtype=float)
    l_z_mean = np.zeros_like(q_hat_mean)
    n_z_mean = np.zeros_like(q_hat_mean)
    deliver_prob = np.zeros_like(q_hat_mean)
    for m in range(n_modes):
        coded_source_bits = bits_mean[m] + float(args.cc_header_bits)
        channel_bits = coded_source_bits / max(float(code_rates[m]), 1.0e-9)       # after LDPC
        l_z = channel_bits / float(args.num_bits_per_symbol)                       # complex channel uses
        n_z = 2.0 * l_z                                                            # real symbols
        num_blocks = max(1.0, np.ceil(coded_source_bits / float(args.ldpc_info_bits)))
        for s in range(snr_grid.size):
            p_block_ok = float(np.clip(1.0 - bler[m, s], 0.0, 1.0))
            p_deliver = p_block_ok ** num_blocks
            deliver_prob[m, s] = p_deliver
            q_hat_mean[m, s] = p_deliver * psnr_h264_mean[m] + (1.0 - p_deliver) * outage_db
            l_z_mean[m, s] = l_z
            n_z_mean[m, s] = n_z

    return {
        "schema_version": "harl_semantic_profile_v1",
        "semantic_model_set": str(args.semantic_model_set),
        "dataset_name": str(args.dataset_name),
        "quality_metric": "psnr",
        "communication_model": "h264_ldpc",
        "mode_ids": np.asarray([f"h264_qp{qp}_ldpc{rate:.3f}" for qp, rate in zip(qp_modes, code_rates)], dtype=object),
        "h264_qp_modes": np.asarray(qp_modes, dtype=float),
        "ldpc_code_rate_modes": np.asarray(code_rates, dtype=float),
        "num_bits_per_symbol": int(args.num_bits_per_symbol),
        "snr_grid_db": snr_grid,
        "q_hat_mean": q_hat_mean,
        "q_hat_std": np.zeros_like(q_hat_mean),
        "bar_ls_main_mean": l_z_mean,                  # complex channel uses L_z (schema-compatible)
        "avg_kept_real_symbols_mean": n_z_mean,        # real symbols n_z (schema-compatible)
        "deliver_prob": deliver_prob,
        "bler": bler,
        "h264_coded_bits_mean": bits_mean,
        "psnr_h264_mean": psnr_h264_mean,
        "psnr_outage_db": float(outage_db),
        "sample_count": float(n_clips),
        "seed": int(args.seed),
        "max_batches": int(args.max_batches),
        "crl_root": str(Path(args.crl_root).expanduser().resolve()),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
    }


def write_outputs(profile: Dict[str, Any], output_npz: Path) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **profile)
    summary = {}
    for key, value in profile.items():
        if isinstance(value, np.ndarray):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if value.size and np.issubdtype(value.dtype, np.number):
                summary[key].update(
                    min=float(np.min(value)), max=float(np.max(value)), mean=float(np.mean(value))
                )
        else:
            summary[key] = value
    output_npz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crl-root", default=str(DEFAULT_CRL_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-name", default="nfs_block_rgb_256_8f")
    parser.add_argument("--semantic-model-set", default="h264_ldpc_5modes")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--snr-grid-db", default="0,5,10,15,20")
    parser.add_argument("--qp-modes", default="42,38,34,30,26")
    parser.add_argument("--ldpc-code-rate-modes", default="0.5,0.6666667,0.75,0.8333333,0.8333333")
    parser.add_argument("--num-bits-per-symbol", type=int, default=2)  # QPSK
    parser.add_argument("--ldpc-info-bits", type=int, default=1024)
    parser.add_argument("--mc-blocks", type=int, default=2000)
    parser.add_argument("--cc-header-bits", type=float, default=512.0)
    parser.add_argument("--outage-mode", choices=["freeze", "floor"], default="freeze")
    parser.add_argument("--outage-floor-db", type=float, default=8.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video-l", type=int, default=8)
    parser.add_argument("--video-h", type=int, default=256)
    parser.add_argument("--video-w", type=int, default=256)
    parser.add_argument("--gt", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    profile = build_profile(args)
    write_outputs(profile, Path(args.output_npz).expanduser())


DEFAULT_TON_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CRL_ROOT = DEFAULT_TON_ROOT / "CRL-SemCom-VidCI"
DEFAULT_DATA_ROOT = DEFAULT_CRL_ROOT / "data"


if __name__ == "__main__":
    main()
