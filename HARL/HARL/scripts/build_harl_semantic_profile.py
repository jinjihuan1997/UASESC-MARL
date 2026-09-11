#!/usr/bin/env python3
"""Build a HARL offline semantic profile from CRL-SemCom checkpoints.

The HARL environment consumes a compact NPZ table indexed by semantic mode and
effective backhaul SNR. This script evaluates the CRL-SemCom validation set
offline and writes that table without modifying the CRL-SemCom repository.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

from harl.envs.uav_escs.semantic_models.semantic_registry import SemanticModeLibrary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_snr_grid(text: str) -> List[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def make_crl_args(args: argparse.Namespace, mode: Dict[str, Any], snr_db: float) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=str(args.data_root),
        dataset_name=str(args.dataset_name),
        block_size=[int(args.video_l), int(args.video_h), int(args.video_w)],
        gt=int(args.gt),
        local=True,
        test=False,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        interp=None,
        init=str(args.shutter_init),
        resume="00-00-00",
        decoder=str(args.decoder),
        shutter=str(args.shutter),
        loss="l2",
        use_semantic_comm=True,
        comm_sensor_channels=int(args.comm_sensor_channels),
        comm_latent_channels=int(mode.get("comm_latent_channels", args.comm_latent_channels)),
        comm_hidden_channels=int(args.comm_hidden_channels),
        comm_rate_levels=int(mode.get("comm_rate_levels", args.comm_rate_levels)),
        comm_channel_coding_rate=float(args.comm_channel_coding_rate),
        comm_modulation_order=int(args.comm_modulation_order),
        comm_snr_db=float(snr_db),
        comm_forced_rate_level=None,
        eval_forced_rate_level=args.eval_forced_rate_level,
        mu_comm=float(mode.get("mu_comm", args.mu_comm)),
        sci_cost_weight=float(args.sci_cost_weight),
        shutter_loss_weight=float(args.shutter_loss_weight),
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    current_state = model.state_dict()
    missing = set(current_state) - set(state)
    # Historical CUDA checkpoints omit this deterministic all-ones tensor:
    # Parameter(...).to(cuda) was not registered by the original shutter code.
    # This is the only documented compatibility exception; learned keys must
    # all match exactly. Never silently drop incompatible learned tensors.
    constant = "shutter.rand_init_new"
    if missing == {constant} and torch.equal(
        current_state[constant], torch.ones_like(current_state[constant])
    ):
        state = dict(state)
        state[constant] = current_state[constant].clone()
    model.load_state_dict(state, strict=True)


def float_comm(comm_info: Dict[str, Any], key: str) -> float:
    value = comm_info[key]
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def evaluate_one(
    *,
    args: argparse.Namespace,
    crl_modules: Dict[str, Any],
    mode: Dict[str, Any],
    mode_index: int,
    snr_db: float,
    device: torch.device,
) -> Dict[str, float]:
    crl_args = make_crl_args(args, mode, snr_db)
    shutter = crl_modules["models"].define_shutter(crl_args.shutter, crl_args)
    decoder = crl_modules["models"].define_decoder(crl_args.decoder, crl_args)
    model = crl_modules["models"].define_model(shutter, decoder, crl_args)
    checkpoint_path = Path(str(mode.get("checkpoint", ""))).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Mode {mode_index} checkpoint does not exist: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    val_data = crl_modules["dataloading"].loadValDataset(crl_args)
    if val_data is None:
        raise RuntimeError("CRL-SemCom validation dataset is unavailable")
    val_dataloader, _ = val_data

    psnr_values: List[float] = []
    ssim_values: List[float] = []
    kept_values: List[float] = []
    ls_main_values: List[float] = []
    ls_side_values: List[float] = []
    ls_total_values: List[float] = []

    max_batches = int(args.max_batches)
    iterator = enumerate(val_dataloader)
    progress = tqdm(
        iterator,
        total=max_batches if max_batches > 0 else None,
        disable=not args.progress,
        desc=f"mode={mode_index} snr={snr_db:g}dB",
    )
    with torch.no_grad():
        for batch_idx, (_, model_input, gt, ref_input, _, _) in progress:
            if max_batches > 0 and batch_idx >= max_batches:
                break
            model_input = model_input.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            ref_input = ref_input.to(device, non_blocking=True)
            restored, _, _, extra = model(
                [model_input, ref_input],
                train=False,
                forced_rate_level=args.eval_forced_rate_level,
            )
            comm_info = extra["comm"]
            psnr_values.append(float(crl_modules["summary_utils"].get_psnr(restored, gt)))
            ssim_values.append(float(crl_modules["summary_utils"].get_ssim(restored, gt)))
            kept_values.append(float_comm(comm_info, "avg_kept_real_symbols"))
            ls_main_values.append(float_comm(comm_info, "avg_ls_main"))
            ls_side_values.append(float_comm(comm_info, "avg_ls_side"))
            ls_total_values.append(float_comm(comm_info, "avg_ls_total"))

    if not psnr_values:
        raise RuntimeError(f"No validation batches evaluated for mode {mode_index}, snr={snr_db}")
    return {
        "q_mean": float(np.mean(psnr_values)),
        "q_std": float(np.std(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "kept_mean": float(np.mean(kept_values)),
        "ls_main_mean": float(np.mean(ls_main_values)),
        "ls_side_mean": float(np.mean(ls_side_values)),
        "ls_total_mean": float(np.mean(ls_total_values)),
        "sample_count": float(len(psnr_values)),
    }


def import_crl_modules(crl_root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(crl_root))
    from src import dataloading, models, summary_utils  # pylint: disable=import-outside-toplevel

    return {
        "dataloading": dataloading,
        "models": models,
        "summary_utils": summary_utils,
    }


def build_profile(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    crl_modules = import_crl_modules(Path(args.crl_root).expanduser().resolve())
    library = SemanticModeLibrary.from_config(
        enabled=True,
        set_name=args.semantic_model_set,
        registry_path=args.semantic_registry_path or None,
        profile_path=None,
    )
    snr_grid = np.asarray(parse_snr_grid(args.snr_grid_db), dtype=float)
    if snr_grid.size == 0:
        raise ValueError("--snr-grid-db must contain at least one SNR value")

    n_modes = int(library.num_modes)
    n_snr = int(snr_grid.size)
    q_hat_mean = np.zeros((n_modes, n_snr), dtype=float)
    q_hat_std = np.zeros_like(q_hat_mean)
    ssim_mean = np.zeros_like(q_hat_mean)
    ssim_std = np.zeros_like(q_hat_mean)
    kept_mean = np.zeros_like(q_hat_mean)
    ls_main_mean = np.zeros_like(q_hat_mean)
    ls_side_mean = np.zeros_like(q_hat_mean)
    ls_total_mean = np.zeros_like(q_hat_mean)
    sample_count = np.zeros_like(q_hat_mean)

    for mode_index, mode in enumerate(library.modes):
        for snr_index, snr_db in enumerate(snr_grid):
            stats = evaluate_one(
                args=args,
                crl_modules=crl_modules,
                mode=mode,
                mode_index=mode_index,
                snr_db=float(snr_db),
                device=device,
            )
            q_hat_mean[mode_index, snr_index] = stats["q_mean"]
            q_hat_std[mode_index, snr_index] = stats["q_std"]
            ssim_mean[mode_index, snr_index] = stats["ssim_mean"]
            ssim_std[mode_index, snr_index] = stats["ssim_std"]
            kept_mean[mode_index, snr_index] = stats["kept_mean"]
            ls_main_mean[mode_index, snr_index] = stats["ls_main_mean"]
            ls_side_mean[mode_index, snr_index] = stats["ls_side_mean"]
            ls_total_mean[mode_index, snr_index] = stats["ls_total_mean"]
            sample_count[mode_index, snr_index] = stats["sample_count"]

    return {
        "schema_version": "harl_semantic_profile_v1",
        "semantic_model_set": str(args.semantic_model_set),
        "dataset_name": str(args.dataset_name),
        "quality_metric": "psnr",
        "mode_ids": np.asarray(library.ids, dtype=object),
        "mode_names": np.asarray(library.ids, dtype=object),
        "checkpoint_paths": np.asarray(library.checkpoints, dtype=object),
        "mu_comm": np.asarray(library.mu_comm, dtype=float),
        "trained_snr_db": np.asarray(library.trained_snr_db, dtype=float),
        "snr_grid_db": snr_grid,
        "q_hat_mean": q_hat_mean,
        "q_hat_std": q_hat_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "avg_kept_real_symbols_mean": kept_mean,
        "bar_ls_main_mean": ls_main_mean,
        "avg_ls_side_mean": ls_side_mean,
        "bar_ls_total_mean": ls_total_mean,
        "sample_count": sample_count,
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
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if value.size and np.issubdtype(value.dtype, np.number):
                summary[key]["min"] = float(np.min(value))
                summary[key]["max"] = float(np.max(value))
                summary[key]["mean"] = float(np.mean(value))
        else:
            summary[key] = value
    output_npz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crl-root", default=str(DEFAULT_CRL_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-name", default="nfs_block_rgb_256_8f")
    parser.add_argument("--semantic-model-set", default=ALL_DEFAULT_MODEL_SET)
    parser.add_argument("--semantic-registry-path", default="")
    parser.add_argument("--snr-grid-db", default="0,5,10,15,20")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--video-l", type=int, default=8)
    parser.add_argument("--video-h", type=int, default=256)
    parser.add_argument("--video-w", type=int, default=256)
    parser.add_argument("--gt", type=int, default=0)
    parser.add_argument("--decoder", default="MST")
    parser.add_argument("--shutter", default="lsvpe")
    parser.add_argument("--shutter-init", default="quad")
    parser.add_argument("--eval-forced-rate-level", type=int, default=None)
    parser.add_argument("--comm-sensor-channels", type=int, default=24)
    parser.add_argument("--comm-latent-channels", type=int, default=16)
    parser.add_argument("--comm-hidden-channels", type=int, default=64)
    parser.add_argument("--comm-rate-levels", type=int, default=4)
    parser.add_argument("--comm-channel-coding-rate", type=float, default=0.5)
    parser.add_argument("--comm-modulation-order", type=int, default=4)
    parser.add_argument("--mu-comm", type=float, default=1.0e-3)
    parser.add_argument("--sci-cost-weight", type=float, default=0.05)
    parser.add_argument("--shutter-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    profile = build_profile(args)
    write_outputs(profile, Path(args.output_npz).expanduser())


ALL_DEFAULT_MODEL_SET = "ran_adaptive_snr0_15_mu_grid_16modes"
DEFAULT_TON_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CRL_ROOT = DEFAULT_TON_ROOT / "CRL-SemCom-VidCI"
DEFAULT_DATA_ROOT = DEFAULT_CRL_ROOT / "data"


if __name__ == "__main__":
    main()
