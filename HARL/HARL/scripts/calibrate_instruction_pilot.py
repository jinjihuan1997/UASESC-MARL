"""Frozen, small video-disjoint calibration for the instruction A/B pilot.

This measures existing codecs under corrected fixed-symbol-power AWGN. It
does not retrain codecs or claim per-video QoS from a mode/SNR mean table.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "HARL/HARL"))
sys.path.insert(0, str(ROOT / "CRL-SemCom-VidCI"))
import numpy as np
import torch
from skimage.metrics import structural_similarity
from src import models, dataloading
from build_harl_semantic_profile import load_checkpoint
from harl.envs.uav_escs.semantic_models.semantic_registry import SemanticModeLibrary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_args(mode):
    return SimpleNamespace(
        block_size=[8, 256, 256], gt=0, local=True, test=False, batch_size=1,
        num_workers=0, interp=None, init="quad", resume="00-00-00", decoder="MST",
        shutter="lsvpe", loss="l2", use_semantic_comm=True, comm_sensor_channels=24,
        comm_latent_channels=int(mode.get("comm_latent_channels", 48)),
        comm_hidden_channels=64, comm_rate_levels=4, comm_channel_coding_rate=.5,
        comm_modulation_order=4, comm_snr_db=10., comm_forced_rate_level=None,
        mu_comm=float(mode["mu_comm"]), sci_cost_weight=.05, shutter_loss_weight=1.,
        data_root=str(ROOT / "CRL-SemCom-VidCI/data"), dataset_name="nfs_block_rgb_256_8f",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw_measurements.jsonl").exists():
        raise FileExistsError("Use a new calibration output directory; records are immutable")
    torch.set_num_threads(2)
    torch.manual_seed(20260907)
    crl = ROOT / "CRL-SemCom-VidCI"
    original = SemanticModeLibrary.from_config(
        enabled=True, set_name="ran_adaptive_snr0_15_mu_grid_16modes"
    )
    modes, missing = [], []
    for i, item in enumerate(original.modes):
        path = crl / item["checkpoint_rel"]
        if path.is_file():
            modes.append(dict(item, checkpoint=str(path), source_mode_index=i,
                              mode_id=len(modes), checkpoint_sha256=sha(path)))
        else:
            missing.append(item["id"])
    if len(modes) != 5:
        raise RuntimeError(f"Expected the five previously audited weights, found {len(modes)}; review inventory")
    name = "measured_available5_instruction_pilot_v1"
    registry = {"recommended_default": name, "sets": {name: {"modes": modes}}}
    (out / "mode_registry.json").write_text(json.dumps(registry, indent=2))
    _, dataset = dataloading.loadValDataset(model_args(modes[0]))
    dataset.test = True  # deterministic preprocessing; never augment calibration/test
    assert dataset.num_vids == 20
    # Fixed before looking at quality: early diagnostic clips remain in calibration.
    partitions = {"calibration": list(range(8)), "diagnostic": list(range(8, 14)),
                  "reserved_final": list(range(14, 20))}
    selected = []
    offset = 0
    for vid, count in enumerate(dataset.num_clips_each):
        if vid < 8:
            positions = sorted(set([count // 4, (3 * count) // 4]))
            split = "calibration"
        elif vid < 14:
            positions = [count // 2]
            split = "diagnostic"
        else:
            positions = []
            split = "reserved_final"
        for pos in positions:
            selected.append({"index": offset + pos, "video_id": vid,
                             "clip_id": pos, "partition": split,
                             "frame_paths": dataset.vid_dict[vid][pos]})
        offset += count
    grid = [-5., 0., 5., 10., 15., 20., 25.]
    plan = {"schema": "instruction_pilot_calibration_v1", "partitions": partitions,
            "selected_clips": selected, "snr_grid_db": grid, "noise_repeats": 2,
            "mode_ids": [m["id"] for m in modes], "missing_modes_excluded": missing,
            "data_index_sha256": sha(crl / "data/nfs_block_rgb_256_8f/test/nfs_block_file_locations.pt"),
            "channel_sha256": sha(crl / "experiment_scripts/comm/channel.py"),
            "transmission_sha256": sha(crl / "experiment_scripts/comm/transmission.py"),
            "input_geometry": [16, 256, 256], "loader_block_size": [8, 256, 256],
            "quality_definition": "PSNR from global unclipped reconstruction MSE, data_range=1; SSIM from clipped frames",
            "power_definition": "unit active real coordinate power, paired complex=(I+jQ)/sqrt(2)",
            "profile_use": "expected quality/load only; no content-aware or per-video quality guarantee",
            "threshold_rule": "Use calibration-only mean quality envelope at SNR bucket edges; AoI=min, balance=midpoint, quality=max-minus-0.5dB (bounded below by min). No policy returns used."}
    (out / "calibration_plan.json").write_text(json.dumps(plan, indent=2))
    batches = [(s, dataset[s["index"]]) for s in selected]
    records = []
    start = time.monotonic()
    with (out / "raw_measurements.jsonl").open("x") as log:
        for mi, mode in enumerate(modes):
            a = model_args(mode)
            model = models.define_model(models.define_shutter(a.shutter, a), models.define_decoder(a.decoder, a), a)
            load_checkpoint(model, Path(mode["checkpoint"]), torch.device("cpu"))
            model.eval()
            for snr in grid:
                model.transmission.snr_db = snr
                for sample, batch in batches:
                    source, target, ref = batch[1].unsqueeze(0), batch[2].unsqueeze(0), batch[3].unsqueeze(0)
                    for repeat in range(2):
                        # Common noise seeds across modes; never consume the RL RNG.
                        noise_seed = 20260907 + sample["index"] * 100 + grid.index(snr) * 2 + repeat
                        torch.manual_seed(noise_seed)
                        with torch.inference_mode():
                            restored, _, action, extra = model([source, ref], train=False)
                        if not torch.isfinite(restored).all():
                            raise RuntimeError("Non-finite video reconstruction")
                        c = extra["comm"]
                        mse = float((restored - target).square().mean())
                        rec, gt = restored[0].clamp(0, 1).numpy(), target[0].numpy()
                        row = {k: sample[k] for k in ("index", "video_id", "clip_id", "partition")}
                        row.update(mode_index=mi, mode_id=mode["id"], snr_db=snr,
                                   noise_seed=noise_seed, repeat=repeat,
                                   psnr=float(-10 * np.log10(max(mse, 1e-12))),
                                   ssim=float(np.mean([structural_similarity(t, r, data_range=1) for t, r in zip(gt, rec)])),
                                   main_uses=float(c["avg_ls_main"]),
                                   real_symbols=float(c["avg_kept_real_symbols"]),
                                   rate_map_bits=float(c["side_info_bits_per_clip"].mean()),
                                   mode_id_bits=int(np.ceil(np.log2(len(modes)))),
                                   sci_action_hist=torch.bincount(action.long().flatten(), minlength=4).tolist())
                        assert abs(2 * row["main_uses"] - row["real_symbols"]) < 1e-5
                        records.append(row)
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                progress = {"completed": len(records), "total": len(modes)*len(grid)*len(batches)*2,
                            "mode_index": mi, "snr_db": snr, "elapsed_seconds": time.monotonic()-start}
                (out / "progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    q, qstd, load, real = [np.zeros((len(modes), len(grid))) for _ in range(4)]
    for mi in range(len(modes)):
        for si, snr in enumerate(grid):
            rows = [r for r in records if r["partition"] == "calibration" and r["mode_index"] == mi and r["snr_db"] == snr]
            q[mi,si] = np.mean([r["psnr"] for r in rows])
            qstd[mi,si] = np.std([r["psnr"] for r in rows])
            load[mi,si] = np.mean([r["main_uses"] for r in rows])
            real[mi,si] = np.mean([r["real_symbols"] for r in rows])
    np.savez_compressed(out / "profile.npz", schema_version="harl_semantic_profile_v1",
                        semantic_model_set=name, dataset_name="nfs_block_rgb_256_8f_calibration8videos",
                        quality_metric="psnr", mode_ids=np.asarray([m["id"] for m in modes]),
                        snr_grid_db=np.asarray(grid), q_hat_mean=q, q_hat_std=qstd,
                        bar_ls_main_mean=load, avg_kept_real_symbols_mean=real,
                        source_records_sha256=sha(out / "raw_measurements.jsonl"),
                        calibration_plan_sha256=sha(out / "calibration_plan.json"))
    # Fixed calibration-only rule; diagnostic/final data never set service targets.
    anchors = np.asarray([0., 5., 10., 15.])
    anchor_q = np.asarray([np.interp(anchors, grid, row) for row in q])
    low, high = anchor_q.min(axis=0), anchor_q.max(axis=0)
    thresholds = np.stack([(low+high)/2, low, np.maximum(low, high-.5)])
    errors = []
    for r in records:
        if r["partition"] == "diagnostic":
            prediction = float(np.interp(r["snr_db"], grid, q[r["mode_index"]]))
            errors.append(prediction-r["psnr"])
    report = {"completed_forward_passes": len(records), "elapsed_seconds": time.monotonic()-start,
              "mean_quality_by_mode_snr": q.tolist(), "mean_main_uses": load.tolist(),
              "diagnostic_psnr_mae_db": float(np.mean(np.abs(errors))),
              "diagnostic_psnr_rmse_db": float(np.sqrt(np.mean(np.square(errors)))),
              "Q_req_by_instruction_snr_bucket": thresholds.tolist(),
              "Q_min": float(np.floor(q.min())), "Q_max": float(np.ceil(q.max())),
              "profile_sha256": sha(out / "profile.npz"),
              "limitation": "Small expected-quality calibration; diagnostic errors do not establish per-clip QoS. Reserved final videos untouched."}
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
