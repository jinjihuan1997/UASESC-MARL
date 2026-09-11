"""Read-only provenance audit; write new evidence JSON, never rewrite profiles.

Numerical reconstruction establishes table identities, not an unavailable
historical generation command. Historical videos/codecs are not retrained here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "HARL/HARL/harl/envs/uav_escs/semantic_models/profiles"
CALIBRATION = ROOT / "experiments/2026-09-07_happo_instruction_ab/calibration"
FORMAL = ROOT / "experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909"
CRL = ROOT / "CRL-SemCom-VidCI"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def read_profile(path):
    # The audited, local historical NPZ files use object arrays for mode names.
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new evidence output; audit records are immutable")
    paths = {
        "author_sc": PROFILES / "author_sci_profile_snr0_20.npz",
        "derived_sc": PROFILES / "derived_coupled_profile_snr0_20.npz",
        "synthetic_cc": PROFILES / "cc_h264_ldpc_profile_snr0_20.npz",
        "measured_pilot_sc": CALIBRATION / "profile.npz",
        "active_formal_sc": FORMAL / "source/reference/inputs/profile.npz",
    }
    hashes_before = {k: sha(p) for k, p in paths.items()}
    tables = {k: read_profile(p) for k, p in paths.items()}
    result = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "table provenance, numeric reconstruction, recorded-measurement aggregation, active consumer and input geometry",
        "historical_original_generation_command": "NOT_FOUND; formula identities below are independently reconstructed",
        "tables": {k: {"path": str(p), "sha256": hashes_before[k],
                       "mtime_utc": datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat(),
                       "fields": tables[k]} for k, p in paths.items()},
    }
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    log_records = []
    for i in range(1, 5):
        files = list((CRL / f"logs/24-03-08/24-03-08-MST/MST_fixed/v_{i}/summaries").glob("events*"))
        assert len(files) == 1
        p = files[0]
        accumulator = EventAccumulator(str(p), size_guidance={"scalars": 0})
        accumulator.Reload()
        rows = accumulator.Scalars("val/psnr")
        best = max(rows, key=lambda r: r.value)
        log_records.append({"version": i, "path": str(p), "sha256": sha(p),
                            "count": len(rows), "last_value": rows[-1].value,
                            "last_step": rows[-1].step,
                            "last_rounded_2dp": round(rows[-1].value, 2),
                            "maximum_value": best.value, "maximum_step": best.step})
    result["historical_sci_log_evidence"] = log_records
    ceiling = np.array([r["last_rounded_2dp"] for r in log_records])
    author, derived = tables["author_sc"], tables["derived_sc"]
    grid = derived["snr_grid_db"]
    degradation = np.array([4.09, 1.61, .52, .14, 0.])
    sci = np.repeat(np.arange(4), 4)
    rate = np.tile(np.arange(1, 5), 4)
    base_q = ceiling[:, None] - degradation[None, :]
    predicted_author = base_q[sci]
    # The fourth SCI anchor is lower than the third. Directly indexing the
    # fourth anchor as rate-4's cap is contradicted by stored row 11. A monotone
    # rate ceiling reproduces the full table; caps above all SCI ceilings are
    # not identifiable from the table, so this is an equivalent construction.
    rate_caps = np.maximum.accumulate(ceiling)
    predicted_derived = np.minimum(ceiling[sci], rate_caps[rate - 1])[:, None] - degradation
    main = np.broadcast_to(2048. * rate[:, None], (16, 5))
    real = 2. * main
    side = np.broadcast_to(2048. / np.log2(1 + 10 ** (grid / 10)), (16, 5))
    checks = {
        "author_quality_from_last_rounded_psnr_minus_common_penalty": error(author["q_hat_mean"], predicted_author),
        "derived_quality_min_of_sci_and_rate_curves": error(derived["q_hat_mean"], predicted_derived),
        "main_load_2048_times_rate": error(derived["bar_ls_main_mean"], main),
        "real_count_twice_main_load": error(derived["avg_kept_real_symbols_mean"], real),
        "stored_side_load_2048_bits_over_capacity": error(derived["avg_ls_side_mean"], side),
        "stored_total_is_main_plus_side": error(derived["bar_ls_total_mean"], main + side),
    }
    assert max(checks.values()) < 1e-10
    fronts = []
    for col, snr in enumerate(grid):
        q, l = derived["q_hat_mean"][:, col], main[:, col]
        nondominated = [i for i in range(16) if not any(
            q[j] >= q[i] and l[j] <= l[i] and (q[j] > q[i] or l[j] < l[i])
            for j in range(16))]
        unique = {}
        for i in nondominated:
            unique.setdefault((float(q[i]), float(l[i])), []).append(i)
        fronts.append({"snr_db": float(snr), "nondominated_row_ids": nondominated,
                       "unique_tradeoffs": [{"quality_db": pair[0], "main_uses": pair[1],
                                              "equivalent_rows": rows} for pair, rows in unique.items()]})
    result["sc_reconstruction"] = {
        "quality_anchors_match_historical_last_validation_values": ceiling,
        "equivalent_monotone_rate_quality_caps": rate_caps,
        "rate_cap_identification_limit": "Rate-3/4 caps at or above 32.81 are indistinguishable in these data; original rule not uniquely recoverable",
        "rejected_direct_rate_anchor_rule": {"max_abs_error": error(derived["q_hat_mean"], np.minimum(base_q[sci], base_q[rate - 1])),
                                               "mismatched_row": 11, "reason": "rate-4 must not lower SCI-4-measurement quality to the weaker SCI-8-measurement anchor"},
        "common_snr_loss_db": degradation,
        "snr_loss_source": "UNKNOWN; values reconstructed from stored table, no generating measurement record found",
        "max_abs_errors": checks,
        "derived_vs_author_changed_quality_cells": int(np.count_nonzero(derived["q_hat_mean"] != author["q_hat_mean"])),
        "all_load_arrays_unchanged_from_author": all(np.array_equal(derived[k], author[k]) for k in
             ["bar_ls_main_mean", "avg_kept_real_symbols_mean", "avg_ls_side_mean", "bar_ls_total_mean"]),
        "pareto_frontiers": fronts,
        "latent_interpretation": "2048*r complex uses equals C=16, Hs=Ws=32, four uniform rate levels; registry C=48 instead gives 6144*r. Not proof of a matching trained C=16 codec.",
    }
    import yaml
    cfg_path = ROOT / "HARL/HARL/harl/configs/envs_cfgs/uav_escs_cc.yaml"
    cc_config = yaml.safe_load(cfg_path.read_text())
    cc = tables["synthetic_cc"]
    qp, rates, required = [cc[k] for k in ["h264_qp_modes", "ldpc_code_rate_modes", "ldpc_required_snr_db"]]
    # Infer a single mean content proxy from QP42 at high SNR; verify every
    # other quality and load cell, including the independently configured bpp.
    psi = float((cc_config["h264_ref_psnr_db"] - cc["q_hat_mean"][0, -1]) / cc_config["h264_content_psnr_penalty_db"])
    pixels = cc_config["cc_source_h"] * cc_config["cc_source_w"] * cc_config["cc_source_l"]
    bits = (pixels * cc_config["h264_ref_bpp"] * 2 ** ((cc_config["h264_ref_qp"] - qp) / 6.)
            * (1 + cc_config["h264_content_bpp_gain"] * psi) + cc_config["cc_header_bits"])
    coded = bits / rates
    predicted_load = coded[:, None] / np.log2(1 + 10 ** (cc["snr_grid_db"] / 10))
    predicted_q = (cc_config["h264_ref_psnr_db"]
                   + cc_config["h264_psnr_per_qp_step_db"] * (cc_config["h264_ref_qp"] - qp[:, None])
                   - cc_config["h264_content_psnr_penalty_db"] * psi
                   - 2 * np.maximum(required[:, None] - cc["snr_grid_db"], 0))
    cc_checks = {"quality_formula": error(cc["q_hat_mean"], predicted_q),
                 "load_formula": error(cc["bar_ls_main_mean"], predicted_load),
                 "field_named_real_symbols_is_coded_bits": error(cc["avg_kept_real_symbols_mean"],
                       np.broadcast_to(coded[:, None], cc["bar_ls_main_mean"].shape))}
    assert max(cc_checks.values()) < 1e-8
    result["cc_reconstruction"] = {
        "config_path": cfg_path, "config_sha256": sha(cfg_path),
        "mean_content_proxy_inferred": psi,
        "mean_content_proxy_source": "UNKNOWN; inferred from table, exact identities verified, original averaging population not recovered",
        "source_pixels_per_update": pixels,
        "source_bits_by_mode_including_header": bits,
        "coded_bits_by_mode": coded,
        "max_abs_errors": cc_checks,
        "not_output_of_current_real_cc_builder": {
            "stored_schema": str(cc["schema_version"]),
            "missing_real_builder_fields": [k for k in ["bler", "deliver_prob", "h264_coded_bits_mean", "sample_count", "data_root"] if k not in cc],
            "stored_load_depends_on_snr": True,
            "real_builder_load_fixed_for_fixed_modulation_and_code_rate": True,
        },
        "config_comment": "Calibrated so round-robin CC serves a similar number of streams as SC under the same UAV/SUT topology while preserving H.264/LDPC's lower quality-load efficiency.",
    }
    records_path = CALIBRATION / "raw_measurements.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    measured = tables["measured_pilot_sc"]
    agg_errors = {"quality_mean": 0., "quality_std": 0., "main_load": 0., "real_symbols": 0.}
    counts = np.zeros(measured["q_hat_mean"].shape, dtype=int)
    for mi in range(counts.shape[0]):
        for si, snr in enumerate(measured["snr_grid_db"]):
            rows = [r for r in records if r["partition"] == "calibration" and r["mode_index"] == mi and r["snr_db"] == snr]
            counts[mi, si] = len(rows)
            for key, field, table_key, reducer in [
                ("quality_mean", "psnr", "q_hat_mean", np.mean),
                ("quality_std", "psnr", "q_hat_std", np.std),
                ("main_load", "main_uses", "bar_ls_main_mean", np.mean),
                ("real_symbols", "real_symbols", "avg_kept_real_symbols_mean", np.mean),
            ]:
                agg_errors[key] = max(agg_errors[key], abs(float(reducer([r[field] for r in rows])) - float(measured[table_key][mi, si])))
    assert max(agg_errors.values()) < 1e-10
    plan_path = CALIBRATION / "calibration_plan.json"
    plan = json.loads(plan_path.read_text())
    assert sha(records_path) == str(measured["source_records_sha256"])
    assert sha(plan_path) == str(measured["calibration_plan_sha256"])
    registry = json.loads((CALIBRATION / "mode_registry.json").read_text())
    weights = registry["sets"][str(measured["semantic_model_set"])]["modes"]
    weight_checks = [{"id": w["id"], "path": w["checkpoint"], "sha256": sha(w["checkpoint"]),
                      "matches_recorded_hash": sha(w["checkpoint"]) == w["checkpoint_sha256"]} for w in weights]
    assert all(w["matches_recorded_hash"] for w in weight_checks)
    errors = [float(np.interp(r["snr_db"], measured["snr_grid_db"], measured["q_hat_mean"][r["mode_index"]])) - r["psnr"]
              for r in records if r["partition"] == "diagnostic"]
    result["measured_pilot_reaggregation"] = {
        "record_count": len(records), "calibration_records": sum(r["partition"] == "calibration" for r in records),
        "diagnostic_records": len(errors), "measurements_per_cell": counts,
        "source_records_hash_verified": True, "calibration_plan_hash_verified": True,
        "weight_checks": weight_checks, "max_abs_errors": agg_errors,
        "diagnostic_mae_db": float(np.mean(np.abs(errors))),
        "diagnostic_rmse_db": float(np.sqrt(np.mean(np.square(errors)))),
        "note": "Reaggregated existing raw results; did not rerun the 1540 codec forwards.",
    }
    import torch
    dataset_summary = {}
    for split in ["train", "test"]:
        p = CRL / f"data/nfs_block_rgb_256_8f/{split}/nfs_block_file_locations.pt"
        index = torch.load(p, map_location="cpu", weights_only=True)
        dataset_summary[split] = {"index_path": str(p), "sha256": sha(p), "videos": len(index),
                                  "clips": sum(len(v) for v in index.values()),
                                  "path_counts_per_index_entry": sorted({len(c) for v in index.values() for c in v.values()}),
                                  "first_frame": index[min(index)][min(index[min(index)])][0]}
    assert dataset_summary["test"]["sha256"] == plan["data_index_sha256"]
    sys.path.insert(0, str(CRL))
    from src.dataio import NFS_Video
    dataset = NFS_Video(log_root=str(CRL / "data/nfs_block_rgb_256_8f"), block_size=[8, 256, 256],
                        split="test", color=False, test=True)
    batch = dataset[0]
    dataset_summary["actual_one_sample_shapes"] = {name: list(batch[i].shape) for i, name in enumerate(["average", "model_input", "target", "reference"])}
    result["dataset_geometry"] = dataset_summary
    current = json.loads((FORMAL / "configs/seed_85/IC_HAPPO.json").read_text())["env_args"]
    result["active_formal_consumer"] = {
        "configuration_path": str(FORMAL / "configs/seed_85/IC_HAPPO.json"),
        "profile_path": current["semantic_profile_path"],
        "profile_matches_original_derived_bytes": hashes_before["derived_sc"] == hashes_before["active_formal_sc"],
        "geometry_metadata": {k: current[k] for k in ["video_l", "video_h", "video_w", "video_channels", "semantic_channels"]},
        "actual_video_forward_in_rl": False,
        "runtime_side_info_bits": 2 * int(np.ceil(current["video_h"] / 8)) * int(np.ceil(current["video_w"] / 8))
                                    + int(np.ceil(np.log2(current["n_semantic_modes"]))),
        "runtime_uses_stored_total_load": False,
        "runtime_recomputes_side_load_at_instantaneous_snr": True,
        "runtime_content_dependence_in_Q_load": False,
    }
    hashes_after = {k: sha(p) for k, p in paths.items()}
    assert hashes_before == hashes_after
    result["all_audited_profiles_unchanged"] = True
    result["audit_script_sha256"] = sha(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(clean(result), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "checks": "PASS", "sc_errors": checks,
                      "cc_errors": cc_checks, "measured_reaggregation_errors": agg_errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
