"""Independent arithmetic and saved-bitstream checks; does not import the builder."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.stats import beta


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inspect(out):
    p = read(out / "protocol.json")
    root = Path(p["project_root"])
    profile = {k: v for k, v in np.load(out / "profile.npz", allow_pickle=False).items()}
    for relative, expected in read(out / "manifest.json").items():
        assert sha(out / relative) == expected, relative
    assert sha(p["sc_reference_path"]) == p["sc_reference_sha256"]
    assert sha(root / "HARL/HARL/harl/envs/uav_escs/semantic_models/profiles/cc_h264_ldpc_profile_snr0_20.npz") == "4a191eeeefa0ab97e1c364d04cb61672ebf866d6f387d43cc2fa0ccd7b0eac99"
    assert sha(root / "HARL/HARL/scripts/build_cc_h264_ldpc_profile.py") == sha(out.parents[1] / "source_before/build_cc_h264_ldpc_profile.py")
    splits = {part: [c for c in p["clips"] if c["partition"] == part] for part in ["calibration", "validation"]}
    cal_ids = {c["video_id"] for c in splits["calibration"]}
    val_ids = {c["video_id"] for c in splits["validation"]}
    assert cal_ids == set(range(8)) and val_ids == set(range(8, 14))
    cal_paths = {f["path"] for c in splits["calibration"] for f in c["frames"]}
    val_paths = {f["path"] for c in splits["validation"] for f in c["frames"]}
    assert not cal_paths.intersection(val_paths)
    for c in p["clips"]:
        for f in c["frames"]:
            assert sha(f["path"]) == f["sha256"]
    counts = Counter(c["video_id"] for c in splits["calibration"])
    weights = np.array([1 / len(cal_ids) / counts[c["video_id"]] for c in splits["calibration"]])
    np.testing.assert_allclose(weights.sum(), 1.)
    channel = {}
    for path in sorted((out / "channel").glob("*.json")):
        cell = read(path)
        n, errors = cell["blocks"], cell["block_errors"]
        assert n == sum(b["blocks"] for b in cell["batches"]) == p["mc_blocks_per_cell"]
        assert errors == sum(b["block_errors"] for b in cell["batches"])
        assert all(b["block_errors"] <= b["bit_errors"] <= b["block_errors"] * p["k"] for b in cell["batches"])
        low = beta.ppf(.025, errors, n-errors+1) if errors else 0.
        high = beta.ppf(.975, errors+1, n-errors) if errors < n else 1.
        np.testing.assert_allclose([cell["bler_ci_low"], cell["bler_ci_high"]], [low, high], atol=1e-15)
        energy = sum(b["noise_energy"] for b in cell["batches"])
        symbols = sum(b["complex_symbols"] for b in cell["batches"])
        np.testing.assert_allclose(cell["empirical_es_n0_db"], -10*np.log10(energy/symbols), atol=1e-12)
        channel[cell["rate_index"], cell["snr_index"]] = cell
    assert len(channel) == len(p["rates"]) * len(p["snr_grid_db"])
    max_errors = dict(quality=0., load=0., packet_probability=0.)
    mode_summary = []
    for mode in p["modes"]:
        records = [read(out / "codec" / f"{c['id']}_qp{mode['qp']}.json") for c in splits["calibration"]]
        sizes = np.array([r["h264_bytes"] for r in records])
        block_count = (8*(sizes+16)+1023)//1024
        code_n = math.ceil(Fraction(1024, 1) / Fraction(mode["rate"]) / 2) * 2
        uses = block_count * code_n // 2
        good = np.array([r["psnr_rgb_db"] for r in records])
        bad = np.array([r["outage_psnr_rgb_db"] for r in records])
        mi = mode["mode_id"]
        for si in range(len(p["snr_grid_db"])):
            cell = channel[mode["rate_index"], si]
            prob = np.power(1-cell["block_errors"]/cell["blocks"], block_count)
            quality = np.average(prob*good+(1-prob)*bad, weights=weights)
            load = np.average(uses, weights=weights)
            packet_probability = np.average(prob, weights=weights)
            for name, got, expected in [("quality", profile["q_hat_mean"][mi, si], quality),
                                        ("load", profile["bar_ls_main_mean"][mi, si], load),
                                        ("packet_probability", profile["deliver_prob"][mi, si], packet_probability)]:
                max_errors[name] = max(max_errors[name], abs(float(got)-float(expected)))
                np.testing.assert_allclose(got, expected, rtol=0., atol=1e-10)
        mode_summary.append(dict(**mode, actual_rate=1024/code_n, ldpc_n=code_n,
                                 h264_bytes_mean=float(np.average(sizes, weights=weights)),
                                 load_channel_uses=float(profile["bar_ls_main_mean"][mi, 0]),
                                 q_success_db=float(np.average(good, weights=weights)),
                                 packet_blocks_max=int(block_count.max()),
                                 high_snr_delivery_ci_low=float(profile["deliver_prob_ci_low"][mi, -1])))
    np.testing.assert_array_equal(profile["avg_kept_real_symbols_mean"], 2*profile["bar_ls_main_mean"])
    # Re-decode both extreme QPs for one frozen clip from each of the 14 videos.
    decoded_checks = []
    for video in sorted(cal_ids | val_ids):
        clip = next(c for c in p["clips"] if c["video_id"] == video)
        frames = []
        for f in clip["frames"]:
            with Image.open(f["path"]) as im:
                frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        source = np.stack(frames)
        for qp in (42, 26):
            stem = f"{clip['id']}_qp{qp}"
            record = read(out / "codec" / (stem + ".json"))
            stream = out / "bitstreams" / (stem + ".h264")
            assert sha(stream) == record["h264_sha256"]
            command = [p["ffmpeg"], "-hide_banner", "-loglevel", "error", "-threads", "1", "-filter_threads", "1",
                       "-r", str(p["fps"]), "-f", "h264", "-i", str(stream), "-an", "-pix_fmt", "rgb24",
                       "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
            result = subprocess.run(command, capture_output=True, check=True)
            assert len(result.stdout) == 8*256*256*3 and not result.stderr, result.stderr
            assert hashlib.sha256(result.stdout).hexdigest() == record["decoded_rgb_sha256"]
            difference = source.astype(np.float64)-np.frombuffer(result.stdout, np.uint8).reshape(source.shape)
            q = 10*np.log10(255.**2 / np.mean(difference**2))
            np.testing.assert_allclose(q, record["psnr_rgb_db"], rtol=0., atol=1e-10)
            decoded_checks.append(dict(clip_id=clip["id"], qp=qp, frames=8, psnr_rgb_db=float(q)))
    # Exercise the existing table reader, without creating or training an RL environment.
    sys.path.insert(0, str(root / "HARL/HARL"))
    from harl.envs.uav_escs.CC.uav_escs_env_cc import CCUAVEnv
    reader = CCUAVEnv.__new__(CCUAVEnv)
    reader.n_semantic_modes = len(p["modes"])
    loaded = reader._load_cc_profile(str(out / "profile.npz"))
    np.testing.assert_array_equal(loaded["q_hat_mean"], profile["q_hat_mean"])
    np.testing.assert_array_equal(loaded["l_z_mean"], profile["bar_ls_main_mean"])
    reader.semantic_mode_selection = "all_modes"
    reader.n_mu_modes = len(p["modes"])
    reader.semantic_mode_lookup_by_bucket_mu = reader._build_semantic_mode_lookup()
    reachable = np.unique([reader._bucket_mode_ids(bucket) for bucket in range(4)]).tolist()
    config_path = root / "experiments/2026-09-09_instruction_long_training/runs/three_seed_sc_20260909/configs/seed_85/IC_HAPPO.json"
    env = read(config_path)["env_args"]
    equal_budget = env["delta_T"]*env["backhaul_availability"]*min(env["B_uav_sut"], env["B_sut_sat"]/env["n_uav"])
    max_beta = 1 - (env["n_uav"]-1)*env["beta_sat_lower_bound"]
    max_budget = env["delta_T"]*env["backhaul_availability"]*min(env["B_uav_sut"], max_beta*env["B_sut_sat"])
    budget_screen = []
    for si, snr in enumerate(p["snr_grid_db"]):
        q = profile["q_hat_mean"][:, si]
        load = profile["bar_ls_main_mean"][:, si]
        budget_screen.append(dict(snr_db=snr,
                                  meets_qmin_and_equal_budget=np.flatnonzero((q >= env["Q_min"]) & (load <= equal_budget)).tolist(),
                                  meets_qmin_and_max_budget=np.flatnonzero((q >= env["Q_min"]) & (load <= max_budget)).tolist()))
    report = dict(state="PASS",profile_sha256=sha(out / "profile.npz"),partition_counts={k:len(v) for k,v in splits.items()},
                  source_frame_entries_verified=sum(len(c["frames"]) for c in p["clips"]),
                  independent_reaggregation_max_absolute_error=max_errors,bitstream_redecode_checks=decoded_checks,
                  existing_cc_profile_reader="PASS; array reading only, not RL integration",old_cc_and_sc_unchanged=True,
                  current_cc_mode_lookup=dict(shape=list(reader.semantic_mode_lookup_by_bucket_mu.shape),
                                              reachable_mode_ids=reachable,requires_integration_fix=len(reachable)!=len(p["modes"])),
                  mode_summary=mode_summary,active_sc_config_path=str(config_path),active_sc_config_sha256=sha(config_path),
                  budget_diagnostic=dict(equal_per_uav=equal_budget,max_single_uav=max_budget,q_min=env["Q_min"],
                                         meaning="Static average-load screen, not packet simulation or training performance",by_snr=budget_screen),
                  checks_source_sha256=sha(__file__))
    (out / "inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("mode_summary","bitstream_redecode_checks","budget_diagnostic")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    inspect(parser.parse_args().output.resolve())
