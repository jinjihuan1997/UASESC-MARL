"""Analyze the frozen HAPPO instruction pilot without changing any source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


METRICS = ["reward", "aoi", "p95_aoi", "aoi_exceedance", "deliveries_per_slot",
           "predicted_psnr_db", "channel_uses_per_slot"]
GROUPS = {
    "R": ["R_fixed_projection"],
    "G": ["G_local_greedy"],
    "A": [f"A_no_explicit_instruction_seed{i}" for i in (1, 2, 3)],
    "B": [f"B_explicit_instruction_seed{i}" for i in (1, 2, 3)],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((differences(a.get(k), b.get(k), f"{prefix}.{k}".strip("."))
                    for k in sorted(a.keys() | b.keys())), [])
    return [] if a == b else [prefix]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    exp, out = args.experiment.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = exp.parents[1]
    spec_path = Path(__file__).with_name("analysis_plan.json")
    spec = json.loads(spec_path.read_text())
    manifest_path = exp / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_manifest_path = exp / "source_frozen_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    inputs = {}

    def track(path):
        path = Path(path)
        digest = sha(path)
        inputs[str(path)] = digest
        return digest

    for path in (spec_path, manifest_path, source_manifest_path):
        track(path)
    assert track(exp / "calibration/profile.npz") == manifest["profile_sha256"]
    assert track(exp / "calibration/mode_registry.json") == manifest["registry_sha256"]
    source_issues = []
    for name, digest in source_manifest.items():
        live = Path(name)
        frozen = exp / "source_frozen" / live.relative_to(root)
        if track(frozen) != digest:
            source_issues.append({"path": str(frozen), "issue": "frozen_hash_mismatch"})
        if track(live) != digest:
            source_issues.append({"path": str(live), "issue": "live_hash_mismatch"})
    assert not source_issues, source_issues

    configs, jobs, curves = {}, {}, {}
    for name, digest in manifest["config_hashes"].items():
        path = Path(name)
        assert track(path) == digest
        policy = path.stem
        config = json.loads(path.read_text())
        configs[policy] = config
        job_path = exp / "jobs" / f"{policy}.json"
        track(job_path)
        job = json.loads(job_path.read_text())
        assert job["state"] == "complete" and job["completed_steps"] == 2_000_000
        assert job["finite_parameters"]
        for model, expected in job["checkpoint_hashes"].items():
            assert track(Path(job["run_dir"]) / "models" / model) == expected
        log_path = Path(job["run_dir"]) / "logs/summary.json"
        track(log_path)
        series = json.loads(log_path.read_text())
        selected = [k for k in series if k.endswith("critic/average_step_rewards")]
        assert len(selected) == 1
        curve = np.asarray(series[selected[0]], dtype=float)
        assert curve.shape == (100, 3) and np.isfinite(curve).all()
        assert np.all(np.diff(curve[:, 1]) > 0) and curve[-1, 1] == 2_000_000
        curves[policy] = {
            "steps": curve[:, 1].astype(int).tolist(), "reward": curve[:, 2].tolist(),
            "first_10_mean": float(curve[:10, 2].mean()),
            "previous_10_mean": float(curve[-20:-10, 2].mean()),
            "last_10_mean": float(curve[-10:, 2].mean()),
            "last_20_min_max": [float(curve[-20:, 2].min()), float(curve[-20:, 2].max())],
            "last_20_slope_per_million_steps": float(np.polyfit(curve[-20:, 1] / 1e6, curve[-20:, 2], 1)[0]),
            "elapsed_seconds": job["elapsed_seconds"],
        }
        jobs[policy] = job
    ab_checks = []
    for a, b in zip(GROUPS["A"], GROUPS["B"]):
        changed = differences(configs[a], configs[b])
        assert changed == ["env_args.actor_observe_instruction", "main_args.exp_name"], changed
        assert jobs[a]["initial_actor_hashes"] == jobs[b]["initial_actor_hashes"]
        assert jobs[a]["initial_critic_hash"] == jobs[b]["initial_critic_hash"]
        ab_checks.append({"A": a, "B": b, "config_differences": changed, "same_initial_parameters": True})

    summary_path = exp / "evaluation/episode_summary.json"
    track(summary_path)
    original = json.loads(summary_path.read_text())
    policies = sum(GROUPS.values(), [])
    scenarios, seeds = manifest["scenarios"], manifest["evaluation_seeds"]
    expected_keys = {(p, c, s) for p in policies for c in scenarios for s in seeds}
    keys = [(r["policy"], r["scenario"], r["seed"]) for r in original]
    assert len(original) == len(expected_keys) == 800
    assert len(set(keys)) == len(keys) and set(keys) == expected_keys
    all_csv = set((exp / "evaluation").glob("*/*.csv"))
    expected_csv = {exp / "evaluation" / p / f"{c}_seed{s}.csv" for p, c, s in expected_keys}
    assert all_csv == expected_csv
    paired_hashes = {}
    records, time_series = [], {}
    max_reconciliation_error = Counter()
    numeric = ["slot", "instruction", "reward", "avg_aoi", "max_aoi", "p95_aoi",
               "aoi_target", "aoi_exceedance_fraction", "scheduled", "predicted_quality_sum", "channel_uses"]
    for i, old in enumerate(original):
        policy, scenario, seed = old["policy"], old["scenario"], old["seed"]
        paired_hashes.setdefault((scenario, seed), set()).add(old["trajectory_sha256"])
        path = exp / "evaluation" / policy / f"{scenario}_seed{seed}.csv"
        track(path)
        frame = pd.read_csv(path)
        assert len(frame) == old["steps"] == 600
        assert np.array_equal(frame.slot.to_numpy(), np.arange(600))
        assert np.isfinite(frame[numeric].to_numpy()).all()
        target_id = {"fixed_balance": 0, "fixed_aoi": 1, "fixed_quality": 2}.get(scenario)
        instructions = (np.full(600, target_id) if target_id is not None else
                        np.r_[np.zeros(300), np.full(300, 1 if scenario.endswith("aoi") else 2)])
        assert np.array_equal(frame.instruction.to_numpy(), instructions), (policy, scenario, seed)
        assert (frame.scheduled >= 0).all() and (frame.channel_uses >= 0).all()
        fractions = np.asarray([json.loads(x) for x in frame.resource_fractions])
        executed = np.asarray([json.loads(x) for x in frame.executed_modes], dtype=int)
        proposed = np.asarray([json.loads(x) for x in frame.proposed_modes], dtype=int)
        scheduling = np.asarray([json.loads(x) for x in frame.proposed_scheduling_logits])
        assert fractions.shape == executed.shape == proposed.shape == (600, 3)
        assert scheduling.shape == (600, 3, 3)
        assert np.isfinite(fractions).all() and np.isfinite(scheduling).all()
        np.testing.assert_allclose(fractions.sum(axis=1), 1., rtol=0, atol=1e-12)
        assert fractions.min() >= .05 - 1e-12
        assert ((executed >= -1) & (executed < 5)).all()
        assert ((proposed >= 0) & (proposed < 5)).all()
        recomputed = {
            "mean_reward": frame.reward.mean(), "mean_aoi": frame.avg_aoi.mean(),
            "mean_p95_aoi": frame.p95_aoi.mean(),
            "aoi_exceedance_fraction": frame.aoi_exceedance_fraction.mean(),
            "deliveries": frame.scheduled.sum(), "channel_uses": frame.channel_uses.sum(),
            "predicted_quality_sum": frame.predicted_quality_sum.sum(),
        }
        for field, value in recomputed.items():
            np.testing.assert_allclose(value, old[field], rtol=1e-11, atol=1e-10,
                                       err_msg=f"{policy}/{scenario}/{seed}/{field}")
            max_reconciliation_error[field] = max(max_reconciliation_error[field], float(abs(value - old[field])))
        windows = {"whole": (0, 600)}
        if scenario.startswith("switch"):
            windows.update({k: tuple(v) for k, v in spec["switch_windows"].items()})
        for window, (lo, hi) in windows.items():
            part = frame.iloc[lo:hi]
            count = int(part.scheduled.sum())
            assert count > 0, "Zero-delivery episodes require explicitly undefined quality"
            em, pm = executed[lo:hi], proposed[lo:hi]
            active = em >= 0
            rec = {
                "policy": policy, "scenario": scenario, "eval_seed": seed, "window": window,
                "reward": float(part.reward.mean()), "aoi": float(part.avg_aoi.mean()),
                "p95_aoi": float(part.p95_aoi.mean()),
                "aoi_exceedance": float(part.aoi_exceedance_fraction.mean()),
                "deliveries_per_slot": count / (hi-lo),
                "predicted_psnr_db": float(part.predicted_quality_sum.sum() / count),
                "channel_uses_per_slot": float(part.channel_uses.mean()),
                "mode_argmax_disagreement_active": float((em[active] != pm[active]).mean()) if active.any() else None,
                "no_executed_mode_fraction": float((~active).mean()),
                "executed_mode_counts": np.bincount(em[active], minlength=5).tolist(),
                "proposed_mode_counts": np.bincount(pm.ravel(), minlength=5).tolist(),
                "mean_resource_max_minus_min": float(np.ptp(fractions[lo:hi], axis=1).mean()),
                "mean_resource_fractions": fractions[lo:hi].mean(axis=0).tolist(),
                "scheduling_raw_std_by_uav_component": scheduling[lo:hi].std(axis=0).tolist(),
            }
            records.append(rec)
        traces = frame[["reward", "avg_aoi", "scheduled", "predicted_quality_sum", "channel_uses"]].to_numpy().T
        time_series.setdefault((policy, scenario), []).append(traces)
        if (i+1) % 100 == 0:
            print(f"Validated {i+1}/800 raw episodes", flush=True)
    assert len(paired_hashes) == 100 and all(len(v) == 1 for v in paired_hashes.values())
    lookup = {(r["policy"], r["scenario"], r["eval_seed"], r["window"]): r for r in records}

    def cube(group, selected_scenarios, window):
        return np.array([[[[lookup[(p, c, s, window)][m] for m in METRICS] for s in seeds]
                          for c in selected_scenarios] for p in GROUPS[group]])

    summaries, contrasts = [], []
    comparisons = [("B", "A"), ("A", "G"), ("B", "G"), ("A", "R"), ("B", "R"), ("G", "R")]
    rng = np.random.default_rng(spec["uncertainty"]["rng_seed"])
    repeats = spec["uncertainty"]["repetitions"]
    # Crossed resampling: the same environment-seed draw is used for every
    # training seed and every scenario. Scenarios are fixed, not IID replicates.
    boot_eval = rng.integers(0, len(seeds), size=(repeats, len(seeds)))
    boot_train = rng.integers(0, 3, size=(repeats, 3))
    t975_df2 = np.sqrt(2 * .95**2 / (1 - .95**2))
    strata = [("all_scenarios", scenarios, "whole")] + [(c, [c], "whole") for c in scenarios]
    strata += [(c, [c], w) for c in scenarios if c.startswith("switch") for w in spec["switch_windows"]]
    for label, chosen, window in strata:
        arrays = {g: cube(g, chosen, window) for g in GROUPS}
        for g, arr in arrays.items():
            summaries.append({"group": g, "scenario": label, "window": window,
                              "mean": dict(zip(METRICS, arr.mean(axis=(0, 1, 2)).tolist())),
                              "per_training_seed": [dict(zip(METRICS, row.tolist())) for row in arr.mean(axis=(1, 2))]})
        for left, right in comparisons:
            delta = arrays[left] - arrays[right]
            effect = delta.mean(axis=(0, 1, 2))
            seed_effect = delta.mean(axis=(1, 2))
            # Average fixed scenarios first, retaining the crossed seed design.
            reduced = delta.mean(axis=1)
            sampled = []
            for start in range(0, repeats, 1000):
                ev = boot_eval[start:start+1000]
                if delta.shape[0] == 3:
                    tr = boot_train[start:start+1000]
                    samples = reduced[tr[:, :, None], ev[:, None, :], :].mean(axis=(1, 2))
                else:
                    samples = reduced[0, ev, :].mean(axis=1)
                sampled.append(samples)
            intervals = np.quantile(np.concatenate(sampled), [.025, .975], axis=0)
            if delta.shape[0] == 3:
                radius = t975_df2 * seed_effect.std(axis=0, ddof=1) / np.sqrt(3)
                t_ci = np.stack([effect-radius, effect+radius], axis=0)
            else:
                t_ci = None
            contrasts.append({
                "contrast": f"{left}-{right}", "scenario": label, "window": window,
                "effect": dict(zip(METRICS, effect.tolist())),
                "per_training_seed_effect": [dict(zip(METRICS, r.tolist())) for r in seed_effect],
                "crossed_bootstrap_95_ci": {m: intervals[:, j].tolist() for j, m in enumerate(METRICS)},
                "training_seed_t_95_ci_conditional_on_eval_set": None if t_ci is None else
                    {m: t_ci[:, j].tolist() for j, m in enumerate(METRICS)},
                "positive_seed_count": dict(zip(METRICS, (seed_effect > 0).sum(axis=0).tolist())),
                "paired_episode_win_fraction": dict(zip(METRICS, (delta > 0).mean(axis=(0, 1, 2)).tolist())),
            })
    behavior = []
    for g, members in GROUPS.items():
        for label, chosen, window in strata:
            rows = [lookup[(p, c, s, window)] for p in members for c in chosen for s in seeds]
            em = np.sum([r["executed_mode_counts"] for r in rows], axis=0)
            pm = np.sum([r["proposed_mode_counts"] for r in rows], axis=0)
            behavior.append({
                "group": g, "scenario": label, "window": window,
                "mode_argmax_disagreement_active": float(np.mean([r["mode_argmax_disagreement_active"] for r in rows])),
                "no_executed_mode_fraction": float(np.mean([r["no_executed_mode_fraction"] for r in rows])),
                "executed_mode_fraction": (em/em.sum()).tolist(),
                "proposed_mode_fraction": (pm/pm.sum()).tolist(),
                "mean_resource_max_minus_min": float(np.mean([r["mean_resource_max_minus_min"] for r in rows])),
                "mean_resource_fractions": np.mean([r["mean_resource_fractions"] for r in rows], axis=0).tolist(),
            })
    means = [{"policy": p, "scenario": c, "columns": ["reward", "aoi", "scheduled", "predicted_quality_sum", "channel_uses"],
              "values": np.mean(v, axis=0).tolist()} for (p, c), v in time_series.items()]
    integrity = {
        "episodes": len(original), "raw_slots": 600 * len(original), "trained_models": len(jobs),
        "paired_channel_hash_groups": len(paired_hashes), "all_paired_channel_hashes_match": True,
        "source_manifest_files": len(source_manifest), "frozen_and_live_sources_match": True,
        "config_profile_registry_model_hashes_match": True, "ab_checks": ab_checks,
        "summary_raw_max_absolute_error": dict(max_reconciliation_error),
        "missing_duplicate_or_nonfinite_records": 0, "all_switches_at_slot_300": True,
        "scope": "The stored pairing hash covers backhaul channel draws only; raw CSV does not contain the complete content/topology streams. Their action independence is established by the frozen implementation and earlier tests, not by this hash alone.",
    }
    write_json(out / "integrity.json", integrity)
    write_json(out / "episode_metrics.json", records)
    write_json(out / "results.json", {"metrics": METRICS, "summaries": summaries, "contrasts": contrasts, "behavior": behavior})
    write_json(out / "training_curves.json", curves)
    write_json(out / "mean_trajectories.json", means)
    inputs[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    write_json(out / "input_manifest.json", inputs)
    for row in summaries:
        if row["scenario"] == "all_scenarios":
            print(json.dumps(row, ensure_ascii=False), flush=True)
    for row in contrasts:
        if row["scenario"] == "all_scenarios" and row["contrast"] in {"B-A", "A-G", "B-G"}:
            print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"Saved analysis to {out}")


if __name__ == "__main__":
    main()
