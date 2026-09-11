"""Post-hoc mechanism checks using unchanged traces and the frozen profile."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
exp = here.parents[1]
manifest = json.loads((exp / "experiment_manifest.json").read_text())
checks = []
for group in ["A", "B"]:
    for training_seed in [1, 2, 3]:
        policy = ("A_no_explicit_instruction" if group == "A" else "B_explicit_instruction") + f"_seed{training_seed}"
        for target in ["fixed_aoi", "switch300_aoi", "switch300_quality"]:
            counts = {name: 0 for name in ["avg_aoi", "scheduled", "executed_modes", "resource_fractions"]}
            prefix_identical = 0
            delta_after_aoi = []
            for seed in manifest["evaluation_seeds"]:
                base = pd.read_csv(exp / "evaluation" / policy / f"fixed_balance_seed{seed}.csv")
                alt = pd.read_csv(exp / "evaluation" / policy / f"{target}_seed{seed}.csv")
                for name in counts:
                    counts[name] += int(base[name].equals(alt[name]))
                names = list(counts)
                prefix_identical += int(base[names].iloc[:300].equals(alt[names].iloc[:300]))
                delta_after_aoi.append(float(alt.avg_aoi.iloc[300:].mean() - base.avg_aoi.iloc[300:].mean()))
            checks.append({"group": group, "training_seed": training_seed, "comparison": target + " vs fixed_balance",
                           "exactly_identical_600_slot_trajectories_out_of_20": counts,
                           "identical_first_300_slots_out_of_20": prefix_identical,
                           "mean_after_switch_aoi_difference": float(np.mean(delta_after_aoi))})

with np.load(exp / "calibration/profile.npz", allow_pickle=False) as profile:
    quality = profile["q_hat_mean"]
    loads = profile["bar_ls_main_mean"]
    dominated = []
    for a in range(len(quality)):
        for b in range(len(quality)):
            if a != b and np.all(quality[b] > quality[a]) and np.all(loads[b] < loads[a]):
                dominated.append({"dominated_mode_index": a, "dominating_mode_index": b,
                                  "minimum_quality_gap_db": float((quality[b]-quality[a]).min()),
                                  "minimum_main_channel_uses_saved": float((loads[a]-loads[b]).min())})
    mode_ids = profile["mode_ids"].tolist()

payload = {
    "analysis_type": "Post-hoc diagnostic; not a new confirmatory endpoint or new policy evaluation",
    "scenario_invariance": checks,
    "strictly_dominated_modes_at_all_seven_calibration_snr_points": dominated,
    "mode_ids": mode_ids,
    "limits": ["Dominance concerns calibration mean quality and mean main load; it is not a per-video dominance claim.",
               "Equal traces establish the observed behavior on these seeds, not universal invariance.",
               "The CSV proposed-mode field uses argmax. Disagreement with execution may include tie-breaking, feasibility changes and no-service cases; it is not a clean causal estimate of projection contribution."],
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(here / "results/mechanism_checks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"A_exact_aoi_sequences": [r for r in checks if r["group"] == "A" and r["comparison"].startswith("fixed_aoi")],
                  "mode_dominance": dominated}, ensure_ascii=False))
