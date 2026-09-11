"""Run a fixed, counterbalanced 3-seed device throughput matrix."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    modes = ["cpu", "gpu", "cpu_vector", "gpu_vector", "hybrid_vector"]
    cases = [(seed, mode) for seed, offset in [(1, 0), (2, 3), (3, 2)]
             for mode in modes[offset:] + modes[:offset]]
    script = OUT / "probe.py"
    source_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    if (OUT / "plan.json").exists():
        raise FileExistsError("Use a new output directory to repeat the matrix")
    vector_script = OUT / "vectorized_simplex.py"
    vector_hash = hashlib.sha256(vector_script.read_bytes()).hexdigest()
    plan = {"cases": cases, "steps_per_case": 16000, "updates_per_case": 4,
            "warmup_updates_excluded_from_steady_metrics": 1,
            "rollout_workers": 10, "rollout_steps": 400,
            "probe_sha256": source_hash,
            "vectorized_simplex_sha256": vector_hash,
            "timing": "perf_counter at existing synchronous CPU-output stage boundaries; no per-step additional CUDA synchronization",
            "background": "Frozen formal CPU A/B queue continues; do not treat this as a machine-exclusive benchmark",
            "invariants": "Same architecture, simulator, measured profile, reward, clipping, minibatches and epochs. Vector variants replace per-row masked Dirichlet log-probability and entropy with equivalent batched formulas, preserving sampling. Optimized GPU variants fix ValueNorm statistics registration; hybrid uses CPU inference and GPU updates. No learning-rule changes.",
            "scope": "Throughput and fixed-input numerical checks, not comparative policy performance"}
    write_json(OUT / "plan.json", plan)
    (OUT / "probe_frozen.py.txt").write_bytes(script.read_bytes())
    (OUT / "vectorized_simplex_frozen.py.txt").write_bytes(vector_script.read_bytes())
    results = []
    for index, (seed, mode) in enumerate(cases):
        assert hashlib.sha256(script.read_bytes()).hexdigest() == source_hash
        assert hashlib.sha256(vector_script.read_bytes()).hexdigest() == vector_hash
        write_json(OUT / "status.json", {"state": "running", "completed_cases": index,
                   "total_cases": len(cases), "active_case": f"{mode}_seed{seed}"})
        with (OUT / f"{mode}_seed{seed}.log").open("x") as log:
            subprocess.run([sys.executable, "-u", str(script), "--mode", mode,
                            "--seed", str(seed), "--updates", "4"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
        result = json.loads((OUT / f"{mode}_seed{seed}/result.json").read_text())
        results.append(result)
        print(json.dumps({"case": f"{mode}_seed{seed}", "steady_fps": result["steady_steps_per_second"]}), flush=True)
    summary = {}
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode]
        times = sum(r["steady_seconds"] for r in rows)
        stage_names = set().union(*(r["steady_stage_seconds"] for r in rows))
        fps = [r["steady_steps_per_second"] for r in rows]
        updates = sum(len(r["updates"])-1 for r in rows)
        summary[mode] = {"runs": len(rows), "steady_steps": sum(r["steady_steps"] for r in rows),
                         "pooled_steady_fps": sum(r["steady_steps"] for r in rows)/times,
                         "seed_fps": fps, "median_fps": statistics.median(fps),
                         "min_fps": min(fps), "max_fps": max(fps),
                         "mean_stage_seconds_per_update": {name: sum(r["steady_stage_seconds"].get(name, 0.) for r in rows)/updates for name in stage_names},
                         "normalizer_checkpoint_complete": all(len(r["saved_normalizer_state_keys"]) == 3 for r in rows),
                         "finite_parameters_and_metrics": all(r["finite_parameters_and_metrics"] for r in rows)}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "status.json", {"state": "complete", "completed_cases": len(cases), "total_cases": len(cases)})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_json(OUT / "status.json", {"state": "failed", "error": repr(exc)})
        raise
