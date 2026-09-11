"""Prepare, train, and evaluate the frozen HAPPO instruction-input A/B pilot."""
from pathlib import Path
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HARL = ROOT / "HARL/HARL"
sys.path.insert(0, str(HARL))
sys.path.insert(0, str(HARL / "examples"))


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def prepare(out):
    from harl.utils.configs_tools import get_defaults_yaml_args
    calibration = out / "calibration"
    report = json.loads((calibration / "calibration_report.json").read_text())
    registry = json.loads((calibration / "mode_registry.json").read_text())
    algo, env = get_defaults_yaml_args("happo", "uav_escs_sc")
    env.update(semantic_model_set=registry["recommended_default"],
               semantic_registry_path=str(calibration / "mode_registry.json"),
               semantic_profile_path=str(calibration / "profile.npz"),
               semantic_mode_selection="all_modes", n_semantic_modes=5,
               env_name="measured5_expected_quality_instruction_pilot",
               Q_req_by_instruction_snr_bucket=report["Q_req_by_instruction_snr_bucket"],
               Q_min=report["Q_min"], Q_max=report["Q_max"], Q_min_eval=report["Q_min"],
               constraint_penalty_Q=0., constraint_penalty_Lambda=0.,
               actor_observe_instruction=True,
               T=600., instruction_switch_min_step=200, instruction_switch_max_step=400,
               include_instruction_id_in_obs=True, include_instruction_constraints_in_obs=True)
    algo["device"].update(cuda=False, cuda_deterministic=False, torch_threads=1)
    # 400 rollout steps x 10 workers x 500 updates = exactly 2,000,000 steps.
    algo["train"].update(n_rollout_threads=10, episode_length=400, num_env_steps=2000000,
                         model_dir=None, log_interval=5, eval_interval=25)
    algo["eval"].update(use_eval=False)
    algo["algo"].update(instruction_conditioned_critic=False, instruction_critic_type="single",
                        use_instruction_adv_norm=False)
    algo["logger"]["log_dir"] = str(out / "training")
    jobs = []
    for seed in (1, 2, 3):
        for variant, visible in (("A_no_explicit_instruction", False), ("B_explicit_instruction", True)):
            config = {"main_args": {"algo": "happo", "env": "uav_escs_sc", "exp_name": variant},
                      "algo_args": copy.deepcopy(algo), "env_args": copy.deepcopy(env)}
            config["algo_args"]["seed"].update(seed=seed, seed_specify=True)
            config["env_args"]["actor_observe_instruction"] = visible
            path = out / "configs" / f"{variant}_seed{seed}.json"
            if path.exists():
                raise FileExistsError(path)
            write_json(path, config)
            jobs.append(str(path))
    # Test seeds are disjoint from all seed + 1000*worker training seeds.
    manifest = {"experiment": "happo_instruction_input_ab_v1", "created_utc": stamp(),
                "jobs": jobs, "train_seeds": [1, 2, 3],
                "evaluation_seeds": list(range(20261001, 20261021)),
                "scenarios": ["fixed_balance", "fixed_aoi", "fixed_quality", "switch300_aoi", "switch300_quality"],
                "steps_per_job": 2000000, "total_training_steps": 12000000,
                "profile_sha256": sha(calibration / "profile.npz"),
                "registry_sha256": sha(calibration / "mode_registry.json"),
                "config_hashes": {p: sha(p) for p in jobs},
                "differences": "Within each seed, only exp_name and actor_observe_instruction differ. Critic keeps full instruction state in both groups.",
                "scope": "Five available measured modes, expected-quality scheduling. No per-video QoS or new PPO claim.",
                "ablation_boundary": "A masks explicit task ID and target features; physical SNR bucket, task-derived availability masks and pending-load observations remain available to both groups.",
                "changes_from_old_default": ["corrected active-symbol power and masked AWGN", "measured five-mode profile", "all five modes selectable at every SNR", "channel sampled once per slot", "independent episode/topology/channel/content/instruction RNG streams", "pre-action masks do not use previous SUT budget", "zero-valued quality/load penalties explicitly disabled", "ordinary single-head critic and global advantage normalization"]}
    write_json(out / "experiment_manifest.json", manifest)
    print(json.dumps({"prepared_jobs": len(jobs), "total_steps": 12000000}), flush=True)


def freeze(out):
    if (out / "source_frozen_manifest.json").exists():
        raise FileExistsError("Frozen experiment already exists; use a new experiment directory")
    paths = []
    for name in ("algorithms", "common", "models", "runners", "utils", "envs/uav_escs"):
        paths.extend((HARL / "harl" / name).rglob("*.py"))
    paths.extend([HARL / "harl/envs/env_wrappers.py", HARL / "harl/envs/__init__.py",
                  HARL / "examples/evaluate_instruction_constraints.py",
                  HARL / "scripts/calibrate_instruction_pilot.py",
                  HARL / "scripts/build_harl_semantic_profile.py", Path(__file__).resolve()])
    for name in ("src", "experiment_scripts/comm"):
        paths.extend((ROOT / "CRL-SemCom-VidCI" / name).rglob("*.py"))
    for p in sorted(set(paths)):
        target = out / "source_frozen" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write_json(out / "source_frozen_manifest.json", {str(p): sha(p) for p in sorted(set(paths))})


def verify_frozen(out):
    manifest = json.loads((out / "experiment_manifest.json").read_text())
    hashes = json.loads((out / "source_frozen_manifest.json").read_text())
    hashes.update(manifest["config_hashes"])
    hashes[str(out / "calibration/profile.npz")] = manifest["profile_sha256"]
    hashes[str(out / "calibration/mode_registry.json")] = manifest["registry_sha256"]
    changed = [p for p, expected in hashes.items() if not Path(p).is_file() or sha(p) != expected]
    if changed:
        raise RuntimeError(f"Frozen inputs changed; refuse mixed-version experiment: {changed}")


def state_hash(network):
    h = hashlib.sha256()
    for key, value in sorted(network.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_one(out, config_path, smoke=False):
    import numpy as np
    import torch
    import evaluate_instruction_constraints  # suppress Gym notice before runner import
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner
    config = json.loads(config_path.read_text())
    main, algo, env = config["main_args"], config["algo_args"], config["env_args"]
    job = config_path.stem
    if smoke:
        main["exp_name"] = "smoke_" + main["exp_name"]
        algo["logger"]["log_dir"] = str(out / "smoke_training")
        algo["train"].update(num_env_steps=8000, log_interval=1, eval_interval=2)
        job = "smoke_" + job
    status_path = out / "jobs" / f"{job}.json"
    started = time.monotonic()

    class PilotRunner(OnPolicyHARunner):
        def __init__(self):
            super().__init__(main, algo, env)
            self.completed_updates = 0
            self.initial_hashes = [state_hash(actor.actor) for actor in self.actor]
            self.initial_critic_hash = state_hash(self.critic.critic)
            self.report("initializing")

        def report(self, state, **extra):
            steps = self.completed_updates * algo["train"]["episode_length"] * algo["train"]["n_rollout_threads"]
            write_json(status_path, {"job": job, "state": state, "updated_utc": stamp(),
                       "completed_steps": steps, "target_steps": algo["train"]["num_env_steps"],
                       "elapsed_seconds": time.monotonic()-started, "run_dir": str(self.run_dir),
                       "config": str(config_path), "initial_actor_hashes": self.initial_hashes,
                       "initial_critic_hash": self.initial_critic_hash, **extra})

        def train(self):
            result = super().train()
            for network in [*(a.actor for a in self.actor), self.critic.critic]:
                if not all(torch.isfinite(p).all() for p in network.parameters()):
                    raise FloatingPointError("Non-finite learned parameters")
            if not np.isfinite(self.critic_buffer.returns).all():
                raise FloatingPointError("Non-finite critic return")
            for metrics in [*result[0], result[1]]:
                for key, value in metrics.items():
                    if not np.isfinite(float(value)):
                        raise FloatingPointError(f"Non-finite training metric {key}")
            self.completed_updates += 1
            self.report("training", finite_parameters=True)
            return result

    runner = None
    try:
        runner = PilotRunner()
        runner.run()
        runner.save()
        final_hashes = [state_hash(a.actor) for a in runner.actor]
        if any(a == b for a, b in zip(runner.initial_hashes, final_hashes)):
            raise RuntimeError("An actor did not update")
        runner.report("complete", finite_parameters=True, final_actor_hashes=final_hashes,
                      checkpoint_hashes={p.name: sha(p) for p in Path(runner.save_dir).glob("*.pt")})
    except BaseException as exc:
        if runner is not None:
            runner.report("failed", error=repr(exc))
        else:
            write_json(status_path, {"state": "failed", "error": repr(exc), "updated_utc": stamp()})
        raise
    finally:
        if runner is not None:
            runner.close()


def evaluate(out, smoke=False):
    import numpy as np
    import torch
    from evaluate_instruction_constraints import _load_actors, _actor_actions
    from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
    torch.set_num_threads(1)
    manifest = json.loads((out / "experiment_manifest.json").read_text())
    destination = out / ("smoke_evaluation" if smoke else "evaluation")
    destination.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(manifest["jobs"][1]).read_text())
    policies = [("R_fixed_projection", None, base), ("G_local_greedy", None, base)]
    for path in manifest["jobs"]:
        config_path = Path(path)
        record = out / "jobs" / (("smoke_" if smoke else "") + config_path.stem + ".json")
        if not record.exists():
            if smoke:
                continue
            raise FileNotFoundError(record)
        state = json.loads(record.read_text())
        if state["state"] != "complete":
            raise RuntimeError(f"Training is incomplete: {record}")
        policies.append((config_path.stem, Path(state["run_dir"]) / "models", json.loads(config_path.read_text())))
    seeds = manifest["evaluation_seeds"][:1] if smoke else manifest["evaluation_seeds"]
    scenarios = manifest["scenarios"]
    summary = []
    for label, weights, config in policies:
        actors = None
        for scenario in scenarios:
            for seed in seeds:
                args = copy.deepcopy(config["env_args"])
                if scenario.startswith("fixed_"):
                    args.update(instruction_mode_strategy="fixed", fixed_instruction_id={"fixed_balance": 0, "fixed_aoi": 1, "fixed_quality": 2}[scenario])
                else:
                    args.update(instruction_mode_strategy="switch_once", instruction_switch_step=300,
                                instruction_before_id=0, instruction_after_id=1 if scenario.endswith("aoi") else 2,
                                instruction_after_mode_strategy="fixed")
                env = SCUAVEnv(args)
                env.seed(seed)
                obs, shared, masks = env.reset()
                if weights is not None and actors is None:
                    actors = _load_actors(algo="happo", algo_args=config["algo_args"], env=env,
                                          model_dir=weights, device=torch.device("cpu"))
                recurrent_n = config["algo_args"]["model"]["recurrent_n"]
                hidden = config["algo_args"]["model"]["hidden_sizes"][-1]
                recurrent = np.zeros((env.n_agents, 1, recurrent_n, hidden), dtype=np.float32)
                actor_masks = np.ones((env.n_agents, 1, 1), dtype=np.float32)
                rows = []
                trace = hashlib.sha256()
                horizon = env.max_steps
                for slot in range(horizon):
                    trace.update(env.gamma_bh.tobytes())
                    if actors is not None:
                        action, recurrent = _actor_actions(actors, obs, masks, recurrent, actor_masks, True)
                    else:
                        action = rule_actions(label, env, obs, masks)
                    obs, shared, reward, done, infos, masks = env.step(action)
                    info = infos[0]
                    chi = np.asarray(info["chi"])
                    count = int(chi.sum())
                    quality_sum = float((chi * np.asarray(info["Q_hat_rec"])).sum())
                    used = float((chi * np.asarray(info["Lambda_sem"])).sum())
                    aoi = np.asarray(info["A_rcc"])
                    rows.append({"slot": slot, "instruction": int(info["instruction_id"]),
                                 "reward": float(reward[0, 0]), "avg_aoi": float(aoi.mean()),
                                 "max_aoi": float(aoi.max()), "p95_aoi": float(np.quantile(aoi, .95)),
                                 "aoi_target": float(info["A_limit_g"]),
                                 "aoi_exceedance_fraction": float(np.mean(aoi > info["A_limit_g"])),
                                 "scheduled": count, "predicted_quality_sum": quality_sum,
                                 "channel_uses": used, "executed_modes": json.dumps(env.last_selected_modes.tolist()),
                                 "proposed_modes": json.dumps([int(np.argmax(a[:env.n_mu_modes])) for a in action[1:]]),
                                 "proposed_scheduling_logits": json.dumps([np.asarray(a[env.uav_act_slices["scheduling_weights"]]).tolist() for a in action[1:]]),
                                 "proposed_resource_logits": json.dumps(np.asarray(action[0]).tolist()),
                                 "resource_fractions": json.dumps(env.beta_sut_sat.tolist())})
                path = destination / label / f"{scenario}_seed{seed}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                summary.append({"policy": label, "scenario": scenario, "seed": seed,
                                "trajectory_sha256": trace.hexdigest(), "steps": horizon,
                                "mean_reward": float(np.mean([r["reward"] for r in rows])),
                                "mean_aoi": float(np.mean([r["avg_aoi"] for r in rows])),
                                "mean_p95_aoi": float(np.mean([r["p95_aoi"] for r in rows])),
                                "aoi_exceedance_fraction": float(np.mean([r["aoi_exceedance_fraction"] for r in rows])),
                                "deliveries": sum(r["scheduled"] for r in rows),
                                "channel_uses": sum(r["channel_uses"] for r in rows),
                                "predicted_quality_sum": sum(r["predicted_quality_sum"] for r in rows)})
                write_json(destination / "progress.json", {"completed_episodes": len(summary), "updated_utc": stamp(), "last_policy": label})
                env.close()
    trajectories = {}
    for row in summary:
        key = (row["scenario"], row["seed"])
        trajectories.setdefault(key, set()).add(row["trajectory_sha256"])
    assert all(len(hashes) == 1 for hashes in trajectories.values()), "Unpaired exogenous channels"
    write_json(destination / "episode_summary.json", summary)


def rule_actions(label, env, obs, masks):
    """Use only actor-observable quantities and fixed public constants."""
    import numpy as np
    actions = [np.zeros(env.sut_act_dim_total)]
    if label == "G_local_greedy":
        # SUT sees per-UAV pending load and max AoI, not their raw local state.
        n = env.n_uav
        scores = obs[0, n+1:2*n+1] + obs[0, 2*n+1:3*n+1]
        actions[0] = (2*(scores-scores.min()) / max(float(np.ptp(scores)), 1e-9) - 1.)
    for n in range(env.n_uav):
        a = np.zeros(env.uav_act_dim_total)
        preferences = np.linspace(1., -1., env.n_mu_modes)
        if label == "G_local_greedy":
            d, m = env.ds_per_uav, env.n_semantic_modes
            row = obs[n+1]
            budget = row[0] * env.Lambda_ref  # previous SUT allocation, as for actor
            cache = row[1:1+d] > .5
            ages = row[1+2*d:1+3*d]
            loads = row[1+3*d:1+3*d+m] * env.Lambda_ref
            quality = row[1+3*d+m:1+3*d+m+d*m].reshape(d, m) * env.Q_max
            priorities = np.argsort(ages)[::-1]
            preferences = np.full(m, -1e9)
            for mode in range(m):
                if not masks[n+1, mode]:
                    continue
                room, score = float(budget), 0.
                for k in priorities:
                    if cache[k] and loads[mode] <= room:
                        room -= loads[mode]
                        qnorm = np.clip((quality[k, mode]-env.Q_min) / max(env.Q_max-env.Q_min, 1e-9), 0., 1.)
                        score += env.omega_Q*qnorm + env.omega_A*ages[k] - env.omega_Lambda*loads[mode]/env.Lambda_ref
                preferences[mode] = score
            finite = preferences > -1e8
            scaled = np.full(m, -1.)
            if finite.any():
                values = preferences[finite]
                span = float(np.ptp(values))
                scaled[finite] = 0. if span == 0 else -.9 + 1.8*(values-values.min())/span
            preferences = scaled
        a[env.uav_act_slices["mu_logits"]] = preferences
        actions.append(a)
    return actions


def run_queue(out):
    verify_frozen(out)
    manifest = json.loads((out / "experiment_manifest.json").read_text())
    for index, path in enumerate(manifest["jobs"]):
        verify_frozen(out)
        job = Path(path).stem
        state = {"state": "training", "updated_utc": stamp(), "active_job": job,
                 "completed_jobs": index, "total_jobs": len(manifest["jobs"]),
                 "progress_file": str(out / "jobs" / f"{job}.json")}
        write_json(out / "status.json", state)
        print(json.dumps(state), flush=True)
        with (out / f"{job}.log").open("x") as log:
            result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                     "--experiment", str(out), "--action", "train", "--config", path],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=HARL)
        if result.returncode:
            write_json(out / "status.json", dict(state, state="failed", returncode=result.returncode))
            raise RuntimeError(f"Training failed: {job}; see its log")
    write_json(out / "status.json", {"state": "evaluating", "completed_jobs": 6, "updated_utc": stamp()})
    verify_frozen(out)
    evaluate(out)
    write_json(out / "status.json", {"state": "complete", "completed_jobs": 6, "updated_utc": stamp(),
                                     "summary": str(out / "evaluation/episode_summary.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--action", choices=["prepare", "freeze", "train", "evaluate", "run"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.experiment.resolve()
    if args.action == "prepare": prepare(out)
    elif args.action == "freeze": freeze(out)
    elif args.action == "train": train_one(out, args.config.resolve(), args.smoke)
    elif args.action == "evaluate": evaluate(out, args.smoke)
    elif args.action == "run":
        try:
            run_queue(out)
        except BaseException as exc:
            current = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
            write_json(out / "status.json", dict(current, state="failed", error=repr(exc), updated_utc=stamp()))
            raise


if __name__ == "__main__":
    main()
