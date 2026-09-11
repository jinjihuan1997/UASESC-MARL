"""Paired evaluation on full 600-slot episodes and a common external objective."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import time
from protocol import (activate_runtime, METHODS, RULES, read, sha, stamp,
                      verify_checkpoint, verify_run, write)


def external_metrics(env, info):
    """Independently calculate physical violations and the shared objective."""
    import numpy as np
    chi = np.asarray(info["chi"])
    quality = np.asarray(info["Q_hat_rec"])
    loads = np.asarray(info["Lambda_sem"])
    before, after = np.asarray(info["A_rcc_pre"]), np.asarray(info["A_rcc"])
    target = float(info["A_limit_g"])
    delivered = int(chi.sum())
    q_gain = (chi * np.maximum((quality-env.Q_min_eval)/(env.Q_max-env.Q_min_eval), 0)).sum()/env.n_ds
    uses = float((chi*loads).sum())
    aoi_term = (env.aoi_mean_weight*np.mean(after/env.aoi_reward_ref)
                + env.aoi_max_weight*np.max(after/env.aoi_reward_ref)
                + env.aoi_tail_weight*np.mean(np.maximum((after-env.aoi_tail_threshold)/env.aoi_reward_ref, 0)))
    wq, wa, wl = env.reward_weights_by_instruction[info["instruction_id"]]
    base = wq*q_gain - wl*uses/env.Lambda_ref/env.n_uav - wa*aoi_term
    aoi_excess = max(0, (float(after.max())-target)/target)
    bonus = env.eta_recv_aoi_bonus*np.mean(np.maximum(before-after, 0))/target
    common = base-env.constraint_penalty_A_by_instruction[info["instruction_id"]]*aoi_excess+bonus
    if not np.isclose(common, info["common_evaluation_reward"], atol=1e-9, rtol=0):
        raise AssertionError("Independent common reward disagrees with environment")
    budget_bad = int(np.count_nonzero((chi*loads).sum(axis=(1, 2)) > np.asarray(info["Phi_bh"])+1e-8))
    q_bad = int(np.count_nonzero((chi > 0) & (quality < np.asarray(info["Q_req_gb"])[:, None, None]-1e-8)))
    cache_bad = int(np.count_nonzero(chi.sum(axis=2) > np.asarray(info["q_cache_pre"])))
    if budget_bad or q_bad or cache_bad:
        raise AssertionError(f"Executed constraints violated: {budget_bad}, {q_bad}, {cache_bad}")
    return dict(common_reward=float(common), base_reward=float(base),
                training_reward=float(info["total_reward"]), recv_aoi_bonus=float(bonus),
                mean_aoi=float(after.mean()), max_aoi=float(after.max()),
                p95_aoi=float(np.quantile(after, .95)),
                aoi_exceedance_fraction=float(np.mean(after > target)),
                max_aoi_violation=float(aoi_excess), deliveries=delivered,
                predicted_quality_sum=float((chi*quality).sum()), channel_uses=uses,
                quality_violations=q_bad, budget_violations=budget_bad, cache_violations=cache_bad,
                infeasible_uav_fraction=float(info["all_mu_infeasible_count"])/env.n_uav)


def aggregate(rows):
    import numpy as np
    result = {key: float(np.mean([r[key] for r in rows])) for key in (
        "common_reward", "base_reward", "training_reward", "recv_aoi_bonus", "mean_aoi",
        "max_aoi", "p95_aoi", "aoi_exceedance_fraction", "max_aoi_violation", "infeasible_uav_fraction")}
    for key in ("deliveries", "predicted_quality_sum", "channel_uses", "quality_violations", "budget_violations", "cache_violations"):
        result[key] = sum(r[key] for r in rows)
    result["delivered_predicted_psnr"] = (result["predicted_quality_sum"]/result["deliveries"]
                                          if result["deliveries"] else None)
    result["deliveries_per_slot"] = result["deliveries"] / sum(r.get("steps", 1) for r in rows)
    result["channel_uses_per_slot"] = result["channel_uses"] / sum(r.get("steps", 1) for r in rows)
    return result


def load_actors(config, env, model_dir):
    import torch
    from harl.algorithms.actors import ALGO_REGISTRY
    settings = config["algo_args"]
    actors = []
    for i in range(env.n_agents):
        agent = ALGO_REGISTRY[config["main_args"]["algo"]](
            {**settings["model"], **settings["algo"]}, env.observation_space[i],
            env.action_space[i], device=torch.device("cpu"))
        agent.actor.load_state_dict(torch.load(model_dir/f"actor_agent{i}.pt", map_location="cpu", weights_only=True))
        agent.prep_rollout()
        actors.append(agent)
    return actors


def evaluate(out, attempt):
    manifest = verify_run(out)
    activate_runtime()
    import numpy as np
    import torch
    from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv
    from harl.envs.uav_escs.SC.rules import rule_actions
    torch.set_num_threads(1)
    destination = out / "evaluation" / f"attempt_{attempt:02d}"
    destination.mkdir(parents=True, exist_ok=False)
    names = [m[0] for m in METHODS] + RULES
    summaries, pairing = [], {}
    started = time.monotonic()
    total = len(names)*len(manifest["scenarios"])*len(manifest["evaluation_seeds"])
    checkpoint_hashes = {}
    for name in names:
        learned = name not in RULES
        config = read(out / "configs" / f"{name if learned else 'IC_HAPPO'}.json")
        env_args = copy.deepcopy(config["env_args"])
        env_args["scheduler"] = "round_robin" if name == "RoundRobin" else "aoi"
        env_args["instruction_mode_strategy"] = "explicit_evaluation"
        env_args["explicit_instruction_schedule"] = [[0, 0]]
        env = SCUAVEnv(env_args)
        actors = None
        if learned:
            status = read(out / "jobs" / name / "status.json")
            verify_checkpoint(status)
            checkpoint_hashes[name] = status["checkpoint_hashes"]
            actors = load_actors(config, env, Path(status["model_dir"]))
        for scenario, schedule in manifest["scenarios"].items():
            env.explicit_instruction_schedule = schedule
            for seed in manifest["evaluation_seeds"]:
                env.seed(seed)
                obs, shared, available = env.reset()
                trace = hashlib.sha256()
                for array in (env.pos_ds, env.pos_uav, env.owner_uav, env.q_cache, env.tau_cache):
                    trace.update(array.tobytes())
                rnn = np.zeros((env.n_agents, 1, 1, 256), dtype=np.float32)
                masks = np.ones((env.n_agents, 1, 1), dtype=np.float32)
                rng = np.random.default_rng(np.random.SeedSequence([seed, 719]))
                rows = []
                for slot in range(env.max_steps):
                    trace.update(env.gamma_uav_sut.tobytes())
                    trace.update(np.asarray([env.gamma_sut_sat, env.current_instruction_id], dtype=np.float64).tobytes())
                    if actors is None:
                        action = rule_actions(env, obs, available, name, rng)
                    else:
                        action = []
                        with torch.no_grad():
                            for i, actor in enumerate(actors):
                                a, state = actor.act(obs[i:i+1], rnn[i], masks[i],
                                                     available[i:i+1], deterministic=True)
                                action.append(a.cpu().numpy().reshape(-1))
                                rnn[i] = state.cpu().numpy()
                    obs, shared, reward, done, infos, available = env.step(action)
                    if bool(np.all(done)) != (slot == env.max_steps-1):
                        raise AssertionError("Episode did not terminate at 600 slots")
                    info = infos[0]
                    expected_g = next(g for t, g in reversed(schedule) if t <= slot)
                    if info["instruction_id"] != expected_g:
                        raise AssertionError("Wrong instruction at schedule boundary")
                    row = dict(slot=slot, instruction_id=expected_g, **external_metrics(env, info))
                    row.update(proposed_sut_logits=json.dumps(env.last_sut_raw_action.tolist()),
                               proposed_modes=json.dumps(np.argmax(env.last_uav_raw_actions, axis=1).tolist()),
                               executed_modes=json.dumps(env.last_selected_modes.tolist()),
                               selected_ds=json.dumps(info["selected_ds"]),
                               resource_fractions=json.dumps(env.beta_sut_sat.tolist()))
                    rows.append(row)
                pair_key = (scenario, seed)
                if pair_key in pairing and pairing[pair_key] != trace.hexdigest():
                    raise AssertionError(f"Unpaired external trajectory for {name} {pair_key}")
                pairing[pair_key] = trace.hexdigest()
                file = destination / "traces" / name / f"{scenario}_seed{seed}.csv"
                file.parent.mkdir(parents=True, exist_ok=True)
                with file.open("x", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                summary = dict(method=name, scenario=scenario, seed=seed, steps=len(rows),
                               external_trajectory_sha256=trace.hexdigest(), trace_file=str(file),
                               trace_sha256=sha(file), **aggregate(rows))
                summary["by_instruction"] = {str(g): dict(steps=len(part), **aggregate(part))
                    for g in range(3) if (part := [r for r in rows if r["instruction_id"] == g])}
                summaries.append(summary)
                write(destination / "progress.json", dict(state="evaluating", method=name, scenario=scenario,
                      completed_episodes=len(summaries), total_episodes=total,
                      elapsed_seconds=time.monotonic()-started, updated_utc=stamp()))
            print(f"evaluation {name}: {scenario}; {len(summaries)}/{total} episodes", flush=True)
        env.close()
    grouped = {name: aggregate([row for row in summaries if row["method"] == name]) for name in names}
    by_scenario = {name: {s: aggregate([r for r in summaries if r["method"] == name and r["scenario"] == s])
                          for s in manifest["scenarios"]} for name in names}
    write(destination / "episode_summary.json", summaries)
    write(destination / "comparison.json", dict(overall=grouped, by_scenario=by_scenario,
          note="One training seed. Episode means do not measure variation across training seeds. Overall scenarios are equally weighted; inspect per-scenario results."))
    flat = [dict(method=name, **grouped[name]) for name in names]
    with (destination / "comparison.csv").open("x", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    baseline = {(r["scenario"], r["seed"]): r for r in summaries if r["method"] == "IC_HAPPO"}
    paired = [dict(method=r["method"], scenario=r["scenario"], seed=r["seed"],
                   common_reward_difference=r["common_reward"]-baseline[(r["scenario"], r["seed"])]["common_reward"],
                   mean_aoi_difference=r["mean_aoi"]-baseline[(r["scenario"], r["seed"])]["mean_aoi"])
              for r in summaries if r["method"] != "IC_HAPPO"]
    write(destination / "paired_differences_from_IC_HAPPO.json", paired)
    lines = ["单种子方法对比", "", f"训练种子 {manifest['seed']}；每个学习方法 {manifest['steps_per_method']:,} 步。",
             "评估统一使用最终检查点，不按评估成绩选模型。PSNR 是平均表预测值。", "",
             "| 方法 | 共同奖励↑ | 平均 AoI↓ | 每时隙 p95 AoI 的均值↓ | AoI 超限比例↓ | 发送预测 PSNR↑ | 发送数/时隙↑ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in grouped.items():
        q = "N/A" if r["delivered_predicted_psnr"] is None else f"{r['delivered_predicted_psnr']:.3f}"
        lines.append(f"| {name} | {r['common_reward']:.6f} | {r['mean_aoi']:.3f} | {r['p95_aoi']:.3f} | {r['aoi_exceedance_fraction']:.3%} | {q} | {r['deliveries_per_slot']:.3f} |")
    lines += ["", "共同奖励由外部评估重新计算。no_task_aux_reward 的训练奖励不同，不能按其训练曲线直接排名。",
              "单训练种子与烟测只能用于开发检查，不能证明方法优越或训练稳定。各场景与逐时隙结果见同目录 JSON/CSV。"]
    (destination / "comparison.md").write_text("\n".join(lines)+"\n")
    report = dict(state="complete", evaluation_dir=str(destination), completed_episodes=len(summaries),
                  total_evaluation_slots=sum(r["steps"] for r in summaries),
                  paired_external_trajectories=len(pairing), all_executed_constraints_passed=True,
                  checkpoint_hashes=checkpoint_hashes, elapsed_seconds=time.monotonic()-started,
                  summary_hashes={p.name: sha(p) for p in destination.glob("*") if p.suffix in ("json", "csv", "md") and p.name not in ("progress.json", "verification.json")},
                  updated_utc=stamp())
    write(destination / "verification.json", report)
    write(out / "evaluation/status.json", report)
    print(f"Comparison: {destination / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attempt", type=int, required=True)
    args = p.parse_args()
    evaluate(args.output.resolve(), args.attempt)
