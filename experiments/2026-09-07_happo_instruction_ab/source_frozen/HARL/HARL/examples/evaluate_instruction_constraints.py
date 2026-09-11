"""Evaluate instruction-conditioned constraint modes in UAV-ESCS SC/CC."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _suppress_gym_notice_banner() -> None:
    if os.environ.get("HARL_SHOW_GYM_NOTICE", "0") == "1":
        return
    pkg_name = "gym_notices"
    mod_name = "gym_notices.notices"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        if not hasattr(mod, "notices"):
            setattr(mod, "notices", {})
        return
    notices_mod = types.ModuleType(mod_name)
    notices_mod.notices = {}
    pkg_mod = sys.modules.get(pkg_name)
    if pkg_mod is None:
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = []
        sys.modules[pkg_name] = pkg_mod
    sys.modules[mod_name] = notices_mod
    setattr(pkg_mod, "notices", notices_mod)


_suppress_gym_notice_banner()

from harl.algorithms.actors import ALGO_REGISTRY  # noqa: E402
from harl.envs.uav_escs.CC.uav_escs_env_cc import CCUAVEnv  # noqa: E402
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv  # noqa: E402
from harl.utils.configs_tools import get_defaults_yaml_args, update_args  # noqa: E402
from harl.utils.models_tools import init_device


BASE_SCENARIOS = (
    ("fixed_balance", {"instruction_mode_strategy": "fixed", "fixed_instruction_id": 0}),
    ("fixed_aoi", {"instruction_mode_strategy": "fixed", "fixed_instruction_id": 1}),
    ("fixed_quality", {"instruction_mode_strategy": "fixed", "fixed_instruction_id": 2}),
    (
        "switch_200_balance_aoi",
        {
            "instruction_mode_strategy": "switch_once",
            "instruction_switch_step": 200,
            "instruction_before_id": 0,
            "instruction_after_id": 1,
        },
    ),
    (
        "switch_200_balance_quality",
        {
            "instruction_mode_strategy": "switch_once",
            "instruction_switch_step": 200,
            "instruction_before_id": 0,
            "instruction_after_id": 2,
        },
    ),
    (
        "random_switch_once_middle",
        {
            "instruction_mode_strategy": "random_switch_once",
            "instruction_switch_min_step": 200,
            "instruction_switch_max_step": 400,
            "instruction_before_candidates": [0],
            "instruction_after_candidates": [1, 2],
            "allow_same_instruction_after_switch": False,
        },
    ),
)


def _parse_int_csv(text: str) -> List[int]:
    out: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(float(item)))
    return out


def _build_scenarios(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    scenarios = list(BASE_SCENARIOS)
    if not _as_bool(args.include_switch_grid):
        return scenarios
    existing = {name for name, _ in scenarios}
    for switch_step in _parse_int_csv(args.switch_grid_steps):
        for after_id, after_name in ((1, "aoi"), (2, "quality")):
            name = f"switch_{switch_step}_balance_{after_name}"
            if name in existing:
                continue
            scenarios.append(
                (
                    name,
                    {
                        "instruction_mode_strategy": "switch_once",
                        "instruction_switch_step": int(switch_step),
                        "instruction_before_id": 0,
                        "instruction_after_id": int(after_id),
                    },
                )
            )
            existing.add(name)
    return scenarios


def _process_cli_value(value: str) -> Any:
    try:
        return eval(value)  # noqa: S307 - match examples/train.py CLI behavior.
    except Exception:
        return value


def _parse_unknown_args(items: Iterable[str]) -> Dict[str, Any]:
    items = list(items)
    keys = [item[2:] for item in items[0::2] if item.startswith("--")]
    values = [_process_cli_value(item) for item in items[1::2]]
    return {key: value for key, value in zip(keys, values)}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return bool(value)


def _normalize_model_dir(model_dir: str) -> Path:
    path = Path(model_dir).expanduser().resolve()
    if (path / "models").is_dir():
        return path / "models"
    return path


def _load_config(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    main_args = {"algo": args.algo, "env": args.env, "exp_name": args.exp_name}
    config_path = Path(args.load_config).expanduser() if args.load_config else None
    model_dir = _normalize_model_dir(args.model_dir) if args.model_dir else None
    if config_path is None and model_dir is not None:
        for candidate in (model_dir.parent / "config.json", model_dir / "config.json"):
            if candidate.exists():
                config_path = candidate
                break
    if config_path is not None and str(config_path):
        with config_path.open("r", encoding="utf-8") as file:
            all_config = json.load(file)
        main_args.update(all_config.get("main_args", {}))
        return main_args, all_config["algo_args"], all_config["env_args"]
    algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)
    return main_args, algo_args, env_args


def _find_latest_model_dir(env_name: str, algo: str) -> Optional[Path]:
    root = Path(__file__).resolve().parent / "results"
    if not root.exists():
        return None
    pattern = f"{env_name}/**/{algo}/**/models/actor_agent0.pt"
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0].parent if matches else None


def _make_env(env_name: str, env_args: Dict[str, Any]):
    if env_name == "uav_escs_sc":
        return SCUAVEnv(env_args)
    if env_name == "uav_escs_cc":
        return CCUAVEnv(env_args)
    raise ValueError(f"Unsupported UAV-ESCS env: {env_name}")


def _load_actors(
    *,
    algo: str,
    algo_args: Dict[str, Any],
    env,
    model_dir: Path,
    device: torch.device,
) -> Optional[List[Any]]:
    actor_files = [model_dir / f"actor_agent{i}.pt" for i in range(env.n_agents)]
    missing = [str(path) for path in actor_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Requested policy has missing actor weights: {missing}")
    actor_args = {**algo_args["model"], **algo_args["algo"]}
    actors = []
    for agent_id in range(env.n_agents):
        agent = ALGO_REGISTRY[algo](
            actor_args,
            env.observation_space[agent_id],
            env.action_space[agent_id],
            device=device,
        )
        state_dict = torch.load(actor_files[agent_id], map_location=device, weights_only=True)
        agent.actor.load_state_dict(state_dict)
        agent.prep_rollout()
        actors.append(agent)
    return actors


def _actor_actions(
    actors: List[Any],
    obs: np.ndarray,
    available_actions: np.ndarray,
    rnn_states: np.ndarray,
    masks: np.ndarray,
    deterministic: bool,
) -> Tuple[List[np.ndarray], np.ndarray]:
    actions = []
    next_rnn_states = rnn_states.copy()
    with torch.no_grad():
        for agent_id, actor in enumerate(actors):
            action, rnn_state = actor.act(
                obs[agent_id : agent_id + 1],
                rnn_states[agent_id],
                masks[agent_id],
                available_actions[agent_id : agent_id + 1],
                deterministic=deterministic,
            )
            actions.append(action.detach().cpu().numpy().reshape(-1))
            next_rnn_states[agent_id] = rnn_state.detach().cpu().numpy()
    return actions, next_rnn_states


def _safe_min_lambda(env, n: int, k: int) -> float:
    if hasattr(env, "feasible_modes_for_stream"):
        feasible = np.asarray(env.feasible_modes_for_stream(n, k), dtype=bool)
    else:
        gid = env._clip_instruction_id(env.current_instruction_id)
        bucket_id = env._effective_snr_bucket_for_uav(n)
        q_req = env._q_req_for_instruction_bucket(gid, bucket_id)
        feasible = np.asarray(env.Q_hat_rec[n, k] >= q_req - 1.0e-9, dtype=bool)
    if not np.any(feasible):
        return 0.0
    return float(np.min(env.Lambda_sem[n, k, feasible]))


def _heuristic_actions(env, available_actions: np.ndarray) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    sut_logits = np.zeros(env.sut_act_dim_total, dtype=np.float32)
    for n in range(env.n_uav):
        ds = env.ds_by_uav[n]
        pending_load = sum(_safe_min_lambda(env, n, int(k)) for k in ds if env.q_cache[n, k] > 0)
        max_aoi = float(np.max(env.A_rcc[ds]) / env.A_max) if ds.size else 0.0
        sut_logits[n] = float(pending_load / max(env.Lambda_ref, 1.0) + max_aoi)
    actions.append(sut_logits)

    for n in range(env.n_uav):
        action = np.full(env.uav_act_dim_total, -1.0, dtype=np.float32)
        ds = env.ds_by_uav[n]
        mode_scores = np.full(env.n_semantic_modes, -10.0, dtype=np.float32)
        bucket_id = env._effective_snr_bucket_for_uav(n) if hasattr(env, "_effective_snr_bucket_for_uav") else 0
        gid = env._clip_instruction_id(env.current_instruction_id)
        q_req = env._q_req_for_instruction_bucket(gid, bucket_id)
        for m in range(env.n_semantic_modes):
            if hasattr(env, "candidate_mask"):
                full_candidate = env.candidate_mask(n, m, env.q_cache, q_req, include_load_budget=True)
                candidate = full_candidate[ds]
            else:
                candidate = (env.q_cache[n, ds] > 0) & (env.Q_hat_rec[n, ds, m] >= q_req - 1.0e-9)
                candidate &= env.Lambda_sem[n, ds, m] <= env.Phi_bh[n] + 1.0e-9
            if not np.any(candidate):
                continue
            q_score = float(np.mean(env.Q_hat_rec[n, ds[candidate], m] / env.Q_max))
            l_score = float(np.mean(env.Lambda_sem[n, ds[candidate], m] / env.Lambda_ref))
            mode_scores[m] = q_score - 0.35 * l_score
        mu_dim = int(getattr(env, "n_mu_modes", min(4, env.n_semantic_modes)))
        mu_scores = np.full(mu_dim, -10.0, dtype=np.float32)
        for mu_id in range(mu_dim):
            mode_id = env._global_mode_id(bucket_id, mu_id) if hasattr(env, "_global_mode_id") else mu_id
            mu_scores[mu_id] = mode_scores[mode_id]
        action[env.uav_act_slices["mode_logits"]] = mu_scores
        if "ds_priority_logits" in env.uav_act_slices:
            action[env.uav_act_slices["ds_priority_logits"]] = (env.A_rcc[ds] / max(float(env.A_max), 1.0)).astype(
                np.float32
            )
        else:
            action[env.uav_act_slices["scheduling_weights"]] = np.asarray([1.0, 1.0, 0.5], dtype=np.float32)
        mask = available_actions[1 + n, : env.uav_act_dim_total] > 0.5
        action = np.where(mask, action, -10.0).astype(np.float32)
        actions.append(action)
    return actions


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_json_ready(key)): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _suffix_path(path: Path, suffix: str) -> Path:
    suffix = str(suffix).strip()
    if not suffix:
        return path
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _scheduled_q_hat(info: Dict[str, Any]) -> List[float]:
    chi = np.asarray(info.get("chi", []), dtype=int)
    q_hat = np.asarray(info.get("Q_hat_rec", []), dtype=float)
    if chi.ndim != 3 or q_hat.shape != chi.shape:
        return []
    return [float(q_hat[n, k, m]) for n, k, m in np.argwhere(chi > 0)]


def _scheduled_modes_and_buckets(info: Dict[str, Any]) -> tuple[List[int], List[int]]:
    chi = np.asarray(info.get("chi", []), dtype=int)
    snr_bucket = np.asarray(info.get("snr_bucket_ids", []), dtype=int).reshape(-1)
    if chi.ndim != 3:
        return [], []
    modes: List[int] = []
    buckets: List[int] = []
    for n, _k, m in np.argwhere(chi > 0):
        modes.append(int(m))
        buckets.append(int(np.clip(snr_bucket[n], 0, 3)) if n < snr_bucket.size else 0)
    return modes, buckets


def _summarize_rows(rows: List[Dict[str, Any]], n_mu: int, n_modes: int) -> Dict[str, Any]:
    mu_counts = np.zeros(n_mu, dtype=int)
    mode_counts = np.zeros(n_modes, dtype=int)
    bucket_mode_counts = np.zeros((4, n_modes), dtype=int)
    alpha_sum = np.zeros(3, dtype=float)
    alpha_count = 0
    q_hats: List[float] = []

    for row in rows:
        q_hats.extend(float(v) for v in row.get("raw_q_hat_values", []))
        for mu in row.get("selected_mu_ids", []):
            mu = int(mu)
            if 0 <= mu < n_mu:
                mu_counts[mu] += 1
        modes = [int(v) for v in row.get("scheduled_modes", [])]
        buckets = [int(v) for v in row.get("scheduled_snr_buckets", [])]
        for idx, mode in enumerate(modes):
            if 0 <= mode < n_modes:
                mode_counts[mode] += 1
                bucket = int(np.clip(buckets[idx], 0, 3)) if idx < len(buckets) else 0
                bucket_mode_counts[bucket, mode] += 1
        alphas = np.asarray(row.get("scheduling_alphas", []), dtype=float)
        if alphas.ndim == 2 and alphas.shape[1] >= 3:
            alpha_sum += np.sum(alphas[:, :3], axis=0)
            alpha_count += alphas.shape[0]

    scheduled_total = float(np.sum([r["scheduled_count"] for r in rows])) if rows else 0.0
    lambda_total = float(np.sum([r["lambda_usage"] for r in rows])) if rows else 0.0
    return {
        "slots": len(rows),
        "reward": float(np.mean([r["reward"] for r in rows])) if rows else 0.0,
        "avg_quality_gain": float(np.mean([r["avg_quality_gain"] for r in rows])) if rows else 0.0,
        "raw_Q_hat_mean": float(np.mean(q_hats)) if q_hats else 0.0,
        "mean_AoI": float(np.mean([r["mean_aoi"] for r in rows])) if rows else 0.0,
        "max_AoI": float(np.mean([r["max_aoi"] for r in rows])) if rows else 0.0,
        "scheduled_count": float(np.mean([r["scheduled_count"] for r in rows])) if rows else 0.0,
        "lambda_usage": float(np.mean([r["lambda_usage"] for r in rows])) if rows else 0.0,
        "lambda_per_stream": float(lambda_total / max(scheduled_total, 1.0e-9)) if rows else 0.0,
        "selected_mu_id_distribution": {str(i): int(v) for i, v in enumerate(mu_counts)},
        "selected_mode_distribution": {str(i): int(v) for i, v in enumerate(mode_counts) if v > 0},
        "mode_distribution_by_SNR_bucket": {
            str(b): {str(i): int(v) for i, v in enumerate(bucket_mode_counts[b]) if v > 0}
            for b in range(4)
        },
        "scheduling_alphas_mean": (alpha_sum / max(alpha_count, 1)).astype(float).tolist(),
        "quality_violation_bucketwise": float(np.mean([r["quality_violation_bucketwise"] for r in rows])) if rows else 0.0,
        "aoi_violation_max": float(np.mean([r["aoi_violation_max"] for r in rows])) if rows else 0.0,
        "load_violation": float(np.mean([r["load_violation"] for r in rows])) if rows else 0.0,
        "recv_aoi_bonus": float(np.mean([r["recv_aoi_bonus"] for r in rows])) if rows else 0.0,
        "reward_quality": float(np.mean([r["reward_quality"] for r in rows])) if rows else 0.0,
        "reward_load_cost": float(np.mean([r["reward_load_cost"] for r in rows])) if rows else 0.0,
        "reward_aoi_penalty": float(np.mean([r["reward_aoi_penalty"] for r in rows])) if rows else 0.0,
        "penalty_quality_violation": float(np.mean([r["penalty_quality_violation"] for r in rows])) if rows else 0.0,
        "penalty_aoi_violation": float(np.mean([r["penalty_aoi_violation"] for r in rows])) if rows else 0.0,
        "penalty_load_violation": float(np.mean([r["penalty_load_violation"] for r in rows])) if rows else 0.0,
        "bonus_aoi_reduction": float(np.mean([r["bonus_aoi_reduction"] for r in rows])) if rows else 0.0,
        "fallback_to_best_feasible_mode_count": float(
            np.mean([r["fallback_to_best_feasible_mode_count"] for r in rows])
        )
        if rows
        else 0.0,
    }


def _switch_segment_summaries(
    rows: List[Dict[str, Any]],
    switch_step: int,
    n_mu: int,
    n_modes: int,
) -> Dict[str, Any]:
    segments = {
        "pre_switch_last_100": [
            row for row in rows if switch_step - 100 <= int(row["slot"]) <= switch_step - 1
        ],
        "post_switch_first_100": [
            row for row in rows if switch_step <= int(row["slot"]) <= switch_step + 99
        ],
        "post_switch_all": [row for row in rows if int(row["slot"]) >= switch_step],
    }
    return {name: _summarize_rows(segment_rows, n_mu, n_modes) for name, segment_rows in segments.items()}


def _switch_segment_summaries_from_rows(
    rows: List[Dict[str, Any]],
    n_mu: int,
    n_modes: int,
) -> Dict[str, Any]:
    segments = {
        "pre_switch_last_100": [],
        "post_switch_first_100": [],
        "post_switch_all": [],
    }
    for row in rows:
        switch_step = int(row.get("instruction_switch_step", 200))
        slot = int(row["slot"])
        if switch_step - 100 <= slot <= switch_step - 1:
            segments["pre_switch_last_100"].append(row)
        if switch_step <= slot <= switch_step + 99:
            segments["post_switch_first_100"].append(row)
        if slot >= switch_step:
            segments["post_switch_all"].append(row)
    return {name: _summarize_rows(segment_rows, n_mu, n_modes) for name, segment_rows in segments.items()}


def _run_scenario(
    *,
    name: str,
    scenario_overrides: Dict[str, Any],
    base_main_args: Dict[str, Any],
    base_algo_args: Dict[str, Any],
    base_env_args: Dict[str, Any],
    actors,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    env_args = dict(base_env_args)
    env_args.update(
        {
            "use_instruction_constraints": True,
            "include_instruction_id_in_obs": True,
            "include_instruction_constraints_in_obs": True,
            "use_direct_ds_logits": False,
            "scheduling_weight_transform": "softmax",
        }
    )
    env_args.update(scenario_overrides)
    if args.episode_length > 0:
        env_args["T"] = float(args.episode_length) * float(env_args.get("delta_T", 1.0))

    env = _make_env(base_main_args["env"], env_args)
    if actors is not None:
        actors = _load_actors(
            algo=base_main_args["algo"],
            algo_args=base_algo_args,
            env=env,
            model_dir=args._resolved_model_dir,
            device=args._device,
        )

    recurrent_n = int(base_algo_args["model"].get("recurrent_n", 1))
    hidden_size = int(base_algo_args["model"].get("hidden_sizes", [128])[-1])
    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        env.seed(int(args.seed) + ep)
        obs, _, available_actions = env.reset()
        rnn_states = np.zeros((env.n_agents, 1, recurrent_n, hidden_size), dtype=np.float32)
        masks = np.ones((env.n_agents, 1, 1), dtype=np.float32)
        done = False
        while not done:
            if actors is not None:
                actions, rnn_states = _actor_actions(
                    actors,
                    obs,
                    available_actions,
                    rnn_states,
                    masks,
                    deterministic=bool(args.deterministic),
                )
            else:
                actions = _heuristic_actions(env, available_actions)
            obs, _, _, dones, infos, available_actions = env.step(actions)
            info = infos[0]
            raw_q_hat_values = _scheduled_q_hat(info)
            scheduled_modes, scheduled_snr_buckets = _scheduled_modes_and_buckets(info)
            rows.append(
                {
                    "scenario": name,
                    "episode": ep,
                    "slot": int(info["slot"]),
                    "instruction_mode_strategy": str(info.get("instruction_mode_strategy", "")),
                    "instruction_switch_step": int(info.get("instruction_switch_step", 0)),
                    "instruction_before_id": int(info.get("instruction_before_id", 0)),
                    "instruction_after_id": int(info.get("instruction_after_id", 0)),
                    "is_instruction_switch_step": bool(info.get("is_instruction_switch_step", False)),
                    "instruction_id": int(info.get("instruction_id", 0)),
                    "instruction_name": str(info.get("instruction_name", "")),
                    "reward": float(info["reward"]),
                    "avg_quality_gain": float(info.get("avg_quality_gain", 0.0)),
                    "raw_q_hat_mean": float(np.mean(raw_q_hat_values)) if raw_q_hat_values else 0.0,
                    "raw_q_hat_values": raw_q_hat_values,
                    "mean_aoi": float(info.get("mean_AoI", info.get("avg_aoi", 0.0))),
                    "max_aoi": float(info.get("max_AoI", info.get("max_aoi", 0.0))),
                    "scheduled_count": int(info.get("scheduled_count", 0)),
                    "lambda_usage": float(np.sum(np.asarray(info.get("lambda_usage", []), dtype=float))),
                    "selected_mu_ids": np.asarray(info.get("selected_mu_ids", []), dtype=int).reshape(-1).tolist(),
                    "selected_modes": np.asarray(info.get("selected_modes", []), dtype=int).reshape(-1).tolist(),
                    "scheduled_modes": scheduled_modes,
                    "scheduled_snr_buckets": scheduled_snr_buckets,
                    "scheduling_alphas": np.asarray(info.get("scheduling_alphas", []), dtype=float).tolist(),
                    "quality_violation_bucketwise": float(info.get("quality_violation_bucketwise", 0.0)),
                    "aoi_violation_max": float(info.get("aoi_violation_max", 0.0)),
                    "load_violation": float(info.get("load_violation", 0.0)),
                    "recv_aoi_bonus": float(info.get("recv_aoi_bonus", 0.0)),
                    "feasibility_rule": str(info.get("feasibility_rule", "")),
                    "quality_feasibility_reference": str(info.get("quality_feasibility_reference", "")),
                    "lambda_constraint": str(info.get("lambda_constraint", "")),
                    "reward_quality": float(info.get("reward_quality", info.get("quality_contrib", 0.0))),
                    "reward_load_cost": float(info.get("reward_load_cost", info.get("load_penalty", 0.0))),
                    "reward_aoi_penalty": float(info.get("reward_aoi_penalty", info.get("aoi_penalty", 0.0))),
                    "penalty_quality_violation": float(info.get("penalty_quality_violation", 0.0)),
                    "penalty_aoi_violation": float(info.get("penalty_aoi_violation", 0.0)),
                    "penalty_load_violation": float(info.get("penalty_load_violation", 0.0)),
                    "bonus_aoi_reduction": float(info.get("bonus_aoi_reduction", 0.0)),
                    "reward_component_sum": float(info.get("reward_component_sum", info["reward"])),
                    "eval_seed": int(args.seed),
                    "eval_config_path": str(Path(args.load_config).expanduser().resolve()) if args.load_config else "defaults",
                    "eval_checkpoint_path": str(args._resolved_model_dir) if args._resolved_model_dir is not None else "heuristic",
                    "fallback_to_best_feasible_mode_count": int(
                        info.get("fallback_to_best_feasible_mode_count", 0)
                    ),
                }
            )
            done = bool(np.all(dones))
            masks[:] = 0.0 if done else 1.0

    env.close()
    summary = {"scenario": name, **_summarize_rows(rows, int(env.n_mu_modes), int(env.n_semantic_modes))}
    if scenario_overrides.get("instruction_mode_strategy") in ("switch_once", "random_switch_once"):
        switch_step = int(scenario_overrides.get("instruction_switch_step", env_args.get("instruction_switch_step", 200)))
        if scenario_overrides.get("instruction_mode_strategy") == "random_switch_once":
            summary["switch_segments"] = _switch_segment_summaries_from_rows(
                rows,
                int(env.n_mu_modes),
                int(env.n_semantic_modes),
            )
            summary["sampled_switch_steps"] = sorted(
                {
                    int(row.get("instruction_switch_step", switch_step))
                    for row in rows
                    if int(row.get("slot", -1)) == 0
                }
            )
        else:
            summary["switch_segments"] = _switch_segment_summaries(
                rows,
                switch_step,
                int(env.n_mu_modes),
                int(env.n_semantic_modes),
            )
    summary["rows"] = rows
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, default="uav_escs_sc", choices=["uav_escs_sc", "uav_escs_cc"])
    parser.add_argument("--algo", type=str, default="happo", choices=["happo", "mappo"])
    parser.add_argument("--exp_name", type=str, default="instruction_eval")
    parser.add_argument("--load_config", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="heuristic", help="'heuristic', 'latest', run dir, or models dir.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode_length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="examples/instruction_eval")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="consistent_mask",
        help="Suffix for corrected-result files. Use an empty string only in a fresh output directory.",
    )
    parser.add_argument("--deterministic", type=_as_bool, default=True)
    parser.add_argument("--cuda", type=_as_bool, default=False)
    parser.add_argument(
        "--include_switch_grid",
        type=_as_bool,
        default=True,
        help="Also evaluate deterministic switch_once trajectories at --switch_grid_steps.",
    )
    parser.add_argument(
        "--switch_grid_steps",
        type=str,
        default="250,300,350",
        help="Comma-separated switch steps for balance->AoI and balance->quality trajectory probes.",
    )
    args, unknown = parser.parse_known_args()

    overrides = _parse_unknown_args(unknown)
    main_args, algo_args, env_args = _load_config(args)
    update_args(overrides, algo_args, env_args, override=True)
    main_args["env"] = args.env
    main_args["algo"] = args.algo
    algo_args.setdefault("device", {})
    algo_args["device"]["cuda"] = bool(args.cuda)
    algo_args["device"].setdefault("torch_threads", 1)
    args._device = init_device(algo_args["device"])

    model_text = args.model_dir.strip().lower()
    if model_text in ("", "none", "heuristic"):
        args._resolved_model_dir = None
        actors_probe = None
    elif model_text == "latest":
        args._resolved_model_dir = _find_latest_model_dir(main_args["env"], main_args["algo"])
        actors_probe = object() if args._resolved_model_dir is not None else None
    else:
        args._resolved_model_dir = _normalize_model_dir(args.model_dir)
        actors_probe = object()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_rows: List[Dict[str, Any]] = []
    for scenario_name, scenario_overrides in _build_scenarios(args):
        summary = _run_scenario(
            name=scenario_name,
            scenario_overrides=scenario_overrides,
            base_main_args=main_args,
            base_algo_args=algo_args,
            base_env_args=env_args,
            actors=actors_probe,
            args=args,
        )
        all_rows.extend(summary.pop("rows"))
        summaries.append(summary)

    csv_path = _suffix_path(output_dir / "instruction_constraint_rollout.csv", args.output_suffix)
    summary_path = _suffix_path(output_dir / "instruction_constraint_summary.json", args.output_suffix)
    _write_csv(csv_path, all_rows)
    run_metadata = {
        "seed": int(args.seed),
        "config_path": str(Path(args.load_config).expanduser().resolve()) if args.load_config else "defaults",
        "checkpoint_path": str(args._resolved_model_dir) if args._resolved_model_dir is not None else "heuristic",
        "feasibility_rule": "Q_hat_rec >= Q_req(instruction, snr_bucket)",
        "output_suffix": str(args.output_suffix),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready({"metadata": run_metadata, "scenarios": summaries}), file, indent=2, sort_keys=True)

    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")
    for item in summaries:
        print(
            f"{item['scenario']}: quality={item['avg_quality_gain']:.4f}, "
            f"raw_Q={item['raw_Q_hat_mean']:.3f}, mean_AoI={item['mean_AoI']:.3f}, "
            f"max_AoI={item['max_AoI']:.3f}, scheduled={item['scheduled_count']:.2f}, "
            f"lambda={item['lambda_usage']:.1f}, fallback={item['fallback_to_best_feasible_mode_count']:.3f}"
        )
        for segment_name, segment in item.get("switch_segments", {}).items():
            print(
                f"  {segment_name}: slots={segment['slots']}, "
                f"quality={segment['avg_quality_gain']:.4f}, raw_Q={segment['raw_Q_hat_mean']:.3f}, "
                f"mean_AoI={segment['mean_AoI']:.3f}, max_AoI={segment['max_AoI']:.3f}, "
                f"scheduled={segment['scheduled_count']:.2f}, lambda={segment['lambda_usage']:.1f}, "
                f"fallback={segment['fallback_to_best_feasible_mode_count']:.3f}"
            )


if __name__ == "__main__":
    main()
