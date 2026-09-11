"""Evaluate algorithm-level baselines for UAV-ESCS.

These baselines compare decision rules on top of the same environment model:
SUT budget allocation, semantic-mode selection, and per-UAV stream scheduling.
They intentionally do not change the communication/channel layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

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

from harl.envs.uav_escs.CC.uav_escs_env_cc import CCUAVEnv  # noqa: E402
from harl.envs.uav_escs.SC.uav_escs_env_sc import SCUAVEnv  # noqa: E402
from harl.utils.configs_tools import get_defaults_yaml_args, update_args  # noqa: E402


BaselineFn = Callable[[Any, np.ndarray, np.random.Generator], List[np.ndarray]]


def _process_cli_value(value: str) -> Any:
    try:
        return eval(value)  # noqa: S307 - match examples/train.py CLI behavior.
    except Exception:
        return value


def _parse_unknown_args(items: Iterable[str]) -> Dict[str, Any]:
    items = list(items)
    keys = [item[2:] for item in items[0::2] if item.startswith("--")]
    values = [_process_cli_value(item) for item in items[1::2]]
    return {k: v for k, v in zip(keys, values)}


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


def _load_config(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    main_args = {"algo": args.algo, "env": args.env, "exp_name": args.exp_name}
    if args.load_config:
        with Path(args.load_config).expanduser().open("r", encoding="utf-8") as file:
            all_config = json.load(file)
        main_args.update(all_config.get("main_args", {}))
        return main_args, all_config["algo_args"], all_config["env_args"]

    algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)
    return main_args, algo_args, env_args


def _make_env(env_name: str, env_args: Dict[str, Any]):
    if env_name == "uav_escs_sc":
        return SCUAVEnv(env_args)
    if env_name == "uav_escs_cc":
        return CCUAVEnv(env_args)
    raise ValueError(f"Unsupported UAV-ESCS env: {env_name}")


def _mask_action(action: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[: mask.size] = np.where(mask > 0.5, out[: mask.size], -10.0)
    return out


def _empty_uav_action(env, fill: float = -1.0) -> np.ndarray:
    return np.full(env.uav_act_dim_total, fill, dtype=np.float32)


def _soft_priority_logits(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.astype(np.float32)
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(scores, dtype=np.float32)
    finite = np.where(np.isfinite(scores), scores, np.nanmin(scores[np.isfinite(scores)]))
    spread = float(np.max(finite) - np.min(finite))
    if spread <= 1.0e-12:
        return np.zeros_like(finite, dtype=np.float32)
    return (2.0 * (finite - np.min(finite)) / spread - 1.0).astype(np.float32)


def _q_req_for_uav(env, n: int) -> float:
    gid = env._clip_instruction_id(env.current_instruction_id)
    bucket_id = env._effective_snr_bucket_for_uav(n)
    return float(env._q_req_for_instruction_bucket(gid, bucket_id))


def _candidate_mask(env, n: int, mode: int, q_cache: np.ndarray | None = None) -> np.ndarray:
    q = env.q_cache if q_cache is None else q_cache
    if hasattr(env, "candidate_mask"):
        return env.candidate_mask(n, mode, q, _q_req_for_uav(env, n), include_load_budget=True)
    return env.owner_mask[n] & (q[n] > 0) & (env.Q_hat_rec[n, :, mode] >= _q_req_for_uav(env, n) - 1.0e-9) & (
        env.Lambda_sem[n, :, mode] <= env.Phi_bh[n] + 1.0e-9
    )


def _q_min(env) -> float:
    return float(getattr(env, "Q_min", 30.0))


def _omega_tuple(env) -> Tuple[float, float, float]:
    return (
        float(getattr(env, "omega_Q", 0.50)),
        float(getattr(env, "omega_Lambda", 0.10)),
        float(getattr(env, "omega_A", 0.40)),
    )


def _stream_quality_value(env, n: int, k: int, mode: int) -> float:
    q_min = _q_min(env)
    denom = max(float(env.Q_max - q_min), 1.0e-9)
    return max((float(env.Q_hat_rec[n, k, mode]) - q_min) / denom, 0.0)


def _stream_load_cost(env, n: int, k: int, mode: int) -> float:
    return float(env.Lambda_sem[n, k, mode]) / max(float(env.Lambda_ref), 1.0e-9)


def _stream_aoi_priority(env, n: int, k: int) -> float:
    cache_age = 0.0
    if int(env.tau_cache[n, k]) >= 0:
        cache_age = max(0.0, float(env.current_step - int(env.tau_cache[n, k]))) / max(float(env.A_max), 1.0)
    return float(env.A_rcc[k]) / max(float(env.A_max), 1.0) + cache_age


def _best_mode_by_score(env, n: int, mode_scores: np.ndarray) -> int:
    mask = np.isfinite(mode_scores)
    if not np.any(mask):
        return 0
    return int(np.argmax(np.where(mask, mode_scores, -np.inf)))


def _mode_logits_from_scores(env, n: int, mode_scores: np.ndarray) -> np.ndarray:
    mu_dim = int(getattr(env, "n_mu_modes", min(4, env.n_semantic_modes)))
    logits = np.full(mu_dim, -10.0, dtype=np.float32)
    bucket_id = env._effective_snr_bucket_for_uav(n) if hasattr(env, "_effective_snr_bucket_for_uav") else 0
    mu_scores = np.full(mu_dim, -np.inf, dtype=float)
    for mu_id in range(mu_dim):
        mode_id = env._global_mode_id(bucket_id, mu_id) if hasattr(env, "_global_mode_id") else mu_id
        if 0 <= mode_id < len(mode_scores):
            mu_scores[mu_id] = float(mode_scores[mode_id])
    finite = np.isfinite(mu_scores)
    if np.any(finite):
        logits[finite] = _soft_priority_logits(mu_scores[finite])
    return logits


def _set_scheduling_weights(env, action: np.ndarray, alpha_a: float, alpha_q: float, alpha_l: float) -> None:
    if "ds_priority_logits" in env.uav_act_slices:
        n = int(getattr(env, "_baseline_action_uav", 0))
        ds = env.ds_by_uav[n]
        scores = np.zeros(ds.size, dtype=float)
        q_min = _q_min(env)
        denom_q = max(float(env.Q_max - q_min), 1.0e-9)
        for idx, k in enumerate(ds):
            if hasattr(env, "feasible_modes_for_stream"):
                feasible = env.feasible_modes_for_stream(n, int(k))
            else:
                feasible = env.Q_hat_rec[n, int(k)] >= _q_req_for_uav(env, n) - 1.0e-9
            feasible_modes = np.where(feasible & (env.Lambda_sem[n, int(k)] <= env.Phi_bh[n] + 1.0e-9))[0]
            norm_q = 0.0
            norm_l = 0.0
            if feasible_modes.size:
                q_vals = env.Q_hat_rec[n, int(k), feasible_modes]
                l_vals = env.Lambda_sem[n, int(k), feasible_modes]
                best = int(np.argmax(q_vals))
                norm_q = max((float(q_vals[best]) - q_min) / denom_q, 0.0)
                norm_l = float(l_vals[best]) / max(float(env.Lambda_ref), 1.0e-9)
            norm_a = float(env.A_rcc[int(k)]) / max(float(env.A_max), 1.0e-9)
            scores[idx] = float(alpha_a) * norm_a + float(alpha_q) * norm_q - float(alpha_l) * norm_l
            if env.q_cache[n, int(k)] <= 0:
                scores[idx] = -10.0
        action[env.uav_act_slices["ds_priority_logits"]] = _soft_priority_logits(scores)
        return
    key = "scheduling_weights" if "scheduling_weights" in env.uav_act_slices else "schedule_scores"
    sl = env.uav_act_slices[key]
    values = np.asarray([alpha_a, alpha_q, alpha_l], dtype=np.float32)
    width = int(sl.stop - sl.start)
    action[sl] = values[:width] if width <= values.size else np.pad(values, (0, width - values.size))


def _mode_scores_aoi(env, n: int) -> np.ndarray:
    scores = np.full(env.n_semantic_modes, -np.inf, dtype=float)
    for mode in range(env.n_semantic_modes):
        candidate = _candidate_mask(env, n, mode)
        if not np.any(candidate):
            continue
        ds = np.where(candidate)[0]
        # AoI baseline keeps the cheapest feasible mode among stale cached streams.
        scores[mode] = float(np.mean([_stream_aoi_priority(env, n, int(k)) for k in ds])) - 0.15 * float(
            np.mean(env.Lambda_sem[n, ds, mode] / max(env.Lambda_ref, 1.0e-9))
        )
    return scores


def _mode_scores_quality(env, n: int) -> np.ndarray:
    scores = np.full(env.n_semantic_modes, -np.inf, dtype=float)
    for mode in range(env.n_semantic_modes):
        candidate = _candidate_mask(env, n, mode)
        if not np.any(candidate):
            continue
        ds = np.where(candidate)[0]
        scores[mode] = float(np.mean([_stream_quality_value(env, n, int(k), mode) for k in ds]))
    return scores


def _simulate_mode_order_value(env, n: int, mode: int, per_stream_score: np.ndarray) -> float:
    candidate = _candidate_mask(env, n, mode)
    if not np.any(candidate):
        return -np.inf
    ordered_ds = np.where(candidate)[0]
    ordered_ds = ordered_ds[np.argsort(per_stream_score[ordered_ds])[::-1]]
    budget_left = float(env.Phi_bh[n])
    total = 0.0
    for k in ordered_ds:
        load = float(env.Lambda_sem[n, k, mode])
        if load <= budget_left + 1.0e-9:
            budget_left -= load
            total += float(per_stream_score[k])
    return total


def random_baseline(env, available_actions: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    actions: List[np.ndarray] = [rng.normal(0.0, 1.0, size=env.sut_act_dim_total).astype(np.float32)]
    for n in range(env.n_uav):
        action = rng.normal(0.0, 1.0, size=env.uav_act_dim_total).astype(np.float32)
        actions.append(_mask_action(action, available_actions[1 + n, : env.uav_act_dim_total]))
    return actions


def round_robin_baseline(env, available_actions: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    del rng
    actions: List[np.ndarray] = [np.zeros(env.sut_act_dim_total, dtype=np.float32)]
    for n in range(env.n_uav):
        env._baseline_action_uav = n
        action = _empty_uav_action(env, fill=-1.0)
        mode_scores = np.full(env.n_semantic_modes, -np.inf, dtype=float)
        for mode in range(env.n_semantic_modes):
            candidate = _candidate_mask(env, n, mode)
            if np.any(candidate):
                mode_scores[mode] = -float(np.mean(env.Lambda_sem[n, np.where(candidate)[0], mode]))
        action[env.uav_act_slices["mode_logits"]] = _mode_logits_from_scores(env, n, mode_scores)

        _set_scheduling_weights(env, action, 1.0, 0.5, 0.5)
        actions.append(_mask_action(action, available_actions[1 + n, : env.uav_act_dim_total]))
    return actions


def aoi_greedy_baseline(env, available_actions: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    del rng
    beta_scores = np.zeros(env.n_uav, dtype=np.float32)
    for n in range(env.n_uav):
        ds = env.ds_by_uav[n]
        beta_scores[n] = float(np.max(env.A_rcc[ds]) / max(env.A_max, 1.0)) if ds.size else 0.0
        beta_scores[n] += float(np.sum(env.q_cache[n, ds] > 0)) / max(float(ds.size), 1.0)
    actions: List[np.ndarray] = [_soft_priority_logits(beta_scores)]

    for n in range(env.n_uav):
        env._baseline_action_uav = n
        action = _empty_uav_action(env, fill=-1.0)
        mode_scores = _mode_scores_aoi(env, n)
        action[env.uav_act_slices["mode_logits"]] = _mode_logits_from_scores(env, n, mode_scores)

        _set_scheduling_weights(env, action, 2.0, 0.3, 0.2)
        actions.append(_mask_action(action, available_actions[1 + n, : env.uav_act_dim_total]))
    return actions


def quality_greedy_baseline(env, available_actions: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    del rng
    beta_scores = np.zeros(env.n_uav, dtype=np.float32)
    per_uav_mode_scores = []
    for n in range(env.n_uav):
        mode_scores = _mode_scores_quality(env, n)
        per_uav_mode_scores.append(mode_scores)
        beta_scores[n] = float(np.nanmax(np.where(np.isfinite(mode_scores), mode_scores, np.nan))) if np.any(
            np.isfinite(mode_scores)
        ) else 0.0
    actions: List[np.ndarray] = [_soft_priority_logits(beta_scores)]

    for n in range(env.n_uav):
        env._baseline_action_uav = n
        action = _empty_uav_action(env, fill=-1.0)
        mode_scores = per_uav_mode_scores[n]
        action[env.uav_act_slices["mode_logits"]] = _mode_logits_from_scores(env, n, mode_scores)

        _set_scheduling_weights(env, action, 0.3, 2.0, 0.2)
        actions.append(_mask_action(action, available_actions[1 + n, : env.uav_act_dim_total]))
    return actions


def load_greedy_baseline(env, available_actions: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    del rng
    beta_scores = np.zeros(env.n_uav, dtype=np.float32)
    mode_scores_by_uav: List[np.ndarray] = []
    for n in range(env.n_uav):
        mode_scores = np.full(env.n_semantic_modes, -np.inf, dtype=float)
        for mode in range(env.n_semantic_modes):
            candidate = _candidate_mask(env, n, mode)
            if not np.any(candidate):
                continue
            ds = np.where(candidate)[0]
            mode_scores[mode] = -float(np.mean([_stream_load_cost(env, n, int(k), mode) for k in ds]))
        mode_scores_by_uav.append(mode_scores)
        beta_scores[n] = float(np.sum(env.q_cache[n, env.ds_by_uav[n]] > 0))
    actions: List[np.ndarray] = [_soft_priority_logits(beta_scores)]

    for n in range(env.n_uav):
        env._baseline_action_uav = n
        action = _empty_uav_action(env, fill=-1.0)
        mode_scores = mode_scores_by_uav[n]
        action[env.uav_act_slices["mode_logits"]] = _mode_logits_from_scores(env, n, mode_scores)

        _set_scheduling_weights(env, action, 0.2, 0.3, 2.0)
        actions.append(_mask_action(action, available_actions[1 + n, : env.uav_act_dim_total]))
    return actions


def utility_greedy_baseline(env, available_actions: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    del rng
    omega_q, omega_l, omega_a = _omega_tuple(env)

    mode_scores_by_uav: List[np.ndarray] = []
    stream_scores_by_uav: List[np.ndarray] = []
    for n in range(env.n_uav):
        per_stream = np.full(env.n_ds, -np.inf, dtype=float)
        mode_scores = np.full(env.n_semantic_modes, -np.inf, dtype=float)
        for mode in range(env.n_semantic_modes):
            candidate = _candidate_mask(env, n, mode)
            if not np.any(candidate):
                continue
            scores = np.full(env.n_ds, -np.inf, dtype=float)
            for k in np.where(candidate)[0]:
                quality = _stream_quality_value(env, n, int(k), mode)
                load = _stream_load_cost(env, n, int(k), mode)
                freshness = _stream_aoi_priority(env, n, int(k))
                scores[k] = omega_q * quality - omega_l * load + 0.15 * omega_a * freshness
            mode_scores[mode] = _simulate_mode_order_value(env, n, mode, scores)
            if mode_scores[mode] >= np.nanmax(np.where(np.isfinite(mode_scores), mode_scores, np.nan)):
                per_stream = scores
        mode_scores_by_uav.append(mode_scores)
        stream_scores_by_uav.append(per_stream)

    # Keep SUT allocation simple and fixed for the rule baseline; the
    # environment projects zero logits to an equal beta split.
    actions: List[np.ndarray] = [np.zeros(env.sut_act_dim_total, dtype=np.float32)]
    for n in range(env.n_uav):
        env._baseline_action_uav = n
        action = _empty_uav_action(env, fill=-1.0)
        mode_scores = mode_scores_by_uav[n]
        action[env.uav_act_slices["mode_logits"]] = _mode_logits_from_scores(env, n, mode_scores)

        _set_scheduling_weights(env, action, omega_a, omega_q, omega_l)
        actions.append(_mask_action(action, available_actions[1 + n, : env.uav_act_dim_total]))
    return actions


BASELINES: Dict[str, BaselineFn] = {
    "random": random_baseline,
    "round_robin": round_robin_baseline,
    "utility_greedy": utility_greedy_baseline,
}

DEFAULT_BASELINES = ["random", "round_robin", "utility_greedy"]


def _row_from_info(
    baseline: str,
    episode: int,
    info: Dict[str, Any],
    scenario: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    beta = np.asarray(info["beta_sut_sat"], dtype=float)
    usage = np.asarray(info["lambda_usage"], dtype=float)
    modes = np.asarray(info["selected_modes"], dtype=int)
    mu_ids = np.asarray(info.get("selected_mu_ids", []), dtype=int)
    bucket_ids = np.asarray(info.get("snr_bucket_ids", []), dtype=int)
    effective_snr = np.asarray(info.get("effective_snr_db", []), dtype=float)
    scheduling_alphas = np.asarray(info.get("scheduling_alphas", []), dtype=float)
    y = np.asarray(info["y"], dtype=int)
    row: Dict[str, Any] = {
        "scenario": scenario,
        "baseline": baseline,
        "episode": episode,
        "slot": int(info["slot"]),
        "instruction_mode_strategy": str(info.get("instruction_mode_strategy", "")),
        "instruction_switch_step": int(info.get("instruction_switch_step", 0)),
        "instruction_before_id": int(info.get("instruction_before_id", 0)),
        "instruction_after_id": int(info.get("instruction_after_id", 0)),
        "is_instruction_switch_step": bool(info.get("is_instruction_switch_step", False)),
        "instruction_id": int(info.get("instruction_id", 0)),
        "instruction_name": str(info.get("instruction_name", "")),
        "reward_type": str(info.get("reward_type", "fixed_multi_objective")),
        "reward": float(info["reward"]),
        "quality_gain": float(info["quality_gain"]),
        "avg_quality_gain": float(info.get("avg_quality_gain", 0.0)),
        "load_cost": float(info["load_cost"]),
        "A_fair": float(info["A_fair"]),
        "quality_term": float(info.get("quality_term", info["quality_gain"])),
        "load_term": float(info.get("load_term", info["load_cost"])),
        "aoi_term": float(info.get("aoi_term", info["A_fair"])),
        "aoi_state_term": float(info.get("aoi_state_term", info.get("aoi_term", info["A_fair"]))),
        "aoi_mean_term": float(info.get("aoi_mean_term", 0.0)),
        "aoi_max_term": float(info.get("aoi_max_term", 0.0)),
        "aoi_tail_term": float(info.get("aoi_tail_term", 0.0)),
        "fresh_gain_term": float(info.get("fresh_gain_term", 0.0)),
        "quality_contrib": float(info.get("quality_contrib", 0.0)),
        "load_penalty": float(info.get("load_penalty", 0.0)),
        "aoi_penalty": float(info.get("aoi_penalty", 0.0)),
        "avg_aoi": float(info["avg_aoi"]),
        "max_aoi": float(info["max_aoi"]),
        "mean_AoI": float(info.get("mean_AoI", info["avg_aoi"])),
        "max_AoI": float(info.get("max_AoI", info["max_aoi"])),
        "scheduled_count": int(info["scheduled_count"]),
        "selected_modes": modes.tolist(),
        "selected_global_mode_ids": np.asarray(info.get("selected_global_mode_ids", modes), dtype=int).tolist(),
        "selected_mu_ids": mu_ids.tolist(),
        "snr_bucket_ids": bucket_ids.tolist(),
        "effective_snr_db": effective_snr.tolist(),
        "all_mu_infeasible_count": int(info.get("all_mu_infeasible_count", 0)),
        "mode_usage_by_snr_bucket_mu": np.asarray(
            info.get("mode_usage_by_snr_bucket_mu", []), dtype=int
        ).tolist(),
        "mu_selection_distribution": np.asarray(
            info.get("mu_selection_distribution", []), dtype=int
        ).tolist(),
        "scheduled_count_by_mu": np.asarray(info.get("scheduled_count_by_mu", []), dtype=int).tolist(),
        "quality_by_mu": np.asarray(info.get("quality_by_mu", []), dtype=float).tolist(),
        "load_by_mu": np.asarray(info.get("load_by_mu", []), dtype=float).tolist(),
        "scheduling_alphas": scheduling_alphas.tolist(),
        "beta": beta.tolist(),
        "avg_lambda_usage": float(info["avg_lambda_usage"]),
        "avg_phi_budget": float(info["avg_phi_budget"]),
        "sum_backhaul_rate_bps": float(info["sum_backhaul_rate"]),
        "omega_Q": float(info.get("omega_Q", 0.0)),
        "omega_Lambda": float(info.get("omega_Lambda", 0.0)),
        "omega_A": float(info.get("omega_A", 0.0)),
        "Q_min": float(info.get("Q_min", 0.0)),
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
        "A_max": float(info.get("A_max", 0.0)),
    }
    if metadata:
        for key, value in metadata.items():
            row[f"eval_{key}"] = value
    for n in range(beta.size):
        row[f"uav{n}_beta"] = float(beta[n])
        row[f"uav{n}_lambda_usage"] = float(usage[n])
        row[f"uav{n}_selected_mode"] = int(modes[n])
        if n < mu_ids.size:
            row[f"uav{n}_selected_mu_id"] = int(mu_ids[n])
        if n < bucket_ids.size:
            row[f"uav{n}_snr_bucket_id"] = int(bucket_ids[n])
        if n < effective_snr.size:
            row[f"uav{n}_effective_snr_db"] = float(effective_snr[n])
        if scheduling_alphas.ndim == 2 and n < scheduling_alphas.shape[0]:
            row[f"uav{n}_alpha_A"] = float(scheduling_alphas[n, 0])
            row[f"uav{n}_alpha_Q"] = float(scheduling_alphas[n, 1])
            row[f"uav{n}_alpha_Lambda"] = float(scheduling_alphas[n, 2])
        row[f"uav{n}_scheduled_count"] = int(np.sum(y[n]))
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
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


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for baseline in sorted({str(row["baseline"]) for row in rows}):
        br = [row for row in rows if row["baseline"] == baseline]

        def segment_stats(segment_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            rewards = np.asarray([row["reward"] for row in segment_rows], dtype=float)
            scheduled = np.asarray([row["scheduled_count"] for row in segment_rows], dtype=float)
            quality = np.asarray([row["quality_term"] for row in segment_rows], dtype=float)
            load = np.asarray([row["load_term"] for row in segment_rows], dtype=float)
            aoi = np.asarray([row["aoi_state_term"] for row in segment_rows], dtype=float)
            aoi_mean = np.asarray([row["aoi_mean_term"] for row in segment_rows], dtype=float)
            aoi_max = np.asarray([row["aoi_max_term"] for row in segment_rows], dtype=float)
            aoi_tail = np.asarray([row["aoi_tail_term"] for row in segment_rows], dtype=float)
            fresh_gain = np.asarray([row["fresh_gain_term"] for row in segment_rows], dtype=float)
            reward_quality = np.asarray([row["reward_quality"] for row in segment_rows], dtype=float)
            reward_load_cost = np.asarray([row["reward_load_cost"] for row in segment_rows], dtype=float)
            reward_aoi_penalty = np.asarray([row["reward_aoi_penalty"] for row in segment_rows], dtype=float)
            penalty_q = np.asarray([row["penalty_quality_violation"] for row in segment_rows], dtype=float)
            penalty_a = np.asarray([row["penalty_aoi_violation"] for row in segment_rows], dtype=float)
            penalty_l = np.asarray([row["penalty_load_violation"] for row in segment_rows], dtype=float)
            bonus_a = np.asarray([row["bonus_aoi_reduction"] for row in segment_rows], dtype=float)
            max_aoi = np.asarray([row["max_AoI"] for row in segment_rows], dtype=float)
            lambda_usage = np.asarray([row["avg_lambda_usage"] for row in segment_rows], dtype=float)
            avg_quality_gain = np.asarray([row["avg_quality_gain"] for row in segment_rows], dtype=float)
            alpha_a = np.asarray(
                [row[key] for row in segment_rows for key in row if key.endswith("_alpha_A")],
                dtype=float,
            )
            alpha_q = np.asarray(
                [row[key] for row in segment_rows for key in row if key.endswith("_alpha_Q")],
                dtype=float,
            )
            alpha_l = np.asarray(
                [row[key] for row in segment_rows for key in row if key.endswith("_alpha_Lambda")],
                dtype=float,
            )
            return {
                "steps": int(len(segment_rows)),
                "reward_mean": float(np.mean(rewards)) if rewards.size else 0.0,
                "reward_sum": float(np.sum(rewards)) if rewards.size else 0.0,
                "scheduled_mean": float(np.mean(scheduled)) if scheduled.size else 0.0,
                "scheduled_total": int(np.sum(scheduled)) if scheduled.size else 0,
                "quality_term_mean": float(np.mean(quality)) if quality.size else 0.0,
                "load_term_mean": float(np.mean(load)) if load.size else 0.0,
                "aoi_state_term_mean": float(np.mean(aoi)) if aoi.size else 0.0,
                "aoi_mean_term_mean": float(np.mean(aoi_mean)) if aoi_mean.size else 0.0,
                "aoi_max_term_mean": float(np.mean(aoi_max)) if aoi_max.size else 0.0,
                "aoi_tail_term_mean": float(np.mean(aoi_tail)) if aoi_tail.size else 0.0,
                "fresh_gain_term_mean": float(np.mean(fresh_gain)) if fresh_gain.size else 0.0,
                "reward_quality_mean": float(np.mean(reward_quality)) if reward_quality.size else 0.0,
                "reward_load_cost_mean": float(np.mean(reward_load_cost)) if reward_load_cost.size else 0.0,
                "reward_aoi_penalty_mean": float(np.mean(reward_aoi_penalty)) if reward_aoi_penalty.size else 0.0,
                "penalty_quality_violation_mean": float(np.mean(penalty_q)) if penalty_q.size else 0.0,
                "penalty_aoi_violation_mean": float(np.mean(penalty_a)) if penalty_a.size else 0.0,
                "penalty_load_violation_mean": float(np.mean(penalty_l)) if penalty_l.size else 0.0,
                "bonus_aoi_reduction_mean": float(np.mean(bonus_a)) if bonus_a.size else 0.0,
                "total_load": float(np.sum(load)) if load.size else 0.0,
                "lambda_usage_mean": float(np.mean(lambda_usage)) if lambda_usage.size else 0.0,
                "quality_gain_mean": float(np.mean(quality)) if quality.size else 0.0,
                "avg_quality_gain": float(np.mean(avg_quality_gain)) if avg_quality_gain.size else 0.0,
                "avg_AoI": float(np.mean(aoi_mean)) if aoi_mean.size else 0.0,
                "max_aoi": float(np.max(max_aoi)) if max_aoi.size else 0.0,
                "alpha_A_mean": float(np.mean(alpha_a)) if alpha_a.size else 0.0,
                "alpha_Q_mean": float(np.mean(alpha_q)) if alpha_q.size else 0.0,
                "alpha_Lambda_mean": float(np.mean(alpha_l)) if alpha_l.size else 0.0,
            }

        out[baseline] = {"overall": segment_stats(br)}
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, default="uav_escs_sc", choices=["uav_escs_sc", "uav_escs_cc"])
    parser.add_argument("--algo", type=str, default="happo", help="Only used to load default config files.")
    parser.add_argument("--exp_name", type=str, default="algorithm_baselines")
    parser.add_argument("--load_config", type=str, default="")
    parser.add_argument(
        "--baselines",
        type=str,
        default="default",
        help="Comma-separated names, 'default' for random+round_robin+utility_greedy, or 'all'.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode_length", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenario_name", type=str, default="", help="Optional scenario label written to rollout rows.")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="consistent_mask",
        help="Suffix for corrected-result files. Use an empty string only in a fresh output directory.",
    )
    parser.add_argument("--dump_episode_trace", type=_as_bool, default=False)
    parser.add_argument("--trace_episode_id", type=int, default=0)
    parser.add_argument("--trace_max_episodes", type=int, default=1)
    parser.add_argument("--trace_output_dir", type=str, default="")
    parser.add_argument("--trace_format", type=str, default="jsonl", choices=["jsonl"])
    args, unknown = parser.parse_known_args()

    overrides = _parse_unknown_args(unknown)
    main_args, algo_args, env_args = _load_config(args)
    update_args(overrides, algo_args, env_args, override=True)
    main_args["env"] = args.env if not args.load_config else main_args.get("env", args.env)

    if args.episode_length > 0:
        env_args["T"] = float(args.episode_length) * float(env_args.get("delta_T", 1.0))

    env_args.setdefault("env_name", "semantic_video_acquisition_fixed_multiobj")
    env_args.setdefault("Q_min", 30.0)
    env_args.setdefault("Q_max", 45.0)
    env_args.setdefault("A_max", 600.0)
    env_args.setdefault("omega_Q", 0.50)
    env_args.setdefault("omega_Lambda", 0.10)
    env_args.setdefault("omega_A", 0.40)
    env_args.setdefault("aoi_mean_weight", 0.4)
    env_args.setdefault("aoi_max_weight", 0.3)
    env_args.setdefault("aoi_tail_weight", 0.3)
    env_args.setdefault("aoi_tail_threshold", 10.0)
    env_args.setdefault("use_direct_ds_logits", False)
    env_args.setdefault("scheduling_weight_transform", "softmax")
    env_args.setdefault("initial_cache_prob", 1.0)

    baseline_arg = args.baselines.strip().lower()
    if baseline_arg == "all":
        baseline_names = list(BASELINES.keys())
    elif baseline_arg in ("", "default"):
        baseline_names = list(DEFAULT_BASELINES)
    else:
        baseline_names = [name.strip() for name in args.baselines.split(",") if name.strip()]
    unknown_baselines = [name for name in baseline_names if name not in BASELINES]
    if unknown_baselines:
        raise ValueError(f"Unknown baselines: {unknown_baselines}. Available: {sorted(BASELINES)}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = Path(__file__).resolve().parent / "baseline_results" / main_args["env"]
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_output_dir = Path(args.trace_output_dir).expanduser().resolve() if args.trace_output_dir else output_dir
    if args.dump_episode_trace:
        trace_output_dir.mkdir(parents=True, exist_ok=True)
        env_args["enable_episode_trace"] = True

    rows: List[Dict[str, Any]] = []
    run_metadata = {
        "seed": int(args.seed),
        "config_path": str(Path(args.load_config).expanduser().resolve()) if args.load_config else "defaults",
        "checkpoint_path": "",
        "feasibility_rule": "Q_hat_rec >= Q_req(instruction, snr_bucket)",
        "output_suffix": str(args.output_suffix),
    }
    for baseline_idx, baseline_name in enumerate(baseline_names):
        policy = BASELINES[baseline_name]
        env = _make_env(main_args["env"], dict(env_args))
        for episode in range(args.episodes):
            # Keep the episode-level environment seed shared across baselines so
            # initial exogenous states are aligned. Action randomness, when a
            # baseline needs it, remains policy-specific.
            env.seed(int(args.seed) + episode)
            rng = np.random.default_rng(int(args.seed) + 1009 * baseline_idx + episode)
            _, _, available_actions = env.reset()
            done = False
            should_trace = (
                bool(args.dump_episode_trace)
                and episode >= int(args.trace_episode_id)
                and episode < int(args.trace_episode_id) + max(0, int(args.trace_max_episodes))
            )
            trace_file = None
            if should_trace:
                trace_path = trace_output_dir / f"trace_seed{int(args.seed)}_{baseline_name}_episode{episode}.jsonl"
                trace_file = trace_path.open("w", encoding="utf-8")
            try:
                while not done:
                    actions = policy(env, available_actions, rng)
                    _, _, _, dones, infos, available_actions = env.step(actions)
                    rows.append(_row_from_info(baseline_name, episode, infos[0], args.scenario_name, run_metadata))
                    if trace_file is not None:
                        trace = dict(infos[0].get("trace", {}))
                        trace["episode"] = int(episode)
                        trace["policy"] = baseline_name
                        trace["seed"] = int(args.seed)
                        trace_file.write(json.dumps(_json_ready(trace), sort_keys=True) + "\n")
                    done = bool(np.all(dones))
            finally:
                if trace_file is not None:
                    trace_file.close()
        env.close()

    csv_path = _suffix_path(output_dir / "algorithm_baselines_rollout.csv", args.output_suffix)
    rollout_json_path = _suffix_path(output_dir / "algorithm_baselines_rollout.json", args.output_suffix)
    summary_path = _suffix_path(output_dir / "algorithm_baselines_summary.json", args.output_suffix)
    _write_csv(csv_path, rows)
    summary = _summary(rows)
    with rollout_json_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready({"metadata": run_metadata, "rows": rows}), file, indent=2, sort_keys=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready({"metadata": run_metadata, "summary": summary}), file, indent=2, sort_keys=True)

    print(f"csv: {csv_path}")
    print(f"rollout_json: {rollout_json_path}")
    print(f"summary: {summary_path}")
    for name in baseline_names:
        item = summary[name]["overall"]
        print(
            f"{name}: reward_mean={item['reward_mean']:.4f}, "
            f"scheduled_mean={item['scheduled_mean']:.2f}, "
            f"quality_term_mean={item['quality_term_mean']:.3f}, "
            f"total_load={item['total_load']:.3f}, "
            f"aoi_state_term_mean={item['aoi_state_term_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
