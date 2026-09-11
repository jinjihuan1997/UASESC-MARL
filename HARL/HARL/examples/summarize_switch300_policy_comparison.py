#!/usr/bin/env python3
"""Focused switch-300 policy comparison for reward, quality, and AoI."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ("switch_300_balance_aoi", "switch_300_balance_quality")
METHOD_ORDER = ["random", "round_robin", "utility_greedy", "IC-MAPPO", "IC-HAPPO"]
METRICS = {
    "reward": ("reward", "Reward"),
    "quality": ("quality", "Reconstruction quality"),
    "aoi": ("aoi", "Mean AoI"),
}
COLORS = {
    "random": "#4D4D4D",
    "round_robin": "#7E2F8E",
    "utility_greedy": "#EDB120",
    "IC-MAPPO": "#D95319",
    "IC-HAPPO": "#0072BD",
}
MARKERS = {
    "random": "x",
    "round_robin": "D",
    "utility_greedy": "^",
    "IC-MAPPO": "s",
    "IC-HAPPO": "o",
}


def parse_seed_list(text: str) -> set[str]:
    if not text:
        return set()
    return {str(int(float(item.strip()))) for item in text.split(",") if item.strip()}


def seed_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.search(r"(?:train|unseen)?_?seed[_-]?(\d+)", part, flags=re.I)
        if match:
            return str(int(match.group(1)))
    return "unknown"


def seed_group(seed: str, path: Path, train_seeds: set[str], unseen_seeds: set[str]) -> str:
    if seed in train_seeds:
        return "train"
    if seed in unseen_seeds:
        return "unseen"
    text = " ".join(str(p).lower() for p in path.parts)
    if "unseen" in text:
        return "unseen"
    if "train" in text:
        return "train"
    return "unknown"


def canon_method(value: Any, path: Path) -> str:
    text = f"{value} {' '.join(path.parts)}".lower()
    if "round_robin" in text or "roundrobin" in text:
        return "round_robin"
    if "utility_greedy" in text or "greedy" in text:
        return "utility_greedy"
    if re.search(r"(^|[/_\-\s])random($|[/_\-\s])", text):
        return "random"
    if "mappo" in text:
        return "IC-MAPPO"
    if "happo" in text:
        return "IC-HAPPO"
    return str(value)


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        col = lower.get(name.lower())
        if col is not None:
            return col
    return None


def normalize_file(path: Path, train_seeds: set[str], unseen_seeds: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    scenario_col = first_existing(df, ("scenario",))
    slot_col = first_existing(df, ("slot", "t", "step", "timestep"))
    reward_col = first_existing(df, ("reward", "slot_reward", "mean_reward"))
    quality_col = first_existing(df, ("avg_quality_gain", "quality_gain", "avg_quality", "mean_quality", "raw_q_hat_mean"))
    aoi_col = first_existing(df, ("mean_aoi", "mean_AoI", "avg_aoi", "avg_AoI", "aoi"))
    if slot_col is None or reward_col is None or quality_col is None or aoi_col is None:
        return pd.DataFrame()

    seed = seed_from_path(path)
    if "algorithm_baselines_rollout" in path.name:
        method_col = first_existing(df, ("baseline", "method", "scheme"))
        method_values = df[method_col] if method_col is not None else "baseline"
    else:
        method_values = pd.Series(canon_method("", path), index=df.index)

    out = pd.DataFrame(
        {
            "source_file": str(path),
            "seed": seed,
            "seed_group": seed_group(seed, path, train_seeds, unseen_seeds),
            "method": [canon_method(v, path) for v in method_values],
            "scenario": df[scenario_col].astype(str) if scenario_col is not None else "",
            "slot": pd.to_numeric(df[slot_col], errors="coerce"),
            "reward": pd.to_numeric(df[reward_col], errors="coerce"),
            "quality": pd.to_numeric(df[quality_col], errors="coerce"),
            "aoi": pd.to_numeric(df[aoi_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["slot", "reward", "quality", "aoi"])


def load_data(results_root: Path, train_seeds: set[str], unseen_seeds: set[str]) -> pd.DataFrame:
    files = sorted(results_root.rglob("algorithm_baselines_rollout*.csv")) + sorted(
        results_root.rglob("instruction_constraint_rollout*.csv")
    )
    frames = [normalize_file(path, train_seeds, unseen_seeds) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    return data[data["scenario"].isin(SCENARIOS) & data["method"].isin(METHOD_ORDER)].copy()


def aggregate_lines(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    per_seed = (
        data.groupby(["scenario", "seed_group", "method", "seed", "slot"], dropna=False)[metric]
        .mean()
        .reset_index(name="seed_mean")
    )
    per_slot = (
        per_seed.groupby(["scenario", "seed_group", "method", "slot"], dropna=False)["seed_mean"]
        .agg(mean="mean", std="std", n_seed="count")
        .reset_index()
        .fillna({"std": 0.0})
    )
    per_slot["interval_start"] = np.floor(per_slot["slot"] / 50.0) * 50.0
    lines = (
        per_slot.groupby(["scenario", "seed_group", "method", "interval_start"], dropna=False)
        .agg(mean=("mean", "mean"), std=("std", "mean"), n_seed=("n_seed", "max"), n_points=("mean", "count"))
        .reset_index()
        .sort_values(["scenario", "seed_group", "method", "interval_start"])
    )
    lines["interval_end"] = lines["interval_start"] + 50.0
    lines["slot"] = lines["interval_start"] + 25.0
    return lines[["scenario", "seed_group", "method", "slot", "mean", "std", "n_seed", "interval_start", "interval_end", "n_points"]]


def aggregate_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment, selector in (
        ("full_episode", data["slot"].ge(0)),
        ("pre_switch_100", data["slot"].between(200, 299)),
        ("post_switch_100", data["slot"].between(300, 399)),
        ("post_switch_all", data["slot"].ge(300)),
    ):
        part = data[selector].copy()
        if part.empty:
            continue
        per_seed = (
            part.groupby(["scenario", "seed_group", "method", "seed"], dropna=False)[["reward", "quality", "aoi"]]
            .mean()
            .reset_index()
        )
        summary = (
            per_seed.groupby(["scenario", "seed_group", "method"], dropna=False)[["reward", "quality", "aoi"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.columns = [
            "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
            for col in summary.columns
        ]
        summary.insert(2, "segment", segment)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def plot_metric(lines: pd.DataFrame, metric: str, ylabel: str, out_base: Path) -> None:
    for (scenario, group), sub in lines.groupby(["scenario", "seed_group"], dropna=False):
        fig, ax = plt.subplots(figsize=(5.2, 3.35))
        for method in METHOD_ORDER:
            g = sub[sub["method"] == method].sort_values("slot")
            if g.empty:
                continue
            ax.plot(
                g["slot"],
                g["mean"],
                label=method,
                color=COLORS.get(method),
                linewidth=1.65,
                linestyle="-",
                marker=MARKERS.get(method, "o"),
                markersize=7.2,
                markerfacecolor="none" if method != "random" else COLORS.get(method),
                markeredgewidth=1.35,
            )
        ax.axvline(300, color="0.20", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Slot")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{scenario.replace('_', ' ')} ({group})")
        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.20)
        ax.margins(x=0.02, y=0.18)
        leg = ax.legend(
            fontsize=7.0,
            loc="upper left" if metric in ("aoi", "quality") else "upper right",
            ncol=2,
            frameon=True,
            fancybox=False,
            framealpha=0.50,
            edgecolor="0.45",
            facecolor="0.90",
        )
        leg.get_frame().set_linewidth(0.6)
        target = out_base / f"fig_{scenario}_{group}_{metric}"
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(target.with_suffix(".png"), dpi=600, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results_root", type=Path, default=Path("examples/results/paper_trajectory_eval"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("examples/results/plot_results/semantic_video_paper_figures/00_switch300_policy_comparison"),
    )
    parser.add_argument("--train_seeds", type=str, default="1,2,3")
    parser.add_argument("--unseen_seeds", type=str, default="101,102,103")
    args = parser.parse_args()

    train_seeds = parse_seed_list(args.train_seeds)
    unseen_seeds = parse_seed_list(args.unseen_seeds)
    data = load_data(args.results_root.resolve(), train_seeds, unseen_seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if data.empty:
        print("No switch_300 trajectory rows found.")
        return 1

    data.to_csv(args.output_dir / "switch300_policy_comparison_rows.csv", index=False)
    summary = aggregate_summary(data)
    summary.to_csv(args.output_dir / "switch300_policy_comparison_summary.csv", index=False)

    for metric, (_field, ylabel) in METRICS.items():
        lines = aggregate_lines(data, metric)
        lines.to_csv(args.output_dir / f"switch300_{metric}_timeseries.csv", index=False)
        plot_metric(lines, metric, ylabel, args.output_dir)

    print(f"rows: {args.output_dir / 'switch300_policy_comparison_rows.csv'}")
    print(f"summary: {args.output_dir / 'switch300_policy_comparison_summary.csv'}")
    print(f"figures: {args.output_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
