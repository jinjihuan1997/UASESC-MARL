"""Plot SC/CC training reward trend comparison from existing training logs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_semantic_video_paper_figures import EXPORT_DPI, FIG_SIZE, finalize_line_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "examples" / "results"
OUTPUT_DIR = RESULTS_ROOT / "plot_results" / "cc_sc_training_reward_comparison"
OUTPUT_BASE = OUTPUT_DIR / "fig_cc_sc_training_reward_trend"

RUN_GLOBS = {
    ("SC", "IC-HAPPO"): "uav_escs_sc/semantic_video_acquisition_fixed_multiobj/happo/*/seed-*/train_semantic_metrics.csv",
    ("SC", "IC-MAPPO"): "uav_escs_sc/semantic_video_acquisition_fixed_multiobj/mappo/*/seed-*/train_semantic_metrics.csv",
    (
        "CC, H.264+LDPC",
        "IC-HAPPO",
    ): "uav_escs_cc/semantic_video_acquisition_fixed_multiobj/happo/*/seed-*/train_semantic_metrics.csv",
    (
        "CC, H.264+LDPC",
        "IC-MAPPO",
    ): "uav_escs_cc/semantic_video_acquisition_fixed_multiobj/mappo/*/seed-*/train_semantic_metrics.csv",
}

COLORS = {
    "IC-HAPPO": "#0072BD",
    "IC-MAPPO": "#D95319",
}
MARKERS = {
    "IC-HAPPO": "o",
    "IC-MAPPO": "s",
}
MAX_TRAINING_STEPS = 10_000_000
MARK_INTERVAL_STEPS = 1_000_000
REWARD_COLUMN = "average_step_rewards"
ENV_PANEL_TITLES = {
    "SC": "SC",
    "CC, H.264+LDPC": "CC, H.264+LDPC",
}


def latest_file(paths: Iterable[Path]) -> Path | None:
    paths = list(paths)
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def load_training_logs() -> pd.DataFrame:
    frames = []
    for (env_label, algo_label), pattern in RUN_GLOBS.items():
        path = latest_file(RESULTS_ROOT.glob(pattern))
        if path is None:
            print(f"WARNING: missing training metrics for {env_label} {algo_label}: {pattern}")
            continue

        df = pd.read_csv(path)
        required = {"total_num_steps", REWARD_COLUMN}
        missing = required - set(df.columns)
        if missing:
            print(f"WARNING: skip {path}, missing columns: {sorted(missing)}")
            continue

        df = df[["total_num_steps", REWARD_COLUMN]].copy()
        df["env_label"] = env_label
        df["algo_label"] = algo_label
        df["source_file"] = str(path)
        frames.append(df)

    if not frames:
        raise RuntimeError("No usable SC/CC training reward logs were found.")

    data = pd.concat(frames, ignore_index=True)
    data["total_num_steps"] = pd.to_numeric(data["total_num_steps"], errors="coerce")
    data[REWARD_COLUMN] = pd.to_numeric(data[REWARD_COLUMN], errors="coerce")
    data = data.dropna(subset=["total_num_steps", REWARD_COLUMN])
    data = data[data["total_num_steps"] <= MAX_TRAINING_STEPS].copy()
    return data.sort_values(["env_label", "algo_label", "total_num_steps"])


def interval_step_indices(steps: Sequence[float]) -> list[int]:
    """Return data indices closest to each fixed training-step marker location."""
    steps_arr = np.asarray(steps, dtype=float)
    if steps_arr.size == 0:
        return []

    targets = np.arange(0, MAX_TRAINING_STEPS + 1, MARK_INTERVAL_STEPS, dtype=float)
    indices = []
    for target in targets:
        idx = int(np.abs(steps_arr - target).argmin())
        if idx not in indices:
            indices.append(idx)
    return indices


def smoothed_reward(group: pd.DataFrame) -> pd.Series:
    return group[REWARD_COLUMN].rolling(window=7, min_periods=1, center=True).mean()


def plot(data: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_BASE.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for env_label, algo_label in (
        ("SC", "IC-MAPPO"),
        ("SC", "IC-HAPPO"),
        ("CC, H.264+LDPC", "IC-MAPPO"),
        ("CC, H.264+LDPC", "IC-HAPPO"),
    ):
        group = data[(data["env_label"] == env_label) & (data["algo_label"] == algo_label)].copy()
        if group.empty:
            continue

        group = group.sort_values("total_num_steps")
        smooth = smoothed_reward(group)
        steps = group["total_num_steps"].to_numpy(dtype=float)
        point_indices = interval_step_indices(steps)
        x_points = steps[point_indices] / 1.0e6
        y_points = smooth.to_numpy(dtype=float)[point_indices]
        ax.plot(
            x_points,
            y_points,
            label=f"{algo_label} ({ENV_PANEL_TITLES[env_label]})",
            color=COLORS[algo_label],
            linestyle="-",
            linewidth=2.2 if env_label == "SC" else 1.75,
            marker=MARKERS[algo_label],
            markersize=8.8,
            markerfacecolor="none" if env_label == "SC" else COLORS[algo_label],
            markeredgecolor=COLORS[algo_label],
            markeredgewidth=1.65,
            zorder=4,
        )

    ax.set_xlim(0.0, MAX_TRAINING_STEPS / 1.0e6)
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1.0e-6)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.22 * span)
    finalize_line_axis(
        ax,
        "Training steps (million)",
        "Average reward per slot",
        legend=True,
        legend_loc="upper center",
        legend_markerscale=0.85,
    )
    ax.set_xlim(0.0, MAX_TRAINING_STEPS / 1.0e6)

    fig.tight_layout()
    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)

def main() -> int:
    data = load_training_logs()
    plot(data)
    print(f"Generated: {OUTPUT_BASE.with_suffix('.pdf')}")
    print(f"Generated: {OUTPUT_BASE.with_suffix('.png')}")
    print(f"Generated: {OUTPUT_BASE.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
