#!/usr/bin/env python3
"""Trajectory-based paper figures for UAV-assisted semantic video backhaul.

This script only targets two paper subsections:

1. Effectiveness of Cooperative Learning Policies
2. Instruction Response Analysis

It is deliberately read-only for raw experiment outputs. Figures and aggregate
CSVs are written under a dedicated plot_results directory.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paper subsection names and output directories
# ============================================================

SUBSECTION_POLICY = "Effectiveness of Cooperative Learning Policies"
SUBSECTION_INSTRUCTION = "Instruction Response Analysis"

POLICY_DIR = "01_effectiveness_cooperative_learning_policies"
INSTRUCTION_DIR = "02_instruction_response_analysis"

SCHEME_ORDER = ["random", "round_robin", "utility_greedy", "IC-MAPPO", "IC-HAPPO"]
IC_SCHEMES = {"IC-MAPPO", "IC-HAPPO"}


# ============================================================
# IEEE / MATLAB-like plotting style
# ============================================================

FIG_SIZE = (6.4, 4.2)
EXPORT_DPI = 600
PLOT_INTERVAL = 50.0
BASE_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 13
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 10.8
BAR_LEGEND_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 13


def setup_ieee_matlab_style() -> None:
    """Use a clean MATLAB-like style for IEEE paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": BASE_FONT_SIZE,
            "font.weight": "bold",
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": AXIS_LABEL_FONT_SIZE,
            "axes.titleweight": "bold",
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.2,
            "ytick.major.size": 4.2,
            "lines.linewidth": 2.0,
            "lines.markersize": 8.5,
            "figure.dpi": 300,
            "savefig.dpi": EXPORT_DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


setup_ieee_matlab_style()


COLORS = {
    "IC-HAPPO": "#0072BD",
    "IC-MAPPO": "#D95319",
    "utility_greedy": "#EDB120",
    "round_robin": "#7E2F8E",
    "random": "#4D4D4D",
}

LINESTYLES = {
    "IC-HAPPO": "-",
    "IC-MAPPO": "-",
    "utility_greedy": "-",
    "round_robin": "-",
    "random": "-",
}

MARKERS = {
    "IC-HAPPO": "o",
    "IC-MAPPO": "s",
    "utility_greedy": "^",
    "round_robin": "D",
    "random": "x",
}

MARKER_FACES = {
    "IC-HAPPO": "none",
    "IC-MAPPO": "none",
    "utility_greedy": "none",
    "round_robin": "none",
    "random": "none",
}

LINE_WIDTHS = {
    "IC-HAPPO": 2.2,
    "IC-MAPPO": 2.2,
    "utility_greedy": 2.0,
    "round_robin": 2.0,
    "random": 2.0,
}

DISPLAY_LABELS = {
    "random": "Random",
    "round_robin": "Round Robin",
    "utility_greedy": "Utility Greedy",
}


ALIASES: Dict[str, Sequence[str]] = {
    "slot": ("slot", "t", "timestep", "step"),
    "reward": ("reward", "slot_reward", "mean_reward", "avg_reward", "episode_reward"),
    "aoi": ("avg_aoi", "mean_aoi", "aoi", "aoi_state", "A_state"),
    "max_aoi": ("max_aoi",),
    "quality": (
        "avg_quality_gain",
        "quality_gain",
        "avg_quality",
        "mean_quality",
        "quality",
        "reconstruction_quality",
        "q_rec",
        "raw_q_hat_mean",
        "raw_Q_mean",
    ),
    "load": (
        "semantic_load",
        "avg_semantic_load",
        "backhaul_load",
        "lambda_sem",
        "Lambda_sem",
        "lambda_usage",
        "avg_lambda_usage",
        "load",
    ),
    "scheduled": ("scheduled_count", "num_scheduled", "scheduled_streams", "service_count"),
    "instruction": ("instruction", "instruction_id", "instruction_name", "instruction_mode", "g", "g_t"),
    "scheme": ("scheme", "algorithm", "algo", "exp_name", "policy_name", "baseline", "method", "policy"),
    "scenario": ("scenario", "eval_scenario", "test_scenario"),
    "seed": ("seed", "train_seed", "eval_seed", "test_seed"),
    "episode": ("episode", "episode_id", "eval_episode", "rollout_id", "run_id"),
    "instruction_mode_strategy": ("instruction_mode_strategy",),
    "transition": ("instruction_transition",),
    "before": ("instruction_before", "instruction_before_mode", "instruction_before_id"),
    "after": ("instruction_after", "instruction_after_mode", "instruction_after_id"),
    "switch_step": ("instruction_switch_step", "switch_step"),
    "mu_ids": ("selected_mu_ids", "mu_ids", "mu_id", "mu"),
    "mode_ids": ("scheduled_modes", "selected_modes", "selected_mode", "mode_id", "semantic_mode"),
    "alphas": ("scheduling_alphas", "scheduling_alpha", "alphas"),
}

CONSISTENCY_KEYS = (
    "env",
    "n_uav",
    "n_ds",
    "T",
    "episode_length",
    "B_sut_sat",
    "Q_min",
    "Q_min_by_instruction",
    "instruction_mode_strategy",
    "instruction_switch_step",
    "instruction_before_mode",
    "instruction_before_id",
    "instruction_after_mode",
    "instruction_after_id",
    "instruction_after_mode_strategy",
)

INSTRUCTION_NAMES = {0: "balance", 1: "AoI", 2: "quality"}


@dataclass
class SectionLog:
    subsection: str
    input_files: List[str] = field(default_factory=list)
    schemes: List[str] = field(default_factory=list)
    seeds: List[str] = field(default_factory=list)
    train_seeds: List[str] = field(default_factory=list)
    unseen_seeds: List[str] = field(default_factory=list)
    unknown_seeds: List[str] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    generated: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        print(f"WARNING [{self.subsection}]: {msg}")
        self.warnings.append(msg)

    def skip(self, fig: str, reason: str) -> None:
        print(f"SKIP [{self.subsection}] {fig}: {reason}")
        self.skipped.append(f"{fig}: {reason}")

    def ok(self, path: Path) -> None:
        print(f"GENERATED [{self.subsection}] {path}")
        self.generated.append(str(path))


@dataclass
class RunLog:
    data_files: List[str] = field(default_factory=list)
    schemes: List[str] = field(default_factory=list)
    seeds: List[str] = field(default_factory=list)
    train_seeds: List[str] = field(default_factory=list)
    unseen_seeds: List[str] = field(default_factory=list)
    unknown_seeds: List[str] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    generated: List[str] = field(default_factory=list)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_seed_list(text: str | None) -> set[str]:
    if not text:
        return set()
    out = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.add(str(int(item)))
        except ValueError:
            out.add(item)
    return out


def first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        col = lower.get(str(name).lower())
        if col is not None:
            return col
    return None


def scalar(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    if nums:
        return float(np.mean([float(x) for x in nums]))
    return text


def numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(series.map(scalar), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def read_config_near(path: Path) -> Dict[str, Any]:
    for parent in [path.parent, *path.parents]:
        cfg = parent / "config.json"
        if cfg.exists():
            try:
                with cfg.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        if parent.name == "examples":
            break
    return {}


def nested_get(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    env = cfg.get("env_args", {}) if isinstance(cfg.get("env_args", {}), dict) else {}
    algo = cfg.get("algo_args", {}) if isinstance(cfg.get("algo_args", {}), dict) else {}
    main = cfg.get("main_args", {}) if isinstance(cfg.get("main_args", {}), dict) else {}

    flat = {}
    for key in CONSISTENCY_KEYS:
        if key in env:
            flat[key] = env[key]

    if "episode_length" not in flat:
        flat["episode_length"] = nested_get(algo, ("train", "episode_length"))
    if "env" not in flat:
        flat["env"] = main.get("env")
    if "algo" not in flat:
        flat["algo"] = main.get("algo")
    if "exp_name" not in flat:
        flat["exp_name"] = main.get("exp_name")

    return flat


def infer_scheme(path: Path, cfg: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
    parts = [str(x) for x in path.parts]
    main = cfg.get("main_args", {}) if isinstance(cfg.get("main_args", {}), dict) else {}
    parts.extend([str(main.get("algo", "")), str(main.get("exp_name", ""))])

    if df is not None:
        col = first_col(df, ALIASES["scheme"])
        if col is not None and df[col].notna().any():
            parts.append(str(df[col].dropna().iloc[0]))

    text = " ".join(parts).lower()

    if "round_robin" in text or "roundrobin" in text:
        return "round_robin"
    if "utility_greedy" in text or "greedy" in text:
        return "utility_greedy"
    if re.search(r"(^|[/_\-\s])random($|[/_\-\s])", text) and "random_switch" not in text and "random_fixed" not in text:
        return "random"
    if "ic_mappo" in text or "ic-mappo" in text or "icmappo" in text:
        return "IC-MAPPO"
    if "ic_happo" in text or "ic-happo" in text or "ichappo" in text:
        return "IC-HAPPO"
    if "mappo" in text:
        return "IC-MAPPO"
    if "happo" in text:
        return "IC-HAPPO"

    return path.parent.name


def canonical_scheme(value: Any) -> str:
    text = str(value).strip().lower()

    if "round_robin" in text or "roundrobin" in text:
        return "round_robin"
    if "utility_greedy" in text or "greedy" in text:
        return "utility_greedy"
    if text == "random" or re.search(r"(^|[/_\-\s])random($|[/_\-\s])", text):
        return "random"
    if "ic_mappo" in text or "ic-mappo" in text or "icmappo" in text:
        return "IC-MAPPO"
    if "ic_happo" in text or "ic-happo" in text or "ichappo" in text:
        return "IC-HAPPO"
    if "mappo" in text:
        return "IC-MAPPO"
    if "happo" in text:
        return "IC-HAPPO"

    return str(value)


def infer_seed(path: Path, df: Optional[pd.DataFrame] = None) -> str:
    if df is not None:
        col = first_col(df, ALIASES["seed"])
        if col is not None and df[col].notna().any():
            return clean_seed(df[col].dropna().iloc[0])

    for part in path.parts:
        m = re.search(r"seed[-_]?(\d+)", part, flags=re.I)
        if m:
            return clean_seed(m.group(1))

    return "unknown"


def clean_seed(value: Any) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def infer_seed_group(path: Path, df: pd.DataFrame, seed: str, train_seeds: set[str], unseen_seeds: set[str]) -> str:
    if seed in train_seeds:
        return "train"
    if seed in unseen_seeds:
        return "unseen"

    text = " ".join(str(x).lower() for x in path.parts)

    for col in ("seed_group", "scenario", "eval_type"):
        if col in df.columns and df[col].notna().any():
            text += " " + str(df[col].dropna().iloc[0]).lower()

    if "unseen" in text or "non_train" in text or "non-training" in text or "nontraining" in text:
        return "unseen"
    if "train_seed" in text or "training_seed" in text:
        return "train"

    return "unknown"


def read_json_like(path: Path) -> Optional[pd.DataFrame]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return pd.json_normalize(obj)

    if isinstance(obj, dict):
        for key in ("trajectory", "trajectories", "rows", "records", "data", "steps", "slots", "episodes", "summary"):
            val = obj.get(key)
            if isinstance(val, list):
                return pd.json_normalize(val)

        scalars = {k: v for k, v in obj.items() if not isinstance(v, (dict, list))}
        if scalars:
            return pd.DataFrame([scalars])

    return None


def read_file(path: Path, section: SectionLog) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)

        if path.suffix.lower() == ".json" and path.name != "config.json":
            return read_json_like(path)

        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=True)
            cols = {}
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.ndim <= 1:
                    cols[key] = arr.reshape(-1)
            return pd.DataFrame(cols) if cols else None

        if path.suffix.lower() == ".pkl":
            with path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                return obj
            if isinstance(obj, list):
                return pd.json_normalize(obj)
            if isinstance(obj, dict):
                return pd.json_normalize(obj)

    except Exception as exc:
        section.warn(f"failed to read {path}: {exc}")

    return None


def file_kind(path: Path, df: pd.DataFrame) -> str:
    name = path.name.lower()

    if first_col(df, ALIASES["slot"]) is not None:
        return "trajectory"

    if any(key in name for key in ("trajectory", "traj", "trace", "rollout")):
        return "trajectory_candidate"

    if name == "train_semantic_metrics.csv" or "train" in name:
        return "train_metrics"

    if "summary" in name:
        return "summary"

    return "table"


def normalize(path: Path, df: pd.DataFrame, train_seeds: set[str], unseen_seeds: set[str], section: SectionLog) -> pd.DataFrame:
    cfg = read_config_near(path)
    out = pd.DataFrame(index=df.index)

    for key, aliases in ALIASES.items():
        col = first_col(df, aliases)
        if col is None:
            continue

        if key in {
            "scheme",
            "scenario",
            "instruction",
            "instruction_mode_strategy",
            "transition",
            "before",
            "after",
            "mu_ids",
            "mode_ids",
            "alphas",
        }:
            out[key] = df[col].astype(str)
        else:
            out[key] = numeric(df[col])

    seed = infer_seed(path, df)
    scheme = infer_scheme(path, cfg, df)

    out["source_file"] = str(path)
    out["source_dir"] = str(path.parent)

    if "scheme" in out:
        out["scheme"] = out["scheme"].map(canonical_scheme)
        out["scheme"] = out["scheme"].replace({"nan": np.nan}).fillna(canonical_scheme(scheme))
    else:
        out["scheme"] = canonical_scheme(scheme)

    out["seed"] = seed
    out["seed_group"] = infer_seed_group(path, df, seed, train_seeds, unseen_seeds)
    out["file_kind"] = file_kind(path, df)
    out["episode"] = out["episode"].fillna(str(path.parent)) if "episode" in out else str(path.parent)

    flat = flatten_config(cfg)
    for key, value in flat.items():
        if value is not None:
            out[f"meta_{key}"] = str(value)

    if "switch_step" not in out:
        sw = flat.get("instruction_switch_step", np.nan)
        out["switch_step"] = scalar(sw)

    return out.reset_index(drop=True)


def discover(results_root: Path, output_dir: Path, train_seeds: set[str], unseen_seeds: set[str], section: SectionLog) -> pd.DataFrame:
    exts = {".csv", ".json", ".npz", ".pkl"}

    files = [
        p
        for p in results_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and p.name != "config.json"
        and "plot_results" not in p.parts
        and output_dir.name not in p.parts
    ]

    frames = []

    for path in sorted(files):
        df = read_file(path, section)
        if df is None or df.empty:
            continue

        norm = normalize(path, df, train_seeds, unseen_seeds, section)
        metric_cols = {"reward", "aoi", "max_aoi", "quality", "load", "scheduled"} & set(norm.columns)

        if not metric_cols:
            continue

        frames.append(norm)
        section.input_files.append(str(path))

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)

    section.schemes = sorted(data["scheme"].dropna().unique().tolist())
    section.seeds = sorted(data["seed"].dropna().unique().tolist(), key=lambda x: (x == "unknown", x))
    section.train_seeds = sorted(data.loc[data["seed_group"] == "train", "seed"].dropna().unique().tolist())
    section.unseen_seeds = sorted(data.loc[data["seed_group"] == "unseen", "seed"].dropna().unique().tolist())
    section.unknown_seeds = sorted(data.loc[data["seed_group"] == "unknown", "seed"].dropna().unique().tolist())

    return data


def usable_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "slot" not in df.columns:
        return pd.DataFrame()

    out = df[(df["file_kind"] == "trajectory") & df["slot"].notna()].copy()

    if out.empty:
        return out

    out["slot"] = numeric(out["slot"])
    return out.dropna(subset=["slot"])


def ensure_same_conditions(df: pd.DataFrame, section: SectionLog) -> None:
    if df.empty:
        return

    for key in CONSISTENCY_KEYS:
        col = f"meta_{key}"
        if col not in df.columns:
            continue

        vals_by_scheme = (
            df[["scheme", col]]
            .dropna()
            .drop_duplicates()
            .groupby("scheme")[col]
            .apply(lambda s: sorted(set(map(str, s))))
            .to_dict()
        )

        flat_vals = {tuple(v) for v in vals_by_scheme.values() if v}

        if len(flat_vals) > 1:
            section.warn(f"policy comparison metadata mismatch for {key}: {vals_by_scheme}")


def common_slot_range(df: pd.DataFrame, section: SectionLog) -> pd.DataFrame:
    if df.empty:
        return df

    ranges = df.groupby("scheme")["slot"].agg(["min", "max"]).dropna()

    if ranges.empty:
        return df

    lo = float(ranges["min"].max())
    hi = float(ranges["max"].min())

    if lo > float(ranges["min"].min()) or hi < float(ranges["max"].max()):
        section.warn(f"different slot ranges detected; plotting common interval [{lo:g}, {hi:g}]")

    return df[(df["slot"] >= lo) & (df["slot"] <= hi)].copy()


def smooth(y: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return y
    return y.rolling(window=window, center=True, min_periods=1).mean()


def interval_average_points(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    interval: float = PLOT_INTERVAL,
) -> pd.DataFrame:
    """Return one plotted point per interval; y is the interval mean."""
    if df.empty:
        return pd.DataFrame(columns=[x_col, y_col, "interval_start", "interval_end", "n_points"])

    tmp = df[[x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[x_col, y_col])

    if tmp.empty:
        return pd.DataFrame(columns=[x_col, y_col, "interval_start", "interval_end", "n_points"])

    tmp["interval_start"] = np.floor(tmp[x_col] / interval) * interval
    out = (
        tmp.groupby("interval_start", dropna=False)
        .agg(**{y_col: (y_col, "mean"), "n_points": (y_col, "count")})
        .reset_index()
        .sort_values("interval_start")
    )
    out["interval_end"] = out["interval_start"] + interval
    out[x_col] = out["interval_start"] + 0.5 * interval
    return out[[x_col, y_col, "interval_start", "interval_end", "n_points"]]


def get_scheme_style(scheme: str) -> Dict[str, Any]:
    color = COLORS.get(scheme, "black")
    return {
        "color": color,
        "linestyle": LINESTYLES.get(scheme, "-"),
        "marker": MARKERS.get(scheme, "o"),
        "markerfacecolor": MARKER_FACES.get(scheme, "none"),
        "markeredgecolor": color,
        "markeredgewidth": 1.65,
        "linewidth": LINE_WIDTHS.get(scheme, 1.15),
        "markersize": 8.8,
    }


def display_label(name: Any) -> str:
    text = str(name)
    return DISPLAY_LABELS.get(text, text)


def apply_axis_font_style(ax: plt.Axes) -> None:
    """Apply the larger bold paper font style consistently to one axis."""
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_weight("bold")
    ax.title.set_size(AXIS_LABEL_FONT_SIZE)
    ax.title.set_weight("bold")

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(TICK_FONT_SIZE)
        tick_label.set_fontweight("bold")

    ax.xaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
    ax.yaxis.get_offset_text().set_fontweight("bold")


def apply_legend_font_style(leg: Any) -> None:
    """Apply the larger bold paper font style consistently to one legend."""
    if leg is None:
        return

    for text in leg.get_texts():
        text.set_fontsize(LEGEND_FONT_SIZE)
        text.set_fontweight("bold")

    title = leg.get_title()
    if title is not None:
        title.set_fontsize(LEGEND_FONT_SIZE)
        title.set_fontweight("bold")


def apply_bar_legend_font_style(leg: Any) -> None:
    """Apply a larger bold style to bar-chart legends."""
    if leg is None:
        return

    for text in leg.get_texts():
        text.set_fontsize(BAR_LEGEND_FONT_SIZE)
        text.set_fontweight("bold")

    title = leg.get_title()
    if title is not None:
        title.set_fontsize(BAR_LEGEND_FONT_SIZE)
        title.set_fontweight("bold")


def finalize_line_axis(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    legend: bool = True,
    legend_loc: str = "upper center",
    legend_markerscale: float = 1.0,
) -> None:
    """Make the axis look like the reference figure."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_axis_font_style(ax)

    ax.grid(True, which="major", linestyle="-", linewidth=0.4, color="0.85", alpha=0.65)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.25")

    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    ax.margins(x=0.02, y=0.18)

    if legend:
        leg = ax.legend(
            loc=legend_loc,
            ncol=2,
            markerscale=legend_markerscale,
            frameon=True,
            fancybox=False,
            framealpha=0.50,
            edgecolor="0.45",
            facecolor="0.90",
            handlelength=2.2,
            handletextpad=0.78,
            columnspacing=1.35,
            borderpad=0.45,
            labelspacing=0.45,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )
        leg.get_frame().set_linewidth(0.6)
        apply_legend_font_style(leg)


def finalize_bar_axis(
    ax: plt.Axes,
    ylabel: Optional[str] = None,
    legend: bool = False,
) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    apply_axis_font_style(ax)

    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="0.85", alpha=0.65)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.25")

    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)

    if legend:
        leg = ax.legend(
            loc="best",
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.50,
            edgecolor="0.45",
            facecolor="0.90",
            handlelength=2.2,
            handletextpad=0.78,
            columnspacing=1.35,
            borderpad=0.45,
            labelspacing=0.45,
            prop={"size": BAR_LEGEND_FONT_SIZE, "weight": "bold"},
        )
        leg.get_frame().set_linewidth(0.6)
        apply_bar_legend_font_style(leg)


def add_instruction_switch_line(
    ax: plt.Axes,
    switch_step: float,
    switch_label: Optional[str] = None,
    label_side: str = "right",
    label_y_frac: float = 0.04,
) -> None:
    """Draw the instruction switch as a vertical dashed line."""
    ax.axvline(
        float(switch_step),
        color="0.20",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    if switch_label:
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        x_offset = 0.008 * (xmax - xmin)
        y = ymin + float(label_y_frac) * (ymax - ymin)
        if label_side == "left":
            x = float(switch_step) - x_offset
            ha = "right"
        else:
            x = float(switch_step) + x_offset
            ha = "left"
        va = "center" if 0.20 <= float(label_y_frac) <= 0.80 else "bottom"

        ax.text(
            x,
            y,
            switch_label,
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight="bold",
            rotation=90,
            va=va,
            ha=ha,
            color="0.15",
        )


# Only coordinate-independent layout hints (switch-label placement) remain here.
# Y-axis bounds are intentionally omitted so each plot autoscales to its own data.
INSTRUCTION_RESPONSE_LINE_OVERRIDES = {
    "fig_instruction_response_aoi_random_middle_unseen_seed_aoi": {
        "switch_label_side": "left",
    },
    "fig_instruction_response_quality_random_middle_unseen_seed_aoi": {
        "switch_label_side": "left",
        "switch_label_y_frac": 0.50,
    },
}


def apply_line_plot_overrides(ax: plt.Axes, out_base: Path) -> dict:
    """Apply targeted paper-layout adjustments for selected instruction-response plots."""
    override = INSTRUCTION_RESPONSE_LINE_OVERRIDES.get(out_base.name, {})
    ylim = override.get("ylim")
    if ylim is not None:
        ymin, ymax = ax.get_ylim()
        lower = ymin if ylim[0] is None else float(ylim[0])
        upper = ymax if ylim[1] is None else float(ylim[1])
        ax.set_ylim(lower, upper)
    return override


def save_paper_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")


def aggregate_timeseries(df: pd.DataFrame, metric: str, group_cols: Sequence[str]) -> pd.DataFrame:
    tmp = df.dropna(subset=["slot", metric, "scheme", "seed"]).copy()

    if tmp.empty:
        return pd.DataFrame()

    tmp[metric] = numeric(tmp[metric])
    tmp = tmp.dropna(subset=[metric])

    per_seed = (
        tmp.groupby([*group_cols, "scheme", "seed", "slot"], dropna=False)[metric]
        .mean()
        .reset_index(name="seed_mean")
    )

    agg = (
        per_seed.groupby([*group_cols, "scheme", "slot"], dropna=False)["seed_mean"]
        .agg(mean_raw="mean", std="std", n_seed="count")
        .reset_index()
    )

    agg["std"] = agg["std"].fillna(0.0)
    return agg


def plot_timeseries(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_base: Path,
    section: SectionLog,
    args: argparse.Namespace,
    schemes: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = (),
    switch_step: Optional[float] = None,
    switch_label: Optional[str] = None,
    interval: float = PLOT_INTERVAL,
) -> bool:
    if metric not in df.columns:
        section.skip(out_base.name, f"missing metric field: {metric}")
        return False

    agg = aggregate_timeseries(df, metric, group_cols)

    if agg.empty:
        section.skip(out_base.name, f"no slot-level numeric data for metric: {metric}")
        return False

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    scheme_order = list(schemes) if schemes else SCHEME_ORDER
    present = [s for s in scheme_order if s in set(agg["scheme"].astype(str))]
    present += sorted(set(agg["scheme"].astype(str)) - set(present))
    plot_frames: List[pd.DataFrame] = []

    for scheme in present:
        g = agg[agg["scheme"].astype(str) == scheme].sort_values("slot")

        if g.empty:
            continue

        style = get_scheme_style(scheme)
        points = interval_average_points(g, "slot", "mean_raw", interval=interval)
        if points.empty:
            continue
        points["scheme"] = scheme
        plot_frames.append(points[["scheme", "slot", "mean_raw", "interval_start", "interval_end", "n_points"]])

        ax.plot(
            points["slot"],
            points["mean_raw"],
            label=display_label(scheme),
            **style,
        )

    if args.save_csv and plot_frames:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(plot_frames, ignore_index=True, sort=False).to_csv(out_base.with_suffix(".csv"), index=False)
        section.ok(out_base.with_suffix(".csv"))

    is_instruction_response = INSTRUCTION_DIR in out_base.parts
    is_policy_plot = POLICY_DIR in out_base.parts
    is_training_policy_plot = is_policy_plot and "training_seeds" in out_base.parts
    is_non_training_policy_plot = is_policy_plot and "non_training_seeds" in out_base.parts

    if ylabel in {"Reward per slot", "Average reward per slot"} and not is_instruction_response:
        legend_loc = "upper right"
    elif "AoI" in ylabel or "Normalized reconstruction quality" in ylabel:
        legend_loc = "upper left"
    else:
        legend_loc = "upper center"

    # Y-axis bounds are left to matplotlib autoscale (finalize_line_axis applies a
    # uniform 18% vertical margin), so the range adapts to the data instead of the
    # previously hand-tuned fixed limits.

    line_override = apply_line_plot_overrides(ax, out_base)
    if switch_step is not None:
        add_instruction_switch_line(
            ax,
            float(switch_step),
            switch_label,
            label_side=str(line_override.get("switch_label_side", "right")),
            label_y_frac=float(line_override.get("switch_label_y_frac", 0.04)),
        )

    legend_markerscale = 0.75 if POLICY_DIR in out_base.parts else 1.0
    finalize_line_axis(
        ax,
        "Slot",
        ylabel,
        legend=True,
        legend_loc=legend_loc,
        legend_markerscale=legend_markerscale,
    )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_base)
    plt.close(fig)

    section.ok(out_base.with_suffix(".pdf"))
    section.ok(out_base.with_suffix(".png"))
    return True


def literal_list(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return list(parsed)
        return [parsed]
    except Exception:
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
        return [float(x) for x in nums]


def flatten_nested(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []

    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            out.extend(flatten_nested(value))
        elif value is not None and not (isinstance(value, float) and np.isnan(value)):
            out.append(value)

    return out


def instruction_window_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["relative_slot"] = numeric(out["slot"]) - numeric(out["switch_step_resolved"])

    return out[
        (out["relative_slot"] >= -int(args.before_window))
        & (out["relative_slot"] <= int(args.after_window))
    ].copy()


def plot_event_aligned_response(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_base: Path,
    section: SectionLog,
    args: argparse.Namespace,
    switch_label: str,
    interval: float = PLOT_INTERVAL,
) -> bool:
    if metric not in df.columns:
        section.skip(out_base.name, f"missing metric field: {metric}")
        return False

    tmp = instruction_window_rows(df.dropna(subset=["slot", "switch_step_resolved", metric]).copy(), args)

    if tmp.empty:
        section.skip(out_base.name, "no rows in the event-aligned switch window")
        return False

    tmp[metric] = numeric(tmp[metric])
    tmp = tmp.dropna(subset=[metric])

    base_cols = ["source_file", "scenario", "scheme", "seed", "episode"]
    base_cols = [c for c in base_cols if c in tmp.columns]

    pre = tmp[(tmp["relative_slot"] >= -int(args.before_window)) & (tmp["relative_slot"] < 0)]

    if pre.empty:
        section.skip(out_base.name, "no pre-switch rows for baseline normalization")
        return False

    baseline = pre.groupby(base_cols, dropna=False)[metric].mean().reset_index(name="pre_mean")
    tmp = tmp.merge(baseline, on=base_cols, how="left")
    tmp = tmp.dropna(subset=["pre_mean"])

    eps = 1e-8
    tmp["response_delta"] = tmp[metric] - tmp["pre_mean"]
    tmp["response_ratio_pct"] = 100.0 * tmp["response_delta"] / (tmp["pre_mean"].abs() + eps)
    tmp["relative_slot"] = np.rint(tmp["relative_slot"]).astype(int)

    per_seed = (
        tmp.groupby(["scheme", "seed", "relative_slot"], dropna=False)
        .agg(metric_seed_mean=(metric, "mean"), response_seed_mean=("response_ratio_pct", "mean"))
        .reset_index()
    )

    agg = (
        per_seed.groupby(["scheme", "relative_slot"], dropna=False)
        .agg(
            mean_metric_raw=("metric_seed_mean", "mean"),
            mean_response_raw=("response_seed_mean", "mean"),
            std_response=("response_seed_mean", "std"),
            n_seed=("seed", "count"),
        )
        .reset_index()
    )

    agg["std_response"] = agg["std_response"].fillna(0.0)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    present = [s for s in ["IC-MAPPO", "IC-HAPPO"] if s in set(agg["scheme"].astype(str))]
    present += sorted(set(agg["scheme"].astype(str)) - set(present))
    plot_frames = []

    for scheme in present:
        g = agg[agg["scheme"].astype(str) == scheme].sort_values("relative_slot")

        if g.empty:
            continue

        style = get_scheme_style(scheme)
        points = interval_average_points(g, "relative_slot", "mean_response_raw", interval=interval)
        if points.empty:
            continue
        points["scheme"] = scheme
        plot_frames.append(
            points[["scheme", "relative_slot", "mean_response_raw", "interval_start", "interval_end", "n_points"]]
        )

        ax.plot(
            points["relative_slot"],
            points["mean_response_raw"],
            label=display_label(scheme),
            **style,
        )

    if args.save_csv and plot_frames:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(plot_frames, ignore_index=True, sort=False).to_csv(out_base.with_suffix(".csv"), index=False)
        section.ok(out_base.with_suffix(".csv"))

    add_instruction_switch_line(ax, 0.0, switch_label)
    is_instruction_response = INSTRUCTION_DIR in out_base.parts

    if ylabel in {"Reward per slot", "Average reward per slot"} and not is_instruction_response:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, -0.05), 0.30)
        legend_loc = "upper right"
    elif "AoI" in ylabel or "Normalized reconstruction quality" in ylabel:
        legend_loc = "upper left"
    else:
        legend_loc = "upper center"
    finalize_line_axis(ax, "Slot relative to switch", ylabel, legend=True, legend_loc=legend_loc)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_base)
    plt.close(fig)

    section.ok(out_base.with_suffix(".pdf"))
    section.ok(out_base.with_suffix(".png"))
    return True


def before_after_delta_table(df: pd.DataFrame, metrics: Sequence[str], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    tmp = df.dropna(subset=["slot", "switch_step_resolved"]).copy()

    if tmp.empty:
        return pd.DataFrame()

    tmp["relative_slot"] = numeric(tmp["slot"]) - numeric(tmp["switch_step_resolved"])

    group_cols = ["scheme", "seed", "episode"]
    group_cols = [c for c in group_cols if c in tmp.columns]

    for metric in metrics:
        if metric not in tmp.columns:
            continue

        valid = tmp.dropna(subset=[metric]).copy()
        valid[metric] = numeric(valid[metric])

        pre = valid[(valid["relative_slot"] >= -int(args.before_window)) & (valid["relative_slot"] < 0)]
        post = valid[(valid["relative_slot"] >= 0) & (valid["relative_slot"] < int(args.after_window))]

        if pre.empty or post.empty:
            continue

        pre_m = pre.groupby(group_cols, dropna=False)[metric].mean().reset_index(name="pre_mean")
        post_m = post.groupby(group_cols, dropna=False)[metric].mean().reset_index(name="post_mean")
        merged = pre_m.merge(post_m, on=group_cols, how="inner")

        if merged.empty:
            continue

        merged["delta"] = merged["post_mean"] - merged["pre_mean"]
        merged["relative_delta_pct"] = 100.0 * merged["delta"] / (merged["pre_mean"].abs() + 1e-8)
        merged["metric"] = metric
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    per_episode = pd.concat(rows, ignore_index=True, sort=False)

    return (
        per_episode.groupby(["scheme", "metric"], dropna=False)
        .agg(
            pre_mean=("pre_mean", "mean"),
            post_mean=("post_mean", "mean"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            relative_delta_pct_mean=("relative_delta_pct", "mean"),
            n=("delta", "count"),
        )
        .reset_index()
        .fillna({"delta_std": 0.0})
    )


def plot_before_after_delta(
    df: pd.DataFrame,
    out_base: Path,
    section: SectionLog,
    args: argparse.Namespace,
) -> bool:
    metric_order = [m for m in ("reward", "aoi", "quality", "scheduled", "load") if m in df.columns]

    if not metric_order:
        section.skip(out_base.name, "missing reward/aoi/quality/scheduled/load fields")
        return False

    table = before_after_delta_table(df, metric_order, args)

    if table.empty:
        section.skip(out_base.name, "no before/after switch windows available")
        return False

    if args.save_csv:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_base.with_suffix(".csv"), index=False)
        section.ok(out_base.with_suffix(".csv"))

    fig, axes = plt.subplots(
        1,
        len(metric_order),
        figsize=(max(3.45, 1.35 * len(metric_order)), 2.35),
        sharex=False,
    )

    if len(metric_order) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_order):
        sub = table[table["metric"] == metric].copy()
        schemes = [s for s in ["IC-MAPPO", "IC-HAPPO"] if s in set(sub["scheme"].astype(str))]

        xpos = np.arange(len(schemes))
        vals = [
            float(sub.loc[sub["scheme"] == s, "relative_delta_pct_mean"].iloc[0])
            if s in set(sub["scheme"])
            else 0.0
            for s in schemes
        ]
        colors = [COLORS.get(s, "black") for s in schemes]

        ax.bar(xpos, vals, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, color="0.20", linewidth=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(schemes, rotation=25, ha="right")
        ax.set_xlabel(metric)
        finalize_bar_axis(ax)

    axes[0].set_ylabel("Post-switch change (%)")
    for ax in axes:
        apply_axis_font_style(ax)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_base)
    plt.close(fig)

    section.ok(out_base.with_suffix(".pdf"))
    section.ok(out_base.with_suffix(".png"))
    return True


def distribution_shift_table(
    df: pd.DataFrame,
    field: str,
    values: Sequence[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if field not in df.columns:
        return pd.DataFrame()

    tmp = df.dropna(subset=["slot", "switch_step_resolved", field]).copy()

    if tmp.empty:
        return pd.DataFrame()

    tmp["relative_slot"] = numeric(tmp["slot"]) - numeric(tmp["switch_step_resolved"])

    rows = []

    for scheme, sg in tmp.groupby("scheme", dropna=False):
        for segment, cond in (
            ("pre", (sg["relative_slot"] >= -int(args.before_window)) & (sg["relative_slot"] < 0)),
            ("post", (sg["relative_slot"] >= 0) & (sg["relative_slot"] < int(args.after_window))),
        ):
            vals = flatten_nested(literal_list(v) for v in sg.loc[cond, field].dropna())
            ints = []

            for val in vals:
                try:
                    ints.append(int(float(val)))
                except Exception:
                    continue

            total = max(len(ints), 1)

            for item in values:
                rows.append(
                    {
                        "scheme": scheme,
                        "segment": segment,
                        "value": item,
                        "ratio": ints.count(item) / total,
                        "count": ints.count(item),
                        "total": len(ints),
                    }
                )

    return pd.DataFrame(rows)


def plot_distribution_shift(
    df: pd.DataFrame,
    field: str,
    values: Sequence[int],
    value_prefix: str,
    out_base: Path,
    section: SectionLog,
    args: argparse.Namespace,
) -> bool:
    table = distribution_shift_table(df, field, values, args)

    if table.empty:
        section.skip(out_base.name, f"missing or empty behavior field: {field}")
        return False

    if args.save_csv:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_base.with_suffix(".csv"), index=False)
        section.ok(out_base.with_suffix(".csv"))

    labels = []
    groups = []

    for scheme in ["IC-MAPPO", "IC-HAPPO"]:
        if scheme not in set(table["scheme"].astype(str)):
            continue
        for segment in ("pre", "post"):
            labels.append(f"{scheme}\n{segment}")
            groups.append((scheme, segment))

    if not groups:
        section.skip(out_base.name, "no IC-HAPPO/IC-MAPPO behavior rows")
        return False

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    bottom = np.zeros(len(groups))
    cmap = plt.get_cmap("tab20")

    for i, value in enumerate(values):
        vals = []

        for scheme, segment in groups:
            row = table[
                (table["scheme"].astype(str) == scheme)
                & (table["segment"] == segment)
                & (table["value"] == value)
            ]
            vals.append(float(row["ratio"].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            np.arange(len(groups)),
            vals,
            bottom=bottom,
            width=0.65,
            color=cmap(i),
            edgecolor="black",
            linewidth=0.3,
            label=f"{value_prefix}{value}",
        )
        bottom += np.asarray(vals)

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Selection ratio")

    finalize_bar_axis(ax, legend=True)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_base)
    plt.close(fig)

    section.ok(out_base.with_suffix(".pdf"))
    section.ok(out_base.with_suffix(".png"))
    return True


def alpha_shift_table(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if "alphas" not in df.columns:
        return pd.DataFrame()

    tmp = df.dropna(subset=["slot", "switch_step_resolved", "alphas"]).copy()

    if tmp.empty:
        return pd.DataFrame()

    tmp["relative_slot"] = numeric(tmp["slot"]) - numeric(tmp["switch_step_resolved"])
    rows = []

    for _, row in tmp.iterrows():
        vals = literal_list(row["alphas"])

        if not vals:
            continue

        arr = np.asarray(flatten_nested(vals), dtype=float)

        if arr.size < 3:
            continue

        arr = arr.reshape(-1, 3)
        mean_alpha = np.nanmean(arr, axis=0)

        if -int(args.before_window) <= row["relative_slot"] < 0:
            segment = "pre"
        elif 0 <= row["relative_slot"] < int(args.after_window):
            segment = "post"
        else:
            segment = None

        if segment is None:
            continue

        rows.append(
            {
                "scheme": row["scheme"],
                "seed": row["seed"],
                "episode": row["episode"],
                "segment": segment,
                "alpha_A": mean_alpha[0],
                "alpha_Q": mean_alpha[1],
                "alpha_Lambda": mean_alpha[2],
            }
        )

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)

    return raw.groupby(["scheme", "segment"], dropna=False)[
        ["alpha_A", "alpha_Q", "alpha_Lambda"]
    ].mean().reset_index()


def plot_alpha_shift(df: pd.DataFrame, out_base: Path, section: SectionLog, args: argparse.Namespace) -> bool:
    table = alpha_shift_table(df, args)

    if table.empty:
        section.skip(out_base.name, "missing or empty scheduling alpha field")
        return False

    if args.save_csv:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_base.with_suffix(".csv"), index=False)
        section.ok(out_base.with_suffix(".csv"))

    groups = [
        (scheme, segment)
        for scheme in ["IC-MAPPO", "IC-HAPPO"]
        for segment in ("pre", "post")
        if not table[(table["scheme"] == scheme) & (table["segment"] == segment)].empty
    ]

    if not groups:
        section.skip(out_base.name, "no IC-HAPPO/IC-MAPPO alpha rows")
        return False

    x = np.arange(len(groups))
    width = 0.22

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    for offset, col, label, color in (
        (-width, "alpha_A", r"$\alpha_A$", "blue"),
        (0.0, "alpha_Q", r"$\alpha_Q$", "red"),
        (width, "alpha_Lambda", r"$\alpha_\Lambda$", "black"),
    ):
        vals = []

        for scheme, segment in groups:
            row = table[(table["scheme"] == scheme) & (table["segment"] == segment)]
            vals.append(float(row[col].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            x + offset,
            vals,
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{seg}" for s, seg in groups])
    ax.set_ylabel("Mean scheduling weight")

    finalize_bar_axis(ax, legend=True)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_base)
    plt.close(fig)

    section.ok(out_base.with_suffix(".pdf"))
    section.ok(out_base.with_suffix(".png"))
    return True


def policy_effectiveness(data: pd.DataFrame, out_dir: Path, args: argparse.Namespace, section: SectionLog) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = usable_trajectory(data)

    if traj.empty:
        section.warn(
            "Policy effectiveness plots require slot-level trajectory data; "
            "train_semantic_metrics.csv is not used as a slot trajectory."
        )
        for fig in (
            "fig_policy_reward_expectation_timeseries",
            "fig_policy_aoi_timeseries",
            "fig_policy_quality_timeseries",
        ):
            section.skip(fig, "no slot-level trajectory data with slot/timestep field")
        return

    traj = traj[traj["scheme"].isin(SCHEME_ORDER)].copy()

    if traj.empty:
        section.skip("policy effectiveness", "no recognized policy/baseline schemes in trajectory data")
        return

    random_mask = pd.Series(False, index=traj.index)
    if "scenario" in traj.columns:
        random_mask |= traj["scenario"].astype(str).str.contains("random_switch_once", case=False, na=False)
    if "instruction_mode_strategy" in traj.columns:
        random_mask |= traj["instruction_mode_strategy"].astype(str).str.contains(
            "random_switch_once", case=False, na=False
        )

    if random_mask.any():
        traj = traj[random_mask].copy()
    else:
        section.warn("Policy effectiveness plots now require random_switch_once trajectories.")
        traj = traj.iloc[0:0].copy()

    if traj.empty:
        section.skip("policy effectiveness", "no random-switch trajectory data after filtering fixed scenarios")
        return

    ensure_same_conditions(traj, section)
    traj = common_slot_range(traj, section)

    if args.post_switch_only:
        traj = traj[traj["slot"] >= float(args.switch_step)].copy()

    def render_policy_set(
        plot_df: pd.DataFrame,
        target_dir: Path,
        label: str,
        filename_tag: str = "",
    ) -> None:
        if plot_df.empty:
            section.skip(f"policy effectiveness/{label}", f"no {label} trajectory rows")
            return

        target_dir.mkdir(parents=True, exist_ok=True)
        section.warn(
            f"Policy effectiveness '{label}' uses seeds: "
            f"{', '.join(sorted(plot_df['seed'].astype(str).dropna().unique().tolist()))}"
        )

        plot_timeseries(
            plot_df,
            "reward",
            "Average reward per slot",
            "",
            target_dir / f"fig_policy_{filename_tag}reward_expectation_timeseries",
            section,
            args,
            SCHEME_ORDER,
        )

        plot_timeseries(
            plot_df,
            "aoi",
            "Average AoI",
            "",
            target_dir / f"fig_policy_{filename_tag}aoi_timeseries",
            section,
            args,
            SCHEME_ORDER,
        )

        if "aoi" not in plot_df.columns and "max_aoi" in plot_df.columns:
            plot_timeseries(
                plot_df,
                "max_aoi",
                "Maximum AoI",
                "",
                target_dir / f"fig_policy_{filename_tag}max_aoi_timeseries",
                section,
                args,
                SCHEME_ORDER,
            )

        plot_timeseries(
            plot_df,
            "quality",
            "Normalized reconstruction quality",
            "",
            target_dir / f"fig_policy_{filename_tag}quality_timeseries",
            section,
            args,
            SCHEME_ORDER,
        )

    render_policy_set(traj, out_dir, "all seeds")

    if "seed_group" in traj.columns:
        render_policy_set(
            traj[traj["seed_group"].astype(str) == "train"].copy(),
            out_dir / "training_seeds",
            "training seeds",
            "training_",
        )
        render_policy_set(
            traj[traj["seed_group"].astype(str) == "unseen"].copy(),
            out_dir / "non_training_seeds",
            "non-training seeds",
            "non-training_",
        )
    else:
        section.skip("policy effectiveness/seed split", "missing seed_group field")


def training_search_roots(results_root: Path) -> List[Path]:
    roots = [results_root]
    if results_root.name == "paper_trajectory_eval":
        roots.append(results_root.parent)
    parent_results = results_root.parent if results_root.parent.name == "results" else None
    if parent_results is not None:
        roots.append(parent_results)

    out: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if root.exists() and key not in seen:
            out.append(root)
            seen.add(key)
    return out


def training_scheme_from_path(path: Path) -> Optional[str]:
    text = str(path).lower()
    if "mappo" in text:
        return "IC-MAPPO"
    if "happo" in text:
        return "IC-HAPPO"
    return None


def plot_ic_training_curves(
    results_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
    section: SectionLog,
) -> None:
    """Plot IC-HAPPO/IC-MAPPO training curves from train_semantic_metrics.csv."""
    files: List[Path] = []
    for root in training_search_roots(results_root):
        files.extend(root.rglob("train_semantic_metrics.csv"))
    sc_files = [p for p in files if "uav_escs_sc" in p.parts]
    if sc_files:
        files = sc_files

    frames: List[pd.DataFrame] = []
    for path in sorted(set(files)):
        scheme = training_scheme_from_path(path)
        if scheme not in IC_SCHEMES:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            section.warn(f"failed to read training metrics {path}: {exc}")
            continue
        if df.empty or "total_num_steps" not in df.columns:
            continue

        required = {"average_episode_rewards", "avg_quality_gain", "avg_aoi"}
        if not required & set(df.columns):
            continue

        keep_cols = ["total_num_steps", *[c for c in required if c in df.columns]]
        part = df[keep_cols].copy()
        part["scheme"] = scheme
        part["source_file"] = str(path)
        frames.append(part)
        section.input_files.append(str(path))

    if not frames:
        section.skip("fig_ic_training_curves", "no IC-HAPPO/IC-MAPPO train_semantic_metrics.csv found")
        return

    data = pd.concat(frames, ignore_index=True, sort=False)
    data["total_num_steps"] = numeric(data["total_num_steps"])
    data = data.dropna(subset=["total_num_steps"])
    if data.empty:
        section.skip("fig_ic_training_curves", "training metrics have no numeric total_num_steps")
        return

    metric_specs = [
        ("average_episode_rewards", "Episode reward"),
        ("avg_quality_gain", "Normalized reconstruction quality"),
        ("avg_aoi", "Average AoI"),
    ]
    metric_specs = [(m, y) for m, y in metric_specs if m in data.columns]
    if not metric_specs:
        section.skip("fig_ic_training_curves", "missing training reward/quality/AoI metrics")
        return

    csv_rows: List[pd.DataFrame] = []
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5.8, 2.45), squeeze=False)
    for ax, (metric, ylabel) in zip(axes[0], metric_specs):
        for scheme in ["IC-MAPPO", "IC-HAPPO"]:
            g = data[data["scheme"] == scheme].copy()
            if g.empty:
                continue
            g[metric] = numeric(g[metric])
            g = g.dropna(subset=[metric]).sort_values("total_num_steps")
            if g.empty:
                continue

            y = g[metric].rolling(args.smooth_window, min_periods=1).mean() if args.smooth_window > 1 else g[metric]
            ax.plot(
                g["total_num_steps"],
                y,
                label=scheme,
                color=COLORS.get(scheme, "black"),
                linestyle=LINESTYLES.get(scheme, "-"),
                linewidth=1.65,
            )

            out = pd.DataFrame(
                {
                    "scheme": scheme,
                    "total_num_steps": g["total_num_steps"].to_numpy(),
                    "metric": metric,
                    "metric_label": ylabel,
                    "value_raw": g[metric].to_numpy(),
                    "value_smooth": np.asarray(y),
                }
            )
            csv_rows.append(out)

        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        apply_axis_font_style(ax)
        ax.grid(True, which="major", linestyle="-", linewidth=0.4, color="0.85", alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("0.25")
        ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        leg = axes[0][-1].legend(
            handles,
            labels,
            loc="best",
            frameon=True,
            fancybox=False,
            framealpha=0.50,
            edgecolor="0.45",
            facecolor="0.90",
            borderpad=0.35,
            labelspacing=0.35,
            columnspacing=1.15,
            handletextpad=0.65,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
        )
        leg.get_frame().set_linewidth(0.6)
        apply_legend_font_style(leg)

    out_base = out_dir / "fig_ic_training_curves"
    if args.save_csv and csv_rows:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(csv_rows, ignore_index=True, sort=False).to_csv(out_base.with_suffix(".csv"), index=False)
        section.ok(out_base.with_suffix(".csv"))

    fig.tight_layout(w_pad=1.0)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_base)
    plt.close(fig)
    section.ok(out_base.with_suffix(".pdf"))
    section.ok(out_base.with_suffix(".png"))


def instruction_value_to_name(value: Any) -> str:
    text = str(value).strip()

    try:
        idx = int(float(text))
        return INSTRUCTION_NAMES.get(idx, f"instruction_{idx}")
    except Exception:
        pass

    text_l = text.lower()

    if text_l in {"1", "aoi"}:
        return "AoI"
    if text_l in {"2", "quality"}:
        return "quality"
    if text_l in {"0", "balance"}:
        return "balance"

    return text if text else "unknown"


def post_instruction_key_from_label(label: str) -> str:
    """Return a compact key based on the post-switch instruction only."""
    text = str(label).replace(" random middle", "").strip()

    if "\u2192" in text:
        text = text.split("\u2192", 1)[1].strip()
    elif "->" in text:
        text = text.split("->", 1)[1].strip()

    text = text.split("@", 1)[0].strip()
    name = instruction_value_to_name(text)
    key = name.lower().replace(" ", "_").replace("-", "_")
    key = re.sub(r"[^a-z0-9_]+", "", key)
    return key or "instruction"


def transition_from_group(group: pd.DataFrame, default_switch: int) -> Tuple[str, float, str]:
    switch_vals = numeric(group["switch_step"]) if "switch_step" in group else pd.Series(dtype=float)
    switch = float(switch_vals.dropna().iloc[0]) if not switch_vals.dropna().empty else float(default_switch)

    scenario = (
        str(group["scenario"].dropna().iloc[0])
        if "scenario" in group.columns and group["scenario"].notna().any()
        else ""
    )

    strategy = (
        str(group["instruction_mode_strategy"].dropna().iloc[0])
        if "instruction_mode_strategy" in group.columns and group["instruction_mode_strategy"].notna().any()
        else ""
    )

    label = None
    key = None

    if "transition" in group.columns and group["transition"].notna().any():
        label = str(group["transition"].dropna().iloc[0])

    elif (
        "before" in group.columns
        and "after" in group.columns
        and group["before"].notna().any()
        and group["after"].notna().any()
    ):
        before = instruction_value_to_name(group["before"].dropna().iloc[0])
        after = instruction_value_to_name(group["after"].dropna().iloc[0])
        label = f"{before} \u2192 {after}"

    elif "instruction" in group.columns and group["instruction"].notna().any():
        before_rows = group[group["slot"] < switch]
        after_rows = group[group["slot"] >= switch]

        if not before_rows.empty and not after_rows.empty:
            before = instruction_value_to_name(before_rows["instruction"].mode(dropna=True).iloc[0])
            after = instruction_value_to_name(after_rows["instruction"].mode(dropna=True).iloc[0])
            label = f"{before} \u2192 {after}"

    if not label:
        label = "Instruction switch"
        key = "unknown_transition"

    if key is None:
        key = post_instruction_key_from_label(label)

    if "switch_once" in strategy and "random_switch_once" not in strategy:
        step_id = int(round(float(switch)))
        label = f"{label} @ {step_id}"
        key = f"{key}_step{step_id}"

    elif "random_switch" in scenario.lower() or "random_switch_once" in strategy:
        key = f"{key}_random_middle"
        label = f"{label} random middle"

    return key, switch, label


def prepare_instruction_data(data: pd.DataFrame, args: argparse.Namespace, section: SectionLog) -> pd.DataFrame:
    traj = usable_trajectory(data)

    if traj.empty:
        section.warn("Instruction response plots require slot-level trajectory data with slot and instruction fields.")
        return pd.DataFrame()

    traj = traj[traj["scheme"].isin(IC_SCHEMES)].copy()

    if traj.empty:
        section.warn("Instruction response plots only use IC-HAPPO and IC-MAPPO; no such trajectory data found.")
        return pd.DataFrame()

    if "instruction" not in traj.columns or not traj["instruction"].notna().any():
        section.warn("Instruction response plots require slot-level trajectory data with slot and instruction fields.")
        return pd.DataFrame()

    switch_mask = pd.Series(False, index=traj.index)

    if "scenario" in traj.columns:
        switch_mask |= traj["scenario"].astype(str).str.contains("switch", case=False, na=False)

    if "instruction_mode_strategy" in traj.columns:
        switch_mask |= traj["instruction_mode_strategy"].astype(str).str.contains("switch", case=False, na=False)

    if switch_mask.any():
        traj = traj[switch_mask].copy()
    else:
        section.warn("Instruction response plots require real switch trajectories; fixed-instruction trajectories are ignored.")
        return pd.DataFrame()

    random_mask = pd.Series(False, index=traj.index)
    if "scenario" in traj.columns:
        random_mask |= traj["scenario"].astype(str).str.contains("random_switch_once", case=False, na=False)
    if "instruction_mode_strategy" in traj.columns:
        random_mask |= traj["instruction_mode_strategy"].astype(str).str.contains(
            "random_switch_once", case=False, na=False
        )
    traj = traj[random_mask].copy()
    if traj.empty:
        section.warn("Instruction response plots require random_switch_once trajectories; no such rows were found.")
        return pd.DataFrame()

    pieces = []
    group_keys = ["source_file"]

    for col in ("scenario", "seed", "episode"):
        if col in traj.columns and traj[col].notna().any():
            group_keys.append(col)

    for _, group in traj.groupby(group_keys, dropna=False):
        key, switch, label = transition_from_group(group, args.switch_step)
        group = group.copy()
        group["transition_key"] = key
        group["transition_label"] = label
        group["switch_step_resolved"] = switch
        pieces.append(group)

    out = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    section.transitions = sorted(out["transition_label"].dropna().unique().tolist()) if not out.empty else []

    return out


def instruction_switch_display_label(label: str) -> str:
    """Use the active post-switch instruction name as the switch label."""
    text = str(label).replace(" random middle", "").strip()
    if "\u2192" in text:
        after = text.split("\u2192", 1)[1].strip()
    elif "->" in text:
        after = text.split("->", 1)[1].strip()
    else:
        after = text
    after = after.split("@", 1)[0].strip()
    return after or "Instruction"


def instruction_legend_label(instruction: str) -> str:
    text = str(instruction).strip()
    if text.lower() == "quality":
        return "Quality"
    if text.lower() == "aoi":
        return "AoI"
    return text


def _with_post_switch_instruction(traj: pd.DataFrame) -> pd.DataFrame:
    out = traj.copy()
    out["instruction"] = out["transition_label"].map(instruction_switch_display_label)
    out = out[out["instruction"].isin(["AoI", "quality"])].copy()
    out["selection_seed"] = out["seed"].map(clean_seed).astype(str)
    out["selection_episode"] = out["episode"].map(clean_seed).astype(str)
    out["selection_key"] = (
        out["seed_group"].astype(str)
        + "|"
        + out["selection_seed"]
        + "|"
        + out["selection_episode"]
        + "|"
        + out["instruction"].astype(str)
    )
    return out


def select_random_instruction_episodes(
    traj: pd.DataFrame,
    section: SectionLog,
    bar_samples_per_instruction: int = 3,
    line_samples_per_instruction: int = 1,
    selection_seed: int = 20260511,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select balanced random-switch episodes for bar summaries and example traces."""
    tmp = _with_post_switch_instruction(traj)
    if tmp.empty:
        section.warn("No AoI/Quality random-switch episodes were available for balanced sampling.")
        return tmp, tmp, pd.DataFrame()

    rng = np.random.default_rng(selection_seed)
    rows = []
    bar_keys: set[str] = set()
    line_keys: set[str] = set()
    required_schemes = {"IC-MAPPO", "IC-HAPPO"}

    episode_table = (
        tmp.groupby(
            [
                "seed_group",
                "selection_seed",
                "selection_episode",
                "instruction",
                "selection_key",
                "switch_step_resolved",
                "transition_key",
                "transition_label",
            ],
            dropna=False,
        )["scheme"]
        .agg(lambda s: sorted(set(map(str, s))))
        .reset_index(name="schemes")
    )
    episode_table = episode_table[
        episode_table["schemes"].map(lambda schemes: required_schemes.issubset(set(schemes)))
    ].copy()

    for seed_group in ("train", "unseen"):
        for instruction in ("AoI", "quality"):
            candidates = episode_table[
                (episode_table["seed_group"].astype(str) == seed_group)
                & (episode_table["instruction"].astype(str) == instruction)
            ].sort_values(["selection_seed", "selection_episode", "switch_step_resolved"])

            if candidates.empty:
                section.warn(f"No common IC-MAPPO/IC-HAPPO random-switch candidates for {seed_group}/{instruction}.")
                continue

            cand_indices = candidates.index.to_numpy()
            bar_n = min(int(bar_samples_per_instruction), len(cand_indices))
            line_n = min(int(line_samples_per_instruction), len(cand_indices))

            if len(cand_indices) < int(bar_samples_per_instruction):
                section.warn(
                    f"Only {len(cand_indices)} random-switch candidates for {seed_group}/{instruction}; "
                    f"using all available candidates for bar statistics."
                )

            selected_bar = rng.choice(cand_indices, size=bar_n, replace=False)
            selected_line = rng.choice(cand_indices, size=line_n, replace=False)

            for purpose, selected in (("bar", selected_bar), ("line", selected_line)):
                for idx in selected:
                    row = candidates.loc[idx]
                    key = str(row["selection_key"])
                    if purpose == "bar":
                        bar_keys.add(key)
                    else:
                        line_keys.add(key)
                    rows.append(
                        {
                            "purpose": purpose,
                            "seed_group": seed_group,
                            "instruction": instruction,
                            "seed": str(row["selection_seed"]),
                            "episode": str(row["selection_episode"]),
                            "switch_step": float(row["switch_step_resolved"]),
                            "transition_key": str(row["transition_key"]),
                            "transition_label": str(row["transition_label"]),
                            "selection_key": key,
                            "selection_seed": int(selection_seed),
                            "available_candidates": int(len(cand_indices)),
                        }
                    )

    manifest = pd.DataFrame(rows)
    bar_traj = tmp[tmp["selection_key"].isin(bar_keys)].copy()
    line_traj = tmp[tmp["selection_key"].isin(line_keys)].copy()
    return bar_traj, line_traj, manifest


def plot_instruction_metric_bar(
    traj: pd.DataFrame,
    out_base: Path,
    metric: str,
    metric_label: str,
    section: SectionLog,
    args: argparse.Namespace,
) -> bool:
    """Summarize one metric under different post-switch instructions."""
    if metric not in traj.columns:
        section.skip(out_base.name, f"missing metric field: {metric}")
        return False

    tmp = traj.dropna(subset=["scheme", "seed_group", "seed", "transition_label", metric]).copy()
    if tmp.empty:
        section.skip(out_base.name, f"no IC trajectory data for metric: {metric}")
        return False

    tmp[metric] = numeric(tmp[metric])
    tmp = tmp[tmp["seed_group"].isin(["train", "unseen"])].dropna(subset=[metric])
    if tmp.empty:
        section.skip(out_base.name, f"no seen/unseen IC data for metric: {metric}")
        return False

    if "instruction" not in tmp.columns:
        tmp["instruction"] = tmp["transition_label"].map(instruction_switch_display_label)
        tmp = tmp[tmp["instruction"].isin(["AoI", "quality"])].copy()
    if tmp.empty:
        section.skip(out_base.name, "no AoI/quality random-switch instruction data")
        return False

    group_cols = ["scheme", "seed_group", "selection_key", "instruction"]
    sample_means = (
        tmp.groupby(group_cols, dropna=False)
        .agg(sample_mean=(metric, "mean"), seed=("seed", "first"), episode=("episode", "first"))
        .reset_index()
    )
    table = (
        sample_means.groupby(["scheme", "seed_group", "instruction"], dropna=False)
        .agg(
            mean=("sample_mean", "mean"),
            std=("sample_mean", "std"),
            n_sample=("selection_key", "count"),
            n_seed=("seed", "nunique"),
        )
        .reset_index()
    )
    table["std"] = table["std"].fillna(0.0)
    table["metric"] = metric
    table["metric_label"] = metric_label

    if args.save_csv:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_base.with_suffix(".csv"), index=False)
        sample_means["metric"] = metric
        sample_means.to_csv(out_base.with_name(out_base.name + "_selected_samples").with_suffix(".csv"), index=False)
        section.ok(out_base.with_suffix(".csv"))
        section.ok(out_base.with_name(out_base.name + "_selected_samples").with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(6.25, 3.75))
    seed_groups = ["train", "unseen"]
    seed_labels = ["Training seeds", "Unseen seeds"]
    instructions = [i for i in ["AoI", "quality"] if i in set(table["instruction"].astype(str))]
    schemes = [s for s in ["IC-MAPPO", "IC-HAPPO"] if s in set(table["scheme"].astype(str))]

    bar_keys = [(inst, scheme) for inst in instructions for scheme in schemes]
    width = 0.16 if len(bar_keys) > 2 else 0.22
    x = np.arange(len(seed_groups))
    hatches = {"AoI": "", "quality": "//"}
    bar_annotations: List[Tuple[Any, float, float]] = []

    for idx, (inst, scheme) in enumerate(bar_keys):
        offset = (idx - (len(bar_keys) - 1) / 2.0) * width
        vals = []
        errs = []
        for group in seed_groups:
            row = table[
                (table["seed_group"] == group)
                & (table["instruction"] == inst)
                & (table["scheme"] == scheme)
            ]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else np.nan)

        label = f"{scheme} ({instruction_legend_label(inst)})"
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs,
            error_kw={
                "ecolor": "0.10",
                "elinewidth": 0.9,
                "capthick": 0.9,
                "capsize": 3.0,
            },
            label=label,
            color=COLORS.get(scheme, "0.5"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.88,
            hatch=hatches.get(inst, ""),
        )
        for rect, value, err in zip(bars, vals, errs):
            if np.isfinite(value):
                bar_annotations.append((rect, float(value), float(err) if np.isfinite(err) else 0.0))

    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels)
    ax.set_ylabel(metric_label)
    finite_means = table["mean"].replace([np.inf, -np.inf], np.nan)
    finite_stds = table["std"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    finite_low = (finite_means - finite_stds).dropna()
    finite_high = (finite_means + finite_stds).dropna()
    if not finite_low.empty and not finite_high.empty:
        ymin = float(finite_low.min())
        ymax = float(finite_high.max())
        if ymin >= 0.0:
            ax.set_ylim(0.0, ymax * 1.65 if ymax > 0.0 else 1.0)
        else:
            span = max(ymax - ymin, 1e-6)
            ax.set_ylim(ymin - 0.15 * span, ymax + 0.65 * span)

    ymin_plot, ymax_plot = ax.get_ylim()
    if ymin_plot < 0.0 < ymax_plot:
        ax.axhline(0.0, color="0.10", linewidth=1.2, zorder=4)

    label_offset = 0.025 * max(ymax_plot - ymin_plot, 1e-6)
    for rect, value, err in bar_annotations:
        if metric == "aoi":
            value_text = f"{value:.2f}"
        else:
            value_text = f"{value:.3f}"
        y = value + err + label_offset if value >= 0.0 else value - err - label_offset
        va = "bottom" if value >= 0.0 else "top"
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            y,
            value_text,
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            rotation=0,
            color="0.15",
            clip_on=False,
        )
    apply_axis_font_style(ax)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.4, color="0.85", alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.25")
    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=True,
            fancybox=False,
            framealpha=0.50,
            edgecolor="0.45",
            facecolor="0.90",
            borderpad=0.45,
            labelspacing=0.45,
            columnspacing=1.35,
            handletextpad=0.78,
            prop={"size": BAR_LEGEND_FONT_SIZE, "weight": "bold"},
        )
        leg.get_frame().set_linewidth(0.6)
        apply_bar_legend_font_style(leg)

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    save_paper_figure(fig, out_base)
    plt.close(fig)
    section.ok(out_base.with_suffix(".pdf"))
    section.ok(out_base.with_suffix(".png"))
    return True


def instruction_response(data: pd.DataFrame, out_dir: Path, args: argparse.Namespace, section: SectionLog) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = prepare_instruction_data(data, args, section)

    if traj.empty:
        for fig in (
            "fig_instruction_response_train_seed_reward",
            "fig_instruction_response_train_seed_aoi",
            "fig_instruction_response_train_seed_quality",
            "fig_instruction_response_unseen_seed_reward",
            "fig_instruction_response_unseen_seed_aoi",
            "fig_instruction_response_unseen_seed_quality",
        ):
            section.skip(fig, "missing slot-level trajectory data with slot and instruction fields")
        return

    if args.post_switch_only:
        traj = traj[traj["slot"] >= traj["switch_step_resolved"]].copy()

    bar_traj, line_traj, selection_manifest = select_random_instruction_episodes(traj, section)
    if args.save_csv and not selection_manifest.empty:
        selection_path = out_dir / "fig_instruction_random-switch_selection_manifest.csv"
        selection_manifest.to_csv(selection_path, index=False)
        section.ok(selection_path)

    plot_instruction_metric_bar(
        bar_traj,
        out_dir / "fig_instruction_random-switch_aoi_bar",
        "aoi",
        "Average AoI",
        section,
        args,
    )
    plot_instruction_metric_bar(
        bar_traj,
        out_dir / "fig_instruction_random-switch_quality_bar",
        "quality",
        "Average normalized reconstruction quality",
        section,
        args,
    )
    plot_instruction_metric_bar(
        bar_traj,
        out_dir / "fig_instruction_random-switch_reward_bar",
        "reward",
        "Average reward per slot",
        section,
        args,
    )

    transitions = sorted(line_traj["transition_key"].dropna().unique().tolist())
    multi_transition = len(transitions) > 1

    for key in transitions:
        g = line_traj[line_traj["transition_key"] == key].copy()

        label = (
            str(g["transition_label"].dropna().iloc[0])
            if g["transition_label"].notna().any()
            else "Instruction switch"
        )
        switch_display_label = instruction_switch_display_label(label)

        switch_values = numeric(g["switch_step_resolved"]).dropna()
        switch = float(switch_values.median()) if not switch_values.empty else float(args.switch_step)

        if switch_values.nunique() > 1:
            section.warn(
                f"{label} has varying switch steps; raw slot plots mark the median switch step {switch:g}, "
                "while event-aligned plots use each episode's own switch step."
            )

        target_dir = out_dir / key if multi_transition else out_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        for seed_group, prefix in (("train", "train_seed"), ("unseen", "unseen_seed")):
            sg = g[g["seed_group"] == seed_group].copy()

            if sg.empty:
                section.skip(f"{key}/{prefix}", f"no {seed_group} seed trajectory data")
                continue

            plot_timeseries(
                sg,
                "reward",
                "Average reward per slot",
                "",
                target_dir / f"fig_instruction_response_{key}_{prefix}_reward",
                section,
                args,
                ["IC-MAPPO", "IC-HAPPO"],
                group_cols=("seed_group",),
                switch_step=switch,
                switch_label=switch_display_label,
                interval=25.0,
            )

            plot_timeseries(
                sg,
                "aoi",
                "Average AoI",
                "",
                target_dir / f"fig_instruction_response_{key}_{prefix}_aoi",
                section,
                args,
                ["IC-MAPPO", "IC-HAPPO"],
                group_cols=("seed_group",),
                switch_step=switch,
                switch_label=switch_display_label,
                interval=25.0,
            )

            plot_timeseries(
                sg,
                "quality",
                "Normalized reconstruction quality",
                "",
                target_dir / f"fig_instruction_response_{key}_{prefix}_quality",
                section,
                args,
                ["IC-MAPPO", "IC-HAPPO"],
                group_cols=("seed_group",),
                switch_step=switch,
                switch_label=switch_display_label,
                interval=25.0,
            )


def write_section_readme(out_dir: Path, section: SectionLog) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {section.subsection}",
        "",
        "## Input data files",
        *[f"- `{p}`" for p in sorted(set(section.input_files))],
        "",
        "## Recognized schemes",
        *[f"- {s}" for s in sorted(set(section.schemes))],
        "",
        "## Recognized seeds",
        *[f"- {s}" for s in sorted(set(section.seeds))],
        "",
        "## Seed groups",
        f"- training seeds: {', '.join(section.train_seeds) if section.train_seeds else 'none'}",
        f"- unseen seeds: {', '.join(section.unseen_seeds) if section.unseen_seeds else 'none'}",
        f"- unknown seeds: {', '.join(section.unknown_seeds) if section.unknown_seeds else 'none'}",
        "",
        "## Instruction transitions",
        *[f"- {t}" for t in sorted(set(section.transitions))],
        "",
        "## Generated figures and CSVs",
        *[f"- `{p}`: {section.subsection}" for p in section.generated],
        "",
        "## Skipped figures",
        *[f"- {s}" for s in section.skipped],
        "",
        "## Warnings",
        *[f"- {w}" for w in section.warnings],
    ]

    if not section.input_files:
        lines.insert(3, "- none")

    if not section.generated:
        lines.insert(lines.index("## Generated figures and CSVs") + 1, "- none")

    if not section.skipped:
        lines.insert(lines.index("## Skipped figures") + 1, "- none")

    if not section.warnings:
        lines.insert(lines.index("## Warnings") + 1, "- none")

    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    section.ok(path)


def write_root_readme(out_dir: Path, policy: SectionLog, instr: SectionLog, run: RunLog) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Semantic Video Paper Figures",
        "",
        "This directory contains trajectory-based figures for the paper result subsections.",
        "",
        "## Subsections",
        f"- `01_effectiveness_cooperative_learning_policies/`: {SUBSECTION_POLICY}",
        f"- `02_instruction_response_analysis/`: {SUBSECTION_INSTRUCTION}",
        "",
        "## Found data files",
        *[f"- `{p}`" for p in sorted(set(run.data_files))],
        "",
        "## Recognized schemes",
        *[f"- {s}" for s in sorted(set(run.schemes))],
        "",
        "## Recognized seeds",
        *[f"- {s}" for s in sorted(set(run.seeds))],
        "",
        "## Seed groups",
        f"- training seeds: {', '.join(run.train_seeds) if run.train_seeds else 'none'}",
        f"- unseen seeds: {', '.join(run.unseen_seeds) if run.unseen_seeds else 'none'}",
        f"- unknown seeds: {', '.join(run.unknown_seeds) if run.unknown_seeds else 'none'}",
        "",
        "## Instruction transitions",
        *[f"- {t}" for t in sorted(set(run.transitions))],
        "",
        "## Generated outputs",
        *[f"- `{p}`" for p in run.generated],
        "",
        "## Skipped figures and warnings",
        *[f"- SKIPPED: {s}" for s in run.skipped],
        *[f"- WARNING: {w}" for w in run.warnings],
    ]

    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"GENERATED root README {path}")


def collect_run_log(policy: SectionLog, instr: SectionLog) -> RunLog:
    run = RunLog()

    for section in (policy, instr):
        run.data_files.extend(section.input_files)
        run.schemes.extend(section.schemes)
        run.seeds.extend(section.seeds)
        run.train_seeds.extend(section.train_seeds)
        run.unseen_seeds.extend(section.unseen_seeds)
        run.unknown_seeds.extend(section.unknown_seeds)
        run.transitions.extend(section.transitions)
        run.warnings.extend(f"{section.subsection}: {w}" for w in section.warnings)
        run.skipped.extend(f"{section.subsection}: {s}" for s in section.skipped)
        run.generated.extend(section.generated)

    for attr in (
        "data_files",
        "schemes",
        "seeds",
        "train_seeds",
        "unseen_seeds",
        "unknown_seeds",
        "transitions",
        "generated",
    ):
        setattr(run, attr, sorted(set(getattr(run, attr))))

    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--results_root", type=Path, default=Path("examples/results"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("examples/results/plot_results/semantic_video_paper_figures"),
    )
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--switch_step", type=int, default=200)
    parser.add_argument(
        "--plot_std",
        type=str2bool,
        default=False,
        help="If True, draw a light mean ± std band around each trajectory curve.",
    )
    parser.add_argument("--save_csv", type=str2bool, default=True)
    parser.add_argument("--train_seeds", type=str, default="")
    parser.add_argument("--unseen_seeds", type=str, default="")
    parser.add_argument("--only_policy_effectiveness", action="store_true")
    parser.add_argument("--only_instruction_response", action="store_true")
    parser.add_argument("--post_switch_only", type=str2bool, default=False)
    parser.add_argument("--before_window", type=int, default=100)
    parser.add_argument("--after_window", type=int, default=100)

    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()

    policy_dir = output_dir / POLICY_DIR
    instr_dir = output_dir / INSTRUCTION_DIR

    train_seeds = parse_seed_list(args.train_seeds)
    unseen_seeds = parse_seed_list(args.unseen_seeds)

    print(f"Scanning results root: {results_root}")
    print(f"Output directory: {output_dir}")

    if not results_root.exists():
        print(f"ERROR: results_root does not exist: {results_root}")
        return 2

    policy_log = SectionLog(SUBSECTION_POLICY)
    instr_log = SectionLog(SUBSECTION_INSTRUCTION)

    data_policy = discover(results_root, output_dir, train_seeds, unseen_seeds, policy_log)
    data_instr = discover(results_root, output_dir, train_seeds, unseen_seeds, instr_log)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_policy.empty:
        normalized = output_dir / "paper_figure_normalized_data.csv"
        data_policy.to_csv(normalized, index=False)
        policy_log.ok(normalized)

    run_all = not (args.only_policy_effectiveness or args.only_instruction_response)

    if run_all or args.only_policy_effectiveness:
        policy_effectiveness(data_policy, policy_dir, args, policy_log)

    if run_all or args.only_instruction_response:
        instruction_response(data_instr, instr_dir, args, instr_log)

    write_section_readme(policy_dir, policy_log)
    write_section_readme(instr_dir, instr_log)

    run = collect_run_log(policy_log, instr_log)
    write_root_readme(output_dir, policy_log, instr_log, run)

    print("\nSummary")

    print("Found data files:")
    for p in run.data_files:
        print(f"  {p}")

    print("Recognized schemes:", ", ".join(run.schemes) if run.schemes else "none")
    print("Recognized seeds:", ", ".join(run.seeds) if run.seeds else "none")

    print("Seed groups:")
    print("  training seeds:", ", ".join(run.train_seeds) if run.train_seeds else "none")
    print("  unseen seeds:", ", ".join(run.unseen_seeds) if run.unseen_seeds else "none")
    print("  unknown seeds:", ", ".join(run.unknown_seeds) if run.unknown_seeds else "none")

    print("Instruction transitions:", ", ".join(run.transitions) if run.transitions else "none")

    print("Generated outputs:")
    for p in run.generated:
        print(f"  {p}")

    print("Skipped:")
    for s in run.skipped:
        print(f"  {s}")

    print("Warnings:")
    for w in run.warnings:
        print(f"  {w}")

    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
