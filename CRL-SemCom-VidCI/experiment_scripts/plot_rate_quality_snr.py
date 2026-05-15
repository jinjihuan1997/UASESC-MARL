import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_mu_label(mu_value: float) -> str:
    return f"mu={mu_value:g}"


def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    required_columns = {"mu_comm", "comm_snr_db", "psnr", "bar_ls_paper"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    return df.sort_values(["mu_comm", "comm_snr_db"]).reset_index(drop=True)


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Keep nondominated points: lower rate and higher PSNR are preferred."""
    frontier_indices = []
    rows = df.reset_index(drop=True)

    for idx, row in rows.iterrows():
        dominated = False
        for jdx, other in rows.iterrows():
            if idx == jdx:
                continue

            better_or_equal_rate = other["bar_ls_paper"] <= row["bar_ls_paper"]
            better_or_equal_psnr = other["psnr"] >= row["psnr"]
            strictly_better = (
                other["bar_ls_paper"] < row["bar_ls_paper"]
                or other["psnr"] > row["psnr"]
            )
            if better_or_equal_rate and better_or_equal_psnr and strictly_better:
                dominated = True
                break

        if not dominated:
            frontier_indices.append(idx)

    frontier = rows.loc[frontier_indices].copy()
    frontier = frontier.sort_values(["bar_ls_paper", "psnr"], ascending=[True, True]).reset_index(drop=True)
    return frontier


def plot_rate_quality_snr(df: pd.DataFrame, output_path: Path) -> None:
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)

    mu_values = sorted(df["mu_comm"].unique())
    color_map = plt.get_cmap("tab10")
    color_by_mu = {mu: color_map(idx % 10) for idx, mu in enumerate(mu_values)}

    for mu in mu_values:
        subset = df[df["mu_comm"] == mu].sort_values("comm_snr_db")
        color = color_by_mu[mu]
        label = build_mu_label(mu)

        axes[0].plot(
            subset["bar_ls_paper"],
            subset["psnr"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        for _, row in subset.iterrows():
            axes[0].annotate(
                f"{int(row['comm_snr_db'])}dB",
                (row["bar_ls_paper"], row["psnr"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=color,
            )

        axes[1].plot(
            subset["comm_snr_db"],
            subset["psnr"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

        axes[2].plot(
            subset["comm_snr_db"],
            subset["bar_ls_paper"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

    axes[0].set_title("Rate vs Quality")
    axes[0].set_xlabel("bar_l_s")
    axes[0].set_ylabel("PSNR (dB)")

    axes[1].set_title("SNR vs Quality")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("PSNR (dB)")

    axes[2].set_title("SNR vs Rate")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("bar_l_s")

    for ax in axes:
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("SemCom Rate-Quality-SNR Comparison", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_frontier(df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)

    mu_values = sorted(df["mu_comm"].unique())
    color_map = plt.get_cmap("tab10")
    color_by_mu = {mu: color_map(idx % 10) for idx, mu in enumerate(mu_values)}

    for mu in mu_values:
        subset = df[df["mu_comm"] == mu].sort_values("bar_ls_paper")
        if subset.empty:
            continue

        color = color_by_mu[mu]
        label = build_mu_label(mu)
        ax.plot(
            subset["bar_ls_paper"],
            subset["psnr"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

        for _, row in subset.iterrows():
            ax.annotate(
                f"SNR {int(row['comm_snr_db'])} dB",
                (row["bar_ls_paper"], row["psnr"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=color,
            )

    ax.set_title("Pareto Frontier: Rate vs Quality")
    ax.set_xlabel("bar_l_s")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(loc="best", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot SemCom rate-quality-SNR comparisons from a CSV summary.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("logs/rate_quality_mcs_map_2026-04-22/all_rate_quality_mcs_map_2026-04-22.csv"),
        help="Input CSV with columns including mu_comm, comm_snr_db, psnr, bar_ls_paper.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/rate_quality_mcs_map_2026-04-22/rate_quality_snr_comparison_2026-04-22.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--pareto-only",
        action="store_true",
        help="Plot only the Pareto frontier in the rate-quality plane.",
    )
    parser.add_argument(
        "--pareto-csv",
        type=Path,
        default=Path("logs/rate_quality_mcs_map_2026-04-22/rate_quality_pareto_frontier_2026-04-22.csv"),
        help="Output CSV path for Pareto frontier rows.",
    )
    args = parser.parse_args()
    df = load_summary(args.csv)
    if args.pareto_only:
        frontier = compute_pareto_frontier(df)
        args.pareto_csv.parent.mkdir(parents=True, exist_ok=True)
        frontier.to_csv(args.pareto_csv, index=False)
        plot_pareto_frontier(frontier, args.output)
        return

    plot_rate_quality_snr(df, args.output)


if __name__ == "__main__":
    main()
