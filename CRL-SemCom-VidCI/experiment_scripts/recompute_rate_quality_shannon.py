import argparse
import json
import math
from pathlib import Path

import pandas as pd


def recompute_side_term(height: int, width: int, snr_db: float) -> float:
    side_info_bits = float(height) * float(width) / 32.0
    snr_linear = 10 ** (float(snr_db) / 10.0)
    return side_info_bits / math.log2(1.0 + snr_linear)


def recompute_table(df: pd.DataFrame, height: int, width: int) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        new_row = row.to_dict()
        avg_ls_side = recompute_side_term(height=height, width=width, snr_db=float(row["comm_snr_db"]))
        new_row["avg_ls_side"] = avg_ls_side

        if "bar_ls_main" in row and not pd.isna(row["bar_ls_main"]):
            bar_ls_main = float(row["bar_ls_main"])
        elif "avg_kept_real_symbols" in row and not pd.isna(row["avg_kept_real_symbols"]):
            bar_ls_main = float(row["avg_kept_real_symbols"]) / 2.0
            new_row["bar_ls_main"] = bar_ls_main
        else:
            raise ValueError("Input CSV must contain either bar_ls_main or avg_kept_real_symbols")

        bar_ls_total = bar_ls_main + avg_ls_side
        new_row["bar_ls_total"] = bar_ls_total
        new_row["bar_ls_paper"] = bar_ls_total
        new_row["ls_side_formula"] = "l_b/log2(1+snr)"
        new_row["side_info_bits"] = float(height) * float(width) / 32.0
        rows.append(new_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Recompute rate-quality tables with l_s_side = l_b / log2(1 + snr).")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("logs/rate_quality_mcs_map_2026-04-22/all_rate_quality_mcs_map_2026-04-22.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/rate_quality_shannon_2026-04-22"),
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out_df = recompute_table(df, height=args.height, width=args.width)
    out_df = out_df.sort_values(["mu_comm", "comm_snr_db"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "all_rate_quality_shannon_2026-04-22.csv"
    json_path = args.output_dir / "all_rate_quality_shannon_2026-04-22.json"

    out_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(out_df.to_dict(orient="records"), f, indent=2)

    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
