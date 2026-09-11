"""
可视化工具 v3.8 (CSV 直读 + 标准 kbps 对齐):
1. SC 独立图 (SC Only)
2. TC 独立图 (TC Only)
3. SC + TC 对比图 (Combined)

[逻辑说明]:
- SC 码率: 直接读取 CSV 中的 'coding_rate_target_fps_kbps'，该字段已按标准定义
  1 kbps = 1000 bits/s 计算（在 realtime_demo_all.py 中完成）。
- TC 码率: 直接读取 measure_tc.py 输出的 'bitrate_kbps'（或按 target_fps 重算）。
- 不再进行 1024/1000 的单位修正，两侧完全在同一 kbps 定义下对齐。

前提：
- 必须先运行修改后的 realtime_demo_all.py 生成包含 'coding_rate_target_fps_kbps' 的 CSV。
- 必须先运行 measure_tc.py 生成 TC 的 CSV。

运行指令示例:
python visualize_benchmark.py --result_root realtime_result/ocean/285W --tc_csv results_tc/tc_h264_ocean.csv --out_dir figs_sc_tc --lpips_thr 0.35 --fps_thr 20

"""

import os
import glob
import argparse
import subprocess
import sys
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== 尝试导入 adjustText ==================
try:
    from adjustText import adjust_text
except ImportError:
    print("[Info] 'adjustText' 库未找到，正在自动安装...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "adjustText"])
        from adjustText import adjust_text
    except Exception as e:
        print(f"[Error] 无法安装 'adjustText'。请手动运行 pip install adjustText。")
        exit(1)

# ================== 全局样式 ==================
import matplotlib

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.linewidth': 1.0,
    'scatter.edgecolors': 'black',
})


# ================== 数据处理函数 ==================
def merge_from_result_root(result_root: str) -> pd.DataFrame:
    pattern_root = os.path.join(result_root, "benchmark_summary.csv")
    csv_paths = glob.glob(pattern_root)
    if not csv_paths:
        pattern_sub = os.path.join(result_root, "*", "benchmark_summary.csv")
        csv_paths = glob.glob(pattern_sub)
    if not csv_paths:
        raise FileNotFoundError(f"在 {result_root} 下没有找到任何 benchmark_summary.csv")
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "power_tag" not in df.columns:
            power = os.path.basename(os.path.dirname(p))
            df["power_tag"] = power
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def ensure_rank_interval(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rank" not in df.columns or "interval" not in df.columns:
        import re
        ranks, intervals = [], []
        for _, row in df.iterrows():
            source = str(row.get("prompt_dir", row.get("config", "")))
            m = re.search(r"rank(\d+)_interval(\d+)", source)
            if m:
                ranks.append(int(m.group(1)))
                intervals.append(int(m.group(2)))
            else:
                ranks.append(-1)
                intervals.append(-1)
        df["rank"] = ranks
        df["interval"] = intervals
    return df


def add_config_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "power_tag" in df.columns:
        df["config_label"] = df.apply(lambda x: f"{x['power_tag']}_r{x['rank']}_i{x['interval']}", axis=1)
    else:
        df["config_label"] = df.apply(lambda x: f"r{x['rank']}_i{x['interval']}", axis=1)
    return df


def load_tc_csv(tc_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(tc_csv_path):
        raise FileNotFoundError(f"找不到 TC CSV 文件：{tc_csv_path}")
    return pd.read_csv(tc_csv_path)


# ================== 核心计算逻辑 (直读标准 kbps) ==================

def compute_coding_rate_kbps_sc(df: pd.DataFrame, target_fps: Optional[float] = None) -> np.ndarray:
    """
    SC 侧码率（kbps）计算：

    - 在 realtime_demo_all.py 中，'coding_rate_target_fps_kbps' 已按标准定义计算：
        bits_per_frame = total_theoretical_bytes * 8 / total_interval_frames
        kbps = bits_per_frame * target_fps / 1000

      因此该字段已经是“1 kbps = 1000 bits/s”的标准网络码率。
    - 这里直接读取该列，不再做任何 1024/1000 的单位修正。
    """
    if "coding_rate_target_fps_kbps" not in df.columns:
        raise ValueError("CSV 中缺少 'coding_rate_target_fps_kbps' 列。请先运行最新版 realtime_demo_all.py。")

    return df["coding_rate_target_fps_kbps"].to_numpy(dtype=float)


def compute_coding_rate_kbps_tc(df_tc: pd.DataFrame, target_fps: Optional[float] = None) -> np.ndarray:
    """
    TC 侧码率（kbps）计算：

    - 若 target_fps is None，则直接返回 measure_tc.py 里计算的 'bitrate_kbps'：
        bitrate_kbps = (total_bits / duration_sec) / 1000

    - 若指定 target_fps，则先用 'payload_bits_tc' 推出每帧 bit 数，
      再按 target_fps 重算 kbps：
        kbps = (payload_bits_tc * target_fps) / 1000
    """
    if target_fps is None:
        return df_tc["bitrate_kbps"].to_numpy(dtype=float)

    target_fps = float(target_fps)
    if "payload_bits_tc" in df_tc.columns:
        bits_per_frame = df_tc["payload_bits_tc"].to_numpy(dtype=float)
    else:
        # Fallback: recalculate bits per frame from bitrate
        bitrate = df_tc["bitrate_kbps"].to_numpy(dtype=float)
        fps_video = df_tc["fps_video"].to_numpy(dtype=float)
        bits_per_frame = bitrate * 1000.0 / fps_video

    # TC 已经是标准 kbps (bits * fps / 1000)
    return bits_per_frame * target_fps / 1000.0


# ================== 绘图引擎 ==================

def draw_rd_curve(
        df_sc: Optional[pd.DataFrame],
        df_tc: Optional[pd.DataFrame],
        save_path: str,
        lpips_thr: float,
        fps_thr: float,
        target_fps: Optional[float] = None
):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    all_texts = []
    all_lpips_for_limit = []
    export_data: List[Dict] = []

    # ----------- 1. 处理 TC (H.264) -----------
    if df_tc is not None:
        sub_tc = df_tc.dropna(subset=["lpips_tc"]).copy()
        lpips_tc = sub_tc["lpips_tc"].values
        rate_tc = compute_coding_rate_kbps_tc(sub_tc, target_fps=target_fps)

        if target_fps is None:
            fps_tc_check = sub_tc["fps_video"].values if "fps_video" in sub_tc.columns else np.full_like(lpips_tc, 20.0)
        else:
            fps_tc_check = np.full_like(lpips_tc, float(target_fps))

        good_tc = (lpips_tc <= lpips_thr) & (fps_tc_check >= fps_thr)
        bad_tc = ~good_tc

        # Label 生成
        if "target_bitrate_kbps" in sub_tc.columns:
            labels_tc = [f"{int(b)}k" for b in sub_tc["target_bitrate_kbps"]]
        else:
            labels_tc = [f"Idx{i}" for i in range(len(sub_tc))]

        # 收集数据
        for i in range(len(rate_tc)):
            export_data.append({
                "Model": "TC",
                "Power": "N/A",
                "Label": labels_tc[i],
                "CodingRate_kbps": rate_tc[i],
                "LPIPS": lpips_tc[i],
                "FPS": fps_tc_check[i],
                "Pass": bool(good_tc[i])
            })

        # 绘图
        if np.any(bad_tc):
            ax.scatter(rate_tc[bad_tc], lpips_tc[bad_tc], marker="X", c="#e0e0e0", s=40, alpha=0.5, zorder=1)

        sort_idx = np.argsort(rate_tc)
        ax.plot(rate_tc[sort_idx], lpips_tc[sort_idx], color='tab:orange', linestyle='--', linewidth=1.5, alpha=0.5,
                zorder=2, label='H.264 Trend')

        if np.any(good_tc):
            ax.scatter(rate_tc[good_tc], lpips_tc[good_tc], c="tab:orange", marker="s", s=90, edgecolors='white',
                       linewidth=1.2, zorder=3, label='TC (H.264)')
            indices = np.where(good_tc)[0]
            for i in indices:
                t = ax.text(rate_tc[i], lpips_tc[i], labels_tc[i], fontsize=9, color='#d35400', fontweight='bold')
                all_texts.append(t)

        all_lpips_for_limit.extend(lpips_tc)

    # ----------- 2. 处理 SC (Promptus) -----------
    if df_sc is not None:
        sub_sc = df_sc.dropna(subset=["lpips", "fps"]).copy()
        sub_sc = add_config_label(ensure_rank_interval(sub_sc))

        POWER_COLORS = {
            "120W": "#2ca02c",  # Green
            "200W": "#1f77b4",  # Blue
            "280W": "#9467bd",  # Purple
            "285W": "#9467bd",
        }
        DEFAULT_COLOR = "#7f7f7f"

        unique_powers = sorted(sub_sc["power_tag"].unique())

        for p_tag in unique_powers:
            mask_p = (sub_sc["power_tag"] == p_tag)
            df_p = sub_sc[mask_p].copy()

            lpips_p = df_p["lpips"].values
            fps_p = df_p["fps"].values

            # [核心]：直接读取标准 kbps
            rate_p = compute_coding_rate_kbps_sc(df_p, target_fps=target_fps)
            labels_p = df_p["config_label"].tolist()

            good_p = (lpips_p <= lpips_thr) & (fps_p >= fps_thr)
            bad_p = ~good_p

            # 收集数据
            for i in range(len(rate_p)):
                export_data.append({
                    "Model": "SC",
                    "Power": p_tag,
                    "Label": labels_p[i],
                    "CodingRate_kbps": rate_p[i],
                    "LPIPS": lpips_p[i],
                    "FPS": fps_p[i],
                    "Pass": bool(good_p[i])
                })

            # 绘图
            if np.any(bad_p):
                ax.scatter(rate_p[bad_p], lpips_p[bad_p], c="#e0e0e0", s=40, alpha=0.5, edgecolors='none', zorder=1)

            if np.any(good_p):
                color = POWER_COLORS.get(p_tag, DEFAULT_COLOR)
                label_legend = f"SC ({p_tag})"

                ax.scatter(rate_p[good_p], lpips_p[good_p], c=color, marker="o", s=110, edgecolors='white',
                           linewidth=1.2, zorder=4, label=label_legend)

                indices = np.where(good_p)[0]
                for i in indices:
                    clean_label = labels_p[i].replace(f"{p_tag}_", "")
                    t = ax.text(rate_p[i], lpips_p[i], clean_label, fontsize=8, color=color, fontweight='bold',
                                alpha=0.9)
                    all_texts.append(t)

            all_lpips_for_limit.extend(lpips_p)

    # ----------- 3. 装饰与保存图片 -----------
    fps_info = f"@{target_fps}FPS" if target_fps else "(Dynamic FPS)"
    ax.set_title(
        f"Performance Comparison {fps_info}\n(Threshold: LPIPS≤{lpips_thr}, FPS≥{fps_thr})",
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel("Coding Rate (kbps)", fontweight='bold')
    ax.set_ylabel("LPIPS (Lower is better)", fontweight='bold')

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right', framealpha=0.9, fancybox=True, edgecolor='gray', title="Configuration")

    if all_lpips_for_limit:
        y_min, y_max = min(all_lpips_for_limit), max(all_lpips_for_limit)
        margin = (y_max - y_min) * 0.15
        ax.set_ylim(max(0, y_min - margin), y_max + margin)

    if all_texts:
        print(f"[Processing] Optimizing labels for {os.path.basename(save_path)}...")
        adjust_text(
            all_texts,
            ax=ax,
            force_text=(0.5, 1.0),
            force_points=(0.2, 0.5),
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Saved Image] {save_path}")

    # ----------- 4. 保存 CSV 数据 -----------
    if export_data:
        csv_path = os.path.splitext(save_path)[0] + ".csv"
        df_export = pd.DataFrame(export_data)

        cols_order = ["Model", "Power", "Label", "CodingRate_kbps", "LPIPS", "FPS", "Pass"]
        for c in cols_order:
            if c not in df_export.columns:
                df_export[c] = None
        df_export = df_export[cols_order]

        df_export.to_csv(csv_path, index=False, float_format="%.4f", encoding="utf-8-sig")
        print(f"[Saved Data]  {csv_path}")


# ================== Main ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--result_root", type=str, default="realtime_result")
    parser.add_argument("--auto_merge", action="store_true")
    parser.add_argument("--tc_csv", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="figs_v3_8_final")
    parser.add_argument("--lpips_thr", type=float, default=0.35)
    parser.add_argument("--fps_thr", type=float, default=20.0)
    parser.add_argument("--target_fps", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载 SC 数据
    df_sc = None
    try:
        if args.auto_merge:
            df_sc = merge_from_result_root(args.result_root)
        elif args.csv:
            df_sc = pd.read_csv(args.csv)
        else:
            # 默认尝试
            try:
                df_sc = merge_from_result_root(args.result_root)
            except Exception:
                pass

        if df_sc is not None:
            print(f"[Info] Original SC data: {len(df_sc)} rows")

    except Exception as e:
        print(f"[Warning] SC Data Load Failed: {e}")

    # 2. 加载 TC 数据
    df_tc = None
    if args.tc_csv and os.path.exists(args.tc_csv):
        df_tc = load_tc_csv(args.tc_csv)
        print(f"[Info] Loaded TC data: {len(df_tc)} rows")
    else:
        print("[Warning] No TC csv provided or not found.")

    # ================== 执行绘图 ==================

    if df_sc is not None and not df_sc.empty:
        draw_rd_curve(
            df_sc=df_sc,
            df_tc=None,
            save_path=os.path.join(args.out_dir, "plot_sc_colored.png"),
            lpips_thr=args.lpips_thr,
            fps_thr=args.fps_thr,
            target_fps=args.target_fps
        )

    if df_tc is not None:
        draw_rd_curve(
            df_sc=None,
            df_tc=df_tc,
            save_path=os.path.join(args.out_dir, "plot_tc_only.png"),
            lpips_thr=args.lpips_thr,
            fps_thr=args.fps_thr,
            target_fps=args.target_fps
        )

    if df_sc is not None and not df_sc.empty and df_tc is not None:
        draw_rd_curve(
            df_sc=df_sc,
            df_tc=df_tc,
            save_path=os.path.join(args.out_dir, "plot_combined_colored.png"),
            lpips_thr=args.lpips_thr,
            fps_thr=args.fps_thr,
            target_fps=args.target_fps
        )


if __name__ == "__main__":
    main()
