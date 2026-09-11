#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DJI 电池能耗可视化（Plot-only v2.7，移除“累计能量”作图；仅使用 Clock:offsetTime）
----------------------------------------------------------------
功能要点：
- 只认 Clock:offsetTime 为时间轴（无则报错）
- 自动/手动识别：电压、电流、SoC（不包含高度）
- 单位鲁棒纠偏：mV/mA→V/A；NaN/异常稳健处理
- 功率定义：默认“放电为正”；可 --abs_power 或 --raw_power
- 平滑体系：mean/median/sg/exp；支持 --smooth_sec（按秒）与 --smooth（点数）
- 统计口径：同时输出“原始平均功率”和“平滑/下采样后平均功率”
- 电流统计：同时输出按功率口径的一致化平均电流和 |I| 平均
- 时间异常：检测 Clock:offsetTime 非单调段并做“分段展平”处理，同时警告
- 缺口限制：前向填充 ffill 对缺口长度设上限（按时间估计步数）
- 图像导出（PNG+PDF）：
  1) 双轴：功率&SoC
  2) 双轴：电压&电流
  3) 总览：两行子图（功率&SoC / 电压&电流）
  4) 单变量：Power / SoC / Voltage / Current
（已完全移除“累计能量”曲线及相关图片）
"""

import argparse
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import EngFormatter, ScalarFormatter

VERSION = "dji-energy v2.7 (plot-only, no-energy, offsetTime-only)"

# ---------- 可选 SciPy ----------
try:
    from scipy.signal import savgol_filter as _sg
    _HAVE_SG = True
except Exception:
    _HAVE_SG = False


# ---------- 字体：优先注册微软雅黑 ----------
def force_cjk_font() -> str:
    yahei_path = r"C:\\Windows\\Fonts\\msyh.ttc"
    if os.path.exists(yahei_path):
        try:
            font_manager.fontManager.addfont(yahei_path)
        except Exception:
            pass
    candidates = [
        "Microsoft YaHei", "Microsoft YaHei UI",
        "Yu Gothic UI", "Yu Gothic", "Meiryo",
        "SimHei", "SimSun",
        "Noto Sans CJK SC", "Noto Sans CJK JP",
        "PingFang SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42
            return name
    name = "DejaVu Sans"
    matplotlib.rcParams["font.family"] = name
    matplotlib.rcParams["font.sans-serif"] = [name]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    return name


# ---------- 多语言 ----------
STR = {
    "zh": {
        "time_s": "时间 (s)",
        "left_axis": "左轴曲线",
        "right_axis": "右轴曲线",
        "power": "功率 (W)",
        "soc": "荷电状态 SoC (%)",
        "volt": "电压 (V)",
        "curr": "电流 (A)",
        "title1": "功率 与 SoC 随时间变化",
        "title2": "电压 与 电流 随时间变化",
        "panel_title": "能耗可视化总览",
        "summary": "统计摘要",
        "avg_power": "平均功率 (W)",
        "avg_power_raw": "原始平均功率 (W)",
        "dur": "时长 (s)",
        "avg_v": "平均电压 (V)",
        "avg_i_signed": "平均电流(放电为正) (A)",
        "avg_i_abs": "平均电流(|I|) (A)",
        "picked": "选用列",
        "units": "单位修正倍率",
        "chosen_font": "使用字体",
        "note": "注",
        "legend_note": "平滑={smooth}  下采样={down}  功率={pdef}",
        "warn_sg": "未安装 SciPy，'sg' 平滑退回 'mean'",
        "warn_tmon": "发现 Clock:offsetTime 非单调，已分段展平，段数={nseg}",
        "warn_vsrc": "选到的电压可能不是电池端电压(95%分位≈{v95:.2f}V)，建议改用 battery_info:*Voltage。",
    },
    "en": {
        "time_s": "Time (s)",
        "left_axis": "Left-axis curves",
        "right_axis": "Right-axis curves",
        "power": "Power (W)",
        "soc": "State of Charge SoC (%)",
        "volt": "Voltage (V)",
        "curr": "Current (A)",
        "title1": "Power & SoC vs Time",
        "title2": "Voltage & Current vs Time",
        "panel_title": "Overview of Energy Analytics",
        "summary": "Summary",
        "avg_power": "Avg Power (W)",
        "avg_power_raw": "Avg Power (raw, W)",
        "dur": "Duration (s)",
        "avg_v": "Avg Voltage (V)",
        "avg_i_signed": "Avg Current(discharge-positive) (A)",
        "avg_i_abs": "Avg Current(|I|) (A)",
        "picked": "Picked columns",
        "units": "Unit scales",
        "chosen_font": "Chosen font",
        "note": "Note",
        "legend_note": "smooth={smooth}  down={down}  power={pdef}",
        "warn_sg": "SciPy not installed, 'sg' smoothing falls back to 'mean'",
        "warn_tmon": "Non-monotonic Clock:offsetTime detected; flattened by segments, n={nseg}",
        "warn_vsrc": "Voltage may not be battery-side (p95≈{v95:.2f}V). Prefer battery_info:*Voltage.",
    },
}


# ---------- 列名匹配 ----------
def _normalize_colname(c: str) -> str:
    return re.sub(r":[A-Z]$", "", c)


def _build_name_maps(columns):
    norm2orig = {}
    for c in columns:
        n = _normalize_colname(c)
        if n not in norm2orig or ("battery_info" in c and "battery_info" not in norm2orig[n]):
            norm2orig[n] = c
    return norm2orig


def _pick_first_existing(df, candidates):
    cols = set(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    norm2orig = _build_name_maps(cols)
    for cand in candidates:
        n = _normalize_colname(cand)
        if n in norm2orig:
            return norm2orig[n]
    patt = re.compile("|".join([re.escape(_normalize_colname(c)) for c in candidates]), re.I)
    for c in cols:
        if patt.search(_normalize_colname(c)):
            return c
    return None


def find_voltage_current_cols(df, force_v=None, force_i=None) -> Tuple[str, str]:
    if force_v and force_v in df.columns and force_i and force_i in df.columns:
        return force_v, force_i
    v_candidates = [
        "battery_info:BatVoltage",
        "battery_info:Voltage",
        "system_voltage:system_voltage",
    ]
    i_candidates = [
        "battery_info:BatCurrent",
        "battery_info:Current",
    ]
    vcol = _pick_first_existing(df, v_candidates)
    icol = _pick_first_existing(df, i_candidates)
    return vcol, icol


def find_soc_col(df, force_soc=None):
    if force_soc and force_soc in df.columns:
        return force_soc
    soc_candidates = [
        "battery_info:CapPercentage",
        "battery_info:sopPercentage",
        "battery_info:soc",
        "battery_info:SoC",
    ]
    return _pick_first_existing(df, soc_candidates)


# ---------- 时间轴：只接受 Clock:offsetTime；非单调段展平 ----------
def _to_seconds_like(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").ffill()
    xmax = float(np.nanmax(x))
    if xmax > 1e9:      # ns
        x = x / 1e9
    elif xmax > 1e6:    # ms
        x = x / 1e3
    return x


def extract_time_seconds(df, lang="zh"):
    if "Clock:offsetTime" not in df.columns:
        raise RuntimeError("缺少 Clock:offsetTime（本绘图脚本只接受该时间列）。请先用剪枝脚本生成 _slim.csv。")
    S = STR[lang]
    raw = _to_seconds_like(df["Clock:offsetTime"]).astype(float)
    # 分段展平：当出现回跳时，从该点起整体平移一个段偏移
    t = raw.to_numpy().copy()
    if len(t) == 0:
        return t
    offset = 0.0
    nseg = 1
    for k in range(1, len(t)):
        if not np.isfinite(t[k-1]) or not np.isfinite(t[k]):
            continue
        if t[k] + offset < t[k-1]:
            # 发生回跳：开启新段，段偏移 = (上一点 - 当前点) + 1e-6
            gap = (t[k-1] - (t[k] + offset)) + 1e-6
            offset += gap
            nseg += 1
        t[k] += offset
    # 起点归零
    t = t - t[0]
    if nseg > 1:
        print("[WARN]", S["warn_tmon"].format(nseg=nseg))
    return t


# ---------- 平滑 ----------
def _window_from_seconds(t: np.ndarray, smooth_sec: float) -> int:
    if smooth_sec is None or smooth_sec <= 0 or len(t) < 2:
        return 1
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 1
    med = float(np.median(dt))
    if med <= 0:
        return 1
    w = int(round(smooth_sec / med))
    return max(1, w)


def _rolling_mean(s: pd.Series, win: int) -> pd.Series:
    if win <= 1:
        return s
    return s.rolling(win, center=True, min_periods=1).mean()


def _rolling_median_then_mean(s: pd.Series, win: int) -> pd.Series:
    if win <= 1:
        return s
    med = s.rolling(win, center=True, min_periods=1).median()
    k = max(1, min(5, win // 3))
    return med.rolling(k, center=True, min_periods=1).mean()


def _sg_filter(y: np.ndarray, win: int, poly: int) -> np.ndarray:
    if not _HAVE_SG or win <= 2:
        return y
    if win % 2 == 0:
        win += 1
    win = max(win, poly + 2 + (poly % 2 == 0))
    win = min(win, len(y) - (1 - len(y) % 2)) if len(y) > 2 else win
    if win < 3:
        return y
    try:
        return _sg(y, window_length=win, polyorder=poly, mode="interp")
    except Exception:
        return y


def apply_smoothing(series: pd.Series,
                    win_pts: int,
                    kind: str,
                    sg_poly: int = 2,
                    exp_alpha: float = None,
                    exp_halflife: float = None) -> pd.Series:
    if series is None:
        return None
    if kind == "median":
        return _rolling_median_then_mean(series, win_pts)
    if kind == "sg":
        if not _HAVE_SG:
            print("[WARN]", STR["zh"]["warn_sg"])  # 用中文提示即可
            return _rolling_mean(series, win_pts)
        y = series.to_numpy(dtype=float)
        y = _sg_filter(y, max(3, win_pts), sg_poly)
        return pd.Series(y, index=series.index)
    if kind == "exp":
        if exp_halflife and exp_halflife > 0:
            return series.ewm(halflife=exp_halflife, adjust=False).mean()
        a = exp_alpha if exp_alpha is not None else 0.3
        a = min(max(a, 1e-3), 0.99)
        return series.ewm(alpha=a, adjust=False).mean()
    return _rolling_mean(series, win_pts)


# ---------- 单位鲁棒纠偏 ----------
def unit_fix(v: pd.Series, i: pd.Series):
    v_scale = i_scale = 1.0
    qv = np.nanquantile(np.abs(v), 0.95)
    qi = np.nanquantile(np.abs(i), 0.95)

    if np.isfinite(qv) and qv > 100:  # 可能是 mV
        v = v / 1000.0
        v_scale = 1 / 1000.0
    if np.isfinite(qi) and qi > 100:  # 可能是 mA
        i = i / 1000.0
        i_scale = 1 / 1000.0

    # 二次兜底
    if np.nanmean(np.abs(v)) > 100:
        v = v / 1000.0
        v_scale *= 1 / 1000.0
    if np.nanmean(np.abs(i)) > 100:
        i = i / 1000.0
        i_scale *= 1 / 1000.0

    return v, i, v_scale, i_scale


# ---------- 公共绘图 ----------
def beautify(ax):
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))


def note_box(ax, text: str):
    ax.text(0.99, 0.02, text, transform=ax.transAxes,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
            fontsize=9)


# ---------- 缺口受限的前向填充 ----------
def ffill_with_limit(s: pd.Series, t: np.ndarray, seconds_limit: float = 2.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if len(t) < 2 or not np.isfinite(t).any():
        return s.ffill()
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return s.ffill()
    med = float(np.median(dt))
    if med <= 0:
        return s.ffill()
    max_steps = max(1, int(round(seconds_limit / med)))
    return s.ffill(limit=max_steps)


# ---------- 主流程 ----------
def run(csv_path: str, outdir: str, dpi: int, lang: str, show: bool,
        vcol_force: str, icol_force: str, soc_force: str,
        smooth_win: int, smooth_sec: float, smooth_kind: str,
        exp_alpha: float, exp_halflife: float, sg_poly: int,
        down_step: int, clip_q: float,
        abs_power: bool, raw_power: bool,
        no_panel: bool, no_pairs: bool, no_singles: bool):
    S = STR[lang]
    font_name = force_cjk_font()
    print(f"[{VERSION}]")
    print(f"{S['chosen_font']}: {font_name}")

    # 读取 CSV（稳健）
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False, on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(csv_path, dtype=str, low_memory=False, on_bad_lines="skip", encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(csv_path, dtype=str, low_memory=False, on_bad_lines="skip", encoding="latin1")

    # 选择列
    vcol, icol = find_voltage_current_cols(df, vcol_force, icol_force)
    soccol = find_soc_col(df, soc_force)

    if not vcol or not icol:
        bat_cols = [c for c in df.columns if "battery" in c.lower() or "system_voltage" in c.lower()]
        msg = "未找到电压/电流列；请用 --vcol/--icol 指定 \n示例：\n" \
              "  --vcol \"battery_info:BatVoltage:D\" --icol \"battery_info:BatCurrent:D\"\n\n" \
              "可参考电池相关列（节选）:\n  - " + "\n  - ".join(bat_cols[:40])
        raise RuntimeError(msg)

    # 时间轴（仅 offsetTime；非单调展平）
    t = extract_time_seconds(df, lang=lang).astype(float)

    # 数值化 & 缺口受限 ffill
    v = ffill_with_limit(df[vcol], t, seconds_limit=2.0)
    i = ffill_with_limit(df[icol], t, seconds_limit=2.0)
    soc = ffill_with_limit(df[soccol], t, seconds_limit=2.0) if soccol else None

    # SoC 归一化到 0–100
    if soc is not None:
        qmax = np.nanquantile(soc, 0.98)
        if np.isfinite(qmax):
            if qmax <= 1.5:
                soc = soc * 100.0
            soc = soc.clip(lower=0, upper=100)

    # 单位纠偏
    v, i, v_scale, i_scale = unit_fix(v, i)

    # 电压来源合理性提醒
    if vcol == "system_voltage:system_voltage":
        v95 = np.nanquantile(v, 0.95)
        if np.isfinite(v95) and not (9.0 <= v95 <= 30.0):
            print("[WARN]", S["warn_vsrc"].format(v95=v95))

    # 功率定义
    if abs_power:
        p = np.abs(v * i)
        p_def = "abs(V*I)"
        i_eff = i.copy()
    elif raw_power:
        p = v * i
        p_def = "raw V*I"
        i_eff = i.copy()
    else:
        sign = -1.0 if np.nanmedian(i) < 0 else 1.0
        p = (sign * v * i)
        p_def = "discharge-positive"
        i_eff = sign * i

    # 平滑窗口：秒优先
    win_pts = smooth_win if (smooth_sec is None or smooth_sec <= 0) else _window_from_seconds(t, smooth_sec)
    win_pts = max(1, int(win_pts))

    # 平滑（先平滑再下采样）
    v_s = apply_smoothing(v, win_pts, smooth_kind, sg_poly, exp_alpha, exp_halflife)
    i_s = apply_smoothing(i, win_pts, smooth_kind, sg_poly, exp_alpha, exp_halflife)
    p_s = apply_smoothing(pd.Series(p, index=v.index), win_pts, smooth_kind, sg_poly, exp_alpha, exp_halflife)
    soc_s = apply_smoothing(soc, win_pts, smooth_kind, sg_poly, exp_alpha, exp_halflife) if soc is not None else None

    # 同步下采样
    step = max(1, int(down_step))
    sl = slice(0, len(t), step)
    t = t[sl]
    v_s = v_s.to_numpy()[sl]
    i_s = i_s.to_numpy()[sl]
    p_s = p_s.to_numpy()[sl]
    if soc_s is not None:
        soc_s = soc_s.to_numpy()[sl]

    # 统计（功率提供原始与平滑两口径；平滑口径可选剪裁）
    if 0 < clip_q < 0.5 and np.isfinite(p_s).any():
        lo, hi = np.nanquantile(p_s, [clip_q, 1 - clip_q])
        p_for_stat = np.clip(p_s, lo, hi)
    else:
        p_for_stat = p_s

    avg_power_raw = float(np.nanmean(v.to_numpy() * i.to_numpy())) if (len(v) and len(i)) else np.nan
    avg_power = float(np.nanmean(p_for_stat))
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    avg_v = float(np.nanmean(v_s))
    avg_i_signed = float(np.nanmean(i_eff))
    avg_i_abs = float(np.nanmean(np.abs(i_s)))

    # 输出目录
    csvp = Path(csv_path).resolve()
    default_out = csvp.parent / "figure"   # 默认输出到 CSV 同目录下 figure
    outdir = Path(outdir) if outdir else default_out
    outdir.mkdir(parents=True, exist_ok=True)
    stem = csvp.stem

    # 文件名
    f1_png = outdir / f"{stem}_power_soc.png"
    f1_pdf = outdir / f"{stem}_power_soc.pdf"
    f2_png = outdir / f"{stem}_volt_curr.png"
    f2_pdf = outdir / f"{stem}_volt_curr.pdf"
    f_over_png = outdir / f"{stem}_overview.png"
    f_over_pdf = outdir / f"{stem}_overview.pdf"

    # 绘图公共样式
    eng = EngFormatter(unit="")
    if smooth_kind != "exp":
        smooth_label = f"{smooth_kind}({win_pts})"
    else:
        smooth_label = f"exp(alpha={exp_alpha})" if (exp_alpha is not None) else f"exp(halflife={exp_halflife})"
    note_txt = S["legend_note"].format(smooth=smooth_label, down=down_step, pdef=p_def)

    out_files = []

    # ---------- 双轴：功率 & SoC ----------
    if not no_pairs:
        fig1, ax1 = plt.subplots(figsize=(10.6, 5.6))
        l1, = ax1.plot(t, p_s, linewidth=1.9, label=S["power"])
        ax1.set_xlabel(S["time_s"])
        ax1.set_ylabel(S["power"], color=l1.get_color())
        ax1.tick_params(axis='y', labelcolor=l1.get_color())
        ax1.yaxis.set_major_formatter(eng)
        beautify(ax1)
        if soc_s is not None:
            ax1r = ax1.twinx()
            l2, = ax1r.plot(t, soc_s, linewidth=1.6, linestyle="--", label=S["soc"])
            ax1r.set_ylabel(S["soc"], color=l2.get_color())
            ax1r.tick_params(axis='y', labelcolor=l2.get_color())
            beautify(ax1r)
            ax1r.set_ylim(0, 100)
            leg1 = ax1.legend([l1], [l1.get_label()], loc="upper left", frameon=True, title=S["left_axis"])
            leg2 = ax1r.legend([l2], [l2.get_label()], loc="upper right", frameon=True, title=S["right_axis"])
            ax1.add_artist(leg1)
        else:
            ax1.legend(loc="upper left", frameon=True, title=S["left_axis"])
        note_box(ax1, f"{S['note']}: {note_txt}")
        ax1.set_title(S["title1"])
        fig1.tight_layout()
        fig1.savefig(f1_png, dpi=dpi, bbox_inches="tight")
        fig1.savefig(f1_pdf, dpi=dpi, bbox_inches="tight")
        plt.close(fig1)
        out_files += [f1_png, f1_pdf]

    # ---------- 双轴：电压 & 电流 ----------
    if not no_pairs:
        fig2, ax2 = plt.subplots(figsize=(10.6, 5.6))
        lv, = ax2.plot(t, v_s, linewidth=1.9, label=S["volt"])
        ax2.set_xlabel(S["time_s"])
        ax2.set_ylabel(S["volt"], color=lv.get_color())
        ax2.tick_params(axis='y', labelcolor=lv.get_color())
        beautify(ax2)
        ax2r = ax2.twinx()
        li, = ax2r.plot(t, i_s, linewidth=1.6, linestyle="--", label=S["curr"])
        ax2r.set_ylabel(S["curr"], color=li.get_color())
        ax2r.tick_params(axis='y', labelcolor=li.get_color())
        beautify(ax2r)
        leg1 = ax2.legend([lv], [lv.get_label()], loc="upper left", frameon=True, title=S["left_axis"])
        leg2 = ax2r.legend([li], [li.get_label()], loc="upper right", frameon=True, title=S["right_axis"])
        ax2.add_artist(leg1)
        note_box(ax2, f"{S['note']}: {note_txt}")
        ax2.set_title(S["title2"])
        fig2.tight_layout()
        fig2.savefig(f2_png, dpi=dpi, bbox_inches="tight")
        fig2.savefig(f2_pdf, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        out_files += [f2_png, f2_pdf]

    # ---------- 总览面板（两行） ----------
    if not no_panel:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11.4, 8.6), sharex=True)

        # 1) 功率 & SoC
        ax = axes[0]
        l1, = ax.plot(t, p_s, linewidth=1.8, label=S["power"])
        ax.set_ylabel(S["power"], color=l1.get_color())
        ax.tick_params(axis='y', labelcolor=l1.get_color())
        beautify(ax)
        legL0 = ax.legend([l1], [l1.get_label()], loc="upper left", frameon=True, title=S["left_axis"])
        ax.add_artist(legL0)
        if soc_s is not None:
            axr = ax.twinx()
            l2, = axr.plot(t, soc_s, linewidth=1.4, linestyle="--", label=S["soc"])
            axr.set_ylabel(S["soc"], color=l2.get_color())
            axr.tick_params(axis='y', labelcolor=l2.get_color())
            beautify(axr)
            axr.set_ylim(0, 100)
            axr.legend([l2], [l2.get_label()], loc="upper right", frameon=True, title=S["right_axis"])
        ax.set_title(S["title1"])

        # 2) 电压 & 电流
        ax = axes[1]
        lv, = ax.plot(t, v_s, linewidth=1.8, label=S["volt"])
        ax.set_xlabel(S["time_s"])
        ax.set_ylabel(S["volt"], color=lv.get_color())
        ax.tick_params(axis='y', labelcolor=lv.get_color())
        beautify(ax)
        ax.legend([lv], [lv.get_label()], loc="upper left", frameon=True, title=S["left_axis"])
        axr = ax.twinx()
        li, = axr.plot(t, i_s, linewidth=1.4, linestyle="--", label=S["curr"])
        axr.set_ylabel(S["curr"], color=li.get_color())
        axr.tick_params(axis='y', labelcolor=li.get_color())
        beautify(axr)
        axr.legend([li], [li.get_label()], loc="upper right", frameon=True, title=S["right_axis"])
        ax.set_title(S["title2"])

        # 顶部标题 + 右上角摘要框（已移除总能量）
        fig.suptitle(S["panel_title"], fontsize=14, y=0.99)
        summary = (f"{S['avg_power_raw']}: {avg_power_raw:,.2f} W\n"
                   f"{S['avg_power']}: {avg_power:,.2f} W\n"
                   f"{S['dur']}: {duration:,.2f} s\n"
                   f"{S['avg_v']}: {avg_v:,.2f} V\n"
                   f"{S['avg_i_signed']}: {avg_i_signed:,.2f} A\n"
                   f"{S['avg_i_abs']}: {avg_i_abs:,.2f} A")
        axes[0].text(0.995, 0.02, summary, transform=axes[0].transAxes,
                     ha="right", va="bottom",
                     bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
                     fontsize=10)

        axes[1].text(0.005, 0.02, f"{S['note']}: {note_txt}", transform=axes[1].transAxes,
                     ha="left", va="bottom",
                     bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
                     fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(f_over_png, dpi=dpi, bbox_inches="tight")
        fig.savefig(f_over_pdf, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        out_files += [f_over_png, f_over_pdf]

    # ---------- 单变量图（不含能量） ----------
    if not no_singles:
        def single_plot(x, y, y_label, title, png_path, pdf_path):
            fig, ax = plt.subplots(figsize=(10.6, 5.0))
            (ln,) = ax.plot(x, y, linewidth=1.9, label=y_label)
            ax.set_xlabel(S["time_s"])
            ax.set_ylabel(y_label)
            beautify(ax)
            ax.legend(loc="upper left", frameon=True)
            note_box(ax, f"{S['note']}: {note_txt}")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        # Power
        single_plot(t, p_s, S["power"], S["power"], outdir / f"{stem}_pwr.png", outdir / f"{stem}_pwr.pdf")
        # SoC（可选）
        if soc_s is not None:
            single_plot(t, soc_s, S["soc"], S["soc"], outdir / f"{stem}_soc.png", outdir / f"{stem}_soc.pdf")
        # Voltage
        single_plot(t, v_s, S["volt"], S["volt"], outdir / f"{stem}_volt.png", outdir / f"{stem}_volt.pdf")
        # Current
        single_plot(t, i_s, S["curr"], S["curr"], outdir / f"{stem}_curr.png", outdir / f"{stem}_curr.pdf")

        out_files += [
            outdir / f"{stem}_pwr.png", outdir / f"{stem}_pwr.pdf",
            (outdir / f"{stem}_soc.png") if soc_s is not None else None,
            (outdir / f"{stem}_soc.pdf") if soc_s is not None else None,
            outdir / f"{stem}_volt.png", outdir / f"{stem}_volt.pdf",
            outdir / f"{stem}_curr.png", outdir / f"{stem}_curr.pdf",
        ]

    # ---------- 控制台摘要 ----------
    print(f"{S['picked']}: V=<{vcol}>, I=<{icol}>" + (f", SoC=<{soccol}>" if soccol else ""))
    print(f"{S['units']}: V_scale={v_scale:g}, I_scale={i_scale:g}")
    print(f" - {S['avg_power_raw']}: {avg_power_raw:,.3f} W")
    print(f" - {S['avg_power']}: {avg_power:,.3f} W")
    print(f" - {S['dur']}: {duration:,.3f} s")
    print(f" - {S['avg_v']}: {avg_v:,.3f} V")
    print(f" - {S['avg_i_signed']}: {avg_i_signed:,.3f} A")
    print(f" - {S['avg_i_abs']}: {avg_i_abs:,.3f} A")
    print("导出文件:")
    for f in out_files:
        if f is not None:
            print(" -", f)

    if show:
        plt.show()
    plt.close('all')


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="DJI 电池能耗可视化（Plot-only，无累计能量作图，offsetTime-only）")
    parser.add_argument("--csv", required=True, help="输入 CSV 路径（建议使用剪枝后的 *_slim.csv）")
    parser.add_argument("--out", default="", help="输出目录（默认写到 CSV 同目录下的 figure/）")
    parser.add_argument("--dpi", type=int, default=200, help="导出 DPI（默认200）")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="界面语言")
    parser.add_argument("--show", action="store_true", help="绘制完成后显示窗口")

    # 列名强制指定（可省略，自动匹配）
    parser.add_argument("--vcol", type=str, default=None, help="手动指定电压列名")
    parser.add_argument("--icol", type=str, default=None, help="手动指定电流列名")
    parser.add_argument("--soccol", type=str, default=None, help="手动指定 SoC 列名")

    # 平滑控制
    parser.add_argument("--smooth_kind", choices=["mean", "median", "sg", "exp"], default="mean",
                        help="平滑类型（默认 mean）")
    parser.add_argument("--smooth", type=int, default=1, help="滚动平滑窗口（点数，=1表示关闭）")
    parser.add_argument("--smooth_sec", type=float, default=0.0, help="按秒指定平滑窗口（优先于 --smooth）")
    parser.add_argument("--sg_poly", type=int, default=2, help="Savitzky–Golay 的多项式阶数（默认2）")
    parser.add_argument("--exp_alpha", type=float, default=None, help="EWMA 的 alpha（0~1，优先于 halflife）")
    parser.add_argument("--exp_halflife", type=float, default=None, help="EWMA 的半衰期（秒概念）")

    # 采样与统计
    parser.add_argument("--down", type=int, default=1, help="下采样步长（默认1=不下采样）")
    parser.add_argument("--clipq", type=float, default=0.01, help="统计功率剪裁分位（0~0.5，默认0.01）")

    # 功率口径
    parser.add_argument("--abs_power", action="store_true", help="功率=|V*I|")
    parser.add_argument("--raw_power", action="store_true", help="功率=V*I（不改号）")

    # 图像开关
    parser.add_argument("--no_panel", action="store_true", help="不导出总览大图（默认会导出）")
    parser.add_argument("--no_pairs", action="store_true", help="不导出两张双轴图（默认会导出）")
    parser.add_argument("--no_singles", action="store_true", help="不导出单变量图（默认会导出）")

    args = parser.parse_args()
    run(args.csv, args.out, args.dpi, args.lang, args.show,
        args.vcol, args.icol, args.soccol,
        args.smooth, args.smooth_sec, args.smooth_kind,
        args.exp_alpha, args.exp_halflife, args.sg_poly,
        args.down, args.clipq,
        args.abs_power, args.raw_power,
        args.no_panel, args.no_pairs, args.no_singles)


if __name__ == "__main__":
    main()
