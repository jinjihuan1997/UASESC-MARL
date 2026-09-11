#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Monitor (Window Power + Full-session Power)
- 打开本地 CSV（HWiNFO / nvidia-smi）
- Tab1：功率 (W) —— 窗口内 CPU / GPU / Total
- Tab2：全程功率 (W，0→现在) —— 会话起点到当前，独立于窗口
- 支持 EMA 平滑、暂停/继续、保存截图（功率/全程功率）、导出窗口指标（含能量估计列，可选）

运行：
  pip install pandas matplotlib numpy
  python energy_monitor_tk_plus_power.py
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

APP_TITLE = "Energy Monitor (Window Power + Full-session Power)"
DEFAULT_WINDOW_SEC = 60
DEFAULT_REFRESH_MS = 200
CONFIG_PATH = Path.home() / ".energy_monitor_tk.json"

# 全程功率曲线抽样上限，避免极长会话卡顿
MAX_POINTS_FULL = 20000

# ------------------- CSV utils -------------------

CAND_SEP = [",", ";", "\t", "|"]
CAND_DEC = [".", ","]


def detect_sep_and_decimal(sample_text: str) -> Tuple[str, str]:
    header_line = sample_text.splitlines()[0] if sample_text else ""
    if ";" in header_line:
        lines = sample_text.splitlines()[1:5]
        joined = " ".join(lines)
        dec = "," if ("," in joined and "." not in joined) else "."
        return ";", dec
    for sep in CAND_SEP:
        try:
            df0 = pd.read_csv(io.StringIO(sample_text), nrows=0, sep=sep, engine="python")
            if df0.shape[1] > 1:
                lines = sample_text.splitlines()[1:5]
                joined = " ".join(lines)
                dec = "."
                if "," in joined and (joined.count(",") > joined.count(".")):
                    dec = ","
                return sep, dec
        except Exception:
            continue
    return ",", "."


def read_header_and_formats(path: Path) -> Tuple[Optional[List[str]], str, str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join([next(f) for _ in range(5)])
    except Exception:
        try:
            with open(path, "r", errors="ignore") as f:
                head = "".join([next(f) for _ in range(5)])
        except Exception:
            return None, ",", "."
    sep, dec = detect_sep_and_decimal(head)
    try:
        df0 = pd.read_csv(io.StringIO(head), nrows=0, sep=sep, engine="python")
        cols = list(df0.columns)
        if len(cols) <= 1:
            df0 = pd.read_csv(io.StringIO(head), nrows=0, sep=",", engine="python")
            cols = list(df0.columns)
    except Exception:
        cols = None
    return cols, sep, dec


def parse_time_columns(df: pd.DataFrame) -> Optional[pd.Series]:
    ts = None
    if "Date" in df.columns and "Time" in df.columns:
        combo = (df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip())
        tried = [
            ("%d.%m.%Y %H:%M:%S.%f", True), ("%d.%m.%Y %H:%M:%S", True),
            ("%Y-%m-%d %H:%M:%S.%f", None), ("%Y-%m-%d %H:%M:%S", None),
            ("%m/%d/%Y %H:%M:%S.%f", False), ("%m/%d/%Y %H:%M:%S", False),
            (None, True), (None, False)
        ]
        for fmt, dayfirst in tried:
            try:
                ts = pd.to_datetime(combo, format=fmt, dayfirst=dayfirst, errors="coerce")
            except Exception:
                ts = None
            if ts is not None and ts.notna().any():
                break
    elif "Time" in df.columns:
        ts = pd.to_datetime(df["Time"].astype(str), errors="coerce")
        if ts.isna().all():
            for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
                ts = pd.to_datetime(df["Time"].astype(str), format=fmt, errors="coerce")
                if ts.notna().any():
                    break
    if ts is None or ts.dropna().empty:
        return None
    ts = ts.dropna()
    t0 = ts.iloc[0]
    return (ts - t0).dt.total_seconds().rename("t_sec")


def tail_bytes(path: Path, approx_lines: int = 1000, chunk_size: int = 65536) -> bytes:
    with open(path, "rb") as f:
        f.seek(0,  os.SEEK_END)
        data = b""
        nl = 0
        pos = f.tell()
        while pos > 0 and nl < approx_lines:
            read_size = chunk_size if pos - chunk_size > 0 else pos
            pos -= read_size
            f.seek(pos, os.SEEK_SET)
            chunk = f.read(read_size)
            data = chunk + data
            nl += chunk.count(b"\n")
        return data


def pick_default_cols(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    cpu_col = None
    gpu_col = None
    cpu_keys = ["cpu package power", "processor power", "pkg power"]
    gpu_keys = ["gpu power", "total board power"]
    for c in cols:
        cl = c.lower()
        if cpu_col is None and any(k in cl for k in cpu_keys):
            cpu_col = c
        if gpu_col is None and any(k in cl for k in gpu_keys):
            gpu_col = c
        if cpu_col and gpu_col:
            break
    return cpu_col, gpu_col


def ema(series: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return series
    out = np.empty_like(series, dtype=float)
    if len(series) == 0:
        return out
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def trapezoid_energy_wh(t: np.ndarray, p: np.ndarray) -> float:
    """用于导出窗口能量估计（Wh）；UI 不显示能量，只在 CSV 用。"""
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    pm = 0.5 * (p[:-1] + p[1:])
    return float(np.sum(pm * dt) / 3600.0)


# ------------------- Tkinter App -------------------

class EnergyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x900")

        # State
        self.csv_path: Optional[Path] = None
        self.sep = ","
        self.dec = "."
        self.header: Optional[List[str]] = None
        self.cpu_col: Optional[str] = None
        self.gpu_col: Optional[str] = None
        self.window_sec = DEFAULT_WINDOW_SEC
        self.refresh_ms = DEFAULT_REFRESH_MS
        self.last_ts: Optional[float] = None
        self.buffer_df: Optional[pd.DataFrame] = None   # 窗口数据
        self.session_df: Optional[pd.DataFrame] = None  # 全程数据
        self.paused = False
        self.ema_on = False
        self.ema_alpha = 0.2

        self.load_config()

        # ===== Top Controls =====
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=6)
        self.path_var = tk.StringVar(value="未选择 CSV" if not self.csv_path else str(self.csv_path))
        ttk.Button(top, text="打开 CSV…", command=self.choose_csv).pack(side="left")
        ttk.Label(top, textvariable=self.path_var).pack(side="left", padx=8)

        ttk.Label(top, text="窗口(s):").pack(side="left", padx=(20, 2))
        self.window_var = tk.IntVar(value=self.window_sec)
        ttk.Spinbox(top, from_=10, to=3600, textvariable=self.window_var, width=6, command=self.apply_window).pack(side="left")

        ttk.Label(top, text="刷新(ms):").pack(side="left", padx=(20, 2))
        self.refresh_var = tk.IntVar(value=self.refresh_ms)
        ttk.Spinbox(top, from_=50, to=5000, increment=50, textvariable=self.refresh_var, width=6, command=self.apply_refresh).pack(side="left")

        self.ema_on_var = tk.BooleanVar(value=self.ema_on)
        ttk.Checkbutton(top, text="EMA 平滑", variable=self.ema_on_var, command=self.toggle_ema).pack(side="left", padx=(20, 4))
        ttk.Label(top, text="α:").pack(side="left")
        self.ema_var = tk.DoubleVar(value=self.ema_alpha)
        ttk.Spinbox(top, from_=0.0, to=1.0, increment=0.05, textvariable=self.ema_var, width=4, command=self.apply_ema).pack(side="left")

        self.btn_pause = ttk.Button(top, text="暂停", command=self.toggle_pause)
        self.btn_pause.pack(side="left", padx=(20, 4))

        ttk.Button(top, text="保存截图", command=self.save_pngs).pack(side="left", padx=4)
        ttk.Button(top, text="导出窗口指标", command=self.export_window_metrics).pack(side="left", padx=4)
        ttk.Button(top, text="选择列…", command=self.select_columns).pack(side="right")

        # ===== Tabs =====
        notebook = ttk.Notebook(self)
        tab_power = ttk.Frame(notebook)
        tab_full = ttk.Frame(notebook)  # 全程功率
        notebook.add(tab_power, text="功率 (W)")
        notebook.add(tab_full, text="全程功率 (W，0→现在)")
        notebook.pack(side="top", fill="both", expand=True, padx=8, pady=6)

        # --- Power Figure (window) ---
        self.fig_pow = Figure(figsize=(10, 6.4), dpi=100)
        self.ax_cpu = self.fig_pow.add_subplot(311)
        self.ax_gpu = self.fig_pow.add_subplot(312)
        self.ax_tot = self.fig_pow.add_subplot(313)
        for ax, title in [(self.ax_cpu, "CPU Power (W)"),
                          (self.ax_gpu, "GPU Power (W)"),
                          (self.ax_tot, "Total Power (W) = CPU + GPU")]:
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power (W)")
            ax.grid(alpha=0.3)
        self.line_cpu, = self.ax_cpu.plot([], [], lw=1.5)
        self.line_gpu, = self.ax_gpu.plot([], [], lw=1.5)
        self.line_tot, = self.ax_tot.plot([], [], lw=1.5)
        canvas_pow = FigureCanvasTkAgg(self.fig_pow, master=tab_power)
        canvas_pow.draw()
        canvas_pow.get_tk_widget().pack(side="top", fill="both", expand=True)

        # --- Full-session POWER Figure ---
        self.fig_full = Figure(figsize=(10, 6.4), dpi=100)
        self.ax_cpu_f = self.fig_full.add_subplot(311)
        self.ax_gpu_f = self.fig_full.add_subplot(312)
        self.ax_tot_f = self.fig_full.add_subplot(313)
        for ax, title in [(self.ax_cpu_f, "CPU Power (W) — Full session"),
                          (self.ax_gpu_f, "GPU Power (W) — Full session"),
                          (self.ax_tot_f, "Total Power (W) = CPU + GPU — Full session")]:
            ax.set_title(title)
            ax.set_xlabel("Time since start (s)")
            ax.set_ylabel("Power (W)")
            ax.grid(alpha=0.3)
        self.line_cpu_f, = self.ax_cpu_f.plot([], [], lw=1.5)
        self.line_gpu_f, = self.ax_gpu_f.plot([], [], lw=1.5)
        self.line_tot_f, = self.ax_tot_f.plot([], [], lw=1.5)
        canvas_full = FigureCanvasTkAgg(self.fig_full, master=tab_full)
        canvas_full.draw()
        canvas_full.get_tk_widget().pack(side="top", fill="both", expand=True)

        # KPI bar（仅显示 W）
        kpi = ttk.Frame(self)
        kpi.pack(side="top", fill="x", padx=8, pady=6)
        self.kpi_cpu = tk.StringVar(value="CPU: -")
        self.kpi_gpu = tk.StringVar(value="GPU: -")
        self.kpi_tot = tk.StringVar(value="Total: -")
        ttk.Label(kpi, textvariable=self.kpi_cpu).pack(side="left", padx=8)
        ttk.Label(kpi, textvariable=self.kpi_gpu).pack(side="left", padx=8)
        ttk.Label(kpi, textvariable=self.kpi_tot).pack(side="left", padx=8)

        # Status bar
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(side="bottom", fill="x", padx=8, pady=4)

        self.after(self.refresh_ms, self.loop)

    # ----- Config -----
    def load_config(self):
        if CONFIG_PATH.exists():
            try:
                cfg = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
                self.window_sec = int(cfg.get("window_sec", DEFAULT_WINDOW_SEC))
                self.refresh_ms = int(cfg.get("refresh_ms", DEFAULT_REFRESH_MS))
                csv = cfg.get("csv_path", "")
                self.csv_path = Path(csv) if csv else None
                self.cpu_col = cfg.get("cpu_col") or None
                self.gpu_col = cfg.get("gpu_col") or None
                self.ema_on = bool(cfg.get("ema_on", False))
                self.ema_alpha = float(cfg.get("ema_alpha", 0.2))
            except Exception:
                pass

    def save_config(self):
        try:
            cfg = {
                "window_sec": self.window_sec,
                "refresh_ms": self.refresh_ms,
                "csv_path": str(self.csv_path) if self.csv_path else "",
                "cpu_col": self.cpu_col,
                "gpu_col": self.gpu_col,
                "ema_on": self.ema_on,
                "ema_alpha": self.ema_alpha,
            }
            json.dump(cfg, open(CONFIG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ----- UI actions -----
    def choose_csv(self):
        file = filedialog.askopenfilename(title="选择 HWiNFO/nvidia-smi CSV",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not file:
            return
        file = file.strip().strip('"').strip("'")
        p = Path(file)
        if not p.exists():
            messagebox.showerror("错误", "文件不存在。")
            return
        self.csv_path = p
        self.path_var.set(str(p))
        self.last_ts = None
        self.buffer_df = None
        self.session_df = None
        self.status_var.set("正在解析表头…")
        cols, sep, dec = read_header_and_formats(p)
        if not cols:
            messagebox.showerror("错误", "无法解析 CSV 表头。")
            return
        self.header, self.sep, self.dec = cols, sep, dec
        cpu, gpu = pick_default_cols(cols)
        self.cpu_col = self.cpu_col or cpu
        self.gpu_col = self.gpu_col or gpu
        if self.cpu_col is None or self.gpu_col is None:
            self.select_columns()
        self.status_var.set(
            f"已选择：{p.name} | 分隔符: '{self.sep}' 小数: '{self.dec}' | CPU={self.cpu_col} GPU={self.gpu_col}")
        self.save_config()

    def select_columns(self):
        if not self.header:
            messagebox.showinfo("提示", "请先打开 CSV。")
            return
        dlg = tk.Toplevel(self)
        dlg.title("选择功率列")
        dlg.geometry("560x220")
        ttk.Label(dlg, text="CPU 列：").grid(row=0, column=0, padx=6, pady=8, sticky="e")
        ttk.Label(dlg, text="GPU 列：").grid(row=1, column=0, padx=6, pady=8, sticky="e")
        options = [c for c in self.header if c not in ("Date", "Time")]
        cpu_var = tk.StringVar(value=self.cpu_col or "")
        gpu_var = tk.StringVar(value=self.gpu_col or "")
        cpu_cb = ttk.Combobox(dlg, values=options, textvariable=cpu_var, width=60)
        gpu_cb = ttk.Combobox(dlg, values=options, textvariable=gpu_var, width=60)
        cpu_cb.grid(row=0, column=1, padx=6, pady=8, sticky="w")
        gpu_cb.grid(row=1, column=1, padx=6, pady=8, sticky="w")

        def ok():
            self.cpu_col = cpu_var.get() or None
            self.gpu_col = gpu_var.get() or None
            dlg.destroy()
            self.save_config()

        ttk.Button(dlg, text="确定", command=ok).grid(row=2, column=0, columnspan=2, pady=10)

    def apply_window(self):
        self.window_sec = max(10, int(self.window_var.get()))
        self.save_config()

    def apply_refresh(self):
        self.refresh_ms = max(50, int(self.refresh_var.get()))
        self.save_config()

    def toggle_ema(self):
        self.ema_on = bool(self.ema_on_var.get())
        self.save_config()

    def apply_ema(self):
        try:
            self.ema_alpha = float(self.ema_var.get())
            self.save_config()
        except Exception:
            pass

    def toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.config(text="继续" if self.paused else "暂停")

    def save_pngs(self):
        # 保存两张图（功率 / 全程功率）
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_pow = filedialog.asksaveasfilename(title="保存功率图 PNG", defaultextension=".png",
                                                initialfile=f"power_window_{ts}.png",
                                                filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if file_pow:
            self.fig_pow.tight_layout()
            self.fig_pow.savefig(file_pow, dpi=150)
        file_full = filedialog.asksaveasfilename(title="保存全程功率图 PNG", defaultextension=".png",
                                                 initialfile=f"power_full_{ts}.png",
                                                 filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if file_full:
            self.fig_full.tight_layout()
            self.fig_full.savefig(file_full, dpi=150)
        if file_pow or file_full:
            self.status_var.set("PNG 已保存。")

    def export_window_metrics(self):
        if self.buffer_df is None or self.buffer_df.empty or self.cpu_col is None or self.gpu_col is None:
            messagebox.showinfo("提示", "当前窗口没有数据。")
            return
        df = self.buffer_df
        t = df["t_sec"].to_numpy()
        cpu = pd.to_numeric(df[self.cpu_col], errors="coerce").interpolate().to_numpy()
        gpu = pd.to_numeric(df[self.gpu_col], errors="coerce").interpolate().to_numpy()
        tot = cpu + gpu
        if self.ema_on:
            cpu = ema(cpu, self.ema_alpha)
            gpu = ema(gpu, self.ema_alpha)
            tot = ema(tot, self.ema_alpha)

        def metrics(p):
            last = float(p[-1]) if len(p) else 0.0
            avg = float(np.mean(p)) if len(p) else 0.0
            peak = float(np.max(p)) if len(p) else 0.0
            wh = trapezoid_energy_wh(t, p)  # 可用于估计窗口能量（可选）
            return last, avg, peak, wh

        c_last, c_avg, c_peak, c_wh = metrics(cpu)
        g_last, g_avg, g_peak, g_wh = metrics(gpu)
        t_last, t_avg, t_peak, t_wh = metrics(tot)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file = filedialog.asksaveasfilename(title="导出窗口指标 CSV", defaultextension=".csv",
                                            initialfile=f"window_metrics_{ts}.csv",
                                            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not file:
            return
        rows = [
            ["metric", "CPU", "GPU", "Total"],
            ["last_power_W", c_last, g_last, t_last],
            ["avg_power_W",  c_avg,  g_avg,  t_avg],
            ["peak_power_W", c_peak, g_peak, t_peak],
            ["energy_Wh",    c_wh,   g_wh,   t_wh],  # 如不需要能量列，可自行删除这一行
        ]
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(file, index=False)
        self.status_var.set(f"已导出窗口指标：{file}")

    # ----- Data reading -----
    def read_tail_df(self, approx_lines: int = 1200) -> Optional[pd.DataFrame]:
        if not self.csv_path:
            return None
        try:
            tb = tail_bytes(self.csv_path, approx_lines=approx_lines)
            df = pd.read_csv(io.StringIO(tb.decode(errors="ignore")),
                             sep=self.sep, decimal=self.dec,
                             on_bad_lines="skip", names=self.header, header=None, engine="python")
        except Exception as e:
            self.status_var.set(f"读取失败：{e}")
            return None
        t_series = parse_time_columns(df)
        if t_series is None or t_series.isna().all():
            return None
        df = df.join(t_series).dropna(subset=["t_sec"])
        return df

    # ----- Live loop -----
    def loop(self):
        try:
            if not self.paused and self.csv_path and self.cpu_col and self.gpu_col:
                df_tail = self.read_tail_df(approx_lines=int(self.window_sec * 12))
                if df_tail is not None and not df_tail.empty:
                    if self.last_ts is not None:
                        df_new = df_tail[df_tail["t_sec"] > self.last_ts]
                    else:
                        df_new = df_tail
                    if not df_new.empty:
                        self.last_ts = float(df_new["t_sec"].iloc[-1])
                        if self.buffer_df is None:
                            self.buffer_df = df_new.copy()
                        else:
                            self.buffer_df = pd.concat([self.buffer_df, df_new], ignore_index=True)
                        # 保留窗口
                        t_end = self.buffer_df["t_sec"].iloc[-1]
                        self.buffer_df = self.buffer_df[self.buffer_df["t_sec"] >= max(0.0, t_end - self.window_sec)]
                        # 会话累计（供全程功率使用）
                        if self.session_df is None:
                            self.session_df = self.buffer_df.copy()
                        else:
                            self.session_df = pd.concat([self.session_df, df_new], ignore_index=True)
                        # 绘图 & KPI
                        self.update_plots()
                        self.update_kpis()
                        self.status_var.set(f"已更新至 t={t_end:.1f}s")
        except Exception as e:
            self.status_var.set(f"循环错误：{e}")
        finally:
            self.after(self.refresh_ms, self.loop)

    # ----- Plotting -----
    def update_plots(self):
        if self.buffer_df is None or self.buffer_df.empty:
            return

        # 窗口功率
        dfw = self.buffer_df
        t_w = dfw["t_sec"].to_numpy()
        cpu_w = pd.to_numeric(dfw[self.cpu_col], errors="coerce").interpolate().to_numpy()
        gpu_w = pd.to_numeric(dfw[self.gpu_col], errors="coerce").interpolate().to_numpy()
        tot_w = cpu_w + gpu_w
        if self.ema_on:
            cpu_w = ema(cpu_w, self.ema_alpha)
            gpu_w = ema(gpu_w, self.ema_alpha)
            tot_w = ema(tot_w, self.ema_alpha)

        self.line_cpu.set_data(t_w, cpu_w)
        self.line_gpu.set_data(t_w, gpu_w)
        self.line_tot.set_data(t_w, tot_w)
        for ax, y in [(self.ax_cpu, cpu_w), (self.ax_gpu, gpu_w), (self.ax_tot, tot_w)]:
            t0, t1 = max(0.0, t_w[-1] - self.window_sec), t_w[-1]
            ax.set_xlim(t0, max(t1, t0 + 1.0))
            if np.isfinite(y).any():
                ymin = float(np.nanmin(y)); ymax = float(np.nanmax(y))
                if ymax - ymin < 1e-3: ymax = ymin + 1.0
                ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

        # 全程功率（0→现在）
        if self.session_df is not None and not self.session_df.empty:
            dfs = self.session_df
            t_s = dfs["t_sec"].to_numpy()
            if len(t_s) > 0:
                t_s = t_s - t_s[0]  # 重基线：从 0 开始
            cpu_s = pd.to_numeric(dfs[self.cpu_col], errors="coerce").interpolate().to_numpy()
            gpu_s = pd.to_numeric(dfs[self.gpu_col], errors="coerce").interpolate().to_numpy()
            tot_s = cpu_s + gpu_s
            if self.ema_on:
                cpu_s = ema(cpu_s, self.ema_alpha)
                gpu_s = ema(gpu_s, self.ema_alpha)
                tot_s = ema(tot_s, self.ema_alpha)

            # 轻量抽样，避免 UI 卡顿
            if len(t_s) > MAX_POINTS_FULL:
                step = max(1, len(t_s) // MAX_POINTS_FULL)
                t_s = t_s[::step]; cpu_s = cpu_s[::step]; gpu_s = gpu_s[::step]; tot_s = tot_s[::step]

            self.line_cpu_f.set_data(t_s, cpu_s)
            self.line_gpu_f.set_data(t_s, gpu_s)
            self.line_tot_f.set_data(t_s, tot_s)
            for ax, y in [(self.ax_cpu_f, cpu_s), (self.ax_gpu_f, gpu_s), (self.ax_tot_f, tot_s)]:
                if len(t_s) > 0:
                    ax.set_xlim(0.0, max(t_s[-1], 1.0))
                if np.isfinite(y).any():
                    ymin = float(np.nanmin(y)); ymax = float(np.nanmax(y))
                    if ymax - ymin < 1e-3: ymax = ymin + 1.0
                    ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
        else:
            self.line_cpu_f.set_data([], [])
            self.line_gpu_f.set_data([], [])
            self.line_tot_f.set_data([], [])

        self.fig_pow.tight_layout();  self.fig_pow.canvas.draw_idle()
        self.fig_full.tight_layout(); self.fig_full.canvas.draw_idle()

    def update_kpis(self):
        df = self.buffer_df
        t = df["t_sec"].to_numpy()
        cpu = pd.to_numeric(df[self.cpu_col], errors="coerce").interpolate().to_numpy()
        gpu = pd.to_numeric(df[self.gpu_col], errors="coerce").interpolate().to_numpy()
        tot = cpu + gpu
        if self.ema_on:
            cpu = ema(cpu, self.ema_alpha)
            gpu = ema(gpu, self.ema_alpha)
            tot = ema(tot, self.ema_alpha)
        last_c, avg_c, peak_c = cpu[-1], float(np.mean(cpu)), float(np.max(cpu))
        last_g, avg_g, peak_g = gpu[-1], float(np.mean(gpu)), float(np.max(gpu))
        last_t, avg_t, peak_t = tot[-1], float(np.mean(tot)), float(np.max(tot))
        self.kpi_cpu.set(f"CPU 现在/均/峰 (W)：{last_c:.1f} / {avg_c:.1f} / {peak_c:.1f}")
        self.kpi_gpu.set(f"GPU 现在/均/峰 (W)：{last_g:.1f} / {avg_g:.1f} / {peak_g:.1f}")
        self.kpi_tot.set(f"Total 现在/均/峰 (W)：{last_t:.1f} / {avg_t:.1f} / {peak_t:.1f}")


if __name__ == "__main__":
    app = EnergyApp()
    app.mainloop()
