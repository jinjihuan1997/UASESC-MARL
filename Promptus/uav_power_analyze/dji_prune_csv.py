#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dji_prune_csv.py  (force offsetTime only)
- 仅做“剪枝”：从大型 DJI CSV 中保留绘图所需的最小列集合（时间/电压/电流/SoC）
- 时间列【强制只保留】Clock:offsetTime（真实秒）；明确不保留 GPS 时间与 Clock:Tick#
- 明确不保留高度相关列
- 生成精简 CSV（可选 Parquet）
- 可选：剪枝完成后自动调用原绘图脚本 dji_energy_plot.py（绘图逻辑不改）

示例：
  只剪枝：
    python uav_power_analyze/dji_prune_csv.py --csv uav_power_analyze/raw_data/condition_1/FLY967.csv
  剪枝并自动绘图：
    python uav_power_analyze/dji_prune_csv.py --csv uav_power_analyze/raw_data/condition_1/FLY967.csv --auto_plot --plot_args "--lang zh --dpi 200"
"""

import argparse
import json
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

# ---------- 工具 ----------
_SUFFIX_RE = re.compile(r":[A-Z]$")  # 去掉列名末尾类似 ":D" 的后缀（DJI 常见）


def _normalize_colname(c: str) -> str:
    return _SUFFIX_RE.sub("", c)


def _build_norm_map(columns: List[str]) -> Dict[str, str]:
    """规范名 -> 真实名 映射。"""
    norm2orig = {}
    for c in columns:
        n = _normalize_colname(c)
        if n not in norm2orig or ("battery_info" in c and "battery_info" not in norm2orig[n]):
            norm2orig[n] = c
    return norm2orig


def _pick_first_existing(all_cols: List[str], candidates: List[str]) -> Optional[str]:
    """按候选列表在表头里找第一命中的真实列名。"""
    cols = set(all_cols)
    # 精确
    for cand in candidates:
        if cand in cols:
            return cand
    # 规范名
    norm2orig = _build_norm_map(all_cols)
    for cand in candidates:
        n = _normalize_colname(cand)
        if n in norm2orig:
            return norm2orig[n]
    # 宽松包含
    patt = re.compile("|".join([re.escape(_normalize_colname(c)) for c in candidates]), re.I)
    for c in all_cols:
        if patt.search(_normalize_colname(c)):
            return c
    return None


# ---------- 只保留【offsetTime / V / I / SoC】，明确剔除高度 ----------
TIME_FORCE = ["Clock:offsetTime"]  # 只允许这个时间列

V_CAND = ["battery_info:BatVoltage", "battery_info:Voltage", "system_voltage:system_voltage"]
I_CAND = ["battery_info:BatCurrent", "battery_info:Current"]
SOC_CAND = ["battery_info:CapPercentage", "battery_info:sopPercentage", "battery_info:soc", "battery_info:SoC"]

ALT_BAN_PATTS = [
    "IMU_ATTI(0):relativeHeight",
    "osd_data:relativeHeight",
    "GPS:heightMSL",
    "IMUCalcs(0):height",
    "IMU_ATTI(0):absoluteHeight",
]

# 同时明确列出不允许进入 keep 的“时间”类列（即使存在也不保留）
TIME_BAN_PATTS = [
    "Clock:Tick#",
    "GPS:dateTimeStamp",
    "GPS:Date",
    "GPS:Time",
]


def decide_keep_columns(header_cols: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    只保留：
      - 时间：强制 Clock:offsetTime
      - 电压/电流/SoC：各 1 列（如存在）
      - 明确剔除：高度相关列、GPS 时间、Clock:Tick#
    """
    keep: List[str] = []
    explain: Dict[str, List[str]] = {"time": [], "voltage": [], "current": [], "soc": []}

    # 先把禁止列（高度 + 其它时间列）剔除
    banned = set()
    for b in ALT_BAN_PATTS + TIME_BAN_PATTS:
        for c in header_cols:
            if _normalize_colname(c) == _normalize_colname(b):
                banned.add(c)
    candidates = [c for c in header_cols if c not in banned]

    # 时间：必须命中 Clock:offsetTime
    tcol = _pick_first_existing(candidates, TIME_FORCE)
    if not tcol:
        raise RuntimeError(
            "未找到时间列 Clock:offsetTime（已按要求禁用 GPS 时间与 Clock:Tick#）。\n"
            "请确认原始 CSV 是否包含该列，或先用外部工具生成 offsetTime。"
        )
    keep.append(tcol)
    explain["time"] = [tcol]

    # V / I / SoC
    v = _pick_first_existing(candidates, V_CAND)
    if v:
        keep.append(v); explain["voltage"] = [v]
    i = _pick_first_existing(candidates, I_CAND)
    if i:
        keep.append(i); explain["current"] = [i]
    soc = _pick_first_existing(candidates, SOC_CAND)
    if soc:
        keep.append(soc); explain["soc"] = [soc]

    # 去重且保持顺序
    out, seen = [], set()
    for c in keep:
        if c not in seen:
            out.append(c); seen.add(c)
    return out, explain


# ---------- CSV 剪枝（分块） ----------
def prune_csv(src: Path, dst_csv: Path, keep_cols: List[str], chunksize: int = 200_000,
              encodings_try=("utf-8", "utf-8-sig", "latin1"),
              progress: bool = True) -> int:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    if dst_csv.exists():
        dst_csv.unlink()

    total_lines = None
    if progress:
        try:
            with open(src, "rb") as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            total_lines = None

    wrote = 0
    t0 = time.time()
    header_written = False

    for enc in encodings_try:
        try:
            reader = pd.read_csv(src, dtype=str, low_memory=False, usecols=keep_cols,
                                 chunksize=chunksize, encoding=enc, on_bad_lines="skip")
            for chunk in reader:
                chunk = chunk[keep_cols]
                chunk.to_csv(dst_csv, index=False, mode="a", header=not header_written)
                header_written = True
                wrote += len(chunk)
                if progress:
                    if total_lines:
                        pct = 100.0 * min(wrote + 1, total_lines) / total_lines
                        sys.stdout.write(f"\r[prune] {wrote} rows written ({pct:.1f}%) ...")
                    else:
                        sys.stdout.write(f"\r[prune] {wrote} rows written ...")
                    sys.stdout.flush()
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            raise

    if progress:
        dt = time.time() - t0
        sys.stdout.write(f"\n[prune] done. rows={wrote} time={dt:.2f}s -> {dst_csv}\n")
    return wrote


def maybe_write_parquet(dst_csv: Path, dst_parquet: Path) -> bool:
    try:
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq  # noqa
        df = pd.read_csv(dst_csv, dtype=str, low_memory=False)
        df.to_parquet(dst_parquet, index=False)
        print(f"[parquet] wrote {dst_parquet}")
        return True
    except Exception as e:
        print("[parquet] skip (pyarrow missing or failed):", e)
        return False


def write_manifest(manifest_path: Path, src: Path, slim: Path,
                   keep_cols: List[str], explain: Dict[str, List[str]], auto_plot_cmd: List[str]):
    data = {
        "source_csv": str(src),
        "slim_csv": str(slim),
        "kept_columns": keep_cols,
        "mapping": explain,
        "auto_plot_cmd": auto_plot_cmd,
        "ts": time.time(),
        "note": "Time column strictly kept as Clock:offsetTime; GPS times and Clock:Tick# are explicitly excluded; altitude columns removed.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[manifest] wrote {manifest_path}")


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="DJI CSV 剪枝（仅保留 Clock:offsetTime）")
    ap.add_argument("--csv", required=True, help="原始 DJI CSV 路径")
    ap.add_argument("--out", default="", help="输出目录（默认 CSV 同目录下 out_data/clean/）")
    ap.add_argument("--chunksize", type=int, default=200000, help="按行分块大小（默认200k）")
    ap.add_argument("--parquet", action="store_true", help="同时写出 Parquet（若系统支持）")
    ap.add_argument("--auto_plot", action="store_true",
                    help="剪枝完成后自动调用 dji_energy_plot.py（绘图脚本本身不改动）")
    ap.add_argument("--plotter", default="uav_power_analyze/dji_energy_plot.py",
                    help="原始绘图脚本路径（默认 uav_power_analyze/dji_energy_plot.py）")
    ap.add_argument("--plot_args", default="", help="传给绘图脚本的附加参数，如：\"--lang zh --dpi 200\"")
    args = ap.parse_args()

    src = Path(args.csv).expanduser().resolve()
    if not src.exists():
        print("源 CSV 不存在：", src); sys.exit(2)

    base_out = Path(args.out) if args.out else src.parent / "out_data" / "clean"
    base_out.mkdir(parents=True, exist_ok=True)

    stem = src.stem
    dst_csv = base_out / f"{stem}_slim.csv"
    dst_parquet = base_out / f"{stem}_slim.parquet"
    manifest = base_out / f"{stem}_slim.manifest.json"

    # 读表头决定保留列
    header = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            header = pd.read_csv(src, nrows=0, dtype=str, low_memory=False, encoding=enc,
                                 on_bad_lines="skip").columns.tolist()
            break
        except UnicodeDecodeError:
            continue
    if header is None:
        header = pd.read_csv(src, nrows=0, dtype=str, low_memory=False, on_bad_lines="skip").columns.tolist()

    try:
        keep_cols, explain = decide_keep_columns(header)
    except RuntimeError as e:
        print(str(e)); sys.exit(3)

    print("[keep] columns:")
    for c in keep_cols:
        print("  -", c)
    print("[time] chosen:", ", ".join(explain["time"]))
    if explain.get("voltage"): print("[voltage] chosen:", ", ".join(explain["voltage"]))
    if explain.get("current"): print("[current] chosen:", ", ".join(explain["current"]))
    if explain.get("soc"):     print("[soc] chosen:", ", ".join(explain["soc"]))

    # 剪枝
    prune_csv(src, dst_csv, keep_cols=keep_cols, chunksize=args.chunksize)

    # 可选 Parquet
    if args.parquet:
        maybe_write_parquet(dst_csv, dst_parquet)

    # Manifest & 自动绘图
    plot_cmd: List[str] = []
    if args.auto_plot:
        plot_cmd = [sys.executable, str(Path(args.plotter)), "--csv", str(dst_csv)]
        if args.plot_args:
            import shlex
            plot_cmd += shlex.split(args.plot_args)
        write_manifest(manifest, src, dst_csv, keep_cols, explain, plot_cmd)
        print("[auto_plot] running:", " ".join(plot_cmd))
        try:
            subprocess.run(plot_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("[auto_plot] 绘图脚本返回非零退出码：", e.returncode); sys.exit(e.returncode)
    else:
        write_manifest(manifest, src, dst_csv, keep_cols, explain, [])
        print("\n下一步（手动绘图，原脚本不变），示例：")
        print(f"  python uav_power_analyze/dji_energy_plot.py --csv \"{dst_csv}\" --lang zh --dpi 200")


if __name__ == "__main__":
    main()
