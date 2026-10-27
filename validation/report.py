"""Aggregate the CSVs written by validation.opencap into summary tables.

    python -m validation.report
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results"
ENGINES = ["mediapipe", "rtmpose-lightweight", "rtmpose-balanced", "rtmpose-performance"]
MOVES = ["squats", "STS", "DJ", "walking"]
JOINTS = ["knee", "hip", "ankle"]


def load() -> list[dict]:
    rows = []
    for e in ENGINES:
        for mv in MOVES:
            f = OUT / f"{e}_{mv}.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(open(f)):
                r["move"] = mv
                for k in ("mae", "rmse", "bias", "sd", "r", "rom_est", "rom_ref", "view_deg", "fps_proc", "lag_s", "detected"):
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def mean(rows, key):
    v = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def table(rows) -> str:
    out = []
    for j in JOINTS:
        out.append(f"\n{j.upper()}  — MAE° (bias°) [ROM error %]   n = trials x cameras")
        out.append(f"{'engine':22s}" + "".join(f"{m:>24s}" for m in MOVES))
        for e in ENGINES:
            line = f"{e:22s}"
            for mv in MOVES:
                rs = [r for r in rows if r["engine"] == e and r["move"] == mv and r["joint"] == f"{j}_aligned"]
                if not rs:
                    line += f"{'—':>24s}"
                    continue
                rom = np.mean([(r["rom_est"] - r["rom_ref"]) / r["rom_ref"] * 100 for r in rs if r["rom_ref"] > 1])
                line += f"{mean(rs, 'mae'):6.1f} ({mean(rs, 'bias'):+5.1f}) [{rom:+4.0f}%] n={len(rs):<3d}"
            out.append(line)
    return "\n".join(out)


def by_view(rows) -> str:
    """Error as a function of how far the camera is from a true profile view."""
    out = ["\nKNEE error vs camera view (0° = perfect profile). Squats + STS + DJ, all engines pooled per engine"]
    bins = [(0, 30), (30, 40), (40, 50), (50, 91)]
    out.append(f"{'engine':22s}" + "".join(f"{f'{a}-{b}°':>20s}" for a, b in bins) + f"{'corr(view, MAE)':>18s}")
    for e in ENGINES:
        rs = [r for r in rows if r["engine"] == e and r["joint"] == "knee_aligned" and r["move"] != "walking"
              and np.isfinite(r["view_deg"])]
        if not rs:
            continue
        line = f"{e:22s}"
        for a, b in bins:
            g = [r for r in rs if a <= r["view_deg"] < b]
            line += f"{(f'{mean(g, chr(109)+chr(97)+chr(101)):.1f}° (n={len(g)})' if g else '—'):>20s}"
        c = np.corrcoef([r["view_deg"] for r in rs], [r["mae"] for r in rs])[0, 1]
        out.append(line + f"{c:18.2f}")
    return "\n".join(out)


def views(rows) -> str:
    out = ["\nEstimated camera view angle (degrees from profile), mean by movement and camera (MediaPipe run)"]
    rs = [r for r in rows if r["engine"] == "mediapipe" and r["joint"] == "knee_aligned"]
    for mv in MOVES:
        for cam in ("Cam0", "Cam4"):
            g = [r["view_deg"] for r in rs if r["move"] == mv and r["cam"] == cam and np.isfinite(r["view_deg"])]
            if g:
                out.append(f"  {mv:8s} {cam}: {np.mean(g):4.0f}°  (min {np.min(g):.0f}°, max {np.max(g):.0f}°)")
    return "\n".join(out)


def main():
    rows = load()
    text = table(rows) + "\n" + views(rows) + "\n" + by_view(rows)
    print(text)
    (OUT / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
