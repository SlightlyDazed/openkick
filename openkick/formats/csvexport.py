"""CSV export: raw landmarks (same layout as the original MediaPipe scripts) and
per-frame measurement series."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core.project import Project


def _fmt(v, nd=4):
    return "" if v is None or not np.isfinite(v) else f"{float(v):.{nd}f}"


def export_landmarks(project: Project, path: str | Path):
    """frame, timestamp_ms, <name>_x/_y/_z/_visibility with x, y normalized 0-1
    (compatible with the csv_a_kva / csv_a_trc scripts)."""
    v, pose = project.video, project.pose
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["frame", "timestamp_ms"]
        for name in L.NAMES:
            head += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"]
        w.writerow(head)
        for i in range(project.frame_count):
            row = [i, int(round(v.time_of(i) * 1000))]
            for j in range(L.COUNT):
                x, y = pose.xy[i, j]
                row += [_fmt(x / v.width, 6) if np.isfinite(x) else "",
                        _fmt(y / v.height, 6) if np.isfinite(y) else "",
                        _fmt(pose.z[i, j], 6), _fmt(pose.visibility[i, j], 4)]
            w.writerow(row)


def export_measurements(project: Project, path: str | Path):
    """One row per frame: time, phase and every measurement value."""
    ms = project.measurements
    t = A.times(project)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["frame", "time_s", "phase"]
        series = []
        for m in ms:
            s = A.series(project, m)
            head.append(f"{m.name} ({s.unit})")
            series.append(s.value)
            if m.kind == "angle":
                head.append(f"{m.name} (°/s)")
                series.append(s.rate)
            if m.kind in ("trajectory", "point"):
                head += [f"{m.name} x (px)", f"{m.name} y (px)"]
                series += [s.xy[:, 0], s.xy[:, 1]]
        w.writerow(head)
        for i in range(project.frame_count):
            ph = project.phase_at(i)
            w.writerow([i, f"{t[i]:.4f}", ph.name if ph else ""] + [_fmt(s[i], 3) for s in series])
