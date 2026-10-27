"""Several analyses side by side: the same person in different conditions (two saddle heights,
before and after a treatment) or different people.

Each analysis is a file: a 2D project (.okick) or a 3D session (.ok3d). From each one come its
cycles (pedal strokes or strides) leg by leg, measured the same way whatever the file, and the
mean curve of the cycle. Comparing is done leg against leg: the right leg of one with the right
leg of the other.

Different people: angles compare as they are; lengths can be put against each person's height.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openkick.core import cycling as C
from openkick.i18n import tr

SAMPLES = 101                  # a cycle resampled from 0 to 100 %
SIDES = ("left", "right")
GAIT_KEYS = ("stride_time", "cadence", "stance", "swing", "knee_contact", "knee_swing", "knee_rom", "hip_rom",
             "ankle_rom", "trunk", "step_length", "step_width", "stride_length", "speed")
LENGTH_UNITS = ("cm", "m")     # measures put against height when the people differ


@dataclass
class Analysis:
    path: str
    label: str
    kind: str                                    # "2D" or "3D"
    activity: str | None = None                  # "cycling", "gait" or None (nothing found)
    subject: object = None
    cycles: dict = field(default_factory=dict)   # side -> list of cycle dicts
    curves: dict = field(default_factory=dict)   # side -> {"knee": (mean, sd)...} over 0..100 % of the cycle
    note: str = ""                               # why something is missing
    reliable: dict = field(default_factory=dict)  # side -> False when that leg's reconstruction is doubtful

    def sides(self) -> list[str]:
        return [s for s in SIDES if self.cycles.get(s)]


def _resample(series: np.ndarray, cycles: list[dict], first_key: str = "first", last_key: str = "last") -> tuple:
    """Mean and spread of a per-frame series over every cycle, from 0 to 100 % of it."""
    rows = []
    for c in cycles:
        a, b = int(c[first_key]), int(c[last_key]) + 1
        seg = series[a:b + 1]
        if len(seg) < 5 or not np.isfinite(seg).mean() > 0.8:
            continue
        x = np.linspace(0, 1, len(seg))
        ok = np.isfinite(seg)
        rows.append(np.interp(np.linspace(0, 1, SAMPLES), x[ok], seg[ok]))
    if not rows:
        return None
    rows = np.array(rows)
    return rows.mean(axis=0), rows.std(axis=0), len(rows)


# ── 2D projects ────────────────────────────────────────────────────────

def _cycling_2d(project) -> tuple[dict, dict, str]:
    """Pedal strokes of the leg turned to the camera, in the side-on parts of a 2D project."""
    from openkick.core import analysis as A
    from openkick.core import landmarks as L
    from openkick.plugins.builtin.cycling import MIN_STRETCH_S, _runs
    p = project
    fps = p.video.fps
    side_on = A.side_on_frames(p.pose, fps)
    t = A.times(p)
    by_side, curves = {}, {}
    xy = p.pose.xy.astype(np.float64)
    cal = p.calibration.px_per_m if p.calibration else None
    for a, b in _runs(side_on, int(MIN_STRETCH_S * fps)):
        side = A.facing_side(p, a, b)
        seg = xy[a:b + 1]

        def P(n):
            return seg[:, L.INDEX[f"{side}_{n}"]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            knee = A.lowpass(A.angle_between(P("hip"), P("knee"), P("ankle")), fps)
            up = P("shoulder") - P("hip")
            sole = P("foot_index") - P("heel")
            series = {"hip": A.lowpass(A.angle_between(P("shoulder"), P("hip"), P("knee")), fps),
                      "ankle": A.lowpass(A.angle_between(P("knee"), P("ankle"), P("foot_index")), fps),
                      "foot": A.lowpass(np.degrees(np.arctan2(sole[:, 1], np.abs(sole[:, 0]))), fps),
                      "trunk": np.degrees(np.arctan2(-up[:, 1], np.abs(up[:, 0])))}
            if cal:
                series["hip_up"] = A.lowpass(-P("hip")[:, 1] / cal, fps)
        for s in C.strokes(knee, t[a:b + 1], fps):
            s.update(C.per_stroke(s, series))
            s.update({"side": side, "first": a + s["a"], "last": a + s["b"] - 1, "top_frame": a + s["top"],
                      "time": float(t[a + s["a"]])})
            by_side.setdefault(side, []).append(s)
        full = np.full(p.frame_count, np.nan)
        for name, values in (("knee", knee), ("hip", series["hip"]), ("ankle", series["ankle"])):
            curves.setdefault(side, {}).setdefault(name, full.copy())[a:b + 1] = values
    return by_side, curves, "" if by_side else tr("no pedal strokes seen from the side")


def _gait_2d(project) -> tuple[dict, dict, str]:
    from openkick.core import analysis as A
    from openkick.plugins import PluginContext
    from openkick.plugins.builtin import gait as GP
    from openkick.plugins.builtin.cycling import _runs
    ctx = PluginContext(project)
    p = project
    fps = p.video.fps
    by_side = {}
    for a, b in _runs(GP._side_on(ctx), int(GP.MIN_STRETCH_S * fps)):
        near, forward = A.facing_side(p, a, b), GP._forward(ctx, a, b)
        for side in (near, GP._other(near)):
            for s in GP._strides(ctx, a, b, side, forward):
                by_side.setdefault(side, []).append(s)
    curves = {}
    from openkick.core import landmarks as L
    xy = p.pose.xy.astype(np.float64)
    for side in by_side:
        curves[side] = {"knee": A.lowpass(A.angle_between(xy[:, L.INDEX[f"{side}_hip"]], xy[:, L.INDEX[f"{side}_knee"]],
                                                          xy[:, L.INDEX[f"{side}_ankle"]]), fps)}
    return by_side, curves, "" if by_side else tr("no strides seen from the side")


# ── 3D sessions ────────────────────────────────────────────────────────

def _leg_spread(X, side: str) -> float:
    from openkick.multiview.geometry import IDX
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a, b in (("hip", "knee"), ("knee", "ankle")):
            v = np.linalg.norm(X[:, IDX[f"{side}_{a}"]] - X[:, IDX[f"{side}_{b}"]], axis=1)
            v = v[np.isfinite(v)]
            if len(v) > 10:
                mid = np.median(v)
                out.append(100 * 1.4826 * np.median(np.abs(v - mid)) / mid)
    return float(max(out)) if out else float("inf")


def from_session(s, path: str, name: str, activity: str | None = None) -> Analysis:
    """The cycles of a 3D session (open or read from a file)."""
    item = Analysis(str(path), name, "3D", subject=s.subject)
    kind = activity or s.template
    if kind not in ("cycling", "gait"):
        from openkick.multiview.cycling import looks_like_pedalling
        kind = "cycling" if looks_like_pedalling(s.X, s.cams) else "gait"
    item.activity = kind
    for side in SIDES:          # on the points as detected: fixed bones would hide a doubtful leg
        item.reliable[side] = _leg_spread(s.X_raw, side) <= 10.0
    if kind == "cycling":
        from openkick.multiview.cycling import analyse
        c = analyse(s.X, s.t, s.cams, s.quality())
        if c is None:
            item.note = tr("no pedal strokes found")
            return item
        item.cycles = {k: v for k, v in c.by_side.items() if v}
        for side, ser in c.series.items():
            item.curves[side] = {k: _resample(v, item.cycles.get(side, [])) for k, v in ser.items()}
    else:
        from openkick.multiview.gait import analyse
        g = analyse(s.X, s.t)
        if g is None:
            item.note = tr("no strides found")
            return item
        for st in g.every:
            item.cycles.setdefault(st["side"], []).append(st)
    return item


def load(path: str, activity: str | None = None, label: str | None = None) -> Analysis:
    """Read one analysis and find its cycles. `activity` forces what is looked for."""
    path = str(path)
    name = label or Path(path).stem
    if path.lower().endswith(".ok3d"):
        from openkick.multiview.session import Session
        return from_session(Session.load(path, require_videos=False), path, name, activity)
    from openkick.core.project import Project
    p = Project.load(path)
    item = Analysis(path, name, "2D", subject=p.subject)
    kind = activity or (p.template if p.template in ("cycling", "gait") else None) or "cycling"
    item.activity = kind
    by_side, curves, note = _cycling_2d(p) if kind == "cycling" else _gait_2d(p)
    item.cycles, item.note = by_side, note
    item.reliable = {side: True for side in by_side}
    for side, ser in curves.items():
        item.curves[side] = {k: _resample(v, by_side.get(side, [])) for k, v in ser.items()}
    return item


# ── tables ─────────────────────────────────────────────────────────────

def _measures(activity: str):
    if activity == "gait":
        from openkick.core import gait as G
        return G, GAIT_KEYS
    return C, C.POSITION + C.DYNAMICS


def _value(cycle: dict, key: str, unit: str, item: Analysis, per_height: bool) -> float:
    v = cycle.get(key)
    if v is None or not np.isfinite(v):
        return float("nan")
    if per_height and unit in LENGTH_UNITS:
        h = getattr(item.subject, "height_m", None)
        if not h:
            return float("nan")
        return 100 * (v / 100 if unit == "cm" else v) / h       # % of the height
    return float(v)


def table(items: list[Analysis], side: str, per_height: bool = False):
    """One leg of every analysis: the typical value of each measure, the difference from the first
    analysis, and a dot where that difference is more than what changes from one cycle to the next."""
    from openkick.plugins import Table
    activity = items[0].activity
    M, keys = _measures(activity)
    cols = [tr("Measure")] + [i.label for i in items] + [tr("{b} − {a}", a=items[0].label, b=i.label) for i in items[1:]]
    rows = []
    for key in keys:
        unit = M.MEASURES[key][1]
        typ = []
        for i in items:
            cyc = i.cycles.get(side) or []
            typ.append(C.typical(_value(c, key, unit, i, per_height) for c in cyc))
        if not any(np.isfinite(m) for m, _ in typ):
            continue
        shown_unit = "% " + tr("height") if per_height and unit in LENGTH_UNITS else unit

        def fmt(v):
            if not np.isfinite(v):
                return "—"
            return f"{v:.1f}°" if shown_unit == "°" else f"{v:.{2 if shown_unit in ('s', 'm', 'm/s') else 1}f} {tr(shown_unit)}"
        row = [tr(M.MEASURES[key][0])] + [f"{fmt(m)} ± {sd:.1f}" if np.isfinite(m) else "—" for m, sd in typ]
        a, sa = typ[0]
        for b, sb in typ[1:]:
            row.append(f"{b - a:+.1f}{' ●' if C.beyond(a, sa, b, sb) else ''}" if np.isfinite(a) and np.isfinite(b) else "—")
        rows.append(row)
    leg = tr("Left leg") if side == "left" else tr("Right leg")
    return Table(cols, rows, leg)


def subjects_table(items: list[Analysis]):
    from openkick.core.subject import describe
    from openkick.plugins import Table
    labels = []
    per = []
    for i in items:
        d = dict(describe(i.subject, tr)) if i.subject is not None else {}
        per.append(d)
        labels += [k for k in d if k not in labels]
    if not labels:
        return None
    return Table([tr("Item")] + [i.label for i in items], [[k] + [d.get(k, "—") for d in per] for k in labels],
                 tr("Patients"))


def summary(items: list[Analysis]) -> list[str]:
    out = []
    for i in items:
        n = {s: len(c) for s, c in i.cycles.items() if c}
        what = tr("pedal strokes") if i.activity == "cycling" else tr("strides")
        legs = ", ".join(f"{tr('left') if s == 'left' else tr('right')} {k}" for s, k in n.items()) or "0"
        doubt = [s for s, ok in i.reliable.items() if not ok and s in n]
        line = f"{i.label} ({i.kind}): {what} {legs}"
        if doubt:
            line += " · " + tr("the 3D of the {side} leg is doubtful (its thigh or shank changes length by more "
                               "than 10 %)", side=" / ".join(tr("left") if s == "left" else tr("right") for s in doubt))
        if i.note:
            line += f" · {i.note}"
        out.append(line)
    return out
