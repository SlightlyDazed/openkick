"""Pedal strokes and their measures, shared by the 2D plugin, the 3D reconstruction and the
comparison of several analyses.

A stroke runs from one bottom of the stroke (leg most stretched) to the next. Its top (leg most
bent) splits it into recovery (the leg comes up) and propulsion (it pushes down). The bottom and
top fall between frames, so they are read off the parabola through three samples; the cadence
is timed where the knee closes fastest, halfway down, which is known far better than the flat
bottom.
"""

from __future__ import annotations

import warnings

import numpy as np

from openkick.i18n import tr

COLOR_RECOVERY = "#2563EB"
COLOR_PROPULSION = "#10B981"

# key: (name, unit). Angles are measured inside the joint (180° is a straight limb); "*" marks what
# the detector places least reliably. The 3D-only ones need two or more cameras.
MEASURES = {
    "cadence": ("Cadence", "rpm"),
    "knee_bottom": ("Knee, bottom", "°"),
    "knee_top": ("Knee, top", "°"),
    "hip_top": ("Hip, top", "°"),
    "ankle_bottom": ("Ankle, bottom *", "°"),
    "foot_bottom": ("Toes down, bottom *", "°"),
    "ankle_travel": ("Ankle travel *", "°"),
    "trunk": ("Trunk tilt", "°"),
    "knee_lateral": ("Knee side-to-side travel", "cm"),       # 3D: across the bike, during one stroke
    "knee_inward": ("Knee towards the bike", "cm"),           # 3D: + inside the hip-ankle line, - outside
    "pelvis_rock": ("Pelvis rocking", "°"),                   # 3D: tilt of the hip line, range in one stroke
    "trunk_sway": ("Trunk side-to-side sway", "cm"),          # 3D: shoulders across the bike, against the pelvis
    "hip_travel": ("Hip bounce", "cm"),                       # up and down in one stroke
}
POSITION = ("knee_bottom", "knee_top", "hip_top", "ankle_bottom", "foot_bottom", "ankle_travel", "trunk")
DYNAMICS = ("cadence", "knee_lateral", "knee_inward", "pelvis_rock", "trunk_sway", "hip_travel")


def name(key: str) -> str:
    return tr(MEASURES[key][0])


def unit(key: str) -> str:
    return MEASURES[key][1]


def show(v, key: str) -> str:
    u = unit(key)
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.1f}°" if u == "°" else f"{v:.1f} {u}"


def typical(values) -> tuple[float, float]:
    """Median and the usual distance from it (a standard deviation one bad stroke cannot inflate)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return float("nan"), float("nan")
    mid = float(np.median(v))
    return mid, float(1.4826 * np.median(np.abs(v - mid)))


def vertex(y: np.ndarray, i: int) -> tuple[float, float]:
    """Where the extreme near sample `i` really is (a fraction of a frame) and its value."""
    if i <= 0 or i >= len(y) - 1 or not np.isfinite(y[i - 1:i + 2]).all():
        return float(i), float(y[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    bend = a - 2 * b + c
    if abs(bend) < 1e-9:
        return float(i), float(b)
    shift = 0.5 * (a - c) / bend
    return i + shift, float(b - 0.25 * (a - c) * shift)


def _descent(y: np.ndarray, start: int, stop: int, level: float) -> float | None:
    for f in range(start, min(stop, len(y) - 1)):
        if y[f] >= level > y[f + 1]:
            return f + (y[f] - level) / (y[f] - y[f + 1])
    return None


def strokes(knee: np.ndarray, t: np.ndarray, fps: float) -> list[dict]:
    """Every stroke of a knee angle series (degrees, already filtered; NaN where unseen).
    Each is {"a", "b": first and last frame, "top": frame of the top, "seconds", "cadence",
    "knee_bottom", "knee_top"} with frames counted within the series."""
    from scipy.signal import find_peaks
    seen = np.isfinite(knee)
    if seen.sum() < fps:
        return []
    frames = np.arange(len(knee))
    filled = np.interp(frames, frames[seen], knee[seen])
    swing = np.percentile(filled, 95) - np.percentile(filled, 5)
    bottoms, _ = find_peaks(filled, distance=max(2, int(0.3 * fps)), prominence=max(10.0, 0.3 * swing))
    if len(bottoms) < 2:
        return []
    gaps = np.diff(bottoms)
    tops = [i + int(np.argmin(filled[i:j + 1])) for i, j in zip(bottoms[:-1], bottoms[1:])]
    tops.append(bottoms[-1] + int(np.argmin(filled[bottoms[-1]:bottoms[-1] + int(gaps[-1]) + 1])))
    halfway = [_descent(filled, b, k, (filled[b] + filled[k]) / 2) for b, k in zip(bottoms, tops)]
    out = []
    for n, (i, j) in enumerate(zip(bottoms[:-1], bottoms[1:])):
        usual = float(np.median(gaps[max(0, n - 3):n + 4]))         # the cadence may change along the clip
        if not 0.6 * usual <= j - i <= 1.6 * usual or seen[i:j + 1].mean() < 0.8:
            continue
        at_i, knee_bottom = vertex(filled, i)
        at_j, _ = vertex(filled, j)
        a, b = halfway[n], halfway[n + 1]
        if a is not None and b is not None and 0.6 * (j - i) <= b - a <= 1.6 * (j - i):
            at_i, at_j = a, b
        _, knee_top = vertex(filled, tops[n])
        seconds = float(np.interp(at_j, frames, t) - np.interp(at_i, frames, t))
        out.append({"a": int(i), "b": int(j), "top": int(tops[n]), "seconds": seconds,
                    "cadence": 60.0 / seconds if seconds > 0 else float("nan"),
                    "knee_bottom": knee_bottom, "knee_top": knee_top})
    return out


def per_stroke(s: dict, series: dict) -> dict:
    """Measures of one stroke from per-frame series (any of: "hip", "ankle", "foot", "trunk",
    "knee_ml", "pelvis_tilt", "trunk_ml", "hip_up"), over the frames of the stroke."""
    a, b, top = s["a"], s["b"], s["top"]
    sl = slice(a, b + 1)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if "hip" in series:
            seg = series["hip"][sl]
            out["hip_top"] = float(np.nanmin(seg)) if np.isfinite(seg).any() else float("nan")
        if "ankle" in series:
            v = series["ankle"]
            out["ankle_bottom"] = float(v[a])
            out["ankle_travel"] = float(np.nanmax(v[sl]) - np.nanmin(v[sl]))
        if "foot" in series:
            out["foot_bottom"] = float(series["foot"][a])
        if "trunk" in series:
            out["trunk"] = float(np.nanmean(series["trunk"][sl]))
        if "knee_ml" in series:
            v = series["knee_ml"][sl]
            out["knee_lateral"] = float(100 * (np.nanmax(v) - np.nanmin(v)))
            out["knee_inward"] = float(100 * np.nanmean(v))
        if "pelvis_tilt" in series:
            v = series["pelvis_tilt"][sl]
            out["pelvis_rock"] = float(np.nanmax(v) - np.nanmin(v))
        if "trunk_ml" in series:
            v = series["trunk_ml"][sl]
            out["trunk_sway"] = float(100 * (np.nanmax(v) - np.nanmin(v)))
        if "hip_up" in series:
            v = series["hip_up"][sl]
            out["hip_travel"] = float(100 * (np.nanmax(v) - np.nanmin(v)))
    return out


def phases(strokes_: list[dict], colors: list[str]) -> tuple[list, list]:
    """One phase per stroke ("first" to "last" frame), split at its top into recovery and propulsion."""
    from openkick.core.project import Phase
    main, sub = [], []
    for i, s in enumerate(strokes_, start=1):
        first, last = int(s["first"]), int(s["last"])
        top = max(first + 1, min(last, int(s["top_frame"])))
        main.append(Phase(name=tr("Stroke {n}", n=i), start=first, end=last, color=colors[(i - 1) % len(colors)]))
        sub.append(Phase(name=tr("Recovery"), start=first, end=top - 1, color=COLOR_RECOVERY))
        sub.append(Phase(name=tr("Propulsion"), start=top, end=last, color=COLOR_PROPULSION))
    return main, sub


# ── tables ─────────────────────────────────────────────────────────────

def cycle_table(strokes_: list[dict], keys, title: str) -> "Table":
    from openkick.plugins import Table
    rows = []
    for key in keys:
        v = [s[key] for s in strokes_ if key in s and np.isfinite(s[key])]
        if not v:
            continue
        mid, sd = typical(v)
        rows.append([name(key), show(mid, key), f"± {sd:.1f}", show(min(v), key), show(max(v), key)])
    return Table([tr("Measure"), tr("Typical stroke"), tr("From stroke to stroke"), tr("Lowest"), tr("Highest")],
                 rows, title)


def sides_table(by_side: dict) -> "Table | None":
    """Left leg against right leg."""
    from openkick.plugins import Table
    left, right = by_side.get("left") or [], by_side.get("right") or []
    if len(left) < 2 or len(right) < 2:
        return None
    rows = []
    for key in POSITION + ("knee_lateral", "knee_inward"):
        (a, _), (b, _) = typical(s.get(key) for s in left), typical(s.get(key) for s in right)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        rows.append([name(key), show(a, key), show(b, key), f"{b - a:+.1f}"])
    return Table([tr("Measure"), tr("Left"), tr("Right"), tr("Difference")], rows, tr("Left and right"))


def beyond(a: float, sa: float, b: float, sb: float) -> bool:
    """Is the change from a to b more than what changes from one stroke to the next?"""
    return abs(b - a) > max(2 * max(sa, sb), 1.0)


def compare_table(groups: list[tuple[str, list[dict]]], keys, title: str) -> "Table":
    """Several analyses side by side (each: a name and its strokes): the typical value of each,
    its difference from the first one and whether it is more than the stroke-to-stroke variation."""
    from openkick.plugins import Table
    ref_name, ref = groups[0]
    cols = [tr("Measure")] + [g for g, _ in groups]
    for g, _ in groups[1:]:
        cols.append(tr("{b} − {a}", a=ref_name, b=g))
    rows = []
    for key in keys:
        typ = [typical(s.get(key) for s in st) for _, st in groups]
        if not any(np.isfinite(m) for m, _ in typ):
            continue
        row = [name(key)] + [f"{show(m, key)} ± {sd:.1f}" if np.isfinite(m) else "—" for m, sd in typ]
        (a, sa) = typ[0]
        for (b, sb) in typ[1:]:
            if np.isfinite(a) and np.isfinite(b):
                row.append(f"{b - a:+.1f}{' ●' if beyond(a, sa, b, sb) else ''}")
            else:
                row.append("—")
        rows.append(row)
    return Table(cols, rows, title)
