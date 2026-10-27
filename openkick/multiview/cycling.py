"""Pedal strokes from the 3D reconstruction of several videos.

The same strokes and measures as the 2D plugin (openkick.core.cycling), measured in 3D: the
angles are those of the real limbs, not of their shadow on one picture, and both legs are
measured, not only the one turned to a camera. 3D also shows what a side view cannot: the knee
moving across the bike, the pelvis rocking and the trunk swaying.

Directions: "up" is where the phones' own vertical points (on tripods they stand upright,
and a rider on a trainer never stands up straight to show it otherwise); "across" is the line
from the right hip to the left one, made level. Lengths are in metres.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from openkick.core import cycling as C
from openkick.core.analysis import angle_between, lowpass
from openkick.i18n import tr
from openkick.multiview.geometry import IDX

MIN_STROKES = 5


@dataclass
class Cycling:
    side: str                                         # the leg whose strokes are the phases
    by_side: dict                                     # "left"/"right" -> strokes
    strokes: list[dict]                               # that leg's strokes
    phases: list = field(default_factory=list)        # one per stroke (rows)
    subphases: list = field(default_factory=list)     # recovery and propulsion of each
    result: object = None                             # openkick.plugins.Result with every table
    up: np.ndarray | None = None
    series: dict = field(default_factory=dict)        # "left"/"right" -> {"knee", "hip", "ankle"}: per-row angles
    t: np.ndarray | None = None

    def phase_at(self, row: int):
        return next((p for p in self.phases if p.start <= row <= p.end), None)

    def subphase_at(self, row: int):
        return next((p for p in self.subphases if p.start <= row <= p.end), None)


def _unit(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def vertical(cams: list) -> np.ndarray:
    """Up, in the reconstruction's axes (those of camera 1), from the phones' own vertical."""
    downs = np.array([np.asarray(R, float).T @ np.array([0.0, 1.0, 0.0]) for R, _ in cams])
    return -_unit(np.median(downs, axis=0))


def looks_like_pedalling(X: np.ndarray, cams: list) -> bool:
    """Is this someone pedalling rather than walking? Against the hip, a pedalling ankle goes round
    a circle, about as tall as it is long; a walking ankle sweeps a long, low arc."""
    up = vertical(cams)
    ratios = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side in ("left", "right"):
            rel = X[:, IDX[f"{side}_ankle"]] - X[:, IDX[f"{side}_hip"]]
            rel = rel[np.isfinite(rel).all(axis=1)]
            if len(rel) < 30:
                continue
            height = rel @ up
            flat = rel - height[:, None] * up
            flat = flat - flat.mean(axis=0)
            along = flat @ np.linalg.svd(flat, full_matrices=False)[2][0]       # the main horizontal direction
            tall = np.percentile(height, 95) - np.percentile(height, 5)
            long = np.percentile(along, 95) - np.percentile(along, 5)
            if long > 0:
                ratios.append(tall / long)
    return bool(ratios) and float(np.median(ratios)) > 0.5


def analyse(X: np.ndarray, t: np.ndarray, cams: list, quality: dict | None = None) -> Cycling | None:
    """Strokes of both legs in a reconstruction (X: rows × 26 points × 3, metres; t: seconds).
    None when nobody pedals in it."""
    from openkick.core.project import PHASE_COLORS
    if len(t) < 10:
        return None
    fps = 1.0 / float(np.median(np.diff(t)))
    X = np.asarray(X, np.float64)
    up = vertical(cams)

    def P(n):
        return X[:, IDX[n]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lh, rh = P("left_hip"), P("right_hip")
        hips = (lh + rh) / 2
        shoulders = (P("left_shoulder") + P("right_shoulder")) / 2
        across = np.nanmedian(lh - rh, axis=0)
        across = _unit(across - (across @ up) * up)                    # level, from right to left
        trunk_v = _unit(shoulders - hips)
        shared = {
            "trunk": np.degrees(np.arcsin(np.clip(trunk_v @ up, -1, 1))),
            "pelvis_tilt": lowpass(np.degrees(np.arcsin(np.clip(_unit(lh - rh) @ up, -1, 1))), fps),
            "trunk_ml": lowpass((shoulders - hips) @ across, fps),
            "hip_up": lowpass(hips @ up, fps),
        }
        by_side, curves = {}, {}
        for side in ("left", "right"):
            hip, knee, ankle = P(f"{side}_hip"), P(f"{side}_knee"), P(f"{side}_ankle")
            toe, heel = P(f"{side}_toe"), P(f"{side}_heel")
            knee_a = lowpass(angle_between(hip, knee, ankle), fps)
            inward = 1.0 if side == "right" else -1.0                 # towards the bike: to the other side
            series = dict(shared,
                          hip=lowpass(angle_between(P(f"{side}_shoulder"), hip, knee), fps),
                          ankle=lowpass(angle_between(knee, ankle, toe), fps),
                          foot=lowpass(-np.degrees(np.arcsin(np.clip(_unit(toe - heel) @ up, -1, 1))), fps),
                          knee_ml=lowpass(inward * ((knee - (hip + ankle) / 2) @ across), fps))
            curves[side] = {"knee": knee_a, "hip": series["hip"], "ankle": series["ankle"]}
            found = []
            for s in C.strokes(knee_a, t, fps):
                s.update(C.per_stroke(s, series))
                s.update({"side": side, "first": s["a"], "last": s["b"] - 1, "top_frame": s["top"],
                          "time": float(t[s["a"]] - t[0])})
                found.append(s)
            by_side[side] = found
    side = max(("right", "left"), key=lambda k: len(by_side[k]))
    strokes = by_side[side]
    if len(strokes) < MIN_STROKES:
        return None
    phases, subphases = C.phases(strokes, PHASE_COLORS)
    c = Cycling(side=side, by_side=by_side, strokes=strokes, phases=phases, subphases=subphases, up=up,
                series=curves, t=np.asarray(t, float))
    c.result = _report(c, quality)
    return c


def _report(c: Cycling, quality: dict | None):
    from openkick.plugins import Result, Table
    every = c.by_side["left"] + c.by_side["right"]
    tables = [C.cycle_table(c.strokes, C.POSITION, tr("Position")),
              C.cycle_table(c.strokes, C.DYNAMICS, tr("Pedalling"))]
    sides = C.sides_table(c.by_side)
    if sides:
        tables.append(sides)
    shown = ("cadence", "knee_bottom", "knee_top", "ankle_bottom", "knee_lateral", "pelvis_rock")
    rows = [[n, f"{s['time']:.2f} s"] + [C.show(s.get(k), k) for k in shown] for n, s in enumerate(c.strokes, start=1)]
    tables.append(Table([tr("Stroke"), tr("At")] + [C.name(k) for k in shown], rows, tr("Stroke by stroke")))
    rpm, spread = C.typical(s["cadence"] for s in c.strokes)
    leg = tr("Right leg") if c.side == "right" else tr("Left leg")
    summary = [tr("{leg} · {n} pedal strokes at {rpm} rpm (± {sd}) · {m} strokes of both legs measured in 3D.",
                  leg=leg, n=len(c.strokes), rpm=f"{rpm:.0f}", sd=f"{spread:.1f}", m=len(every)),
               tr("Angles are measured inside the joint, so 180° is a straight limb. Up is where the phones' "
                  "own vertical points: on tripods they should stand upright.")]
    if quality and quality.get("level") == "poor":
        summary.append(tr("The 3D reconstruction is of poor quality: correct the doubtful points before "
                          "relying on these numbers."))
    return Result(title=tr("Pedal stroke analysis in 3D"), summary="\n".join(summary), tables=tables)
