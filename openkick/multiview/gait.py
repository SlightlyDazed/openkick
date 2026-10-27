"""Gait phases and measures from the 3D reconstruction of several videos.

The same events as the 2D gait plugin (openkick.core.gait), found along the direction of walking
in 3D: the heel furthest ahead of the pelvis at initial contact, the toes furthest behind it at
toe off. The vertical is the one the 3D view uses (feet to head where the person stands tallest)
and the direction of walking comes from the feet, so the cameras may stand anywhere, and every
length is in metres. Rows are those of the reconstruction (frames of the first video)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from openkick.core import gait as G
from openkick.core.analysis import angle_between, lowpass
from openkick.i18n import tr
from openkick.multiview.geometry import IDX

KEYS = ("stride_time", "cadence", "stance", "swing", "step_length", "step_width", "stride_length", "speed",
        "knee_contact", "knee_swing", "knee_rom", "hip_rom", "ankle_rom", "trunk")
HEADING_HZ = 0.5          # the direction of walking changes slower than this (it may turn round)


@dataclass
class Gait:
    side: str                                         # the leg whose strides are the phases
    strides: list[dict]                               # that leg's strides
    every: list[dict]                                 # both legs'
    phases: list = field(default_factory=list)        # one per stride (rows)
    subphases: list = field(default_factory=list)     # stance and swing of each
    result: object = None                             # openkick.plugins.Result with every table

    def phase_at(self, row: int):
        return next((p for p in self.phases if p.start <= row <= p.end), None)

    def subphase_at(self, row: int):
        return next((p for p in self.subphases if p.start <= row <= p.end), None)


def _unit(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=-1)


def analyse(X: np.ndarray, t: np.ndarray) -> Gait | None:
    """Strides of both legs in a reconstruction (X: rows × 26 points × 3, metres; t: seconds).
    None when there is no walking to be found."""
    from openkick.core.project import PHASE_COLORS
    if len(t) < 10:
        return None
    fps = 1.0 / float(np.median(np.diff(t)))
    X = np.asarray(X, np.float64)

    def P(name):
        return X[:, IDX[name]]
    from openkick.multiview.export import to_world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pelvis = np.where(np.isfinite(P("hip")), P("hip"), (P("left_hip") + P("right_hip")) / 2)
        trunk = P("neck") - pelvis
        if np.isfinite(X).all(axis=2).sum() == 0:
            return None
        up = to_world(X)[1]["R"][1]                   # the same vertical as the 3D view: feet to head, standing tallest

        def flat(v):                                  # the horizontal part of a vector
            return v - _dot(v, up)[..., None] * up
        feet = sum(_unit(flat(P(f"{s}_toe") - P(f"{s}_heel"))) for s in ("left", "right"))
        heading = _unit(flat(np.stack([lowpass(feet[:, k], fps, cutoff=HEADING_HZ) for k in range(3)], axis=1)))
        if not np.isfinite(heading).any():
            return None
        across = np.cross(up, heading)
        legs = [np.linalg.norm(P(f"{s}_hip") - P(f"{s}_ankle"), axis=1) for s in ("left", "right")]
        leg = float(np.nanmedian(np.concatenate(legs)))
        lean = lowpass(np.degrees(np.arctan2(_dot(trunk, heading), _dot(trunk, up))), fps)
    if not np.isfinite(leg) or leg <= 0:
        return None

    # each foot's events and positions, for the eight phases (gait_events needs the other foot's too):
    # "down" stands for the image's downward axis of the 2D plugin, "_x" for the way of walking
    feet = {}
    for side in ("left", "right"):
        heel, toe = P(f"{side}_heel"), P(f"{side}_toe")
        feet[side] = {"contacts": [f for f, _ in G.events(lowpass(_dot(heel - pelvis, heading), fps), fps, leg)],
                      "offs": [f for f, _ in G.events(lowpass(-_dot(toe - pelvis, heading), fps), fps, leg)],
                      "heel_y": lowpass(-_dot(heel, up[None, :].repeat(len(heel), 0)), fps),
                      "ankle_x": lowpass(_dot(P(f"{side}_ankle"), heading), fps),
                      "knee_x": lowpass(_dot(P(f"{side}_knee"), heading), fps)}
    every = []
    for side in ("left", "right"):
        other = "right" if side == "left" else "left"
        heel, toe = P(f"{side}_heel"), P(f"{side}_toe")
        contacts = G.events(lowpass(_dot(heel - pelvis, heading), fps), fps, leg)
        offs = G.events(lowpass(-_dot(toe - pelvis, heading), fps), fps, leg)
        if len(contacts) < 2 or not offs:
            continue
        hip, knee, ankle = P(f"{side}_hip"), P(f"{side}_knee"), P(f"{side}_ankle")
        knee_a = lowpass(angle_between(hip, knee, ankle), fps)
        hip_a = lowpass(angle_between(P(f"{side}_shoulder"), hip, knee), fps)
        ankle_a = lowpass(angle_between(knee, ankle, toe), fps)
        for s in G.pair(contacts, offs, t, np.isfinite(knee_a)):
            a, b = s["a"], s["b"]
            gap = heel[a] - P(f"{other}_heel")[a]          # both feet on the ground at contact
            travel = float(np.linalg.norm(flat(pelvis[b] - pelvis[a])))
            stride = float(np.linalg.norm(flat(heel[b] - heel[a]))) if travel >= G.TRAVEL * leg else float("nan")
            found = G.gait_events(s, feet[side], feet[other], 1.0, leg)
            every.append({"side": side, "first": a, "last": b - 1, "off": s["o"], "time": float(t[a]),
                          "events": found["events"], "estimated": found["estimated"],
                          **G.joint_measures(s, knee_a, hip_a, ankle_a, lean),
                          "step_length": float(_dot(gap, heading[a])),
                          "step_width": 100 * abs(float(_dot(gap, across[a]))),
                          "stride_length": stride, "speed": stride / s["seconds"]})
    left = [s for s in every if s["side"] == "left"]
    right = [s for s in every if s["side"] == "right"]
    side, strides = ("left", left) if len(left) > len(right) else ("right", right)
    if len(strides) < 2:
        return None
    strides.sort(key=lambda s: s["first"])
    phases, subphases = G.phases(strides, PHASE_COLORS)
    g = Gait(side=side, strides=strides, every=every, phases=phases, subphases=subphases)
    g.result = _report(g, X, t)
    return g


def _report(g: Gait, X: np.ndarray, t: np.ndarray):
    from openkick.plugins import Result, Table
    ok, review = tr("OK"), tr("Review")
    feet = [IDX[f"{s}_{k}"] for s in ("left", "right") for k in ("heel", "toe", "ankle")]
    seen = float(np.isfinite(X[:, feet]).all(axis=(1, 2)).mean())
    mid, _ = G.typical(s["stride_time"] for s in g.strides)
    per = mid / float(np.median(np.diff(t)))
    checks = Table([tr("Check"), tr("Found"), tr("Verdict")], [
        [tr("Feet reconstructed"), f"{seen * 100:.0f} %", ok if seen >= 0.9 else review],
        [tr("Strides"), str(len(g.strides)), ok if len(g.strides) >= 3 else review],
        [tr("Frames in each stride"), f"{per:.0f}", ok if per >= 25 else review]], tr("Recording check"))
    tables = [G.cycle_table(g.strides, KEYS)]
    eight = G.phase_table(g.strides)
    if eight:
        tables.append(eight)
    sides = G.sides_table(g.every)
    if sides:
        tables.append(sides)
    tables += [checks, G.stride_table(g.strides)]
    cadence, _ = G.typical(s["cadence"] for s in g.strides)
    stance, _ = G.typical(s["stance"] for s in g.strides)
    summary = [
        tr("{side} · {n} strides · {c} steps/min · stance {s} % of the stride.",
           side=tr("Left leg") if g.side == "left" else tr("Right leg"), n=len(g.strides),
           c=f"{cadence:.0f}", s=f"{stance:.0f}"),
        tr("Found in 3D, so the cameras may stand anywhere: the direction of walking is taken from the feet "
           "and every length is in metres, scaled from the height of the person. Initial contact and toe off "
           "come from where the heel and the toes are against the pelvis (Zeni et al., 2008). Angles are "
           "measured inside the joint, so 180° is a straight limb; the measures marked * are the least reliable.")]
    return Result(title=tr("Gait analysis"), summary="\n".join(summary), tables=tables)
