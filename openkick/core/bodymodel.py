"""Body segments, derived points (center of mass) and point-to-segment attachment.

Center of mass: segment masses and CoM locations from de Leva (1996),
"Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters",
J. Biomech. 29(9):1223-1230. MediaPipe landmarks approximate the anatomical
endpoints (e.g. mid-ears for the head CoM, mid-shoulders for C7), so the result
is a 2D estimate suited to comparing movements, not a lab-grade measurement.
"""

from __future__ import annotations

import math

import numpy as np

from openkick.core import landmarks as L

I = L.INDEX

# Segments a user marker can be attached to: (proximal, distal) -> display name (English source).
SEGMENTS = {
    (I["right_shoulder"], I["right_elbow"]): "Right upper arm",
    (I["right_elbow"], I["right_wrist"]): "Right forearm",
    (I["left_shoulder"], I["left_elbow"]): "Left upper arm",
    (I["left_elbow"], I["left_wrist"]): "Left forearm",
    (I["right_wrist"], I["right_index"]): "Right hand",
    (I["left_wrist"], I["left_index"]): "Left hand",
    (I["right_hip"], I["right_knee"]): "Right thigh",
    (I["right_knee"], I["right_ankle"]): "Right shank",
    (I["left_hip"], I["left_knee"]): "Left thigh",
    (I["left_knee"], I["left_ankle"]): "Left shank",
    (I["right_heel"], I["right_foot_index"]): "Right foot",
    (I["left_heel"], I["left_foot_index"]): "Left foot",
    (I["right_shoulder"], I["right_hip"]): "Trunk (right side)",
    (I["left_shoulder"], I["left_hip"]): "Trunk (left side)",
    (I["right_shoulder"], I["left_shoulder"]): "Shoulders",
    (I["right_hip"], I["left_hip"]): "Pelvis",
    (I["nose"], I["right_ear"]): "Head",
    (I["nose"], I["left_ear"]): "Head",
}

# (mass fraction, CoM fraction from proximal) — de Leva 1996
DE_LEVA = {
    "m": {"head": (0.0694, None), "trunk": (0.4346, 0.4486), "upper_arm": (0.0271, 0.5772),
          "forearm": (0.0162, 0.4574), "hand": (0.0061, 0.7900), "thigh": (0.1416, 0.4095),
          "shank": (0.0433, 0.4459), "foot": (0.0137, 0.4415)},
    "f": {"head": (0.0668, None), "trunk": (0.4257, 0.4964), "upper_arm": (0.0255, 0.5754),
          "forearm": (0.0138, 0.4559), "hand": (0.0056, 0.7474), "thigh": (0.1478, 0.3612),
          "shank": (0.0481, 0.4416), "foot": (0.0129, 0.4014)},
}

DERIVED = {"com:m": "Center of mass (male model)", "com:f": "Center of mass (female model)"}


def _mid(xy, a, b):
    return (xy[..., a, :] + xy[..., b, :]) / 2


def center_of_mass(xy: np.ndarray, sex: str = "m") -> np.ndarray:
    """Whole-body CoM for pose arrays (..., 33, 2). NaN where a segment is missing."""
    p = DE_LEVA[sex]
    parts = []

    def seg(prox, dist, key):
        m, c = p[key]
        parts.append((m, prox + (dist - prox) * c))

    parts.append((p["head"][0], _mid(xy, I["left_ear"], I["right_ear"])))
    seg(_mid(xy, I["left_shoulder"], I["right_shoulder"]), _mid(xy, I["left_hip"], I["right_hip"]), "trunk")
    for side in ("left", "right"):
        s = lambda n: xy[..., I[f"{side}_{n}"], :]  # noqa: E731
        seg(s("shoulder"), s("elbow"), "upper_arm")
        seg(s("elbow"), s("wrist"), "forearm")
        seg(s("wrist"), s("index"), "hand")
        seg(s("hip"), s("knee"), "thigh")
        seg(s("knee"), s("ankle"), "shank")
        seg(s("heel"), s("foot_index"), "foot")
    total = sum(m for m, _ in parts)
    return sum(m * pos for m, pos in parts) / total


def derived_point(name: str, xy_frame: np.ndarray) -> tuple[float, float] | None:
    if name.startswith("com:"):
        c = center_of_mass(xy_frame, name.split(":")[1])
        if not np.isfinite(c).all():
            return None
        return float(c[0]), float(c[1])
    return None


def derived_series(name: str, xy: np.ndarray) -> np.ndarray:
    if name.startswith("com:"):
        return center_of_mass(xy.astype(np.float64), name.split(":")[1])
    return np.full((xy.shape[0], 2), np.nan)


def nearest_segment(xy_frame: np.ndarray, p: tuple[float, float]):
    """(a, b, distance_px) of the body segment closest to point p in this frame."""
    best = None
    for (a, b) in SEGMENTS:
        pa, pb = xy_frame[a], xy_frame[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        ux, uy = pb - pa
        l2 = ux * ux + uy * uy
        if l2 < 1:
            continue
        t = max(0.0, min(1.0, ((p[0] - pa[0]) * ux + (p[1] - pa[1]) * uy) / l2))
        d = math.hypot(p[0] - (pa[0] + t * ux), p[1] - (pa[1] + t * uy))
        if best is None or d < best[2]:
            best = (a, b, d)
    return best
