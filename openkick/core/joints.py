"""Joint angles the way clinicians read them, from a side-on 2D video.

The interior angle between three points (180° = a straight limb) folds over at 180°, so a hip
in extension reads the same as one in flexion. Here every joint is measured with a sign, from a
neutral position of 0°:

  knee      flexion: the shank against the line of the thigh (0 = straight, + = bent)
  hip       flexion: the thigh against the vertical ("vertical"), or against the vertical tipped
            with three quarters of the trunk's lean ("pelvis", standing in for the pelvis, which the
            detector does not see); + = in front of the body, − = behind it
  ankle     dorsiflexion: the sole (heel to toes) against the shank, 0 = foot at a right angle
  shoulder  flexion: the upper arm against the vertical (with "vertical") or against the trunk's
            axis, mid-shoulders to mid-hips (with "pelvis"); + = in front, − = behind
  elbow     flexion: the forearm against the line of the upper arm

Chosen by measuring against the laboratory's OpenSim angles on the OpenCap dataset (see
docs/VALIDATION.md). A template can keep the interior angles ("interior"), as cycling does:
bike fitting is written in them.
"""

from __future__ import annotations

import numpy as np

from openkick.core import landmarks as L

HIP_LEAN = 0.75            # share of the trunk's lean taken as the pelvis's tilt
JOINTS = ("hip", "knee", "ankle", "shoulder", "elbow")
HIP_REFERENCES = ("pelvis", "vertical")


def signed(ref: np.ndarray, v: np.ndarray, forward: np.ndarray | float) -> np.ndarray:
    """Angle (degrees) turning from `ref` to `v`, positive towards the front of the person.
    Image coordinates (y down); forward = +1 when the person faces image-right, −1 left."""
    cross = ref[..., 0] * v[..., 1] - ref[..., 1] * v[..., 0]
    dot = (ref * v).sum(-1)
    with np.errstate(invalid="ignore"):
        out = np.degrees(np.arctan2(-forward * cross, dot))
    out[(np.linalg.norm(ref, axis=-1) < 1e-6) | (np.linalg.norm(v, axis=-1) < 1e-6)] = np.nan
    return out


def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def facing(xy: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Per frame, +1 when the person faces image-right and −1 when image-left, from where the
    toes are against the heels (both feet, over about a second, so a turn is followed)."""
    import warnings

    from scipy.ndimage import median_filter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # frames with no feet
        d = np.nanmean(np.stack([xy[:, L.INDEX[f"{s}_foot_index"], 0] - xy[:, L.INDEX[f"{s}_heel"], 0]
                                 for s in ("left", "right")]), axis=0) if len(xy) else np.zeros(0)
    ok = np.isfinite(d)
    whole = 1.0 if not ok.any() or np.median(d[ok]) >= 0 else -1.0
    if not ok.any():
        return np.full(len(xy), whole)
    k = np.arange(len(d))
    d = np.interp(k, k[ok], d[ok])
    d = median_filter(d, size=max(3, int(fps) | 1), mode="nearest")
    return np.where(d > 0, 1.0, np.where(d < 0, -1.0, whole))


def joint_of(a: int, o: int, b: int) -> tuple[str, str] | None:
    """(joint, side) when the three landmarks are one of the standard joint angles, else None."""
    preset = L.ANGLE_PRESETS.get(o)
    if preset is None or (L.INDEX[preset[0]], L.INDEX[preset[2]]) not in ((a, b), (b, a)):
        return None
    side, joint = L.NAMES[o].split("_", 1)
    return (joint, side) if joint in JOINTS else None


def trunk_lean(xy: np.ndarray, forward) -> np.ndarray:
    """The trunk's lean from the vertical (mid-hips to mid-shoulders), + = forward."""
    I = L.INDEX
    mid_hip = (xy[:, I["left_hip"]] + xy[:, I["right_hip"]]) / 2
    mid_sh = (xy[:, I["left_shoulder"]] + xy[:, I["right_shoulder"]]) / 2
    up = np.broadcast_to(np.array([0.0, -1.0]), mid_hip.shape)
    return -signed(up, mid_sh - mid_hip, forward)


def references(xy: np.ndarray, joint: str, side: str, forward, hip: str = "pelvis") -> tuple[np.ndarray, np.ndarray]:
    """(reference direction, moving segment) for each frame; the angle is the turn from one to
    the other. Used both to measure and to draw the arc."""
    P = {n: xy[:, L.INDEX[f"{side}_{n}"]] for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle",
                                                     "heel", "foot_index")}
    if joint == "knee":
        return P["knee"] - P["hip"], P["ankle"] - P["knee"]
    if joint == "elbow":
        return P["elbow"] - P["shoulder"], P["wrist"] - P["elbow"]
    if joint == "shoulder":
        if hip == "vertical":
            down = np.broadcast_to(np.array([0.0, 1.0]), P["shoulder"].shape)
        else:                                   # the trunk's axis, from the mid-shoulders to the mid-hips
            I = L.INDEX
            down = (xy[:, I["left_hip"]] + xy[:, I["right_hip"]] - xy[:, I["left_shoulder"]] - xy[:, I["right_shoulder"]]) / 2
        return down, P["elbow"] - P["shoulder"]
    if joint == "hip":
        down = np.broadcast_to(np.array([0.0, 1.0]), P["hip"].shape).copy()
        if hip == "pelvis":
            tilt = np.radians(HIP_LEAN * np.nan_to_num(trunk_lean(xy, forward)))
            f = np.broadcast_to(forward, tilt.shape)
            # leaning forward tips the line down the trunk backwards: the vertical turned back by
            # the tilt (forward is +x when facing right), so the thigh reads more flexed
            down = np.stack([-f * np.sin(tilt), np.cos(tilt)], axis=-1)
        return down, P["knee"] - P["hip"]
    if joint == "ankle":
        shank_up = P["knee"] - P["ankle"]
        f = np.broadcast_to(forward, shank_up.shape[:1])[:, None]
        # the foot at a right angle to the shank, pointing forward: the shank turned by 90°
        neutral = np.stack([-shank_up[:, 1], shank_up[:, 0]], axis=-1)
        neutral = np.where(f * neutral[:, :1] >= 0, neutral, -neutral)
        return neutral, P["foot_index"] - P["heel"]
    raise ValueError(joint)


def clinical(xy: np.ndarray, joint: str, side: str, forward, hip: str = "pelvis") -> np.ndarray:
    """Per-frame angle of one joint (degrees), in the convention described at the top."""
    I = L.INDEX
    if joint == "ankle":
        sole = xy[:, I[f"{side}_foot_index"]] - xy[:, I[f"{side}_heel"]]
        return 90.0 - between(xy[:, I[f"{side}_knee"]] - xy[:, I[f"{side}_ankle"]], sole)
    ref, seg = references(xy, joint, side, forward, hip)
    if joint == "knee":
        return signed(ref, seg, -np.asarray(forward))
    if joint == "elbow":
        return signed(ref, seg, np.asarray(forward))
    return signed(ref, seg, forward)


# ── 3D ─────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def pelvis_axes(X: np.ndarray, idx: dict, up: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An estimate of the pelvis per row of a 3D skeleton: (to the right, up, forward). Right is
    from the left hip to the right one; up is the room's vertical turned towards the trunk by
    `alpha` (0 = the vertical, 1 = the trunk), both square to the right."""
    rh, lh = X[:, idx["right_hip"]], X[:, idx["left_hip"]]
    neck = (X[:, idx["right_shoulder"]] + X[:, idx["left_shoulder"]]) / 2
    right = _unit(rh - lh)

    def square(v):
        return _unit(v - (v * right).sum(-1, keepdims=True) * right)
    trunk = square(neck - (rh + lh) / 2)
    vertical = square(np.broadcast_to(np.asarray(up, np.float64), right.shape))
    up_p = _unit((1 - alpha) * vertical + alpha * trunk)
    fwd = np.cross(up_p, right)
    toes = sum(X[:, idx[f"{s}_toe"]] - X[:, idx[f"{s}_heel"]] for s in ("left", "right"))
    with np.errstate(invalid="ignore"):
        if np.nanmedian((fwd * toes).sum(-1)) < 0:
            fwd = -fwd
    return right, up_p, fwd


def clinical_3d(X: np.ndarray, idx: dict, joint: str, side: str, up: np.ndarray, hip: str = "pelvis") -> np.ndarray:
    """Per-row angle of one joint of a 3D skeleton (metres), in the convention at the top: hip
    flexion in an estimate of the pelvis (the vertical for "vertical", tipped three quarters of
    the way to the trunk for "pelvis"), knee and elbow flexion, ankle dorsiflexion with the sole
    from heel to toes, shoulder elevation from the side of the trunk."""
    P = {n: X[:, idx[f"{side}_{n}"]] for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "toe")}
    if joint == "hip":
        _, up_p, fwd = pelvis_axes(X, idx, up, HIP_LEAN if hip == "pelvis" else 0.0)
        thigh = _unit(P["knee"] - P["hip"])
        with np.errstate(invalid="ignore"):
            return np.degrees(np.arctan2((thigh * fwd).sum(-1), -(thigh * up_p).sum(-1)))
    if joint == "knee":
        return 180.0 - between(P["hip"] - P["knee"], P["ankle"] - P["knee"])
    if joint == "elbow":
        return 180.0 - between(P["shoulder"] - P["elbow"], P["wrist"] - P["elbow"])
    if joint == "ankle":
        return 90.0 - between(P["knee"] - P["ankle"], P["toe"] - P["heel"])
    if joint == "shoulder":
        return between(P["hip"] - P["shoulder"], P["elbow"] - P["shoulder"])
    raise ValueError(joint)
