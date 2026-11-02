"""Biomechanical calculations on a Project: angle series, speeds, phase statistics,
smoothing and point-edit propagation. Pure numpy/scipy, no GUI."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from openkick.core.project import (SRC_AUTO, SRC_MANUAL, SRC_SMOOTHED, Measurement, Phase, Pose,
                                   Project)

# ── series ─────────────────────────────────────────────────────────────


def times(project: Project) -> np.ndarray:
    v = project.video
    return np.array([v.time_of(i) for i in range(v.frame_count)], np.float64)


def _points(project: Project, ref, n: int, _depth: int = 0) -> np.ndarray:
    """Per-frame positions (n, 2) of any kind of point reference."""
    k = ref.kind
    xy = project.pose.xy
    if k == "landmark":
        return xy[:, ref.landmark, :].astype(np.float64)
    if k == "segment":
        a, b, al, ac = ref.segment
        pa, pb = xy[:, a, :].astype(np.float64), xy[:, b, :].astype(np.float64)
        u = pb - pa
        return np.stack([pa[:, 0] + al * u[:, 0] - ac * u[:, 1], pa[:, 1] + al * u[:, 1] + ac * u[:, 0]], axis=1)
    if k == "derived":
        from openkick.core.bodymodel import derived_series
        return derived_series(ref.derived, xy)
    if k == "marker":
        m = project.measurement(ref.marker)
        if m is None or _depth > 4:
            return np.full((n, 2), np.nan)
        return _points(project, m.points[0], n, _depth + 1)
    out = np.full((n, 2), np.nan)
    if ref.keys:
        frames = np.array(sorted(ref.keys))
        vals = np.array([ref.keys[f] for f in frames], np.float64)
        idx = np.arange(n)
        out[:, 0] = np.interp(idx, frames, vals[:, 0])
        out[:, 1] = np.interp(idx, frames, vals[:, 1])
    return out


def angle_between(a: np.ndarray, o: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Interior angle at vertex `o` (degrees, 0-180) for arrays of shape (N, 2)."""
    v1 = a - o
    v2 = b - o
    dot = (v1 * v2).sum(axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
    ang[(n1 < 1e-6) | (n2 < 1e-6)] = np.nan
    return ang


def lowpass(values: np.ndarray, fps: float, cutoff: float = 6.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (the biomechanics standard). NaN gaps are
    bridged for filtering and restored afterwards."""
    from scipy.signal import butter, filtfilt
    values = np.asarray(values, np.float64)
    out = values.copy()
    finite = np.isfinite(values)
    if finite.sum() < 3 * (order + 1) + 1 or cutoff >= fps / 2:
        return out
    idx = np.arange(len(values))
    filled = np.interp(idx, idx[finite], values[finite])
    b, a = butter(order, cutoff / (fps / 2), btype="low")
    out = filtfilt(b, a, filled)
    out[~finite] = np.nan
    return out


# A detector jump (a point that leaves for a frame or two and comes back) becomes a false peak of
# speed. Before differentiating, values further than 3 robust SDs from the median of the 0.2 s
# around them are dropped (a Hampel filter), but only when that is more than 5° for an angle or
# 2 % of the leg for a point, so a fast real movement is kept. Against the OpenCap lab (2D,
# walking) it took the knee's angular speed error from 114 to 89 °/s and the ankle's speed error
# from 0.70 to 0.56 m/s; in squats, sit-to-stand and drop jumps it changed by less than 1 °/s.
DESPIKE_S = 0.2
DESPIKE_SD = 3.0
DESPIKE_FLOOR_DEG = 5.0
DESPIKE_FLOOR_LEG = 0.02


def despike(values: np.ndarray, fps: float, floor: float) -> np.ndarray:
    """`values` with the detector's jumps set to NaN (see DESPIKE_S)."""
    from scipy.ndimage import median_filter
    x = np.asarray(values, np.float64).copy()
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return x
    idx = np.arange(len(x))
    y = np.interp(idx, idx[ok], x[ok])
    size = 2 * max(1, int(DESPIKE_S * fps)) + 1
    med = median_filter(y, size=size, mode="nearest")
    spread = 1.4826 * median_filter(np.abs(y - med), size=size, mode="nearest")
    x[ok & (np.abs(y - med) > DESPIKE_SD * np.maximum(spread, floor))] = np.nan
    return x


def leg_length(project: Project) -> float:
    """Median hip-to-ankle distance of the skeleton, in pixels (NaN without one)."""
    import warnings

    from openkick.core.landmarks import INDEX
    cache = _cache(project)
    if "_leg" not in cache:
        xy = project.pose.xy.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cache["_leg"] = float(np.nanmedian(np.concatenate(
                [np.linalg.norm(xy[:, INDEX[f"{s}_hip"]] - xy[:, INDEX[f"{s}_ankle"]], axis=1)
                 for s in ("left", "right")]))) if len(xy) else float("nan")
    return cache["_leg"]


def _cache(project: Project) -> dict:
    c = getattr(project, "_analysis_cache", None)
    if c is None or c.get("_rev") != project.revision:
        c = {"_rev": project.revision}
        project._analysis_cache = c
    return c


@dataclass
class Series:
    unit: str
    value: np.ndarray                 # per-frame main value (angle, speed or distance)
    xy: np.ndarray | None = None      # trajectories: per-frame positions
    rate: np.ndarray | None = None    # angles: angular velocity (deg/s)
    accel: np.ndarray | None = None   # trajectories: size of the acceleration (unit/s)


def series(project: Project, m: Measurement) -> Series:
    cache = _cache(project)
    if m.id in cache:
        return cache[m.id]
    n = project.frame_count
    t = times(project)
    fps = project.video.fps
    scale = project.calibration.px_per_m if project.calibration else None
    if m.kind == "angle":
        val = _clinical(project, m)
        if val is None:
            a, o, b = (_points(project, r, n) for r in m.points)
            val = angle_between(a, o, b)
        rate = _gradient(lowpass(despike(val, fps, DESPIKE_FLOOR_DEG), fps, cutoff(project)), t)
        s = Series("°", val, rate=rate)
    elif m.kind in ("trajectory", "point"):
        p = _points(project, m.points[0], n)
        leg = leg_length(project)
        floor = DESPIKE_FLOOR_LEG * leg if np.isfinite(leg) else 0.0
        fc = cutoff(project)
        if floor > 0:
            smooth = np.stack([lowpass(despike(p[:, k], fps, floor), fps, fc) for k in range(2)], axis=1)
        else:
            smooth = np.stack([lowpass(p[:, 0], fps, fc), lowpass(p[:, 1], fps, fc)], axis=1)
        vx, vy = _gradient(smooth[:, 0], t), _gradient(smooth[:, 1], t)
        speed = np.hypot(vx, vy)
        # the acceleration is the second derivative of the position (its size), not the change of
        # the speed: a point going round a circle at a steady speed is accelerating
        accel = np.hypot(_gradient(vx, t), _gradient(vy, t))
        unit = "px/s"
        if scale:
            speed, accel = speed / scale, accel / scale
            unit = "m/s"
        s = Series(unit, speed, xy=p, accel=accel)
    elif m.kind == "distance":
        p, q = (_points(project, r, n) for r in m.points)
        d = np.linalg.norm(p - q, axis=1)
        unit = "px"
        if scale:
            d = d / scale
            unit = "m"
        s = Series(unit, d)
    else:
        raise ValueError(m.kind)
    cache[m.id] = s
    return s


# Cutoff of the low-pass filter before speeds, in Hz. 6 Hz suits walking, squats and pedalling
# (Winter); a fast movement loses its peaks to it: against the OpenCap lab, the drop jump's peak
# knee angular speed came out 25 % low, and a kicking foot at 20 m/s (simulated) reads 14 m/s.
# A template can ask for more (`cutoff_hz`); it stays below the video's Nyquist frequency.
CUTOFF_HZ = 6.0


def cutoff(project: Project) -> float:
    """The low-pass cutoff for this project's speeds (Hz), from its template."""
    cache = _cache(project)
    if "_cutoff" not in cache:
        from openkick.core.templates import find
        t = find(project.template)
        fc = (t.cutoff_hz if t is not None and t.cutoff_hz else CUTOFF_HZ)
        cache["_cutoff"] = float(min(fc, 0.45 * project.video.fps)) if project.video.fps else float(fc)
    return cache["_cutoff"]


def convention(project: Project) -> tuple[str, str]:
    """How the project's standard joint angles are read: ("clinical" or "interior", hip
    reference), from its template (see core/joints.py)."""
    cache = _cache(project)
    if "_convention" not in cache:
        from openkick.core.templates import find
        t = find(project.template)
        cache["_convention"] = ("clinical", "pelvis") if t is None else (t.convention or "clinical",
                                                                            t.hip_reference or "pelvis")
    return cache["_convention"]


def joint(m: Measurement) -> tuple[str, str] | None:
    """(joint, side) when an angle is one of the standard joints on the skeleton, else None."""
    from openkick.core.joints import joint_of
    if m.kind != "angle" or len(m.points) != 3 or not all(r.kind == "landmark" for r in m.points):
        return None
    return joint_of(*(r.landmark for r in m.points))


def facing(project: Project) -> np.ndarray:
    """Per frame, +1 when the person faces image-right, −1 when image-left."""
    from openkick.core.joints import facing as _facing
    cache = _cache(project)
    if "_facing" not in cache:
        cache["_facing"] = _facing(project.pose.xy.astype(np.float64), project.video.fps)
    return cache["_facing"]


def _clinical(project: Project, m: Measurement) -> np.ndarray | None:
    """A standard joint angle in the clinical convention, or None to measure it inside the joint."""
    from openkick.core.joints import clinical
    found = joint(m)
    style, hip = convention(project)
    if found is None or style == "interior":
        return None
    name, side = found
    return clinical(project.pose.xy.astype(np.float64), name, side, facing(project), hip)


def _gradient(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(v) < 2:
        return np.full_like(v, np.nan)
    with np.errstate(invalid="ignore"):
        return np.gradient(v, t)


def value_at(project: Project, m: Measurement, frame: int) -> float | None:
    v = series(project, m).value[frame]
    return None if not np.isfinite(v) else float(v)


def format_value(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "°":
        return f"{v:.0f}°"
    if unit in ("m/s", "m"):
        return f"{v:.2f} {unit}"
    return f"{v:.0f} {unit}"


# ── statistics ─────────────────────────────────────────────────────────


@dataclass
class Stats:
    minimum: float | None
    maximum: float | None
    t_min: float | None
    t_max: float | None
    mean: float | None

    @property
    def range(self) -> float | None:
        if self.minimum is None or self.maximum is None:
            return None
        return self.maximum - self.minimum


def stats(project: Project, m: Measurement, start: int = 0, end: int | None = None) -> Stats:
    end = project.frame_count - 1 if end is None else end
    v = series(project, m).value[start:end + 1]
    t = times(project)[start:end + 1]
    ok = np.isfinite(v)
    if not ok.any():
        return Stats(None, None, None, None, None)
    vv, tt = v[ok], t[ok]
    i0, i1 = int(np.argmin(vv)), int(np.argmax(vv))
    return Stats(float(vv[i0]), float(vv[i1]), float(tt[i0]), float(tt[i1]), float(vv.mean()))


def phase_stats(project: Project, m: Measurement, phase: Phase) -> Stats:
    return stats(project, m, phase.start, phase.end)


def low_confidence_frames(project: Project, threshold: float = 0.5) -> list[int]:
    """Frames where a body joint used by a measurement (or any main joint) is doubtful."""
    from openkick.core.landmarks import JOINTS
    pose = project.pose
    used = {r.landmark for m in project.measurements for r in m.points if r.is_landmark}
    joints = sorted(used) if used else [j for j in JOINTS if j >= 11]
    vis = pose.visibility[:, joints]
    detected = np.isfinite(pose.xy[:, joints, 0])
    bad = ((vis < threshold) & detected).any(axis=1)
    manual = (pose.source[:, joints] == SRC_MANUAL).any(axis=1)
    return [int(i) for i in np.nonzero(bad & ~manual)[0]]


def frame_confidence(project: Project) -> np.ndarray:
    """Mean visibility of the main body joints per frame (0..1, NaN = no person)."""
    from openkick.core.landmarks import JOINTS
    joints = [j for j in JOINTS if j >= 11]
    vis = project.pose.visibility[:, joints].astype(np.float64)
    detected = np.isfinite(project.pose.xy[:, joints, 0]).any(axis=1)
    out = vis.mean(axis=1)
    out[~detected] = np.nan
    return out


# ── point of view ──────────────────────────────────────────────────────

SIDE_ON = 0.35      # shoulders closer together than this fraction of the trunk: seen from the side


def side_on_frames(pose: Pose, fps: float = 30.0) -> np.ndarray:
    """True for the frames where the camera sees the person from the side: the two shoulders
    fall almost on top of each other. Flickers shorter than half a second are ignored."""
    from scipy.ndimage import median_filter

    from openkick.core.landmarks import INDEX
    xy = pose.xy.astype(np.float64)
    ls, rs, lh, rh = (xy[:, INDEX[k]] for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    trunk = np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.abs(ls[:, 0] - rs[:, 0]) / trunk
    seen = np.isfinite(spread)
    if not seen.any():
        return seen
    size = max(3, int(fps / 2) | 1)
    return (median_filter(np.where(seen, spread, 9.0), size=size, mode="nearest") < SIDE_ON) & seen


def facing_side(project: Project, start: int = 0, end: int | None = None) -> str:
    """"left" or "right": the side of the body turned to the camera between two frames, from
    the detector's depth estimate of the legs. "right" when there is nothing to go by."""
    import warnings

    from openkick.core.landmarks import INDEX
    end = project.frame_count - 1 if end is None else end
    z = project.pose.z[start:end + 1]
    side = side_on_frames(project.pose, project.video.fps)[start:end + 1]
    if side.any():
        z = z[side]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        left, right = (np.nanmean(z[:, [INDEX[f"{s}_{k}"] for k in ("hip", "knee", "ankle")]])
                       for s in ("left", "right"))
    return "left" if np.isfinite(left) and np.isfinite(right) and left < right else "right"


# ── editing ────────────────────────────────────────────────────────────


@dataclass
class PoseEdit:
    """Undo record for a change to the pose arrays over a frame range."""
    start: int
    end: int
    old_xy: np.ndarray
    old_source: np.ndarray
    new_xy: np.ndarray
    new_source: np.ndarray

    def undo(self, pose: Pose):
        pose.xy[self.start:self.end + 1] = self.old_xy
        pose.source[self.start:self.end + 1] = self.old_source

    def redo(self, pose: Pose):
        pose.xy[self.start:self.end + 1] = self.new_xy
        pose.source[self.start:self.end + 1] = self.new_source


def edit_point(pose: Pose, frame: int, lm: int, new: tuple[float, float], scope: str = "frame",
               radius: int = 3, phase: Phase | None = None, old: tuple[float, float] | None = None) -> PoseEdit:
    """Move landmark `lm` at `frame` to `new`.

    scope: "frame"     only this frame
           "neighbors" spread the correction over ±radius frames with a smooth falloff
           "phase"     shift the joint by the same offset over the whole phase
    Frames already corrected by hand are never overwritten by propagation.
    `old` is the position before the drag (the live drag may have already moved it)."""
    n = pose.frame_count
    if scope == "neighbors":
        start, end = max(0, frame - radius), min(n - 1, frame + radius)
    elif scope == "phase" and phase is not None:
        start, end = phase.start, phase.end
    else:
        start, end = frame, frame
    old_xy = pose.xy[start:end + 1].copy()
    old_src = pose.source[start:end + 1].copy()
    if old is not None:
        old_xy[frame - start, lm] = old
    before = old_xy[frame - start, lm]
    delta = np.array(new, np.float32) - before
    if not np.isfinite(delta).all():
        delta = np.zeros(2, np.float32)
    for f in range(start, end + 1):
        if f == frame:
            pose.xy[f, lm] = new
            pose.source[f, lm] = SRC_MANUAL
            continue
        if pose.source[f, lm] == SRC_MANUAL or not np.isfinite(pose.xy[f, lm]).all():
            continue
        if scope == "neighbors":
            w = 0.5 * (1 + math.cos(math.pi * abs(f - frame) / (radius + 1)))
        else:
            w = 1.0
        pose.xy[f, lm] = old_xy[f - start, lm] + delta * w
        pose.source[f, lm] = SRC_SMOOTHED
    return PoseEdit(start, end, old_xy, old_src, pose.xy[start:end + 1].copy(), pose.source[start:end + 1].copy())


def revert_point(pose: Pose, frame: int, lm: int) -> PoseEdit:
    """Restore MediaPipe's original detection for one joint in one frame."""
    old_xy = pose.xy[frame:frame + 1].copy()
    old_src = pose.source[frame:frame + 1].copy()
    pose.xy[frame, lm] = pose.raw_xy[frame, lm]
    pose.source[frame, lm] = SRC_AUTO if np.isfinite(pose.raw_xy[frame, lm]).all() else 0
    return PoseEdit(frame, frame, old_xy, old_src, pose.xy[frame:frame + 1].copy(), pose.source[frame:frame + 1].copy())


def smooth_pose(pose: Pose, fps: float, cutoff: float = 6.0) -> PoseEdit:
    """Butterworth low-pass on every joint trajectory. Hand corrections are kept as they are."""
    old_xy, old_src = pose.xy.copy(), pose.source.copy()
    for lm in range(pose.xy.shape[1]):
        for c in range(2):
            col = pose.xy[:, lm, c]
            filt = lowpass(col, fps, cutoff).astype(np.float32)
            manual = pose.source[:, lm] == SRC_MANUAL
            filt[manual] = col[manual]
            pose.xy[:, lm, c] = filt
        auto = pose.source[:, lm] == SRC_AUTO
        pose.source[auto, lm] = SRC_SMOOTHED
    n = pose.frame_count
    return PoseEdit(0, n - 1, old_xy, old_src, pose.xy.copy(), pose.source.copy())
