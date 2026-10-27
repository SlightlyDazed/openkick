"""What each camera sees: the 26-point skeleton (RTMPose, body with feet) of every frame and,
for a handheld camera, how the picture moved."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from openkick.core.pose import data_dir
from openkick.core.video import VideoMeta, iter_frames

MODES = ("lightweight", "balanced", "performance")


def cache_file(meta: VideoMeta, kind: str) -> Path:
    p = Path(meta.path)
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    return data_dir() / "cache" / "multiview" / f"{key}-{kind}.npz"


def detect(meta: VideoMeta, mode: str = "balanced", progress: Callable | None = None,
           cancelled: Callable[[], bool] | None = None, use_cache: bool = True):
    """times (n,), keypoints (n, 26, 2) in pixels and confidence (n, 26) of the main person.
    `progress(i, n, rgb, xy)` is called for every frame. Returns None if cancelled."""
    cache = cache_file(meta, f"pose26-{mode}")
    n = meta.frame_count
    t = np.array([meta.time_of(i) for i in range(n)])
    if use_cache and cache.exists():
        z = np.load(cache)
        if len(z["xy"]) == n:
            return t, z["xy"], z["score"]
    import cv2
    from rtmlib import BodyWithFeet, PoseTracker
    model = PoseTracker(BodyWithFeet, det_frequency=10, tracking=False, mode=mode, backend="onnxruntime",
                        device="cpu", to_openpose=False)
    xy = np.full((n, 26, 2), np.nan, np.float32)
    score = np.zeros((n, 26), np.float32)
    prev = None
    for i, rgb in iter_frames(meta):
        if cancelled and cancelled():
            return None
        k, s = model(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if k is not None and len(k):
            k, s = np.asarray(k), np.asarray(s)
            c = k[:, [11, 12]].mean(axis=1)
            # the athlete: the tallest person at first, then whoever is closest to where they were
            j = int(np.argmax(np.ptp(k[:, :, 1], axis=1))) if prev is None else int(np.argmin(np.linalg.norm(c - prev, axis=1)))
            prev = c[j]
            xy[i], score[i] = k[j], s[j]
        if progress:
            progress(i, n, rgb, xy[i])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, xy=xy, score=score)
    return t, xy, score


def _person_mask(shape, pts, margin: float):
    """Everything but the person (and a margin round them), where the background can be trusted."""
    import cv2
    mask = np.full(shape, 255, np.uint8)
    if pts is not None and np.isfinite(pts).any():
        x0, y0 = np.nanmin(pts, axis=0)
        x1, y1 = np.nanmax(pts, axis=0)
        mx, my = (x1 - x0) * margin + 10, (y1 - y0) * margin + 10
        cv2.rectangle(mask, (int(x0 - mx), int(y0 - my)), (int(x1 + mx), int(y1 + my)), 0, -1)
    return mask


def _similarity(M: np.ndarray) -> np.ndarray:
    """(scale, angle, tx, ty) of a 2x3 similarity transform: these can be averaged and interpolated."""
    a, b = M[0, 0], M[1, 0]
    return np.array([np.hypot(a, b), np.arctan2(b, a), M[0, 2], M[1, 2]])


def _matrix(q: np.ndarray) -> np.ndarray:
    s, r, tx, ty = q
    T = np.eye(3)
    T[:2] = [[s * np.cos(r), -s * np.sin(r), tx], [s * np.sin(r), s * np.cos(r), ty]]
    return T


def camera_motion(meta: VideoMeta, person_xy: np.ndarray | None = None, margin: float = 0.25,
                  progress: Callable | None = None, cancelled: Callable[[], bool] | None = None,
                  use_cache: bool = True) -> np.ndarray | None:
    """(n, 3, 3) transforms from each frame's pixels to the first frame's pixels, from the
    background (the person is masked out). Identity everywhere for a camera on a tripod.

    Each frame is matched straight against the first one. Chaining frame to frame instead adds
    up every small error: a phone that never moved was found to drift 17 px over a minute, and
    that "correction" spoilt the 3D. A frame that cannot be matched (the background hidden or
    blurred) takes the place of the frames around it."""
    cache = cache_file(meta, "motion-direct")
    n = meta.frame_count
    if use_cache and cache.exists():
        T = np.load(cache)["T"]
        if len(T) == n:
            return T
    import cv2
    orb = cv2.ORB_create(2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    stride = max(1, int(np.ceil(n / 3000)))              # long videos: every few frames, the rest in between
    q = np.full((n, 4), np.nan)
    ref = None
    for i, rgb in iter_frames(meta):
        if cancelled and cancelled():
            return None
        if progress:
            progress(i, n)
        if i % stride and i != n - 1 and ref is not None:
            continue
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pts = person_xy[i] if person_xy is not None else None
        kp, des = orb.detectAndCompute(gray, _person_mask(gray.shape, pts, margin))
        if ref is None:
            if des is None or len(kp) < 30:
                continue                                  # nothing to hold on to yet: try the next frame
            ref = (np.float32([k.pt for k in kp]), des)
            q[i] = (1.0, 0.0, 0.0, 0.0)
            continue
        if des is None or len(kp) < 12:
            continue
        m = matcher.match(ref[1], des)
        if len(m) < 12:
            continue
        a = ref[0][[x.queryIdx for x in m]]
        b = np.float32([kp[x.trainIdx].pt for x in m])
        M, inliers = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is not None and inliers is not None and inliers.sum() >= 12:
            q[i] = _similarity(M)                         # this frame -> the first one
    seen = np.isfinite(q[:, 0])
    T = np.tile(np.eye(3), (n, 1, 1))
    if seen.sum() >= 2:
        frames = np.arange(n)
        q[:, 1] = np.where(seen, np.unwrap(np.where(seen, q[:, 1], 0.0)), np.nan)
        filled = np.stack([np.interp(frames, frames[seen], q[seen, k]) for k in range(4)], axis=1)
        from scipy.ndimage import median_filter
        filled = median_filter(filled, size=(5, 1), mode="nearest")      # one bad match does not make a jolt
        T = np.stack([_matrix(r) for r in filled])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, T=T)
    return T


def is_handheld(T: np.ndarray, meta: VideoMeta, tolerance: float = 0.004) -> bool:
    """Did the picture move more than a tripod would let it? (drift above 0.4 % of the diagonal)"""
    corners = np.array([[0, 0, 1], [meta.width, 0, 1], [0, meta.height, 1], [meta.width, meta.height, 1.0]])
    moved = np.einsum("nij,kj->nki", T, corners)[..., :2] - corners[None, :, :2]
    return float(np.percentile(np.linalg.norm(moved, axis=2), 95)) > tolerance * np.hypot(meta.width, meta.height)


def max_drift(T: np.ndarray, meta: VideoMeta) -> float:
    """Largest displacement of the picture centre, in pixels."""
    c = np.array([meta.width / 2, meta.height / 2, 1.0])
    return float(np.linalg.norm((T @ c)[:, :2] - c[:2], axis=1).max())
