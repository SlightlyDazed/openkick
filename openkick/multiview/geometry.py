"""Geometry of two calibrated views: normalization, triangulation, calibration from matching
body points, and left/right consistency. Pure numpy/opencv/scipy, no GUI."""

from __future__ import annotations

import warnings

import numpy as np

HALPE26 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
           "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
           "right_knee", "left_ankle", "right_ankle", "head", "neck", "hip", "left_toe", "right_toe",
           "left_small_toe", "right_small_toe", "left_heel", "right_heel"]
IDX = {n: i for i, n in enumerate(HALPE26)}
# points shown and reviewed (eyes and ears help the solver but are too small to check by eye)
BODY = [i for i, n in enumerate(HALPE26) if "eye" not in n and "ear" not in n]

BONES = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
         ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_ankle", "left_heel"), ("left_heel", "left_toe"), ("right_ankle", "right_heel"),
         ("right_heel", "right_toe"), ("neck", "head")]

# Left/right pairs, in the groups a pose network tends to swap together
LEGS = [("left_hip", "right_hip"), ("left_knee", "right_knee"), ("left_ankle", "right_ankle"),
        ("left_heel", "right_heel"), ("left_toe", "right_toe"), ("left_small_toe", "right_small_toe")]
ARMS = [("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"), ("left_wrist", "right_wrist")]
FACE = [("left_eye", "right_eye"), ("left_ear", "right_ear")]
GROUPS = {"legs": LEGS, "arms": ARMS, "face": FACE}


def side(name: str) -> str:
    return "L" if name.startswith("left") else "R" if name.startswith("right") else "C"


def swap_indices(groups=("legs", "arms", "face")) -> np.ndarray:
    """Permutation of the 26 points that exchanges left and right for the given groups."""
    perm = np.arange(len(HALPE26))
    for g in groups:
        for a, b in GROUPS[g]:
            perm[IDX[a]], perm[IDX[b]] = IDX[b], IDX[a]
    return perm


def guess_intrinsics(width: int, height: int, equiv_mm: float = 26.0) -> np.ndarray:
    """Camera matrix assuming a phone's main lens (about 26 mm full-frame equivalent), centred
    principal point and no distortion: what can be assumed without calibrating anything."""
    f = (np.hypot(width, height) / 2) / np.tan(np.arctan(21.63 / equiv_mm))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def normalize(xy: np.ndarray, K: np.ndarray, dist: np.ndarray | None = None) -> np.ndarray:
    """Pixels -> normalized image coordinates (lens distortion removed when given)."""
    if dist is None or not np.any(dist):
        return (np.asarray(xy, float) - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    import cv2
    shape = xy.shape
    pts = xy.reshape(-1, 1, 2).astype(np.float64)
    ok = np.isfinite(pts).all(axis=(1, 2))
    out = np.full((len(pts), 2), np.nan)
    if ok.any():
        out[ok] = cv2.undistortPoints(pts[ok], K, dist).reshape(-1, 2)
    return out.reshape(shape)


def _solve_rays(A: np.ndarray) -> np.ndarray:
    """The point closest to every ray, from the rows of A (M, rows, 4) of A @ [X, 1] = 0, by least
    squares. Much faster than a singular value decomposition per point, and the same answer for
    points in front of the cameras (all of them here)."""
    B, c = A[..., :3], -A[..., 3]
    N = np.einsum("mri,mrj->mij", B, B)
    r = np.einsum("mri,mr->mi", B, c)
    X = np.full(r.shape, np.nan)
    good = np.abs(np.linalg.det(N)) > 1e-18
    if good.any():
        X[good] = np.linalg.solve(N[good], r[good][..., None])[..., 0]
    return X


def triangulate(xa: np.ndarray, xb: np.ndarray, Ra, ta, Rb, tb, wa=None, wb=None) -> np.ndarray:
    """Linear triangulation of normalized points (..., 2) seen by two cameras -> (..., 3).
    `wa`, `wb` (...,) weigh each view: a point placed by hand pulls the result towards its ray."""
    Pa, Pb = np.hstack([Ra, np.reshape(ta, (3, 1))]), np.hstack([Rb, np.reshape(tb, (3, 1))])
    shape = xa.shape[:-1]
    a, b = xa.reshape(-1, 2), xb.reshape(-1, 2)
    A = np.stack([a[:, 0:1] * Pa[2] - Pa[0], a[:, 1:2] * Pa[2] - Pa[1],
                  b[:, 0:1] * Pb[2] - Pb[0], b[:, 1:2] * Pb[2] - Pb[1]], axis=1)
    if wa is not None:
        A[:, :2] *= np.reshape(wa, (-1, 1, 1))
    if wb is not None:
        A[:, 2:] *= np.reshape(wb, (-1, 1, 1))
    ok = np.isfinite(A).all(axis=(1, 2))
    X = np.full((len(a), 3), np.nan)
    if ok.any():
        X[ok] = _solve_rays(A[ok])
    return X.reshape(*shape, 3)


def reproject(X: np.ndarray, R, t) -> np.ndarray:
    xc = X @ np.asarray(R).T + np.asarray(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        return xc[..., :2] / xc[..., 2:3]


def _pose_params(R, d) -> np.ndarray:
    import cv2
    d = np.ravel(d) / np.linalg.norm(d)
    return np.concatenate([cv2.Rodrigues(np.asarray(R, float))[0].ravel(),
                           [np.arccos(np.clip(d[2], -1, 1)), np.arctan2(d[1], d[0])]])


def _pose_from(p):
    import cv2
    R = cv2.Rodrigues(np.asarray(p[:3], float))[0]
    return R, np.array([np.sin(p[3]) * np.cos(p[4]), np.sin(p[3]) * np.sin(p[4]), np.cos(p[3])])


def calibrate_from_body(na: np.ndarray, nb: np.ndarray, weights: np.ndarray, focal: float,
                        trusted: np.ndarray | None = None, init=None, threshold_px: float = 3.0):
    """Pose of camera B relative to camera A from matching body points.

    na, nb   (M, 2) normalized coordinates of the same points in both views
    weights  (M,) confidence of each match
    focal    focal length in pixels, to express errors in pixels
    trusted  (M,) bool, matches placed by hand: never rejected and weighted more
    init     (R, d) of a previous solution: refine it instead of starting from scratch
    Returns R, d (unit baseline, camera A at the origin), RMS reprojection error (px) of the
    matches that agree with the solution, and the fraction of matches that agree."""
    import cv2
    from scipy.optimize import least_squares
    I, Z = np.eye(3), np.zeros(3)
    trusted = np.zeros(len(na), bool) if trusted is None else np.asarray(trusted, bool)
    if len(na) < 8:
        raise ValueError("not enough matching points between the two videos")
    if init is None:
        cv2.setRNGSeed(0)
        E, inl = cv2.findEssentialMat(na, nb, I, method=cv2.RANSAC, prob=0.999, threshold=threshold_px / focal)
        if E is None or E.shape != (3, 3):
            raise ValueError("the two videos do not show a consistent movement")
        _, R0, t0, _ = cv2.recoverPose(E, na, nb, I, mask=inl.copy())
        inl = inl.ravel().astype(bool) | trusted
        p0 = _pose_params(R0, t0)
    else:
        p0 = _pose_params(*init)
        _, e = reprojection_error(na, nb, init[0], init[1], focal, focal)
        inl = (e < 2 * threshold_px) | trusted
        if inl.sum() < 8:
            inl = np.ones(len(na), bool)
    w = np.sqrt(np.where(trusted, 4.0, 1.0) * weights)[:, None]

    def residual(p):
        R, d = _pose_from(p)
        X = triangulate(na, nb, I, Z, R, d)
        r = np.concatenate([(reproject(X, I, Z) - na) * w, (reproject(X, R, d) - nb) * w], axis=1) * focal
        r[~inl] = 0
        return np.nan_to_num(r).ravel()

    sol = least_squares(residual, p0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    R, d = _pose_from(sol.x)
    X = triangulate(na, nb, I, Z, R, d)
    if np.nanmedian((X @ R.T + d)[inl][:, 2]) < 0 or np.nanmedian(X[inl][:, 2]) < 0:     # behind the cameras: mirror solution
        d = -d
        X = triangulate(na, nb, I, Z, R, d)
    e = np.concatenate([reproject(X, I, Z) - na, reproject(X, R, d) - nb], axis=1)[inl] * focal
    return R, d, float(np.sqrt(np.nanmean(e ** 2) * 2)), float(inl.mean())


def bone_spread(X: np.ndarray) -> float:
    """How much the rigid bones of a reconstruction change length from frame to frame: the median,
    over the bones, of their robust spread (%). A real bone keeps its length, so a wrong camera
    geometry shows here even when it reprojects as well as the right one."""
    import warnings
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a, b in RIGID[:8]:                                  # legs and arms: the long bones
            v = np.linalg.norm(X[:, IDX[a]] - X[:, IDX[b]], axis=1)
            v = v[np.isfinite(v)]
            if len(v) > 10:
                mid = np.median(v)
                out.append(100 * 1.4826 * np.median(np.abs(v - mid)) / mid if mid > 0 else np.inf)
    return float(np.median(out)) if out else float("inf")


def around(yaw_deg: float, depth: float = 1.0):
    """A guess of camera B: level like camera A and turned `yaw_deg` round the person, who stands
    `depth` in front of A. (R, d) with unit baseline, as calibrate_from_body() returns them."""
    y = np.radians(yaw_deg)
    R = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    p = np.array([0.0, 0.0, depth])                              # the person, in A's axes
    c = p - R.T @ np.array([0.0, 0.0, depth])                    # B looks at them from the same distance
    norm = np.linalg.norm(c)
    if norm < 1e-9:
        return None
    return R, -R @ (c / norm)


def epipolar_agreement(na: np.ndarray, nb: np.ndarray, focal: float, threshold_px: float = 3.0,
                       max_points: int = 400) -> float:
    """Fraction of matches that fit one single camera geometry: high only when the two sets of
    points really are the same instants seen from two places."""
    import cv2
    if len(na) < 30:
        return 0.0
    if len(na) > max_points:
        pick = np.linspace(0, len(na) - 1, max_points).astype(int)
        na, nb = na[pick], nb[pick]
    cv2.setRNGSeed(0)
    E, inl = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.99, threshold=threshold_px / focal,
                                  maxIters=150)
    if E is None or inl is None:
        return 0.0
    return float(inl.mean())


def reprojection_error(na: np.ndarray, nb: np.ndarray, R, d, focal_a: float, focal_b: float):
    """3D points and per-point reprojection error in pixels (worst of the two views), (N, K)."""
    X = triangulate(na, nb, np.eye(3), np.zeros(3), R, d)
    ea = np.linalg.norm(reproject(X, np.eye(3), np.zeros(3)) - na, axis=-1) * focal_a
    eb = np.linalg.norm(reproject(X, R, d) - nb, axis=-1) * focal_b
    return X, np.maximum(ea, eb)


def resolve_swaps(na: np.ndarray, nb: np.ndarray, R, d, focal: float, locked_a=None, locked_b=None,
                  margin: float = 0.6, min_gain_px: float = 4.0) -> dict:
    """Find left/right confusions.

    A pose network sometimes labels the left leg as right (typical when the legs cross in a side
    view). Two cues tell which labelling is right:
      1. agreement between the views: if exchanging left and right in one view makes the two views
         fit clearly better, one of them is confused in that frame;
      2. continuity in time: a limb does not jump to the other limb's place between two frames.
         This decides which of the two views was the confused one.
    Frames fixed by hand (`locked_*`, (N,) bool) are trusted as they are.
    Returns {group: {"a": (N,) bool, "b": (N,) bool}}: where each view should be exchanged."""
    n = len(na)
    locked_a = np.zeros(n, bool) if locked_a is None else np.asarray(locked_a, bool)
    locked_b = np.zeros(n, bool) if locked_b is None else np.asarray(locked_b, bool)
    I, Z = np.eye(3), np.zeros(3)
    out = {}
    for group in ("legs", "arms"):
        perm = swap_indices((group,))
        ids = np.array([IDX[x] for pair in GROUPS[group] for x in pair])
        # 1) do the views disagree?
        _, e0 = reprojection_error(na[:, ids], nb[:, ids], R, d, focal, focal)
        _, e1 = reprojection_error(na[:, ids], nb[:, perm][:, ids], R, d, focal, focal)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            m0, m1 = np.nanmean(e0, axis=1), np.nanmean(e1, axis=1)
        rel = np.isfinite(m0) & np.isfinite(m1) & (m1 < margin * m0) & (m0 - m1 > min_gain_px)
        rel &= ~(locked_a & locked_b)
        nb_rel = nb.copy()
        nb_rel[rel] = nb_rel[rel][:, perm]
        # 2) which labelling keeps every limb in its place over time? (two states per frame:
        #    as it is, or both views exchanged; shortest path through the frames)
        X = triangulate(na, nb_rel, I, Z, R, d)
        left = [IDX[a] for a, _ in GROUPS[group]]
        right = [IDX[b] for _, b in GROUPS[group]]
        Lx, Rx = X[:, left], X[:, right]
        ok = np.isfinite(Lx).all(axis=(1, 2)) & np.isfinite(Rx).all(axis=(1, 2))
        sep = np.nanmedian(np.linalg.norm(Lx - Rx, axis=2)) if ok.any() else 1.0
        prior = 0.05 * sep * len(left)
        # a frame fixed by hand in one view pins that view's labelling
        forced = np.full(n, -1)
        forced[locked_a] = 0                       # view A stays -> state 0
        forced[locked_b & ~locked_a] = rel[locked_b & ~locked_a].astype(int)   # view B stays
        both = np.zeros(n, bool)
        idx = np.nonzero(ok)[0]
        if len(idx) > 1:
            cost = np.zeros((len(idx), 2))
            back = np.zeros((len(idx), 2), int)
            for k, i in enumerate(idx):
                unary = np.array([0.0, prior])
                if forced[i] >= 0:
                    unary[1 - forced[i]] = np.inf
                if k == 0:
                    cost[k] = unary
                    continue
                j = idx[k - 1]
                keep = np.linalg.norm(Lx[i] - Lx[j], axis=1).sum() + np.linalg.norm(Rx[i] - Rx[j], axis=1).sum()
                trade = np.linalg.norm(Lx[i] - Rx[j], axis=1).sum() + np.linalg.norm(Rx[i] - Lx[j], axis=1).sum()
                for s in (0, 1):
                    c = cost[k - 1] + np.array([keep if s == 0 else trade, trade if s == 0 else keep])
                    back[k, s] = int(np.argmin(c))
                    cost[k, s] = c[back[k, s]] + unary[s]
            s = int(np.argmin(cost[-1]))
            for k in range(len(idx) - 1, -1, -1):
                both[idx[k]] = bool(s)
                s = back[k, s]
        out[group] = {"a": both, "b": both ^ rel}
    return out


def apply_swaps(xy: np.ndarray, swaps: dict, view: str) -> np.ndarray:
    """Keypoints (N, K, ...) with left and right exchanged where `swaps` says so."""
    out = xy.copy()
    for group, s in swaps.items():
        m = s[view]
        if m.any():
            out[m] = out[m][:, swap_indices((group,))]
    return out


def sync_offset(ta, xya, tb, xyb, max_s: float = 1.5) -> tuple[float, float]:
    """Seconds to add to camera B's clock to match camera A, from the vertical motion of the
    body (rise and fall look alike from any viewpoint). Returns (offset, correlation)."""
    def height(t, xy):
        size = np.nanmedian(np.abs(xy[:, IDX["neck"], 1] - xy[:, IDX["hip"], 1]))
        y = np.nanmean(xy[:, [IDX[k] for k in ("hip", "neck", "left_knee", "right_knee")], 1], axis=1)
        return -(y - np.nanmedian(y)) / size
    ya, yb = height(ta, xya), height(tb, xyb)
    oka, okb = np.isfinite(ya), np.isfinite(yb)
    if oka.sum() < 10 or okb.sum() < 10:
        return 0.0, 0.0
    ya, yb = np.interp(ta, ta[oka], ya[oka]), np.interp(tb, tb[okb], yb[okb])
    best, best_r = 0.0, -np.inf
    lo = max(-max_s, ta[0] - tb[-1] + 0.5)
    hi = min(max_s, ta[-1] - tb[0] - 0.5)
    for off in np.arange(lo, hi + 1e-9, 0.005):
        g = np.interp(ta, tb + off, yb, left=np.nan, right=np.nan)
        ok = np.isfinite(g)
        if ok.sum() < 20:
            continue
        a, b = ya[ok] - ya[ok].mean(), g[ok] - g[ok].mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den < 1e-9:
            continue
        r = float(a @ b / den) * min(1.0, ok.sum() / (0.6 * len(ta)))   # prefer long overlaps
        if r > best_r:
            best, best_r = float(off), r
    return best, best_r


def resample(t, xy, score, t_new):
    """Keypoints (N, K, 2) and scores (N, K) interpolated at new times."""
    out = np.empty((len(t_new), xy.shape[1], 2))
    sc = np.empty((len(t_new), xy.shape[1]))
    for j in range(xy.shape[1]):
        ok = np.isfinite(xy[:, j]).all(axis=1)
        if ok.sum() < 2:
            out[:, j], sc[:, j] = np.nan, 0
            continue
        for c in range(2):
            out[:, j, c] = np.interp(t_new, t[ok], xy[ok, j, c])
        sc[:, j] = np.interp(t_new, t[ok], score[ok, j])
    return out, sc


def scale_from_height(X: np.ndarray, height_m: float) -> float:
    """Factor from reconstruction units to metres, from the person's height (top of the head
    to the heels in the most upright frames)."""
    h = np.linalg.norm(X[:, IDX["head"]] - (X[:, IDX["left_heel"]] + X[:, IDX["right_heel"]]) / 2, axis=1)
    h = h[np.isfinite(h)]
    return height_m / float(np.percentile(h, 90)) if len(h) else 1.0


def bone_variation(X: np.ndarray) -> dict:
    """How much each bone's reconstructed length changes over time (%, lower is better):
    a real bone keeps its length, so this measures quality without any reference system."""
    out = {}
    for name, a, b in (("thigh", "hip", "knee"), ("shank", "knee", "ankle"), ("upper_arm", "shoulder", "elbow"),
                       ("forearm", "elbow", "wrist")):
        vals = []
        for s in ("left", "right"):
            v = np.linalg.norm(X[:, IDX[f"{s}_{a}"]] - X[:, IDX[f"{s}_{b}"]], axis=1)
            v = v[np.isfinite(v)]
            if len(v) > 5:
                vals.append(100 * np.std(v) / np.median(v))
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out


# Segments that cannot change length, and which of them meet at each point
RIGID = [("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"),
         ("right_knee", "right_ankle"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_ankle", "left_heel"),
         ("left_heel", "left_toe"), ("left_ankle", "left_toe"), ("left_toe", "left_small_toe"),
         ("right_ankle", "right_heel"), ("right_heel", "right_toe"), ("right_ankle", "right_toe"),
         ("right_toe", "right_small_toe"), ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")]


def bone_inconsistency(X: np.ndarray) -> np.ndarray:
    """(N, 26): how far the bones meeting at each point are from their usual length, in the
    units of X. A point is charged only when EVERY bone that meets there is off, which is what
    happens when that point (and not its neighbour) is misplaced. This catches what the
    agreement between two cameras cannot: a point displaced along the line of sight."""
    n = len(X)
    dev = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a, b in RIGID:
            length = np.linalg.norm(X[:, IDX[a]] - X[:, IDX[b]], axis=1)
            d = np.abs(length - np.nanmedian(length)) if np.isfinite(length).sum() > 5 else np.full(n, np.nan)
            dev.setdefault(IDX[a], []).append(d)
            dev.setdefault(IDX[b], []).append(d)
        out = np.full((n, X.shape[1]), np.nan)
        for j, ds in dev.items():
            out[:, j] = np.nanmin(np.stack(ds), axis=0)
    return out


def jumps(X: np.ndarray) -> np.ndarray:
    """(N, K): how far each point is from where its neighbours in time say it should be (the
    midpoint of the frames before and after), in the units of X. A point the detector got wrong
    in a single frame jumps away and back; real movement, even fast, does not."""
    n = len(X)
    out = np.full(X.shape[:2], np.nan)
    if n < 5:
        return out
    near = np.full(X.shape[:2], np.nan)
    far = np.full(X.shape[:2], np.nan)
    near[1:-1] = np.linalg.norm(X[1:-1] - (X[:-2] + X[2:]) / 2, axis=2)
    far[2:-2] = np.linalg.norm(X[2:-2] - (X[:-4] + X[4:]) / 2, axis=2)
    return np.fmin(near, far)          # the wider pair keeps the neighbours of a jump clean


# ── more than two cameras ──────────────────────────────────────────────


SPIKE_M = 0.03             # a point this far from its neighbours' path (or 4 of its usual tremor) is a one-off error


def path_frames(fps: float) -> int:
    """How many frames (odd) the path of a point is taken over: 0.15 s, at least 5."""
    return max(5, int(round(0.15 * fps)) | 1)


def bridged(X: np.ndarray) -> np.ndarray:
    """(N, K, 3) with each point's gaps in time filled by straight lines."""
    Y = np.array(X, float)
    idx = np.arange(len(Y))
    for k in range(Y.shape[1]):
        ok = np.isfinite(Y[:, k]).all(axis=1)
        if 2 <= ok.sum() < len(Y):
            for c in range(3):
                Y[:, k, c] = np.interp(idx, idx[ok], Y[ok, k, c])
    return Y


def path_of(X: np.ndarray, frames: int) -> np.ndarray:
    """Where the neighbouring frames put each point: the running median over `frames`. A point
    wrong for a frame or two does not move it."""
    from scipy.ndimage import median_filter
    return median_filter(bridged(X), size=(frames, 1, 1), mode="nearest")


def smooth(X: np.ndarray, fps: float, cutoff_hz: float, spikes: bool = True) -> np.ndarray:
    """The 3D points (metres) without the tremor faster than `cutoff_hz` and, with `spikes`,
    without one-off jumps: a point far from its neighbours' path is dropped and bridged. Then
    every trajectory goes through a zero-phase Butterworth low-pass (the biomechanics standard).
    In a very fast gesture a real peak can look like a jump: leave `spikes` off there.
    Points that were never reconstructed stay NaN."""
    from openkick.core.analysis import lowpass
    if len(X) < 10:
        return np.array(X, float)
    missing = ~np.isfinite(X).all(axis=2)
    Y = np.array(X, float)
    if spikes:
        frames = path_frames(fps)
        for _ in range(2):                   # a jump drags the path a little: look again once it is gone
            d = np.linalg.norm(bridged(Y) - path_of(Y, frames), axis=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                tremor = 1.4826 * np.nanmedian(d, axis=0)            # each point's own, over the clip
            Y[d > np.maximum(4 * np.nan_to_num(tremor), SPIKE_M)] = np.nan
    Y = bridged(Y)
    for k in range(Y.shape[1]):
        for c in range(3):
            Y[:, k, c] = lowpass(Y[:, k, c], fps, cutoff_hz)
    Y[missing] = np.nan
    return Y


FOOT = ("heel", "toe", "small_toe")      # the points that turn with the foot about the ankle



def _kabsch(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations R with B ≈ A @ R.T, frame by frame. A, B (N, P, 3), w (N, P)."""
    H = np.einsum("np,npi,npj->nij", w, A, B)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nji,nkj->nik", Vt, U)))
    D = np.zeros_like(H)
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    return np.einsum("nji,njk,nlk->nil", Vt, D, U)


def rigid_feet(X: np.ndarray, unsure: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Each foot as one piece, like a shoe with a stiff sole: heel, toe and small toe keep one
    shape (the typical one in the clip) and turn together about the ankle. The ankle stays
    where it was, so the leg is untouched; the foot still turns as much as it does, only its
    shape stops changing. Each frame's turn is the one that fits its points best, the points
    the cameras are less sure of (`unsure`, (N, K)) counting less."""
    Y = np.array(X, float)
    u = np.maximum(np.nan_to_num(np.asarray(unsure, float), nan=1.0), 0.0) + 1.0
    for side in ("left", "right"):
        a = IDX[f"{side}_ankle"]
        idx = [IDX[f"{side}_{p}"] for p in FOOT]
        V = Y[:, idx] - Y[:, a][:, None]                                   # (N, 3, 3) from the ankle
        ok = np.isfinite(V).all(axis=(1, 2))
        if ok.sum() < 10:
            continue
        w = 1.0 / u[:, idx]
        # the typical shape: start from the frame whose distances are the usual ones, then average
        # every frame turned onto it (median, so wrong frames do not pull it)
        dist = np.linalg.norm(V[:, [0, 0, 1]] - V[:, [1, 2, 2]], axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = int(np.nanargmin(np.where(ok, np.nansum(np.abs(dist - np.nanmedian(dist[ok], axis=0)), axis=1), np.inf)))
        T = V[ref]
        for _ in range(rounds):
            R = _kabsch(np.broadcast_to(T, V[ok].shape), V[ok], w[ok])
            T = np.median(np.einsum("nij,npi->npj", R, V[ok]), axis=0)     # each frame turned back: V @ R
        R = _kabsch(np.broadcast_to(T, V[ok].shape), V[ok], w[ok])
        fitted = np.einsum("pj,nij->npi", T, R)                              # T @ R.T
        rows = np.nonzero(ok)[0]
        Y[rows[:, None], np.array(idx)[None, :]] = Y[rows, a][:, None] + fitted
    return Y


def triangulate_n(x: np.ndarray, cams: list, w: np.ndarray | None = None) -> np.ndarray:
    """Linear triangulation from any number of cameras.

    x     (V, ..., 2) normalized points, one block per camera (NaN where a camera has no point)
    cams  V poses (R, t) with x_cam = R X + t
    w     (V, ...) weight of each observation (0 leaves it out)
    A point needs at least two cameras; otherwise it is NaN."""
    V = len(cams)
    shape = x.shape[1:-1]
    xf = np.asarray(x, float).reshape(V, -1, 2)
    M = xf.shape[1]
    wf = np.ones((V, M)) if w is None else np.asarray(w, float).reshape(V, -1)
    ok = np.isfinite(xf).all(axis=2) & np.isfinite(wf) & (wf > 0)
    A = np.zeros((M, 2 * V, 4))
    for v, (R, t) in enumerate(cams):
        P = np.hstack([R, np.reshape(t, (3, 1))])
        u = np.where(ok[v][:, None], xf[v], 0.0)
        wv = np.where(ok[v], wf[v], 0.0)[:, None]
        A[:, 2 * v] = (u[:, 0:1] * P[2] - P[0]) * wv
        A[:, 2 * v + 1] = (u[:, 1:2] * P[2] - P[1]) * wv
    good = ok.sum(axis=0) >= 2
    X = np.full((M, 3), np.nan)
    if good.any():
        X[good] = _solve_rays(A[good])
    return X.reshape(*shape, 3)


def baseline_scale(X: np.ndarray, x: np.ndarray, R, d) -> float:
    """A camera's orientation R and the direction d towards it are known, but not how far it is.
    Given 3D points X already reconstructed by other cameras and where this one sees them (x,
    normalized), return the distance s such that x ~ R X + s d."""
    a = X.reshape(-1, 3) @ np.asarray(R).T
    u = x.reshape(-1, 2)
    lhs = np.concatenate([u[:, 0] * d[2] - d[0], u[:, 1] * d[2] - d[1]])
    rhs = np.concatenate([a[:, 0] - u[:, 0] * a[:, 2], a[:, 1] - u[:, 1] * a[:, 2]])
    ok = np.isfinite(lhs) & np.isfinite(rhs)
    if ok.sum() < 20:
        raise ValueError("too few points in common with the other videos")
    lhs, rhs = lhs[ok], rhs[ok]
    s = float(lhs @ rhs / (lhs @ lhs))
    for _ in range(3):                                  # drop the points that do not fit and solve again
        r = np.abs(lhs * s - rhs)
        keep = r <= 2.5 * np.median(r) + 1e-12
        s = float(lhs[keep] @ rhs[keep] / (lhs[keep] @ lhs[keep]))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("its position does not fit the other videos")
    return s


def resect(X: np.ndarray, x: np.ndarray, seen: np.ndarray, focal: float, threshold_px: float = 4.0):
    """Where a camera is, from 3D points other cameras already rebuilt and where it sees them
    (normalized coordinates): (R, t) such that x ~ R X + t, or None when too few points agree.
    Placing a third camera this way keeps it consistent with the first two; working it out
    separately from the body and joining it afterwards can leave the three disagreeing."""
    import cv2
    X = np.asarray(X, float).reshape(-1, 3)
    u = np.asarray(x, float).reshape(-1, 2)
    ok = np.asarray(seen, bool).reshape(-1) & np.isfinite(X).all(axis=1) & np.isfinite(u).all(axis=1)
    if ok.sum() < 30:
        return None
    P, Q = X[ok].astype(np.float64), u[ok].astype(np.float64)
    good, rvec, tvec, inliers = cv2.solvePnPRansac(P, Q, np.eye(3), None, reprojectionError=threshold_px / focal,
                                                   iterationsCount=500, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not good or inliers is None or len(inliers) < 30:
        return None
    idx = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(P[idx], Q[idx], np.eye(3), None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    return R, tvec.ravel()


def view_errors(X: np.ndarray, x: np.ndarray, cams: list, focals) -> np.ndarray:
    """(V, ...) reprojection error of each camera in pixels (NaN where it has no point)."""
    return np.stack([np.linalg.norm(reproject(X, R, t) - x[v], axis=-1) * focals[v] for v, (R, t) in enumerate(cams)])


LEG_BONES = RIGID[:4]      # thighs and shanks: what the cameras must keep the same length
BONE_WEIGHT = 50.0         # a bone 2 % off its usual length costs as much as 1 px of reprojection error


def refine_cameras(x: np.ndarray, w: np.ndarray, cams: list, base: int, focals, trusted: np.ndarray | None = None,
                   threshold_px: float = 3.0, max_nfev: int = 30, shape: tuple | None = None):
    """Adjust all cameras together so that every one agrees with the rest.

    x        (V, M, 2) normalized points (NaN where a camera has no point)
    w        (V, M) confidence of each observation
    cams     current poses; camera 0 is the origin and stays fixed
    base     the camera whose distance to camera 0 defines the unit of length
    trusted  (V, M) bool, points placed by hand: never rejected and weighted more
    shape    (frames, points) when x holds whole skeletons, frame after frame: then the bones of
             the legs must also keep their length. Reprojection alone has several almost equally
             good answers when the body hardly changes shape (a rider on a trainer); only the
             right one keeps the thighs and shanks the same length from frame to frame.
    Returns the new poses, the RMS reprojection error (px) and the fraction of observations
    that agree with the solution."""
    from scipy.optimize import least_squares
    V = len(cams)
    focals = np.asarray(focals, float)
    trusted = np.zeros(x.shape[:2], bool) if trusted is None else np.asarray(trusted, bool)
    ok = np.isfinite(x).all(axis=2) & (w > 0)
    with np.errstate(invalid="ignore"):
        e0 = view_errors(triangulate_n(x, cams, ok), x, cams, focals)
        inl = ok & ((e0 < 2 * threshold_px) | trusted)
    inl &= inl.sum(axis=0) >= 2
    wt = np.sqrt(np.where(trusted, 4.0, 1.0) * np.where(inl, w, 0.0))[..., None] * focals[:, None, None]

    def pack(cs):
        out = []
        for i in range(1, V):
            out.append(_pose_params(*cs[i]) if i == base else
                       np.concatenate([__import__("cv2").Rodrigues(np.asarray(cs[i][0], float))[0].ravel(), np.ravel(cs[i][1])]))
        return np.concatenate(out)

    def unpack(p):
        import cv2
        cs, k = [(np.eye(3), np.zeros(3))], 0
        for i in range(1, V):
            if i == base:
                cs.append(_pose_from(p[k:k + 5]))
                k += 5
            else:
                cs.append((cv2.Rodrigues(np.asarray(p[k:k + 3], float))[0], np.array(p[k + 3:k + 6], float)))
                k += 6
        return cs

    bones = [(IDX[a], IDX[b]) for a, b in LEG_BONES] if shape is not None else []

    def residual(p):
        cs = unpack(p)
        X = triangulate_n(x, cs, inl)
        r = np.nan_to_num(np.stack([(reproject(X, R, t) - x[v]) for v, (R, t) in enumerate(cs)]) * wt).ravel()
        if not bones:
            return r
        S = X.reshape(*shape, 3)
        with np.errstate(invalid="ignore"):
            L = np.stack([np.linalg.norm(S[:, a] - S[:, b], axis=1) for a, b in bones], axis=1)
            mid = np.nanmedian(L, axis=0)
            rb = BONE_WEIGHT * (L - mid) / mid
        return np.concatenate([r, np.nan_to_num(rb).ravel()])

    scale = np.linalg.norm(cams[base][1])
    start = [(R, np.asarray(t, float) / scale) for R, t in cams]
    sol = least_squares(residual, pack(start), loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=max_nfev)
    out = unpack(sol.x)
    e = view_errors(triangulate_n(x, out, inl), x, out, focals)[inl]
    return out, float(np.sqrt(np.nanmean(e ** 2))), float(inl.sum() / max(ok.sum(), 1))
