"""A 3D session from two or more videos: what each camera saw, the corrections made by hand,
and the 3D reconstruction solved from all of them. No GUI here."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openkick.core.video import VideoMeta
from openkick.multiview import geometry as G

SRC_NONE, SRC_AUTO, SRC_MANUAL, SRC_SPREAD = 0, 1, 2, 3     # same meaning as in the 2D editor
SRC_FOLLOW = 4             # put by the program where a correction in another video says (Session.follow)
SRC_LEARNED = 5            # put by the program on the path it learned of that point over the pedal stroke
MIN_SCORE = 0.5            # detections below this confidence are not used unless fixed by hand
MANUAL_WEIGHT = 3.0        # how much more a hand-placed point pulls the 3D point towards its ray
PROBLEM_CM = 6.0           # the cameras disagree by more than this -> worth a look
REJECT_CM = 12.0           # three or more cameras: one that puts a point this far from the rest is left out
OFF_PATH_CM = 4.0          # a 3D point this far from the path its neighbouring frames give it has jumped
SMOOTH_HZ = 8.0            # 3D smoothing: the tremor faster than this goes (0: none)
FAST_SMOOTH_HZ = 12.0      # for fast gestures, whose peaks a lower cut-off would flatten
GUIDE_PCT = 5.0            # a thigh or shank this much off its usual length shows where its point should be
FAST = ("kick", "jump")    # the templates of fast gestures
FOLLOW_LIMIT = 0.03        # a video follows a correction only if that moves its point less than this share of the body
LEARN_PARTS = 8            # the pedal stroke in parts: each needs a hand-placed point before its path is learned
LEARN_ORDER = 3            # harmonics of a learned path (the error repeats once per turn, and smoothly)
STROKE_PARTS = ("at the top", "going down", "going down", "at the bottom", "at the bottom", "going up", "going up",
                "at the top")  # what each part is called, from the top round


def smoothing_for(template: str | None) -> float:
    """The 3D smoothing an analysis starts with."""
    return FAST_SMOOTH_HZ if template in FAST else SMOOTH_HZ
#                            (only gross errors: against the laboratory, leaving out smaller disagreements
#                            made the angles worse, because each camera sees a joint slightly differently)
MAX_VIEWS = 6
FORMAT = 2
I3, Z3 = np.eye(3), np.zeros(3)


class View:
    """One camera: its video, the detected keypoints and the user's corrections."""

    def __init__(self, meta: VideoMeta, t: np.ndarray, xy: np.ndarray, score: np.ndarray,
                 motion: np.ndarray | None = None, K: np.ndarray | None = None):
        self.meta = meta
        self.t = np.asarray(t, float)
        self.raw_xy = np.asarray(xy, np.float32).copy()
        self.raw_score = np.asarray(score, np.float32).copy()
        self.xy = self.raw_xy.copy()
        self.score = self.raw_score.copy()
        self.source = np.where(np.isfinite(self.xy).all(axis=2), SRC_AUTO, SRC_NONE).astype(np.uint8)
        self.motion = motion        # (n, 3, 3): pixels of frame i -> pixels of frame 0 (handheld camera)
        self.K = G.guess_intrinsics(meta.width, meta.height) if K is None else np.asarray(K, float)

    @property
    def frame_count(self) -> int:
        return len(self.t)

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    @property
    def handheld(self) -> bool:
        return self.motion is not None

    def stable(self, xy: np.ndarray | None = None) -> np.ndarray:
        """Keypoints (these, or the current ones) as if the camera had not moved (first frame's coordinates)."""
        xy = self.xy if xy is None else xy
        if self.motion is None:
            return xy.astype(float)
        h = np.concatenate([xy.astype(float), np.ones((*xy.shape[:2], 1))], axis=2)
        return np.einsum("nij,nkj->nki", self.motion, h)[..., :2]

    def to_frame(self, frame: int, pts: np.ndarray) -> np.ndarray:
        """Points in the first frame's coordinates -> pixels of `frame`."""
        if self.motion is None:
            return np.asarray(pts, float)
        inv = np.linalg.inv(self.motion[frame])
        h = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1)
        return (h @ inv.T)[..., :2]

    def from_frame(self, frame: int, pts: np.ndarray) -> np.ndarray:
        if self.motion is None:
            return np.asarray(pts, float)
        h = np.concatenate([pts, np.ones((*np.shape(pts)[:-1], 1))], axis=-1)
        return (h @ self.motion[frame].T)[..., :2]

    def valid(self) -> np.ndarray:
        fixed = np.isin(self.source, (SRC_MANUAL, SRC_SPREAD, SRC_FOLLOW, SRC_LEARNED))
        return np.isfinite(self.xy).all(axis=2) & ((self.score >= MIN_SCORE) | fixed)

    def manual(self) -> np.ndarray:
        return self.source == SRC_MANUAL

    def body_px(self) -> float:
        """Typical height of the person in this video, in pixels."""
        xy = self.stable()
        h = np.linalg.norm(xy[:, G.IDX["head"]] - (xy[:, G.IDX["left_heel"]] + xy[:, G.IDX["right_heel"]]) / 2, axis=1)
        h = h[np.isfinite(h)]
        return float(np.percentile(h, 90)) if len(h) else float(self.meta.height)


def observe(views: list, offsets: list, step: int = 1) -> dict:
    """All views on the first one's clock.
    rows, t   frames of the first view inside the time span common to all, and their times
    n         (V, N, K, 2) normalized points (NaN where a view has no usable point)
    ok, w, manual  (V, N, K) usable / confidence / placed by hand
    follow    (V, N, K) put there to follow a correction in another video: it agrees with the 3D
              by construction, so it has no say in where the cameras are (w is 0)"""
    a = views[0]
    keep = np.ones(len(a.t), bool)
    for v, off in zip(views[1:], offsets[1:]):
        keep &= (a.t >= v.t[0] + off - 1e-9) & (a.t <= v.t[-1] + off + 1e-9)
    rows = np.nonzero(keep)[0][::step]
    t = a.t[rows]
    V, n, k = len(views), len(t), a.xy.shape[1]
    N = np.full((V, n, k, 2), np.nan)
    ok, man, w = np.zeros((V, n, k), bool), np.zeros((V, n, k), bool), np.zeros((V, n, k))
    fol = np.zeros((V, n, k), bool)
    for i, (v, off) in enumerate(zip(views, offsets)):
        if i == 0:
            s = v.stable()[rows]
            ok[0], man[0], w[0] = v.valid()[rows], v.manual()[rows], v.score[rows]
            fol[0] = v.source[rows] == SRC_FOLLOW
        else:
            s_all, v_all, m_all = v.stable(), v.valid(), v.manual()
            tv = v.t + off
            near = np.abs(tv[None, :] - t[:, None]).argmin(axis=1) if n else np.zeros(0, int)
            s = np.full((n, k, 2), np.nan)
            for j in range(k):
                good = v_all[:, j]
                if good.sum() < 2:
                    continue
                for c in range(2):
                    s[:, j, c] = np.interp(t, tv[good], s_all[good, j, c])
                ok[i, :, j] = np.interp(t, tv, good.astype(float)) > 0.5
                w[i, :, j] = np.interp(t, tv[good], v.score[good, j])
                man[i, :, j] = m_all[near, j]
                fol[i, :, j] = v.source[near, j] == SRC_FOLLOW
        w[i] = np.where(fol[i], 0.0, np.where(man[i], 1.0, w[i]))
        N[i] = np.where(ok[i][..., None], G.normalize(s, v.K), np.nan)
    return {"rows": rows, "t": t, "n": N, "ok": ok, "w": w, "manual": man, "follow": fol}


class Pair:
    """Two cameras: their time offset and where the second is with respect to the first."""

    def __init__(self, a: View, b: View):
        self.views = [a, b]
        self.offset: float | None = None     # seconds to add to B's clock to match A's
        self.R: np.ndarray | None = None     # pose of camera B relative to camera A (unit baseline)
        self.d: np.ndarray | None = None
        self.rms = float("nan")
        self.agreement = float("nan")
        self.by_sound = False                # the offset came from the videos' sound

    def observe(self, offset: float, step: int = 1) -> dict:
        o = observe(self.views, [0.0, offset], step)
        both = o["ok"][0] & o["ok"][1]
        return {"rows": o["rows"], "t": o["t"], "ok": both, "w": np.minimum(o["w"][0], o["w"][1]),
                "na": np.where(both[..., None], o["n"][0], np.nan), "nb": np.where(both[..., None], o["n"][1], np.nan),
                "ma": o["manual"][0], "mb": o["manual"][1]}

    def synchronize(self, max_s: float | None = None, progress=None) -> float:
        """Find the time offset between the two videos with no help from the user.

        First the sound, when both videos have it and it agrees clearly: the phones heard the
        same room, and that lines them up whatever the person does. Without it, two cues from
        the movement, the second more reliable than the first:
          1. the up-and-down movement of the body looks alike from any camera;
          2. only at the right offset do the two skeletons fit one single pair of cameras.
        A movement that repeats (pedalling) can fool the movement cues by one whole turn: the
        sound cannot be fooled that way. Either way the result is refined later (refine_offset).
        Returns the offset (seconds added to B's clock)."""
        a, b = self.views
        self.by_sound = False
        from openkick.multiview import audio as AU
        heard, sure = None, 0.0
        if Path(a.meta.path).resolve() != Path(b.meta.path).resolve():      # one file twice: its sound says nothing
            try:
                heard, sure = AU.offset(a.meta.path, b.meta.path)
            except Exception:  # noqa: BLE001 - no sound to go by: the movement decides
                heard, sure = None, 0.0
        overlap = (min(a.t[-1], b.t[-1] + heard) - max(a.t[0], b.t[0] + heard)) if heard is not None else 0.0
        if heard is not None and sure >= AU.SURE and overlap >= 2.0:
            self.offset, self.by_sound = float(heard), True
            return self.offset
        span = min(a.t[-1] - a.t[0], b.t[-1] - b.t[0])
        lo, hi = a.t[0] - b.t[-1] + 0.6 * span, a.t[-1] - b.t[0] - 0.6 * span     # keep 60 % overlap
        if max_s is not None:
            lo, hi = max(lo, -max_s), min(hi, max_s)
        if hi <= lo:
            self.offset = 0.0
            return 0.0
        dt = float(np.median(np.diff(a.t)))
        step = max(dt, 1 / 30)
        offsets = np.arange(lo, hi + 1e-9, step)
        sub = max(1, int(round(len(a.t) / 120)))
        scores = np.zeros(len(offsets))
        focal = a.focal
        for i, off in enumerate(offsets):
            o = self.observe(float(off), sub)
            m = o["ok"]
            scores[i] = G.epipolar_agreement(o["na"][m], o["nb"][m], focal) * min(1.0, m.sum() / 200)
            if progress:
                progress(i + 1, len(offsets))
        # the body's vertical movement breaks ties between offsets that fit equally well
        motion, _ = G.sync_offset(a.t, a.stable(), b.t, b.stable(), max_s=max(abs(lo), abs(hi)))
        best = float(scores.max())
        good = np.nonzero(scores >= 0.9 * best)[0]
        if best < 0.25:
            pick = motion
        else:
            pick = float(offsets[good[np.abs(offsets[good] - motion).argmin()]]) if len(good) > 1 and \
                np.abs(offsets[good] - motion).min() <= 2 * step else float(offsets[int(scores.argmax())])
        self.offset = pick
        return pick

    def refine_offset(self, window: float = 0.12, step: float = 1 / 240) -> float:
        """Fine time offset: the one where the two cameras agree best (needs solved cameras)."""
        base = self.offset
        best, best_e = base, np.inf
        fa, fb = self.views[0].focal, self.views[1].focal
        for off in np.arange(base - window, base + window + 1e-9, step):
            o = self.observe(float(off))
            if o["ok"].sum() < 50:
                continue
            _, e = G.reprojection_error(o["na"], o["nb"], self.R, self.d, fa, fb)
            m = float(np.nanmedian(e))
            if m < best_e:
                best, best_e = float(off), m
        self.offset = best
        return best

    def calibrate(self, warm: bool = True):
        """Where is camera B with respect to camera A? Solved from the body itself.

        From scratch, several starting guesses are tried: the one the matches suggest and
        camera B turned round the person every 30 degrees. When the body hardly changes shape
        (a rider on a trainer moves the legs in two flat planes and little else) several
        geometries fit the matches almost equally well; only the right one keeps the bones the
        same length from frame to frame, so among those that fit, that decides."""
        o = self.observe(self.offset)
        step = max(1, int(round(len(o["rows"]) / 250)))
        m = o["ok"][::step]
        trusted = (o["ma"] | o["mb"])[::step][m]
        args = (o["na"][::step][m], o["nb"][::step][m], o["w"][::step][m], self.views[0].focal)
        if warm and self.R is not None:
            self.R, self.d, self.rms, self.agreement = G.calibrate_from_body(*args, trusted=trusted, init=(self.R, self.d))
            return
        # every guess on all the frames: tried on a third of them, one walking trial of the laboratory
        # data kept a guess 28 degrees off (50 once refined) where all the frames give 10
        guesses = [None] + [g for g in (G.around(y) for y in range(30, 360, 30)) if g is not None]
        found = []
        for init in guesses:
            try:
                found.append(G.calibrate_from_body(*args, trusted=trusted, init=init))
            except ValueError:
                continue
        found = [f for f in found if np.isfinite(f[2]) and np.isfinite(f[3])]
        if not found:
            raise ValueError("the two videos do not show a consistent movement")
        best_rms = min(f[2] for f in found)
        most = max(f[3] for f in found)
        fits = [f for f in found if f[2] <= 1.5 * best_rms + 0.5 and f[3] >= 0.5 * most] or found
        sample = max(1, int(round(len(o["rows"]) / 150)))

        def spread(f):
            X = G.triangulate(o["na"][::sample], o["nb"][::sample], I3, Z3, f[0], f[1])
            return G.bone_spread(X)
        best = min(fits, key=spread)
        self.R, self.d, self.rms, self.agreement = best

    def solve(self, progress=None):
        self.R = self.d = None
        self.synchronize(progress=progress)
        self.calibrate(warm=False)
        for _ in range(2):
            before = self.offset
            self.refine_offset()
            self.calibrate(warm=True)
            if abs(self.offset - before) < 1 / 240:
                break


@dataclass
class Edit:
    """Undo record of a change to one view over a frame range."""
    view: int
    start: int
    end: int
    old: tuple
    new: tuple

    def apply(self, session: "Session", new: bool = True):
        v = session.views[self.view]
        xy, score, source = self.new if new else self.old
        s = slice(self.start, self.end + 1)
        v.xy[s], v.score[s], v.source[s] = xy, score, source


def _snap(v: View, start: int, end: int) -> tuple:
    s = slice(start, end + 1)
    return v.xy[s].copy(), v.score[s].copy(), v.source[s].copy()


def _two_lengths(A: np.ndarray, B: np.ndarray, la: float, lb: float, near: np.ndarray) -> np.ndarray:
    """The point `la` from A and `lb` from B nearest to `near` (a knee between a hip and an
    ankle). Out of reach (A and B too far apart, or too close), the nearest that can be."""
    v = B - A
    d = float(np.linalg.norm(v))
    if d < 1e-9:
        return near
    u = v / d
    if d >= la + lb or d <= abs(la - lb):
        return A + u * (la if d >= la + lb else np.sign(la - lb) * la)
    a = (la ** 2 - lb ** 2 + d ** 2) / (2 * d)
    centre, radius = A + a * u, np.sqrt(max(la ** 2 - a ** 2, 0.0))
    w = (near - centre) - np.dot(near - centre, u) * u
    n = float(np.linalg.norm(w))
    if n < 1e-9:
        return centre
    return centre + w / n * radius


class Session:
    def __init__(self, views: list, height_m: float = 1.70):
        if not 2 <= len(views) <= MAX_VIEWS:
            raise ValueError(f"between 2 and {MAX_VIEWS} videos are needed")
        from openkick.core.subject import Subject
        self.views = list(views)
        self.height_m = float(height_m)
        self.template: str | None = None     # what is analysed ("cycling", "gait"...); None: found by itself
        self.subject = Subject(height_m=self.height_m)
        self.smooth_hz = SMOOTH_HZ           # cut-off of the 3D smoothing (0: the points as triangulated)
        self.feet = ""                       # "" free, "rigid": each foot one piece (a stiff sole)
        self.report_notes = ""               # the evaluator's observations, for the report
        self.report_evaluator = ""           # who evaluated it
        self.reviewed: set = set()           # (frame of the first view, joint) checked by the user and fine
        self.offsets: list | None = None     # seconds to add to each view's clock to match the first one's
        self.cams: list | None = None        # pose (R, t) of every camera relative to the first one
        self.base = 1                        # the camera whose distance to the first one is the unit of length
        self.rms = float("nan")
        self.agreement = float("nan")
        self.path: Path | None = None
        self.revision = 0
        self.saved_revision = 0
        # filled by update(): one row per frame of the first view inside the common time span
        V = len(self.views)
        self.rows = np.zeros(0, int)
        self.t = np.zeros(0)
        self.X = np.zeros((0, 26, 3))        # metres, first camera's axes (x right, y down, z away), smoothed
        self.X_raw = np.zeros((0, 26, 3))    # the same before smoothing: what the doubtful points are found on
        self.followed = np.zeros((0, 26), bool)   # some video follows a correction made in another one there
        self.last_follow = {"moved_px": 0.0, "refused": {}}   # what the last follow() did
        self.err_px = [np.zeros((0, 26)) for _ in range(V)]
        self.used = np.zeros((V, 0, 26), bool)   # which cameras each 3D point was built from
        self.err_cm = np.zeros((0, 26))      # how much the cameras disagree on each point (with three or
        #                                      more: how far the second most distant camera is from the solution)
        self.bone_cm = np.zeros((0, 26))     # how much the bones meeting at each point are off their length
        self.jump_cm = np.zeros((0, 26))     # how far each point jumps away from its neighbours in time
        self.issue_cm = np.zeros((0, 26))    # the largest of the three: what is worth checking
        self.scale = 1.0

    # two-camera shorthands
    @property
    def offset(self) -> float | None:
        return None if self.offsets is None else self.offsets[1]

    @property
    def R(self):
        return None if self.cams is None else self.cams[1][0]

    @property
    def d(self):
        return None if self.cams is None else self.cams[1][1]

    # ── time ───────────────────────────────────────────────
    @property
    def frame_count(self) -> int:
        return len(self.rows)

    @property
    def solved(self) -> bool:
        return self.cams is not None and len(self.rows) > 0

    @property
    def modified(self) -> bool:
        return self.revision != self.saved_revision

    def frame_of(self, view: int, row: int) -> int:
        """Video frame of `view` shown at reconstruction row `row`."""
        if view == 0:
            return int(self.rows[row])
        return int(np.abs(self.views[view].t - (self.t[row] - self.offsets[view])).argmin())

    def row_of(self, view: int, frame: int) -> int:
        t = self.views[view].t[frame] + self.offsets[view]
        return int(np.abs(self.t - t).argmin()) if len(self.t) else 0

    def grid(self, offsets: list):
        """Frames of the first view inside the time span common to all views, and their times."""
        a = self.views[0]
        keep = np.ones(len(a.t), bool)
        for v, off in zip(self.views[1:], offsets[1:]):
            keep &= (a.t >= v.t[0] + off - 1e-9) & (a.t <= v.t[-1] + off + 1e-9)
        rows = np.nonzero(keep)[0]
        return rows, a.t[rows]

    def _pair(self) -> Pair:
        p = Pair(self.views[0], self.views[1])
        p.offset = self.offsets[1]
        if self.cams is not None:
            p.R, p.d = self.cams[1]
        return p

    # ── solving ────────────────────────────────────────────
    def solve(self, progress=None):
        """Everything from scratch: synchronize, locate the cameras, reconstruct.
        Every video is first matched with the first one; with three or more, the cameras are
        then put on one common scale and adjusted together."""
        V = len(self.views)
        self.cams = None
        pairs = []
        for i in range(1, V):
            p = Pair(self.views[0], self.views[i])
            try:
                p.solve(progress=(lambda k, n, i=i: progress((i - 1) * n + k, (V - 1) * n)) if progress else None)
            except ValueError as e:
                raise ValueError(f"video {i + 1}: {e}") from e
            pairs.append(p)
        self.offsets = [0.0] + [p.offset for p in pairs]
        if V == 2:
            self.base, self.cams = 1, [(I3, Z3), (pairs[0].R, pairs[0].d)]
            self.rms, self.agreement = pairs[0].rms, pairs[0].agreement
        else:
            # the 3D starts from the pair whose bones keep their length best; the unit of length
            # is the distance between its two cameras
            o = self._sample()

            def spread(i, p):
                return G.bone_spread(G.triangulate(o["n"][0], o["n"][i], I3, Z3, p.R, p.d))
            self.base = 1 + int(np.argmin([spread(i, p) for i, p in enumerate(pairs, start=1)]))
            self.cams = self._chain(pairs)
            self._refine()
            self._refine(max_nfev=20)          # again, now that more points agree with the cameras
        self.update(recalibrate=False)

    def _sample(self, frames: int = 150) -> dict:
        o = observe(self.views, self.offsets)
        step = max(1, int(round(len(o["rows"]) / frames)))
        return {k: (v[:, ::step] if k in ("n", "ok", "w", "manual", "follow") else v[::step]) for k, v in o.items()}

    def _chain(self, pairs: list) -> list:
        """Each pair knows the direction to its camera but in its own unit of length: express
        them all in the base pair's unit, using the 3D points that pair reconstructs."""
        o = self._sample()
        pb = pairs[self.base - 1]
        X = G.triangulate(o["n"][0], o["n"][self.base], I3, Z3, pb.R, pb.d)
        cams = [(I3, Z3)]
        for i, p in enumerate(pairs, start=1):
            if i == self.base:
                cams.append((p.R, p.d))
                continue
            placed = G.resect(X, o["n"][i], o["ok"][i] & o["ok"][0] & o["ok"][self.base], self.views[i].focal)
            if placed is not None:                  # put it where the first pair's 3D says it is
                cams.append(placed)
                continue
            try:
                cams.append((p.R, p.d * G.baseline_scale(X, o["n"][i], p.R, p.d)))
            except ValueError as e:
                raise ValueError(f"video {i + 1}: {e}") from e
        return cams

    def _refine(self, frames: int = 120, max_nfev: int = 30):
        """Adjust all cameras together (three or more)."""
        o = self._sample(frames)
        V = len(self.views)
        x = o["n"].reshape(V, -1, 2)
        seen = o["ok"].reshape(V, -1)
        self.cams, self.rms, self.agreement = G.refine_cameras(
            x, o["w"].reshape(V, -1) * seen, self.cams, self.base, [v.focal for v in self.views],
            o["manual"].reshape(V, -1), max_nfev=max_nfev, shape=o["n"].shape[1:3])

    def calibrate(self, warm: bool = True):
        """Locate the cameras again from the current points (after a correction)."""
        if len(self.views) == 2:
            p = self._pair()
            if not warm:
                p.R = p.d = None
            p.calibrate(warm=warm)
            self.cams = [(I3, Z3), (p.R, p.d)]
            self.rms, self.agreement = p.rms, p.agreement
        elif warm:
            self._refine(frames=60, max_nfev=5)      # a correction moves the cameras very little: a quick touch-up
        else:
            self._refine()

    def update(self, recalibrate: bool = True):
        """Recompute the 3D points after a correction (and, optionally, the cameras)."""
        if self.offsets is None:
            return
        if recalibrate or self.cams is None:
            try:
                self.calibrate(warm=self.cams is not None)
            except ValueError:
                if self.cams is None:
                    raise
        V = len(self.views)
        o = observe(self.views, self.offsets)
        focals = [v.focal for v in self.views]
        cm = np.array([100 * self.height_m / v.body_px() for v in self.views])[:, None, None]
        weight = np.where(o["manual"], MANUAL_WEIGHT, 1.0)
        used = o["ok"] & (o["ok"].sum(axis=0) >= 2)
        X = G.triangulate_n(o["n"], self.cams, weight * used)
        with np.errstate(invalid="ignore"):
            err = G.view_errors(X, o["n"], self.cams, focals) * cm          # (V, N, K) centimetres
            if V >= 3:
                self._outvote(o, weight, used, X, err, cm, focals, self.scale_for(X))
            for v, (R, t) in enumerate(self.cams):
                X[((X @ np.asarray(R).T + t)[..., 2] <= 0) & used[v]] = np.nan      # behind a camera
        err[:, ~np.isfinite(X).all(axis=-1)] = np.nan
        self.scale = self.scale_for(X)
        self.rows, self.t, self.X_raw, self.used = o["rows"], o["t"], X * self.scale, used
        self.followed = o["follow"].any(axis=0)        # (rows, 26): some video follows a correction there
        self.err_px = [err[v] / cm[v] for v in range(V)]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if V == 2:
                self.err_cm = np.nanmax(np.where(used, err, np.nan), axis=0)
            else:           # at least two cameras this far from the solution (one alone is outvoted by the rest)
                second = np.sort(np.where(used & np.isfinite(err), err, -np.inf), axis=0)[-2]
                self.err_cm = np.where(np.isfinite(second), second, np.nan)
        self.bone_cm = G.bone_inconsistency(self.X_raw) * 100      # found before smoothing hides them
        if self.feet == "rigid":            # a foot's shape is not the detector's any more (the ankle still is)
            self.bone_cm[:, [G.IDX[f"{side}_{p}"] for side in ("left", "right") for p in G.FOOT]] = np.nan
        self.jump_cm = G.jumps(self.X_raw) * 100
        self.issue_cm = np.fmax(np.fmax(self.err_cm, self.bone_cm), self.jump_cm)
        self.issue_cm[~np.isfinite(self.err_cm)] = np.nan
        X = self.X_raw
        if self.feet == "rigid":
            X = G.rigid_feet(X, np.nan_to_num(self.err_cm, nan=PROBLEM_CM))
        fps = 1.0 / float(np.median(np.diff(self.t))) if len(self.t) > 1 else 30.0
        # light smoothing is the one for fast gestures: it leaves the jumps, a real peak can look like one
        self.X = G.smooth(X, fps, self.smooth_hz, spikes=self.smooth_hz < FAST_SMOOTH_HZ) if self.smooth_hz else X
        self.revision += 1

    def bone_lengths(self, source: str = "typical") -> dict:
        """(joint, joint) -> metres of the thighs and shanks: their typical length in this clip (the
        median of the detection), or, with "measured", the tape's where the record has it."""
        tape = self.subject.scale_lengths() if self.subject and source == "measured" else {}
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for a, b in G.LEG_BONES:
                side = a.split("_")[0]
                bone = "thigh" if a.endswith("hip") else "shank"
                L = np.linalg.norm(self.X_raw[:, G.IDX[a]] - self.X_raw[:, G.IDX[b]], axis=1)
                out[(a, b)] = tape.get(f"{side}_{bone}", float(np.nanmedian(L)) if np.isfinite(L).any() else np.nan)
        return {k: v for k, v in out.items() if np.isfinite(v)}

    def _outvote(self, o: dict, weight, used, X, err, cm, focals, unit_m: float):
        """Three or more cameras: leave out the camera that got a point wrong.

        A camera is wrong where, without it, the rest agree and it puts the point more than
        REJECT_CM away from where they do. It is left out of the 3D point only where the point
        built with all the cameras jumps off the path its neighbouring frames give it (the
        running median) and leaving the camera out lands it closer to that path. Elsewhere the
        point keeps every camera, and the wrong one only shows in the errors (what the
        doubtful points are found on), measured from where the others put the point.
        Leaving a camera out wherever the rest agreed made a cyclist filmed from the side,
        behind and in front lose the side camera for a frame here and there: the two others
        agree with each other easily, front to back is what they see worst, and the point
        jumped. Averaging every camera was as accurate as outvoting on the laboratory data.
        Points placed by hand are never left out. Changes used, X and err in place.
        `unit_m`: metres per unit of X."""
        worst = np.where(used, err, -np.inf).max(axis=0)
        fps = 1.0 / float(np.median(np.diff(o["t"]))) if len(o["t"]) > 1 else 30.0
        frames, near = G.path_frames(fps), OFF_PATH_CM / 100 / unit_m
        X0 = X.copy()

        def off(Y, path):
            d = np.linalg.norm(Y - path, axis=-1)
            return np.where(d <= near, 0.0, d)         # on the path: keeps every camera
        path = G.path_of(X0, frames)
        # only the points where the cameras disagree or that leave their path need the trials
        idx = np.nonzero((used.sum(axis=0) >= 3) & ((worst > PROBLEM_CM) | (off(X0, path) > 0)))
        if not len(idx[0]):
            return
        n, u = o["n"][:, idx[0], idx[1]], used[:, idx[0], idx[1]]            # (V, P, 2), (V, P)
        w, manual, hit = weight[:, idx[0], idx[1]], o["manual"][:, idx[0], idx[1]], worst[idx]
        tries = []                            # (camera, X without it, errors, where it may be left out)
        for v in range(len(self.views)):
            far = u[v] & ~manual[v]
            if not far.any():
                continue
            trial = u.copy()
            trial[v] = False
            Xv = G.triangulate_n(n, self.cams, w * trial)
            ev = G.view_errors(Xv, n, self.cams, focals) * cm[:, :, 0]
            far &= ev[v] > REJECT_CM                   # far from where the others put the point
            rest = np.where(trial & np.isfinite(ev), ev, -np.inf).max(axis=0)
            tries.append((v, Xv, ev, far, rest))
        if not tries:
            return
        # the wrong camera, frame by frame: the one without which the others agree best
        wrong, best = np.full(len(hit), -1), np.full(len(hit), np.inf)
        for k, (_, _, _, far, rest) in enumerate(tries):
            better = far & (hit > PROBLEM_CM) & (rest < best)
            wrong[better], best[better] = k, rest[better]
        wrong[~(best < 0.5 * hit)] = -1
        # the 3D point: all the cameras, unless it leaves its path (twice: the second path is cleaner)
        for _ in range(2):
            gap = off(X0[idx], path[idx])
            choice, chosen = np.full(len(hit), -1), X0[idx]
            for k, (_, Xv, _, far, _) in enumerate(tries):
                d = np.linalg.norm(Xv - path[idx], axis=-1)
                better = far & np.isfinite(d) & (d < gap)
                choice[better], gap[better], chosen[better] = k, d[better], Xv[better]
            X[...] = X0
            X[idx] = chosen
            path = G.path_of(X, frames)
        for k, (v, _, ev, _, _) in enumerate(tries):
            m = choice == k
            used[v, idx[0][m], idx[1][m]] = False
            shown = m | ((choice < 0) & (wrong == k))
            err[:, idx[0][shown], idx[1][shown]] = ev[:, shown]

    def view_error_cm(self, view: int, row: int, joint: int) -> float:
        """How far `view` puts this point from where the 3D solution has it, in centimetres."""
        return float(self.err_px[view][row, joint] * 100 * self.height_m / self.views[view].body_px())

    def worst_view(self, row: int, joint: int) -> int:
        """The video that disagrees most with the 3D solution on this point."""
        e = [self.view_error_cm(v, row, joint) for v in range(len(self.views))]
        return int(np.nanargmax(e)) if np.isfinite(e).any() else 0

    def project(self, view: int, row: int, P: np.ndarray) -> np.ndarray:
        """3D points (metres, (..., 3)) seen in `view`'s video frame of `row` (pixels)."""
        v = self.views[view]
        n = G.reproject(np.asarray(P, float) / self.scale, *self.cams[view])
        px = n * np.array([v.K[0, 0], v.K[1, 1]]) + v.K[:2, 2]
        return v.to_frame(self.frame_of(view, row), np.atleast_2d(px)).reshape(np.shape(px))

    # ── bone lengths ───────────────────────────────────────
    def bone_deviation(self) -> np.ndarray:
        """(rows, 4): how much longer (+) or shorter (−) than its usual length each thigh and shank
        (G.LEG_BONES) is, as a fraction, on the points as detected. Usual: the length measured with a
        tape in the patient's record, or else the typical length in the clip (tape_bones() says which).
        With a pedal stroke the typical length can be wrong all along: a hip the detector slides along
        the thigh with the leg up gives a median halfway between the two."""
        key = (self.revision, tuple(sorted((self.subject.scale_lengths() if self.subject else {}).items())))
        if getattr(self, "_deviation_key", None) != key:
            ref = self.bone_lengths("measured")
            out = np.full((len(self.X_raw), len(G.LEG_BONES)), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                for k, (a, b) in enumerate(G.LEG_BONES):
                    if (a, b) in ref:
                        L = np.linalg.norm(self.X_raw[:, G.IDX[a]] - self.X_raw[:, G.IDX[b]], axis=1)
                        out[:, k] = L / ref[(a, b)] - 1
            self._deviation, self._deviation_ref, self._deviation_key = out, ref, key
        return self._deviation

    def tape_bones(self) -> set:
        """The thighs and shanks (G.LEG_BONES pairs) whose usual length is the tape's."""
        tape = self.subject.scale_lengths() if self.subject else {}
        return {(a, b) for a, b in G.LEG_BONES if f"{a.split('_')[0]}_{'thigh' if a.endswith('hip') else 'shank'}" in tape}

    def bone_guides(self, row: int, pct: float = GUIDE_PCT) -> list[dict]:
        """Where to move a point so that the thighs and shanks off their usual length by `pct` % or
        more in this row get it back. For each bone, the end to move is the one the cameras are
        least sure of (their disagreement, a jump, and whether all its bones are off). A knee that
        both its bones blame gets one place that suits both (of the circle where it keeps both
        lengths, the point nearest where it is now). Each guide: joint (index), bones [(name a,
        name b, dev fraction, length m, usual m)], target (3D, metres), long (the worst bone is
        too long)."""
        dev = self.bone_deviation()
        if not 0 <= row < len(dev):
            return []
        ref = self._deviation_ref
        blamed: dict = {}
        for k, (a, b) in enumerate(G.LEG_BONES):
            d = dev[row, k]
            if not np.isfinite(d) or abs(d) * 100 < pct:
                continue
            ia, ib = G.IDX[a], G.IDX[b]

            def unsure(j):
                return float(np.nansum([self.bone_cm[row, j], self.err_cm[row, j], self.jump_cm[row, j]]))
            ua, ub = unsure(ia), unsure(ib)
            # alike: the hip of a thigh, the ankle of a shank (the knee is shared: its bones would both be off)
            suspect = ia if ua > ub or (ua == ub and a.endswith("hip")) else ib
            trusted = ib if suspect == ia else ia
            L = float(np.linalg.norm(self.X_raw[row, ia] - self.X_raw[row, ib]))
            blamed.setdefault(suspect, []).append((a, b, float(d), L, ref[(a, b)], trusted))
        out = []
        for j, bones in blamed.items():
            P = self.X_raw[row, j]
            if len(bones) == 2:                      # a knee: both lengths at once
                (_, _, _, _, l1, h), (_, _, _, _, l2, f) = bones
                target = _two_lengths(self.X_raw[row, h], self.X_raw[row, f], l1, l2, P)
            else:
                _, _, _, _, usual, t = bones[0]
                Pt = self.X_raw[row, t]
                target = Pt + (P - Pt) / np.linalg.norm(P - Pt) * usual
            worst = max(bones, key=lambda x: abs(x[2]))
            out.append({"joint": j, "bones": [x[:5] for x in bones], "target": target, "long": worst[2] > 0})
        return out

    def projection(self, view: int, row: int) -> np.ndarray:
        """The 3D skeleton of `row` drawn back on `view`'s video frame (pixels), (26, 2)."""
        v = self.views[view]
        n = G.reproject(self.X[row] / self.scale, *self.cams[view])
        px = n * np.array([v.K[0, 0], v.K[1, 1]]) + v.K[:2, 2]
        return v.to_frame(self.frame_of(view, row), px)

    def epipolar_line(self, view: int, row: int, joint: int, other: int | None = None):
        """Where a point of `view` can be in the video `other`: two pixels of that line."""
        other = 1 - view if other is None else other
        v, o = self.views[view], self.views[other]
        f = self.frame_of(view, row)
        p = v.xy[f, joint].astype(float)
        if not np.isfinite(p).all() or self.cams is None:
            return None
        n = G.normalize(v.from_frame(f, p[None])[0], v.K)
        (Rv, tv), (Ro, to) = self.cams[view], self.cams[other]
        unit = np.linalg.norm(self.cams[self.base][1])
        depths = np.geomspace(0.02, 40.0, 60)[:, None] * unit
        world = -Rv.T @ tv + (Rv.T @ np.array([n[0], n[1], 1.0])) * depths       # the ray of that point
        cam = world @ Ro.T + to
        cam = cam[cam[:, 2] > 1e-3 * unit]                      # the part in front of the other camera
        if len(cam) < 2:
            return None
        px = cam[[0, -1], :2] / cam[[0, -1], 2:3] * np.array([o.K[0, 0], o.K[1, 1]]) + o.K[:2, 2]
        px = o.to_frame(self.frame_of(other, row), px)
        return px[0], px[1]

    # ── quality ────────────────────────────────────────────
    def quality(self) -> dict:
        """Numbers that tell how far the reconstruction can be trusted, with no reference system."""
        e = self.err_cm[:, G.BODY]
        ok = np.isfinite(e)
        followed = getattr(self, "followed", np.zeros(self.err_cm.shape, bool))[:, G.BODY]
        e = np.where(followed, np.nan, e)          # made to agree: it would make the videos look closer than they are
        bones = G.bone_variation(self.X_raw)               # as detected: fixing the bones would hide it
        legs = float(np.nanmean([bones["thigh"], bones["shank"]]))
        med = float(np.nanmedian(e)) if np.isfinite(e).any() else float("nan")
        level = "good" if med < 2.0 and legs < 6 else "fair" if med < 4.0 and legs < 12 else "poor"
        return {"mismatch_cm": med, "mismatch_p90_cm": float(np.nanpercentile(e, 90)) if np.isfinite(e).any() else float("nan"),
                "reprojection_px": self.rms, "agreement": self.agreement, "bones": bones, "legs_variation": legs,
                "reconstructed": float(ok.mean()) if e.size else 0.0,
                "problems": int(self.doubtful().sum()),
                "manual": int(sum(v.manual().sum() for v in self.views)),
                "followed": int(sum((v.source == SRC_FOLLOW).sum() for v in self.views)), "level": level}

    def frame_error(self) -> np.ndarray:
        """Worst issue of each row, in centimetres (NaN where nothing was reconstructed)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(self.issue_cm[:, G.BODY], axis=1) if self.issue_cm.size else np.zeros(0)

    def problems(self) -> list[tuple[int, int, float]]:
        """(row, joint, cm) of the points worth checking, worst first: the two cameras disagree
        on them, the bones that meet there do not have their usual length, or they jump away
        from where the frames before and after put them."""
        e = self.issue_cm[:, G.BODY]
        r, k = np.nonzero(self.doubtful()[:, G.BODY])
        order = np.argsort(-e[r, k])
        return [(int(r[i]), int(G.BODY[k[i]]), float(e[r[i], k[i]])) for i in order]

    def worst_joint(self, row: int) -> int:
        """The point of `row` most worth checking."""
        return int(G.BODY[int(np.nanargmax(self.issue_cm[row, G.BODY]))])

    def next_problem(self, row: int) -> int | None:
        """The next row after `row` with a point worth checking (wraps around)."""
        bad = np.nonzero(self.doubtful().any(axis=1))[0]
        if not len(bad):
            return None
        later = bad[bad > row]
        return int(later[0]) if len(later) else int(bad[0])

    # ── review ─────────────────────────────────────────────
    def _reviewed_mask(self) -> np.ndarray:
        m = np.zeros(self.issue_cm.shape, bool)
        if self.reviewed and len(self.rows):
            f, j = np.array(sorted(self.reviewed)).T
            r = np.searchsorted(self.rows, f)
            ok = (r < len(self.rows)) & (self.rows[np.minimum(r, len(self.rows) - 1)] == f)
            m[r[ok], j[ok]] = True
        return m

    def doubtful(self, reviewed: bool = False) -> np.ndarray:
        """(rows, 26): the points worth checking; without `reviewed`, not those marked as fine."""
        with np.errstate(invalid="ignore"):
            bad = self.issue_cm > PROBLEM_CM
        keep = np.zeros(bad.shape[1], bool)
        keep[G.BODY] = True
        bad &= keep
        return bad if reviewed else bad & ~self._reviewed_mask()

    def why(self, row: int, joint: int) -> str:
        """Why a point is doubtful: "jump", "bone" or "disagree"."""
        cm, bone, jump = self.err_cm[row, joint], self.bone_cm[row, joint], self.jump_cm[row, joint]
        if not cm > PROBLEM_CM and jump > PROBLEM_CM and not bone > jump:
            return "jump"
        if not cm > PROBLEM_CM and bone > PROBLEM_CM:
            return "bone"
        return "disagree"

    def stretches(self) -> list[dict]:
        """The doubtful points in stretches: one per point and run of frames, worst first. Runs
        less than 0.15 s apart are one: a point wrong for a few frames in a row only jumps at the
        two ends (in between it is wrong by the same amount), and away and back means the frames
        between are wrong too. Each: joint, first and last row, its worst row and how far off it
        is there (cm), why, the video that disagrees most there, and whether the user marked it
        as fine."""
        bad, mine = self.doubtful(reviewed=True), self._reviewed_mask()
        fps = 1.0 / float(np.median(np.diff(self.t))) if len(self.t) > 1 else 30.0
        gap = G.path_frames(fps)
        out = []
        for j in G.BODY:
            rows = np.nonzero(bad[:, j])[0]
            if not len(rows):
                continue
            for run in np.split(rows, np.nonzero(np.diff(rows) > gap)[0] + 1):
                a, b = int(run[0]), int(run[-1])
                worst = a + int(np.nanargmax(np.where(bad[a:b + 1, j], self.issue_cm[a:b + 1, j], -np.inf)))
                out.append({"joint": int(j), "first": a, "last": b, "worst": worst,
                            "cm": float(self.issue_cm[worst, j]), "why": self.why(worst, j),
                            "view": self.worst_view(worst, j), "reviewed": bool(mine[run, j].all())})
        return sorted(out, key=lambda e: -e["cm"])

    def stretch_at(self, row: int, joint: int) -> tuple[int, int] | None:
        """First and last row of the doubtful stretch of `joint` around `row`."""
        for e in self.stretches():
            if e["joint"] == joint and e["first"] - 1 <= row <= e["last"] + 1:
                return e["first"], e["last"]
        return None

    def set_reviewed(self, joint: int, first: int, last: int, on: bool = True):
        """Mark the rows first..last of `joint` as checked and fine (or not)."""
        keys = {(int(self.rows[r]), int(joint)) for r in range(first, last + 1)}
        self.reviewed = (self.reviewed | keys) if on else (self.reviewed - keys)
        self._touch()

    # ── corrections ────────────────────────────────────────
    def _touch(self):
        self.revision += 1

    def move_point(self, view: int, frame: int, joint: int, new, scope: str = "frame", radius: int = 3,
                   old=None) -> Edit:
        """Put `joint` of `view` at pixel `new` in `frame`.
        scope "frame": only this frame; "neighbors": spread the correction over ±radius frames
        with a smooth falloff; "stretch": the same correction over the whole doubtful stretch of
        this point (stretches()), fading out over radius frames on either side. Points already
        placed by hand are never moved by the spreading."""
        v = self.views[view]
        r = radius if scope in ("neighbors", "stretch") else 0
        lo = hi = frame                                   # frames that take the whole correction
        if scope == "stretch" and len(self.t):
            found = self.stretch_at(self.row_of(view, frame), joint)
            if found:
                lo, hi = sorted((self.frame_of(view, found[0]), self.frame_of(view, found[1])))
                lo, hi = min(lo, frame), max(hi, frame)
        start, end = max(0, lo - r), min(v.frame_count - 1, hi + r)
        before = _snap(v, start, end)
        if old is not None:
            before[0][frame - start, joint] = old
        delta = np.asarray(new, np.float32) - before[0][frame - start, joint]
        if not np.isfinite(delta).all():
            delta = np.zeros(2, np.float32)
        for f in range(start, end + 1):
            if f == frame:
                v.xy[f, joint], v.source[f, joint], v.score[f, joint] = new, SRC_MANUAL, 1.0
            elif v.source[f, joint] != SRC_MANUAL and np.isfinite(v.xy[f, joint]).all():
                gap = lo - f if f < lo else f - hi if f > hi else 0
                w = 0.5 * (1 + np.cos(np.pi * gap / (r + 1)))
                v.xy[f, joint] = before[0][f - start, joint] + delta * w
                v.source[f, joint] = SRC_SPREAD
        self._touch()
        return Edit(view, start, end, before, _snap(v, start, end))

    def follow(self, view: int, edit: Edit, joint: int) -> list[Edit]:
        """After a correction of `joint` in `view` (`edit`, as move_point returns it), the other
        videos follow it, so that they do not come apart from it.

        Over the moments the correction covers, each frame of every other video gets the point
        where the corrected 3D point falls in it. That 3D point lies on the ray of the hand-placed
        point; how deep along it is decided by the videos that were not corrected (the point of the
        ray they agree with best), so a video that was right barely moves: only as much as it
        takes to agree with the correction. Where two or more videos were corrected by hand there,
        the 3D point comes from those alone. A point placed by hand is never moved; the ones moved
        here are marked SRC_FOLLOW. The cameras do not move.

        Where a video would have to move more than FOLLOW_LIMIT of the person's height in it, it
        does not agree with the correction at that moment and keeps its point there
        (self.last_follow["refused"]: video -> how far it would have had to move, pixels), for the
        user to look at."""
        if self.cams is None or self.offsets is None or len(self.views) < 2:
            return []
        HARD = 1e3                                      # a hand-placed point: the 3D point stays on its ray
        v = self.views[view]
        t0, t1 = v.t[edit.start] + self.offsets[view], v.t[edit.end] + self.offsets[view]
        # a point that follows an earlier correction has no say of its own: where the detector put it does
        stable = [u.stable(np.where((u.source == SRC_FOLLOW)[..., None], u.raw_xy, u.xy)) for u in self.views]
        valid = [np.where(u.source == SRC_FOLLOW, np.isfinite(u.raw_xy).all(axis=2) & (u.raw_score >= MIN_SCORE),
                          u.valid()) for u in self.views]
        clocks = [u.t + off for u, off in zip(self.views, self.offsets)]

        def seen(u: int, tau: float):
            """View u's point at time tau: normalized point, placed by hand, weight (None: none)."""
            tu, good = clocks[u], valid[u][:, joint]
            if tau < tu[0] - 1e-6 or tau > tu[-1] + 1e-6 or not good.any():
                return None
            near = int(np.abs(tu - tau).argmin())
            k = int(np.clip(np.searchsorted(tu, tau), 1, len(tu) - 1))
            # the frame at that moment, or the two on either side of it
            around = (near,) if abs(tu[near] - tau) < 1e-6 or len(tu) == 1 else (k - 1, k)
            if good.sum() >= 2:
                if np.interp(tau, tu, good.astype(float)) <= 0.5:
                    return None
                px = np.array([np.interp(tau, tu[good], stable[u][good, joint, c]) for c in range(2)])
            elif good[near]:
                px = stable[u][near, joint]
            else:
                return None
            hard = any(self.views[u].source[i, joint] in (SRC_MANUAL, SRC_SPREAD, SRC_LEARNED) for i in around)
            return G.normalize(px, self.views[u].K), hard, float(self.views[u].score[near, joint])

        def corrected_point(tau: float):
            """The corrected 3D point at time tau (camera units), or None where the correction does not reach."""
            obs = [seen(u, tau) for u in range(len(self.views))]
            if not obs[view] or not obs[view][1]:
                return None
            hard = sum(o is not None and o[1] for o in obs)
            x = np.full((len(obs), 1, 2), np.nan)
            wt = np.zeros((len(obs), 1))
            for u, o in enumerate(obs):
                if o is None or (hard >= 2 and not o[1]):
                    continue
                x[u, 0], wt[u, 0] = o[0], HARD if o[1] else max(o[2], 0.05)
            if (wt[:, 0] > 0).sum() < 2:
                return None
            P = G.triangulate_n(x, self.cams, wt)[0]
            return P if np.isfinite(P).all() else None

        def stable_at(u: int, tau: float):
            """View u's point at tau from its own frames (first-frame pixels), None if it has no usable one."""
            tu, good = clocks[u], valid[u][:, joint]
            if good.sum() < 2 or np.interp(tau, tu, good.astype(float)) <= 0.5:
                return None
            return np.array([np.interp(tau, tu[good], stable[u][good, joint, c]) for c in range(2)])

        # the 3D exists at the first video's frames; every other video is read between two of its own.
        # The moments of the 3D the correction reaches, and where the corrected point is at each
        T = clocks[0]
        hit = [i for i in range(max(0, int(np.searchsorted(T, t0 - 1e-6)) - 1), min(len(T), int(np.searchsorted(T, t1 + 1e-6)) + 1))
               if corrected_point(float(T[i])) is not None]
        if not hit:
            self.last_follow = {"moved_px": 0.0, "refused": {}}
            return []
        target = {i: corrected_point(float(T[i])) for i in hit}
        out = []
        self.last_follow = {"moved_px": 0.0, "refused": {}}
        for w, vw in enumerate(self.views):
            if w == view:
                continue
            tw = clocks[w]
            first, last = float(T[hit[0]]), float(T[hit[-1]])
            a = max(0, int(np.searchsorted(tw, first - 1e-6)) - 1)
            b = min(len(tw) - 1, int(np.searchsorted(tw, last + 1e-6)))
            frames = [g for g in range(a, b + 1)
                      if valid[w][g, joint] and vw.source[g, joint] not in (SRC_MANUAL, SRC_SPREAD, SRC_LEARNED)]
            if not frames:
                continue
            col = {g: k for k, g in enumerate(frames)}
            # how far the video's point has to move at each moment of the 3D its frames touch (none
            # where the correction does not reach: those keep their point), and how its frames are
            # read there; the frame shifts that best give that
            lo = tw[max(0, frames[0] - 1)]
            hi = tw[min(len(tw) - 1, frames[-1] + 1)]
            rows = [i for i in range(len(T)) if lo < T[i] < hi]
            A, y, wts = [], [], []
            R, t = self.cams[w]
            cap = FOLLOW_LIMIT * vw.body_px()
            refused = 0.0
            for i in rows:
                here = stable_at(w, float(T[i]))
                if here is None:
                    continue
                k = int(np.clip(np.searchsorted(tw, T[i]), 1, len(tw) - 1))
                f = (T[i] - tw[k - 1]) / (tw[k] - tw[k - 1]) if tw[k] > tw[k - 1] else 0.0
                line = np.zeros(len(frames))
                for g, c in ((k - 1, 1 - f), (k, f)):
                    if g in col:
                        line[col[g]] += c
                if not line.any():
                    continue
                if i in target:
                    cam = np.asarray(R) @ target[i] + np.asarray(t)
                    if cam[2] <= 0:
                        continue
                    want = cam[:2] / cam[2] * np.array([vw.K[0, 0], vw.K[1, 1]]) + vw.K[:2, 2] - here
                    if np.linalg.norm(want) <= cap:
                        A.append(line), y.append(want), wts.append(1.0)
                    else:                               # it does not agree with the correction at this moment
                        refused = max(refused, float(np.linalg.norm(want)))
                        A.append(line), y.append(np.zeros(2)), wts.append(0.1)
                else:
                    A.append(line), y.append(np.zeros(2)), wts.append(0.1)    # beyond the correction: stay
            if refused:
                self.last_follow["refused"][w] = refused
            if not A or not any(c == 1.0 for c in wts):
                continue
            Wq = np.sqrt(np.array(wts))[:, None]
            M = np.vstack([np.array(A) * Wq, 0.05 * np.eye(len(frames))])     # of the ways that give it, the smallest
            Y = np.vstack([np.array(y) * Wq, np.zeros((len(frames), 2))])
            d = np.linalg.lstsq(M, Y, rcond=None)[0]
            start, end = frames[0], frames[-1]
            before = _snap(vw, start, end)
            most = float(np.max(np.linalg.norm(d, axis=1)))
            for g, k in col.items():
                vw.xy[g, joint] = vw.to_frame(g, (stable[w][g, joint] + d[k])[None])[0]
                vw.source[g, joint] = SRC_FOLLOW
            out.append(Edit(w, start, end, before, _snap(vw, start, end)))
            self.last_follow["moved_px"] = max(self.last_follow["moved_px"], most)
        if out:
            self._touch()
        return out

    # ── a point's path over the pedal stroke ───────────────
    def pedal_turns(self, side: str) -> np.ndarray | None:
        """(rows,) pedal turns at each row, from that side's ankle going round the crank: it goes up by
        one every turn and is a whole number with the ankle at the top. None if it does not go round."""
        from openkick.multiview.cycling import vertical
        if self.cams is None or not len(self.t):
            return None
        A = self.X_raw[:, G.IDX[f"{side}_ankle"]]
        ok = np.isfinite(A).all(axis=1)
        if ok.sum() < 30:
            return None
        c = np.median(A[ok], axis=0)
        _, S, Vt = np.linalg.svd(A[ok] - c, full_matrices=False)
        if S[1] < 0.35 * S[0]:
            return None                                   # a line, not a circle: no pedalling
        normal, up = Vt[2], vertical(self.cams)
        e_up = up - (up @ normal) * normal
        if np.linalg.norm(e_up) < 0.3:
            return None
        e_up /= np.linalg.norm(e_up)
        e_on = np.cross(normal, e_up)
        k = np.arange(len(A))
        Af = np.stack([np.interp(k, k[ok], A[ok, i]) for i in range(3)], axis=1) - c
        a = np.unwrap(np.arctan2(Af @ e_on, Af @ e_up))        # 0 with the ankle at the top
        if a[-1] < a[0]:
            a = -a
        if a[-1] - a[0] < 2 * np.pi * 2:
            return None                                   # fewer than two turns
        return a / (2 * np.pi)

    def learning(self, view: int, joint: int) -> dict | None:
        """What there is to learn the path of `joint` in `view` over the pedal stroke from: the pedal
        turns at each of its frames, its hand-placed frames and the parts of the stroke they cover
        (LEARN_PARTS, from the top). None where the point has no side or nobody pedals."""
        side = G.HALPE26[joint].split("_")[0]
        if side not in ("left", "right"):
            return None
        turns = self.pedal_turns(side)
        if turns is None:
            return None
        v = self.views[view]
        ft = np.interp(v.t + self.offsets[view], self.t, turns, left=np.nan, right=np.nan)
        hand = (v.source[:, joint] == SRC_MANUAL) & np.isfinite(ft) & np.isfinite(v.raw_xy[:, joint]).all(axis=1)
        parts = set(np.floor((ft[hand] % 1) * LEARN_PARTS).astype(int).tolist())
        return {"turns": ft, "hand": hand, "parts": parts,
                "missing": sorted(set(range(LEARN_PARTS)) - parts)}

    def learn_path(self, view: int, joint: int) -> Edit | None:
        """Learn how the detector misplaces `joint` in `view` at each moment of the pedal stroke, from
        the user's corrections over at least one whole turn, and correct every turn with it.

        What is learned is the correction (hand-placed point minus the detector's), as a smooth
        function that repeats every turn (LEARN_ORDER harmonics): so a point the detector puts too
        far along the thigh with the leg up is moved back with the leg up in every turn, and where
        the detector was right at that moment it is left nearly where it was. Hand-placed points
        are never moved; the others of that point in that video are marked SRC_LEARNED. None if
        some part of the stroke has no correction yet. self.last_learn: frames changed, hand-placed
        points learned from, how close the learned path passes to them and the median move (px)."""
        info = self.learning(view, joint)
        if not info or info["missing"]:
            return None
        v = self.views[view]
        raw, now = v.stable(v.raw_xy)[:, joint], v.stable()[:, joint]
        angle = 2 * np.pi * (info["turns"] % 1)

        def basis(p):
            return np.stack([np.ones_like(p)] + [f(k * p) for k in range(1, LEARN_ORDER + 1) for f in (np.cos, np.sin)], axis=1)
        hand = info["hand"]
        B, Y = basis(angle[hand]), now[hand] - raw[hand]
        ridge = 0.1 * np.eye(B.shape[1])
        ridge[0, 0] = 0.0
        coef = np.linalg.lstsq(np.vstack([B, ridge]), np.vstack([Y, np.zeros((B.shape[1], 2))]), rcond=None)[0]
        fit = float(np.median(np.linalg.norm(B @ coef - Y, axis=1)))
        change = np.isfinite(info["turns"]) & np.isfinite(v.raw_xy[:, joint]).all(axis=1) & (v.source[:, joint] != SRC_MANUAL)
        before = _snap(v, 0, v.frame_count - 1)
        shift = basis(angle[change]) @ coef
        for f, p in zip(np.nonzero(change)[0], raw[change] + shift):
            v.xy[f, joint] = v.to_frame(int(f), p[None])[0]
        v.source[change, joint] = SRC_LEARNED
        self.last_learn = {"frames": int(change.sum()), "hand": int(hand.sum()), "fit_px": fit,
                           "moved_px": float(np.median(np.linalg.norm(shift, axis=1))) if len(shift) else 0.0}
        self._touch()
        return Edit(view, 0, v.frame_count - 1, before, _snap(v, 0, v.frame_count - 1))

    def revert_point(self, view: int, frame: int, joint: int) -> Edit:
        """Back to what the automatic detection said."""
        v = self.views[view]
        before = _snap(v, frame, frame)
        v.xy[frame, joint], v.score[frame, joint] = v.raw_xy[frame, joint], v.raw_score[frame, joint]
        v.source[frame, joint] = SRC_AUTO if np.isfinite(v.raw_xy[frame, joint]).all() else SRC_NONE
        self._touch()
        return Edit(view, frame, frame, before, _snap(v, frame, frame))

    def discard_point(self, view: int, frame: int, joint: int) -> Edit:
        """Tell the program not to use this point here (hidden behind the body, for example)."""
        v = self.views[view]
        before = _snap(v, frame, frame)
        v.score[frame, joint], v.source[frame, joint] = 0.0, SRC_AUTO
        self._touch()
        return Edit(view, frame, frame, before, _snap(v, frame, frame))

    def swap_sides(self, view: int, start: int, end: int, group: str = "legs") -> Edit:
        """Exchange left and right in `view` over a frame range (the detector confused them)."""
        v = self.views[view]
        start, end = max(0, start), min(v.frame_count - 1, end)
        before = _snap(v, start, end)
        perm = G.swap_indices((group,))
        s = slice(start, end + 1)
        v.xy[s], v.score[s], v.source[s] = v.xy[s][:, perm], v.score[s][:, perm], v.source[s][:, perm]
        self._touch()
        return Edit(view, start, end, before, _snap(v, start, end))

    def find_swaps(self) -> dict:
        """Left/right confusions the program can see by itself: {group: [rows of each view]}."""
        V = len(self.views)
        o = observe(self.views, self.offsets)
        if V == 2:
            na = np.where((o["ok"][0] & o["ok"][1])[..., None], o["n"][0], np.nan)
            nb = np.where((o["ok"][0] & o["ok"][1])[..., None], o["n"][1], np.nan)
            found = G.resolve_swaps(na, nb, self.R, self.d, self.views[0].focal,
                                    o["manual"][0].any(axis=1), o["manual"][1].any(axis=1))
            return {g: [s["a"], s["b"]] for g, s in found.items()}
        # three or more: for every frame, would the videos agree better if ONE of them had its sides
        # exchanged? The one that makes them agree is the confused one.
        import warnings
        focals = [v.focal for v in self.views]
        out = {}
        for group in ("legs", "arms"):
            perm = G.swap_indices((group,))
            ids = np.array([G.IDX[x] for pair in G.GROUPS[group] for x in pair])

            def cost(n, ok):
                e = G.view_errors(G.triangulate_n(n, self.cams, ok), n, self.cams, focals)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    return np.nanmean(e, axis=(0, 2))                       # one number per frame

            n0, ok0 = o["n"][:, :, ids], o["ok"][:, :, ids]
            base = cost(n0, ok0)
            best, who = base.copy(), np.full(len(base), -1)
            for v in range(V):
                n1, ok1 = n0.copy(), ok0.copy()
                n1[v], ok1[v] = o["n"][v][:, perm][:, ids], o["ok"][v][:, perm][:, ids]
                c = cost(n1, ok1)
                better = np.isfinite(c) & (c < best)
                best[better], who[better] = c[better], v
            clear = np.isfinite(base) & (best < 0.6 * base) & (base - best > 4.0)
            out[group] = [clear & (who == v) & ~o["manual"][v].any(axis=1) for v in range(V)]
        return out

    def fix_swaps(self) -> tuple[list[Edit], int]:
        """Apply find_swaps(). Returns the undo records and how many frames changed."""
        found = self.find_swaps()
        edits, count = [], 0
        for group, per_view in found.items():
            for view, mask in enumerate(per_view):
                frames = sorted({self.frame_of(view, int(r)) for r in np.nonzero(mask)[0]})
                count += len(frames)
                for f in frames:
                    edits.append(self.swap_sides(view, f, f, group))
        return edits, count

    def scale_for(self, X: np.ndarray) -> float:
        """Reconstruction units to metres. From the thigh and shank lengths measured on the person
        when there are any (right for anyone, standing or not), else from their height (which
        needs them to stand up straight at some moment of the clip)."""
        measured = self.subject.scale_lengths() if self.subject else {}
        ratios = []
        with np.errstate(invalid="ignore"):
            for key, metres in measured.items():
                side, bone = key.split("_", 1)
                a, b = ("hip", "knee") if bone == "thigh" else ("knee", "ankle")
                v = np.linalg.norm(X[:, G.IDX[f"{side}_{a}"]] - X[:, G.IDX[f"{side}_{b}"]], axis=1)
                v = v[np.isfinite(v)]
                if len(v) > 10 and np.median(v) > 0:
                    ratios.append(metres / float(np.median(v)))
        if ratios:
            return float(np.median(ratios))
        return G.scale_from_height(X, self.height_m)

    # ── files ──────────────────────────────────────────────
    def save(self, path: str | Path):
        """Write the session (points, corrections, cameras) to one file. The videos are not
        copied: their location is kept, relative to the session file when they are in its
        folder or below, so the folder can be moved or sent as a whole."""
        path = Path(path)
        folder = path.resolve().parent
        videos = []
        for v in self.views:
            d = v.meta.to_dict()
            full = Path(v.meta.path).resolve()
            d["path"] = str(full)
            try:
                d["relative"] = str(full.relative_to(folder))
            except ValueError:
                d["relative"] = None
            videos.append(d)
        meta = {"format": FORMAT, "height_m": self.height_m, "offsets": self.offsets, "base": self.base, "videos": videos,
                "template": self.template, "subject": self.subject.to_dict() if self.subject else {},
                "smooth_hz": self.smooth_hz, "feet": self.feet, "report_notes": self.report_notes,
                "report_evaluator": self.report_evaluator,
                "reviewed": sorted([int(f), int(j)] for f, j in self.reviewed),
                "cams": None if self.cams is None else [[np.asarray(R).tolist(), np.asarray(t).tolist()] for R, t in self.cams]}
        arrays = {"meta": np.array(json.dumps(meta))}
        for i, v in enumerate(self.views):
            arrays.update({f"t{i}": v.t, f"raw_xy{i}": v.raw_xy, f"raw_score{i}": v.raw_score, f"xy{i}": v.xy,
                           f"score{i}": v.score, f"source{i}": v.source, f"K{i}": v.K})
            if v.motion is not None:
                arrays[f"motion{i}"] = v.motion
        tmp = path.with_name(path.name + ".tmp")           # never leave a half-written session behind
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(path)
        self.path = path
        self.saved_revision = self.revision

    @staticmethod
    def read(path: str | Path) -> dict:
        """What a session file holds, without rebuilding anything: {"meta", "arrays", "videos"}.
        "videos" lists, for each one, where it should be and whether it is there."""
        path = Path(path)
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        except Exception as e:  # noqa: BLE001 - any unreadable file gets the same plain message
            raise ValueError("this is not an OpenKick 3D session, or the file is damaged") from e
        if meta.get("format", 1) > FORMAT:
            raise ValueError("this session was saved with a newer version of OpenKick")
        videos = []
        for vm in meta["videos"]:
            # where it was, where it would be if the folder was moved, and next to the session by name
            tries = [Path(vm["path"])]
            if vm.get("relative"):
                tries.insert(0, path.resolve().parent / vm["relative"])
            tries.append(path.resolve().parent / Path(vm["path"]).name)
            found = next((str(t) for t in tries if t.exists()), None)
            videos.append({"name": Path(vm["path"]).name, "path": found, "frames": len(vm["pts"])})
        return {"meta": meta, "arrays": arrays, "videos": videos, "file": path}

    @classmethod
    def load(cls, path: str | Path, videos: list | None = None, require_videos: bool = True) -> "Session":
        """Open a saved session. `videos` gives the place of each video when it is not where the
        session expects it (one path per video, None to use the expected place). With
        require_videos=False a missing video is not an error (the points can still be used)."""
        data = cls.read(path)
        meta, z = data["meta"], data["arrays"]
        views = []
        for i, (vm, info) in enumerate(zip(meta["videos"], data["videos"])):
            where = (videos[i] if videos and videos[i] else None) or info["path"]
            if where is None:
                if require_videos:
                    raise FileNotFoundError(info["name"])
                where = vm["path"]
            vm = dict(vm, path=str(where))
            vm.pop("relative", None)
            v = View(VideoMeta.from_dict(vm), z[f"t{i}"], z[f"raw_xy{i}"], z[f"raw_score{i}"],
                     z[f"motion{i}"] if f"motion{i}" in z else None, z[f"K{i}"])
            v.xy, v.score, v.source = z[f"xy{i}"].copy(), z[f"score{i}"].copy(), z[f"source{i}"].copy()
            views.append(v)
        s = cls(views, meta["height_m"])
        from openkick.core.subject import Subject
        s.template = meta.get("template")
        s.subject = Subject.from_dict(meta.get("subject")) if meta.get("subject") else Subject(height_m=s.height_m)
        s.smooth_hz = float(meta.get("smooth_hz", SMOOTH_HZ) or 0.0)     # older sessions get it too
        s.feet = meta.get("feet") or ""
        s.report_notes = meta.get("report_notes") or ""
        s.report_evaluator = meta.get("report_evaluator") or ""
        s.reviewed = {(int(f), int(j)) for f, j in meta.get("reviewed", [])}
        if meta.get("format", 1) == 1:                  # sessions saved when only two videos were possible
            s.offsets = [0.0, meta["offset"]]
            cams = None if meta["R"] is None else [[np.eye(3), np.zeros(3)], [meta["R"], meta["d"]]]
        else:
            s.offsets, s.base, cams = meta["offsets"], meta["base"], meta["cams"]
        if cams is not None:
            s.cams = [(np.array(R, float), np.array(t, float)) for R, t in cams]
            s.update(recalibrate=False)
        s.path = Path(path)
        s.saved_revision = s.revision
        return s

    @staticmethod
    def same_video(path: str | Path, frames: int) -> bool:
        """Is the file at `path` the video a session was made from? (same number of frames)"""
        from openkick.core.video import probe
        try:
            return probe(path).frame_count == frames
        except Exception:  # noqa: BLE001
            return False
