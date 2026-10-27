"""3D reconstruction from two or more cameras scored against the laboratory (OpenCap dataset).

"ceiling"  triangulate with the dataset's own camera calibration (checkerboard).
"body"     what OpenKick does (openkick.multiview): no calibration at all; time offset, camera
           pose and scale are solved from the two skeletons and the person's height.

Both are scored with 3D joint angles against the marker-based mocap, and the recovered cameras
and time offset are compared with the true ones.

    python -m validation.multiview --subject subject2 --trial squats1 --cams Cam0 Cam4
    python -m validation.multiview --batch squats DJ walking STS
    python -m validation.multiview --batch squats --cams Cam0 Cam2 Cam4
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from validation import opencap as V

from openkick.multiview import geometry as G
from openkick.multiview.geometry import (HALPE26, IDX, guess_intrinsics, normalize, reproject,  # noqa: F401
                                         resample, scale_from_height, sync_offset, triangulate)
from openkick.multiview.session import Session, View

I3, Z3 = np.eye(3), np.zeros(3)


# ── 2D detection (all 26 points, cached) ───────────────────────────────


def detect_full(video: Path, mode: str = "balanced"):
    """times (N,), keypoints (N, 26, 2) in pixels, scores (N, 26)."""
    key = "_".join(video.parts[-6:]).replace(".avi", "")
    cache = V.CACHE / f"halpe26-{mode}" / f"{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["xy"], z["score"]
    import cv2
    from rtmlib import BodyWithFeet, PoseTracker

    from openkick.core.video import iter_frames, probe
    model = PoseTracker(BodyWithFeet, det_frequency=10, tracking=False, mode=mode, backend="onnxruntime",
                        device="cpu", to_openpose=False)
    meta = probe(video)
    n = meta.frame_count
    xy = np.full((n, 26, 2), np.nan)
    score = np.zeros((n, 26))
    prev = None
    for i, rgb in iter_frames(meta):
        k, s = model(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if k is None or len(k) == 0:
            continue
        k, s = np.asarray(k), np.asarray(s)
        c = k[:, [11, 12]].mean(axis=1)
        j = int(np.argmax(np.ptp(k[:, :, 1], axis=1))) if prev is None else int(np.argmin(np.linalg.norm(c - prev, axis=1)))
        prev = c[j]
        xy[i], score[i] = k[j], s[j]
    t = np.array([meta.time_of(i) for i in range(n)])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=t, xy=xy, score=score)
    return t, xy, score


# ── cameras ────────────────────────────────────────────────────────────


def load_camera(subject: str, trial: str, cam: str) -> dict:
    """The dataset's calibration: K (3x3), dist (5,), R (3x3), t (3,) with x_cam = R X + t (mm)."""
    s = V.session_of(subject, trial)
    p = pickle.load(open(V.ROOT / subject / "VideoData" / s / cam / "cameraIntrinsicsExtrinsics.pickle", "rb"))
    return {"K": np.asarray(p["intrinsicMat"], float), "dist": np.ravel(p["distortion"]).astype(float),
            "R": np.asarray(p["rotation"], float), "t": np.ravel(p["translation"]).astype(float)}


def relative_pose(ca: dict, cb: dict):
    R = cb["R"] @ ca["R"].T
    return R, cb["t"] - R @ ca["t"]


def pose_difference(R_est, t_est, R_true, t_true) -> dict:
    """How far a recovered relative camera pose is from the true one."""
    dR = R_est @ R_true.T
    rot = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    a, b = t_est / np.linalg.norm(t_est), t_true / np.linalg.norm(t_true)
    return {"rotation_deg": float(rot), "baseline_direction_deg": float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))),
            "baseline_true_m": float(np.linalg.norm(t_true) / 1000)}


# ── 3D angles ──────────────────────────────────────────────────────────


def angle3(a, o, b):
    v1, v2 = a - o, b - o
    c = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def between3(v1, v2):
    c = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def angles_from_keypoints(X: np.ndarray) -> dict:
    """3D joint angles from triangulated HALPE-26 points (N, 26, 3)."""
    g = lambda n: X[:, IDX[n]]  # noqa: E731
    out = {}
    for side, S in (("r", "right"), ("l", "left")):
        out[f"knee_{side}"] = 180 - angle3(g(f"{S}_hip"), g(f"{S}_knee"), g(f"{S}_ankle"))
        out[f"hip_{side}"] = 180 - angle3(g(f"{S}_shoulder"), g(f"{S}_hip"), g(f"{S}_knee"))
        out[f"ankle_{side}"] = between3(g(f"{S}_knee") - g(f"{S}_ankle"), g(f"{S}_toe") - g(f"{S}_heel"))
        out[f"elbow_{side}"] = 180 - angle3(g(f"{S}_shoulder"), g(f"{S}_elbow"), g(f"{S}_wrist"))
        out[f"shoulder_{side}"] = angle3(g(f"{S}_hip"), g(f"{S}_shoulder"), g(f"{S}_elbow"))     # arm away from the trunk
    return out


def reference_3d(subject: str, trial: str) -> dict:
    """The same 3D angles from the laboratory markers (no projection)."""
    t, m = V.read_trc(V.ROOT / subject / "MarkerData" / "Mocap" / f"{trial}.trc")
    out = {"t": t}
    for side, S, s in (("r", "R", "r"), ("l", "L", "L")):
        out[f"knee_{side}"] = 180 - angle3(m[f"{S}_HJC"], m[f"{s}_knee"], m[f"{s}_ankle"])
        out[f"hip_{side}"] = 180 - angle3(m[f"{S}_Shoulder"], m[f"{S}_HJC"], m[f"{s}_knee"])
        out[f"ankle_{side}"] = between3(m[f"{s}_knee"] - m[f"{s}_ankle"], m[f"{s}_toe"] - m[f"{s}_calc"])
        if f"{S}_elbow_lat" in m and f"{S}_wrist_radius" in m:
            elbow = (m[f"{S}_elbow_lat"] + m[f"{S}_elbow_med"]) / 2
            wrist = (m[f"{S}_wrist_radius"] + m[f"{S}_wrist_ulna"]) / 2
            out[f"elbow_{side}"] = 180 - angle3(m[f"{S}_Shoulder"], elbow, wrist)
            out[f"shoulder_{side}"] = angle3(m[f"{S}_HJC"], m[f"{S}_Shoulder"], elbow)
    return out


# ── time alignment with the laboratory ─────────────────────────────────

_FIT = [("right_hip", "R_HJC"), ("left_hip", "L_HJC"), ("right_knee", "r_knee"), ("left_knee", "L_knee"),
        ("right_ankle", "r_ankle"), ("left_ankle", "L_ankle"), ("right_shoulder", "R_Shoulder"),
        ("left_shoulder", "L_Shoulder"), ("right_heel", "r_calc"), ("left_heel", "L_calc"),
        ("right_toe", "r_toe"), ("left_toe", "L_toe")]


def _rigid_residual(P: np.ndarray, Q: np.ndarray) -> float:
    """RMS distance after the best rotation + translation of P onto Q (no mirror, no scale)."""
    ok = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
    P, Q = P[ok], Q[ok]
    if len(P) < 30:
        return float("inf")
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return float(np.sqrt(np.mean(np.sum(((P - pc) @ (Vt.T @ D @ U.T).T - (Q - qc)) ** 2, axis=1))))


def mocap_lag(t: np.ndarray, X_mm: np.ndarray, subject: str, trial: str, max_s: float = 0.8):
    """Time offset between the video clock and the laboratory clock, and how well the 3D points
    then fit the markers (mm). Found with a rigid fit of the whole body in space: unlike matching
    one angle curve, it cannot be fooled by a periodic movement such as walking (half a cycle
    later with the legs exchanged is a mirror image, which a rigid fit rejects)."""
    tm, m = V.read_trc(V.ROOT / subject / "MarkerData" / "Mocap" / f"{trial}.trc")
    unit = 1.0 if np.nanmedian(np.linalg.norm(m["R_HJC"] - m["r_knee"], axis=1)) > 100 else 1000.0
    best = (float("inf"), 0.0)
    for lag in np.arange(-max_s, max_s + 1e-9, 1 / 120):
        P, Q = [], []
        for h, mk in _FIT:
            Q.append(np.stack([np.interp(t - lag, tm, m[mk][:, c] * unit, left=np.nan, right=np.nan) for c in range(3)], axis=1))
            P.append(X_mm[:, IDX[h]])
        P, Q = np.concatenate(P), np.concatenate(Q)
        if (np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)).sum() < 0.5 * len(P):
            continue
        best = min(best, (_rigid_residual(P, Q), float(lag)))
    return best[1], best[0]


def score_angles(t: np.ndarray, est: dict, ref: dict, lag: float) -> dict:
    """Errors per joint (both legs pooled) at the given video/mocap time offset."""
    out = {"lag_s": lag}
    for j in ("knee", "hip", "ankle"):
        ms = [V.metrics(est[f"{j}_{s}"], np.interp(t - lag, ref["t"], ref[f"{j}_{s}"], left=np.nan, right=np.nan)) for s in "rl"]
        ms = [m for m in ms if "mae" in m]
        out[j] = {k: float(np.mean([m[k] for m in ms])) if ms else float("nan") for k in ("mae", "bias", "sd", "r")}
    return out


# ── experiment ─────────────────────────────────────────────────────────


def session_for(subject: str, trial: str, cams, height_m: float) -> Session:
    """The videos as OpenKick would see them: detections only, lens guessed."""
    from fractions import Fraction

    from openkick.core.video import VideoMeta
    views = []
    for c in cams:
        video = V.video_path(subject, trial, c)
        t, xy, score = detect_full(video)
        views.append(View(VideoMeta(str(video), 720, 1280, 60.0, Fraction(1, 1000)), t, xy, score))
    return Session(views, height_m)


def _observed(s: Session, offsets: list, cal: list):
    """All views on the first one's clock, normalized with the dataset's intrinsics and distortion."""
    rows, t = s.grid(offsets)
    out = []
    for i, (v, off) in enumerate(zip(s.views, offsets)):
        if i == 0:
            xy, sc = v.xy[rows].astype(float), v.score[rows]
        else:
            xy, sc = resample(v.t + off, v.xy.astype(float), v.score.astype(float), t)
        xy[sc < 0.5] = np.nan
        out.append(normalize(xy, cal[i]["K"], cal[i]["dist"]))
    return t, np.stack(out)


def ceiling(s: Session, cal: list):
    """Reconstruction with the dataset's calibration, each camera at the time offset where it
    agrees best with the first one. Returns offsets, times, points (mm), the reprojection error
    (px, median) and the cameras relative to the first one."""
    rel = [(I3, Z3)] + [relative_pose(cal[0], c) for c in cal[1:]]
    offsets, worst = [0.0], 0.0
    for i in range(1, len(cal)):
        pair = Session([s.views[0], s.views[i]], s.height_m)
        best = (float("inf"), 0.0)
        for off in np.arange(-1.5, 1.5001, 1 / 120):
            if len(pair.grid([0.0, float(off)])[0]) < 25:
                continue
            _, n = _observed(pair, [0.0, float(off)], [cal[0], cal[i]])
            both = np.isfinite(n).all(axis=(0, 3))
            na, nb = np.where(both[..., None], n[0], np.nan), np.where(both[..., None], n[1], np.nan)
            err = G.reprojection_error(na, nb, *rel[i], cal[0]["K"][0, 0], cal[i]["K"][0, 0])[1]
            best = min(best, (float(np.nanmedian(err)), float(off)))
        offsets.append(best[1])
        worst = max(worst, best[0])
    t, n = _observed(s, offsets, cal)
    return offsets, t, G.triangulate_n(n, rel), worst, rel


def run(subject: str, trial: str, cams=("Cam0", "Cam4"), verbose=True) -> dict:
    import yaml
    meta = yaml.safe_load(open(V.ROOT / subject / "sessionMetadata.yaml"))
    height = float(meta["height_m"])
    cal = [load_camera(subject, trial, c) for c in cams]
    ref = reference_3d(subject, trial)
    s = session_for(subject, trial, cams, height)

    offsets, t, X, reproj, rel = ceiling(s, cal)
    lag, fit = mocap_lag(t, X, subject, trial)
    res = {"height_m": height, "true_offsets_s": offsets, "true_offset_s": offsets[1], "mocap_lag_s": lag,
           "baseline_true_m": float(np.linalg.norm(rel[1][1]) / 1000),
           "angle_between_cameras_deg": float(np.degrees(np.arccos(np.clip((np.trace(rel[1][0]) - 1) / 2, -1, 1))))}
    res["ceiling"] = score_angles(t, angles_from_keypoints(X), ref, lag)
    res["ceiling"].update({"reprojection_px": reproj, "fit_mm": fit})

    s.solve()
    q = s.quality()
    _, fit2 = mocap_lag(s.t, s.X * 1000, subject, trial)
    out = score_angles(s.t, angles_from_keypoints(s.X), ref, lag)
    diffs = [pose_difference(*s.cams[i], *rel[i]) for i in range(1, len(cams))]
    true_base = float(np.linalg.norm(rel[s.base][1]) / 1000)
    out.update({"rotation_deg": float(np.mean([x["rotation_deg"] for x in diffs])),
                "rotation_max_deg": float(np.max([x["rotation_deg"] for x in diffs])),
                "baseline_est_m": float(s.scale * np.linalg.norm(s.cams[s.base][1])), "baseline_true_m": true_base,
                "offset_s": s.offsets[1], "offset_error_ms": float(max((a - b) * 1000 for a, b in zip(s.offsets[1:], offsets[1:]))
                                                                   if len(cams) == 2 else max(abs(a - b) * 1000 for a, b in zip(s.offsets[1:], offsets[1:]))),
                "reprojection_px": s.rms, "agreement": s.agreement, "fit_mm": fit2, "mismatch_cm": q["mismatch_cm"],
                "legs_variation": q["legs_variation"], "level": q["level"], "problems": q["problems"],
                "reconstructed": q["reconstructed"]})
    res["body"] = out
    if verbose:
        report(res, subject, trial, cams)
    return res


def report(res: dict, subject, trial, cams):
    c, b = res["ceiling"], res["body"]
    print(f"\n{subject} · {trial} · {' + '.join(cams)}: true offset {res['true_offset_s'] * 1000:+.0f} ms, "
          f"video/mocap lag {res['mocap_lag_s'] * 1000:+.0f} ms")
    print(f"  {'calibration':22s} {'knee':>6s} {'hip':>6s} {'ankle':>6s}  fit to markers")
    print(f"  {'ceiling (dataset)':22s} {c['knee']['mae']:6.1f} {c['hip']['mae']:6.1f} {c['ankle']['mae']:6.1f}  {c['fit_mm']:4.0f} mm")
    print(f"  {'body (OpenKick)':22s} {b['knee']['mae']:6.1f} {b['hip']['mae']:6.1f} {b['ankle']['mae']:6.1f}  {b['fit_mm']:4.0f} mm | "
          f"offset error {b['offset_error_ms']:+.0f} ms, camera rotation error {b['rotation_deg']:.1f}° (max {b['rotation_max_deg']:.1f}°), "
          f"baseline {b['baseline_est_m']:.2f} m (true {b['baseline_true_m']:.2f}), quality {b['level']} "
          f"(mismatch {b['mismatch_cm']:.1f} cm, legs {b['legs_variation']:.1f} %, reconstructed {100 * b['reconstructed']:.0f} %)")


def batch(kinds, cams=("Cam0", "Cam4"), out: str | None = None) -> dict:
    """Run every subject for the given movement types and print the means."""
    import json
    subjects = sorted((p.name for p in V.ROOT.glob("subject*")), key=lambda s: int(s[7:]))
    results = {}
    for kind in kinds:
        rows = []
        for s in subjects:
            for tr in V.trials_of_type(s, kind):
                if V.video_path(s, tr, cams[0]) is None:
                    continue
                try:
                    r = run(s, tr, cams, verbose=False)
                except Exception as e:  # noqa: BLE001
                    print(f"{s} {tr}: failed ({e!r})", flush=True)
                    continue
                r["subject"], r["trial"] = s, tr
                rows.append(r)
                c, b = r["ceiling"], r["body"]
                print(f"{s:10s} {tr:10s} lag {r['mocap_lag_s'] * 1000:+5.0f} | ceiling knee {c['knee']['mae']:5.1f} hip {c['hip']['mae']:5.1f} "
                      f"ankle {c['ankle']['mae']:5.1f} | body knee {b['knee']['mae']:5.1f} hip {b['hip']['mae']:5.1f} ankle {b['ankle']['mae']:5.1f} "
                      f"| offset err {b['offset_error_ms']:+5.0f} ms rot err {b['rotation_deg']:5.1f}° {b['level']:4s} "
                      f"mismatch {b['mismatch_cm']:4.1f} cm legs {b['legs_variation']:4.1f} %", flush=True)
        results[kind] = rows
        if not rows:
            continue
        print(f"\n{kind}: {len(rows)} trials, {'+'.join(cams)} (degrees vs laboratory)")
        print(f"{'calibration':20s} {'knee':>6s} {'hip':>6s} {'ankle':>6s} | median knee | fit mm")
        for n in ("ceiling", "body"):
            g = lambda j: float(np.nanmean([r[n][j]["mae"] for r in rows]))  # noqa: E731
            print(f"{n:20s} {g('knee'):6.1f} {g('hip'):6.1f} {g('ankle'):6.1f} | {np.nanmedian([r[n]['knee']['mae'] for r in rows]):11.1f} | "
                  f"{np.mean([r[n]['fit_mm'] for r in rows]):6.0f}")
        b = [r["body"] for r in rows]
        print(f"body: camera rotation error {np.mean([x['rotation_deg'] for x in b]):.1f}° (max {np.max([x['rotation_deg'] for x in b]):.1f}°), "
              f"baseline error {np.mean([abs(x['baseline_est_m'] - x['baseline_true_m']) / x['baseline_true_m'] * 100 for x in b]):.1f} %, "
              f"offset error median {np.median(np.abs([x['offset_error_ms'] for x in b])):.0f} ms (max {np.max(np.abs([x['offset_error_ms'] for x in b])):.0f} ms)\n", flush=True)
    if out:
        json.dump(results, open(out, "w"), indent=1)
    return results


def save_lags(results: dict, cams=("Cam0", "Cam4")):
    """Keep the video/mocap time offset of every trial and camera for the 2D harness."""
    import json
    lags = json.loads(V.LAG_FILE.read_text()) if V.LAG_FILE.exists() else {}
    for rows in results.values():
        for r in rows:
            for cam, off in zip(cams, r["true_offsets_s"]):
                lags[f"{r['subject']}/{r['trial']}/{cam}"] = r["mocap_lag_s"] - off
    V.LAG_FILE.write_text(json.dumps(lags, indent=1, sort_keys=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="subject2")
    ap.add_argument("--trial", default="squats1")
    ap.add_argument("--cams", nargs="+", default=["Cam0", "Cam4"], help="two or more cameras")
    ap.add_argument("--batch", nargs="+", metavar="TYPE", help="run every subject for these movement types")
    ap.add_argument("--out", help="save the batch results as JSON")
    a = ap.parse_args(argv)
    if a.batch:
        save_lags(batch(a.batch, tuple(a.cams), a.out), tuple(a.cams))
    else:
        run(a.subject, a.trial, tuple(a.cams))


if __name__ == "__main__":
    main()
