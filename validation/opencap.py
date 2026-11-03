"""Validation of OpenKick's 2D joint angles against the OpenCap "LabValidation" dataset
(marker-based motion capture synchronized with 5 smartphone cameras).

Reference
    The laboratory's own OpenSim inverse kinematics, shipped with the data: knee_angle,
    hip_flexion, ankle_angle, arm_flex (shoulder) and elbow_flex, the clinical convention
    (0 = neutral, flexion and dorsiflexion positive). Until November 2026 the reference was built
    from the markers the way OpenKick built its own angles; the skin markers sit 3-8 cm outside
    the joints, so that reference was itself 4.5-5.5 deg off the lab's OpenSim knee.

OpenKick
    The angles the app shows, automatic output (no smoothing, no manual correction):
    openkick.core.joints.clinical, with the hip's reference of the template a user would pick
    for the movement (TEMPLATE).

Timing
    Video and mocap are only roughly synchronized. The offset is measured in 3D
    (mocap_lag.json, see validation.multiview) and refined by cross-correlation of the knee;
    errors are given both as-is and after removing that offset ("aligned").

Usage
    python -m validation.opencap --engine mediapipe --trials squats1 STS1
    python -m validation.opencap --engine rtmpose-balanced --subjects subject2 subject3

Download the dataset (free account) from https://simtk.org/projects/opencap and point
OPENKICK_OPENCAP to the LabValidation_withVideos folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("OPENKICK_OPENCAP",
                           Path(__file__).resolve().parents[2] / "datos/opencap/LabValidation_withVideos"))
CACHE = ROOT.parent / "_openkick_cache"
OUT = Path(__file__).resolve().parent / "results"

# Cameras that see the subject in profile (checked on the videos). Which side of the body each
# one sees depends on where the subject faces, so it is deduced from the detection itself.
PROFILE_CAMS = ("Cam0", "Cam4")
JOINTS = ("knee", "hip", "ankle", "shoulder", "elbow")
LAB = {"knee": "knee_angle", "hip": "hip_flexion", "ankle": "ankle_angle", "shoulder": "arm_flex", "elbow": "elbow_flex"}
# The template a user would pick for each movement: it sets the hip's reference and, in 3D, the
# smoothing of the angles
TEMPLATE = {"squats": "squat", "STS": "squat", "DJ": "jump", "walking": "gait"}


# ── dataset readers ────────────────────────────────────────────────────


def read_trc(path: Path):
    lines = path.read_text().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    rows = [ln.split("\t") for ln in lines[5:] if ln.strip()]
    a = np.array([[float(x) if x.strip() else np.nan for x in r] for r in rows])
    t = a[:, 1]
    markers = {n: a[:, 2 + 3 * i:5 + 3 * i] for i, n in enumerate(names)}
    return t, markers


def read_mot(path: Path):
    lines = path.read_text().splitlines()
    i = lines.index("endheader")
    cols = lines[i + 1].split("\t")
    a = np.array([ln.split("\t") for ln in lines[i + 2:] if ln.strip()], float)
    return a[:, 0], {c: a[:, k] for k, c in enumerate(cols)}


def session_of(subject: str, trial: str) -> str | None:
    for s in sorted((ROOT / subject / "VideoData").glob("Session*")):
        if (s / "Cam0" / trial).is_dir():
            return s.name
    return None


def video_path(subject: str, trial: str, cam: str) -> Path | None:
    s = session_of(subject, trial)
    if s is None:
        return None
    p = ROOT / subject / "VideoData" / s / cam / trial / f"{trial}_syncdWithMocap.avi"
    return p if p.exists() else None


def trials_of(subject: str) -> list[str]:
    return sorted(p.stem for p in (ROOT / subject / "MarkerData" / "Mocap").glob("*.trc"))


# Movement types -> trial name pattern (numbers differ between subjects)
TYPES = {"squats": r"squats\d+", "squatsAsym": r"squatsAsym\d+", "STS": r"STS\d+", "STSweakLegs": r"STSweakLegs\d+",
         "DJ": r"DJ\d+", "DJAsym": r"DJAsym\d+", "walking": r"walking\d+", "walkingTS": r"walkingTS\d+"}


def trials_of_type(subject: str, kind: str) -> list[str]:
    import re
    return [t for t in trials_of(subject) if re.fullmatch(TYPES[kind], t) and session_of(subject, t)]


def template_of(trial: str):
    """The app's template for this trial's movement (None for movements not in TEMPLATE)."""
    import re

    from openkick.core.templates import find
    kind = next((k for k, pat in TYPES.items() if re.fullmatch(pat, trial)), None)
    return find(TEMPLATE.get(kind, ""))


def hip_reference(trial: str) -> str:
    t = template_of(trial)
    return (t.hip_reference if t else None) or "pelvis"


# ── angles ─────────────────────────────────────────────────────────────


def reference(subject: str, trial: str) -> dict:
    """The laboratory's OpenSim angles (LAB) at the mocap rate, as {joint_side: series}."""
    t, c = read_mot(ROOT / subject / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot")
    out = {"t": t}
    for j, col in LAB.items():
        for side in "rl":
            if f"{col}_{side}" in c:
                out[f"{j}_{side}"] = c[f"{col}_{side}"]
    # body proportions used to estimate how far a camera is from a true profile view
    _, m = read_trc(ROOT / subject / "MarkerData" / "Mocap" / f"{trial}.trc")
    out["hip_width_over_thigh"] = float(np.nanmedian(
        np.linalg.norm(m["R_HJC"] - m["L_HJC"], axis=1) / np.linalg.norm(m["R_HJC"] - m["r_knee"], axis=1)))
    return out


@dataclass
class Detection:
    t: np.ndarray            # time of each video frame (s)
    points: dict             # name -> (N, 2) pixels, NaN where missing
    seconds: float           # processing time


def landmarks_of(d: Detection) -> np.ndarray:
    """The detection in the app's landmark layout (N, 33, 2), NaN where the engine has no point."""
    from openkick.core import landmarks as L
    xy = np.full((len(d.t), L.COUNT, 2), np.nan)
    for name, v in d.points.items():
        side, part = name.split("_", 1)
        xy[:, L.INDEX[f"{side}_{'foot_index' if part == 'toe' else part}"]] = v
    return xy


def angles_2d(d: Detection, side: str, hip: str = "pelvis") -> dict:
    """The app's clinical angles of one side (openkick.core.joints), the joints the engine sees."""
    from openkick.core import joints
    xy = landmarks_of(d)
    fps = 1 / np.median(np.diff(d.t))
    forward = joints.facing(xy, fps)
    S = "right" if side == "r" else "left"
    out = {}
    with np.errstate(invalid="ignore"):
        for j in JOINTS:
            if j in ("shoulder", "elbow") and f"{S}_elbow" not in d.points:
                continue
            out[j] = joints.clinical(xy, j, S, forward, hip)
    return out


# ── engines ────────────────────────────────────────────────────────────


def detect(engine: str, video: Path) -> Detection:
    """Run a pose engine over a video (results cached on disk)."""
    key = "_".join(video.parts[-6:]).replace(".avi", "")
    cache = CACHE / engine / f"{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return Detection(z["t"], dict(z["points"].item()), float(z["seconds"]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from validation import engines
    t0 = time.time()
    t, points = engines.run(engine, video)
    d = Detection(t, points, time.time() - t0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=d.t, points=np.array(d.points, dtype=object), seconds=d.seconds)
    return d


# ── comparison ─────────────────────────────────────────────────────────


def best_lag(a: np.ndarray, b: np.ndarray, fps: float, max_s: float = 0.25) -> float:
    """Residual time offset (s) between two series, by cross-correlation."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return 0.0
    a, b = a - np.nanmean(a[ok]), b - np.nanmean(b[ok])
    a, b = np.where(ok, a, 0), np.where(ok, b, 0)
    k = int(max_s * fps)
    lags = range(-k, k + 1)
    score = [np.sum(a[max(0, L):len(a) + min(0, L)] * b[max(0, -L):len(b) + min(0, -L)]) for L in lags]
    return list(lags)[int(np.argmax(score))] / fps


def metrics(est: np.ndarray, ref: np.ndarray) -> dict:
    ok = np.isfinite(est) & np.isfinite(ref)
    if ok.sum() < 10:
        return {"n": int(ok.sum())}
    e = est[ok] - ref[ok]
    sd = float(np.std(e, ddof=1))
    r = float(np.corrcoef(est[ok], ref[ok])[0, 1]) if np.std(ref[ok]) > 1e-6 else float("nan")
    return {"n": int(ok.sum()), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)), "sd": sd, "loa_low": float(np.mean(e) - 1.96 * sd),
            "loa_high": float(np.mean(e) + 1.96 * sd), "r": r,
            "rom_est": float(np.ptp(est[ok])), "rom_ref": float(np.ptp(ref[ok])),
            "detected": float(np.isfinite(est).mean())}


def near_side(det: Detection) -> str:
    """Side of the body facing the camera. A person facing image-right shows their right side."""
    p = det.points
    dx = np.concatenate([p["right_toe"][:, 0] - p["right_heel"][:, 0], p["left_toe"][:, 0] - p["left_heel"][:, 0]])
    return "r" if np.nanmedian(dx) > 0 else "l"


def view_angle(det: Detection, side: str, ratio_3d: float) -> float:
    """Degrees between the camera and a true profile view (0 = perfect profile, 90 = frontal).
    In profile both hips overlap in the image; the more frontal the camera, the wider apart they
    appear relative to the thigh length. Uses the frames where the thigh looks longest (upright)."""
    p = det.points
    S = "right" if side == "r" else "left"
    width = np.abs(p["right_hip"][:, 0] - p["left_hip"][:, 0])
    thigh = np.linalg.norm(p[f"{S}_hip"] - p[f"{S}_knee"], axis=1)
    ok = np.isfinite(width) & np.isfinite(thigh)
    if ok.sum() < 5:
        return float("nan")
    top = thigh >= np.nanpercentile(thigh[ok], 70)
    r = np.nanmedian(width[ok & top] / thigh[ok & top])
    return float(np.degrees(np.arcsin(np.clip(r / ratio_3d, 0, 1))))


LAG_FILE = Path(__file__).parent / "results" / "mocap_lag.json"
_LAGS: dict | None = None


def known_lag(subject: str, trial: str, cam: str) -> float | None:
    """Video/mocap time offset measured in 3D (validation.multiview.save_lags), if available.
    Matching one angle curve is not enough for walking: the movement repeats, and half a cycle
    later the other leg draws the same curve."""
    global _LAGS
    if _LAGS is None:
        _LAGS = json.loads(LAG_FILE.read_text()) if LAG_FILE.exists() else {}
    return _LAGS.get(f"{subject}/{trial}/{cam}")


def compare(engine: str, subject: str, trial: str, cam: str, side: str | None = None) -> dict | None:
    video = video_path(subject, trial, cam)
    if video is None:
        return None
    ref = reference(subject, trial)
    det = detect(engine, video)
    side = side or near_side(det)
    est = angles_2d(det, side, hip_reference(trial))
    fps = 1 / np.median(np.diff(det.t))
    known = known_lag(subject, trial, cam)
    if known is None:
        lag = best_lag(est["knee"], np.interp(det.t, ref["t"], ref[f"knee_{side}"]), fps)
    else:           # measured in 3D; only the last few hundredths are tuned on the curve
        lag = known + best_lag(est["knee"], np.interp(det.t - known, ref["t"], ref[f"knee_{side}"]), fps, max_s=0.05)
    out = {"engine": engine, "subject": subject, "trial": trial, "side": side, "cam": cam,
           "view_deg": view_angle(det, side, ref["hip_width_over_thigh"]), "frames": len(det.t),
           "lag_s": lag, "fps_proc": len(det.t) / det.seconds if det.seconds else float("nan"), "series": {}}
    for j in JOINTS:
        if j not in est or f"{j}_{side}" not in ref:
            continue
        r = np.interp(det.t, ref["t"], ref[f"{j}_{side}"], left=np.nan, right=np.nan)
        out[j] = metrics(est[j], r)
        # same comparison after removing the residual video/mocap time offset
        ra = np.interp(det.t - lag, ref["t"], ref[f"{j}_{side}"], left=np.nan, right=np.nan)
        out[j + "_aligned"] = metrics(est[j], ra)
        out["series"][j] = (det.t, est[j], ra)
    return out


# ── reporting ──────────────────────────────────────────────────────────


def summarize(rows: list[dict]) -> str:
    lines = [f"{'joint':14s} {'trials':>6s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} {'SD':>6s} {'r':>6s}  (degrees; mean over trials)"]
    for j in [x for jj in JOINTS for x in (jj, jj + "_aligned")]:
        vals = [r[j] for r in rows if j in r and "mae" in r[j]]
        if not vals:
            continue
        g = lambda k: float(np.nanmean([v[k] for v in vals]))  # noqa: E731
        lines.append(f"{j:14s} {len(vals):6d} {g('mae'):7.1f} {g('rmse'):7.1f} {g('bias'):+7.1f} {g('sd'):6.1f} {g('r'):6.3f}")
    return "\n".join(lines)


def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["n", "mae", "rmse", "bias", "sd", "loa_low", "loa_high", "r", "rom_est", "rom_ref", "detected"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "subject", "trial", "side", "cam", "joint", "lag_s", "fps_proc", "view_deg", "frames"] + keys)
        for r in rows:
            for j in [x for jj in JOINTS for x in (jj, jj + "_aligned")]:
                if j in r:
                    w.writerow([r["engine"], r["subject"], r["trial"], r["side"], r["cam"], j,
                                f"{r['lag_s']:.3f}", f"{r['fps_proc']:.1f}", f"{r['view_deg']:.1f}", r["frames"]] +
                               [f"{r[j].get(k, float('nan')):.3f}" for k in keys])


def plot(row: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    joints = [j for j in JOINTS if j in row["series"]]
    fig, axes = plt.subplots(1, len(joints), figsize=(5 * len(joints), 4), dpi=110)
    for ax, j in zip(axes, joints):
        t, est, ref = row["series"][j]
        ax.plot(t, ref, color="#111", lw=2, label="laboratorio (OpenSim)")
        ax.plot(t, est, color="#E4572E", lw=1.5, label=f"OpenKick · {row['engine']}")
        m = row[j + "_aligned"]
        ax.set_title(f"{j} {row['side']} · MAE {m.get('mae', float('nan')):.1f}° · sesgo {m.get('bias', float('nan')):+.1f}°")
        ax.set_xlabel("tiempo (s)")
        ax.set_ylabel("ángulo (°)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{row['subject']} · {row['trial']} · {row['cam']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="mediapipe")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--trials", nargs="*", default=None, help="exact trial names (e.g. squats1)")
    ap.add_argument("--types", nargs="*", default=None, help=f"movement types: {', '.join(TYPES)}")
    ap.add_argument("--cams", nargs="*", default=list(PROFILE_CAMS), help="cameras to use (default: the two profile ones)")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args(argv)
    subjects = a.subjects or sorted((p.name for p in ROOT.glob("subject*")), key=lambda s: int(s[7:]))
    if not a.trials and not a.types:
        a.types = ["squats"]
    rows = []
    for s in subjects:
        trials = list(a.trials or []) + [t for k in (a.types or []) for t in trials_of_type(s, k)]
        for tr in trials:
            for cam in a.cams:
                r = compare(a.engine, s, tr, cam)
                if r is None:
                    continue
                rows.append(r)
                side = r["side"]
                k = r["knee_aligned"]
                print(f"{s:10s} {tr:14s} {side} {r['cam']}  knee MAE {k.get('mae', float('nan')):5.1f}  "
                      f"hip MAE {r['hip_aligned'].get('mae', float('nan')):5.1f}  ankle MAE {r['ankle_aligned'].get('mae', float('nan')):5.1f}  "
                      f"lag {r['lag_s'] * 1000:+4.0f} ms  view {r['view_deg']:3.0f}°  {r['fps_proc']:.0f} fps", flush=True)
                if a.plots:
                    plot(r, OUT / a.engine / f"{s}_{tr}_{side}.png")
    tag = f"{a.engine}_{'-'.join((a.types or []) + (a.trials or []))}" + \
        ("" if a.cams == list(PROFILE_CAMS) else "_" + "-".join(a.cams))
    if a.subjects:
        tag += "_part" + "-".join(x.replace("subject", "") for x in a.subjects)
    save_csv(rows, OUT / f"{tag}.csv")
    print("\n" + summarize(rows))
    print(f"\nCSV: {OUT / (tag + '.csv')}")
    return rows


if __name__ == "__main__":
    main()
