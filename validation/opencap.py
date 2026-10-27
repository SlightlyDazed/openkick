"""Validation of OpenKick's 2D joint angles against the OpenCap "LabValidation" dataset
(marker-based motion capture synchronized with 5 smartphone cameras).

Reference ("gold standard")
    The same clinical 2D angles OpenKick measures, computed from the laboratory's 3D markers
    projected on the subject's sagittal plane (mocap X = forward, Y = up, Z = right):
      knee  = angle between thigh (hip joint centre -> knee) and shank (knee -> ankle)
      hip   = angle between trunk (hip joint centre -> shoulder) and thigh
      ankle = angle between shank (ankle -> knee) and foot (heel -> toe)
    expressed as flexion: 0 deg = straight (knee, hip); ankle is the shank-foot angle (about 90 deg
    standing, smaller in dorsiflexion).

Timing
    Video and mocap are synchronized in the dataset, but a residual offset of a few frames
    remains in some cameras. It is estimated by cross-correlation of the knee angle and
    reported; errors are given both as-is and after removing that offset ("aligned").
    The knee is also compared with the OpenSim inverse-kinematics result shipped with the data.

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
JOINTS = ("knee", "hip", "ankle")


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


# ── angles ─────────────────────────────────────────────────────────────


def between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle between two direction vectors (degrees) for arrays (N, 2)."""
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.clip((v1 * v2).sum(axis=1) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(c))


def interior(a: np.ndarray, o: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle at vertex o (degrees) for arrays (N, 2)."""
    v1, v2 = a - o, b - o
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.clip((v1 * v2).sum(axis=1) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(c))


def reference(subject: str, trial: str) -> dict:
    """Marker-based sagittal-plane angles (+ OpenSim IK knee) at the mocap rate."""
    t, m = read_trc(ROOT / subject / "MarkerData" / "Mocap" / f"{trial}.trc")
    # Subject's sagittal plane: vertical (lab Y) x forward, where forward is horizontal and
    # perpendicular to the mean pelvis left-right axis over the trial.
    lat = np.nanmean(m["r.ASIS"] - m["L.ASIS"], axis=0)
    lat[1] = 0
    lat /= np.linalg.norm(lat)
    fwd = np.array([lat[2], 0.0, -lat[0]])          # lateral rotated 90 deg about the vertical
    xy = lambda name: np.stack([m[name] @ fwd, m[name][:, 1]], axis=1)  # noqa: E731
    out = {"t": t}
    for side, S, s in (("r", "R", "r"), ("l", "L", "L")):
        hjc, sho = xy(f"{S}_HJC"), xy(f"{S}_Shoulder")
        knee, ank, toe, heel = xy(f"{s}_knee"), xy(f"{s}_ankle"), xy(f"{s}_toe"), xy(f"{s}_calc")
        out[f"knee_{side}"] = 180 - interior(hjc, knee, ank)
        out[f"hip_{side}"] = 180 - interior(sho, hjc, knee)
        out[f"ankle_{side}"] = between(knee - ank, toe - heel)
    # body proportions used to estimate how far a camera is from a true profile view
    out["hip_width_over_thigh"] = float(np.nanmedian(
        np.linalg.norm(m["R_HJC"] - m["L_HJC"], axis=1) / np.linalg.norm(m["R_HJC"] - m["r_knee"], axis=1)))
    ik = ROOT / subject / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot"
    if ik.exists():
        ti, c = read_mot(ik)
        for side in "rl":
            out[f"knee_ik_{side}"] = np.interp(t, ti, c[f"knee_angle_{side}"])
    return out


@dataclass
class Detection:
    t: np.ndarray            # time of each video frame (s)
    points: dict             # name -> (N, 2) pixels, NaN where missing
    seconds: float           # processing time


def angles_2d(d: Detection, side: str) -> dict:
    p = d.points
    S = "right" if side == "r" else "left"
    return {
        "knee": 180 - interior(p[f"{S}_hip"], p[f"{S}_knee"], p[f"{S}_ankle"]),
        "hip": 180 - interior(p[f"{S}_shoulder"], p[f"{S}_hip"], p[f"{S}_knee"]),
        "ankle": between(p[f"{S}_knee"] - p[f"{S}_ankle"], p[f"{S}_toe"] - p[f"{S}_heel"]),
    }


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
    est = angles_2d(det, side)
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
        r = np.interp(det.t, ref["t"], ref[f"{j}_{side}"])
        out[j] = metrics(est[j], r)
        # same comparison after removing the residual video/mocap time offset
        ra = np.interp(det.t - lag, ref["t"], ref[f"{j}_{side}"])
        out[j + "_aligned"] = metrics(est[j], ra)
        out["series"][j] = (det.t, est[j], ra)
    if f"knee_ik_{side}" in ref:
        out["knee_vs_ik"] = metrics(est["knee"], np.interp(det.t, ref["t"], ref[f"knee_ik_{side}"]))
    return out


# ── reporting ──────────────────────────────────────────────────────────


def summarize(rows: list[dict]) -> str:
    lines = [f"{'joint':14s} {'trials':>6s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} {'SD':>6s} {'r':>6s}  (degrees; mean over trials)"]
    for j in [x for jj in JOINTS for x in (jj, jj + "_aligned")] + ["knee_vs_ik"]:
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
            for j in [x for jj in JOINTS for x in (jj, jj + "_aligned")] + ["knee_vs_ik"]:
                if j in r:
                    w.writerow([r["engine"], r["subject"], r["trial"], r["side"], r["cam"], j,
                                f"{r['lag_s']:.3f}", f"{r['fps_proc']:.1f}", f"{r['view_deg']:.1f}", r["frames"]] +
                               [f"{r[j].get(k, float('nan')):.3f}" for k in keys])


def plot(row: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=110)
    for ax, j in zip(axes, JOINTS):
        t, est, ref = row["series"][j]
        ax.plot(t, ref, color="#111", lw=2, label="laboratorio (marcadores)")
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
