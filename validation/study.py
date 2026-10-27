"""Studies on the OpenCap data beyond the main tables (docs/VALIDATION.md explains each):

    python -m validation.study baseline     per-frame results of the two-camera reconstruction, kept for the statistics
    python -m validation.study fps30        the same from videos at 30 frames per second (every other frame)
    python -m validation.study pairs        other camera pairs (squats and walking)
    python -m validation.study holdout      settings chosen on some participants, results on the others
    python -m validation.study stats        tables with confidence intervals, agreement and reliability

Everything runs OpenKick as a user would: only the videos' detections and the person's height.
"""

from __future__ import annotations

import json
import sys
import warnings
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from validation import opencap as V

OUT = Path(__file__).resolve().parent / "results"
SERIES = V.CACHE / "study"
JOINTS = ("knee", "hip", "ankle", "elbow", "shoulder")
KINDS = ("squats", "STS", "DJ", "walking")
TUNE = ("subject2", "subject3", "subject4", "subject5")            # settings may be chosen looking at these
TEST = ("subject7", "subject8", "subject9", "subject10", "subject11")   # results are reported on these
DEFAULTS = {"min_score": 0.5, "threshold_px": 3.0, "lens_mm": 26.0, "div": 1}


def subjects() -> list[str]:
    return sorted((p.name for p in V.ROOT.glob("subject*")), key=lambda s: int(s[7:]))


def trials(kinds=KINDS, who=None, cams=("Cam0", "Cam4")) -> list[tuple]:
    return [(s, tr, k) for k in kinds for s in (who or subjects()) for tr in V.trials_of_type(s, k)
            if V.video_path(s, tr, cams[0]) is not None]


def evaluate(job: tuple) -> dict:
    """One recording through OpenKick with the given settings. Returns the errors of every angle
    and, for the statistics, the angle curves of OpenKick and of the laboratory."""
    subject, trial, kind, cams, opts, tag = job
    warnings.simplefilter("ignore")
    import yaml
    from fractions import Fraction

    from openkick.core.video import VideoMeta
    from openkick.multiview import geometry as G
    from openkick.multiview import session as S
    from validation import multiview as M
    o = dict(DEFAULTS, **opts)
    S.MIN_SCORE = o["min_score"]
    S.G.calibrate_from_body = partial(G.calibrate_from_body.func if isinstance(G.calibrate_from_body, partial)
                                      else G.calibrate_from_body, threshold_px=o["threshold_px"])
    meta = yaml.safe_load(open(V.ROOT / subject / "sessionMetadata.yaml"))
    views = []
    for c in cams:
        t, xy, score = M.detect_full(V.video_path(subject, trial, c))
        d = o["div"]
        views.append(S.View(VideoMeta("", 720, 1280, 60.0 / d, Fraction(1, 1000)), t[::d], xy[::d], score[::d],
                            K=G.guess_intrinsics(720, 1280, o["lens_mm"])))
    s = S.Session(views, float(meta["height_m"]))
    out = {"subject": subject, "trial": trial, "kind": kind, "cams": list(cams), "opts": o}
    try:
        s.solve()
    except Exception as e:  # noqa: BLE001
        out["failed"] = repr(e)
        return out
    lags = [V.known_lag(subject, trial, c) for c in cams]
    lag = lags[0]
    ref = M.reference_3d(subject, trial)
    est = M.angles_from_keypoints(s.X)
    cal = [M.load_camera(subject, trial, c) for c in cams]
    R_true, t_true = M.relative_pose(cal[0], cal[1])
    q = s.quality()
    out.update({"rotation_deg": M.pose_difference(s.R, s.d, R_true, t_true)["rotation_deg"],
                "baseline_err_pct": float(abs(s.scale - np.linalg.norm(t_true) / 1000) / (np.linalg.norm(t_true) / 1000) * 100),
                "offset_err_ms": float((s.offsets[1] - (lags[0] - lags[1])) * 1000), "level": q["level"],
                "reconstructed": q["reconstructed"], "frames": int(s.frame_count)})
    series = {"t": s.t - lag}
    for j in JOINTS:
        per_side = []
        for side in "rl":
            key = f"{j}_{side}"
            if key not in ref:
                continue
            r = np.interp(s.t - lag, ref["t"], ref[key], left=np.nan, right=np.nan)
            series[key + "_est"], series[key + "_ref"] = est[key], r
            m = V.metrics(est[key], r)
            if "mae" in m:
                ok = np.isfinite(est[key]) & np.isfinite(r)
                m["peak_est"], m["peak_ref"] = float(np.max(est[key][ok])), float(np.max(r[ok]))
                per_side.append(m)
        if per_side:
            out[j] = {k: float(np.mean([m[k] for m in per_side])) for k in ("mae", "rmse", "bias", "sd", "r", "rom_est", "rom_ref",
                                                                           "peak_est", "peak_ref")}
    if tag:
        SERIES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SERIES / f"{tag}_{subject}_{trial}.npz", **series)
    return out


def _pair(job: tuple) -> dict:
    subject, trial, kind, cams = job
    warnings.simplefilter("ignore")
    from validation import multiview as M
    try:
        r = M.run(subject, trial, cams, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"subject": subject, "trial": trial, "kind": kind, "cams": list(cams), "failed": repr(e)}
    r.update({"subject": subject, "trial": trial, "kind": kind, "cams": list(cams)})
    return r


def pool_map(fn, jobs: list, workers: int = 6, label: str = "") -> list:
    rows = []
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            rows.append(r)
            if i % 12 == 0 or i == len(jobs):
                print(f"  {label} {i}/{len(jobs)}", flush=True)
    failed = [r for r in rows if "failed" in r]
    for r in failed:
        print(f"  failed: {r['subject']} {r['trial']} {r['failed']}", flush=True)
    return [r for r in rows if "failed" not in r]


def mean_of(rows: list, joint: str, key: str = "mae") -> float:
    v = [r[joint][key] for r in rows if joint in r]
    return float(np.mean(v)) if v else float("nan")


def save(name: str, rows) -> Path:
    OUT.mkdir(exist_ok=True)
    f = OUT / f"study_{name}.json"
    f.write_text(json.dumps(rows, indent=1))
    return f


def load(name: str):
    return json.loads((OUT / f"study_{name}.json").read_text())


# ── the studies ────────────────────────────────────────────────────────


def baseline():
    jobs = [(s, tr, k, ("Cam0", "Cam4"), {}, "base") for s, tr, k in trials()]
    save("baseline", pool_map(evaluate, jobs, label="baseline"))


def fps30():
    jobs = [(s, tr, k, ("Cam0", "Cam4"), {"div": 2}, "") for s, tr, k in trials()]
    save("fps30", pool_map(evaluate, jobs, label="30 fps"))


PAIRS = (("Cam0", "Cam1"), ("Cam0", "Cam2"), ("Cam1", "Cam3"), ("Cam0", "Cam3"), ("Cam2", "Cam4"), ("Cam0", "Cam4"))


def pairs():
    jobs = [(s, tr, k, cams) for cams in PAIRS for s, tr, k in trials(("squats", "walking"))]
    save("pairs", pool_map(_pair, jobs, label="pairs"))


SWEEP = {"min_score": (0.3, 0.5, 0.7), "threshold_px": (2.0, 3.0, 5.0), "lens_mm": (22.0, 26.0, 30.0)}


def holdout():
    """Choose every setting on the tuning participants only (one at a time around the defaults),
    then measure once on the test participants, who played no part in the choice."""
    def score(rows):
        return float(np.mean([mean_of(rows, j) for j in ("knee", "hip", "ankle")]))
    tune = trials(who=TUNE)
    result = {"tune_subjects": TUNE, "test_subjects": TEST, "sweep": {}}
    chosen = {}
    for name, values in SWEEP.items():
        result["sweep"][name] = {}
        for v in values:
            rows = pool_map(evaluate, [(s, tr, k, ("Cam0", "Cam4"), {name: v}, "") for s, tr, k in tune], label=f"{name}={v}")
            result["sweep"][name][str(v)] = {"score": score(rows), **{j: mean_of(rows, j) for j in ("knee", "hip", "ankle")},
                                             "rotation_deg": float(np.mean([r["rotation_deg"] for r in rows]))}
        chosen[name] = min(values, key=lambda v: result["sweep"][name][str(v)]["score"])
    result["chosen"] = chosen
    test = trials(who=TEST)
    for label, opts in (("chosen", chosen), ("defaults", {})):
        rows = pool_map(evaluate, [(s, tr, k, ("Cam0", "Cam4"), opts, "") for s, tr, k in test], label=f"test, {label}")
        result[f"test_{label}"] = {k: {j: mean_of([r for r in rows if r["kind"] == k], j) for j in ("knee", "hip", "ankle")} for k in KINDS}
        result[f"test_{label}"]["all"] = {j: mean_of(rows, j) for j in ("knee", "hip", "ankle")}
    rows = load("baseline")
    for who, name in ((TUNE, "tune_defaults"), (TEST, "test_defaults_check")):
        part = [r for r in rows if r["subject"] in who]
        result[name] = {k: {j: mean_of([r for r in part if r["kind"] == k], j) for j in ("knee", "hip", "ankle")} for k in KINDS}
        result[name]["all"] = {j: mean_of(part, j) for j in ("knee", "hip", "ankle")}
    save("holdout", result)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for name in argv or ["baseline"]:
        if name == "stats":
            from validation import stats
            stats.main()
        else:
            {"baseline": baseline, "fps30": fps30, "pairs": pairs, "holdout": holdout}[name]()


if __name__ == "__main__":
    main()
