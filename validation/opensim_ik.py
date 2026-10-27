"""The whole chain against the laboratory: two videos -> OpenKick 3D points -> OpenSim scaling
and inverse kinematics (Pose2Sim) -> joint angles of the model, compared with the angles
OpenSim gives from the laboratory's markers (the dataset's Mocap/IK files).

The dataset also holds OpenCap's own video results (two cameras, same OpenSim pipeline), scored
here with the same code as a yardstick.

    python -m validation.opensim_ik --types squats STS DJ walking
    python -m validation.opensim_ik --types walking --cams Cam1 Cam3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from validation import opencap as V

OUT = Path(__file__).resolve().parent / "results"
# ours (Pose2Sim model) -> laboratory (OpenCap's model); same sign conventions
COORDS = {"knee": "knee_angle_{s}", "hip_flexion": "hip_flexion_{s}", "hip_adduction": "hip_adduction_{s}",
          "hip_rotation": "hip_rotation_{s}", "ankle": "ankle_angle_{s}"}
OPENCAP_VIDEO = "HRNet/2-cameras"


def work_dir(subject: str, trial: str, cams, augment: bool = False) -> Path:
    return V.CACHE / ("opensim-augmented" if augment else "opensim") / "-".join(cams) / f"{subject}_{trial}"


def reconstruct(subject: str, trial: str, cams, augment: bool = False, cutoff: float = 8.0) -> dict:
    """Run OpenKick + OpenSim for one trial (cached). Returns the motion file and its time base."""
    import yaml

    from openkick.multiview.export import export_trc, run_opensim
    from validation import multiview as M
    meta = yaml.safe_load(open(V.ROOT / subject / "sessionMetadata.yaml"))
    project = work_dir(subject, trial, cams, augment)
    info_file = project / "openkick.json"
    if info_file.exists():
        return json.loads(info_file.read_text())
    s = M.session_for(subject, trial, cams, float(meta["height_m"]))
    s.solve()
    lag = V.known_lag(subject, trial, cams[0])
    if lag is None:                       # this camera pair was not measured yet
        ca, cb = M.load_camera(subject, trial, cams[0]), M.load_camera(subject, trial, cams[1])
        _, t, X, _ = M.ceiling(s, ca, cb)
        lag = M.mocap_lag(t, X, subject, trial)[0]
    (project / "pose-3d").mkdir(parents=True, exist_ok=True)
    trc = project / "pose-3d" / f"{trial}.trc"
    export_trc(s, trc, rate=60, cutoff=cutoff)
    out = run_opensim(project, trc, float(meta["height_m"]), float(meta["mass_kg"]), augment=augment)
    q = s.quality()
    info = {"mot": str(out["mot"][0]), "t0": float(s.t[0]), "lag": float(lag), "level": q["level"],
            "mismatch_cm": q["mismatch_cm"], "legs_variation": q["legs_variation"]}
    err = sorted((project / "kinematics").glob("*_ik_marker_errors.sto"))
    if err:
        _, e = V.read_mot(err[0])
        info["ik_marker_rms_cm"] = float(np.mean(e["marker_error_RMS"]) * 100)
    info_file.write_text(json.dumps(info))
    return info


def _worker(args):
    subject, trial, cams, augment = args
    devnull = os.open(os.devnull, os.O_WRONLY)     # OpenSim and Pose2Sim are very talkative
    saved = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        return subject, trial, reconstruct(subject, trial, cams, augment), None
    except Exception as e:  # noqa: BLE001
        return subject, trial, None, repr(e)
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)


def score(subject: str, trial: str, info: dict) -> dict:
    """Errors of OpenKick's OpenSim angles, and of OpenCap's, against the laboratory's."""
    tr, ref = V.read_mot(V.ROOT / subject / "OpenSimData" / "Mocap" / "IK" / f"{trial}.mot")
    t, est = V.read_mot(Path(info["mot"]))
    tm = t + info["t0"] - info["lag"]              # our frames on the laboratory's clock
    out = {"openkick": {}, "opencap": {}}
    oc_file = V.ROOT / subject / "OpenSimData" / "Video" / OPENCAP_VIDEO / "IK" / f"{trial}.mot"
    oc = V.read_mot(oc_file) if oc_file.exists() else None
    window = (tr >= tm[0]) & (tr <= tm[-1])         # OpenCap scored over the same stretch of movement
    for name, pattern in COORDS.items():
        ours, theirs = [], []
        for side in "rl":
            c = pattern.format(s=side)
            ours.append(V.metrics(est[c], np.interp(tm, tr, ref[c], left=np.nan, right=np.nan)))
            if oc is not None:
                theirs.append(V.metrics(np.interp(tr[window], oc[0], oc[1][c], left=np.nan, right=np.nan), ref[c][window]))
        for key, ms in (("openkick", ours), ("opencap", theirs)):
            ms = [m for m in ms if "mae" in m]
            if ms:
                out[key][name] = {k: float(np.mean([m[k] for m in ms])) for k in ("mae", "bias", "sd", "r", "rom_est", "rom_ref")}
    return out


def main(argv=None):
    from multiprocessing import get_context
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", nargs="+", default=["squats"])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--cams", nargs=2, default=["Cam0", "Cam4"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--augment", action="store_true", help="add the anatomical markers predicted by Stanford's augmenter")
    a = ap.parse_args(argv)
    cams = tuple(a.cams)
    subjects = a.subjects or sorted((p.name for p in V.ROOT.glob("subject*")), key=lambda s: int(s[7:]))
    jobs = [(s, tr, cams, a.augment) for kind in a.types for s in subjects for tr in V.trials_of_type(s, kind)
            if V.video_path(s, tr, cams[0]) is not None]
    kind_of = {(s, tr): kind for kind in a.types for s in subjects for tr in V.trials_of_type(s, kind)}
    rows = []
    with get_context("spawn").Pool(a.workers) as pool:
        for s, tr, info, error in pool.imap(_worker, jobs):
            if error:
                print(f"{s:10s} {tr:10s} failed: {error}", flush=True)
                continue
            r = score(s, tr, info)
            r.update({"subject": s, "trial": tr, "kind": kind_of[(s, tr)], **{k: info[k] for k in info if k != "mot"}})
            rows.append(r)
            ok, oc = r["openkick"], r["opencap"]
            print(f"{s:10s} {tr:10s} OpenKick knee {ok['knee']['mae']:5.1f} hip flex {ok['hip_flexion']['mae']:5.1f} "
                  f"add {ok['hip_adduction']['mae']:5.1f} rot {ok['hip_rotation']['mae']:5.1f} ankle {ok['ankle']['mae']:5.1f}"
                  + (f" | OpenCap knee {oc['knee']['mae']:5.1f} hip flex {oc['hip_flexion']['mae']:5.1f} add {oc['hip_adduction']['mae']:5.1f} "
                     f"rot {oc['hip_rotation']['mae']:5.1f} ankle {oc['ankle']['mae']:5.1f}" if oc else "")
                  + f" | {info['level']} IK markers {info.get('ik_marker_rms_cm', float('nan')):.1f} cm", flush=True)
    print(summary(rows, cams))
    OUT.mkdir(exist_ok=True)
    tag = "-".join(cams) + ("_augmented" if a.augment else "")
    old = json.loads((OUT / f"opensim_ik_{tag}.json").read_text()) if (OUT / f"opensim_ik_{tag}.json").exists() else []
    keep = [r for r in old if (r["subject"], r["trial"]) not in {(x["subject"], x["trial"]) for x in rows}]
    (OUT / f"opensim_ik_{tag}.json").write_text(json.dumps(keep + rows, indent=1))
    return rows


def summary(rows: list[dict], cams) -> str:
    out = [f"\nOpenSim joint angles vs laboratory (marker-based OpenSim IK). MAE in degrees (bias). OpenKick: {cams[0]}+{cams[1]}, "
           f"no calibration. OpenCap: its published result with 2 cameras ({OPENCAP_VIDEO})."]
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    for who in ("openkick", "opencap"):
        out.append(f"\n{who:10s} {'trials':>6s} " + "".join(f"{c:>20s}" for c in COORDS))
        for kind in kinds + (["ALL"] if len(kinds) > 1 else []):
            rs = [r for r in rows if (kind == "ALL" or r["kind"] == kind) and r[who]]
            if not rs:
                continue
            line = f"{kind:10s} {len(rs):6d} "
            for c in COORDS:
                line += f"{np.mean([r[who][c]['mae'] for r in rs]):12.1f} ({np.mean([r[who][c]['bias'] for r in rs]):+5.1f})"
            out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
