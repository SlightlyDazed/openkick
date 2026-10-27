"""Frame by frame, the three skeletons against the laboratory: for every trial the angles and the 3D
points of detected / model / measured, and the laboratory's, on the same clock. One .npz per trial."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from openkick.core.subject import Subject
from openkick.multiview import model_fit as MF
from validation import multiview as M
from validation import opencap as V

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
cams = ("Cam0", "Cam4")
modes = ("detected", "model", "measured")
IK = ["pelvis_tilt", "pelvis_list", "hip_flexion_r", "hip_flexion_l", "hip_adduction_r", "hip_adduction_l",
      "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
LAB = ["R_HJC", "L_HJC", "r_knee", "L_knee", "r_ankle", "L_ankle", "R_Shoulder", "L_Shoulder",
       "r_calc", "L_calc", "r_toe", "L_toe"]
subjects = sorted((p.name for p in V.ROOT.glob("subject*")), key=lambda s: int(s[7:]))
for kind in ("squats", "walking"):
    for subj in subjects:
        for tr in [t for t in V.trials_of_type(subj, kind) if V.video_path(subj, t, cams[0]) is not None][:1]:
            try:
                meta = yaml.safe_load(open(V.ROOT / subj / "sessionMetadata.yaml"))
                height, mass = float(meta["height_m"]), float(meta.get("mass_kg", 70) or 70)
                cal = [M.load_camera(subj, tr, c) for c in cams]
                ref = M.reference_3d(subj, tr)
                s = M.session_for(subj, tr, cams, height)
                offsets, t, X, _, _ = M.ceiling(s, cal)
                lag, fit_mm = M.mocap_lag(t, X, subj, tr)
                s.solve()
                tm, m = V.read_trc(V.ROOT / subj / "MarkerData" / "Mocap" / f"{tr}.trc")
                unit = 0.001 if np.nanmedian(np.linalg.norm(m["R_HJC"] - m["r_knee"], axis=1)) > 10 else 1.0
                lab = {}
                for side, S, sl in (("right", "R", "r"), ("left", "L", "L")):
                    lab[f"thigh_{side}"] = 100 * unit * float(np.nanmedian(np.linalg.norm(m[f"{S}_HJC"] - m[f"{sl}_knee"], axis=1)))
                    lab[f"shank_{side}"] = 100 * unit * float(np.nanmedian(np.linalg.norm(m[f"{sl}_knee"] - m[f"{sl}_ankle"], axis=1)))
                s.subject = Subject(height_m=height, mass_kg=mass, lengths=lab)
                when = s.t - lag
                d = {"t": s.t, "lag": lag, "fit_mm": fit_mm, "lab_lengths": np.array([lab[k] for k in sorted(lab)]),
                     "lab_length_names": np.array(sorted(lab))}
                for k, v in ref.items():
                    if k != "t":
                        d[f"ref_{k}"] = np.interp(when, ref["t"], v, left=np.nan, right=np.nan)
                ti, c = V.read_mot(V.ROOT / subj / "OpenSimData" / "Mocap" / "IK" / f"{tr}.mot")
                for k in IK:
                    d[f"labik_{k}"] = np.interp(when, ti, c[k], left=np.nan, right=np.nan)
                oc = V.ROOT / subj / "OpenSimData" / "Video" / "HRNet" / "2-cameras" / "IK" / f"{tr}.mot"
                if oc.exists():                      # OpenCap's own two cameras, on its own clock
                    to, co = V.read_mot(oc)
                    d["opencap_t"] = to
                    for k in IK:
                        d[f"opencap_{k}"] = co[k]
                    d["labik_t"] = ti
                    for k in IK:
                        d[f"labikraw_{k}"] = c[k]
                for mk in LAB:
                    d[f"lab_{mk}"] = np.stack([np.interp(when, tm, m[mk][:, c] * unit, left=np.nan, right=np.nan)
                                               for c in range(3)], axis=1)
                for mode in modes:
                    s.skeleton = mode
                    if mode != "detected":
                        folder = Path(tempfile.mkdtemp(prefix="frames_"))
                        s.model = MF.fit(s, mode, folder=folder)
                        if not s.model["ok"]:
                            raise RuntimeError(s.model["error"][-300:])
                        z = np.load(folder / "model.npz")
                        labels, data = list(z["labels"]), np.asarray(z["data"], float)
                        d[f"{mode}_ik_t"] = np.asarray(z["times"], float) + s.t[0]      # on the session's clock
                        for k in IK:
                            if k in labels:
                                d[f"{mode}_ik_{k}"] = data[:, labels.index(k)]
                        d[f"{mode}_rms_cm"] = s.model["rms_cm"]
                        d[f"{mode}_lengths"] = np.array([s.model["lengths"].get(k, np.nan) for k in
                                                         ("right_thigh", "right_shank", "left_thigh", "left_shank")])
                        dev = s.model_deviation()
                        d[f"{mode}_deviation"] = dev
                    s.update(recalibrate=False)
                    d[f"{mode}_X"] = s.X.copy()
                    for k, v in M.angles_from_keypoints(s.X).items():
                        d[f"{mode}_{k}"] = v
                np.savez_compressed(OUT / f"{kind}_{subj}_{tr}.npz", **d)
                print(kind, subj, tr, "ok", flush=True)
            except Exception as ex:  # noqa: BLE001
                print(kind, subj, tr, "failed", repr(ex)[:300], flush=True)
