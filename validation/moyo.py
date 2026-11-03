"""MOYO (Max Planck Institute for Intelligent Systems, 2023) as a second laboratory reference.

One yoga instructor, about 170 recordings, 8 calibrated cameras and Vicon markers. The dataset is
not redistributed: register at https://moyo.is.tue.mpg.de/, download it (see
docs/VALIDATION.md) and point OPENKICK_MOYO at the folder `20220923_20220926_with_hands`.

The laboratory reference is built here the way the OpenCap laboratory built its own: the same
OpenSim model (LaiArnoldModified2017, generic, taken from the OpenCap dataset), scaled with a
standing pose and the hip joint centres (functional: see functional_hjc), then inverse kinematics with the same marker
weights. The markers come labelled inside MoSh's fits (`mosh/*/..._stageii.pkl`); the .c3d files
of the dataset are unlabelled.

    python -m validation.moyo reference --seq Tree_Pose_or_Vrksasana_-a
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from validation import opencap as V

ROOT = Path(os.environ.get("OPENKICK_MOYO",
                           Path(__file__).resolve().parents[2] / "datos/moyo/20220923_20220926_with_hands"))
WORK = ROOT.parent / "_openkick"
GENERIC = V.ROOT / "subject2/OpenSimData/Mocap/Model/LaiArnoldModified2017_poly_withArms_weldHand_generic.osim"
LAB_SCALE = V.ROOT / "subject2/OpenSimData/Mocap/Model/LaiArnoldModified2017_poly_withArms_weldHand_setup_scale.xml"
RATE = 60.0
STATIC_SEQ = "220923_yogi_body_hands_03596_Tree_Pose_or_Vrksasana_-a"   # starts standing still
STATIC_S = (0.1, 1.1)                                                     # as long as the lab's static trial


# ── reading MoSh's files without running anything from them ───────────


_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"), ("collections", "OrderedDict")}


class _OnlyArrays(pickle.Unpickler):
    """A pickle can run code when it is loaded; this one only rebuilds numpy arrays and containers."""

    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def load_fit(path: Path) -> dict:
    with open(path, "rb") as f:
        return _OnlyArrays(f, encoding="latin1").load()


def sequences() -> list[str]:
    return sorted(p.name[:-len("_stageii.pkl")] for p in ROOT.glob("mosh/*/*_stageii.pkl"))


def fit_path(seq: str) -> Path:
    hits = [p for p in ROOT.glob(f"mosh/*/{seq}_stageii.pkl")] or [p for p in ROOT.glob(f"mosh/*/*{seq}*_stageii.pkl")]
    if len(hits) != 1:
        raise FileNotFoundError(f"{seq}: {len(hits)} fits found")
    return hits[0]


def markers(seq: str) -> tuple[list[str], np.ndarray]:
    """The labelled Vicon markers: names and (N, K, 3) in metres, laboratory axes (z up)."""
    d = load_fit(fit_path(seq))["stageii_debug_details"]
    labels = [str(s) for s in d["labels_orig"]]
    X = np.asarray(d["markers_orig"], float)
    X[np.all(X == 0, axis=-1)] = np.nan
    return labels, X


# ── from the laboratory's markers to OpenSim's ─────────────────────────


def harrington(rasis, lasis, rpsis, lpsis):
    """Hip joint centres (Harrington et al. 2007) from the pelvis markers, in metres."""
    mid_asis, mid_psis = (rasis + lasis) / 2, (rpsis + lpsis) / 2
    z = rasis - lasis
    pw = np.linalg.norm(z, axis=-1, keepdims=True)
    z = z / pw
    x = mid_asis - mid_psis
    x = x - np.sum(x * z, -1, keepdims=True) * z
    pd = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / pd
    y = np.cross(z, x)
    pw, pd = pw * 1000, pd * 1000                         # the regression is in millimetres
    ax, ay, az = -0.24 * pd - 9.9, -0.30 * pw - 10.9, 0.33 * pw + 7.3
    right = mid_asis + (ax * x + ay * y + az * z) / 1000
    left = mid_asis + (ax * x + ay * y - az * z) / 1000
    return right, left


def pelvis_frame(g: dict):
    """Origin between the front waist markers; x forward, y up, z right (columns of R)."""
    ra, la, rp, lp = g["RFWT"], g["LFWT"], g["RBWT"], g["LBWT"]
    o = (ra + la) / 2
    z = ra - la
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    x = o - (rp + lp) / 2
    x = x - np.sum(x * z, -1, keepdims=True) * z
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return o, np.stack([x, np.cross(z, x), z], -1)


def functional_hjc() -> np.ndarray:
    """Hip joint centre in the pelvis frame, found as the point the thigh turns about.

    MOYO's pelvis markers are "waist" markers, about 7 cm above the anterior iliac spines, so
    Harrington's regression (made for markers on the spines) put the centre 5 cm too high: thigh
    49 cm long for a person about 1.78 m tall. Here the knee and thigh markers of every fourth
    recording are seen from the pelvis and one sphere centre is fitted to them per side (the
    usual functional method; yoga moves the hip in every direction). Both sides are averaged,
    mirrored: they differ by about 2 cm, which is skin motion. Thigh length then 44 cm."""
    f = WORK / "model" / "hjc.npy"
    if f.exists():
        return np.load(f)
    centres = []
    for side, sign in (("R", 1), ("L", -1)):
        rows, rhs, k = [], [], 0
        for seq in sequences()[::4]:
            labels, X = markers(seq)
            g = {n: X[:, labels.index(n)] for n in ("RFWT", "LFWT", "RBWT", "LBWT")}
            o, R = pelvis_frame(g)
            for j, m in enumerate(("KNE", "KNI", "THI")):
                p = np.einsum("nij,ni->nj", R, X[:, labels.index(side + m)] - o)[::6]
                p = p[np.isfinite(p).all(1)]
                e = np.zeros((len(p), 3))
                e[:, j] = 1                                  # each marker its own radius
                rows.append(np.hstack([2 * p, e]))
                rhs.append((p ** 2).sum(1))
        sol = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        centres.append(sol[:3] * np.array([1, 1, sign]))
    c = np.mean(centres, axis=0)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, c)
    return c


def opensim_markers(labels: list[str], X: np.ndarray) -> dict:
    """Laboratory markers plus the virtual ones the scaling uses (hip centres, shoulders), with
    OpenSim's axes: y up, the person facing +x at the start (only the start: IK starts there)."""
    g = {n: X[:, labels.index(n)] for n in labels if not n.startswith("M_")}   # M_*: the yoga mat
    o, R = pelvis_frame(g)
    c = functional_hjc()
    g["RHJC"] = o + np.einsum("nij,j->ni", R, c)
    g["LHJC"] = o + np.einsum("nij,j->ni", R, c * np.array([1, 1, -1]))
    g["RSHO"], g["LSHO"] = (g["RFSH"] + g["RBSH"]) / 2, (g["LFSH"] + g["LBSH"]) / 2
    # laboratory z up -> OpenSim y up, then turned about the vertical so that the pelvis's right
    # points to +z in the first frames
    to_y_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    g = {k: v @ to_y_up.T for k, v in g.items()}
    right = np.nanmean((g["RFWT"] - g["LFWT"])[:30], axis=0)
    a = np.arctan2(right[0], right[2])                     # angle of "right" from +z, about y
    c, s = np.cos(a), np.sin(a)
    turn = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    start = np.nanmean(((g["RHJC"] + g["LHJC"]) / 2)[:30], axis=0) * np.array([1, 0, 1])
    return {k: (v - start) @ turn.T for k, v in g.items()}


def write_trc(path: Path, g: dict, rate: float = RATE, frames: slice = slice(None)):
    names = list(g)
    W = np.stack([g[n][frames] for n in names], axis=1)
    n = len(W)
    t = np.arange(n) / rate
    with open(path, "w", newline="") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate}\t{rate}\t{n}\t{len(names)}\tm\t{rate}\t1\t{n}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n")
        f.write("\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(names) + 1)) + "\n\n")
        for i in range(n):
            vals = "\t".join("" if not np.isfinite(c) else f"{c:.6f}" for c in W[i].ravel())
            f.write(f"{i + 1}\t{t[i]:.6f}\t{vals}\n")


# MOYO's markers on the laboratory model: the lab marker whose place it takes (its body and first
# location come from there; the scaling then puts every marker where the standing pose shows it),
# or (body, location) for those the lab model has no counterpart for
SAME = {"C7": "C7", "RSHO": "R_Shoulder", "LSHO": "L_Shoulder", "CLAV": "R_Sternum",
        "RFWT": "r.ASIS", "LFWT": "L.ASIS", "RBWT": "r.PSIS", "LBWT": "L.PSIS", "RHJC": "R_HJC", "LHJC": "L_HJC",
        "RTHI": "r_thigh1", "LTHI": "L_thigh1", "RKNE": "r_knee", "RKNI": "r_mknee", "LKNE": "L_knee", "LKNI": "L_mknee",
        "RSHN": "r_shank_antsup", "LSHN": "L_shank_antsup", "RANK": "r_ankle", "LANK": "L_ankle",
        "RTOE": "r_toe", "RMT5": "r_5meta", "RHEE": "r_calc", "LTOE": "L_toe", "LMT5": "L_5meta", "LHEE": "L_calc",
        "RELB": "R_elbow_lat", "RELBIN": "R_elbow_med", "LELB": "L_elbow_lat", "LELBIN": "L_elbow_med"}
OWN = {"STRN": ("torso", (0.07, 0.30, 0.0)), "T10": ("torso", (-0.12, 0.22, 0.0)),
       "RFSH": ("torso", (0.04, 0.42, 0.13)), "RBSH": ("torso", (-0.05, 0.42, 0.12)),
       "LFSH": ("torso", (0.04, 0.42, -0.13)), "LBSH": ("torso", (-0.05, 0.42, -0.12)),
       "MFWT": ("pelvis", (0.03, 0.0, 0.0)), "MBWT": ("pelvis", (-0.17, 0.03, 0.0)),
       "RMT1": ("calcn_r", (0.135, 0.0, -0.03)), "LMT1": ("calcn_l", (0.135, 0.0, 0.03)),
       "RUPA": ("humerus_r", (0.0, -0.15, 0.04)), "LUPA": ("humerus_l", (0.0, -0.15, -0.04)),
       "RFRM": ("radius_r", (0.0, -0.10, 0.03)), "LFRM": ("radius_l", (0.0, -0.10, -0.03))}
# the weights of the laboratory's inverse kinematics, for their counterparts
IK_WEIGHT = {"C7": 5, "RSHO": 5, "LSHO": 5, "CLAV": 5, "STRN": 2, "T10": 2,
             "RFWT": 25, "LFWT": 25, "RBWT": 25, "LBWT": 25, "MFWT": 10, "MBWT": 10,
             "RTHI": 4, "LTHI": 4, "RKNE": 20, "LKNE": 20, "RKNI": 0, "LKNI": 0, "RSHN": 4, "LSHN": 4,
             "RANK": 20, "LANK": 20, "RTOE": 7, "LTOE": 7, "RMT5": 7, "LMT5": 7, "RMT1": 7, "LMT1": 7,
             "RHEE": 7, "LHEE": 7, "RELB": 5, "RELBIN": 5, "LELB": 5, "LELBIN": 5,
             "RWRR": 5, "RWRU": 5, "LWRR": 5, "LWRU": 5, "RUPA": 2, "LUPA": 2, "RFRM": 2, "LFRM": 2}


def wrist_names(labels: list[str], X: np.ndarray) -> dict:
    """Which of the inner and outer wrist markers sits on the radius (the thumb's side)."""
    out = {}
    for s in "RL":
        thumb = X[:, labels.index(f"{s}THMB")]
        d = {w: np.nanmedian(np.linalg.norm(X[:, labels.index(f"{s}{w}WR")] - thumb, axis=-1)) for w in "IO"}
        radius = min(d, key=d.get)
        out[f"{s}{radius}WR"], out[f"{s}{'O' if radius == 'I' else 'I'}WR"] = f"{s}WRR", f"{s}WRU"
    return out


def build_model(path: Path) -> Path:
    """The laboratory's generic model with MOYO's markers instead of the laboratory's."""
    tree = ET.parse(GENERIC)
    root = tree.getroot()
    mset = root.find(".//MarkerSet/objects")
    lab = {m.get("name"): m for m in mset.findall("Marker")}
    for m in list(mset):
        mset.remove(m)

    def add(name, body, loc):
        m = ET.SubElement(mset, "Marker", name=name)
        ET.SubElement(m, "socket_parent_frame").text = f"/bodyset/{body}"
        ET.SubElement(m, "location").text = " ".join(f"{v:.6f}" for v in loc)
        ET.SubElement(m, "fixed").text = "false"
    for name, src in SAME.items():
        m = lab[src]
        add(name, m.findtext("socket_parent_frame").split("/")[-1], [float(v) for v in m.findtext("location").split()])
    for name, (body, loc) in OWN.items():
        add(name, body, loc)
    for side, sign in (("R", 1), ("L", -1)):            # wrists: on the radius, from the lab's
        for kind, src in (("WRR", "wrist_radius"), ("WRU", "wrist_ulna")):
            m = lab[f"{side}_{src}"]
            add(f"{side}{kind}", m.findtext("socket_parent_frame").split("/")[-1],
                [float(v) for v in m.findtext("location").split()])
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path)
    return path


RENAME = {"C7": "C7", "R_Shoulder": "RSHO", "L_Shoulder": "LSHO", "R_Sternum": "CLAV", "L_Sternum": None,
          "r.ASIS": "RFWT", "L.ASIS": "LFWT", "r.PSIS": "RBWT", "L.PSIS": "LBWT", "R_HJC": "RHJC", "L_HJC": "LHJC",
          "r_knee": "RKNE", "r_mknee": "RKNI", "L_knee": "LKNE", "L_mknee": "LKNI", "r_ankle": "RANK",
          "L_ankle": "LANK", "r_mankle": None, "L_mankle": None, "r_toe": "RTOE", "r_5meta": "RMT5",
          "r_calc": "RHEE", "L_toe": "LTOE", "L_5meta": "LMT5", "L_calc": "LHEE",
          "R_elbow_lat": "RELB", "R_elbow_med": "RELBIN", "L_elbow_lat": "LELB", "L_elbow_med": "LELBIN",
          "R_wrist_radius": "RWRR", "R_wrist_ulna": "RWRU", "L_wrist_radius": "LWRR", "L_wrist_ulna": "LWRU"}


def scale(model: Path, static_trc: Path, out: Path, mass: float = 60.0) -> Path:
    """The laboratory's scaling setup with MOYO's marker names: the same segment measurements (the
    pairs that use the medial malleolus, which MOYO lacks, are dropped) and marker placement."""
    tree = ET.parse(LAB_SCALE)
    root = tree.getroot()
    tool = root.find("ScaleTool")
    tool.find("mass").text = str(mass)
    tool.find("height").text = "-1"
    # OpenSim reads the file names of a setup relative to the setup's folder, even absolute ones
    rel = lambda f: os.path.relpath(f, out.parent)
    tool.find("GenericModelMaker/model_file").text = rel(model)
    for pair_set in tool.iter("MarkerPairSet"):
        objects = pair_set.find("objects")
        for pair in list(objects):
            names = pair.findtext("markers").split()
            new = [RENAME.get(n) for n in names]
            if None in new:
                objects.remove(pair)
            else:
                if new == ["RWRR", "LELBIN"]:      # the laboratory's file pairs the right wrist with the left elbow
                    new = ["LWRR", "LELBIN"]
                pair.find("markers").text = " ".join(new)
    for task_set in tool.iter("IKTaskSet"):
        objects = task_set.find("objects")
        for task in list(objects.findall("IKMarkerTask")):
            new = RENAME.get(task.get("name"))
            if new is None:
                objects.remove(task)
            else:
                task.set("name", new)
    t0, t1 = STATIC_S
    for sec in ("ModelScaler", "MarkerPlacer"):
        s = tool.find(sec)
        s.find("marker_file").text = rel(static_trc)
        s.find("time_range").text = f"{t0} {t1}"
    tool.find("ModelScaler/output_model_file").text = out.stem + "_scaled_only.osim"
    tool.find("ModelScaler/output_scale_file").text = out.stem + "_scale_factors.xml"
    tool.find("MarkerPlacer/output_model_file").text = out.name
    tool.find("MarkerPlacer/output_motion_file").text = out.stem + "_static.mot"
    for f in ("GenericModelMaker/marker_set_file", "MarkerPlacer/output_marker_file", "ModelScaler/coordinate_file",
              "MarkerPlacer/coordinate_file"):
        e = tool.find(f)
        if e is not None:
            e.text = "Unassigned"
    setup = out.with_name(out.stem + "_setup_scale.xml")
    tree.write(setup)
    import opensim
    opensim.ScaleTool(str(setup)).run()
    if not out.exists():
        raise RuntimeError(f"OpenSim did not write {out}")
    return out


def inverse_kinematics(model: Path, trc: Path, out: Path) -> Path:
    """The laboratory's inverse kinematics (its setup file, its weights) with MOYO's markers."""
    lab = ET.parse(V.ROOT / "subject2/OpenSimData/Mocap/IK/squats1_setup_ik.xml")
    tool = lab.getroot().find("InverseKinematicsTool")
    # unlike the scaling, this tool wants full paths
    tool.find("model_file").text = str(model)
    tool.find("marker_file").text = str(trc)
    tool.find("output_motion_file").text = str(out)
    tool.find("results_directory").text = str(out.parent)
    n = sum(1 for line in open(trc)) - 6
    tool.find("time_range").text = f"0 {(n - 1) / RATE}"
    tool.find("report_marker_locations").text = "true"
    tool.set("name", out.stem)
    tasks = tool.find("IKTaskSet/objects")
    for task in list(tasks):
        tasks.remove(task)
    for name, w in IK_WEIGHT.items():
        e = ET.SubElement(tasks, "IKMarkerTask", name=name)
        ET.SubElement(e, "apply").text = "true" if w > 0 else "false"
        ET.SubElement(e, "weight").text = str(float(w))
    setup = out.with_name(out.stem + "_setup_ik.xml")
    lab.write(setup)
    import opensim
    opensim.InverseKinematicsTool(str(setup)).run()
    if not out.exists():
        raise RuntimeError(f"OpenSim did not write {out}")
    return out


def subject_model() -> Path:
    """The scaled laboratory model of the instructor, built once from the standing start."""
    out = WORK / "model" / "moyo_03596.osim"
    if out.exists():
        return out
    labels, X = markers(STATIC_SEQ)
    g = opensim_markers(labels, X)
    g = {wrist_names(labels, X).get(k, k): v for k, v in g.items()}
    trc = WORK / "model" / "static.trc"
    trc.parent.mkdir(parents=True, exist_ok=True)
    write_trc(trc, g, frames=slice(0, int(1.5 * RATE)))
    return scale(build_model(WORK / "model" / "generic_moyo.osim"), trc, out)


def unclamped(model: Path) -> Path:
    """The same model without joint limits: yoga goes past them (knee flexed beyond 140 degrees,
    hip rotated beyond 40 in the tree pose), and the reference has to follow the real movement."""
    out = model.with_name(model.stem + "_unclamped.osim")
    if not out.exists():
        out.write_text(model.read_text().replace("<clamped>true</clamped>", "<clamped>false</clamped>"))
    return out


def reference(seq: str) -> Path:
    """The laboratory's joint angles for one sequence (OpenSim .mot), computed once."""
    out = WORK / "ik" / f"{seq}.mot"
    if out.exists():
        return out
    model = unclamped(subject_model())
    labels, X = markers(seq)
    g = opensim_markers(labels, X)
    g = {wrist_names(labels, X).get(k, k): v for k, v in g.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    trc = out.with_suffix(".trc")
    write_trc(trc, g)
    return inverse_kinematics(model, trc, out)


# ── the videos, as OpenKick sees them ──────────────────────────────────

VIDEO_WIDTH = 1920          # the images (2K-4K) are turned into 1920-pixel-wide videos, a phone's
IMAGE_FPS = 30.0            # the images are every other frame of the 60 Hz laboratory


def camera_names(seq: str) -> list[str]:
    """The cameras with every image (a download cut short leaves some with fewer)."""
    counts = {p.name: len(frames_of(p)) for p in image_folder(seq).iterdir() if p.is_dir()}
    full = max(counts.values())
    return sorted(c for c, n in counts.items() if n == full)


def image_folder(seq: str) -> Path:
    hits = [p for p in ROOT.glob(f"images/*/{seq}") if p.is_dir()]
    if not hits:
        zips = list(ROOT.glob(f"images/*/{seq}.zip"))
        if not zips:
            raise FileNotFoundError(f"no images for {seq}: download them (datos/moyo/descargar_moyo.sh)")
        import subprocess                 # bsdtar also reads a zip cut short, up to where it ends
        subprocess.run(["bsdtar", "-xf", str(zips[0]), "-C", str(zips[0].parent)], check=False)
        hits = [p for p in ROOT.glob(f"images/*/{seq}") if p.is_dir()]
    return hits[0]


def frames_of(folder: Path) -> list[tuple[int, Path]]:
    """The images of one camera with their frame numbers (the number at the end of the name)."""
    out = []
    for f in folder.glob("*.jpg"):
        num = f.stem.rsplit("_", 1)[-1]
        if num.isdigit():
            out.append((int(num), f))
    return sorted(out)


def calibration(seq: str, cams: list[str]) -> list[dict]:
    """The dataset's cameras for this recording's day, in pixels of the videos made here:
    K (3x3), dist (zeros), R, t with x_cam = R X + t (millimetres, laboratory axes)."""
    import json
    day = "20" + seq[:6]
    path = next(ROOT.glob(f"cameras/{day}/*PROCESSED_CAMERA_PARAMS/cameras_param.json"))
    params = json.loads(path.read_text())
    out = []
    for c in cams:
        p = params[f"cam_{int(c.rsplit('_', 1)[-1])}"]
        w0 = _image_size(image_folder(seq) / c)[0]
        # the calibration was made at the cameras' full resolution, twice that of the published
        # images (its principal point falls beyond their middle)
        full = 2 * w0 if p["princpt"][0] > 0.75 * w0 else w0
        k = VIDEO_WIDTH / full
        K = np.array([[p["focal"] * k, 0, p["princpt"][0] * k], [0, p["focal"] * k, p["princpt"][1] * k], [0, 0, 1]])
        R = np.asarray(p["rotation"], float)
        out.append({"K": K, "dist": np.zeros(5), "R": R, "t": -R @ np.asarray(p["position"], float)})
    return out


def _image_size(folder: Path) -> tuple[int, int]:
    import cv2
    h, w = cv2.imread(str(frames_of(folder)[0][1])).shape[:2]
    return w, h


def make_video(seq: str, cam: str) -> tuple[Path, int]:
    """One camera's images as a video (H.264, quality like a phone's), and the first image's
    frame number. Made once."""
    import json
    import subprocess
    out = WORK / "video" / seq / f"{cam}.mp4"
    info = out.with_suffix(".json")
    if out.exists() and info.exists():
        return out, json.loads(info.read_text())["first"]
    frames = frames_of(image_folder(seq) / cam)
    nums = [n for n, _ in frames]
    if nums != list(range(nums[0], nums[0] + len(nums))):
        raise ValueError(f"{seq} {cam}: missing images")
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{f}'\nduration {1 / IMAGE_FPS}\n" for _, f in frames))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-vf", f"scale={VIDEO_WIDTH}:-2", "-r", str(IMAGE_FPS), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    listing.unlink()
    info.write_text(json.dumps({"first": nums[0], "count": len(nums)}))
    return out, nums[0]


def lens_equivalent(K: np.ndarray, width: int, height: int) -> float:
    """The full-frame equivalent focal length (mm) of a camera matrix: guess_intrinsics' inverse."""
    return float(21.63 * K[0, 0] / (np.hypot(width, height) / 2))


def reference_angles(seq: str) -> dict:
    """The laboratory's angles in the app's names, on the laboratory's clock (t = 0: first frame)."""
    t, m = V.read_mot(reference(seq))
    out = {"t": t}
    for s in "rl":
        out[f"knee_{s}"], out[f"hip_{s}"], out[f"ankle_{s}"] = m[f"knee_angle_{s}"], m[f"hip_flexion_{s}"], m[f"ankle_angle_{s}"]
        out[f"elbow_{s}"] = m[f"elbow_flex_{s}"]
        out[f"shoulder_{s}"] = np.degrees(np.arccos(np.cos(np.radians(m[f"arm_flex_{s}"])) * np.cos(np.radians(m[f"arm_add_{s}"]))))
    return out


def evaluate(seq: str, cams: list[str], lens: str = "phone") -> dict:
    """OpenKick's 3D with these cameras (no calibration; lens "phone" = the 26 mm the app assumes,
    "true" = the camera's real one) and the ceiling (the dataset's calibration, all cameras at the
    same instant: they are hardware-synchronised), against the laboratory's angles."""
    from openkick.multiview import geometry as G
    from openkick.multiview.session import Session
    from validation import multiview as M
    ref = reference_angles(seq)
    res = {"seq": seq, "cams": cams, "lens": lens}
    try:
        s, cal, lag = solve(seq, cams, lens)
    except ValueError as e:
        res["failed"] = repr(e)
        return res
    res["lag_s"] = lag
    res["lens_mm"] = [lens_equivalent(c["K"], v.meta.width, v.meta.height) for c, v in zip(cal, s.views)]
    t_c, n = M._observed(Session(s.views, HEIGHT_M), [0.0] * len(cams), cal)
    rel = [(np.eye(3), np.zeros(3))] + [M.relative_pose(cal[0], c) for c in cal[1:]]
    X_c = G.triangulate_n(n, rel) / 1000
    res["ceiling"] = M.score_angles(t_c, M.angles_from_keypoints(X_c, t_c, "yoga"), ref, lag)
    q = s.quality()
    body = M.score_angles(s.t, M.angles_from_keypoints(s.X, s.t, "yoga"), ref, lag)
    diffs = [M.pose_difference(*s.cams[i], *rel[i]) for i in range(1, len(cams))]
    body.update({"rotation_deg": float(np.mean([d["rotation_deg"] for d in diffs])),
                 "rotation_max_deg": float(np.max([d["rotation_deg"] for d in diffs])),
                 "offsets_ms": [float(o * 1000) for o in s.offsets[1:]], "level": q["level"],
                 "reprojection_px": float(s.rms), "reconstructed": q["reconstructed"]})
    res["body"] = body
    return res


OPENSIM = {"knee": "knee_angle_{s}", "hip_flexion": "hip_flexion_{s}", "hip_adduction": "hip_adduction_{s}",
           "hip_rotation": "hip_rotation_{s}", "ankle": "ankle_angle_{s}"}
TRUNK = {"pelvis_tilt": ("pelvis_tilt", 1), "pelvis_list": ("pelvis_list", 1),
         "lumbar_flexion": ("L5_S1_Flex_Ext", "lumbar_extension")}


def solve(seq: str, cams: list[str], lens: str = "phone"):
    """OpenKick's 3D from these cameras, as a user gets it (no calibration, no corrections)."""
    from fractions import Fraction

    from openkick.core.video import VideoMeta, probe
    from openkick.multiview import geometry as G
    from openkick.multiview.session import Session
    from validation import multiview as M
    cal = calibration(seq, cams)
    views, first = [], None
    for c, cc in zip(cams, cal):
        video, n0 = make_video(seq, c)
        first = n0 if first is None else first
        t, xy, score = M.detect_full(video)
        meta = probe(video)
        K = cc["K"] if lens == "true" else G.guess_intrinsics(meta.width, meta.height)
        views.append(M.View(VideoMeta(str(video), meta.width, meta.height, IMAGE_FPS, Fraction(1, 1000)), t, xy, score, K=K))
    s = Session(views, HEIGHT_M)
    s.solve()
    return s, cal, -(first - 1) / IMAGE_FPS


HEIGHT_M = 1.78                  # the instructor: her head marker stands at 1.70 m


def evaluate_opensim(seq: str, cams: list[str], lens: str = "phone") -> dict:
    """The whole chain: OpenKick's 3D -> OpenSim (Pose2Sim, as the app exports it) -> model
    angles, against the laboratory's OpenSim angles. Also the pelvis and the lumbar spine: the
    question is whether the model's excess of hip flexion is the pelvis's fault."""
    import json

    from openkick.multiview.export import export_trc, run_opensim
    project = WORK / "opensim" / lens / "-".join(c.rsplit("_", 1)[-1] for c in cams) / seq
    info_file = project / "openkick.json"
    if info_file.exists():
        info = json.loads(info_file.read_text())
    else:
        s, _, lag = solve(seq, cams, lens)
        (project / "pose-3d").mkdir(parents=True, exist_ok=True)
        trc = project / "pose-3d" / f"{seq}.trc"
        export_trc(s, trc, rate=60, cutoff=8.0)
        out = run_opensim(project, trc, HEIGHT_M, 60.0)
        info = {"mot": str(out["mot"][0]), "t0": float(s.t[0]), "lag": lag, "level": s.quality()["level"]}
        info_file.write_text(json.dumps(info))
    tr, ref = V.read_mot(reference(seq))
    t, est = V.read_mot(Path(info["mot"]))
    tm = t + info["t0"] - info["lag"]
    res = {"seq": seq, "cams": cams, "lens": lens, "level": info["level"]}
    for name, pattern in OPENSIM.items():
        ms = [V.metrics(est[pattern.format(s=x)], np.interp(tm, tr, ref[pattern.format(s=x)], left=np.nan, right=np.nan))
              for x in "rl"]
        ms = [m for m in ms if "mae" in m]
        res[name] = {k: float(np.mean([m[k] for m in ms])) for k in ("mae", "bias", "sd", "r", "rom_est", "rom_ref")}
    for name, (ours, lab) in TRUNK.items():
        lab_name = lab if isinstance(lab, str) else ours
        sign = -1.0 if name == "lumbar_flexion" else 1.0            # both as flexion: the lab's is extension
        m = V.metrics(est[ours] if name != "lumbar_flexion" else est[ours],
                      sign * np.interp(tm, tr, ref[lab_name], left=np.nan, right=np.nan))
        res[name] = {k: float(m[k]) for k in ("mae", "bias", "sd", "r", "rom_est", "rom_ref")} if "mae" in m else {}
    return res


# The cameras around the instructor (recording of 23 September; azimuth from where she faces):
# 1 front (-3 deg, 5 m), 2 and 6 front diagonals (-53, +50), 3 and 5 sides (-95, +89), 4 behind
# (175), 7 and 8 high in front (4.5 m up, -27 and +22). Sets that a user could have, small to large.
CAMERA_SETS = (("3", "6"), ("2", "6"), ("1", "3"), ("2", "3"), ("2", "3", "6"), ("1", "2", "3", "6"),
               ("1", "2", "3", "5", "6"), ("1", "2", "3", "4", "5", "6"), ("1", "2", "3", "6", "8"))


def camera_sets(have: list[str]) -> list[list[str]]:
    by_number = {str(int(c.rsplit("_", 1)[-1])): c for c in have}
    return [[by_number[n] for n in cs] for cs in CAMERA_SETS if all(n in by_number for n in cs)]


def summary_line(r: dict) -> str:
    cams = "+".join(c.rsplit("_", 1)[-1].lstrip("0") for c in r["cams"])
    if "failed" in r:
        return f"{r['seq'][29:60]:31s} {cams:12s} {r['lens']:5s} FAILED {r['failed'][:80]}"
    j = ("knee", "hip", "ankle")
    c = " / ".join(f"{r['ceiling'][k]['mae']:4.1f}" for k in j)
    b = " / ".join(f"{r['body'][k]['mae']:4.1f}" for k in j)
    line = (f"{r['seq'][29:60]:31s} {cams:12s} {r['lens']:5s} ceiling {c}   openkick {b}   cam rot {r['body']['rotation_deg']:5.1f}"
            f"  {r['body']['level']}")
    if "opensim" in r:
        o = r["opensim"]
        line += "   opensim " + " / ".join(f"{o[k]['mae']:4.1f}" for k in ("knee", "hip_flexion", "ankle"))
    return line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["list", "model", "reference", "videos", "evaluate"])
    ap.add_argument("--lens", nargs="*", default=["phone", "true"])
    ap.add_argument("--opensim", action="store_true", help="also the OpenSim chain")
    ap.add_argument("--seq", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.what == "list":
        print("\n".join(sequences()))
    elif a.what == "model":
        print(subject_model())
    elif a.what == "videos":                   # videos and detections, one camera at a time
        from validation import multiview as M
        for q in a.seq:
            seq = fit_path(q).name[:-len("_stageii.pkl")]
            for c in camera_names(seq):
                video, _ = make_video(seq, c)
                M.detect_full(video)
                print(seq, c, "ok", flush=True)
    elif a.what == "evaluate":
        import json
        rows = []
        for q in a.seq:
            seq = fit_path(q).name[:-len("_stageii.pkl")]
            have = camera_names(seq)
            for cams in camera_sets(have):
                for lens in a.lens:
                    r = evaluate(seq, cams, lens)
                    if a.opensim and "failed" not in r:
                        r["opensim"] = evaluate_opensim(seq, cams, lens)
                    rows.append(r)
                    print(summary_line(r), flush=True)
        out = V.OUT / "moyo_evaluate.json"
        old = json.loads(out.read_text()) if out.exists() else []
        keep = [r for r in old if (r["seq"], tuple(r["cams"]), r["lens"]) not in {(x["seq"], tuple(x["cams"]), x["lens"]) for x in rows}]
        out.write_text(json.dumps(keep + rows, indent=1))
    else:
        for s in a.seq:
            seq = fit_path(s).name[:-len("_stageii.pkl")]
            print(reference(seq))


if __name__ == "__main__":
    sys.exit(main())
