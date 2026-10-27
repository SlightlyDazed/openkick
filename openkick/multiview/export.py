"""From the reconstruction to OpenSim: marker file (.trc) in OpenSim's axes (X forward, Y up,
Z right, metres) and, when Pose2Sim is installed, automatic scaling + inverse kinematics."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from openkick.multiview.geometry import IDX

# Pose2Sim marker names for the HALPE-26 skeleton -> our point names
MARKERS = {"Hip": "hip", "RHip": "right_hip", "RKnee": "right_knee", "RAnkle": "right_ankle", "RBigToe": "right_toe",
           "RSmallToe": "right_small_toe", "RHeel": "right_heel", "LHip": "left_hip", "LKnee": "left_knee",
           "LAnkle": "left_ankle", "LBigToe": "left_toe", "LSmallToe": "left_small_toe", "LHeel": "left_heel",
           "Neck": "neck", "Head": "head", "Nose": "nose", "RShoulder": "right_shoulder", "RElbow": "right_elbow",
           "RWrist": "right_wrist", "LShoulder": "left_shoulder", "LElbow": "left_elbow", "LWrist": "left_wrist"}


def to_world(X: np.ndarray, height_m: float | None = None):
    """Reconstruction in camera A's axes (N, 26, 3; x right, y down, z away) -> OpenSim axes.
    The vertical is the body's own axis in the frames where the person stands tallest; the
    floor is put under the feet. `height_m` rescales (None: the points are already in metres)."""
    import warnings
    head = X[:, IDX["head"]]
    feet = (X[:, IDX["left_heel"]] + X[:, IDX["right_heel"]]) / 2
    axis = head - feet
    length = np.linalg.norm(axis, axis=1)
    ok = np.isfinite(length)
    if ok.sum() < 3:
        up = np.array([0, -1.0, 0])
    else:
        sel = ok & (length >= np.nanpercentile(length, 75))
        up = np.mean(axis[sel] / length[sel, None], axis=0)
        up /= np.linalg.norm(up)
    tilt = float(np.degrees(np.arccos(np.clip(up @ np.array([0, -1.0, 0]), -1, 1))))
    fwd = np.array([1.0, 0, 0]) - (np.array([1.0, 0, 0]) @ up) * up      # camera-right made horizontal
    fwd /= np.linalg.norm(fwd)
    R = np.stack([fwd, up, np.cross(fwd, up)])                            # rows: world axes in camera coords
    scale = 1.0 if height_m is None else height_m / float(np.nanpercentile(length, 90))
    W = (X @ R.T) * scale
    offset = np.zeros(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        floor = np.nanpercentile(W[:, [IDX["left_heel"], IDX["right_heel"], IDX["left_toe"], IDX["right_toe"]], 1], 2)
        if np.isfinite(floor):
            offset = np.array([np.nanmedian(W[:, IDX["hip"], 0]), floor, np.nanmedian(W[:, IDX["hip"], 2])])
            W -= offset
    return W, {"tilt_deg": tilt, "scale": scale, "R": R, "offset": offset}


def from_world(W: np.ndarray, info: dict) -> np.ndarray:
    """OpenSim axes back to the reconstruction's (to_world undone)."""
    return ((np.asarray(W, float) + info["offset"]) / info["scale"]) @ info["R"]


def uniform(t: np.ndarray, W: np.ndarray, rate: float = 30.0, cutoff: float | None = 8.0):
    """Resample to a constant rate (OpenSim needs it), bridge gaps and low-pass filter."""
    from scipy.signal import butter, filtfilt
    tu = np.arange(t[0], t[-1] + 1e-9, 1 / rate)
    out = np.empty((len(tu), W.shape[1], 3))
    for j in range(W.shape[1]):
        for c in range(3):
            v = W[:, j, c]
            ok = np.isfinite(v)
            out[:, j, c] = np.interp(tu, t[ok], v[ok]) if ok.sum() > 1 else np.nan
    if cutoff and len(tu) > 15 and cutoff < rate / 2:
        b, a = butter(4, cutoff / (rate / 2))
        good = np.isfinite(out).all(axis=(0, 2))
        out[:, good] = filtfilt(b, a, out[:, good], axis=0)
    return tu - tu[0], out


def write_trc(path: Path, t: np.ndarray, W: np.ndarray, rate: float):
    path = Path(path)
    names = list(MARKERS)
    n = len(t)
    with open(path, "w", newline="") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate}\t{rate}\t{n}\t{len(names)}\tm\t{rate}\t1\t{n}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t\n")     # Pose2Sim's augmenter needs this ending
        f.write("\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(names) + 1)) + "\n\n")
        for i in range(n):
            vals = "\t".join("\t".join("" if not np.isfinite(c) else f"{c:.6f}" for c in W[i, IDX[MARKERS[m]]]) for m in names)
            f.write(f"{i + 1}\t{t[i]:.6f}\t{vals}\n")


def export_trc(session, path: Path, rate: float | None = None, cutoff: float | None = 8.0) -> dict:
    """Write the session's 3D points as an OpenSim marker file. Returns what was written."""
    W, info = to_world(session.X)
    rate = rate or round(1 / float(np.median(np.diff(session.t))))
    t, U = uniform(session.t, W, rate, cutoff)
    write_trc(Path(path), t, U, rate)
    return {"frames": len(t), "rate": rate, "tilt_deg": info["tilt_deg"], "markers": len(MARKERS)}


def run_opensim(project: Path, trc: Path, height_m: float, mass_kg: float = 70.0, augment: bool = False) -> dict:
    """Scale the Pose2Sim skeleton model to the person and solve inverse kinematics.
    `augment` first adds anatomical markers predicted by a neural network (Stanford's marker
    augmenter, the one OpenCap uses), which tell OpenSim how the pelvis and the feet are turned.
    Returns the paths of the scaled model (.osim) and the motion (.mot)."""
    import Pose2Sim
    import rtoml
    from Pose2Sim import Pose2Sim as P2S
    project, trc = Path(project), Path(trc)
    (project / "pose-3d").mkdir(parents=True, exist_ok=True)
    if trc.resolve().parent != (project / "pose-3d").resolve():
        shutil.copy(trc, project / "pose-3d" / trc.name)
    cfg = rtoml.load(Path(Pose2Sim.__file__).parent / "Demo_SinglePerson" / "Config.toml")
    cfg["project"].update({"participant_height": height_m, "participant_mass": mass_kg, "multi_person": False})
    cfg["pose"]["pose_model"] = "Body_with_feet"
    cfg["kinematics"].update({"use_augmentation": augment, "use_simple_model": True, "filter_ik": False})
    cfg["markerAugmentation"] = {"feet_on_floor": False, "make_c3d": False}
    rtoml.dump(cfg, project / "Config.toml")
    cwd = os.getcwd()
    os.chdir(project)
    try:
        if augment:
            P2S.markerAugmentation()
        P2S.kinematics()
    finally:
        os.chdir(cwd)
    kin = project / "kinematics"
    return {"osim": sorted(kin.glob("*.osim")), "mot": sorted(kin.glob("*.mot")), "dir": kin}
