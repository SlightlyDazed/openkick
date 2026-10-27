"""The OpenSim step, run as a separate process so the window never freezes and OpenSim's own
messages stay out of the way:

    python -m openkick.multiview.opensim_job <folder> <markers.trc> <height_m> [mass_kg]

Scales the skeleton model to the person, solves inverse kinematics (Pose2Sim) and "bakes" the
result: where every bone and every marker of the model is in every frame. Writes
<folder>/result.json."""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

ANGLES = ("knee_angle_r", "knee_angle_l", "hip_flexion_r", "hip_flexion_l", "ankle_angle_r", "ankle_angle_l")


def prepare(folder: Path) -> Path:
    """Empty the working folder of a previous run. Returns where the marker file must go."""
    folder = Path(folder)
    for sub in ("pose-3d", "kinematics"):
        if (folder / sub).is_dir():
            shutil.rmtree(folder / sub)
    for name in ("result.json", "model.npz"):
        (folder / name).unlink(missing_ok=True)
    (folder / "pose-3d").mkdir(parents=True, exist_ok=True)
    return folder / "pose-3d"


def geometry_dir() -> Path:
    import Pose2Sim
    return Path(Pose2Sim.__file__).parent / "OpenSim_Setup" / "Geometry"


def bake(osim_path: Path, mot_path: Path, out: Path) -> dict:
    """Position of every bone mesh in every frame, plus the joint angles, in one .npz file."""
    import opensim as osim
    osim.Logger.setLevelString("error")
    model = osim.Model(str(osim_path))
    state = model.initSystem()
    table = osim.TimeSeriesTable(str(mot_path))
    times = np.array(table.getIndependentColumn())
    labels = list(table.getColumnLabels())
    data = np.array([[table.getRowAtIndex(i)[j] for j in range(len(labels))] for i in range(len(times))])
    coords = {}
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        c = cs.get(i)
        coords[c.getName()] = (c, c.getMotionType() == osim.Coordinate.Rotational)
    gdir = geometry_dir()
    meshes = []
    for comp in model.getComponentsList():
        mesh = osim.Mesh.safeDownCast(comp)
        if mesh is None or not (gdir / mesh.get_mesh_file()).exists():
            continue
        s = mesh.get_scale_factors()
        meshes.append((mesh.get_mesh_file(), mesh.getFrame(), np.array([s.get(0), s.get(1), s.get(2)])))
    M = np.zeros((len(times), len(meshes), 4, 4), np.float32)
    M[:, :, 3, 3] = 1
    ms = model.getMarkerSet()
    marker_names = [ms.get(i).getName() for i in range(ms.getSize())]
    P = np.full((len(times), len(marker_names), 3), np.nan, np.float32)    # the model's markers: fixed on its bones
    js = model.getJointSet()
    joint_names = [js.get(i).getName() for i in range(js.getSize())]
    B = np.full((len(times), len(joint_names), 3), np.nan, np.float32)     # each joint's centre (its child frame)
    for f in range(len(times)):
        for name, v in zip(labels, data[f]):
            if name in coords:
                c, rot = coords[name]
                c.setValue(state, float(np.radians(v) if rot else v), False)
        model.realizePosition(state)
        for i in range(len(marker_names)):
            loc = ms.get(i).getLocationInGround(state)
            P[f, i] = (loc.get(0), loc.get(1), loc.get(2))
        for i in range(len(joint_names)):
            o = js.get(i).getChildFrame().getPositionInGround(state)
            B[f, i] = (o.get(0), o.get(1), o.get(2))
        for k, (_, frame, scale) in enumerate(meshes):
            T = frame.getTransformInGround(state)
            R, p = T.R(), T.p()
            for i in range(3):
                for j in range(3):
                    M[f, k, i, j] = R.get(i, j) * scale[j]
                M[f, k, i, 3] = p.get(i)
    np.savez_compressed(out, times=times, M=M, files=np.array([m[0] for m in meshes]), geometry=np.array(str(gdir)),
                        labels=np.array(labels), data=data.astype(np.float32), markers=P,
                        marker_names=np.array(marker_names), joints=B, joint_names=np.array(joint_names))
    return {"frames": len(times), "bones": len(meshes)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    folder, trc, height = Path(argv[0]), Path(argv[1]), float(argv[2])
    mass = float(argv[3]) if len(argv) > 3 else 70.0
    result = {"ok": False}
    try:
        from openkick.core.video import VideoMeta  # noqa: F401  (fail early if the package is not importable)
        from openkick.multiview.export import run_opensim
        out = run_opensim(folder, trc, height, mass)
        if not out["osim"] or not out["mot"]:
            raise RuntimeError("OpenSim did not produce a model")
        info = bake(out["osim"][0], out["mot"][0], folder / "model.npz")
        errors = sorted(out["dir"].glob("*_ik_marker_errors.sto"))
        rms = float("nan")
        if errors:
            lines = errors[0].read_text().splitlines()
            i = lines.index("endheader")
            cols = lines[i + 1].split("\t")
            a = np.array([ln.split("\t") for ln in lines[i + 2:] if ln.strip()], float)
            rms = float(np.mean(a[:, cols.index("marker_error_RMS")]) * 100)
        result = {"ok": True, "osim": str(out["osim"][0]), "mot": str(out["mot"][0]), "trc": str(trc),
                  "baked": str(folder / "model.npz"), "marker_rms_cm": rms, **info}
    except Exception:  # noqa: BLE001 - reported to the window
        result = {"ok": False, "error": traceback.format_exc()}
    (folder / "result.json").write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
