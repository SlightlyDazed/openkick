"""Export the 33 MediaPipe markers as an OpenSim .trc file (meters, Y up)."""

import numpy as np

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.i18n import tr
from openkick.plugins import ExportPlugin

TRC_NAMES = [
    "Nose", "LEyeInner", "LEye", "LEyeOuter", "REyeInner", "REye", "REyeOuter",
    "LEar", "REar", "MouthL", "MouthR", "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LPinky", "RPinky", "LIndex", "RIndex", "LThumb", "RThumb",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LHeel", "RHeel",
    "LFootIndex", "RFootIndex",
]


class OpenSimTrc(ExportPlugin):
    id = "opensim_trc"
    name = {"es": "Exportar a OpenSim (.trc)", "en": "Export to OpenSim (.trc)"}
    description = {"es": "Genera un .trc con los 33 marcadores, filtrado a 6 Hz, para escalar y hacer cinemática inversa.",
                   "en": "Writes a .trc with the 33 markers, 6 Hz filtered, for scaling and inverse kinematics."}
    file_filter = "OpenSim TRC (*.trc)"
    suffix = ".trc"

    def export(self, ctx, path):
        p = ctx.project
        v = p.video
        if p.calibration:
            px_per_m = p.calibration.px_per_m
        else:
            h = ctx.ask_number(tr("Athlete height (m), used to convert pixels to meters"), 1.75)
            if not h:
                return ""
            px_per_m = _scale_from_height(p, h)
        xy = p.pose.xy.astype(np.float64)
        vis = p.pose.visibility
        z = p.pose.z.astype(np.float64)
        n = p.frame_count
        mid_hip = np.nanmedian(xy[:, [L.INDEX["left_hip"], L.INDEX["right_hip"]], :].reshape(-1, 2), axis=0)
        t = A.times(p)
        data = np.full((n, L.COUNT, 3), np.nan)
        for j in range(L.COUNT):
            x = (xy[:, j, 0] - mid_hip[0]) / px_per_m
            y = -(xy[:, j, 1] - mid_hip[1]) / px_per_m
            zz = -z[:, j] * v.width / px_per_m      # MediaPipe z shares the x scale
            low = vis[:, j] < 0.3
            for c, arr in enumerate((x, y, zz)):
                arr = arr.copy()
                arr[low] = np.nan
                data[:, j, c] = A.lowpass(arr, v.fps)
        with open(path, "w", newline="") as f:
            f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
            f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
            f.write(f"{v.fps:.6f}\t{v.fps:.6f}\t{n}\t{L.COUNT}\tm\t{v.fps:.6f}\t1\t{n}\n")
            f.write("Frame#\tTime\t" + "\t\t\t".join(TRC_NAMES) + "\n")
            f.write("\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, L.COUNT + 1)) + "\n\n")
            for i in range(n):
                vals = "\t".join("" if not np.isfinite(c) else f"{c:.6f}" for c in data[i].ravel())
                f.write(f"{i + 1}\t{t[i]:.6f}\t{vals}\n")
        return tr("OpenSim file written. The depth (Z) comes from a single camera and is approximate.")


def _scale_from_height(p, height_m):
    """px per meter from nose-to-ankle distance in the most upright frames (~94% of height)."""
    xy = p.pose.xy
    nose = xy[:, L.INDEX["nose"], 1]
    ankle = np.nanmean(xy[:, [L.INDEX["left_ankle"], L.INDEX["right_ankle"]], 1], axis=1)
    h = np.abs(ankle - nose)
    h = h[np.isfinite(h)]
    if not len(h):
        return 500.0
    tall = h[h >= np.percentile(h, 75)]
    return float(np.median(tall)) / (height_m * 0.94)


PLUGINS = [OpenSimTrc()]
