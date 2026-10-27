"""Peak linear speeds (trajectories) and angular velocities (angles)."""

import numpy as np

from openkick.i18n import tr
from openkick.plugins import AnalysisPlugin, Result, Table


class Velocities(AnalysisPlugin):
    id = "velocities"
    name = {"es": "Velocidades y aceleraciones", "en": "Velocities and accelerations"}
    description = {"es": "Velocidad lineal y angular máxima, por fase, con filtro Butterworth 6 Hz.",
                   "en": "Peak linear and angular velocity per phase, Butterworth 6 Hz filtered."}

    def run(self, ctx):
        p = ctx.project
        t = ctx.times()
        ranges = [(tr("Whole clip"), 0, p.frame_count - 1)] + [(ph.name, ph.start, ph.end) for ph in p.phases]
        tables = []
        tracks = [m for m in p.measurements if m.kind == "trajectory"]
        if tracks:
            rows = []
            for m in tracks:
                s = ctx.series(m)
                acc = np.gradient(s.value, t) if len(t) > 1 else s.value * np.nan
                for label, a, b in ranges:
                    v, i = _peak(s.value[a:b + 1])
                    ac, _ = _peak(np.abs(acc[a:b + 1]))
                    rows.append([m.name, label, ctx.fmt(v, s.unit), _at(t, a, i),
                                 "—" if ac is None else f"{ac:.1f} {s.unit}²"])
            if p.calibration is None:
                note = tr("Speeds are in pixels per second. Calibrate a known length to get m/s.")
            else:
                note = ""
            tables.append(Table([tr("Point"), tr("Range"), tr("Peak speed"), tr("At"), tr("Peak acceleration")],
                                rows, tr("Linear speed")))
        else:
            note = ""
        angles = [m for m in p.measurements if m.kind == "angle"]
        if angles:
            rows = []
            for m in angles:
                s = ctx.series(m)
                for label, a, b in ranges:
                    v, i = _peak(np.abs(s.rate[a:b + 1]))
                    rows.append([m.name, label, "—" if v is None else f"{v:.0f} °/s", _at(t, a, i)])
            tables.append(Table([tr("Angle"), tr("Range"), tr("Peak angular velocity"), tr("At")], rows,
                                tr("Angular velocity")))
        if not tables:
            return Result(summary=tr("Add angles or trajectories to compute velocities."))
        return Result(title=self.label, summary=note, tables=tables)


def _peak(v):
    ok = np.isfinite(v)
    if not ok.any():
        return None, None
    i = int(np.nanargmax(np.where(ok, v, -np.inf)))
    return float(v[i]), i


def _at(t, a, i):
    return "—" if i is None else f"{t[a + i]:.2f} s"


PLUGINS = [Velocities()]
