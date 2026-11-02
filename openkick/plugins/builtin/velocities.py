"""Peak linear speeds (trajectories) and angular velocities (angles)."""

import numpy as np

from openkick.i18n import tr
from openkick.plugins import AnalysisPlugin, Result, Table


class Velocities(AnalysisPlugin):
    id = "velocities"
    name = {"es": "Velocidades y aceleraciones", "en": "Velocities and accelerations"}
    description = {"es": "Velocidad lineal y angular máxima, por fase, con filtro Butterworth (6 Hz, o el de la plantilla).",
                   "en": "Peak linear and angular velocity per phase, Butterworth filtered (6 Hz, or the template's)."}

    def run(self, ctx):
        p = ctx.project
        t = ctx.times()
        whole = (tr("Whole clip"), 0, p.frame_count - 1)
        if p.template == "gait" and p.phases:
            # walking into and out of the picture is not walking: only the strides found
            whole = (tr("All strides"), min(ph.start for ph in p.phases), max(ph.end for ph in p.phases))
        ranges = [whole] + [(ph.name, ph.start, ph.end) for ph in p.phases]
        tables = []
        tracks = [m for m in p.measurements if m.kind == "trajectory"]
        if tracks:
            rows = []
            for m in tracks:
                s = ctx.series(m)
                acc = s.accel if s.accel is not None else s.value * np.nan
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
        notes = [note] if note else []
        from openkick.core.analysis import cutoff
        from openkick.core.templates import find
        notes.append(tr("Positions and angles are smoothed below {hz} Hz before differentiating.",
                        hz=f"{cutoff(p):.0f}"))
        t_ = find(p.template)
        if t_ is not None and t_.min_fps and p.video.fps < t_.min_fps - 1:
            notes.append(tr("This movement is too fast for {fps} frames per second: its peak speeds come out "
                            "low. Record at {need} frames per second or more (120 is better).",
                            fps=f"{p.video.fps:.0f}", need=f"{t_.min_fps:.0f}"))
        return Result(title=self.label, summary="\n".join(notes), tables=tables)


def _peak(v):
    ok = np.isfinite(v)
    if not ok.any():
        return None, None
    i = int(np.nanargmax(np.where(ok, v, -np.inf)))
    return float(v[i]), i


def _at(t, a, i):
    return "—" if i is None else f"{t[a + i]:.2f} s"


PLUGINS = [Velocities()]
