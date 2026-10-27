"""Range of motion (ROM) report: per angle, for the whole clip and for each phase."""

from openkick.i18n import tr
from openkick.plugins import AnalysisPlugin, Result, Table


class RomReport(AnalysisPlugin):
    id = "rom_report"
    name = {"es": "Informe de rango de movimiento", "en": "Range of motion report"}
    description = {"es": "Tabla de ROM por articulación y fase, lista para el informe.",
                   "en": "ROM table per joint and phase, ready for your report."}

    def run(self, ctx):
        angles = [m for m in ctx.project.measurements if m.kind == "angle"]
        if not angles:
            return Result(summary=tr("There are no angles in this project. Add one with the angle tool."))
        cols = [tr("Angle"), tr("Min"), tr("Max"), tr("ROM"), tr("Mean"), tr("Min at"), tr("Max at")]
        rows = []
        for m in angles:
            s = ctx.stats(m)
            rows.append([m.name, ctx.fmt(s.minimum, "°"), ctx.fmt(s.maximum, "°"), ctx.fmt(s.range, "°"),
                         ctx.fmt(s.mean, "°"), _t(s.t_min), _t(s.t_max)])
        tables = [Table(cols, rows, tr("Whole clip"))]
        if ctx.project.phases:
            cols = [tr("Phase")] + [f"{m.name} · ROM" for m in angles]
            rows = []
            for ph in ctx.project.phases:
                row = [ph.name]
                for m in angles:
                    s = ctx.stats(m, ph)
                    row.append(f"{ctx.fmt(s.range, '°')}  ({ctx.fmt(s.minimum, '°')}–{ctx.fmt(s.maximum, '°')})")
                rows.append(row)
            tables.append(Table(cols, rows, tr("By phase")))
        return Result(title=self.label, tables=tables)


def _t(v):
    return "—" if v is None else f"{v:.2f} s"


PLUGINS = [RomReport()]
