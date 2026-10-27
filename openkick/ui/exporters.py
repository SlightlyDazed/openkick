"""Exports that need Qt rendering: video with drawings and the PDF report."""

from __future__ import annotations

import html
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QMarginsF, QSizeF, QThread, QUrl, Signal
from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPainter, QPdfWriter, QTextDocument

from openkick import AUTHOR, __version__
from openkick.core import analysis as A
from openkick.core import templates
from openkick.core.project import Project
from openkick.core.video import iter_frames
from openkick.i18n import tr
from openkick.plugins import PluginContext
from openkick.ui import overlay


class VideoExportWorker(QThread):
    progress = Signal(int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, project: Project, path: str, layers: dict):
        super().__init__()
        self.project, self.path, self.layers = project, path, dict(layers)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._run()
            self.done.emit("" if self._stop else self.path)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))

    def _run(self):
        p = self.project
        meta = p.video
        w, h = meta.width - meta.width % 2, meta.height - meta.height % 2
        opts = overlay.Options(layers=self.layers, k=max(1.0, meta.width / 1100), show_markers=False)
        view = overlay.View(1.0, 0.0, 0.0)
        with av.open(self.path, "w") as out:
            codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
            stream = out.add_stream(codec, rate=Fraction(meta.fps).limit_denominator(100000))
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            stream.time_base = meta.time_base
            if codec == "libx264":
                stream.options = {"crf": "18", "preset": "medium"}
            else:
                stream.bit_rate = 8_000_000
            first = meta.first_pts
            last_pts = -1
            for i, rgb in iter_frames(meta):
                if self._stop:
                    break
                img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()
                painter = QPainter(img)
                overlay.paint(painter, p, i, view, opts)
                painter.end()
                img = img.convertToFormat(QImage.Format_RGB888)
                arr = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
                arr = arr[:, : img.width() * 3].reshape(img.height(), img.width(), 3)[:h, :w]
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
                pts = meta.pts[i] - first
                if pts <= last_pts:
                    pts = last_pts + 1
                last_pts = pts
                frame.pts = pts
                frame.time_base = meta.time_base
                for packet in stream.encode(frame):
                    out.mux(packet)
                self.progress.emit(i + 1, meta.frame_count)
            for packet in stream.encode():
                out.mux(packet)


# ── PDF report ─────────────────────────────────────────────────────────


CSS = """
body { font-family: 'IBM Plex Sans'; color: #1B1F24; font-size: 10pt; }
h1 { font-family: 'Space Grotesk'; font-size: 22pt; margin: 0 0 2pt 0; }
h2 { font-size: 12pt; margin: 16pt 0 6pt 0; }
.meta { color: #5B6474; }
table { border-collapse: collapse; width: 100%; }
th { background: #F1F3F6; color: #3D4550; text-align: left; padding: 5pt 7pt; font-weight: 600; }
td { padding: 5pt 7pt; border-bottom: 1px solid #E4E7EC; }
td.num { font-family: 'IBM Plex Mono'; }
.kpi { font-family: 'IBM Plex Mono'; font-size: 18pt; }
.foot { color: #8B94A3; font-size: 8pt; }
td.dot { color: #9A3412; font-weight: 600; }
"""


def _table(cols, rows, num_from=1) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f"<td class='{'num' if i >= num_from else ''}{' dot' if '●' in str(v) else ''}'>{html.escape(str(v))}</td>"
            for i, v in enumerate(r)) + "</tr>"
    return f"<table width='100%' cellspacing='0' cellpadding='4'><tr>{head}</tr>{body}</table>"   # Qt ignores CSS padding


def write_pdf(project: Project, chart, path: str):
    p = project
    t = templates.find(p.template)
    doc = QTextDocument()
    doc.setDefaultStyleSheet(CSS)
    parts = [f"<h1>{html.escape(p.title)}</h1>"]
    meta = [Path(p.video.path).name, f"{p.video.duration:.2f} s", tr("{n} frames", n=p.frame_count)]
    if t:
        meta.append(tr("template “{name}”", name=t.label))
    meta.append(tr("calibrated 1 m = {px} px", px=round(p.calibration.px_per_m)) if p.calibration
                else tr("not calibrated (speeds in px/s)"))
    parts.append(f"<p class='meta'>{html.escape(' · '.join(meta))}</p>")

    # the patient
    if not p.subject.empty:
        from openkick.core.subject import describe
        parts.append(f"<h2>{html.escape(tr('Patient'))}</h2>" + _table([tr("Item"), tr("Value")], describe(p.subject, tr), 2))

    # key figures
    kpis = []
    for m in p.measurements:
        s = A.stats(p, m)
        unit = A.series(p, m).unit
        if m.kind == "angle":
            kpis.append((tr("ROM · {name}", name=m.name), A.format_value(s.range, "°"),
                         f"{A.format_value(s.minimum, '°')} – {A.format_value(s.maximum, '°')}"))
        elif m.kind == "trajectory":
            kpis.append((tr("Peak speed · {name}", name=m.name), A.format_value(s.maximum, unit),
                         tr("at {t} s", t=f"{s.t_max:.2f}") if s.t_max is not None else ""))
    if kpis:
        cells = "".join(f"<td><div class='meta'>{html.escape(a)}</div><div class='kpi'>{html.escape(b)}</div>"
                        f"<div class='meta'>{html.escape(c)}</div></td>" for a, b, c in kpis[:4])
        parts.append(f"<table cellspacing='0'><tr>{cells}</tr></table>")

    # charts
    for mode, title in (("angle", tr("Joint angles")), ("speed", tr("Speeds"))):
        old = chart.mode
        chart.mode = mode
        if chart.series():
            img = chart.render_image(900, 340, light=True)
            doc.addResource(QTextDocument.ImageResource, QUrl(f"chart_{mode}.png"), img)
            parts.append(f"<h2>{html.escape(title)}</h2><img src='chart_{mode}.png' width='640'>")
        chart.mode = old

    # phases table
    times = A.times(p)
    ms = [m for m in p.measurements if m.kind in ("angle", "trajectory")]
    ranges = [(ph.name, ph.start, ph.end) for ph in p.phases] + [(tr("Whole clip"), 0, p.frame_count - 1)]
    cols = [tr("Phase"), tr("Interval"), tr("Duration")] + \
           [f"{m.name} " + (tr("min–max") if m.kind == "angle" else tr("peak speed")) for m in ms]
    rows = []
    for name, a, b in ranges:
        row = [name, f"{times[a]:.2f} – {times[b]:.2f} s", f"{times[b] - times[a]:.2f} s"]
        for m in ms:
            s = A.stats(p, m, a, b)
            row.append(f"{A.format_value(s.minimum, '°')} – {A.format_value(s.maximum, '°')}" if m.kind == "angle"
                       else A.format_value(s.maximum, A.series(p, m).unit))
        rows.append(row)
    parts.append(f"<h2>{html.escape(tr('Phases'))}</h2>" + _table(cols, rows))

    # built-in plugin tables (ROM + velocities)
    from openkick.plugins.builtin import rom_report, velocities
    ctx = PluginContext(p)
    for plugin in (rom_report.PLUGINS[0], velocities.PLUGINS[0]):
        res = plugin.run(ctx)
        for tb in res.tables:
            parts.append(f"<h2>{html.escape(plugin.label)} · {html.escape(tb.title)}</h2>" + _table(tb.columns, tb.rows))
    if p.template in ("cycling", "gait"):
        import importlib
        plugin = importlib.import_module(f"openkick.plugins.builtin.{p.template}").PLUGINS[0]
        c_res = plugin.run(ctx)
        if c_res.summary:
            parts.append(f"<p>{html.escape(c_res.summary)}</p>")
        for tb in c_res.tables:
            parts.append(f"<h2>{html.escape(tb.title or plugin.label)}</h2>" + _table(tb.columns, tb.rows))
    parts.append(f"<p class='foot'>{html.escape(tr('Generated with OpenKick {v} · openkick project', v=__version__))}"
                 f" · {html.escape(tr('Developed by {name}', name=AUTHOR))}</p>")
    doc.setHtml("<body>" + "".join(parts) + "</body>")
    print_pdf(doc, path, p.title)


def print_pdf(doc: QTextDocument, path: str, title: str):
    """An A4 PDF of a report written as HTML."""
    writer = QPdfWriter(path)
    writer.setPageSize(QPageSize(QPageSize.A4))
    writer.setPageMargins(QMarginsF(16, 16, 16, 16), QPageLayout.Millimeter)
    writer.setResolution(144)
    writer.setTitle(title)
    writer.setCreator(f"OpenKick {__version__} — {AUTHOR}")
    doc.setPageSize(QSizeF(writer.width(), writer.height()))
    doc.print_(writer)
