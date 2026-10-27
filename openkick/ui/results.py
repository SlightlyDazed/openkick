"""Results page: phase band, angle/speed chart, key figures and per-phase table."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from openkick.core import analysis as A
from openkick.core import templates
from openkick.i18n import tr
from openkick.ui import icons, theme
from openkick.ui.controller import Controller

LIGHT = {"bg": "#FFFFFF", "grid": "#E4E7EC", "text": "#1B1F24", "muted": "#5B6474", "phase_alpha": 55}
DARK = {"bg": theme.PANEL, "grid": theme.LINE, "text": theme.TEXT, "muted": theme.DIM, "phase_alpha": 70}


def nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


class Chart(QWidget):
    """Series over time with the phases shaded behind."""

    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.mode = "angle"
        self.hover_x: float | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(320)

    def series(self):
        p = self.ctl.project
        if not p:
            return []
        out = []
        for m in p.measurements:
            if not m.visible:
                continue
            if self.mode == "angle" and m.kind == "angle":
                out.append((m, A.series(p, m).value, "°"))
            elif self.mode == "speed" and m.kind == "trajectory":
                s = A.series(p, m)
                out.append((m, s.value, s.unit))
        return out

    def paintEvent(self, _):
        p = QPainter(self)
        self.render_to(p, QRectF(self.rect()), DARK, interactive=True)

    def render_image(self, w: int, h: int, light: bool = True) -> QImage:
        img = QImage(w * 2, h * 2, QImage.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(2)
        img.fill(QColor(LIGHT["bg"] if light else DARK["bg"]))
        p = QPainter(img)
        self.render_to(p, QRectF(0, 0, w, h), LIGHT if light else DARK, interactive=False)
        p.end()
        return img

    def render_to(self, p: QPainter, rect: QRectF, pal: dict, interactive: bool):
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(rect, QColor(pal["bg"]))
        proj = self.ctl.project
        if not proj:
            return
        data = self.series()
        t = A.times(proj)
        dur = proj.video.duration or 1
        plot = QRectF(rect.left() + 56, rect.top() + 14, rect.width() - 72, rect.height() - 44)
        if data:
            allv = np.concatenate([v[np.isfinite(v)] for _, v, _ in data if np.isfinite(v).any()] or [np.array([0, 1])])
            lo, hi = float(allv.min()), float(allv.max())
        else:
            lo, hi = 0.0, 180.0
        if self.mode == "angle":
            lo, hi = max(0.0, math.floor(lo / 10) * 10 - 10), min(180.0, math.ceil(hi / 10) * 10 + 10)
        else:
            lo, hi = 0.0, hi * 1.1 if hi > 0 else 1.0
        if hi - lo < 1e-6:
            hi = lo + 1

        def X(time):
            return plot.left() + time / dur * plot.width()

        def Y(val):
            return plot.bottom() - (val - lo) / (hi - lo) * plot.height()

        for ph in proj.phases:
            c = QColor(ph.color)
            c.setAlpha(pal["phase_alpha"])
            x0 = X(t[ph.start])
            x1 = X(t[ph.end + 1]) if ph.end + 1 < len(t) else plot.right()
            p.fillRect(QRectF(x0, plot.top(), x1 - x0, plot.height()), c)
        p.setFont(theme.mono(11))
        step = nice_step(hi - lo, 5)
        v = math.ceil(lo / step) * step
        unit = "°" if self.mode == "angle" else (data[0][2] if data else "")
        while v <= hi + 1e-9:
            y = Y(v)
            p.setPen(QPen(QColor(pal["grid"]), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(pal["muted"]))
            lbl = f"{v:.0f}{unit}" if self.mode == "angle" else f"{v:g}"
            p.drawText(QRectF(rect.left(), y - 8, 50, 16), Qt.AlignRight | Qt.AlignVCenter, lbl)
            v += step
        ts = nice_step(dur, 6)
        tt = 0.0
        while tt <= dur + 1e-9:
            x = X(tt)
            p.setPen(QColor(pal["muted"]))
            p.drawText(QRectF(x - 30, plot.bottom() + 6, 60, 16), Qt.AlignCenter, f"{tt:g} s" if tt == 0 else f"{tt:g}")
            tt += ts
        labels = []
        for m, vals, _unit in data:
            ok = np.isfinite(vals)
            path = QPainterPath()
            started = False
            for i in range(len(vals)):
                if not ok[i]:
                    started = False
                    continue
                q = QPointF(X(t[i]), Y(vals[i]))
                if started:
                    path.lineTo(q)
                else:
                    path.moveTo(q)
                    started = True
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(m.color), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)
            if ok.any():
                i = int(np.nanargmin(vals)) if self.mode == "angle" else int(np.nanargmax(vals))
                c = QPointF(X(t[i]), Y(vals[i]))
                labels.append((m, c, vals, i, _unit))
        for m, c, vals, i, _unit in labels:
            if True:
                p.setPen(QPen(QColor(pal["bg"]), 2))
                p.setBrush(QColor(m.color))
                p.drawEllipse(c, 5, 5)
                label = (tr("min {v} · {t} s", v=A.format_value(float(vals[i]), "°"), t=f"{t[i]:.2f}")
                         if self.mode == "angle" else
                         tr("max {v} · {t} s", v=A.format_value(float(vals[i]), _unit), t=f"{t[i]:.2f}"))
                f = theme.mono(11, 600)
                p.setFont(f)
                fm = QFontMetricsF(f)
                w = fm.horizontalAdvance(label) + 14
                r = QRectF(c.x() + 9, c.y() + (6 if self.mode == "angle" else -26), w, 20)
                if r.right() > plot.right():
                    r.moveRight(c.x() - 9)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(m.color))
                p.drawRoundedRect(r, 5, 5)
                p.setPen(QColor(theme.text_on(m.color)))
                p.drawText(r, Qt.AlignCenter, label)
        if interactive:
            x = X(t[self.ctl.frame])
            p.setPen(QPen(QColor(theme.ACCENT), 1.5, Qt.DashLine))
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        if not data:
            p.setPen(QColor(pal["muted"]))
            p.setFont(theme.font(13))
            msg = tr("Add angles to see them here") if self.mode == "angle" else tr("Add trajectories to see their speed")
            p.drawText(plot, Qt.AlignCenter, msg)
        self._plot = plot

    def mousePressEvent(self, e):
        proj = self.ctl.project
        if not proj or not hasattr(self, "_plot"):
            return
        t = (e.position().x() - self._plot.left()) / self._plot.width() * proj.video.duration
        self.ctl.set_frame(proj.video.index_at_time(t))
        self.update()


class Tile(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet(f"QFrame#Card {{ background: {theme.PANEL}; border: 1px solid #1E232C; border-radius: 14px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(6)
        self.label = QLabel()
        self.label.setStyleSheet(f"color: {theme.MUTED};")
        self.value = QLabel()
        self.value.setFont(theme.mono(32, 500))
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setStyleSheet(f"color: {theme.DIM}; font-size: 12px;")
        for w in (self.label, self.value, self.sub):
            lay.addWidget(w)

    def set(self, label, value, sub, color=theme.TEXT):
        self.label.setText(label)
        self.value.setText(value)
        self.value.setStyleSheet(f"color: {color};")
        self.sub.setText(sub)


class PhaseBand(QWidget):
    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setFixedHeight(52)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        proj = self.ctl.project
        if not proj:
            return
        if not proj.phases:
            p.setPen(QPen(QColor(theme.BORDER), 1, Qt.DashLine))
            p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 8, 8)
            p.setPen(QColor(theme.DIM))
            p.drawText(self.rect(), Qt.AlignCenter, tr("Mark phases in the timeline to compare them here"))
            return
        t = A.times(proj)
        dur = proj.video.duration or 1
        w = self.width()
        for i, ph in enumerate(proj.phases):
            x0 = t[ph.start] / dur * w
            x1 = (t[ph.end + 1] if ph.end + 1 < len(t) else dur) / dur * w
            r = QRectF(x0 + 1.5, 0, max(4, x1 - x0 - 3), self.height())
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ph.color))
            p.drawRoundedRect(r, 8, 8)
            fg = QColor(theme.text_on(ph.color))
            p.setPen(fg)
            full_lbl = ph.name if (ph.name.startswith(f"{i + 1} ") or ph.name.lower().startswith("pedalada") or ph.name.lower().startswith("stroke")) else f"{i + 1} · {ph.name}"
            if r.width() < 60:
                p.setFont(theme.font(12, 600))
                p.drawText(r, Qt.AlignCenter, f"P{i + 1}")
            else:
                p.setFont(theme.font(13, 600))
                fm = QFontMetricsF(p.font())
                p.drawText(r.adjusted(8, 8, -6, -26), Qt.AlignLeft | Qt.AlignVCenter,
                           fm.elidedText(full_lbl, Qt.ElideRight, int(r.width() - 14)))
                p.setFont(theme.mono(11))
                fg.setAlpha(200)
                p.setPen(fg)
                p.drawText(r.adjusted(8, 26, -6, -6), Qt.AlignLeft | Qt.AlignVCenter,
                           f"{t[ph.start]:.2f} – {t[ph.end]:.2f} s")


class ResultsPage(QWidget):
    exportPdf = Signal()
    exportCsv = Signal()
    exportKva = Signal()
    exportVideo = Signal()
    editPhases = Signal()

    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setObjectName("Page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName("Page")
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.setSpacing(22)

        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(6)
        self.title = QLabel()
        self.title.setFont(theme.display(30))
        col.addWidget(self.title)
        self.meta = QLabel()
        self.meta.setStyleSheet(f"color: {theme.FAINT};")
        col.addWidget(self.meta)
        head.addLayout(col, 1)
        lbl = QLabel(tr("Export as"))
        lbl.setStyleSheet(f"color: {theme.MUTED};")
        head.addWidget(lbl)
        for name, text, sig in (("pdf", tr("PDF report"), self.exportPdf), ("table", tr("CSV data"), self.exportCsv),
                                ("export", tr("Kinovea (.kva)"), self.exportKva), ("video", tr("Video with drawings"), self.exportVideo)):
            b = QPushButton(icons.icon(name, theme.TEXT, 16), " " + text)
            b.clicked.connect(sig)
            head.addWidget(b)
        lay.addLayout(head)

        self.band = PhaseBand(ctl)
        lay.addWidget(self.band)

        mid = QHBoxLayout()
        mid.setSpacing(20)
        chart_card = QFrame()
        chart_card.setObjectName("Card")
        chart_card.setStyleSheet(f"QFrame#Card {{ background: {theme.PANEL}; border: 1px solid #1E232C; border-radius: 14px; }}")
        cl = QVBoxLayout(chart_card)
        cl.setContentsMargins(18, 14, 18, 10)
        crow = QHBoxLayout()
        self.chart_title = QLabel()
        self.chart_title.setStyleSheet("font-weight: 600;")
        crow.addWidget(self.chart_title)
        self.legend = QHBoxLayout()
        self.legend.setSpacing(14)
        crow.addSpacing(12)
        crow.addLayout(self.legend)
        crow.addStretch(1)
        self.mode = QComboBox()
        self.mode.addItem(tr("Joint angles"), "angle")
        self.mode.addItem(tr("Speeds"), "speed")
        self.mode.currentIndexChanged.connect(self._mode)
        crow.addWidget(self.mode)
        cl.addLayout(crow)
        self.chart = Chart(ctl)
        cl.addWidget(self.chart)
        mid.addWidget(chart_card, 3)
        tiles = QVBoxLayout()
        tiles.setSpacing(14)
        self.tiles = [Tile(), Tile(), Tile()]
        for t in self.tiles:
            t.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            tiles.addWidget(t)
        tw = QWidget()
        tw.setLayout(tiles)
        tw.setMinimumWidth(300)
        mid.addWidget(tw, 1)
        lay.addLayout(mid)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self.table)

        self.plugin_box = QVBoxLayout()
        self.plugin_box.setSpacing(16)
        lay.addLayout(self.plugin_box)

        lay.addStretch(1)
        ctl.changed.connect(self.refresh)
        ctl.projectLoaded.connect(self.refresh)
        ctl.frameChanged.connect(lambda _: self.chart.update())

    def _mode(self):
        self.chart.mode = self.mode.currentData()
        self.refresh()

    def refresh(self):
        p = self.ctl.project
        if not p or not self.isVisible():
            return
        t = templates.find(p.template)
        self.title.setText(p.title)
        v = p.video
        bits = [f"{v.duration:.2f} s", tr("{n} frames", n=v.frame_count)]
        if t:
            bits.append(tr("template “{name}”", name=t.label))
        if p.calibration:
            bits.append(tr("calibrated 1 m = {px} px", px=round(p.calibration.px_per_m)))
        else:
            bits.append(tr("not calibrated (speeds in px/s)"))
        self.meta.setText(" · ".join(bits))
        self.chart_title.setText(tr("Joint angles") if self.chart.mode == "angle" else tr("Speeds"))
        while self.legend.count():
            item = self.legend.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        for m, _, _ in self.chart.series():
            lbl = QLabel(f"<span style='color:{m.color}; font-size:16px;'>━</span>&nbsp;{m.name}")
            lbl.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 12px;")
            self.legend.addWidget(lbl)
        self.band.update()
        self.chart.update()
        self._tiles()
        self._table()
        self._plugin_results()

    def _tiles(self):
        p = self.ctl.project
        angles = [m for m in p.measurements if m.kind == "angle"]
        tracks = [m for m in p.measurements if m.kind == "trajectory"]
        if angles:
            m = angles[0]
            s = A.stats(p, m)
            self.tiles[0].set(tr("ROM · {name}", name=m.name), A.format_value(s.range, "°"),
                              tr("from {a} ({ta} s) to {b} ({tb} s)", a=A.format_value(s.minimum, "°"),
                                 ta=f"{s.t_min:.2f}" if s.t_min is not None else "—", b=A.format_value(s.maximum, "°"),
                                 tb=f"{s.t_max:.2f}" if s.t_max is not None else "—"), m.color)
        else:
            self.tiles[0].set(tr("Range of motion"), "—", tr("Add an angle to compute it"))
        if tracks:
            m = tracks[0]
            s = A.stats(p, m)
            unit = A.series(p, m).unit
            ph = None
            if s.t_max is not None:
                ph = p.phase_at(p.video.index_at_time(s.t_max))
            sub = tr("at {t} s", t=f"{s.t_max:.2f}") if s.t_max is not None else ""
            if ph:
                sub += " · " + ph.name
            self.tiles[1].set(tr("Peak speed · {name}", name=m.name), A.format_value(s.maximum, unit), sub, m.color)
        else:
            self.tiles[1].set(tr("Peak speed"), "—", tr("Add a trajectory to compute it"))
        dur = p.video.duration
        self.tiles[2].set(tr("Duration"), f"{dur:.2f} s",
                          (tr("1 phase") if len(p.phases) == 1 else tr("{n} phases", n=len(p.phases)))
                          if p.phases else tr("No phases marked"))

    def _table(self):
        p = self.ctl.project
        ms = [m for m in p.measurements if m.kind in ("angle", "trajectory")]
        cols = [tr("Phase"), tr("Interval"), tr("Duration")]
        for m in ms:
            cols.append(f"{m.name} " + (tr("min–max") if m.kind == "angle" else tr("peak speed")))
        ranges = [(ph.name, ph.start, ph.end, ph.color) for ph in p.phases] + \
                 [(tr("Whole clip"), 0, p.frame_count - 1, None)]
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(ranges))
        self.table.setHorizontalHeaderLabels(cols)
        t = A.times(p)
        for r, (name, a, b, color) in enumerate(ranges):
            items = [name, f"{t[a]:.2f} – {t[b]:.2f} s", f"{t[b] - t[a]:.2f} s"]
            for m in ms:
                s = A.stats(p, m, a, b)
                if m.kind == "angle":
                    items.append(f"{A.format_value(s.minimum, '°')} – {A.format_value(s.maximum, '°')}")
                else:
                    items.append(A.format_value(s.maximum, A.series(p, m).unit))
            for c, text in enumerate(items):
                it = QTableWidgetItem(("■  " if (c == 0 and color) else "") + text)
                if c == 0 and color:
                    it.setForeground(QColor(theme.TEXT))
                    it.setData(Qt.DecorationRole, QColor(color))
                    it.setText(text)
                if c > 0:
                    it.setFont(theme.mono(13))
                self.table.setItem(r, c, it)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(self.table.horizontalHeader().height() + 42 * len(ranges) + 6)
        for r in range(len(ranges)):
            self.table.setRowHeight(r, 42)

    def _plugin_results(self):
        while self.plugin_box.count():
            item = self.plugin_box.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().hide()
                        sub.widget().deleteLater()

        p = self.ctl.project
        if not p:
            return

        res = self.ctl.plugin_results.get(p.template) if p.template else None
        if not res and p.template in ("cycling", "gait") and p.pose.has_data:
            res = self.ctl.run_template_plugin()

        if not res:
            return

        h_row = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(icons.pixmap("chart", theme.ACCENT, 20))
        h_row.addWidget(icon_lbl)
        h_title = QLabel(res.title or tr("Plugin results"))
        h_title.setFont(theme.display(20))
        h_row.addWidget(h_title)
        h_row.addStretch(1)
        self.plugin_box.addLayout(h_row)

        if res.summary:
            sum_card = QFrame()
            sum_card.setObjectName("Card")
            sum_card.setStyleSheet(f"QFrame#Card {{ background: {theme.PANEL}; border: 1px solid #1E232C; border-radius: 14px; padding: 14px; }}")
            s_lay = QVBoxLayout(sum_card)
            s_lay.setContentsMargins(16, 14, 16, 14)
            s_lay.setSpacing(8)
            s_lbl = QLabel(res.summary)
            s_lbl.setWordWrap(True)
            s_lbl.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 13px;")
            s_lay.addWidget(s_lbl)
            self.plugin_box.addWidget(sum_card)

        for t in res.tables:
            if not t.rows:
                continue
            if t.title:
                tl = QLabel(t.title)
                tl.setFont(theme.font(15, 600))
                self.plugin_box.addWidget(tl)
            tw = QTableWidget(len(t.rows), len(t.columns))
            tw.setHorizontalHeaderLabels(t.columns)
            tw.verticalHeader().hide()
            tw.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tw.setSelectionMode(QAbstractItemView.NoSelection)
            tw.setAlternatingRowColors(True)
            tw.setShowGrid(False)
            tw.setFocusPolicy(Qt.NoFocus)
            for r, row in enumerate(t.rows):
                for c, v in enumerate(row):
                    it = QTableWidgetItem(str(v))
                    if c > 0:
                        it.setFont(theme.mono(12))
                    tw.setItem(r, c, it)
            hh = tw.horizontalHeader()
            hh.setSectionResizeMode(QHeaderView.Stretch)
            tw.setFixedHeight(hh.height() + 38 * len(t.rows) + 6)
            for r in range(len(t.rows)):
                tw.setRowHeight(r, 38)
            self.plugin_box.addWidget(tw)

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()
