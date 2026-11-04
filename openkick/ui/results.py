"""Results page: a card per joint, the average phase, the whole video, the phase table and the template's analysis."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

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

        for k, ph in enumerate(proj.phases):      # alternate light bands: the phases' own colours mean nothing here
            c = QColor(pal["muted"])
            c.setAlpha(18 if k % 2 == 0 else 34)
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
            if len(labels) <= 2:                  # more labels than that land on each other
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


# ── what the page sums up ──────────────────────────────────────────────

JOINT_NAMES = {"hip": "Hip", "knee": "Knee", "ankle": "Ankle", "shoulder": "Shoulder", "elbow": "Elbow"}
SIDE_COLORS = {"left": theme.LEFT, "right": theme.RIGHT}
FLAG_SPREADS = 3.0        # a phase is pointed out when a value is this many usual spreads from the typical one
FLAG_MIN_DEG = 4.0        # ... and at least this many degrees away
FOLDED_ROWS = 6           # rows of the phase table shown before "show all"


def usable_phases(p) -> list:
    return [ph for ph in p.phases if ph.end - ph.start >= 2]


def angle_groups(p) -> list[tuple[str, dict]]:
    """The visible angles, the two sides of one joint together: [(title, {side or "": measurement})]."""
    out: list[tuple[str, dict]] = []
    where: dict = {}
    for m in p.measurements:
        if m.kind != "angle" or not m.visible:
            continue
        found = A.joint(m)
        if found and found[0] in JOINT_NAMES:
            name, side = found
            if name in where and side not in out[where[name]][1]:
                out[where[name]][1][side] = m
                continue
            where.setdefault(name, len(out))
            out.append((tr(JOINT_NAMES[name]), {side: m}))
        else:
            out.append((m.name, {"": m}))
    return out


def per_phase(p, m, phases) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest value of `m` in every phase (NaN where it is not seen)."""
    v = A.series(p, m).value
    lo, hi = [], []
    for ph in phases:
        seg = v[ph.start:ph.end + 1]
        ok = np.isfinite(seg)
        lo.append(float(seg[ok].min()) if ok.any() else np.nan)
        hi.append(float(seg[ok].max()) if ok.any() else np.nan)
    return np.array(lo), np.array(hi)


def cycles(values: np.ndarray, phases, n: int = 101) -> np.ndarray:
    """Each phase stretched to 0–100 % (rows: phases with the value seen in 80 % of its frames)."""
    out = []
    for ph in phases:
        seg = values[ph.start:ph.end + 1]
        ok = np.isfinite(seg)
        if len(seg) < 3 or ok.mean() < 0.8:
            continue
        x = np.linspace(0, 1, len(seg))
        out.append(np.interp(np.linspace(0, 1, n), x[ok], seg[ok]))
    return np.array(out) if out else np.zeros((0, n))


def flagged(p, phases) -> dict[int, list[str]]:
    """Phases whose lowest or highest value of an angle is far from the typical phase:
    {phase index: [what is off]}. Worth looking at before trusting the summary."""
    from openkick.core.cycling import typical
    out: dict[int, list[str]] = {}
    if len(phases) < 5:
        return out
    for m in p.measurements:
        if m.kind != "angle" or not m.visible:
            continue
        for vals, what in zip(per_phase(p, m, phases), (tr("min"), tr("max"))):
            mid, spread = typical(vals)
            if not np.isfinite(mid):
                continue
            limit = max(FLAG_SPREADS * spread, FLAG_MIN_DEG)
            for i, x in enumerate(vals):
                if np.isfinite(x) and abs(x - mid) > limit:
                    out.setdefault(i, []).append(f"{m.name} {what} {x - mid:+.0f}°")
    return out


def _title(text: str, sub: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 8, 0, 0)
    lay.setSpacing(2)
    h = QLabel(text)
    h.setFont(theme.font(17, 600))
    lay.addWidget(h)
    if sub:
        s = QLabel(sub)
        s.setWordWrap(True)
        s.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
        lay.addWidget(s)
    return w


class JointCard(QFrame):
    """One joint: lowest, highest and range, a row per side, and the difference between sides."""

    def __init__(self, title: str, rows: list[dict], parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        grid = QGridLayout(self)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        head = QLabel(title)
        head.setFont(theme.font(15, 600))
        grid.addWidget(head, 0, 0, 1, 2)
        for c, text in enumerate((tr("Min."), tr("Max."), tr("Range of motion")), start=2):
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(lbl, 0, c)
        for r, row in enumerate(rows, start=1):
            side = QLabel(f"<span style='color:{row['color']}'>●</span>&nbsp; {row['label']}")
            side.setStyleSheet(f"color: {theme.TEXT_2};")
            grid.addWidget(side, r, 0)
            note = QLabel(row.get("note", ""))
            note.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
            grid.addWidget(note, r, 1)
            for c, (v, s) in enumerate((row["min"], row["max"], row["rom"]), start=2):
                cell = QLabel(self._value(v, s, row.get("bold", True)))
                cell.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                cell.setTextFormat(Qt.RichText)
                grid.addWidget(cell, r, c)
        grid.setColumnStretch(1, 1)

    @staticmethod
    def _value(v, spread, bold: bool) -> str:
        if v is None or not np.isfinite(v):
            return f"<span style='color:{theme.DIM}'>—</span>"
        size = 17 if bold else 13
        text = f"<span style='font-size:{size}px; color:{theme.TEXT if bold else theme.TEXT_2}'>{v:+.0f}°</span>" \
            if not bold else f"<span style='font-size:{size}px; color:{theme.TEXT}'>{v:.0f}°</span>"
        if spread is not None and np.isfinite(spread):
            text += f"<span style='color:{theme.FAINT}; font-size:12px'>&nbsp;± {spread:.0f}</span>"
        return text


class CycleChart(QWidget):
    """One joint over the average phase (0–100 %): mean line and ± 1 SD band, one per side."""

    def __init__(self, title: str, curves: list[tuple[str, str, np.ndarray, np.ndarray]], parent=None):
        super().__init__(parent)
        self.title, self.curves = title, curves
        self.setMinimumHeight(230)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        p.fillRect(r, QColor(theme.CARD))
        p.setPen(QPen(QColor(theme.CARD_LINE), 1))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.font(14, 600))
        p.drawText(QRectF(r.left() + 16, r.top() + 10, r.width() - 32, 22), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setFont(theme.font(12))
        x = r.right() - 16
        for label, color, _, _ in reversed(self.curves):           # legend, right-aligned
            w = QFontMetricsF(p.font()).horizontalAdvance(label)
            p.setPen(QColor(theme.TEXT_2))
            p.drawText(QRectF(x - w, r.top() + 10, w, 22), Qt.AlignVCenter, label)
            p.setPen(QPen(QColor(color), 3, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(x - w - 22, r.top() + 21), QPointF(x - w - 8, r.top() + 21))
            x -= w + 40
        plot = QRectF(r.left() + 50, r.top() + 44, r.width() - 66, r.height() - 76)
        vals = np.concatenate([np.r_[m - s, m + s] for _, _, m, s in self.curves]) if self.curves else np.array([0, 1])
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        step = nice_step(max(hi - lo, 1.0), 4)
        lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
        if hi - lo < 1e-6:
            hi = lo + step

        def X(k):
            return plot.left() + k / 100 * plot.width()

        def Y(v):
            return plot.bottom() - (v - lo) / (hi - lo) * plot.height()
        p.setFont(theme.font(11))
        v = lo
        while v <= hi + 1e-9:
            p.setPen(QPen(QColor(theme.LINE), 1))
            p.drawLine(QPointF(plot.left(), Y(v)), QPointF(plot.right(), Y(v)))
            p.setPen(QColor(theme.DIM))
            p.drawText(QRectF(r.left(), Y(v) - 8, 42, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:g}°")
            v += step
        for k in (0, 25, 50, 75, 100):
            p.drawText(QRectF(X(k) - 30, plot.bottom() + 6, 60, 16), Qt.AlignCenter, f"{k} %")
        xs = np.linspace(0, 100, 101)
        for _, color, mean, sd in self.curves:
            band = QPainterPath()
            band.moveTo(X(xs[0]), Y(mean[0] + sd[0]))
            for k in range(1, len(xs)):
                band.lineTo(X(xs[k]), Y(mean[k] + sd[k]))
            for k in range(len(xs) - 1, -1, -1):
                band.lineTo(X(xs[k]), Y(mean[k] - sd[k]))
            band.closeSubpath()
            c = QColor(color)
            c.setAlpha(45)
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawPath(band)
            line = QPainterPath()
            line.moveTo(X(xs[0]), Y(mean[0]))
            for k in range(1, len(xs)):
                line.lineTo(X(xs[k]), Y(mean[k]))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(color), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(line)


class Fold(QWidget):
    """A titled block that opens and closes."""

    def __init__(self, title: str, content: QWidget, open_: bool = False, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.button = QToolButton()
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        self.button.setText(" " + title)
        self.button.setFont(theme.font(14, 600))
        self.button.setStyleSheet(f"QToolButton {{ color: {theme.TEXT_2}; padding: 4px 2px; }}")
        self.button.setChecked(open_)
        self.button.toggled.connect(self._toggle)
        lay.addWidget(self.button, 0, Qt.AlignLeft)
        self.content = content
        lay.addWidget(content)
        content.setVisible(open_)

    def _toggle(self, on: bool):
        self.button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.content.setVisible(on)


def _table_widget(columns: list[str], rows: list[list], row_h: int = 38) -> QTableWidget:
    tw = QTableWidget(len(rows), len(columns))
    tw.setHorizontalHeaderLabels(columns)
    tw.verticalHeader().hide()
    tw.setEditTriggers(QAbstractItemView.NoEditTriggers)
    tw.setSelectionMode(QAbstractItemView.NoSelection)
    tw.setAlternatingRowColors(True)
    tw.setShowGrid(False)
    tw.setFocusPolicy(Qt.NoFocus)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            it = QTableWidgetItem(str(v))
            if c > 0:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tw.setItem(r, c, it)
    hh = tw.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.Stretch)
    tw.setFixedHeight(hh.height() + row_h * len(rows) + 6)
    for r in range(len(rows)):
        tw.setRowHeight(r, row_h)
    return tw


def _clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            _clear(item.layout())


class ResultsPage(QWidget):
    """Summary first (one card per joint), then the average phase, the whole video, the phase
    table and what the template's analysis adds, its long tables folded."""
    exportPdf = Signal()
    exportCsv = Signal()
    exportKva = Signal()
    exportVideo = Signal()
    editPhases = Signal()

    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.show_all = False
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
        lay.setSpacing(16)

        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(6)
        self.title = QLabel()
        self.title.setFont(theme.display(26))
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

        self.summary_box = QVBoxLayout()           # 1. one card per joint
        self.summary_box.setSpacing(10)
        lay.addLayout(self.summary_box)
        self.cycle_box = QVBoxLayout()             # 2. the average phase
        self.cycle_box.setSpacing(10)
        lay.addLayout(self.cycle_box)

        lay.addWidget(_title(tr("Over the whole video")))           # 3. the clip, as recorded
        chart_card = QFrame()
        chart_card.setObjectName("Card")
        cl = QVBoxLayout(chart_card)
        cl.setContentsMargins(18, 12, 18, 10)
        crow = QHBoxLayout()
        self.legend = QHBoxLayout()
        self.legend.setSpacing(14)
        crow.addLayout(self.legend)
        crow.addStretch(1)
        self.mode = QComboBox()
        self.mode.addItem(tr("Joint angles"), "angle")
        self.mode.addItem(tr("Speeds"), "speed")
        self.mode.currentIndexChanged.connect(self._mode)
        crow.addWidget(self.mode)
        cl.addLayout(crow)
        self.chart = Chart(ctl)
        self.chart.setMinimumHeight(260)
        cl.addWidget(self.chart)
        lay.addWidget(chart_card)

        self.table_title = _title(tr("By phase"))                    # 4. phase by phase
        lay.addWidget(self.table_title)
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self.table)
        self.more = QPushButton()
        self.more.setFlat(True)
        self.more.setStyleSheet(f"QPushButton {{ color: {theme.MUTED}; border: 0; background: transparent; text-align: left; }}"
                                f"QPushButton:hover {{ color: {theme.TEXT}; }}")
        self.more.clicked.connect(self._toggle_rows)
        lay.addWidget(self.more, 0, Qt.AlignLeft)

        self.plugin_box = QVBoxLayout()                              # 5. the template's analysis
        self.plugin_box.setSpacing(12)
        lay.addLayout(self.plugin_box)

        lay.addStretch(1)
        ctl.changed.connect(self.refresh)
        ctl.projectLoaded.connect(self._loaded)
        ctl.frameChanged.connect(lambda _: self.chart.update())

    def _loaded(self):
        self.show_all = False
        self.refresh()

    def _mode(self):
        self.chart.mode = self.mode.currentData()
        self.refresh()

    def _toggle_rows(self):
        self.show_all = not self.show_all
        self._table()

    def refresh(self):
        p = self.ctl.project
        if not p or not self.isVisible():
            return
        t = templates.find(p.template)
        self.title.setText(p.title)
        v = p.video
        bits = [f"{v.duration:.2f} s", tr("{n} frames", n=v.frame_count)]
        phases = usable_phases(p)
        if phases:
            bits.append(tr("1 phase") if len(phases) == 1 else tr("{n} phases", n=len(phases)))
        if t:
            bits.append(tr("template “{name}”", name=t.label))
        if p.calibration:
            bits.append(tr("calibrated 1 m = {px} px", px=round(p.calibration.px_per_m)))
        else:
            bits.append(tr("not calibrated (speeds in px/s)"))
        self.meta.setText(" · ".join(bits))
        _clear(self.legend)
        for m, _, _ in self.chart.series():
            lbl = QLabel(f"<span style='color:{m.color}; font-size:16px;'>━</span>&nbsp;{m.name}")
            lbl.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 12px;")
            self.legend.addWidget(lbl)
        self.chart.update()
        self._summary()
        self._cycles()
        self._table()
        self._plugin_results()

    # ── 1. summary ──
    def _summary(self):
        from openkick.core.cycling import typical
        _clear(self.summary_box)
        p = self.ctl.project
        groups = angle_groups(p)
        if not groups:
            return
        phases = usable_phases(p)
        sub = (tr("Typical phase ± how much it changes from one phase to the next ({n} phases).", n=len(phases))
               if len(phases) >= 2 else tr("Over the whole clip. Mark phases to see the typical one and how much it varies."))
        self.summary_box.addWidget(_title(tr("Summary"), sub))
        near = A.facing_side(p) if p.pose.has_data else None
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for k, (title, sides) in enumerate(groups):
            rows, typ = [], {}
            for side in ("left", "right", ""):
                if side not in sides:
                    continue
                m = sides[side]
                if len(phases) >= 2:
                    lo, hi = per_phase(p, m, phases)
                    (a, sa), (b, sb), (c, sc) = typical(lo), typical(hi), typical(hi - lo)
                else:
                    s = A.stats(p, m)
                    a, b = (s.minimum, s.maximum) if s.minimum is not None else (np.nan, np.nan)
                    c, sa, sb, sc = b - a, None, None, None
                typ[side] = (a, b, c)
                rows.append({"label": tr("Left") if side == "left" else tr("Right") if side == "right" else m.name,
                             "color": SIDE_COLORS.get(side, m.color), "min": (a, sa), "max": (b, sb), "rom": (c, sc),
                             "note": tr("far from the camera") if side and near and side != near and len(sides) > 1 else ""})
            if "left" in typ and "right" in typ:
                (a1, b1, c1), (a2, b2, c2) = typ["left"], typ["right"]
                rows.append({"label": tr("Right − left"), "color": theme.DIM, "bold": False,
                             "min": (a2 - a1, None), "max": (b2 - b1, None), "rom": (c2 - c1, None)})
            grid.addWidget(JointCard(title, rows), k // 2, k % 2)
        self.summary_box.addLayout(grid)

    # ── 2. the average phase ──
    def _cycles(self):
        _clear(self.cycle_box)
        p = self.ctl.project
        phases = usable_phases(p)
        if len(phases) < 2:
            return
        charts = []
        for title, sides in angle_groups(p):
            curves = []
            for side in ("left", "right", ""):
                if side not in sides:
                    continue
                C = cycles(A.series(p, sides[side]).value, phases)
                if len(C) >= 2:
                    label = tr("Left") if side == "left" else tr("Right") if side == "right" else sides[side].name
                    curves.append((label, SIDE_COLORS.get(side, sides[side].color), C.mean(0), C.std(0)))
            if curves:
                charts.append(CycleChart(title, curves))
        if not charts:
            return
        self.cycle_box.addWidget(_title(tr("Average phase"),
                                        tr("Each phase stretched from 0 to 100 %. Line: mean of the phases; band: ± 1 standard deviation.")))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for k, c in enumerate(charts):
            grid.addWidget(c, k // 2, k % 2)
        self.cycle_box.addLayout(grid)

    # ── 4. phase by phase ──
    def _table(self):
        from openkick.core.cycling import typical
        p = self.ctl.project
        ms = [m for m in p.measurements if m.kind in ("angle", "trajectory") and m.visible]
        phases = usable_phases(p)
        cols = [tr("Phase"), tr("Interval"), tr("Duration")]
        for m in ms:
            cols.append(f"{m.name} " + (tr("min–max") if m.kind == "angle" else tr("peak speed")))
        t = A.times(p)
        if len(phases) >= 2:
            cols.append("")
        flags = flagged(p, phases)
        rows, marks, which = [], [], []          # marks: "typical", "flag" or ""; which: the phase of the row
        if len(phases) >= 2:
            row = [tr("Typical"), "", f"{typical(t[ph.end] - t[ph.start] for ph in phases)[0]:.2f} s"]
            for m in ms:
                if m.kind == "angle":
                    lo, hi = per_phase(p, m, phases)
                    row.append(f"{A.format_value(typical(lo)[0], '°')} – {A.format_value(typical(hi)[0], '°')}")
                else:
                    peaks = [A.stats(p, m, ph.start, ph.end).maximum for ph in phases]
                    row.append(A.format_value(typical(peaks)[0], A.series(p, m).unit))
            rows.append(row + [""])
            marks.append("typical")
            which.append(None)
            shown = range(len(phases))
            folded = len(phases) > FOLDED_ROWS + 2 and not self.show_all
            if folded:
                shown = sorted(set(list(range(FOLDED_ROWS)) + list(flags)))
            for i in shown:
                ph = phases[i]
                row = [ph.name, f"{t[ph.start]:.2f} – {t[ph.end]:.2f} s", f"{t[ph.end] - t[ph.start]:.2f} s"]
                for m in ms:
                    s = A.stats(p, m, ph.start, ph.end)
                    row.append(f"{A.format_value(s.minimum, '°')} – {A.format_value(s.maximum, '°')}" if m.kind == "angle"
                               else A.format_value(s.maximum, A.series(p, m).unit))
                rows.append(row + [tr("Review") if i in flags else ""])
                marks.append("flag" if i in flags else "")
                which.append(i)
            hidden = len(phases) - len(shown)
            self.more.setVisible(len(phases) > FOLDED_ROWS + 2)
            self.more.setText(tr("Show all {n} phases", n=len(phases)) if hidden else tr("Show fewer"))
        else:
            for name, a, b in [(ph.name, ph.start, ph.end) for ph in phases] + [(tr("Whole clip"), 0, p.frame_count - 1)]:
                row = [name, f"{t[a]:.2f} – {t[b]:.2f} s", f"{t[b] - t[a]:.2f} s"]
                for m in ms:
                    s = A.stats(p, m, a, b)
                    row.append(f"{A.format_value(s.minimum, '°')} – {A.format_value(s.maximum, '°')}" if m.kind == "angle"
                               else A.format_value(s.maximum, A.series(p, m).unit))
                rows.append(row)
                marks.append("")
                which.append(None)
            self.more.hide()
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(cols)
        bold = theme.font(13, 600)
        for r, (row, mark, i) in enumerate(zip(rows, marks, which)):
            for c, text in enumerate(row):
                it = QTableWidgetItem(text)
                if c > 0:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if mark == "typical":
                    it.setFont(bold)
                if mark == "flag" and c == len(row) - 1:
                    it.setForeground(QColor(theme.WARN_TEXT))
                    it.setToolTip("\n".join(flags.get(i, [])))
                self.table.setItem(r, c, it)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFixedHeight(hh.height() + 40 * len(rows) + 6)
        for r in range(len(rows)):
            self.table.setRowHeight(r, 40)

    # ── 5. the template's analysis ──
    def _plugin_results(self):
        _clear(self.plugin_box)
        p = self.ctl.project
        if not p:
            return
        res = self.ctl.plugin_results.get(p.template) if p.template else None
        if not res and p.template in ("cycling", "gait") and p.pose.has_data:
            res = self.ctl.run_template_plugin()
        if not res:
            return
        lines = (res.summary or "").split("\n")
        self.plugin_box.addWidget(_title(res.title or tr("Plugin results"), lines[0] if lines else ""))
        warn = [ln for ln in lines[1:] if ln.startswith("•") or ln.endswith(":")]
        if warn:
            w = QLabel("\n".join(warn))
            w.setWordWrap(True)
            w.setStyleSheet(f"color: {theme.WARN_TEXT}; font-size: 13px;")
            self.plugin_box.addWidget(w)
        for t in res.tables:
            if t.rows and not t.detail:
                self.plugin_box.addWidget(QLabel(f"<b>{t.title}</b>") if t.title else QWidget())
                self.plugin_box.addWidget(_table_widget(t.columns, t.rows))
        for t in res.tables:
            if t.rows and t.detail:
                self.plugin_box.addWidget(Fold(t.title, _table_widget(t.columns, t.rows)))
        notes = [ln for ln in lines[1:] if ln not in warn]
        if notes:
            n = QLabel("\n".join(notes))
            n.setWordWrap(True)
            n.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
            self.plugin_box.addWidget(Fold(tr("How it is measured"), n))

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()
