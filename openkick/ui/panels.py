"""Right-hand panel: measurements, phases and the selected-point inspector."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QButtonGroup, QColorDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QMenu, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core import templates
from openkick.core.project import SRC_MANUAL, SRC_SMOOTHED, Measurement, PointRef
from openkick.i18n import tr
from openkick.ui import icons, theme
from openkick.ui.canvas import delete_measure, rename_measure, revert_joint, timecode
from openkick.ui.controller import Controller


def section(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("SectionTitle")
    return lbl


def dot(color: str, size: int = 10) -> QLabel:
    d = QLabel()
    d.setFixedSize(size, size)
    d.setStyleSheet(f"background: {color}; border-radius: {size // 2}px;")
    return d


def clear(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w:
            w.hide()
            w.deleteLater()
        elif item.layout():
            clear(item.layout())


class MeasureCard(QFrame):
    clicked = Signal(str)

    def __init__(self, ctl: Controller, m: Measurement, parent=None):
        super().__init__(parent)
        self.ctl, self.mid = ctl, m.id
        self.setObjectName("Card")
        self.setCursor(Qt.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 10, 10)
        lay.setSpacing(12)
        lay.addWidget(dot(m.color))
        col = QVBoxLayout()
        col.setSpacing(2)
        name = QLabel(m.name)
        name.setStyleSheet("font-weight: 500;")
        name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        col.addWidget(name)
        attached = all(r.follows_body or (r.kind == "marker" and ctl.project.measurement(r.marker)
                                          and ctl.project.measurement(r.marker).points[0].follows_body)
                       for r in m.points)
        kind = {"angle": tr("Angle"), "trajectory": tr("Trajectory"), "distance": tr("Distance"),
                "point": tr("Marker")}[m.kind]
        if m.kind == "point" and m.points[0].kind == "derived":
            where = tr("calculated")
        elif m.kind == "point" and m.points[0].kind == "segment":
            from openkick.core.bodymodel import SEGMENTS
            where = tr(SEGMENTS.get(m.points[0].segment[:2], "on the skeleton"))
        else:
            where = tr("on the skeleton") if attached else tr("free points")
        sub = QLabel(f"{kind} · {where}")
        sub.setObjectName("Faint")
        sub.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sub.setToolTip(tr("Attached to skeleton: correcting a joint updates this measurement") if attached else
                       tr("Free points: drag them in each frame where they need adjusting"))
        col.addWidget(sub)
        lay.addLayout(col, 1)
        self.value = QLabel("—")
        self.value.setFont(theme.mono(20 if m.kind != "point" else 12, 500))
        self.value.setStyleSheet(f"color: {m.color};")
        lay.addWidget(self.value)
        eye = QToolButton()
        eye.setIcon(icons.icon("eye" if m.visible else "eye_off", theme.FAINT, 16))
        eye.setToolTip(tr("Show / hide"))
        eye.clicked.connect(self._toggle)
        lay.addWidget(eye)
        self.set_selected(ctl.selected_measure == m.id)

    def set_selected(self, on: bool):
        self.setObjectName("CardSelected" if on else "Card")
        self.style().unpolish(self)
        self.style().polish(self)

    def refresh(self):
        m = self.ctl.project.measurement(self.mid)
        if not m:
            return
        s = A.series(self.ctl.project, m)
        if m.kind == "point":
            x, y = s.xy[self.ctl.frame]
            self.value.setText("—" if not np.isfinite(x) else f"x {x:.0f}\ny {y:.0f} px")
            return
        v = s.value[self.ctl.frame]
        self.value.setText(A.format_value(None if not np.isfinite(v) else float(v), s.unit))

    def _toggle(self):
        mid = self.mid
        self.ctl.change(tr("Show / hide"), lambda: setattr(self.ctl.project.measurement(mid), "visible",
                                                           not self.ctl.project.measurement(mid).visible))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.ctl.select_measure(self.mid)
        elif e.button() == Qt.RightButton:
            self._menu(e.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, e):
        m = self.ctl.project.measurement(self.mid)
        if m:
            rename_measure(self, self.ctl, m)

    def _menu(self, gpos):
        m = self.ctl.project.measurement(self.mid)
        menu = QMenu(self)
        menu.addAction(tr("Rename…"), lambda: rename_measure(self, self.ctl, m))
        menu.addAction(tr("Color…"), self._color)
        if m.kind == "trajectory":
            sub = menu.addMenu(tr("Opacity"))
            for pct in (25, 50, 75, 100):
                a = sub.addAction(f"{pct} %")
                a.setCheckable(True)
                a.setChecked(abs(m.opacity - pct / 100) < 0.05)
                a.triggered.connect(lambda _, v=pct: self._set_opacity(v / 100))
        menu.addAction(tr("Delete"), lambda: delete_measure(self.ctl, m))
        menu.exec(gpos)

    def _set_opacity(self, value: float):
        mid = self.mid
        self.ctl.change(tr("Opacity"), lambda: setattr(self.ctl.project.measurement(mid), "opacity", value))

    def _color(self):
        m = self.ctl.project.measurement(self.mid)
        c = QColorDialog.getColor(QColor(m.color), self, tr("Color"))
        if c.isValid():
            mid = self.mid
            self.ctl.change(tr("Color"), lambda: setattr(self.ctl.project.measurement(mid), "color", c.name().upper()))


class Sparkline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = np.array([])
        self.marks: list[int] = []
        self.frame = 0
        self.setFixedHeight(36)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        v = self.values
        if not len(v):
            return
        w, h = self.width(), self.height()
        path = QPainterPath()
        started = False
        for i, val in enumerate(v):
            x = i / max(len(v) - 1, 1) * w
            if not np.isfinite(val):
                started = False
                continue
            y = 3 + (1 - val) * (h - 6)
            if started:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                started = True
        p.setPen(QPen(QColor(theme.OK), 1.8))
        p.drawPath(path)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.WARN))
        for i in self.marks:
            val = v[i]
            if np.isfinite(val):
                p.drawEllipse(QPointF(i / max(len(v) - 1, 1) * w, 3 + (1 - val) * (h - 6)), 3, 3)
        x = self.frame / max(len(v) - 1, 1) * w
        p.setPen(QPen(QColor(theme.ACCENT), 1))
        p.drawLine(QPointF(x, 0), QPointF(x, h))


class MeasurementsTab(QWidget):
    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.cards: list[MeasureCard] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(16, 16, 16, 16)
        self.lay.setSpacing(20)

        head = QHBoxLayout()
        head.addWidget(section(tr("In this frame")))
        head.addStretch(1)
        self.add_btn = QPushButton(icons.icon("plus", theme.TEXT, 14), " " + tr("Add"))
        self.add_btn.setStyleSheet(f"QPushButton {{ min-height: 28px; padding: 0 10px; background: {theme.RAISED}; border: 0; }}")
        self.add_btn.clicked.connect(self._add_menu)
        head.addWidget(self.add_btn)
        box = QVBoxLayout()
        box.setSpacing(10)
        box.addLayout(head)
        self.cards_box = QVBoxLayout()
        self.cards_box.setSpacing(8)
        box.addLayout(self.cards_box)
        self.lay.addLayout(box)

        box = QVBoxLayout()
        box.setSpacing(10)
        box.addWidget(section(tr("Current phase")))
        self.phase_card = QFrame()
        self.phase_card.setObjectName("Card")
        self.phase_lay = QVBoxLayout(self.phase_card)
        self.phase_lay.setContentsMargins(14, 14, 14, 14)
        self.phase_lay.setSpacing(10)
        box.addWidget(self.phase_card)
        self.lay.addLayout(box)

        box = QVBoxLayout()
        box.setSpacing(10)
        box.addWidget(section(tr("Detection quality")))
        q = QFrame()
        q.setObjectName("Card")
        ql = QVBoxLayout(q)
        ql.setContentsMargins(14, 14, 14, 14)
        ql.setSpacing(10)
        self.spark = Sparkline()
        ql.addWidget(self.spark)
        row = QHBoxLayout()
        self.quality_lbl = QLabel()
        self.quality_lbl.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 12px;")
        row.addWidget(self.quality_lbl, 1)
        self.review_btn = QPushButton(tr("Review"))
        self.review_btn.setProperty("warn", True)
        self.review_btn.setToolTip(tr("Jump to the next doubtful frame (Tab)"))
        self.review_btn.clicked.connect(self.next_doubtful)
        row.addWidget(self.review_btn)
        ql.addLayout(row)
        box.addWidget(q)
        self.lay.addLayout(box)

        self.lay.addStretch(1)
        pill = QFrame()
        pill.setStyleSheet(f"QFrame {{ border: 1px dashed {theme.BORDER}; border-radius: 10px; }}"
                           "QLabel { border: 0; }")
        pl = QHBoxLayout(pill)
        pl.setContentsMargins(12, 6, 6, 6)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("template", theme.ACCENT, 16))
        pl.addWidget(ic)
        self.tpl_lbl = QLabel()
        self.tpl_lbl.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        pl.addWidget(self.tpl_lbl, 1)
        change = QPushButton(tr("Change"))
        change.setProperty("ghost", True)
        change.clicked.connect(self._template_menu)
        pl.addWidget(change)
        self.lay.addWidget(pill)

        ctl.projectLoaded.connect(self.rebuild)
        ctl.changed.connect(self.rebuild)
        ctl.frameChanged.connect(self.refresh)
        ctl.selectionChanged.connect(self._selection)

    def rebuild(self):
        clear(self.cards_box)
        self.cards = []
        p = self.ctl.project
        if not p:
            return
        for m in p.measurements:
            c = MeasureCard(self.ctl, m)
            self.cards.append(c)
            self.cards_box.addWidget(c)
        if not p.measurements:
            hint = QLabel(tr("No measurements yet. Pick the Angle tool (A) and click a knee, hip, elbow, "
                             "shoulder or ankle, or the Trajectory tool (T) and click any joint."))
            hint.setWordWrap(True)
            hint.setObjectName("Faint")
            self.cards_box.addWidget(hint)
        t = templates.find(p.template)
        self.tpl_lbl.setText(tr("Template: {name}", name=t.label if t else tr("none")))
        self._phase_key = None
        self.refresh()

    def _selection(self):
        for c in self.cards:
            c.set_selected(c.mid == self.ctl.selected_measure)

    def refresh(self, *_):
        p = self.ctl.project
        if not p:
            return
        for c in self.cards:
            c.refresh()
        ph = p.phase_at(self.ctl.frame)
        sph = p.subphase_at(self.ctl.frame) if hasattr(p, "subphase_at") else None
        key = (ph.id if ph else None, sph.id if sph else None, p.revision)
        if key != getattr(self, "_phase_key", None):
            self._phase_key = key
            self._phase_card(p, ph)
            self._quality(p)
        self.spark.frame = self.ctl.frame
        self.spark.update()

    def _phase_card(self, p, ph):
        clear(self.phase_lay)
        if not p.phases:
            lbl = QLabel(tr("No phases yet."))
            lbl.setObjectName("Faint")
            lbl.setWordWrap(True)
            self.phase_lay.addWidget(lbl)
            return

        if len(p.phases) <= 4:
            phases_row = QHBoxLayout()
            phases_row.setSpacing(6)
            for ph_i in p.phases:
                b = QPushButton(ph_i.name)
                is_cur = ph and ph.id == ph_i.id
                b.setStyleSheet(
                    f"QPushButton {{ background: {ph_i.color if is_cur else theme.CARD}; "
                    f"color: {theme.text_on(ph_i.color) if is_cur else theme.TEXT_2}; "
                    f"border: 1px solid {ph_i.color}; border-radius: 9px; padding: 2px 8px; font-size: 11px; font-weight: 500; }}"
                    f"QPushButton:hover {{ border-color: {theme.ACCENT}; }}"
                )
                b.setToolTip(f"{p.video.time_of(ph_i.start):.2f} – {p.video.time_of(ph_i.end):.2f} s · {tr('Click to jump')}")
                b.clicked.connect(lambda _=False, f=ph_i.start: self.ctl.set_frame(f))
                phases_row.addWidget(b)
            phases_row.addStretch(1)
            self.phase_lay.addLayout(phases_row)
        else:
            nav_row = QHBoxLayout()
            nav_row.setSpacing(6)
            cur_idx = next((i for i, q in enumerate(p.phases) if ph and q.id == ph.id), -1)
            prev_btn = QToolButton()
            prev_btn.setIcon(icons.icon("prev", theme.TEXT_2, 14))
            prev_btn.setToolTip(tr("Previous phase"))
            prev_idx = max(0, cur_idx - 1) if cur_idx > 0 else len(p.phases) - 1
            prev_btn.clicked.connect(lambda _=False, idx=prev_idx: self.ctl.set_frame(p.phases[idx].start))
            nav_row.addWidget(prev_btn)

            title_lbl = QLabel(f"{ph.name if ph else tr('Outside phases')} ({cur_idx + 1 if cur_idx >= 0 else '—'}/{len(p.phases)})")
            title_lbl.setFont(theme.font(12, 600))
            title_lbl.setStyleSheet(f"color: {ph.color if ph else theme.MUTED};")
            nav_row.addWidget(title_lbl, 1, Qt.AlignCenter)

            next_btn = QToolButton()
            next_btn.setIcon(icons.icon("next", theme.TEXT_2, 14))
            next_btn.setToolTip(tr("Next phase"))
            next_idx = min(len(p.phases) - 1, cur_idx + 1) if cur_idx >= 0 and cur_idx < len(p.phases) - 1 else 0
            next_btn.clicked.connect(lambda _=False, idx=next_idx: self.ctl.set_frame(p.phases[idx].start))
            nav_row.addWidget(next_btn)
            self.phase_lay.addLayout(nav_row)

        if getattr(p, "subphases", None):
            sph = p.subphase_at(self.ctl.frame)
            stroke_subs = [s for s in p.subphases if ph and ph.start <= s.start and s.end <= ph.end]
            if stroke_subs:
                sub_row = QHBoxLayout()
                sub_row.setSpacing(6)
                from openkick.core.gait import short_name
                many = len(stroke_subs) > 3              # the eight phases of a stride: short names, all in one row
                if many:
                    sub_row.setSpacing(3)
                for s in stroke_subs:
                    span = f"{p.video.time_of(s.start):.2f}–{p.video.time_of(s.end):.2f} s"
                    sb = QPushButton(short_name(s.name) if many else f"{s.name} ({span})")
                    sb.setToolTip(f"{s.name} · {span}")
                    is_sub_cur = sph and sph.id == s.id
                    sb.setStyleSheet(
                        f"QPushButton {{ background: {s.color if is_sub_cur else theme.CARD}; "
                        f"color: {theme.text_on(s.color) if is_sub_cur else theme.TEXT_2}; "
                        f"border: 1px solid {s.color}; border-radius: 6px; padding: {'2px 1px' if many else '2px 8px'}; "
                        f"font-size: 11px; font-weight: 500; }}"
                        f"QPushButton:hover {{ border-color: {theme.ACCENT}; }}"
                    )
                    sb.clicked.connect(lambda _=False, f=s.start: self.ctl.set_frame(f))
                    sub_row.addWidget(sb)
                self.phase_lay.addLayout(sub_row)
                if many and sph:
                    now = QLabel(f"{sph.name} · {p.video.time_of(sph.start):.2f}–{p.video.time_of(sph.end):.2f} s")
                    now.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 12px;")
                    self.phase_lay.addWidget(now)

        if not ph:
            lbl = QLabel(tr("This frame is not inside any phase."))
            lbl.setObjectName("Faint")
            lbl.setWordWrap(True)
            self.phase_lay.addWidget(lbl)
        else:
            head = QHBoxLayout()
            chip = QLabel(ph.name)
            chip.setStyleSheet(f"background: {ph.color}; color: {theme.text_on(ph.color)}; border-radius: 11px;"
                               "padding: 3px 10px; font-size: 12px; font-weight: 600;")
            head.addWidget(chip)
            head.addStretch(1)
            v = p.video
            rng = QLabel(f"{v.time_of(ph.start):.2f} → {v.time_of(ph.end):.2f} s · "
                         f"{v.time_of(ph.end) - v.time_of(ph.start):.2f} s")
            rng.setFont(theme.mono(12))
            rng.setStyleSheet(f"color: {theme.MUTED};")
            head.addWidget(rng)
            self.phase_lay.addLayout(head)
            rows = 0
            for m in p.measurements:
                s = A.phase_stats(p, m, ph)
                if m.kind == "angle":
                    k, val = tr("ROM {name}", name=m.name.lower()), A.format_value(s.range, "°")
                elif m.kind == "trajectory":
                    k, val = tr("Peak speed {name}", name=m.name.lower()), A.format_value(s.maximum, A.series(p, m).unit)
                else:
                    continue
                row = QHBoxLayout()
                a = QLabel(k)
                a.setStyleSheet(f"color: {theme.MUTED};")
                a.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                b = QLabel(val)
                b.setFont(theme.mono(13, 500))
                row.addWidget(a, 1)
                row.addWidget(b)
                self.phase_lay.addLayout(row)
                rows += 1
            if not rows:
                lbl = QLabel(tr("Add angles or trajectories to see phase values."))
                lbl.setObjectName("Faint")
                self.phase_lay.addWidget(lbl)

        for pid, emoji in (("cycling", "🚲"), ("gait", "🚶")):
            c_res = self.ctl.plugin_results.get(pid)
            if c_res and c_res.summary:
                first_line = c_res.summary.split("\n")[0]
                badge = QLabel(f"<span style='color:{theme.ACCENT}'>{emoji}</span> {first_line}")
                badge.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 11px; padding: 4px 6px; background: {theme.STAGE}; border-radius: 6px;")
                badge.setWordWrap(True)
                self.phase_lay.addWidget(badge)

    def _quality(self, p):
        conf = A.frame_confidence(p)
        low = self.ctl.low_confidence_frames()
        self.spark.values = conf
        self.spark.marks = low
        if not p.pose.has_data:
            self.quality_lbl.setText(tr("No detection yet"))
            self.review_btn.hide()
        elif low:
            self.quality_lbl.setText(tr("1 frame with low confidence") if len(low) == 1
                                     else tr("{n} frames with low confidence", n=len(low)))
            self.review_btn.show()
        else:
            self.quality_lbl.setText(tr("All frames look reliable"))
            self.review_btn.hide()

    def next_doubtful(self):
        low = self.ctl.low_confidence_frames()
        if not low:
            return
        nxt = next((f for f in low if f > self.ctl.frame), low[0])
        self.ctl.set_frame(nxt)

    def _add_menu(self):
        if not self.ctl.project:
            return
        menu = QMenu(self)
        am = menu.addMenu(icons.icon("angle", theme.TEXT, 16), tr("Joint angle"))
        for j, (a, o, b) in L.ANGLE_PRESETS.items():
            am.addAction(tr(L.DISPLAY[o]), lambda a=a, o=o, b=b: self._add("angle", [a, o, b], tr(L.DISPLAY[o])))
        tm = menu.addMenu(icons.icon("track", theme.TEXT, 16), tr("Trajectory of a joint"))
        for j in L.JOINTS:
            n = L.NAMES[j]
            tm.addAction(tr(L.DISPLAY[n]), lambda n=n: self._add("trajectory", [n], tr(L.DISPLAY[n])))
        mm = menu.addMenu(icons.icon("target", theme.TEXT, 16), tr("Marker / point"))
        mm.addAction(icons.icon("com", theme.TEXT, 16), tr("Center of mass (male model)"),
                     lambda: self._add_point(PointRef(derived="com:m"), tr("Center of mass")))
        mm.addAction(icons.icon("com", theme.TEXT, 16), tr("Center of mass (female model)"),
                     lambda: self._add_point(PointRef(derived="com:f"), tr("Center of mass")))
        mm.addSeparator()
        mm.addAction(tr("Midpoint of the hips"), lambda: self._add_point(
            PointRef(segment=(L.INDEX["right_hip"], L.INDEX["left_hip"], 0.5, 0.0)), tr("Mid-hip")))
        mm.addAction(tr("Midpoint of the shoulders"), lambda: self._add_point(
            PointRef(segment=(L.INDEX["right_shoulder"], L.INDEX["left_shoulder"], 0.5, 0.0)), tr("Mid-shoulder")))
        mm.addSeparator()
        mm.addAction(tr("Place a marker on the video (tool P)"), lambda: self.ctl.set_tool("marker"))
        menu.addSeparator()
        menu.addAction(icons.icon("line", theme.TEXT, 16), tr("Distance (tool L)"), lambda: self.ctl.set_tool("distance"))
        menu.exec(self.add_btn.mapToGlobal(self.add_btn.rect().bottomLeft()))

    def _add(self, kind, names, label):
        p = self.ctl.project
        m = Measurement(kind=kind, name=label, color=p.next_color(), points=[PointRef.at(L.INDEX[n]) for n in names])
        self.ctl.change(tr("Add measurement"), lambda: p.measurements.append(m))

    def _add_point(self, ref, name):
        p = self.ctl.project
        m = Measurement(kind="point", name=name, color=p.next_color(), points=[ref])
        self.ctl.change(tr("Add marker"), lambda: p.measurements.append(m))
        self.ctl.select_measure(m.id)

    def _template_menu(self):
        if not self.ctl.project:
            return
        menu = QMenu(self)
        for t in templates.list_templates():
            menu.addAction(icons.icon("template", t.color, 16), t.label,
                           lambda t=t: self.ctl.change(tr("Apply template"),
                                                       lambda: (templates.apply(self.ctl.project, t), self.ctl.run_template_plugin())))
        menu.exec(QCursor.pos())


class PhasesTab(QWidget):
    markPhase = Signal()

    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)
        outer.addWidget(section(tr("Phases of the movement")))
        info = QLabel(tr("A phase starts at the frame where you mark it and lasts until the next one. "
                         "Drag the edges in the timeline to adjust them; double-click to rename."))
        info.setWordWrap(True)
        info.setObjectName("Faint")
        outer.addWidget(info)
        self.list = QVBoxLayout()
        self.list.setSpacing(8)
        outer.addLayout(self.list)
        btn = QPushButton(icons.icon("flag", theme.ON_ACCENT, 16), " " + tr("Mark phase here"))
        btn.setProperty("primary", True)
        btn.clicked.connect(self.markPhase)
        outer.addWidget(btn)
        outer.addStretch(1)
        self._cur = None
        ctl.projectLoaded.connect(self.rebuild)
        ctl.changed.connect(self.rebuild)
        ctl.frameChanged.connect(self._frame)

    def _frame(self, _):
        p = self.ctl.project
        cur = p.phase_at(self.ctl.frame) if p else None
        if (cur.id if cur else None) != self._cur:
            self.rebuild()

    def rebuild(self):
        from openkick.ui.timeline import delete_phase, rename_phase
        clear(self.list)
        p = self.ctl.project
        if not p:
            return
        cur = p.phase_at(self.ctl.frame)
        self._cur = cur.id if cur else None
        for i, ph in enumerate(p.phases):
            card = QFrame()
            card.setObjectName("CardSelected" if ph is cur else "Card")
            lay = QHBoxLayout(card)
            lay.setContentsMargins(12, 8, 6, 8)
            lay.setSpacing(10)
            sw = QLabel()
            sw.setFixedSize(12, 12)
            sw.setStyleSheet(f"background: {ph.color}; border-radius: 3px;")
            lay.addWidget(sw)
            col = QVBoxLayout()
            col.setSpacing(1)
            name = QLabel(f"{i + 1} · {ph.name}")
            name.setStyleSheet("font-weight: 500;")
            col.addWidget(name)
            v = p.video
            t = QLabel(f"{v.time_of(ph.start):.2f} – {v.time_of(ph.end):.2f} s · "
                       f"{tr('{n} frames', n=ph.end - ph.start + 1)}")
            t.setFont(theme.mono(11))
            t.setStyleSheet(f"color: {theme.FAINT};")
            col.addWidget(t)
            lay.addLayout(col, 1)
            go = QToolButton()
            go.setIcon(icons.icon("prev", theme.FAINT, 16))
            go.setToolTip(tr("Go to start"))
            go.clicked.connect(lambda _=False, f=ph.start: self.ctl.set_frame(f))
            lay.addWidget(go)
            ren = QToolButton()
            ren.setIcon(icons.icon("text", theme.FAINT, 16))
            ren.setToolTip(tr("Rename…"))
            ren.clicked.connect(lambda _=False, ph=ph: rename_phase(self, self.ctl, ph))
            lay.addWidget(ren)
            dl = QToolButton()
            dl.setIcon(icons.icon("trash", theme.FAINT, 16))
            dl.setToolTip(tr("Delete phase"))
            dl.clicked.connect(lambda _=False, ph=ph: delete_phase(self.ctl, ph))
            lay.addWidget(dl)
            self.list.addWidget(card)
        if not p.phases:
            lbl = QLabel(tr("No phases yet."))
            lbl.setObjectName("Faint")
            self.list.addWidget(lbl)


class PointTab(QWidget):
    SCOPES = [("frame", "Only this frame", "The rest stays as MediaPipe detected it"),
              ("neighbors", "Smooth with neighbors", "Spreads the change over ±3 frames"),
              ("phase", "Until the end of the phase", "Shifts the joint over the whole current phase")]

    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 18, 16, 16)
        outer.setSpacing(20)
        box = QVBoxLayout()
        box.setSpacing(12)
        box.addWidget(section(tr("Selected point")))
        self.empty = QLabel(tr("Click a joint on the video to inspect it. Drag it to correct MediaPipe; "
                               "the angles and trajectories that use it update by themselves."))
        self.empty.setWordWrap(True)
        self.empty.setObjectName("Faint")
        box.addWidget(self.empty)
        self.detail = QWidget()
        dl = QVBoxLayout(self.detail)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(12)
        head = QHBoxLayout()
        self.dot = dot(theme.RIGHT, 12)
        head.addWidget(self.dot)
        self.name = QLabel()
        self.name.setStyleSheet("font-size: 17px; font-weight: 600;")
        head.addWidget(self.name, 1)
        dl.addLayout(head)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.x_lbl, self.y_lbl = QLabel(), QLabel()
        for i, (cap, lbl) in enumerate((("X", self.x_lbl), ("Y", self.y_lbl))):
            tile = QFrame()
            tile.setObjectName("Card")
            tl = QVBoxLayout(tile)
            tl.setContentsMargins(12, 8, 12, 8)
            tl.setSpacing(0)
            c = QLabel(cap)
            c.setObjectName("Faint")
            tl.addWidget(c)
            lbl.setFont(theme.mono(15))
            tl.addWidget(lbl)
            grid.addWidget(tile, 0, i)
        dl.addLayout(grid)
        crow = QHBoxLayout()
        cl = QLabel(tr("MediaPipe confidence"))
        cl.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        crow.addWidget(cl, 1)
        self.conf_lbl = QLabel()
        self.conf_lbl.setFont(theme.mono(12))
        crow.addWidget(self.conf_lbl)
        dl.addLayout(crow)
        self.conf_bar = QProgressBar()
        self.conf_bar.setFixedHeight(6)
        self.conf_bar.setRange(0, 100)
        dl.addWidget(self.conf_bar)
        self.src_lbl = QLabel()
        self.src_lbl.setWordWrap(True)
        dl.addWidget(self.src_lbl)
        brow = QHBoxLayout()
        self.revert_btn = QPushButton(tr("Revert to MediaPipe"))
        self.revert_btn.clicked.connect(lambda: revert_joint(self.ctl, self.ctl.selected_landmark))
        brow.addWidget(self.revert_btn)
        dl.addLayout(brow)
        box.addWidget(self.detail)
        outer.addLayout(box)

        box = QVBoxLayout()
        box.setSpacing(8)
        box.addWidget(section(tr("Apply corrections to")))
        self.group = QButtonGroup(self)
        for i, (key, title, sub) in enumerate(self.SCOPES):
            rb = QRadioButton(tr(title))
            rb.setStyleSheet("font-weight: 500;")
            rb.setChecked(key == ctl.edit_scope)
            self.group.addButton(rb, i)
            box.addWidget(rb)
            s = QLabel(tr(sub))
            s.setObjectName("Faint")
            s.setContentsMargins(26, 0, 0, 6)
            box.addWidget(s)
        self.group.idClicked.connect(lambda i: setattr(ctl, "edit_scope", self.SCOPES[i][0]))
        outer.addLayout(box)

        box = QVBoxLayout()
        box.setSpacing(8)
        box.addWidget(section(tr("Updates by itself")))
        self.uses = QVBoxLayout()
        self.uses.setSpacing(6)
        box.addLayout(self.uses)
        outer.addLayout(box)
        outer.addStretch(1)
        tip = QLabel(tr("← → change frame · Tab jumps to the next doubtful frame · Esc cancels a tool"))
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.DIM}; font-size: 12px;")
        outer.addWidget(tip)
        ctl.selectionChanged.connect(self.refresh)
        ctl.frameChanged.connect(self.refresh)
        ctl.changed.connect(self.refresh)
        self.refresh()

    def set_scope(self, key: str):
        for i, (k, *_rest) in enumerate(self.SCOPES):
            if k == key:
                self.group.button(i).setChecked(True)

    def refresh(self, *_):
        p = self.ctl.project
        j = self.ctl.selected_landmark
        has = p is not None and j is not None
        self.empty.setVisible(not has)
        self.detail.setVisible(has)
        clear(self.uses)
        if not has:
            return
        f = self.ctl.frame
        self.name.setText(tr(L.DISPLAY[L.NAMES[j]]))
        color = {"R": theme.RIGHT, "L": theme.LEFT}.get(L.side(j), theme.CENTER)
        self.dot.setStyleSheet(f"background: {color}; border-radius: 6px;")
        x, y = p.pose.xy[f, j]
        self.x_lbl.setText("—" if np.isnan(x) else f"{x:.1f} px")
        self.y_lbl.setText("—" if np.isnan(y) else f"{y:.1f} px")
        c = float(p.pose.visibility[f, j])
        self.conf_lbl.setText(f"{round(c * 100)} %")
        low = c < 0.5
        self.conf_lbl.setStyleSheet(f"color: {theme.WARN_TEXT if low else theme.TEXT};")
        self.conf_bar.setValue(round(c * 100))
        self.conf_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {theme.WARN if low else theme.OK}; border-radius: 3px; }}")
        src = p.pose.source[f, j]
        if src == SRC_MANUAL:
            self.src_lbl.setText(tr("Corrected by hand · protected"))
            self.src_lbl.setStyleSheet("background: #1A2410; color: #D9F99D; border-radius: 8px; padding: 9px 12px;")
        elif src == SRC_SMOOTHED:
            self.src_lbl.setText(tr("Adjusted by a nearby correction or smoothing"))
            self.src_lbl.setStyleSheet(f"background: {theme.RAISED}; color: {theme.TEXT_2}; border-radius: 8px; padding: 9px 12px;")
        else:
            self.src_lbl.setText(tr("As detected by MediaPipe"))
            self.src_lbl.setStyleSheet(f"background: {theme.RAISED}; color: {theme.TEXT_2}; border-radius: 8px; padding: 9px 12px;")
        self.revert_btn.setEnabled(bool(src != 1))
        for m in p.measurements:
            if any(r.landmark == j for r in m.points):
                row = QHBoxLayout()
                row.addWidget(dot(m.color, 8))
                n = QLabel(m.name)
                n.setStyleSheet(f"color: {theme.TEXT_2};")
                row.addWidget(n, 1)
                s = A.series(p, m)
                v = s.value[f]
                val = QLabel(A.format_value(None if not np.isfinite(v) else float(v), s.unit))
                val.setFont(theme.mono(13, 500))
                row.addWidget(val)
                self.uses.addLayout(row)
        if not self.uses.count():
            lbl = QLabel(tr("No measurement uses this joint yet."))
            lbl.setObjectName("Faint")
            self.uses.addWidget(lbl)


class RightPanel(QWidget):
    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedWidth(344)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.measures = MeasurementsTab(ctl)
        self.phases = PhasesTab(ctl)
        self.point = PointTab(ctl)
        self.tabs.addTab(self.measures, tr("Measurements"))
        self.tabs.addTab(self.phases, tr("Phases"))
        self.tabs.addTab(self.point, tr("Point"))
        self.tabs.tabBar().setContentsMargins(12, 0, 0, 0)
        lay.addWidget(self.tabs)
        def follow():
            if ctl.selected_landmark is not None:
                self.tabs.setCurrentWidget(self.point)
            elif ctl.selected_measure is not None:
                self.tabs.setCurrentWidget(self.measures)
        ctl.selectionChanged.connect(follow)
