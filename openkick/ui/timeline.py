"""Transport controls and the timeline (phases, measurement curves, frame markers)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QColorDialog, QComboBox, QHBoxLayout, QInputDialog, QLabel, QMenu,
                               QPushButton, QToolButton, QWidget)

from openkick.core import analysis as A
from openkick.core import tracks as T
from openkick.core.project import PHASE_COLORS
from openkick.i18n import tr
from openkick.ui import icons, theme
from openkick.ui.canvas import timecode
from openkick.ui.controller import Controller

LABEL_W = 150
RULER_H = 24
PHASE_H = 34
SUBPHASE_H = 24
CURVE_H = 40
FRAMES_H = 26
NOTES_H = 28
TRACK_H = 26              # a track the user made
HIDDEN_H = 14             # ... folded away
SNAP_PX = 8               # edges and events snap to anything this close
MAX_CURVES = 3


class TransportBar(QWidget):
    markPhase = Signal()
    addNote = Signal()
    newTrack = Signal(str)            # "segments" or "events"

    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setFixedHeight(56)
        self.playing = False
        self.speed = 1.0
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._tick)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(4)

        def btn(name, tip, slot):
            b = QToolButton()
            b.setIcon(icons.icon(name, theme.TEXT_2, 18))
            b.setIconSize(icons.ICON_SIZE)
            b.setFixedSize(34, 34)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        btn("start", tr("Go to start (Home)"), lambda: ctl.set_frame(0))
        btn("prev", tr("Previous frame (←)"), lambda: ctl.step(-1))
        self.play_btn = QToolButton()
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setToolTip(tr("Play / pause (Space)"))
        self.play_btn.setStyleSheet(f"QToolButton {{ background: {theme.ACCENT}; border-radius: 20px; }}"
                                    f"QToolButton:hover {{ background: {theme.ACCENT_HOVER}; }}")
        self.play_btn.clicked.connect(self.toggle)
        lay.addWidget(self.play_btn)
        btn("next", tr("Next frame (→)"), lambda: ctl.step(1))
        btn("end", tr("Go to end (End)"), lambda: ctl.set_frame(10 ** 9))
        lay.addSpacing(12)
        self.time_lbl = QLabel()
        self.time_lbl.setFont(theme.mono(20, 500))
        lay.addWidget(self.time_lbl)
        self.dur_lbl = QLabel()
        self.dur_lbl.setFont(theme.mono(13))
        self.dur_lbl.setStyleSheet(f"color: {theme.DIM};")
        lay.addWidget(self.dur_lbl)
        lay.addSpacing(12)
        self.frame_lbl = QLabel()
        self.frame_lbl.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        lay.addWidget(self.frame_lbl)
        lay.addSpacing(8)
        self.speed_box = QComboBox()
        self.speed_box.setFont(theme.mono(12))
        for s in (0.1, 0.25, 0.5, 1.0, 2.0):
            self.speed_box.addItem(f"{s:g}×", s)
        self.speed_box.setCurrentIndex(3)
        self.speed_box.setToolTip(tr("Playback speed"))
        self.speed_box.currentIndexChanged.connect(lambda _: setattr(self, "speed", self.speed_box.currentData()))
        lay.addWidget(self.speed_box)
        lay.addStretch(1)
        mark = QPushButton(icons.icon("flag", theme.TEXT, 16), " " + tr("Mark phase here"))
        mark.setToolTip(tr("Start a new phase at this frame (F)"))
        mark.clicked.connect(self.markPhase)
        lay.addWidget(mark)
        note = QPushButton(icons.icon("note", theme.TEXT, 16), " " + tr("Note"))
        note.setToolTip(tr("Add a text note on the video"))
        note.clicked.connect(self.addNote)
        lay.addWidget(note)
        track = QPushButton(icons.icon("plus", theme.TEXT, 16), " " + tr("Track"))
        track.setToolTip(tr("Add a row of your own to the timeline: segments of time or single events"))
        tmenu = QMenu(track)
        tmenu.addAction(tr("Segments track (stretches of time)"), lambda: self.newTrack.emit(T.SEGMENTS))
        tmenu.addAction(tr("Events track (single instants)"), lambda: self.newTrack.emit(T.EVENTS))
        track.setMenu(tmenu)
        lay.addWidget(track)
        ctl.frameChanged.connect(self.refresh)
        ctl.projectLoaded.connect(self.refresh)
        self._set_icon()

    def _set_icon(self):
        self.play_btn.setIcon(icons.icon(icons.PAUSE if self.playing else icons.PLAY, theme.ON_ACCENT, 18, fill=True))

    def refresh(self, *_):
        p = self.ctl.project
        if not p:
            return
        f = self.ctl.frame
        self.time_lbl.setText(timecode(p.video.time_of(f)))
        self.dur_lbl.setText("/ " + timecode(p.video.duration))
        self.frame_lbl.setText(tr("Frame {n} of {total}", n=f + 1, total=p.frame_count))

    def toggle(self):
        if not self.ctl.project:
            return
        self.playing = not self.playing
        if self.playing and self.ctl.frame >= self.ctl.project.frame_count - 1:
            self.ctl.set_frame(0)
        self._set_icon()
        self.ctl.playingChanged.emit(self.playing)
        if self.playing:
            self._schedule()
        else:
            self.timer.stop()

    def stop(self):
        if self.playing:
            self.toggle()

    def _schedule(self):
        p = self.ctl.project
        f = self.ctl.frame
        if f + 1 >= p.frame_count:
            self.stop()
            return
        dt = (p.video.time_of(f + 1) - p.video.time_of(f)) / self.speed
        self.timer.start(max(1, int(dt * 1000)))

    def _tick(self):
        if not self.playing or not self.ctl.project:
            return
        self.ctl.step(1)
        self._schedule()


class Timeline(QWidget):
    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.drag = None
        self.hover_x = None
        self.view = None          # (first, last second) shown when zoomed in; None = the whole video
        self.selected = None      # (row key, block id) of the block picked last
        self.snap_frame = None    # where the edge being dragged snapped to, drawn as a guide
        ctl.frameChanged.connect(lambda _: self.update())
        ctl.changed.connect(self._relayout)
        ctl.projectLoaded.connect(self._relayout)
        ctl.selectionChanged.connect(self.update)
        self._relayout()

    def curves(self):
        p = self.ctl.project
        if not p:
            return []
        ms = [m for m in p.measurements if m.visible and m.kind in ("angle", "trajectory", "distance")]
        ms.sort(key=lambda m: {"angle": 0, "trajectory": 1, "distance": 2}[m.kind])
        return ms[:MAX_CURVES]

    def _relayout(self, *_):
        n = len(self.curves())
        notes = NOTES_H if self.ctl.project and self.ctl.project.notes else 0
        subphases = SUBPHASE_H if self.ctl.project and self.ctl.project.subphases else 0
        tracks = sum(HIDDEN_H if t.hidden else TRACK_H for t in self.ctl.project.tracks) if self.ctl.project else 0
        self.setFixedHeight(RULER_H + PHASE_H + subphases + tracks + max(n, 1) * CURVE_H + notes + FRAMES_H + 18)
        self.update()

    # geometry
    def track_rect(self) -> QRectF:
        return QRectF(LABEL_W, 6, self.width() - LABEL_W - 18, self.height() - 12)

    def span(self) -> tuple[float, float]:
        """First and last second on screen."""
        dur = self.ctl.project.video.duration or 1
        return self.view if self.view else (0.0, dur)

    def x_of(self, frame: float) -> float:
        p = self.ctl.project
        frame = int(frame)
        if frame >= p.frame_count:          # the end of the last frame (a phase that lasts to the end)
            return self.x_of_time(p.video.duration or 1)
        return self.x_of_time(p.video.time_of(max(frame, 0)))

    def frame_at(self, x: float) -> int:
        p = self.ctl.project
        r = self.track_rect()
        t0, t1 = self.span()
        t = t0 + (x - r.left()) / max(r.width(), 1) * (t1 - t0)
        return p.video.index_at_time(t)

    def frames_per_px(self) -> float:
        t0, t1 = self.span()
        return (t1 - t0) * self.ctl.project.video.fps / max(self.track_rect().width(), 1)

    def zoom(self, factor: float, at_x: float | None = None):
        """Zoom in (factor < 1) or out around `at_x`. Down to about one second on screen."""
        dur = self.ctl.project.video.duration or 1
        t0, t1 = self.span()
        r = self.track_rect()
        anchor = t0 + ((at_x if at_x is not None else r.center().x()) - r.left()) / max(r.width(), 1) * (t1 - t0)
        width = max(min(dur, 30 / max(self.ctl.project.video.fps, 1)), min(dur, (t1 - t0) * factor))
        a = anchor - (anchor - t0) / (t1 - t0) * width
        a = max(0.0, min(dur - width, a))
        self.view = None if width >= dur - 1e-9 else (a, a + width)
        self.update()

    def scroll(self, seconds: float):
        if not self.view:
            return
        dur = self.ctl.project.video.duration or 1
        t0, t1 = self.view
        d = max(-t0, min(dur - t1, seconds))
        self.view = (t0 + d, t1 + d)
        self.update()

    def rows(self):
        r = self.track_rect()
        y = r.top()
        out = {"ruler": QRectF(r.left(), y, r.width(), RULER_H)}
        y += RULER_H
        out["phases"] = QRectF(r.left(), y, r.width(), PHASE_H)
        out["groups"] = out["phases"]
        y += PHASE_H
        if self.ctl.project and self.ctl.project.subphases:
            out["subphases"] = QRectF(r.left(), y, r.width(), SUBPHASE_H)
            y += SUBPHASE_H
        for t in (self.ctl.project.tracks if self.ctl.project else []):
            h = HIDDEN_H if t.hidden else TRACK_H
            out[f"track:{t.id}"] = QRectF(r.left(), y, r.width(), h)
            y += h
        curves = self.curves()
        for i, m in enumerate(curves):
            out[m.id] = QRectF(r.left(), y, r.width(), CURVE_H)
            y += CURVE_H
        if not curves:
            out["empty"] = QRectF(r.left(), y, r.width(), CURVE_H)
            y += CURVE_H
        if self.ctl.project.notes:
            out["notes"] = QRectF(r.left(), y, r.width(), NOTES_H)
            y += NOTES_H
        out["frames"] = QRectF(r.left(), y, r.width(), FRAMES_H)
        return out

    def x_of_time(self, t: float) -> float:
        r = self.track_rect()
        t0, t1 = self.span()
        return r.left() + (t - t0) / ((t1 - t0) or 1) * r.width()

    def note_bars(self):
        """(note, full-visibility rect, fade-in end x, fade-out end x) per note."""
        rows = self.rows()
        if "notes" not in rows:
            return []
        r = rows["notes"]
        v = self.ctl.project.video
        out = []
        for n in self.ctl.project.notes:
            ts, te = v.time_of(n.start), v.time_of(n.end)
            x0, x1 = self.x_of_time(ts), max(self.x_of_time(ts) + 4, self.x_of_time(te))
            out.append((n, QRectF(x0, r.top() + 5, x1 - x0, r.height() - 10),
                        self.x_of_time(ts + n.fade_in), self.x_of_time(te + n.fade_out)))
        return out

    # painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        project = self.ctl.project
        if not project:
            return
        rows = self.rows()
        self._labels(p, rows)
        tr_ = self.track_rect()
        p.save()
        p.setClipRect(QRectF(tr_.left() - 8, 0, tr_.width() + 16, self.height()))
        self._ruler(p, rows["ruler"])
        self._phases(p, rows["phases"])
        if "subphases" in rows:
            self._subphases(p, rows["subphases"])
        for t in project.tracks:
            self._track(p, t, rows[f"track:{t.id}"])
        for m in self.curves():
            self._curve(p, m, rows[m.id])
        if "empty" in rows:
            p.setPen(QColor(theme.DIM))
            p.setFont(theme.font(12))
            p.drawText(rows["empty"], Qt.AlignVCenter | Qt.AlignLeft,
                       "  " + tr("Measurement curves appear here when you add angles or trajectories"))
        if "notes" in rows:
            self._notes(p)
        self._frames(p, rows["frames"])
        # playhead
        x = self.x_of(self.ctl.frame)
        top, bottom = rows["ruler"].top(), rows["frames"].bottom()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawRect(QRectF(x - 1, top, 2, bottom - top))
        p.drawRoundedRect(QRectF(x - 7, top - 2, 14, 10), 3, 3)
        if self.snap_frame is not None and self.drag:          # the edge snapped here
            sx = self.x_of(self.snap_frame)
            p.setPen(QPen(QColor(theme.ACCENT), 1, Qt.DashLine))
            p.drawLine(QPointF(sx, top), QPointF(sx, bottom))
        p.restore()
        if self.view:                                           # zoomed in: where the view is in the video
            dur = project.video.duration or 1
            r = self.track_rect()
            a, b = self.view
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.LINE))
            p.drawRoundedRect(QRectF(r.left(), self.height() - 6, r.width(), 3), 1.5, 1.5)
            p.setBrush(QColor(theme.MUTED))
            p.drawRoundedRect(QRectF(r.left() + a / dur * r.width(), self.height() - 6,
                                     max(6, (b - a) / dur * r.width()), 3), 1.5, 1.5)

    def _labels(self, p: QPainter, rows):
        p.setFont(theme.font(12))
        lx = 16
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.font(12, 500))
        if self.ctl.project and self.ctl.project.subphases:
            p.drawText(QRectF(lx, rows["groups"].top(), LABEL_W - lx, PHASE_H), Qt.AlignVCenter, tr("Groups"))
            p.drawText(QRectF(lx, rows["subphases"].top(), LABEL_W - lx, SUBPHASE_H), Qt.AlignVCenter, tr("Phases"))
        else:
            p.drawText(QRectF(lx, rows["phases"].top(), LABEL_W - lx, PHASE_H), Qt.AlignVCenter, tr("Phases"))
        p.setFont(theme.font(12))
        for m in self.curves():
            r = rows[m.id]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(m.color))
            p.drawEllipse(QPointF(lx + 4, r.center().y()), 4, 4)
            p.setPen(QColor(theme.TEXT if m.id == self.ctl.selected_measure else theme.MUTED))
            fm = QFontMetricsF(p.font())
            p.drawText(QRectF(lx + 14, r.top(), LABEL_W - lx - 20, CURVE_H), Qt.AlignVCenter,
                       fm.elidedText(m.name, Qt.ElideRight, LABEL_W - lx - 24))
        for t in self.ctl.project.tracks:
            r = rows[f"track:{t.id}"]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(t.color))
            if t.kind == T.EVENTS:
                c = QPointF(lx + 4, r.center().y())
                p.drawPolygon(QPolygonF([c + QPointF(0, -4), c + QPointF(4, 0), c + QPointF(0, 4), c + QPointF(-4, 0)]))
            else:
                p.drawRoundedRect(QRectF(lx, r.center().y() - 3.5, 8, 7), 2, 2)
            p.setPen(QColor(theme.DIM if t.hidden else theme.TEXT_2))
            p.setFont(theme.font(10 if t.hidden else 12))
            fm = QFontMetricsF(p.font())
            name = t.name + ("  🔒" if t.locked else "")
            p.drawText(QRectF(lx + 14, r.top(), LABEL_W - lx - 20, r.height()), Qt.AlignVCenter,
                       fm.elidedText(name, Qt.ElideRight, LABEL_W - lx - 24))
        p.setFont(theme.font(12))
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(lx, rows["frames"].top(), LABEL_W, FRAMES_H), Qt.AlignVCenter, tr("Frames"))
        if "notes" in rows:
            p.drawText(QRectF(lx, rows["notes"].top(), LABEL_W, NOTES_H), Qt.AlignVCenter, tr("Notes"))

    def _ruler(self, p: QPainter, r: QRectF):
        dur = self.ctl.project.video.duration
        if dur <= 0 or r.width() <= 0:              # nothing to measure yet (window still being laid out)
            return
        t0, t1 = self.span()
        step = next((s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600)
                     if r.width() / max((t1 - t0) / s, 1) >= 56), 600)
        p.setFont(theme.mono(11))
        t = int(t0 / step) * step
        while t <= t1 + 1e-9:
            x = self.x_of_time(t)
            p.setPen(QPen(QColor("#2A303A"), 1))
            p.drawLine(QPointF(x, r.top() + 4), QPointF(x, r.bottom()))
            p.setPen(QColor(theme.DIM))
            label = f"{t:.1f} s" if t == 0 else (f"{t:.2f}" if step < 0.1 else f"{t:.1f}" if step < 1 else f"{t:.0f}")
            p.drawText(QPointF(x + 4, r.top() + 14), label)
            t += step

    def _phases(self, p: QPainter, r: QRectF):
        project = self.ctl.project
        cur = project.phase_at(self.ctl.frame)
        p.setFont(theme.font(12, 500))
        fm = QFontMetricsF(p.font())
        for i, ph in enumerate(project.phases):
            x0 = self.x_of(ph.start)
            x1 = self.x_of(ph.end + 1) if ph.end + 1 < project.frame_count else r.right()
            rect = QRectF(x0 + 1, r.top() + 4, max(2, x1 - x0 - 2), r.height() - 8)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ph.color))
            p.drawRoundedRect(rect, 6, 6)
            if ph is cur or self.selected == ("phases", ph.id):
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(theme.ACCENT if ph is cur else theme.TEXT), 1.5))
                p.drawRoundedRect(rect.adjusted(0.75, 0.75, -0.75, -0.75), 6, 6)
            p.setPen(QColor(theme.text_on(ph.color)))
            numbered = ph.name.lower().startswith(("pedalada", "stroke", "zancada", "stride"))   # already "Stride 3"
            full_lbl = ph.name if (ph.name.startswith(f"{i + 1} ") or numbered) else f"{i + 1} · {ph.name}"
            label = fm.elidedText(full_lbl if rect.width() >= 75 else f"P{i + 1}", Qt.ElideRight, int(rect.width() - 8))
            align = Qt.AlignCenter if rect.width() < 75 else (Qt.AlignVCenter | Qt.AlignLeft)
            p.drawText(rect.adjusted(4, 0, -4, 0), align, label)
        if not project.phases:
            p.setPen(QPen(QColor(theme.BORDER), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 4, -1, -4), 6, 6)
            p.setPen(QColor(theme.DIM))
            p.setFont(theme.font(12))
            p.drawText(r, Qt.AlignCenter, tr("No phases yet · move to a frame and press “Mark phase here” (F)"))

    def _subphases(self, p: QPainter, r: QRectF):
        project = self.ctl.project
        if not project or not project.subphases:
            return
        cur = project.subphase_at(self.ctl.frame)
        p.setFont(theme.font(10, 600))
        fm = QFontMetricsF(p.font())
        for ph in project.subphases:
            x0 = self.x_of(ph.start)
            x1 = self.x_of(ph.end + 1) if ph.end + 1 < project.frame_count else r.right()
            rect = QRectF(x0 + 1, r.top() + 2, max(2, x1 - x0 - 2), r.height() - 4)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ph.color))
            p.drawRoundedRect(rect, 4, 4)
            if ph is cur:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(theme.ACCENT), 1.5))
                p.drawRoundedRect(rect.adjusted(0.75, 0.75, -0.75, -0.75), 4, 4)
            p.setPen(QColor(theme.text_on(ph.color)))
            w = rect.width()
            if w >= 55:
                lbl = fm.elidedText(ph.name, Qt.ElideRight, int(w - 4))
                p.drawText(rect.adjusted(2, 0, -2, 0), Qt.AlignCenter, lbl)
            elif w >= 16:
                from openkick.core.gait import short_name
                short = fm.elidedText(short_name(ph.name), Qt.ElideNone, int(w))
                p.drawText(rect, Qt.AlignCenter, short if fm.horizontalAdvance(short) <= w else short[:1])

    def _curve(self, p: QPainter, m, r: QRectF):
        project = self.ctl.project
        s = A.series(project, m)
        v = s.value
        ok = np.isfinite(v)
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        if not ok.any():
            return
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi - lo < 1e-6:
            lo, hi = lo - 1, hi + 1
        pad = 5
        path = QPainterPath()
        started = False
        for i in range(len(v)):
            if not ok[i]:
                started = False
                continue
            x = self.x_of(i)
            y = r.bottom() - pad - (v[i] - lo) / (hi - lo) * (r.height() - 2 * pad)
            if started:
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
                started = True
        sel = m.id == self.ctl.selected_measure
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(m.color), 2.4 if sel else 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
        f = self.ctl.frame
        if ok[f]:
            x = self.x_of(f)
            y = r.bottom() - pad - (v[f] - lo) / (hi - lo) * (r.height() - 2 * pad)
            p.setPen(QPen(QColor(theme.PANEL), 2))
            p.setBrush(QColor(m.color))
            p.drawEllipse(QPointF(x, y), 4, 4)

    def _notes(self, p: QPainter):
        from PySide6.QtGui import QLinearGradient
        p.setFont(theme.font(11, 500))
        fm = QFontMetricsF(p.font())
        for n, rect, x_in, x_out in self.note_bars():
            c = QColor(n.color)
            # fade-out tail after the end of the note
            if x_out > rect.right() + 1:
                g = QLinearGradient(rect.right(), 0, x_out, 0)
                c1, c2 = QColor(c), QColor(c)
                c1.setAlpha(110)
                c2.setAlpha(0)
                g.setColorAt(0, c1)
                g.setColorAt(1, c2)
                p.setPen(Qt.NoPen)
                p.setBrush(g)
                p.drawRect(QRectF(rect.right(), rect.top(), x_out - rect.right(), rect.height()))
            body = QColor(c)
            body.setAlpha(110)
            if x_in > rect.left() + 1:
                g = QLinearGradient(rect.left(), 0, x_in, 0)
                c0 = QColor(c)
                c0.setAlpha(0)
                g.setColorAt(0, c0)
                g.setColorAt(1, body)
                p.setBrush(g)
            else:
                p.setBrush(body)
            p.setPen(QPen(c, 1))
            p.drawRoundedRect(rect, 4, 4)
            p.setPen(QColor(theme.TEXT))
            txt = fm.elidedText(n.text.replace("\n", " "), Qt.ElideRight, int(rect.width() - 10))
            p.drawText(rect.adjusted(6, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, txt)

    def _note_hit(self, pos: QPointF):
        for n, rect, _, _ in reversed(self.note_bars()):
            if rect.adjusted(-5, -2, 5, 2).contains(pos):
                if abs(pos.x() - rect.left()) < 5:
                    return n, "start"
                if abs(pos.x() - rect.right()) < 5:
                    return n, "end"
                return n, "move"
        return None

    def _frames(self, p: QPainter, r: QRectF):
        project = self.ctl.project
        n = project.frame_count
        spacing = 1 / max(self.frames_per_px(), 1e-9)
        if spacing >= 4:
            p.setPen(QPen(QColor(theme.LINE), 1))
            for i in range(n):
                x = self.x_of(i)
                p.drawLine(QPointF(x, r.top() + 6), QPointF(x, r.bottom() - 6))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.WARN))
        for i in self.ctl.low_confidence_frames():
            x = self.x_of(i)
            p.drawPolygon(QPolygonF([QPointF(x, r.top() + 7), QPointF(x - 5, r.top() + 17), QPointF(x + 5, r.top() + 17)]))
        p.setBrush(QColor(theme.ACCENT))
        for i in self.ctl.manual_frames():
            x = self.x_of(int(i))
            c = r.center()
            p.drawPolygon(QPolygonF([QPointF(x, c.y() - 5), QPointF(x + 5, c.y()), QPointF(x, c.y() + 5), QPointF(x - 5, c.y())]))

    # interaction
    def _boundary_at(self, pos: QPointF):
        project = self.ctl.project
        rows = self.rows()
        if project.subphases:           # strides and their phases: one edge moves both rows together
            from openkick.core import phase_edit as E
            band = rows["phases"].united(rows["subphases"]) if "subphases" in rows else rows["phases"]
            if not band.adjusted(0, -2, 0, 2).contains(pos):
                return None
            row = project.subphases if "subphases" in rows and rows["subphases"].contains(pos) else project.phases
            near = [(abs(self.x_of(f) - pos.x()), f) for f in E.edges(row) if 0 < f < project.frame_count]
            near = [n for n in near if n[0] < 5]
            return ("edge", min(near)[1]) if near else None
        if not rows["phases"].adjusted(0, -2, 0, 2).contains(pos):
            return None
        for i, ph in enumerate(project.phases):
            if abs(self.x_of(ph.start) - pos.x()) < 5 and i > 0 and project.phases[i - 1].end == ph.start - 1:
                return ("start", i)
            if abs(self.x_of(ph.end + 1) - pos.x()) < 5:
                return ("end", i)
            if abs(self.x_of(ph.start) - pos.x()) < 5:
                return ("start", i)
        return None

    def _phase_at_pos(self, pos: QPointF):
        project = self.ctl.project
        if not self.rows()["phases"].contains(pos):
            return None
        f = self.frame_at(pos.x())
        return project.phase_at(f)

    def _subphase_at_pos(self, pos: QPointF):
        project = self.ctl.project
        rows = self.rows()
        if not project or "subphases" not in rows or not rows["subphases"].contains(pos):
            return None
        f = self.frame_at(pos.x())
        return project.subphase_at(f)

    def mousePressEvent(self, e):
        project = self.ctl.project
        if not project:
            return
        pos = e.position()
        key = self._row_at(pos)
        if pos.x() < LABEL_W - 4:                   # the names of the rows
            if e.button() == Qt.RightButton and (key in ("phases", "subphases") or (key or "").startswith("track:")):
                self._track_menu(key, e.globalPosition().toPoint())
            return
        t = self._track_of(key)
        if t:                                       # one of the user's tracks
            item, part = self._item_at(t, pos)
            f = self.frame_at(pos.x())
            if e.button() == Qt.RightButton:
                if item is not None:
                    self.selected = (key, item.id)
                    self._item_menu(key, item, e.globalPosition().toPoint())
                elif not t.hidden:
                    self._empty_menu(t, f, e.globalPosition().toPoint())
                self.update()
                return
            if item is not None:
                self.selected = (key, item.id)
                if not t.locked:
                    grab = f - (item.frame if isinstance(item, T.Mark) else item.start)
                    self.drag = ("item", key, item.id, part, grab, self.ctl.begin(), pos.x(), False)
                    self.update()
                    return
            else:
                self.selected = None
            self.drag = ("scrub",)
            self.ctl.set_frame(f)
            self.update()
            return
        if e.button() == Qt.RightButton:
            nh = self._note_hit(pos)
            if nh:
                from openkick.ui.canvas import edit_note
                n = nh[0]
                menu = QMenu(self)
                theme.menu_title(menu, n.text.split("\n")[0][:40])
                menu.addAction(tr("Edit note…"), lambda: edit_note(self, self.ctl, n))
                menu.addAction(tr("Delete note"), lambda: self.ctl.change(
                    tr("Delete note"), lambda: self.ctl.project.notes.remove(n)))
                menu.exec(e.globalPosition().toPoint())
                return
            sph = self._subphase_at_pos(pos)
            if sph:
                menu = QMenu(self)
                theme.menu_title(menu, sph.name)
                menu.addAction(tr("Go to start"), lambda: self.ctl.set_frame(sph.start))
                menu.exec(e.globalPosition().toPoint())
                return
            ph = self._phase_at_pos(pos)
            if ph:
                self.selected = ("phases", ph.id)
                self._item_menu("phases", ph, e.globalPosition().toPoint())
            elif key == "ruler" and self.view:
                menu = QMenu(self)
                menu.addAction(tr("Show the whole video") + "\tCtrl+0", lambda: (setattr(self, "view", None), self.update()))
                menu.exec(e.globalPosition().toPoint())
            return
        b = self._boundary_at(pos)
        if b:
            self.drag = ("boundary", b, self.ctl.begin())
            return
        ph = self._phase_at_pos(pos)
        if ph:                                      # a phase: picked, and moved if dragged
            self.selected = ("phases", ph.id)
            self.drag = ("item", "phases", ph.id, "body", self.frame_at(pos.x()) - ph.start, self.ctl.begin(), pos.x(), False)
            self.ctl.set_frame(self.frame_at(pos.x()))
            return
        nh = self._note_hit(pos)
        if nh:
            n, part = nh
            self.drag = ("note", n.id, part, self.frame_at(pos.x()), n.start, n.end, self.ctl.begin())
            return
        for m in self.curves():
            if self.rows()[m.id].contains(pos):
                self.ctl.select_measure(m.id)
        self.drag = ("scrub",)
        self.ctl.set_frame(self.frame_at(pos.x()))

    def mouseMoveEvent(self, e):
        project = self.ctl.project
        if not project:
            return
        pos = e.position()
        if not self.drag:
            nh = self._note_hit(pos)
            if nh:
                self.setCursor(Qt.SizeHorCursor if nh[1] != "move" else Qt.SizeAllCursor)
                self.setToolTip(nh[0].text)
                return
            t = self._track_of(self._row_at(pos))
            if t:
                item, part = self._item_at(t, pos)
                self.setCursor(Qt.ArrowCursor if item is None or t.locked else
                               Qt.SizeHorCursor if part in ("start", "end") else Qt.OpenHandCursor)
                self.setToolTip(item.name if item is not None else
                                tr("Double click to add here · right click for more"))
                return
            self.setCursor(Qt.SizeHorCursor if self._boundary_at(pos) else Qt.ArrowCursor)
            sph = self._subphase_at_pos(pos)
            if sph:
                ph = project.phase_at(self.frame_at(pos.x()))
                prefix = f"{ph.name} · " if ph else ""
                self.setToolTip(f"{prefix}{sph.name} · {timecode(project.video.time_of(sph.start))} – "
                                f"{timecode(project.video.time_of(sph.end))}")
                return
            ph = self._phase_at_pos(pos)
            self.setToolTip(f"{ph.name} · {timecode(project.video.time_of(ph.start))} – "
                            f"{timecode(project.video.time_of(ph.end))}" if ph else "")
            return
        f = self.frame_at(pos.x())
        if self.drag[0] == "item":
            self._drag_item(pos, f)
            return
        if self.drag[0] == "note":
            _, nid, part, f0, s0, e0, _snap = self.drag
            n = next((x for x in project.notes if x.id == nid), None)
            if n:
                last = project.frame_count - 1
                if part == "start":
                    n.start = max(0, min(f, n.end))
                elif part == "end":
                    n.end = min(last, max(f, n.start))
                else:
                    d = max(-s0, min(last - e0, f - f0))
                    n.start, n.end = s0 + d, e0 + d
                project.touch()
                self.ctl.set_frame(n.start if part != "end" else n.end)
                self.update()
            return
        if self.drag[0] == "scrub":
            self.ctl.set_frame(f)
        elif self.drag[0] == "boundary" and self.drag[1][0] == "edge":
            from openkick.core import phase_edit as E
            edge = self.drag[1][1]
            moving = [b for b in [*project.phases, *project.subphases] if b.start == edge or b.end == edge - 1]
            f = self._snap(f, skip=moving)
            moved = E.drag(project.phases, project.subphases, edge, f, project.frame_count)
            self.drag = ("boundary", ("edge", moved), self.drag[2])
            project.touch()
            self.ctl.set_frame(moved)
            self.update()
        elif self.drag[0] == "boundary":
            side, i = self.drag[1]
            phs = project.phases
            ph = phs[i]
            if side == "start":
                lo = phs[i - 1].start + 1 if i > 0 else 0
                ph.start = max(lo, min(f, ph.end))
                if i > 0 and phs[i - 1].end >= ph.start - 1:
                    phs[i - 1].end = ph.start - 1
            else:
                f = f - 1
                hi = phs[i + 1].end - 1 if i + 1 < len(phs) else project.frame_count - 1
                ph.end = max(ph.start, min(f, hi))
                if i + 1 < len(phs) and phs[i + 1].start <= ph.end + 1:
                    phs[i + 1].start = ph.end + 1
            project.touch()
            self.ctl.set_frame(ph.start if side == "start" else ph.end)
            self.update()

    def mouseReleaseEvent(self, e):
        self.snap_frame = None
        if self.drag and self.drag[0] == "item":
            _, key, bid, part, grab, before, x0, moved = self.drag
            if moved:
                self.ctl.commit(before, tr("Move") if part == "body" else tr("Adjust"))
            elif key != "phases":
                b = self._find(key, bid)
                if b is not None:
                    self.ctl.set_frame(b.frame if isinstance(b, T.Mark) else self.frame_at(e.position().x()))
            self.drag = None
            self.update()
            return
        if self.drag and self.drag[0] == "boundary":
            self.ctl.commit(self.drag[2], tr("Adjust phase"))
        if self.drag and self.drag[0] == "note":
            self.ctl.commit(self.drag[6], tr("Edit note"))
        self.drag = None

    def mouseDoubleClickEvent(self, e):
        key = self._row_at(e.position())
        t = self._track_of(key)
        if t and e.position().x() >= LABEL_W - 4:
            item, _ = self._item_at(t, e.position())
            if item is not None and not t.locked:
                self._rename_item(key, item)
            elif item is None and not t.locked and not t.hidden:
                self._add_here(t, self.frame_at(e.position().x()))
            return
        nh = self._note_hit(e.position())
        if nh:
            from openkick.ui.canvas import edit_note
            edit_note(self, self.ctl, nh[0])
            return
        sph = self._subphase_at_pos(e.position())
        if sph:
            self.ctl.set_frame(sph.start)
            return
        ph = self._phase_at_pos(e.position())
        if ph:
            rename_phase(self, self.ctl, ph)

    def _track(self, p: QPainter, t, r: QRectF):
        """One of the user's tracks: segments as blocks, events as flags."""
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        if t.hidden:
            return
        p.setFont(theme.font(10, 600))
        fm = QFontMetricsF(p.font())
        if t.kind == T.SEGMENTS:
            for b in t.items:
                x0, x1 = self.x_of(b.start), self.x_of(b.end + 1)
                rect = QRectF(x0 + 1, r.top() + 3, max(2, x1 - x0 - 2), r.height() - 6)
                c = QColor(b.color)
                if t.locked:
                    c.setAlpha(150)
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawRoundedRect(rect, 4, 4)
                if self.selected == (f"track:{t.id}", b.id):
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(theme.TEXT), 1.5))
                    p.drawRoundedRect(rect.adjusted(0.75, 0.75, -0.75, -0.75), 4, 4)
                if rect.width() >= 16:
                    p.setPen(QColor(theme.text_on(b.color)))
                    p.drawText(rect.adjusted(3, 0, -3, 0), Qt.AlignCenter,
                               fm.elidedText(b.name, Qt.ElideRight, int(rect.width() - 6)))
            return
        for m in t.items:
            x = self.x_of(m.frame)
            sel = self.selected == (f"track:{t.id}", m.id)
            p.setPen(QPen(QColor(m.color), 2 if sel else 1.2))
            p.drawLine(QPointF(x, r.top() + 2), QPointF(x, r.bottom() - 2))
            p.setPen(QPen(QColor(theme.TEXT) if sel else Qt.NoPen))
            p.setBrush(QColor(m.color))
            c = QPointF(x, r.top() + 8)
            p.drawPolygon(QPolygonF([c + QPointF(0, -5), c + QPointF(5, 0), c + QPointF(0, 5), c + QPointF(-5, 0)]))
            p.setPen(QColor(theme.TEXT_2))
            p.drawText(QRectF(x + 7, r.top(), 140, r.height()), Qt.AlignVCenter | Qt.AlignLeft,
                       fm.elidedText(m.name, Qt.ElideRight, 136))

    # ── the user's tracks: finding things under the mouse ──
    def _row_at(self, pos: QPointF):
        """Key of the row under `pos` ("phases", "subphases", "track:<id>"...) or None."""
        for key, r in self.rows().items():
            if key in ("groups",):
                continue
            if r.top() <= pos.y() < r.bottom():
                return key
        return None

    def _track_of(self, key):
        if not key or not key.startswith("track:"):
            return None
        return next((t for t in self.ctl.project.tracks if t.id == key[6:]), None)

    def _item_at(self, t, pos: QPointF):
        """(block or mark, part) under `pos` on track `t`; part is "start", "end" or "body"."""
        if t.hidden:
            return None, None
        if t.kind == T.EVENTS:
            near = [(abs(self.x_of(m.frame) - pos.x()), m) for m in t.items]
            near = [n for n in near if n[0] <= 6]
            return (min(near, key=lambda n: n[0])[1], "body") if near else (None, None)
        for b in t.items:
            x0, x1 = self.x_of(b.start), self.x_of(b.end + 1)
            if abs(pos.x() - x0) <= 5:
                return b, "start"
            if abs(pos.x() - x1) <= 5:
                return b, "end"
            if x0 < pos.x() < x1:
                return b, "body"
        return None, None

    def _snap(self, frame: int, skip=()) -> int:
        """`frame`, or the edge, event or playhead within SNAP_PX of it. Alt held: no snapping."""
        from PySide6.QtGui import QGuiApplication
        self.snap_frame = None
        if QGuiApplication.keyboardModifiers() & Qt.AltModifier:
            return frame
        project = self.ctl.project
        rows = [project.phases, project.subphases] + [t.items for t in project.tracks if not t.hidden]
        out, hit = T.snap(frame, T.snap_targets(rows, self.ctl.frame, skip), SNAP_PX * self.frames_per_px())
        self.snap_frame = hit
        return out

    def _selected_item(self):
        if not self.selected or not self.ctl.project:
            return None, None
        key, bid = self.selected
        project = self.ctl.project
        if key == "phases":
            return project.phases, next((b for b in project.phases if b.id == bid), None)
        t = self._track_of(key)
        if t:
            return t.items, next((b for b in t.items if b.id == bid), None)
        return None, None

    # ── zoom and scroll ──
    def wheelEvent(self, e):
        if not self.ctl.project:
            return
        dy = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.ControlModifier:
            self.zoom(0.8 if dy > 0 else 1.25, e.position().x())
            e.accept()
        elif self.view:
            t0, t1 = self.view
            self.scroll(-(t1 - t0) * 0.1 * (1 if dy > 0 else -1))
            e.accept()
        else:
            e.ignore()

    # ── keys ──
    def keyPressEvent(self, e):
        blocks, b = self._selected_item()
        if b is not None and e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_item(self.selected[0], b)
        elif b is not None and e.key() == Qt.Key_B and e.modifiers() & Qt.ControlModifier:
            self._split(self.selected[0], b)
        elif e.key() in (Qt.Key_Plus, Qt.Key_Equal) and e.modifiers() & Qt.ControlModifier:
            self.zoom(0.8)
        elif e.key() == Qt.Key_Minus and e.modifiers() & Qt.ControlModifier:
            self.zoom(1.25)
        elif e.key() == Qt.Key_0 and e.modifiers() & Qt.ControlModifier:
            self.view = None
            self.update()
        else:
            super().keyPressEvent(e)

    # ── edits, each one undoable ──
    def _blocks_of(self, key):
        if key == "phases":
            return self.ctl.project.phases
        t = self._track_of(key)
        return t.items if t else None

    def _find(self, key, bid):
        blocks = self._blocks_of(key)
        return next((b for b in blocks if b.id == bid), None) if blocks is not None else None

    def _split(self, key, b):
        f = self.ctl.frame

        def apply():
            blk = self._find(key, b.id)
            new = T.split(self._blocks_of(key), blk, f) if blk is not None and not isinstance(blk, T.Mark) else None
            if new:
                self.selected = (key, new.id)
        self.ctl.change(tr("Split at the playhead"), apply)

    def _merge(self, key, b):
        self.ctl.change(tr("Merge with the next one"), lambda: T.merge_next(self._blocks_of(key), self._find(key, b.id)))

    def _duplicate(self, key, b):
        n = self.ctl.project.frame_count

        def apply():
            new = T.duplicate(self._blocks_of(key), self._find(key, b.id), n)
            if new:
                self.selected = (key, new.id)
        self.ctl.change(tr("Duplicate"), apply)

    def _delete_item(self, key, b):
        if key == "phases":
            delete_phase(self.ctl, b)
        else:
            self.ctl.change(tr("Delete"), lambda: self._blocks_of(key).remove(self._find(key, b.id)))
        self.selected = None

    def _rename_item(self, key, b):
        if key == "phases":
            rename_phase(self, self.ctl, b)
            return
        name, ok = QInputDialog.getText(self, tr("Rename"), tr("Name:"), text=b.name)
        if ok and name.strip():
            self.ctl.change(tr("Rename"), lambda: setattr(self._find(key, b.id), "name", name.strip()))

    def _recolor_item(self, key, b):
        if key == "phases":
            recolor_phase(self, self.ctl, b)
            return
        c = QColorDialog.getColor(QColor(b.color), self, tr("Color"))
        if c.isValid():
            self.ctl.change(tr("Color"), lambda: setattr(self._find(key, b.id), "color", c.name().upper()))

    def _add_here(self, t, frame: int):
        fps = self.ctl.project.video.fps
        n = self.ctl.project.frame_count

        def apply():
            tt = self._track_of(f"track:{t.id}")
            if t.kind == T.EVENTS:
                m = T.add_mark(tt.items, frame, tr("Event {n}", n=len(tt.items) + 1), tt.color)
                self.selected = (f"track:{t.id}", m.id)
            else:
                b = T.add_segment(tt.items, frame, int(round(fps)), n, tr("Segment {n}", n=len(tt.items) + 1), tt.color)
                if b:
                    self.selected = (f"track:{t.id}", b.id)
        self.ctl.change(tr("Add event") if t.kind == T.EVENTS else tr("Add segment"), apply)

    def _new_track(self, kind: str, at: int | None):
        name = tr("Events") if kind == T.EVENTS else tr("Segments")
        tracks = self.ctl.project.tracks
        name = f"{name} {sum(t.kind == kind for t in tracks) + 1}"
        self.ctl.change(tr("New track"), lambda: T.new_track(self.ctl.project.tracks, kind, name, at))
        self._relayout()

    def _track_menu(self, key, gpos):
        """Right click on a row's name: tracks above and below, and the track's own options."""
        project = self.ctl.project
        t = self._track_of(key)
        menu = QMenu(self)
        theme.menu_title(menu, t.name if t else (tr("Phases") if key in ("phases", "subphases") else tr("Timeline")))
        i = project.tracks.index(t) if t else 0
        if t:
            menu.addAction(tr("New segments track above"), lambda: self._new_track(T.SEGMENTS, i))
            menu.addAction(tr("New events track above"), lambda: self._new_track(T.EVENTS, i))
        menu.addAction(tr("New segments track below"), lambda: self._new_track(T.SEGMENTS, i + 1 if t else 0))
        menu.addAction(tr("New events track below"), lambda: self._new_track(T.EVENTS, i + 1 if t else 0))
        if t:
            menu.addSeparator()
            tid = t.id

            def tt():
                return next(x for x in self.ctl.project.tracks if x.id == tid)

            def rename():
                name, ok = QInputDialog.getText(self, tr("Rename track"), tr("Name:"), text=t.name)
                if ok and name.strip():
                    self.ctl.change(tr("Rename track"), lambda: setattr(tt(), "name", name.strip()))

            def recolor():
                c = QColorDialog.getColor(QColor(t.color), self, tr("Track color"))
                if c.isValid():
                    self.ctl.change(tr("Track color"), lambda: setattr(tt(), "color", c.name().upper()))
            menu.addAction(tr("Rename track…"), rename)
            menu.addAction(tr("Track color…"), recolor)
            up = menu.addAction(tr("Move up"), lambda: self.ctl.change(tr("Move track"), lambda: T.move_track(
                self.ctl.project.tracks, tt(), -1)))
            up.setEnabled(i > 0)
            down = menu.addAction(tr("Move down"), lambda: self.ctl.change(tr("Move track"), lambda: T.move_track(
                self.ctl.project.tracks, tt(), 1)))
            down.setEnabled(i < len(project.tracks) - 1)
            menu.addAction(tr("Show") if t.hidden else tr("Hide"), lambda: (self.ctl.change(
                tr("Hide track"), lambda: setattr(tt(), "hidden", not tt().hidden)), self._relayout()))
            menu.addAction(tr("Unlock") if t.locked else tr("Lock"), lambda: self.ctl.change(
                tr("Lock track"), lambda: setattr(tt(), "locked", not tt().locked)))
            menu.addSeparator()
            menu.addAction(tr("Delete track"), lambda: (self.ctl.change(
                tr("Delete track"), lambda: self.ctl.project.tracks.remove(tt())), self._relayout()))
        menu.exec(gpos)

    def _item_menu(self, key, b, gpos):
        """Right click on a block of the phases or of a track of the user."""
        menu = QMenu(self)
        theme.menu_title(menu, b.name)
        t = self._track_of(key)
        locked = bool(t and t.locked)
        menu.addAction(tr("Rename…"), lambda: self._rename_item(key, b)).setEnabled(not locked)
        menu.addAction(tr("Color…"), lambda: self._recolor_item(key, b)).setEnabled(not locked)
        if isinstance(b, T.Mark):
            menu.addAction(tr("Go to it"), lambda: self.ctl.set_frame(b.frame))
        else:
            menu.addAction(tr("Go to start"), lambda: self.ctl.set_frame(b.start))
            menu.addSeparator()
            cut = menu.addAction(tr("Split at the playhead") + "\tCtrl+B", lambda: self._split(key, b))
            cut.setEnabled(not locked and b.start < self.ctl.frame <= b.end)
            later = any(x.start > b.end for x in self._blocks_of(key))
            menu.addAction(tr("Merge with the next one"), lambda: self._merge(key, b)).setEnabled(not locked and later)
            menu.addAction(tr("Duplicate"), lambda: self._duplicate(key, b)).setEnabled(not locked)
        menu.addSeparator()
        menu.addAction(tr("Delete") + "\tSupr", lambda: self._delete_item(key, b)).setEnabled(not locked)
        menu.exec(gpos)

    def _empty_menu(self, t, frame: int, gpos):
        menu = QMenu(self)
        theme.menu_title(menu, t.name)
        add = menu.addAction(tr("Add an event here") if t.kind == T.EVENTS else tr("Add a segment here"),
                             lambda: self._add_here(t, frame))
        add.setEnabled(not t.locked)
        menu.addSeparator()
        menu.addAction(tr("Track options…"), lambda: self._track_menu(f"track:{t.id}", gpos))
        menu.exec(gpos)

    def _drag_item(self, pos: QPointF, f: int):
        """Move a block or an event, or one edge of a block, with snapping."""
        project = self.ctl.project
        _, key, bid, part, grab, before, x0, moved = self.drag
        if not moved and abs(pos.x() - x0) < 3:            # a click, not a drag (yet)
            return
        self.drag = ("item", key, bid, part, grab, before, x0, True)
        blocks = self._blocks_of(key)
        b = self._find(key, bid)
        if b is None:
            return
        n = project.frame_count
        if isinstance(b, T.Mark):
            T.move_mark(blocks, b, self._snap(f - grab, skip=[b]), n)
            self.ctl.set_frame(b.frame)
        elif part == "body":
            children = [c for c in project.subphases if b.start <= c.start and c.end <= b.end] if key == "phases" else []
            length = b.end - b.start + 1
            start = f - grab
            a = self._snap(start, skip=[b, *children])
            hit_a = self.snap_frame
            z = self._snap(start + length, skip=[b, *children]) - length
            if hit_a is not None and (self.snap_frame is None or abs(a - start) <= abs(z - start)):
                start, self.snap_frame = a, hit_a
            elif self.snap_frame is not None:
                start = z
            T.move(blocks, b, start, n, children)
        else:
            edge = self._snap(f if part == "start" else f, skip=[b])
            T.resize(blocks, b, part, edge if part == "start" else edge - 1, n)
            self.ctl.set_frame(b.start if part == "start" else b.end)
        project.touch()
        self.update()

    def _phase_menu(self, ph, gpos):
        menu = QMenu(self)
        theme.menu_title(menu, ph.name)
        menu.addAction(tr("Rename…"), lambda: rename_phase(self, self.ctl, ph))
        menu.addAction(tr("Color…"), lambda: recolor_phase(self, self.ctl, ph))
        menu.addAction(tr("Go to start"), lambda: self.ctl.set_frame(ph.start))
        menu.addAction(tr("Delete phase"), lambda: delete_phase(self.ctl, ph))
        menu.exec(gpos)


def _phase_by_id(ctl, pid):
    return next((p for p in ctl.project.phases if p.id == pid), None)


def rename_phase(parent, ctl: Controller, ph):
    name, ok = QInputDialog.getText(parent, tr("Rename phase"), tr("Name:"), text=ph.name)
    if ok and name.strip():
        ctl.change(tr("Rename phase"), lambda: setattr(_phase_by_id(ctl, ph.id), "name", name.strip()))


def recolor_phase(parent, ctl: Controller, ph):
    c = QColorDialog.getColor(QColor(ph.color), parent, tr("Phase color"))
    if c.isValid():
        ctl.change(tr("Phase color"), lambda: setattr(_phase_by_id(ctl, ph.id), "color", c.name().upper()))


def delete_phase(ctl: Controller, ph):
    def apply():
        phs = ctl.project.phases
        i = next(k for k, p in enumerate(phs) if p.id == ph.id)
        removed = phs.pop(i)
        if i > 0 and phs[i - 1].end == removed.start - 1:
            phs[i - 1].end = removed.end       # the previous phase absorbs the gap
    ctl.change(tr("Delete phase"), apply)


def next_phase_name(ctl: Controller) -> str:
    from openkick.core import templates
    t = templates.find(ctl.project.template)
    used = {p.name for p in ctl.project.phases}
    if t:
        for name in t.phase_names():
            if name not in used:
                return name
    return tr("Phase {n}", n=len(ctl.project.phases) + 1)


def mark_phase(parent, ctl: Controller):
    if not ctl.project:
        return
    name, ok = QInputDialog.getText(parent, tr("Mark phase"), tr("The new phase starts at this frame. Name:"),
                                    text=next_phase_name(ctl))
    if ok and name.strip():
        f = ctl.frame
        ctl.change(tr("Mark phase"), lambda: ctl.project.add_phase_at(f, name.strip()))


__all__ = ["TransportBar", "Timeline", "mark_phase", "PHASE_COLORS"]
