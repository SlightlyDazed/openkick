"""Interactive video canvas: shows the frame with overlays and implements the tools."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QInputDialog, QMenu, QWidget

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core.project import Calibration, Measurement, PointRef, TextNote
from openkick.i18n import tr
from openkick.ui import overlay, theme
from openkick.ui.controller import Controller

HIT_RADIUS = 13


class VideoCanvas(QWidget):
    message = Signal(str)          # short hint for the context bar
    trackRequested = Signal(str, int)   # marker/free-point measurement id, direction (+1 / -1)

    def __init__(self, ctl: Controller, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 200)
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.hover_lm: int | None = None
        self.mouse = QPointF()
        self.drag = None
        self.draft: list[tuple[PointRef, tuple[float, float]]] = []
        self.opts = overlay.Options()
        self.ghost = QSettings().value("ghost", True, type=bool)   # shadow of the neighbouring frames
        self.playing = False
        ctl.playingChanged.connect(self._playing)
        ctl.frameChanged.connect(lambda _: self.update())
        ctl.changed.connect(self.update)
        ctl.selectionChanged.connect(self.update)
        ctl.projectLoaded.connect(self.fit)
        ctl.toolChanged.connect(self._tool_changed)

    def set_ghost(self, on: bool):
        self.ghost = bool(on)
        QSettings().setValue("ghost", self.ghost)
        self.update()

    def _playing(self, on: bool):
        self.playing = on
        self.update()

    @property
    def ghost_shown(self) -> bool:
        """The shadows only help while correcting: with the select tool and the video stopped."""
        p = self.ctl.project
        return bool(self.ghost and not self.playing and self.ctl.tool == "select" and p and p.pose.has_data
                    and self.ctl.layers.get("skeleton", True))

    # ── geometry ───────────────────────────────────────────
    def view(self) -> overlay.View:
        v = self.ctl.project.video
        base = min(self.width() / v.width, self.height() / v.height) * 0.96
        s = base * self.zoom
        ox = (self.width() - v.width * s) / 2 + self.pan.x()
        oy = (self.height() - v.height * s) / 2 + self.pan.y()
        return overlay.View(s, ox, oy)

    def fit(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def zoom_at(self, factor: float, at: QPointF | None = None):
        if not self.ctl.project:
            return
        at = at or QPointF(self.width() / 2, self.height() / 2)
        v = self.view()
        ix, iy = v.inv(at.x(), at.y())
        self.zoom = max(0.25, min(12.0, self.zoom * factor))
        v2 = self.view()
        p2 = v2.map(ix, iy)
        self.pan += at - p2
        self.update()

    def athlete_box(self) -> QRectF | None:
        """Bounding box of the skeleton over the whole clip (robust percentiles)."""
        p = self.ctl.project
        xy = p.pose.xy[:, [j for j in L.JOINTS if j >= 11], :].reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 10:
            return None
        x0, y0 = np.percentile(xy, 1, axis=0)
        x1, y1 = np.percentile(xy, 99, axis=0)
        return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def focus_athlete(self):
        """Zoom so the athlete's whole movement fills the view."""
        p = self.ctl.project
        box = self.athlete_box() if p else None
        if box is None:
            self.fit()
            return
        v = p.video
        box = box.adjusted(-box.width() * 0.25 - 20, -box.height() * 0.18 - 20, box.width() * 0.25 + 20, box.height() * 0.18 + 20)
        base = min(self.width() / v.width, self.height() / v.height) * 0.96
        target = min(self.width() / box.width(), self.height() / box.height()) * 0.96
        self.zoom = max(1.0, min(8.0, target / base))
        self.pan = QPointF(0, 0)
        view = self.view()
        c = view.map(box.center().x(), box.center().y())
        self.pan = QPointF(self.width() / 2 - c.x(), self.height() / 2 - c.y())
        self.update()

    def athlete_is_small(self) -> bool:
        box = self.athlete_box()
        return box is not None and box.height() < 0.5 * self.ctl.project.video.height

    @property
    def zoom_percent(self) -> int:
        return round(self.zoom * 100)

    # ── hit testing ────────────────────────────────────────
    def joint_at(self, pos: QPointF) -> int | None:
        p = self.ctl.project
        if not p or not self.ctl.layers.get("skeleton", True):
            return None
        v = self.view()
        xy = p.pose.xy[self.ctl.frame]
        best, best_d = None, HIT_RADIUS
        for j in (range(L.COUNT) if self.opts.all_points else L.JOINTS):
            if not np.isfinite(xy[j]).all():
                continue
            q = v.map(*xy[j])
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < best_d:
                best, best_d = j, d
        return best

    def free_point_at(self, pos: QPointF):
        p = self.ctl.project
        v = self.view()
        for m in reversed(p.measurements):
            if not m.visible:
                continue
            for i, r in enumerate(m.points):
                if r.is_landmark:
                    continue
                pt = r.free_at(self.ctl.frame)
                if pt is None:
                    continue
                q = v.map(*pt)
                if math.hypot(q.x() - pos.x(), q.y() - pos.y()) < HIT_RADIUS:
                    return m, i
        return None

    def note_at(self, pos: QPointF):
        p = self.ctl.project
        v = self.view()
        for n in reversed(p.notes):
            if n.opacity(p.video, self.ctl.frame) > 0 and overlay.note_rect(None, n, v).contains(pos):
                return n
        return None

    def marker_at(self, pos: QPointF):
        p = self.ctl.project
        if not p or not self.ctl.layers.get("markers", True):
            return None
        v = self.view()
        for m in reversed(p.measurements):
            if m.kind != "point" or not m.visible:
                continue
            pt = p.point(m.points[0], self.ctl.frame)
            if pt is None:
                continue
            q = v.map(*pt)
            if math.hypot(q.x() - pos.x(), q.y() - pos.y()) < HIT_RADIUS + 2:
                return m
        return None

    def measure_label_at(self, pos: QPointF) -> str | None:
        for mid, rect in self.opts.hits.items():
            if rect.adjusted(-4, -4, 4, 4).contains(pos):
                return mid
        return None

    # ── painting ───────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.STAGE))
        project = self.ctl.project
        if not project:
            return
        v = self.view()
        img = self.ctl.image()
        target = QRectF(v.ox, v.oy, project.video.width * v.scale, project.video.height * v.scale)
        p.setRenderHint(QPainter.SmoothPixmapTransform, v.scale < 3)
        p.drawImage(target, img)
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(target)
        self.opts.layers = self.ctl.layers
        self.opts.selected_landmark = self.ctl.selected_landmark
        self.opts.selected_measure = self.ctl.selected_measure
        self.opts.hover_landmark = self.hover_lm
        self.opts.ghost = self.ghost_shown
        self.opts.hits = {}
        overlay.paint(p, project, self.ctl.frame, v, self.opts)
        self._paint_tool(p, v)
        self._paint_hud(p, target)

    def _paint_tool(self, p: QPainter, v: overlay.View):
        tool = self.ctl.tool
        project = self.ctl.project
        if tool == "calibrate" and project.calibration and not self.draft:
            c = project.calibration
            a, b = v.map(*c.a), v.map(*c.b)
            p.setPen(QPen(QColor(theme.ACCENT), 2, Qt.DashLine))
            p.drawLine(a, b)
            overlay.chip(p, QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2 - 18),
                         f"{c.length_m:.2f} m", theme.ACCENT)
        if not self.draft:
            return
        pts = [v.map(*xy) for _, xy in self.draft]
        p.setPen(QPen(QColor(theme.ACCENT), 1.8, Qt.DashLine))
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        p.drawLine(pts[-1], self.mouse)
        p.setPen(QPen(QColor(theme.BG), 1.5))
        p.setBrush(QColor(theme.ACCENT))
        for q in pts:
            p.drawEllipse(q, 4.5, 4.5)

    def _pill(self, p: QPainter, x: float, y: float, text: str, dot: str | None = None, right=False):
        f = theme.mono(12) if dot is None else theme.font(12)
        p.setFont(f)
        fm = QFontMetricsF(f)
        w = fm.horizontalAdvance(text) + 20 + (14 if dot else 0)
        r = QRectF(x - w if right else x, y, w, 26)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(11, 13, 17, 205))
        p.drawRoundedRect(r, 6, 6)
        tx = r.x() + 10
        if dot:
            p.setBrush(QColor(dot))
            p.drawEllipse(QPointF(tx + 3, r.center().y()), 3, 3)
            tx += 14
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(tx, r.y(), w, 26), Qt.AlignVCenter | Qt.AlignLeft, text)
        return r

    def _paint_hud(self, p: QPainter, target: QRectF):
        project = self.ctl.project
        f = self.ctl.frame
        left = max(target.left(), 0) + 12
        top = max(target.top(), 0) + 12
        r = self._pill(p, left, top, tr("Frame {n} · {t}", n=f + 1, t=timecode(project.video.time_of(f))))
        ph = project.phase_at(f)
        sph = project.subphase_at(f) if hasattr(project, "subphase_at") else None
        if ph:
            p.setFont(theme.font(12, 600))
            fm = QFontMetricsF(p.font())
            if sph:
                text = f"{ph.name} · {sph.name} · {project.video.time_of(sph.start):.2f}–{project.video.time_of(sph.end):.2f} s"
                c = sph.color
            else:
                text = f"{ph.name} · {project.video.time_of(ph.start):.2f}–{project.video.time_of(ph.end):.2f} s"
                c = ph.color
            w = fm.horizontalAdvance(text) + 20
            pr = QRectF(r.right() + 8, top, w, 26)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c))
            p.drawRoundedRect(pr, 6, 6)
            p.setPen(QColor(theme.text_on(c)))
            p.drawText(pr, Qt.AlignCenter, text)
        elif project.phases:
            self._pill(p, r.right() + 8, top, tr("{n} phases", n=len(project.phases)), theme.ACCENT)
        right = min(target.right(), self.width()) - 12
        if self.ghost_shown:                      # what the two faint skeletons are
            bottom = min(target.bottom(), self.height()) - 38
            x = left
            for text, color, there in ((tr("previous frame"), overlay.GHOST_PREV, f > 0),
                                       (tr("next frame"), overlay.GHOST_NEXT, f < project.frame_count - 1)):
                if there:
                    x = self._pill(p, x, bottom, text, color).right() + 6
        if not project.pose.has_data:
            self._pill(p, right, top, tr("No skeleton yet · Analysis › Detect pose"), theme.DIM, right=True)
            return
        conf = A.frame_confidence(project)[f]
        if not np.isfinite(conf):
            self._pill(p, right, top, tr("No person detected in this frame"), theme.WARN, right=True)
        else:
            color = theme.OK if conf >= 0.75 else theme.WARN
            self._pill(p, right, top, tr("MediaPipe · confidence {v}%", v=round(conf * 100)), color, right=True)

    # ── interaction ────────────────────────────────────────
    def _tool_changed(self, tool: str):
        self.draft = []
        cursors = {"select": Qt.ArrowCursor, "zoom": Qt.PointingHandCursor}
        self.setCursor(cursors.get(tool, Qt.CrossCursor))
        self.update()

    def cancel(self):
        self.draft = []
        self.drag = None
        self.update()

    def wheelEvent(self, e):
        if not self.ctl.project:
            return
        steps = e.angleDelta().y() / 120
        self.zoom_at(1.15 ** steps, e.position())
        self.message.emit("")

    def mouseDoubleClickEvent(self, e):
        if self.ctl.tool == "select" and self.joint_at(e.position()) is None and not self.note_at(e.position()):
            self.fit()

    def mousePressEvent(self, e):
        project = self.ctl.project
        if not project:
            return
        pos = e.position()
        self.setFocus()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.ShiftModifier
                                              and self.ctl.tool == "select"):
            self.drag = ("pan", pos, QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.RightButton:
            self._context_menu(pos, e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton:
            return
        tool = self.ctl.tool
        v = self.view()
        img_xy = v.inv(pos.x(), pos.y())
        if tool == "select":
            j = self.joint_at(pos)
            if j is not None:
                self.ctl.select_landmark(j)
                old = tuple(project.pose.xy[self.ctl.frame, j])
                self.drag = ("joint", j, old, pos)
                self.setCursor(Qt.ClosedHandCursor)
                return
            mk = self.marker_at(pos)
            if mk:
                self.ctl.select_measure(mk.id)
                if mk.points[0].kind == "derived":
                    self.message.emit(tr("The center of mass is calculated from the skeleton: correct the joints instead"))
                else:
                    self.drag = ("marker", mk.id, self.ctl.begin())
                    self.setCursor(Qt.ClosedHandCursor)
                return
            fp = self.free_point_at(pos)
            if fp:
                m, i = fp
                self.ctl.select_measure(m.id)
                self.drag = ("free", m.id, i, self.ctl.begin())
                return
            mid = self.measure_label_at(pos)
            if mid:
                self.ctl.select_measure(mid)
                return
            n = self.note_at(pos)
            if n:
                self.drag = ("note", n.id, (img_xy[0] - n.x, img_xy[1] - n.y), self.ctl.begin())
                return
            self.ctl.select_landmark(None)
            self.ctl.select_measure(None)
            self.drag = ("pan", pos, QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)
        elif tool == "zoom":
            self.zoom_at(1 / 1.6 if e.modifiers() & Qt.AltModifier else 1.6, pos)
        elif tool == "marker":
            self._add_marker(img_xy, pos)
        elif tool in ("angle", "trajectory", "distance", "calibrate"):
            j = self.joint_at(pos) if tool != "calibrate" else None
            mk = self.marker_at(pos) if (tool != "calibrate" and j is None) else None
            if mk is not None:
                ref = PointRef(marker=mk.id)
                xy = project.point(mk.points[0], self.ctl.frame)
            else:
                ref = PointRef.at(j) if j is not None else PointRef.free(self.ctl.frame, *img_xy)
                xy = tuple(project.pose.xy[self.ctl.frame, j]) if j is not None else img_xy
            custom = bool(e.modifiers() & Qt.ShiftModifier)
            if tool == "angle" and not self.draft and j is not None and j in L.ANGLE_PRESETS and not custom:
                self._add_preset_angle(j)
                return
            if tool == "trajectory":
                if mk is not None:
                    self._follow_marker(mk)
                else:
                    self._add_trajectory(ref, j)
                return
            self.draft.append((ref, xy))
            need = 3 if tool == "angle" else 2
            if len(self.draft) >= need:
                refs = [r for r, _ in self.draft]
                pts = [xy for _, xy in self.draft]
                self.draft = []
                if tool == "angle":
                    self._add_measure("angle", refs, self._angle_name(refs))
                elif tool == "distance":
                    self._add_measure("distance", refs, tr("Distance"))
                else:
                    self._calibrate(pts)
            else:
                hints = {"angle": [tr("Now click the vertex of the angle"), tr("Now click the other end")],
                         "distance": [tr("Click the second point")], "calibrate": [tr("Click the other end of the known length")]}
                self.message.emit(hints[tool][len(self.draft) - 1])
            self.update()
        elif tool == "text":
            new_note(self, self.ctl, img_xy)

    def mouseMoveEvent(self, e):
        project = self.ctl.project
        if not project:
            return
        pos = e.position()
        self.mouse = pos
        v = self.view()
        if self.drag:
            kind = self.drag[0]
            if kind == "pan":
                self.pan = self.drag[2] + (pos - self.drag[1])
            elif kind == "joint":
                x, y = v.inv(pos.x(), pos.y())
                project.pose.xy[self.ctl.frame, self.drag[1]] = (x, y)
                project.touch()
                moved = math.hypot(pos.x() - self.drag[3].x(), pos.y() - self.drag[3].y())
                self.message.emit(tr("{joint} · moved {px} px on screen",
                                     joint=tr(L.DISPLAY[L.NAMES[self.drag[1]]]), px=round(moved)))
            elif kind == "free":
                m = project.measurement(self.drag[1])
                if m:
                    m.points[self.drag[2]].keys[self.ctl.frame] = v.inv(pos.x(), pos.y())
                    project.touch()
            elif kind == "note":
                n = next((n for n in project.notes if n.id == self.drag[1]), None)
                if n:
                    x, y = v.inv(pos.x(), pos.y())
                    n.x, n.y = x - self.drag[2][0], y - self.drag[2][1]
            elif kind == "marker":
                m = project.measurement(self.drag[1])
                if m:
                    self._move_marker(m, v.inv(pos.x(), pos.y()))
                    project.touch()
            self.update()
            return
        j = self.joint_at(pos) if self.ctl.tool in ("select", "angle", "trajectory", "distance") else None
        if j != self.hover_lm:
            self.hover_lm = j
            if j is not None:
                self.setToolTip(tr(L.DISPLAY[L.NAMES[j]]))
            else:
                self.setToolTip("")
        if self.ctl.tool == "select":
            over = j is not None or self.marker_at(pos) or self.free_point_at(pos) or self.note_at(pos)
            self.setCursor(Qt.OpenHandCursor if over else Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        project = self.ctl.project
        if not project or not self.drag:
            return
        kind = self.drag[0]
        drag, self.drag = self.drag, None
        if kind == "joint":
            _, j, old, start = drag
            f = self.ctl.frame
            new = tuple(float(c) for c in project.pose.xy[f, j])
            if math.hypot(e.position().x() - start.x(), e.position().y() - start.y()) < 1.5:
                project.pose.xy[f, j] = old
                project.touch()
            else:
                edit = A.edit_point(project.pose, f, j, new, scope=self.ctl.edit_scope,
                                    phase=project.phase_at(f), old=old)
                self.ctl.push_pose_edit(edit, tr("Move {joint}", joint=tr(L.DISPLAY[L.NAMES[j]])))
            self.setCursor(Qt.OpenHandCursor)
        elif kind == "free":
            self.ctl.commit(drag[3], tr("Move point"))
        elif kind == "note":
            self.ctl.commit(drag[3], tr("Move note"))
        elif kind == "marker":
            self.ctl.commit(drag[2], tr("Move marker"))
            self.setCursor(Qt.OpenHandCursor)
        elif kind == "pan":
            self.setCursor(Qt.ArrowCursor)
        self.message.emit("")
        self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.cancel()
            if self.ctl.tool != "select":
                self.ctl.set_tool("select")
            return
        super().keyPressEvent(e)

    # ── actions ────────────────────────────────────────────
    def _angle_name(self, refs: list[PointRef]) -> str:
        v = refs[1]
        if v.is_landmark:
            return tr(L.DISPLAY[L.NAMES[v.landmark]])
        return tr("Angle")

    def _add_measure(self, kind: str, refs: list[PointRef], name: str):
        project = self.ctl.project
        m = Measurement(kind=kind, name=name, points=refs, color=project.next_color())
        self.ctl.change(tr("Add measurement"), lambda: project.measurements.append(m))
        self.ctl.select_measure(m.id)
        self.message.emit("")

    def _add_preset_angle(self, j: int):
        a, o, b = L.ANGLE_PRESETS[j]
        project = self.ctl.project
        exists = next((m for m in project.measurements if m.kind == "angle"
                       and [r.landmark for r in m.points] == [L.INDEX[a], L.INDEX[o], L.INDEX[b]]), None)
        if exists:
            self.ctl.select_measure(exists.id)
            self.message.emit(tr("That angle is already measured"))
            return
        self._add_measure("angle", [PointRef.at(L.INDEX[a]), PointRef.at(j), PointRef.at(L.INDEX[b])],
                          tr(L.DISPLAY[o]))

    def _add_trajectory(self, ref: PointRef, j: int | None):
        project = self.ctl.project
        if j is not None:
            exists = next((m for m in project.measurements if m.kind == "trajectory"
                           and m.points[0].landmark == j), None)
            if exists:
                self.ctl.select_measure(exists.id)
                return
            name = tr(L.DISPLAY[L.NAMES[j]])
        else:
            name = tr("Free point")
            self.message.emit(tr("Free point: drag it in other frames to follow the object"))
        self._add_measure("trajectory", [ref], name)

    def _add_marker(self, img_xy, pos: QPointF):
        """Place a marker: on the body it is attached to the nearest segment (it will follow the
        skeleton); elsewhere it is a free marker (move it or track it automatically)."""
        from openkick.core.bodymodel import SEGMENTS, nearest_segment
        project = self.ctl.project
        f = self.ctl.frame
        n = sum(1 for m in project.measurements if m.kind == "point") + 1
        near = nearest_segment(project.pose.xy[f], img_xy) if project.pose.has_data else None
        if near and near[2] * self.view().scale < 28:
            a, b, _ = near
            ref = PointRef.on_segment(a, b, project.pose.point(f, a), project.pose.point(f, b), img_xy)
            seg = tr(SEGMENTS[(a, b)])
        else:
            ref = PointRef.free(f, *img_xy)
        self._add_measure("point", [ref], tr("Marker {n}", n=n))
        self.message.emit(tr("Marker attached to {seg}: it will follow the skeleton in every frame", seg=seg)
                          if ref.kind == "segment" else
                          tr("Free marker: drag it in other frames, or right-click › Track automatically"))

    def _move_marker(self, m: Measurement, xy):
        ref = m.points[0]
        f = self.ctl.frame
        if ref.kind == "segment":
            a, b, _, _ = ref.segment
            pa, pb = self.ctl.project.pose.point(f, a), self.ctl.project.pose.point(f, b)
            if pa and pb:
                ref.segment = PointRef.on_segment(a, b, pa, pb, xy).segment
        elif ref.kind == "free":
            ref.keys[f] = (float(xy[0]), float(xy[1]))

    def _follow_marker(self, mk: Measurement):
        project = self.ctl.project
        exists = next((m for m in project.measurements if m.kind == "trajectory"
                       and m.points[0].marker == mk.id), None)
        if exists:
            self.ctl.select_measure(exists.id)
            return
        self._add_measure("trajectory", [PointRef(marker=mk.id)], mk.name)

    def _calibrate(self, pts):
        project = self.ctl.project
        px = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        if px < 2:
            return
        cm, ok = QInputDialog.getDouble(self, tr("Calibrate"), tr("Real length of the segment (cm):"),
                                        100.0, 0.1, 100000.0, 1)
        if not ok:
            return
        cal = Calibration(px_per_m=px / (cm / 100), a=pts[0], b=pts[1], length_m=cm / 100)

        def apply():
            project.calibration = cal
        self.ctl.change(tr("Calibrate"), apply)
        self.message.emit(tr("Calibrated: 1 m = {px} px", px=round(cal.px_per_m)))

    def _context_menu(self, pos: QPointF, gpos):
        project = self.ctl.project
        menu = QMenu(self)
        j = self.joint_at(pos)
        mk = self.marker_at(pos) if j is None else None
        mid = self.measure_label_at(pos)
        n = self.note_at(pos)
        if mk is not None:
            marker_menu(self, self.ctl, mk, menu)
        elif j is not None:
            self.ctl.select_landmark(j)
            theme.menu_title(menu, tr(L.DISPLAY[L.NAMES[j]]))
            menu.addAction(tr("Revert to MediaPipe"), lambda: revert_joint(self.ctl, j))
            if j in L.ANGLE_PRESETS:
                menu.addAction(tr("Measure this angle"), lambda: self._add_preset_angle(j))
            menu.addAction(tr("Follow its trajectory"), lambda: self._add_trajectory(PointRef.at(j), j))
        elif mid:
            m = project.measurement(mid)
            self.ctl.select_measure(mid)
            theme.menu_title(menu, m.name)
            menu.addAction(tr("Rename…"), lambda: rename_measure(self, self.ctl, m))
            menu.addAction(tr("Delete"), lambda: delete_measure(self.ctl, m))
        elif n:
            menu.addAction(tr("Edit note…"), lambda: edit_note(self, self.ctl, n))
            menu.addAction(tr("Show only in this frame"), lambda: self._note_only_here(n))
            menu.addAction(tr("Delete note"), lambda: self.ctl.change(tr("Delete note"),
                                                                      lambda: project.notes.remove(n)))
        else:
            menu.addAction(tr("Fit to window"), self.fit)
        menu.exec(gpos)

    def _note_only_here(self, n):
        f = self.ctl.frame

        def apply():
            for x in self.ctl.project.notes:
                if x.id == n.id:
                    x.start = x.end = f
        self.ctl.change(tr("Edit note"), apply)


# ── helpers shared with panels ─────────────────────────────────────────


def timecode(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def revert_joint(ctl: Controller, j: int):
    edit = A.revert_point(ctl.project.pose, ctl.frame, j)
    ctl.push_pose_edit(edit, tr("Revert {joint}", joint=tr(L.DISPLAY[L.NAMES[j]])))


def rename_measure(parent, ctl: Controller, m: Measurement):
    name, ok = QInputDialog.getText(parent, tr("Rename"), tr("Name:"), text=m.name)
    if ok and name.strip():
        def apply():
            ctl.project.measurement(m.id).name = name.strip()
        ctl.change(tr("Rename"), apply)


def delete_measure(ctl: Controller, m: Measurement):
    """Delete a measurement; measurements built on a marker go with it."""
    gone = {m.id} | {d.id for d in ctl.project.dependents(m.id)}

    def apply():
        ctl.project.measurements = [x for x in ctl.project.measurements if x.id not in gone]
    ctl.change(tr("Delete measurement"), apply)
    ctl.select_measure(None)


def new_note(parent, ctl: Controller, xy):
    from openkick.ui.dialogs import NoteDialog
    p = ctl.project
    f0 = ctl.frame
    end = p.video.index_at_time(p.video.time_of(f0) + 2.0)
    draft = TextNote(text="", x=xy[0], y=xy[1], start=f0, end=max(f0, end), fade_out=0.5)
    dlg = NoteDialog(p.video, draft, f0, parent)
    if dlg.exec():
        for k, val in dlg.values().items():
            setattr(draft, k, val)
        ctl.change(tr("Add note"), lambda: p.notes.append(draft))


def edit_note(parent, ctl: Controller, n: TextNote):
    from openkick.ui.dialogs import NoteDialog
    dlg = NoteDialog(ctl.project.video, n, ctl.frame, parent)
    if dlg.exec():
        vals = dlg.values()

        def apply():
            for x in ctl.project.notes:
                if x.id == n.id:
                    for k, val in vals.items():
                        setattr(x, k, val)
        ctl.change(tr("Edit note"), apply)


def marker_menu(parent, ctl: Controller, m: Measurement, menu: QMenu):
    from openkick.core.bodymodel import SEGMENTS, nearest_segment
    ref = m.points[0]
    ctl.select_measure(m.id)
    theme.menu_title(menu, m.name)
    if ref.kind == "segment":
        seg = tr(SEGMENTS.get(ref.segment[:2], "segment"))
        a = menu.addAction(tr("Attached to {seg}", seg=seg))
        a.setEnabled(False)
        menu.addAction(tr("Make it a free marker"), lambda: _marker_to_free(ctl, m))
    elif ref.kind == "free":
        menu.addAction(tr("Track automatically forward"), lambda: parent.trackRequested.emit(m.id, 1))
        menu.addAction(tr("Track automatically backward"), lambda: parent.trackRequested.emit(m.id, -1))
        xy = ref.free_at(ctl.frame)
        near = nearest_segment(ctl.project.pose.xy[ctl.frame], xy) if xy and ctl.project.pose.has_data else None
        if near:
            menu.addAction(tr("Attach to {seg}", seg=tr(SEGMENTS[near[:2]])), lambda: _marker_to_segment(ctl, m, near[:2]))
    menu.addSeparator()
    menu.addAction(tr("Follow its trajectory"), lambda: parent._follow_marker(m))
    menu.addAction(tr("Rename…"), lambda: rename_measure(parent, ctl, m))
    menu.addAction(tr("Delete"), lambda: delete_measure(ctl, m))


def _marker_to_free(ctl: Controller, m: Measurement):
    """Bake the attached positions of every frame into free keys (then edit freely)."""
    p = ctl.project
    xy = A._points(p, m.points[0], p.frame_count)

    def apply():
        mm = p.measurement(m.id)
        mm.points[0] = PointRef(keys={i: (float(x), float(y)) for i, (x, y) in enumerate(xy)
                                      if np.isfinite(x) and np.isfinite(y)})
    ctl.change(tr("Make it a free marker"), apply)


def _marker_to_segment(ctl: Controller, m: Measurement, seg):
    p = ctl.project
    f = ctl.frame
    a, b = seg
    xy = p.point(m.points[0], f)

    def apply():
        p.measurement(m.id).points[0] = PointRef.on_segment(a, b, p.pose.point(f, a), p.pose.point(f, b), xy)
    ctl.change(tr("Attach marker"), apply)
