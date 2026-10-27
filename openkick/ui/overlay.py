"""Draws skeleton, measurements and notes over a frame.

Shared by the interactive canvas and the video exporter, so what you see is
exactly what gets exported. Everything is drawn in target (widget or image)
coordinates through a View, so line widths stay constant when zooming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core.project import LOW_CONFIDENCE, SRC_MANUAL, Measurement, Project
from openkick.ui import theme


@dataclass
class View:
    scale: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    def map(self, x: float, y: float) -> QPointF:
        return QPointF(float(self.ox + x * self.scale), float(self.oy + y * self.scale))

    def inv(self, px: float, py: float) -> tuple[float, float]:
        return (float((px - self.ox) / self.scale), float((py - self.oy) / self.scale))


@dataclass
class Options:
    layers: dict = field(default_factory=lambda: {"skeleton": True, "angles": True, "trajectories": True,
                                                  "distances": True, "notes": True})
    selected_landmark: int | None = None
    hover_landmark: int | None = None
    selected_measure: str | None = None
    k: float = 1.0                    # size factor (1 on screen, larger for high-res export)
    show_markers: bool = True          # manual / low-confidence hints (off in exports)
    all_points: bool = False           # draw all 33 MediaPipe landmarks (face and hand details too)
    ghost: bool = False                # faint skeleton of the previous and next frame (to line up corrections)
    hits: dict = field(default_factory=dict)   # filled while painting: measurement id -> label rect
    placed: list = field(default_factory=list)  # label rects already drawn this frame (to avoid overlaps)
    deferred: list | None = None                 # labels to paint on top of everything


SIDE_COLOR = {"R": theme.RIGHT, "L": theme.LEFT, "C": theme.CENTER}
GHOST_PREV, GHOST_NEXT = "#FBBF24", "#38BDF8"     # previous frame: amber; next frame: sky blue


def _pen(color, width, style=Qt.SolidLine, alpha=None):
    c = QColor(color)
    if alpha is not None:
        c.setAlpha(alpha)
    pen = QPen(c, width, style, Qt.RoundCap, Qt.RoundJoin)
    return pen


def chip(p: QPainter, anchor: QPointF, text: str, bg: str, k: float = 1.0, align: str = "center",
         avoid: list | None = None, deferred: list | None = None):
    """Rounded value label (e.g. "83°"). Labels listed in `avoid` are not overlapped."""
    f = theme.mono(12 * k, 600)
    p.setFont(f)
    fm = QFontMetricsF(f)
    w = fm.horizontalAdvance(text) + 16 * k
    h = 22 * k
    x = anchor.x() - w / 2 if align == "center" else anchor.x()
    r = QRectF(x, anchor.y() - h / 2, w, h)
    if avoid is not None:
        base = QRectF(r)
        for i in range(1, 12):
            if not any(r.adjusted(-2 * k, -2 * k, 2 * k, 2 * k).intersects(o) for o in avoid):
                break
            step = (h + 4 * k) * ((i + 1) // 2) * (1 if i % 2 else -1)
            r = base.translated(0, step)
        avoid.append(QRectF(r))
    if deferred is not None:
        deferred.append((QRectF(r), text, bg, k))
        return r
    _draw_chip(p, r, text, bg, k)
    return r


def _draw_chip(p: QPainter, r: QRectF, text: str, bg: str, k: float):
    p.setFont(theme.mono(12 * k, 600))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(bg))
    p.drawRoundedRect(r, 6 * k, 6 * k)
    p.setPen(QColor(theme.text_on(bg)))
    p.drawText(r, Qt.AlignCenter, text)


def paint(p: QPainter, project: Project, frame: int, view: View, opts: Options):
    p.setRenderHint(QPainter.Antialiasing)
    opts.placed = []
    opts.deferred = []
    try:
        _paint(p, project, frame, view, opts)
    finally:
        for r, text, bg, k in opts.deferred:
            if text is not None:
                _draw_chip(p, r, text, bg, k)
        for r, text, bg, k in opts.deferred:      # selection outlines on top of their label
            if text is None:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(theme.ACCENT), 1.5 * k))
                p.drawRoundedRect(r.adjusted(-3 * k, -3 * k, 3 * k, 3 * k), 8 * k, 8 * k)
        opts.deferred = None


def _paint(p: QPainter, project: Project, frame: int, view: View, opts: Options):
    layers = opts.layers
    if layers.get("trajectories", True):
        for m in project.measurements:
            if m.kind == "trajectory" and m.visible:
                _trajectory(p, project, m, frame, view, opts)
    if layers.get("skeleton", True) and project.pose.has_data:
        if opts.ghost:
            _ghosts(p, project, frame, view, opts)
        _skeleton(p, project, frame, view, opts)
    if layers.get("angles", True):
        for m in project.measurements:
            if m.kind == "angle" and m.visible:
                _angle(p, project, m, frame, view, opts)
    if layers.get("distances", True):
        for m in project.measurements:
            if m.kind == "distance" and m.visible:
                _distance(p, project, m, frame, view, opts)
    if layers.get("markers", True):
        for m in project.measurements:
            if m.kind == "point" and m.visible:
                _marker(p, project, m, frame, view, opts)
    if layers.get("notes", True):
        for n in project.notes:
            a = n.opacity(project.video, frame)
            if a > 0:
                p.save()
                p.setOpacity(a)
                note_rect(p, n, view, opts.k, draw=True)
                p.restore()


def _ghosts(p: QPainter, project: Project, frame: int, view: View, opts: Options):
    """The skeleton of the previous and the next frame, faint, under the current one: where
    every point comes from and where it goes. For the selected (or hovered) joint the path
    previous -> current -> next is drawn, so a correction can be placed in line with its neighbours."""
    k = opts.k
    pose = project.pose
    joints = range(L.COUNT) if opts.all_points else L.JOINTS
    focus = opts.selected_landmark if opts.selected_landmark is not None else opts.hover_landmark
    cur = pose.xy[frame]
    for f, color in ((frame - 1, GHOST_PREV), (frame + 1, GHOST_NEXT)):
        if not 0 <= f < pose.frame_count:
            continue
        xy = pose.xy[f]
        p.setPen(_pen(color, 1.6 * k, alpha=135))
        for a, b in L.BONES:
            if np.isfinite(xy[a]).all() and np.isfinite(xy[b]).all():
                p.drawLine(view.map(*xy[a]), view.map(*xy[b]))
        if np.isfinite(xy[[0, 11, 12]]).all():
            p.drawLine(view.map(*xy[0]), view.map(*((xy[11] + xy[12]) / 2)))
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(color, 190))
        for j in joints:
            if np.isfinite(xy[j]).all():
                p.drawEllipse(view.map(*xy[j]), 2.2 * k, 2.2 * k)
        if focus is not None and np.isfinite(xy[focus]).all():
            c = view.map(*xy[focus])
            if np.isfinite(cur[focus]).all():
                p.setPen(_pen(color, 1.6 * k, Qt.DashLine, alpha=230))
                p.drawLine(c, view.map(*cur[focus]))
            p.setPen(QPen(QColor(theme.BG), 1.2 * k))
            p.setBrush(QColor(color))
            p.drawEllipse(c, 4.2 * k, 4.2 * k)


def _skeleton(p: QPainter, project: Project, frame: int, view: View, opts: Options):
    k = opts.k
    xy = project.pose.xy[frame]
    vis = project.pose.visibility[frame]
    src = project.pose.source[frame]
    for a, b in L.BONES:
        if not (np.isfinite(xy[a]).all() and np.isfinite(xy[b]).all()):
            continue
        sa, sb = L.side(a), L.side(b)
        color = SIDE_COLOR[sa if sa == sb else "C"]
        alpha = 255 if min(vis[a], vis[b]) >= LOW_CONFIDENCE else 130
        p.setPen(_pen(color, 2.5 * k, alpha=alpha))
        p.drawLine(view.map(*xy[a]), view.map(*xy[b]))
    # neck line: nose to mid-shoulders
    if np.isfinite(xy[[0, 11, 12]]).all():
        mid = (xy[11] + xy[12]) / 2
        p.setPen(_pen(theme.CENTER, 2.5 * k))
        p.drawLine(view.map(*xy[0]), view.map(*mid))
    if opts.all_points:
        p.setPen(QPen(QColor(theme.BG), 1 * k))
        for j in range(L.COUNT):
            if j in L.JOINTS or not np.isfinite(xy[j]).all():
                continue
            p.setBrush(QColor(SIDE_COLOR[L.side(j)]))
            p.drawEllipse(view.map(*xy[j]), 2.6 * k, 2.6 * k)
    for j in L.JOINTS:
        if not np.isfinite(xy[j]).all():
            continue
        c = view.map(*xy[j])
        color = QColor(SIDE_COLOR[L.side(j)])
        low = vis[j] < LOW_CONFIDENCE
        r = (4.5 if j in (23, 24, 25, 26) else 4.0) * k
        if j in (19, 20, 7, 8):
            r = 3.0 * k
        p.setPen(QPen(QColor(theme.BG), 1.5 * k))
        if low and src[j] != SRC_MANUAL:
            p.setBrush(QColor(theme.BG))
            p.setPen(QPen(color, 1.8 * k))
        else:
            p.setBrush(color)
        p.drawEllipse(c, r, r)
        if opts.show_markers and src[j] == SRC_MANUAL:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(theme.ACCENT), 1.4 * k))
            p.drawEllipse(c, r + 3 * k, r + 3 * k)
        if j == opts.hover_landmark and j != opts.selected_landmark:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 170), 1.5 * k))
            p.drawEllipse(c, r + 5 * k, r + 5 * k)
        if j == opts.selected_landmark:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(theme.ACCENT), 2 * k))
            p.drawEllipse(c, r + 6 * k, r + 6 * k)


def _qt_angle(o: QPointF, q: QPointF) -> float:
    return math.degrees(math.atan2(-(q.y() - o.y()), q.x() - o.x()))


def _angle(p: QPainter, project: Project, m: Measurement, frame: int, view: View, opts: Options):
    pts = [project.point(r, frame) for r in m.points]
    if any(pt is None for pt in pts):
        return
    k = opts.k
    A_, O, B = (view.map(*pt) for pt in pts)
    la = math.hypot(A_.x() - O.x(), A_.y() - O.y())
    lb = math.hypot(B.x() - O.x(), B.y() - O.y())
    if la < 1 or lb < 1:
        return
    selected = m.id == opts.selected_measure
    color = QColor(m.color)
    free = not all(r.is_landmark for r in m.points)
    if free or selected:
        p.setPen(_pen(m.color, (2.5 if selected else 2) * k))
        p.drawLine(O, A_)
        p.drawLine(O, B)
    r = min(28 * k, 0.45 * min(la, lb))
    a1, a2 = _qt_angle(O, A_), _qt_angle(O, B)
    sweep = (a2 - a1 + 540) % 360 - 180
    path = QPainterPath(O)
    path.arcTo(QRectF(O.x() - r, O.y() - r, 2 * r, 2 * r), a1, sweep)
    path.closeSubpath()
    fill = QColor(color)
    fill.setAlpha(80 if selected else 62)
    p.setBrush(fill)
    p.setPen(_pen(m.color, 1.5 * k))
    p.drawPath(path)
    if free:
        p.setPen(QPen(QColor(theme.BG), 1.2 * k))
        p.setBrush(color)
        for q in (A_, O, B):
            p.drawEllipse(q, 3.5 * k, 3.5 * k)
    # label along the bisector, outside the arc
    ua = ((A_.x() - O.x()) / la, (A_.y() - O.y()) / la)
    ub = ((B.x() - O.x()) / lb, (B.y() - O.y()) / lb)
    bx, by = ua[0] + ub[0], ua[1] + ub[1]
    n = math.hypot(bx, by)
    if n < 1e-3:
        bx, by = -ua[1], ua[0]
        n = 1.0
    d = r + 22 * k
    anchor = QPointF(O.x() + bx / n * d, O.y() + by / n * d)
    v = A.value_at(project, m, frame)
    rect = chip(p, anchor, A.format_value(v, "°"), m.color, k, avoid=opts.placed, deferred=opts.deferred)
    opts.hits[m.id] = rect
    if selected:
        opts.deferred.append((rect, None, None, k))


def _trajectory(p: QPainter, project: Project, m: Measurement, frame: int, view: View, opts: Options):
    k = opts.k
    s = A.series(project, m)
    xy = s.xy
    ok = np.isfinite(xy).all(axis=1)
    if not ok.any():
        return
    selected = m.id == opts.selected_measure
    past, future = QPainterPath(), QPainterPath()
    started_p = started_f = False
    for i in range(len(xy)):
        if not ok[i]:
            started_p = started_f = False
            continue
        q = view.map(*xy[i])
        if i <= frame:
            if started_p:
                past.lineTo(q)
            else:
                past.moveTo(q)
                started_p = True
        if i >= frame:
            if started_f:
                future.lineTo(q)
            else:
                future.moveTo(q)
                started_f = True
    p.save()
    p.setOpacity(m.opacity)
    p.setBrush(Qt.NoBrush)
    p.setPen(_pen(m.color, 2 * k, Qt.DashLine, alpha=120))
    p.drawPath(future)
    p.setPen(_pen(m.color, (3 if selected else 2.5) * k))
    p.drawPath(past)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(m.color))
    for i in range(0, min(frame, len(xy) - 1) + 1):
        if ok[i]:
            p.drawEllipse(view.map(*xy[i]), 2.2 * k, 2.2 * k)
    if ok[frame]:
        c = view.map(*xy[frame])
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(m.color), 2 * k))
        p.drawEllipse(c, 7 * k, 7 * k)
    p.restore()
    # label chip at full opacity so it's always readable
    if ok[frame]:
        c = view.map(*xy[frame])
        v = s.value[frame]
        label = A.format_value(float(v), s.unit) if s.unit == "m/s" and np.isfinite(v) else m.name
        rect = chip(p, QPointF(c.x() - 14 * k, c.y() - 26 * k), label, m.color, k, avoid=opts.placed, deferred=opts.deferred)
        opts.hits[m.id] = rect
        if selected:
            opts.deferred.append((rect, None, None, k))


def _distance(p: QPainter, project: Project, m: Measurement, frame: int, view: View, opts: Options):
    pts = [project.point(r, frame) for r in m.points]
    if any(pt is None for pt in pts):
        return
    k = opts.k
    P, Q = view.map(*pts[0]), view.map(*pts[1])
    selected = m.id == opts.selected_measure
    p.setPen(_pen(m.color, (2.5 if selected else 2) * k))
    p.drawLine(P, Q)
    dx, dy = Q.x() - P.x(), Q.y() - P.y()
    n = math.hypot(dx, dy) or 1
    nx, ny = -dy / n * 6 * k, dx / n * 6 * k
    for c in (P, Q):
        p.drawLine(QPointF(c.x() - nx, c.y() - ny), QPointF(c.x() + nx, c.y() + ny))
    s = A.series(project, m)
    mid = QPointF((P.x() + Q.x()) / 2 + nx * 3, (P.y() + Q.y()) / 2 + ny * 3)
    opts.hits[m.id] = chip(p, mid, A.format_value(float(s.value[frame]), s.unit), m.color, k, avoid=opts.placed, deferred=opts.deferred)


def note_rect(p: QPainter | None, n, view: View, k: float = 1.0, draw: bool = False) -> QRectF:
    f = theme.font(13 * k, 500)
    fm = QFontMetricsF(f)
    lines = n.text.split("\n")
    w = max(fm.horizontalAdvance(t) for t in lines) + 20 * k
    h = fm.height() * len(lines) + 12 * k
    c = view.map(n.x, n.y)
    r = QRectF(c.x(), c.y(), w, h)
    if draw and p is not None:
        p.setPen(QPen(QColor(n.color), 1.2 * k))
        p.setBrush(QColor(11, 13, 17, 215))
        p.drawRoundedRect(r, 8 * k, 8 * k)
        p.setFont(f)
        p.setPen(QColor(n.color))
        p.drawText(r.adjusted(10 * k, 6 * k, -10 * k, -6 * k), Qt.AlignLeft | Qt.AlignVCenter, n.text)
    return r


def _marker(p: QPainter, project: Project, m: Measurement, frame: int, view: View, opts: Options):
    """User marker: crosshair (free / on a segment) or the center-of-mass symbol."""
    ref = m.points[0]
    pt = project.point(ref, frame)
    if pt is None:
        return
    k = opts.k
    c = view.map(*pt)
    selected = m.id == opts.selected_measure
    color = QColor(m.color)
    if ref.kind == "segment" and selected:
        a, b, al, _ = ref.segment
        pa, pb = project.pose.point(frame, a), project.pose.point(frame, b)
        if pa and pb:
            foot = view.map(pa[0] + al * (pb[0] - pa[0]), pa[1] + al * (pb[1] - pa[1]))
            p.setPen(_pen(m.color, 1.2 * k, Qt.DotLine))
            p.drawLine(c, foot)
    r = 7 * k
    if ref.kind == "derived":
        # classic CoM symbol: circle with two opposite filled quadrants
        p.setPen(QPen(QColor(theme.BG), 1.5 * k))
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(c, r, r)
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        p.drawPie(rect, 0, 90 * 16)
        p.drawPie(rect, 180 * 16, 90 * 16)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(color, 1.6 * k))
        p.drawEllipse(c, r, r)
    else:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(theme.BG), 4 * k))
        p.drawEllipse(c, r, r)
        p.setPen(QPen(color, 2 * k))
        p.drawEllipse(c, r, r)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(c.x() + dx * r * 0.45, c.y() + dy * r * 0.45),
                       QPointF(c.x() + dx * r * 1.6, c.y() + dy * r * 1.6))
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawEllipse(c, 1.8 * k, 1.8 * k)
    if selected:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(theme.ACCENT), 1.6 * k))
        p.drawEllipse(c, r + 5 * k, r + 5 * k)
    rect = chip(p, QPointF(c.x() + r + 8 * k, c.y() - r - 6 * k), m.name, m.color, k, align="left",
                avoid=opts.placed, deferred=opts.deferred)
    opts.hits[m.id] = rect
