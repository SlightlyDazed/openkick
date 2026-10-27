"""3D from two or more videos (experimental). Step 1: the automatic reconstruction and its
correction by dragging skeleton points, as in the 2D editor, until the videos agree. Step 2:
the OpenSim model those points produce, next to them."""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QColor, QFontMetricsF, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut,
                           QUndoCommand, QUndoStack)
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QRadioButton,
                               QScrollArea, QSplitter, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from openkick import APP_NAME, multiview
from openkick.core.video import VideoReader
from openkick.i18n import tr
from openkick.multiview import geometry as G
from openkick.multiview.session import PROBLEM_CM, SRC_FOLLOW, SRC_LEARNED, SRC_MANUAL, STROKE_PARTS, Session
from openkick.ui import icons, theme

HIT_RADIUS = 13
VIDEO_FILTER = "Video (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv *.mpg *.mpeg *.MP4 *.MOV *.AVI)"
SIDE_COLOR = {"R": theme.RIGHT, "L": theme.LEFT, "C": theme.CENTER}

DISPLAY = {"nose": "Nose", "left_eye": "Left eye", "right_eye": "Right eye", "left_ear": "Left ear",
           "right_ear": "Right ear", "left_shoulder": "Left shoulder", "right_shoulder": "Right shoulder",
           "left_elbow": "Left elbow", "right_elbow": "Right elbow", "left_wrist": "Left wrist",
           "right_wrist": "Right wrist", "left_hip": "Left hip", "right_hip": "Right hip", "left_knee": "Left knee",
           "right_knee": "Right knee", "left_ankle": "Left ankle", "right_ankle": "Right ankle",
           "head": "Top of the head", "neck": "Neck", "hip": "Center of the pelvis", "left_toe": "Left toe",
           "right_toe": "Right toe", "left_small_toe": "Left little toe", "right_small_toe": "Right little toe",
           "left_heel": "Left heel", "right_heel": "Right heel"}
HANDLES = G.BODY       # points drawn as handles
MODES = (("balanced", "Balanced"), ("lightweight", "Fast"), ("performance", "Accurate (slow)"))
DRAW_BONES = [(G.IDX[a], G.IDX[b]) for a, b in G.BONES + [
    ("left_ankle", "left_toe"), ("right_ankle", "right_toe"), ("left_toe", "left_small_toe"),
    ("right_toe", "right_small_toe"), ("neck", "hip"), ("nose", "head")]]
SIDE = [G.side(n) for n in G.HALPE26]
GROUP_OF = {G.IDX[n]: g for g, pairs in G.GROUPS.items() if g != "face" for pair in pairs for n in pair}
LEVELS = {"good": ("Good reconstruction", theme.OK), "fair": ("Acceptable reconstruction", theme.WARN),
          "poor": ("Doubtful reconstruction", theme.DANGER)}


def joint_name(j: int) -> str:
    return tr(DISPLAY[G.HALPE26[j]])


# a thigh or shank off its usual length: what is too much, what is missing
LONG_COLOR, SHORT_COLOR = "#EF4444", "#FACC15"
BONE_NAMES = {("left_hip", "left_knee"): "Left thigh", ("left_knee", "left_ankle"): "Left shank",
              ("right_hip", "right_knee"): "Right thigh", ("right_knee", "right_ankle"): "Right shank"}


def bone_text(bones, tape=()) -> str:
    """"Right thigh 46.2 cm · usual 40.5 · +14 %" for each (a, b, dev, length, usual); "tape 43.5" for
    the bones in `tape` (their usual length is the one measured on the patient)."""
    return " · ".join(tr("{bone} {cm} cm · tape {usual} · {pct} %" if (a, b) in tape else "{bone} {cm} cm · usual {usual} · {pct} %",
                         bone=tr(BONE_NAMES[(a, b)]), cm=num(100 * L), usual=num(100 * u), pct=f"{100 * d:+.0f}")
                      for a, b, d, L, u in bones)


def timecode(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def num(value: float, digits: int = 1) -> str:
    from openkick.i18n import DECIMAL_COMMA, language
    s = f"{value:.{digits}f}"
    return s.replace(".", ",") if language() in DECIMAL_COMMA else s


# ── state shared by the widgets ────────────────────────────────────────


class _EditCommand(QUndoCommand):
    def __init__(self, ctl: "MultiviewController", edits: list, text: str):
        super().__init__(text)
        self.ctl, self.edits, self._first = ctl, edits, True

    def redo(self):
        if self._first:               # already applied when pushed
            self._first = False
            return
        for e in self.edits:
            e.apply(self.ctl.session, True)
        self.ctl.version += 1
        self.ctl.recompute(cameras=False)

    def undo(self):
        for e in reversed(self.edits):
            e.apply(self.ctl.session, False)
        self.ctl.version += 1
        self.ctl.recompute(cameras=False)


class MultiviewController(QObject):
    sessionLoaded = Signal()
    rowChanged = Signal(int)
    changed = Signal()                 # points or reconstruction changed
    selectionChanged = Signal()
    message = Signal(str)
    focusChanged = Signal()            # one video is being edited on its own, or all are back
    playingChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Session | None = None
        self.readers: list[VideoReader] = []
        self.row = 0
        self.selected: tuple[int, int] | None = None     # (view, joint)
        self.dragging: tuple[int, int] | None = None
        self.scope = "frame"
        self.follow = QSettings().value("multiview/follow", True, type=bool)   # the other videos follow a correction
        self.show_3d = True
        self.show_lengths = QSettings().value("multiview/lengths", True, type=bool)   # thigh and shank length guides
        self.focus: int | None = None      # the video being edited on its own (None: all of them)
        self.focus_frame = 0               # its current frame: it is stepped frame by frame, not row by row
        self.shadow = QSettings().value("multiview/shadow", True, type=bool)   # previous/next frame's skeleton
        self.playing = False
        self.version = 0                   # goes up with every correction (the OpenSim model follows it)
        self.saved_version = 0             # the version that is in the session file
        self.undo = QUndoStack(self)
        self._images: dict[tuple[int, int], QImage] = {}
        self._world, self._world_key = None, None
        self._gait, self._gait_key = None, None
        self._cycling, self._cycling_key = None, None

    def set_session(self, session: Session):
        self.close()
        self.session = session
        self.readers = [VideoReader(v.meta) for v in session.views]
        self.row, self.selected, self.dragging = 0, None, None
        self.focus = None
        self.version = self.saved_version = 0
        self.undo.clear()
        self.sessionLoaded.emit()
        self.rowChanged.emit(0)

    def activity_kind(self) -> str | None:
        """What the session analyses: the template chosen for it, or, when none was, pedalling or
        walking as the movement shows."""
        s = self.session
        if not s or not s.solved:
            return None
        if s.template in ("cycling", "gait"):
            return s.template
        if s.template and s.template != "other":
            return "template"
        if s.template:
            return None
        key = ("kind", id(s), s.revision)
        if getattr(self, "_kind_key", None) != key:
            from openkick.multiview.cycling import looks_like_pedalling
            try:
                self._kind = "cycling" if looks_like_pedalling(s.X, s.cams) else "gait"
            except Exception:  # noqa: BLE001
                self._kind = "gait"
            self._kind_key = key
        return self._kind

    def activity(self):
        """The cycles of what the session analyses (pedal strokes or strides, with their phases),
        or None. Found again after every correction."""
        kind = self.activity_kind()
        return self.cycling() if kind == "cycling" else self.gait() if kind == "gait" else None

    def template_result(self):
        """The template's angles in 3D (a session made for any template but cycling and gait)."""
        s = self.session
        if not s or not s.solved or self.activity_kind() != "template":
            return None
        from openkick.core import templates as TPL
        tpl = TPL.find(s.template)
        if tpl is None:
            return None
        from openkick.multiview.template3d import report
        return report(s.X, s.t, tpl)

    def cycling(self):
        """Pedal strokes of the reconstruction, or None when nobody pedals in it."""
        s = self.session
        if not s or not s.solved:
            return None
        key = (id(s), s.revision)
        if self._cycling_key != key:
            from openkick.multiview.cycling import analyse
            try:
                self._cycling = analyse(s.X, s.t, s.cams, s.quality())
            except Exception:  # noqa: BLE001 - a failed analysis must never get in the way of correcting
                self._cycling = None
            self._cycling_key = key
        return self._cycling

    def gait(self):
        """Strides and gait phases of the reconstruction, or None when nobody walks in it.
        Found again after every correction."""
        s = self.session
        if not s or not s.solved:
            return None
        key = (id(s), s.revision)
        if self._gait_key != key:
            from openkick.multiview.gait import analyse
            try:
                self._gait = analyse(s.X, s.t)
            except Exception:  # noqa: BLE001 - a failed gait analysis must never get in the way of correcting
                self._gait = None
            self._gait_key = key
        return self._gait

    def close(self):
        for r in self.readers:
            r.close()
        self.readers, self.session = [], None
        self.focus = None
        self._images.clear()
        self.undo.clear()

    def set_row(self, row: int):
        if not self.session or not self.session.frame_count:
            return
        row = max(0, min(int(row), self.session.frame_count - 1))
        moved = row != self.row
        self.row = row
        if self.focus is not None:
            f = self.session.frame_of(self.focus, row)
            moved |= f != self.focus_frame
            self.focus_frame = f
        if moved:
            self.rowChanged.emit(row)

    def focus_range(self) -> tuple[int, int]:
        """First and last frame of the video edited on its own that take part in the 3D."""
        s = self.session
        return s.frame_of(self.focus, 0), s.frame_of(self.focus, s.frame_count - 1)

    def step(self, delta: int):
        if self.focus is None:
            self.set_row(self.row + delta)
            return
        self.set_focus_frame(self.focus_frame + delta)      # one video on its own: every one of its frames

    def set_focus_frame(self, frame: int):
        """Go to an exact frame of the video edited on its own."""
        if self.focus is None:
            return
        lo, hi = self.focus_range()
        f = max(lo, min(hi, int(frame)))
        if f != self.focus_frame:
            self.focus_frame = f
            self.row = self.session.row_of(self.focus, f)
            self.rowChanged.emit(self.row)

    def frame(self, view: int) -> int:
        if view == self.focus:
            return self.focus_frame
        return self.session.frame_of(view, self.row)

    def set_focus(self, view: int | None):
        """Edit one video on its own (None: back to all of them)."""
        if view == self.focus or not self.session:
            return
        self.focus = view
        if view is not None:
            self.focus_frame = self.session.frame_of(view, self.row)
            if self.selected:
                self.selected = (view, self.selected[1])
        self.focusChanged.emit()
        self.selectionChanged.emit()
        self.rowChanged.emit(self.row)

    def set_shadow(self, on: bool):
        self.shadow = bool(on)
        QSettings().setValue("multiview/shadow", self.shadow)
        self.changed.emit()

    def set_playing(self, on: bool):
        self.playing = on
        self.playingChanged.emit(on)

    def image(self, view: int) -> QImage:
        f = self.frame(view)
        img = self._images.get((view, f))
        if img is None:
            arr = self.readers[view].frame(f)
            h, w = arr.shape[:2]
            img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self._images[(view, f)] = img
            if len(self._images) > 48:
                self._images.pop(next(iter(self._images)))
        return img

    @property
    def dirty(self) -> bool:
        """There are corrections that are not in the session file."""
        return self.session is not None and self.version != self.saved_version

    def world(self) -> tuple:
        """The 3D points in upright axes (Y up, floor at zero), their centre and their size (m)."""
        s = self.session
        key = (id(s), s.revision)
        if self._world_key != key:
            from openkick.multiview.export import to_world
            W = to_world(s.X)[0]
            pts = W.reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts):
                lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
                center, size = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)), 0.5)
            else:
                center, size = None, 1.0
            self._world, self._world_key = (W, center, size), key
        return self._world

    def select(self, sel: tuple[int, int] | None):
        self.selected = sel
        self.selectionChanged.emit()

    def recompute(self, cameras: bool = True):
        """Solve again after a change in the points."""
        s = self.session
        keep_t = s.t[self.row] if s.frame_count else 0.0
        s.update(recalibrate=cameras)
        self.row = int(np.abs(s.t - keep_t).argmin()) if s.frame_count else 0
        if self.focus is not None and s.frame_count:
            lo, hi = self.focus_range()
            self.focus_frame = max(lo, min(hi, self.focus_frame))
            self.row = s.row_of(self.focus, self.focus_frame)
        self.changed.emit()

    def push(self, edits: list, text: str):
        """A correction: the 3D follows it, the cameras stay where they are (moving them after every
        drag shifted every other point, and the other videos seemed to come apart). They are
        placed again only when asked (Place the cameras again)."""
        self.undo.push(_EditCommand(self, edits, text))
        self.version += 1
        self.recompute(cameras=False)

    def describe(self, row: int, j: int) -> str:
        """What is wrong with point `j` at `row`, in plain words."""
        s = self.session
        cm, bone, jump = s.err_cm[row, j], s.bone_cm[row, j], s.jump_cm[row, j]
        if not cm > PROBLEM_CM and jump > PROBLEM_CM and not bone > jump:
            return tr("{joint}: it jumps {cm} cm away from the frames before and after", joint=joint_name(j), cm=num(jump))
        if not cm > PROBLEM_CM and bone > PROBLEM_CM:
            return tr("{joint}: the bones that meet here are {cm} cm off their usual length", joint=joint_name(j), cm=num(bone))
        if np.isfinite(cm):
            return tr("{joint}: the videos disagree by {cm} cm", joint=joint_name(j), cm=num(cm))
        return joint_name(j)

    def next_problem(self):
        s = self.session
        r = s.next_problem(self.row)
        if r is None:
            self.message.emit(tr("No points left to check"))
            return
        worst = s.worst_joint(r)
        self.set_row(r)
        self.select((s.worst_view(r, worst) if self.focus is None else self.focus, worst))
        self.message.emit(self.describe(r, worst) + ". " + (
            tr("Look at it in every video and drag it onto the body where it is wrong") if self.focus is None else
            tr("If it is not on the body in this video, drag it to its place")))


# ── one video with its skeleton ────────────────────────────────────────


class ViewCanvas(QWidget):
    def __init__(self, ctl: MultiviewController, view: int, parent=None):
        super().__init__(parent)
        self.ctl, self.index = ctl, view
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(240, 200)
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.hover: int | None = None
        self.drag = None
        # Bound methods only (never lambdas): the canvases are replaced when another session is
        # opened, and a lambda would keep calling the old, deleted one.
        self._wired = [(ctl.rowChanged, self._refresh), (ctl.changed, self._refresh), (ctl.selectionChanged, self._refresh),
                       (ctl.sessionLoaded, self.focus_athlete), (ctl.playingChanged, self._refresh),
                       (ctl.focusChanged, self._focus_changed)]
        for signal, slot in self._wired:
            signal.connect(slot)
        self.solo_btn = QPushButton(self)
        self.solo_btn.setCursor(Qt.PointingHandCursor)
        self.solo_btn.setFocusPolicy(Qt.NoFocus)
        self.solo_btn.clicked.connect(self._toggle_solo)
        self._focus_changed()

    def detach(self):
        """Stop listening: this canvas is about to be replaced."""
        for signal, slot in self._wired:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._wired = []

    @property
    def live(self) -> bool:
        """Still showing a video of the open session."""
        s = self.ctl.session
        return bool(self._wired and s and self.index < len(s.views))

    def _refresh(self, *_):
        self.update()

    def _toggle_solo(self):
        self.ctl.set_focus(None if self.ctl.focus == self.index else self.index)

    @property
    def solo(self) -> bool:
        """This video is the one being edited on its own."""
        return self.ctl.focus == self.index

    def _focus_changed(self):
        on = self.solo
        self.solo_btn.setIcon(icons.icon("chevron_left" if on else "fit", theme.ON_ACCENT if on else theme.TEXT, 14))
        self.solo_btn.setText(" " + (tr("Back to all the videos") if on else tr("Edit this video")))
        self.solo_btn.setToolTip(tr("Back to the 3D view with every video (Esc)") if on else
                                 tr("Open this video on its own, large, to correct it frame by frame"))
        self.solo_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.ACCENT}; color: {theme.ON_ACCENT}; border: 0; border-radius: 13px; min-height: 26px;"
            f" padding: 0 12px; font-size: 12px; font-weight: 600; }} QPushButton:hover {{ background: {theme.ACCENT_HOVER}; }}" if on else
            f"QPushButton {{ background: rgba(11, 13, 17, 215); color: {theme.TEXT}; border: 1px solid {theme.BORDER}; border-radius: 13px;"
            f" min-height: 26px; padding: 0 12px; font-size: 12px; }} QPushButton:hover {{ border-color: {theme.ACCENT}; }}")
        self.solo_btn.adjustSize()
        self._place_button()
        if on:
            QTimer.singleShot(0, self, self.focus_athlete)
        self.update()

    def _place_button(self):
        self.solo_btn.move(self.width() - self.solo_btn.width() - 10, 10)

    @property
    def shadow_shown(self) -> bool:
        """The skeletons of the neighbouring frames help only while one video is corrected, stopped."""
        return bool(self.solo and self.ctl.shadow and not self.ctl.playing)

    @property
    def v(self):
        return self.ctl.session.views[self.index]

    # geometry
    def _base(self) -> float:
        m = self.v.meta
        return min(self.width() / m.width, self.height() / m.height) * 0.96

    def view(self) -> tuple[float, float, float]:
        m = self.v.meta
        s = self._base() * self.zoom
        return s, (self.width() - m.width * s) / 2 + self.pan.x(), (self.height() - m.height * s) / 2 + self.pan.y()

    def map(self, xy) -> QPointF:
        s, ox, oy = self.view()
        return QPointF(float(ox + xy[0] * s), float(oy + xy[1] * s))

    def inv(self, pos: QPointF) -> tuple[float, float]:
        s, ox, oy = self.view()
        return float((pos.x() - ox) / s), float((pos.y() - oy) / s)

    def fit(self):
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.update()

    def focus_athlete(self):
        """Zoom so the person's whole movement fills the view."""
        if not self.live:
            return
        xy = self.v.xy[:, HANDLES].reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 10 or self.width() < 50:
            self.fit()
            return
        x0, y0 = np.percentile(xy, 1, axis=0)
        x1, y1 = np.percentile(xy, 99, axis=0)
        w, h = float(x1 - x0) * 1.5 + 40, float(y1 - y0) * 1.36 + 40
        self.zoom = float(max(1.0, min(8.0, min(self.width() / w, self.height() / h) * 0.96 / self._base())))
        self.pan = QPointF(0, 0)
        c = self.map((float(x0 + x1) / 2, float(y0 + y1) / 2))
        self.pan = QPointF(self.width() / 2 - c.x(), self.height() / 2 - c.y())
        self.update()

    def zoom_at(self, factor: float, at: QPointF):
        ix, iy = self.inv(at)
        self.zoom = max(0.25, min(14.0, self.zoom * factor))
        self.pan += at - self.map((ix, iy))
        self.update()

    def resizeEvent(self, e):
        self._place_button()
        if self.live and e.oldSize().width() < 50:
            self.focus_athlete()

    def joint_at(self, pos: QPointF) -> int | None:
        xy = self.v.xy[self.ctl.frame(self.index)]
        best, best_d = None, HIT_RADIUS
        for j in HANDLES:
            if not np.isfinite(xy[j]).all():
                continue
            q = self.map(xy[j])
            d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
            if d < best_d:
                best, best_d = j, d
        return best

    # painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.STAGE))
        s = self.ctl.session
        if not self.live or not s.frame_count:
            return
        p.setRenderHint(QPainter.Antialiasing)
        scale, ox, oy = self.view()
        m = self.v.meta
        target = QRectF(ox, oy, m.width * scale, m.height * scale)
        p.setRenderHint(QPainter.SmoothPixmapTransform, bool(scale < 3))
        p.drawImage(target, self.ctl.image(self.index))
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(target)
        row, f = self.ctl.row, self.ctl.frame(self.index)
        xy, valid, src = self.v.xy[f], self.v.valid()[f], self.v.source[f]
        used = s.used[self.index, row]                    # built into the 3D point (not left out)
        proj = s.projection(self.index, row)
        err, apart = s.issue_cm[row], s.err_cm[row]
        if self.ctl.show_3d:
            self._ghost(p, proj)
        if self.shadow_shown:
            self._shadow(p, f)
        self._epipolar(p)
        for a, b in DRAW_BONES:
            if not (valid[a] and valid[b]):
                continue
            color = SIDE_COLOR[SIDE[a] if SIDE[a] == SIDE[b] else "C"]
            p.setPen(QPen(QColor(color), 2.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(self.map(xy[a]), self.map(xy[b]))
        if self.ctl.show_lengths:
            self._lengths(p, row, xy)
        sel = self.ctl.selected
        for j in HANDLES:
            if not np.isfinite(xy[j]).all():
                continue
            c = self.map(xy[j])
            color = QColor(SIDE_COLOR[SIDE[j]])
            r = 4.5 if j in (11, 12, 13, 14) else 3.2 if j >= 20 else 4.0
            bad = np.isfinite(err[j]) and err[j] > PROBLEM_CM
            if apart[j] > PROBLEM_CM and self.ctl.show_3d and np.isfinite(proj[j]).all():   # where the 3D solution puts it
                p.setPen(QPen(QColor(theme.WARN), 1.4, Qt.DashLine))
                p.drawLine(c, self.map(proj[j]))
            if valid[j] and (used[j] or not np.isfinite(apart[j])):
                p.setPen(QPen(QColor(theme.BG), 1.5))
                p.setBrush(color)
            else:                                           # not used: too uncertain, or the other videos disagree
                p.setPen(QPen(QColor(theme.DIM), 1.6))
                p.setBrush(QColor(theme.BG))
            p.drawEllipse(c, r, r)
            p.setBrush(Qt.NoBrush)
            if src[j] == SRC_MANUAL:
                p.setPen(QPen(QColor(theme.ACCENT), 1.4))
                p.drawEllipse(c, r + 3, r + 3)
            elif src[j] == SRC_FOLLOW:                     # moved to follow a correction in another video
                p.setPen(QPen(QColor(theme.ACCENT), 1.2, Qt.DotLine))
                p.drawEllipse(c, r + 3, r + 3)
            elif src[j] == SRC_LEARNED:                    # put on the path learned over the pedal stroke
                p.setPen(QPen(QColor(theme.OK), 1.2, Qt.DashLine))
                p.drawEllipse(c, r + 3, r + 3)
            if bad:
                p.setPen(QPen(QColor(theme.DANGER if err[j] > 2 * PROBLEM_CM else theme.WARN), 1.8))
                p.drawEllipse(c, r + 5.5, r + 5.5)
            if sel and sel[1] == j:
                p.setPen(QPen(QColor(theme.ACCENT if sel[0] == self.index else "#FFFFFF"), 2))
                p.drawEllipse(c, r + 8, r + 8)
            elif j == self.hover:
                p.setPen(QPen(QColor(255, 255, 255, 170), 1.5))
                p.drawEllipse(c, r + 7, r + 7)
        self._hud(p, target, f)

    def _lengths(self, p: QPainter, row: int, xy: np.ndarray):
        """Thighs and shanks off their usual length: from the point to move to where it should be,
        red when the bone is too long (that much is too much), dashed yellow when it is too short
        (that much is missing). The point it ends on is where to drag it."""
        s = self.ctl.session
        sel = self.ctl.selected
        for g in s.bone_guides(row):
            j = g["joint"]
            if not np.isfinite(xy[j]).all():
                continue
            to = s.project(self.index, row, g["target"])
            if not np.isfinite(to).all():
                continue
            a, b = self.map(xy[j]), self.map(to)
            color = QColor(LONG_COLOR if g["long"] else SHORT_COLOR)
            p.setPen(QPen(color, 4.0 if g["long"] else 3.0, Qt.SolidLine if g["long"] else Qt.DashLine, Qt.RoundCap))
            p.drawLine(a, b)
            p.setPen(QPen(color, 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(b, 6, 6)
            if (sel and sel[1] == j) or j == self.hover or (self.ctl.dragging and self.ctl.dragging[1] == j):
                text = bone_text(g["bones"], self.ctl.session.tape_bones())
                wide = QFontMetricsF(theme.font(12)).horizontalAdvance(text) + 34
                left = a.x() + 14 + wide > self.width()          # no room on the right: to the left of the point
                self._pill(p, a.x() - 14 if left else a.x() + 14, a.y() - 34, text, color.name(), right=left)

    def _ghost(self, p: QPainter, proj: np.ndarray):
        p.setPen(QPen(QColor(255, 255, 255, 120), 1.2, Qt.SolidLine, Qt.RoundCap))
        for a, b in DRAW_BONES:
            if np.isfinite(proj[[a, b]]).all():
                p.drawLine(self.map(proj[a]), self.map(proj[b]))

    def _shadow(self, p: QPainter, f: int):
        """This video's skeleton in the previous and in the next frame, faint: where every point
        comes from and where it goes, so a correction can be put in line with its neighbours."""
        from openkick.ui.overlay import GHOST_NEXT, GHOST_PREV
        v = self.v
        sel = self.ctl.selected
        focus = sel[1] if sel else self.hover
        cur = v.xy[f]
        for g, color in ((f - 1, GHOST_PREV), (f + 1, GHOST_NEXT)):
            if not 0 <= g < v.frame_count:
                continue
            xy, ok = v.xy[g], v.valid()[g]
            p.setPen(QPen(theme.qcolor(color, 135), 1.6, Qt.SolidLine, Qt.RoundCap))
            for a, b in DRAW_BONES:
                if ok[a] and ok[b]:
                    p.drawLine(self.map(xy[a]), self.map(xy[b]))
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor(color, 190))
            for j in HANDLES:
                if ok[j]:
                    p.drawEllipse(self.map(xy[j]), 2.2, 2.2)
            if focus is not None and np.isfinite(xy[focus]).all():
                c = self.map(xy[focus])
                if np.isfinite(cur[focus]).all():
                    p.setPen(QPen(theme.qcolor(color, 230), 1.6, Qt.DashLine))
                    p.drawLine(c, self.map(cur[focus]))
                p.setPen(QPen(QColor(theme.BG), 1.2))
                p.setBrush(QColor(color))
                p.drawEllipse(c, 4.2, 4.2)

    def _epipolar(self, p: QPainter):
        d = self.ctl.dragging
        if not d or d[0] == self.index:
            return
        line = self.ctl.session.epipolar_line(d[0], self.ctl.row, d[1], self.index)
        if line is None:
            return
        p.setPen(QPen(QColor(theme.ACCENT), 1.4, Qt.DashLine))
        p.setClipRect(self.rect())
        p.drawLine(self.map(line[0]), self.map(line[1]))

    def _pill(self, p: QPainter, x: float, y: float, text: str, dot: str | None = None, right=False):
        f = theme.font(12)
        p.setFont(f)
        w = QFontMetricsF(f).horizontalAdvance(text) + 20 + (14 if dot else 0)
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

    def _hud(self, p: QPainter, target: QRectF, f: int):
        from openkick.ui.overlay import GHOST_NEXT, GHOST_PREV
        left, top = max(target.left(), 0) + 10, max(target.top(), 0) + 10
        self._pill(p, left, top, tr("Video {n} · frame {f}", n=self.index + 1, f=f + 1))
        if self.v.handheld:
            self._pill(p, left, top + 32, tr("Handheld camera · compensated"), theme.OK)
        if self.shadow_shown:                         # what the two faint skeletons are
            x, bottom = left, min(target.bottom(), self.height()) - 36
            for text, color, there in ((tr("previous frame"), GHOST_PREV, f > 0),
                                       (tr("next frame"), GHOST_NEXT, f < self.v.frame_count - 1)):
                if there:
                    self._pill(p, x, bottom, text, color)
                    x += QFontMetricsF(theme.font(12)).horizontalAdvance(text) + 40

    # interaction
    def wheelEvent(self, e):
        if self.live:
            self.zoom_at(1.15 ** (e.angleDelta().y() / 120), e.position())

    def mouseDoubleClickEvent(self, e):
        if self.live and self.joint_at(e.position()) is None:
            self.focus_athlete() if self.zoom == 1.0 else self.fit()

    def mousePressEvent(self, e):
        s = self.ctl.session
        if not self.live or not s.frame_count:
            return
        pos = e.position()
        self.setFocus()
        if e.button() == Qt.RightButton:
            self._menu(pos, e.globalPosition().toPoint())
            return
        j = self.joint_at(pos) if e.button() == Qt.LeftButton and not e.modifiers() & Qt.ShiftModifier else None
        if j is None:
            if e.button() == Qt.LeftButton:
                self.ctl.select(None)
            self.drag = ("pan", pos, QPointF(self.pan))
            self.setCursor(Qt.ClosedHandCursor)
            return
        f = self.ctl.frame(self.index)
        v = self.v
        self.ctl.select((self.index, j))
        self.drag = ("joint", j, f, (v.xy[f, j].copy(), float(v.score[f, j]), int(v.source[f, j])), pos)
        self.ctl.dragging = (self.index, j)
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        s = self.ctl.session
        if not self.live or not s.frame_count:
            return
        pos = e.position()
        if self.drag and self.drag[0] == "pan":
            self.pan = self.drag[2] + (pos - self.drag[1])
            self.update()
            return
        if self.drag:
            _, j, f, _, start = self.drag
            v = self.v
            v.xy[f, j], v.score[f, j], v.source[f, j] = self.inv(pos), 1.0, SRC_MANUAL
            self.ctl.recompute(cameras=False)             # live: the 3D point follows the hand
            cm = s.err_cm[self.ctl.row, j]
            text = tr("{joint} · the videos now disagree by {cm} cm", joint=joint_name(j), cm=num(cm)) \
                if np.isfinite(cm) else joint_name(j)
            touching = [x for g in s.bone_guides(self.ctl.row, pct=0.0) for x in g["bones"] if G.IDX[x[0]] == j or G.IDX[x[1]] == j]
            self.ctl.message.emit(text + (" · " + bone_text(touching, s.tape_bones()) if touching else ""))
            return
        j = self.joint_at(pos)
        if j != self.hover:
            self.hover = j
            self.setToolTip("" if j is None else self.ctl.describe(self.ctl.row, j))
            self.setCursor(Qt.OpenHandCursor if j is not None else Qt.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, e):
        if not self.drag:
            return
        drag, self.drag = self.drag, None
        self.setCursor(Qt.ArrowCursor)
        if drag[0] == "pan":
            return
        _, j, f, (old_xy, old_score, old_src), start = drag
        v = self.v
        new = tuple(float(c) for c in v.xy[f, j])
        v.xy[f, j], v.score[f, j], v.source[f, j] = old_xy, old_score, old_src     # back, then one clean edit
        self.ctl.dragging = None
        pos = e.position()
        if math.hypot(pos.x() - start.x(), pos.y() - start.y()) < 1.5:
            self.ctl.recompute(cameras=False)
            return
        s = self.ctl.session
        edit = s.move_point(self.index, f, j, new, self.ctl.scope)
        followed = s.follow(self.index, edit, j) if self.ctl.follow else []
        self.ctl.push([edit] + followed, tr("Move {joint}", joint=joint_name(j)))
        refused = s.last_follow["refused"] if self.ctl.follow else {}
        if refused:
            self.ctl.message.emit(tr("{joint} corrected · {videos} did not follow everywhere: there it does not agree with "
                                     "this correction (it would have to move {px} px). Check the point there.", joint=joint_name(j),
                                     videos=", ".join(tr("Video {n}", n=w + 1) for w in refused),
                                     px=num(max(refused.values()), 0)))
            return
        if followed:
            self.ctl.message.emit(tr("{joint} corrected · the other videos followed it (they moved up to {px} px)",
                                     joint=joint_name(j), px=num(s.last_follow["moved_px"], 0)))
            return
        cm = s.err_cm[self.ctl.row, j]
        if np.isfinite(cm):
            self.ctl.message.emit(tr("{joint} corrected · the videos now disagree by {cm} cm", joint=joint_name(j), cm=num(cm)))

    def _menu(self, pos: QPointF, gpos):
        ctl, s = self.ctl, self.ctl.session
        menu = QMenu(self)
        j = self.joint_at(pos)
        f = ctl.frame(self.index)
        if j is not None:
            ctl.select((self.index, j))
            theme.menu_title(menu, joint_name(j))
            a = menu.addAction(tr("Back to the automatic detection"),
                               lambda: ctl.push([s.revert_point(self.index, f, j)], tr("Revert {joint}", joint=joint_name(j))))
            a.setEnabled(bool(self.v.source[f, j] != 1 or self.v.score[f, j] != self.v.raw_score[f, j]))
            menu.addAction(tr("Do not use this point here"),
                           lambda: ctl.push([s.discard_point(self.index, f, j)], tr("Discard {joint}", joint=joint_name(j))))
            g = GROUP_OF.get(j)
            if g:
                label = tr("Left and right legs are exchanged") if g == "legs" else tr("Left and right arms are exchanged")
                sub = menu.addMenu(label)
                text = tr("Exchange left and right")
                last = self.v.frame_count - 1
                sub.addAction(tr("Fix only this frame"), lambda: ctl.push([s.swap_sides(self.index, f, f, g)], text))
                sub.addAction(tr("Fix from here to the end"), lambda: ctl.push([s.swap_sides(self.index, f, last, g)], text))
                sub.addAction(tr("Fix from the start to here"), lambda: ctl.push([s.swap_sides(self.index, 0, f, g)], text))
            menu.addSeparator()
        menu.addAction(tr("Focus on the person"), self.focus_athlete)
        menu.addAction(tr("Fit to window"), self.fit)
        menu.exec(gpos)


# ── the 3D views ───────────────────────────────────────────────────────


class Orbit(QObject):
    """From where a 3D view is looked at. Views that share one turn together."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.yaw, self.pitch, self.zoom = 35.0, 12.0, 1.0

    def reset(self):
        self.yaw, self.pitch, self.zoom = 35.0, 12.0, 1.0
        self.changed.emit()

    def basis(self) -> np.ndarray:
        """Rows: the screen's right, up and towards-the-viewer directions, in world axes."""
        y, x = math.radians(self.yaw), math.radians(self.pitch)
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        return Rx @ Ry


class _Orbiting(QWidget):
    """Mouse handling shared by the 3D views: drag turns, wheel zooms, double-click resets."""

    def __init__(self, ctl: MultiviewController, orbit: Orbit | None, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.orbit = orbit or Orbit(self)
        self.setMinimumSize(200, 200)
        self._drag = None
        self.orbit.changed.connect(self.update)
        ctl.rowChanged.connect(lambda _: self.update())
        ctl.changed.connect(self.update)
        ctl.sessionLoaded.connect(self.update)

    def px_per_m(self, size: float) -> float:
        return min(self.width(), self.height()) * 0.86 / size * self.orbit.zoom

    def mousePressEvent(self, e):
        self._drag = (e.position(), self.orbit.yaw, self.orbit.pitch)

    def mouseMoveEvent(self, e):
        if self._drag:
            d = e.position() - self._drag[0]
            self.orbit.yaw = self._drag[1] + d.x() * 0.5
            self.orbit.pitch = max(-89.0, min(89.0, self._drag[2] + d.y() * 0.5))
            self.orbit.changed.emit()

    def mouseReleaseEvent(self, _):
        self._drag = None

    def mouseDoubleClickEvent(self, _):
        self.orbit.reset()

    def wheelEvent(self, e):
        self.orbit.zoom = max(0.4, min(6.0, self.orbit.zoom * 1.15 ** (e.angleDelta().y() / 120)))
        self.orbit.changed.emit()

    def _caption(self, p: QPainter, title: str, foot: str = ""):
        p.setFont(theme.font(11, 600))
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(12, 10, self.width(), 18), Qt.AlignLeft | Qt.AlignVCenter, title)
        if foot:
            p.setFont(theme.font(11))
            p.setPen(QColor(theme.DIM))
            p.drawText(QRectF(0, self.height() - 24, self.width() - 12, 18), Qt.AlignRight | Qt.AlignVCenter, foot)


def paint_skeleton(p: QPainter, w: float, h: float, world: tuple, row: int, basis: np.ndarray, scale: float,
                   selected: int | None = None, k: float = 1.0):
    """The stick figure of `row` on a w x h area, seen along `basis` at `scale` pixels per metre.
    `k` thickens lines and dots for large pictures (exported video)."""
    W, center, size = world

    def project(P):
        q = (P - center) @ basis.T
        return np.stack([w / 2 + q[..., 0] * scale, h * 0.5 - q[..., 1] * scale], axis=-1)

    g = math.ceil(size / 2 / 0.5) * 0.5
    p.setPen(QPen(QColor(theme.LINE), 1 * k))
    for u in np.arange(-g, g + 1e-6, 0.5):
        for a, b in (((center[0] + u, 0, center[2] - g), (center[0] + u, 0, center[2] + g)),
                     ((center[0] - g, 0, center[2] + u), (center[0] + g, 0, center[2] + u))):
            q = project(np.array([a, b]))
            p.drawLine(QPointF(*q[0]), QPointF(*q[1]))
    if selected is not None:                            # path of the selected point
        path = project(W[:, selected])
        p.setPen(QPen(QColor(theme.ACCENT), 1.2 * k))
        for a, b in zip(path, path[1:]):
            if np.isfinite(a).all() and np.isfinite(b).all():
                p.drawLine(QPointF(*a), QPointF(*b))
    q = project(W[row])
    for a, b in DRAW_BONES:
        if np.isfinite(q[[a, b]]).all():
            p.setPen(QPen(QColor(SIDE_COLOR[SIDE[a] if SIDE[a] == SIDE[b] else "C"]), 2.6 * k, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(*q[a]), QPointF(*q[b]))
    for j in HANDLES:
        if not np.isfinite(q[j]).all():
            continue
        p.setPen(QPen(QColor(theme.BG), 1.2 * k))
        p.setBrush(QColor(SIDE_COLOR[SIDE[j]]))
        p.drawEllipse(QPointF(*q[j]), 3.4 * k, 3.4 * k)
        if selected == j:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(theme.ACCENT), 2 * k))
            p.drawEllipse(QPointF(*q[j]), 8 * k, 8 * k)


class Skeleton3D(_Orbiting):
    """The reconstructed points as a stick figure, seen from any side."""

    def __init__(self, ctl: MultiviewController, orbit: Orbit | None = None, title: str = "3D VIEW", parent=None):
        super().__init__(ctl, orbit, parent)
        self.title = title
        self.show_selection = True          # the selected point and the path it follows
        ctl.selectionChanged.connect(self.update)
        self.setToolTip(tr("Drag to turn the skeleton · double-click to reset"))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.STAGE))
        s = self.ctl.session
        if not s or not s.frame_count:
            return
        p.setRenderHint(QPainter.Antialiasing)
        world = self.ctl.world()
        if world[1] is None:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, tr("Nothing reconstructed yet"))
            return
        sel = self.ctl.selected if self.show_selection else None
        paint_skeleton(p, self.width(), self.height(), world, self.ctl.row, self.orbit.basis(), self.px_per_m(world[2]),
                       sel[1] if sel else None)
        self._caption(p, tr(self.title), tr("floor grid: 0.5 m"))


class OsimView(_Orbiting):
    """The OpenSim model (bones) moved by the inverse-kinematics result."""

    def __init__(self, ctl: MultiviewController, orbit: Orbit | None = None, parent=None):
        super().__init__(ctl, orbit, parent)
        self.scene = None
        self.status = ""
        self.busy = False
        self._key, self._image = None, None
        self.setToolTip(tr("Drag to turn the model · double-click to reset"))

    def set_scene(self, scene, status: str = "", busy: bool = False):
        self.scene, self.status, self.busy = scene, status, busy
        self._key = None
        self.update()

    def frame(self) -> int | None:
        s = self.ctl.session
        if self.scene is None or not s or not s.frame_count:
            return None
        return self.scene.frame_at(float(s.t[self.ctl.row] - s.t[0]))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.STAGE))
        f = self.frame()
        _, center, size = self.ctl.world() if self.ctl.session else (None, None, 1.0)
        if f is None or center is None:
            p.setPen(QColor(theme.MUTED))
            p.setFont(theme.font(13))
            p.drawText(self.rect().adjusted(30, 0, -30, 0), Qt.AlignCenter | Qt.TextWordWrap, self.status)
            self._caption(p, tr("OPENSIM MODEL"))
            return
        w, h = max(2, self.width()), max(2, self.height())
        o = self.orbit
        key = (f, w, h, round(o.yaw, 2), round(o.pitch, 2), round(o.zoom, 4), tuple(np.round(center, 3)), round(size, 3))
        if key != self._key:
            B = o.basis()
            g = math.ceil(size / 2 / 0.5) * 0.5
            if getattr(self.scene, "_floor_key", None) != (tuple(np.round(center, 3)), g):
                self.scene.set_floor(float(center[0]), float(center[2]), g)
                self.scene._floor_key = (tuple(np.round(center, 3)), g)
            arr = self.scene.render(f, (w, h), center, B[1], B[2], self.px_per_m(size))
            self._image = QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QImage.Format_RGB888).copy()
            self._key = key
        p.drawImage(0, 0, self._image)
        self._caption(p, tr("OPENSIM MODEL"), tr("floor grid: 0.5 m"))


def _encode_video(path: str | Path, session: Session, speed: float, size: tuple, draw, progress=None) -> int:
    """Write an MP4 of the whole movement. `draw(painter, W, H, row, seconds)` paints each
    picture. `speed` 0.5 is half speed. `progress(done, total)` returning False cancels.
    Returns the frames written (0 if cancelled)."""
    from fractions import Fraction

    import av
    W, H = int(size[0]) - int(size[0]) % 2, int(size[1]) - int(size[1]) % 2
    t0, t1 = float(session.t[0]), float(session.t[-1])
    fps = max(1, round(1 / float(np.median(np.diff(session.t)))))
    total = max(2, int(round((t1 - t0) * fps / speed)) + 1)
    done = 0
    with av.open(str(path), "w") as out:
        codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
        stream = out.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        if codec == "libx264":
            stream.options = {"crf": "18", "preset": "medium"}
        else:
            stream.bit_rate = 10_000_000
        for i in range(total):
            if progress and progress(i, total) is False:
                break
            t = min(t1, t0 + i * speed / fps)
            img = QImage(W, H, QImage.Format_RGB888)
            img.fill(QColor(theme.PANEL))
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            draw(p, W, H, int(np.abs(session.t - t).argmin()), t - t0)
            p.end()
            frame = np.frombuffer(img.constBits(), np.uint8).reshape(H, img.bytesPerLine())[:, :W * 3].reshape(H, W, 3)
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            vf.pts = i
            vf.time_base = Fraction(1, fps)
            for packet in stream.encode(vf):
                out.mux(packet)
            done += 1
        cancelled = done < total
        for packet in stream.encode():
            out.mux(packet)
    return 0 if cancelled else done


def _footer(p: QPainter, W: int, H: int, foot: int, k: float, seconds: float, speed: float, name: str, note: str = ""):
    p.setFont(theme.mono(13 * k))
    p.setPen(QColor(theme.MUTED))
    p.drawText(QRectF(0, H - foot, W, foot), Qt.AlignCenter, timecode(seconds)
               + ("" if speed == 1 else "   ·   " + tr("slow motion ×{s}", s=f"{speed:g}")))
    p.setFont(theme.font(12 * k))
    p.setPen(QColor(theme.DIM))
    p.drawText(QRectF(18 * k, H - foot, W / 2, foot), Qt.AlignVCenter | Qt.AlignLeft, name)
    p.drawText(QRectF(W / 2, H - foot, W / 2 - 18 * k, foot), Qt.AlignVCenter | Qt.AlignRight,
               (note + "   ·   " if note else "") + APP_NAME)


def write_comparison_video(path: str | Path, session: Session, world: tuple, scene, orbit: "Orbit", speed: float = 1.0,
                           size: tuple = (1920, 1080), name: str = "", progress=None) -> int:
    """MP4 with the 3D points on the left and the OpenSim model on the right, both seen from
    where `orbit` looks, for the whole movement."""
    W, H = int(size[0]) - int(size[0]) % 2, int(size[1]) - int(size[1]) % 2
    k = H / 720                                                 # sizes were tuned for 720 lines
    head, foot = round(54 * k), round(40 * k)
    pw, ph = W // 2 - 1, H - head - foot
    _, center, extent = world
    basis = orbit.basis()
    scale = min(pw, ph) * 0.86 / extent * orbit.zoom
    scene.set_floor(float(center[0]), float(center[2]), math.ceil(extent / 2 / 0.5) * 0.5)
    scene._floor_key = None                                     # the window's view sets its own again
    t0 = float(session.t[0])

    def draw(p, W, H, row, seconds):
        p.fillRect(QRectF(0, head, pw, ph), QColor(theme.STAGE))
        p.save()
        p.translate(0, head)
        p.setClipRect(QRectF(0, 0, pw, ph))
        paint_skeleton(p, pw, ph, world, row, basis, scale, None, k)
        p.restore()
        arr = scene.render(scene.frame_at(float(session.t[row]) - t0), (pw, ph), center, basis[1], basis[2], scale)
        p.drawImage(W - pw, head, QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QImage.Format_RGB888))
        p.setFont(theme.font(15 * k, 600))
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(18 * k, 0, pw, head), Qt.AlignVCenter | Qt.AlignLeft, tr("3D points from the videos"))
        p.drawText(QRectF(W - pw + 18 * k, 0, pw, head), Qt.AlignVCenter | Qt.AlignLeft, tr("OpenSim model"))
        _footer(p, W, H, foot, k, seconds, speed, name)

    return _encode_video(path, session, speed, (W, H), draw, progress)


def camera_directions(session: Session) -> list:
    """For every camera: the screen axes (right, up, towards the viewer) of a view that looks at
    the person from where that camera was (never from lower than 9 degrees above the floor), in
    the upright axes of the 3D views."""
    from openkick.multiview.export import to_world
    info = to_world(session.X)[1]
    person = np.nanmedian(session.X[:, G.IDX["hip"]], axis=0)              # metres, first camera's axes
    out = []
    for R, t in session.cams:
        cam = -np.asarray(R).T @ np.asarray(t) * session.scale
        toward = info["R"] @ (cam - person)
        toward /= np.linalg.norm(toward)
        low = math.sin(math.radians(9))                 # a camera level with the hips would see the floor edge-on:
        if toward[1] < low:                             # look from slightly above, so the floor grid shows
            flat = np.array([toward[0], 0.0, toward[2]])
            toward = flat / np.linalg.norm(flat) * math.cos(math.radians(9)) + np.array([0.0, low, 0.0])
        right = np.cross([0.0, 1.0, 0.0], toward)
        right /= np.linalg.norm(right)
        out.append(np.stack([right, np.cross(toward, right), toward]))
    return out


def camera_video_size(count: int) -> tuple[int, int]:
    """Picture size of the video with one column per camera."""
    pw = 800 if count == 2 else 640 if count == 3 else 1920 // count
    ph = 640 if count <= 3 else 520
    k = ph / 520
    return pw * count, 2 * (round(34 * k) + ph) + round(40 * k)


def write_camera_video(path: str | Path, session: Session, world: tuple, scene, readers: list, speed: float = 1.0,
                       size: tuple | None = None, name: str = "", progress=None) -> int:
    """MP4 with one column per camera: above, the video itself; below, the OpenSim model seen
    from where that camera was, with the reconstructed 3D points as green dots."""
    V = len(session.views)
    W, H = size or camera_video_size(V)
    W, H = int(W) - int(W) % 2, int(H) - int(H) % 2
    foot = round(H * 0.03) + 14
    cap = round(H * 0.026) + 12
    pw, ph = W // V, (H - foot - 2 * cap) // 2
    k = ph / 520
    Wd, center, extent = world
    bases = camera_directions(session)
    scale = min(pw, ph) * 0.86 / extent
    scene.set_floor(float(center[0]), float(center[2]), math.ceil(extent / 2 / 0.5) * 0.5)
    scene._floor_key = None
    t0 = float(session.t[0])

    def draw(p, W, H, row, seconds):
        frame = scene.frame_at(float(session.t[row]) - t0)
        for v in range(V):
            x0 = v * pw
            # the video
            rgb = readers[v].frame(session.frame_of(v, row))
            src = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
            f = min((pw - 2) / src.width(), ph / src.height())
            dst = QRectF(x0 + (pw - src.width() * f) / 2, cap + (ph - src.height() * f) / 2, src.width() * f, src.height() * f)
            p.fillRect(QRectF(x0, cap, pw - 2, ph), QColor(theme.STAGE))
            p.drawImage(dst, src)
            # the model from that camera, and the points it was built from
            B = bases[v]
            arr = scene.render(frame, (pw - 2, ph), center, B[1], B[2], scale)
            y1 = 2 * cap + ph
            p.drawImage(x0, y1, QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QImage.Format_RGB888))
            q = (Wd[row][G.BODY] - center) @ B.T
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.ACCENT))
            for a, b in zip(x0 + (pw - 2) / 2 + q[:, 0] * scale, y1 + ph / 2 - q[:, 1] * scale):
                if np.isfinite(a) and np.isfinite(b):
                    p.drawEllipse(QPointF(float(a), float(b)), 3.2 * k, 3.2 * k)
            p.setFont(theme.font(13 * k, 600))
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(x0 + 12 * k, 0, pw, cap), Qt.AlignVCenter | Qt.AlignLeft, tr("Video {n}", n=v + 1))
            p.drawText(QRectF(x0 + 12 * k, cap + ph, pw, cap), Qt.AlignVCenter | Qt.AlignLeft,
                       tr("OpenSim · seen from camera {n}", n=v + 1))
        _footer(p, W, H, foot, k, seconds, speed, name, tr("green dots: what was reconstructed from the videos"))

    return _encode_video(path, session, speed, (W, H), draw, progress)


# ── timeline: where the videos disagree ────────────────────────────────

BARS_H = 74                # the doubt bars: the whole timeline when nobody walks
STRIDES_H = 30             # when the person walks, two rows above them, like the phases of the 2D
STANCE_H = 22              # timeline: the strides, then the stance and swing of each
BONES_H = 14               # a strip over the bars: how far off its length the worst thigh or shank is
GAIT_LABELS_W = 84         # room for the names of those two rows


def show_gait_report(ctl: "MultiviewController", parent: QWidget):
    """Every table of the analysis of what the session shows (pedalling or walking), in the dialog
    the plugins use."""
    from openkick.ui.dialogs import PluginResultDialog
    if ctl.activity_kind() == "template":
        res = ctl.template_result()
        if res:
            PluginResultDialog(res.title, res, parent).exec()
        return
    g = ctl.activity()
    if g:
        title = tr("Pedal stroke analysis in 3D") if ctl.activity_kind() == "cycling" else tr("Gait analysis")
        PluginResultDialog(title, g.result, parent).exec()


def report_label(ctl) -> str:
    kind = ctl.activity_kind()
    return (tr("See the pedal stroke analysis") if kind == "cycling" else tr("See the angles in 3D") if kind == "template"
            else tr("See the gait analysis"))


class ErrorTimeline(QWidget):
    """One bar per step: how doubtful it is. With all the videos, a step is a row of the 3D
    (a frame of video 1). With one video edited on its own, a step is each frame of THAT video,
    so none of its frames is skipped. When the person walks, two rows above the bars show the
    strides and their stance and swing, drawn like the phases of the 2D timeline."""

    def __init__(self, ctl: MultiviewController, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setFixedHeight(BARS_H)
        self.setMouseTracking(True)
        self._cells_key, self._cells = None, None
        ctl.rowChanged.connect(self._refresh)
        ctl.changed.connect(self._refresh)
        ctl.sessionLoaded.connect(self._refresh)
        ctl.focusChanged.connect(self._refresh)
        ctl.selectionChanged.connect(self._refresh)          # the strip follows the selected point's bones
        self.tip = tr("Height: how doubtful each frame is. Dots: frames you corrected")
        self.setToolTip(self.tip)

    def _refresh(self, *_):
        h = BARS_H + (STRIDES_H + STANCE_H if self.ctl.activity() else 0) + (BONES_H if self.ctl.show_lengths else 0)
        if self.height() != h:
            self.setFixedHeight(h)
        self.update()

    def lanes(self) -> dict:
        """Where each row goes. All of them share the x axis of "track"; "strides" and "phases"
        are there only when the person walks."""
        walking = self.ctl.activity() is not None
        left = 14 + (GAIT_LABELS_W if walking else 0)
        track = QRectF(left, 8, self.width() - left - 14, self.height() - 16)
        out = {"track": track}
        y = track.y()
        if walking:
            out["strides"] = QRectF(track.x(), y, track.width(), STRIDES_H)
            out["phases"] = QRectF(track.x(), y + STRIDES_H, track.width(), STANCE_H)
            y += STRIDES_H + STANCE_H
        if self.ctl.show_lengths:
            out["bones"] = QRectF(track.x(), y, track.width(), BONES_H)
            y += BONES_H
        out["bars"] = QRectF(track.x(), y, track.width(), max(0.0, track.bottom() - y))
        return out

    def bone_columns(self) -> list[int]:
        """The thighs and shanks the strip shows: those of the selected point, else all four."""
        sel = self.ctl.selected
        if sel:
            mine = [k for k, (a, b) in enumerate(G.LEG_BONES) if sel[1] in (G.IDX[a], G.IDX[b])]
            if mine:
                return mine
        return list(range(len(G.LEG_BONES)))

    @staticmethod
    def bone_color(dev: float) -> str | None:
        """The strip's colour for a bone this far off (fraction): yellow from 5 %, orange from 10 %,
        red from 15 %."""
        a = abs(dev) * 100
        return None if not np.isfinite(a) or a < 5 else SHORT_COLOR if a < 10 else theme.WARN if a < 15 else LONG_COLOR

    def track(self) -> QRectF:
        return self.lanes()["track"]

    def cells(self) -> dict:
        """What the bars stand for: count, current one, doubt of each (cm), which were corrected
        by hand, the row of the 3D each one shows, and the number shown for the current one."""
        c, s = self.ctl, self.ctl.session
        key = (id(s), s.revision, c.focus)
        if self._cells_key != key:
            err = s.frame_error()
            if c.focus is None:
                manual = np.zeros(s.frame_count, bool)
                for view, v in enumerate(s.views):
                    for f in np.nonzero(v.manual().any(axis=1))[0]:
                        manual[s.row_of(view, int(f))] = True
                self._cells = {"n": s.frame_count, "first": 0, "err": err, "manual": manual,
                               "rows": np.arange(s.frame_count)}
            else:
                lo, hi = c.focus_range()
                rows = np.array([s.row_of(c.focus, f) for f in range(lo, hi + 1)])
                self._cells = {"n": hi - lo + 1, "first": lo, "err": err[rows], "rows": rows,
                               "manual": s.views[c.focus].manual().any(axis=1)[lo:hi + 1]}
            self._cells_key = key
        cur = c.row if c.focus is None else c.focus_frame - self._cells["first"]
        return dict(self._cells, current=cur)

    def index_at(self, x: float) -> int:
        r, n = self.track(), self.cells()["n"]
        return max(0, min(n - 1, int((x - r.x()) / max(r.width(), 1) * n)))

    def go(self, index: int):
        if self.ctl.focus is None:
            self.ctl.set_row(index)
        else:
            self.ctl.set_focus_frame(self.cells()["first"] + index)

    @staticmethod
    def _span(rows: np.ndarray, ph) -> tuple[int, int]:
        """First cell of a phase and the one after its last (a cell may be a frame of one video)."""
        return int(np.searchsorted(rows, ph.start, "left")), int(np.searchsorted(rows, ph.end, "right"))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        s = self.ctl.session
        if not s or not s.frame_count:
            return
        p.setRenderHint(QPainter.Antialiasing)
        cells = self.cells()
        lanes = self.lanes()
        r, n = lanes["track"], cells["n"]
        w = r.width() / n
        g = self.ctl.activity()
        if g is not None and "strides" in lanes:
            self._gait(p, g, lanes, cells, w)
        if "bones" in lanes:
            self._bones(p, lanes["bones"], cells, w)
        area = lanes["bars"]
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.CARD_2))
        p.drawRoundedRect(area, 6, 6)
        bars = QRectF(area.x(), area.y() + 16, area.width(), area.height() - 20)
        top = 3 * PROBLEM_CM
        for i in range(n):
            e = cells["err"][i]
            if not np.isfinite(e):
                continue
            h = max(2.0, min(1.0, e / top) * bars.height())
            color = theme.OK if e <= PROBLEM_CM else theme.WARN if e <= 2 * PROBLEM_CM else theme.DANGER   # as the rings
            p.setBrush(theme.qcolor(color, 220 if e > PROBLEM_CM else 110))
            p.drawRect(QRectF(r.x() + i * w + 0.5, bars.bottom() - h, max(1.0, w - 1), h))
        y = bars.bottom() - PROBLEM_CM / top * bars.height()          # the "worth a look" level
        p.setPen(QPen(QColor(theme.BORDER), 1, Qt.DashLine))
        p.drawLine(QPointF(r.x(), y), QPointF(r.right(), y))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        for i in np.nonzero(cells["manual"])[0]:
            p.drawEllipse(QPointF(r.x() + (i + 0.5) * w, area.y() + 8), 2, 2)
        cur = cells["current"]
        x = r.x() + (cur + 0.5) * w
        p.setPen(Qt.NoPen)                                            # the current step, highlighted
        p.setBrush(QColor(255, 255, 255, 34))
        p.drawRect(QRectF(r.x() + cur * w, r.y(), max(2.0, w), r.height()))
        p.setPen(QPen(QColor(theme.TEXT), 1.5))
        p.drawLine(QPointF(x, r.y() - 3), QPointF(x, r.bottom() + 3))

    def _bones(self, p: QPainter, lane: QRectF, cells: dict, w: float):
        dev = self.ctl.session.bone_deviation()
        if not len(dev):
            return
        cols = self.bone_columns()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            worst = np.nanmax(np.abs(dev[cells["rows"]][:, cols]), axis=1)
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(theme.OK, 45))
        p.drawRoundedRect(lane.adjusted(0, 3, 0, -3), 3, 3)
        for i, d in enumerate(worst):
            color = self.bone_color(d)
            if color:
                p.setBrush(QColor(color))
                p.drawRect(QRectF(lane.x() + i * w, lane.y() + 3, max(1.0, w), lane.height() - 6))
        if "strides" in self.lanes():                 # the rows have names when the person pedals or walks
            p.setPen(QColor(theme.TEXT))
            p.setFont(theme.font(10, 500))
            p.drawText(QRectF(14, lane.y(), GAIT_LABELS_W - 8, lane.height()), Qt.AlignVCenter | Qt.AlignLeft, tr("Lengths"))

    def bone_tip(self, pos: QPointF) -> str | None:
        lanes = self.lanes()
        if "bones" not in lanes or not lanes["bones"].contains(pos):
            return None
        row = int(self.cells()["rows"][self.index_at(pos.x())])
        cols = self.bone_columns()
        dev = self.ctl.session.bone_deviation()[row]
        off = [k for k in cols if np.isfinite(dev[k]) and abs(dev[k]) >= 0.05]
        if not off:
            return tr("Thighs and shanks at their usual length (within 5 %)")
        return " · ".join(f"{tr(BONE_NAMES[G.LEG_BONES[k]])} {100 * dev[k]:+.0f} %" for k in off)

    def _gait(self, p: QPainter, g, lanes: dict, cells: dict, w: float):
        """The strides and their stance and swing, like the phases of the 2D timeline: a block of
        its colour for each, the current one outlined."""
        rows = cells["rows"]
        now = int(rows[max(0, min(cells["current"], len(rows) - 1))])
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.font(12, 500))
        cycles = tr("Strokes") if self.ctl.activity_kind() == "cycling" else tr("Strides")
        for lane, text in (("strides", cycles), ("phases", tr("Phases"))):
            band = lanes[lane]
            p.drawText(QRectF(14, band.y(), GAIT_LABELS_W - 8, band.height()), Qt.AlignVCenter | Qt.AlignLeft, text)
        for lane, items, current, pad, radius, font in (
                ("strides", g.phases, g.phase_at(now), 3, 6, theme.font(12, 500)),
                ("phases", g.subphases, g.subphase_at(now), 2, 4, theme.font(10, 600))):
            band = lanes[lane]
            p.setFont(font)
            fm = QFontMetricsF(font)
            for i, ph in enumerate(items):
                a, b = self._span(rows, ph)
                if b <= a:
                    continue
                rect = QRectF(band.x() + a * w + 1, band.y() + pad, max(2.0, (b - a) * w - 2), band.height() - 2 * pad)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(ph.color))
                p.drawRoundedRect(rect, radius, radius)
                if ph is current:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(theme.ACCENT), 1.5))
                    p.drawRoundedRect(rect.adjusted(0.75, 0.75, -0.75, -0.75), radius, radius)
                p.setPen(QColor(theme.text_on(ph.color)))
                if lane == "strides":
                    wide = rect.width() >= 75
                    label, align = (ph.name, Qt.AlignVCenter | Qt.AlignLeft) if wide else (str(i + 1), Qt.AlignCenter)
                else:
                    label = ph.name if rect.width() >= 55 else ph.name[:1].upper() if rect.width() >= 16 else ""
                    align = Qt.AlignCenter
                p.drawText(rect.adjusted(4, 0, -4, 0), align, fm.elidedText(label, Qt.ElideRight, int(rect.width() - 8)))

    def gait_at(self, pos: QPointF):
        """(row name, phase) of the stride or the stance/swing block under `pos`, or None."""
        g = self.ctl.activity()
        if g is None or not self.ctl.session:
            return None
        lanes = self.lanes()
        for lane in ("strides", "phases"):
            if lane in lanes and lanes[lane].contains(pos):
                row = int(self.cells()["rows"][self.index_at(pos.x())])
                ph = g.phase_at(row) if lane == "strides" else g.subphase_at(row)
                return (lane, ph) if ph else None
        return None

    def describe(self, lane: str, ph) -> str:
        """What the tooltip says about a stride or a stance/swing block."""
        g, s = self.ctl.activity(), self.ctl.session
        span = f"{timecode(s.t[ph.start] - s.t[0])} – {timecode(s.t[ph.end] - s.t[0])}"
        leg = tr("Left leg") if g.side == "left" else tr("Right leg")
        stride = ph if lane == "strides" else g.phase_at(ph.start)
        parts = [stride.name] if stride else []
        if lane != "strides":
            parts.append(ph.name)
        return " · ".join(parts + [leg, span])

    def go_to_start(self, ph):
        cells = self.cells()
        self.go(min(self._span(cells["rows"], ph)[0], cells["n"] - 1))

    def mousePressEvent(self, e):
        if not (self.ctl.session and self.ctl.session.frame_count):
            return
        if e.button() == Qt.RightButton:
            hit = self.gait_at(e.position())
            if hit:
                menu = QMenu(self)
                theme.menu_title(menu, hit[1].name)
                menu.addAction(tr("Go to start"), lambda: self.go_to_start(hit[1]))
                menu.addAction(report_label(self.ctl), lambda: show_gait_report(self.ctl, self))
                menu.exec(e.globalPosition().toPoint())
                return
        self.go(self.index_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.mousePressEvent(e)
            return
        hit = self.gait_at(e.position())
        bones = self.bone_tip(e.position())
        self.setToolTip(self.describe(*hit) if hit else bones if bones else self.tip)

    def mouseDoubleClickEvent(self, e):
        hit = self.gait_at(e.position())
        if hit:
            self.go_to_start(hit[1])


# ── building a session ─────────────────────────────────────────────────


class BuildWorker(QThread):
    progress = Signal(str, int, int, QImage)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, videos: list, height_m: float, mode: str):
        super().__init__()
        self.args = (list(videos), height_m, mode)
        self._stop = False
        self._last = 0.0

    def stop(self):
        self._stop = True

    def run(self):
        from openkick.multiview.build import build
        try:
            self.done.emit(build(*self.args, progress=self._progress, cancelled=lambda: self._stop))
        except Exception as e:  # noqa: BLE001 - reported to the user
            self.failed.emit(str(e))

    def _progress(self, step, i, n, rgb=None, xy=None):
        now = time.monotonic()
        if now - self._last < 0.06 and i < n:
            return
        self._last = now
        self.progress.emit(step, int(i), int(n), _preview(rgb, xy) if rgb is not None else QImage())


def _preview(rgb: np.ndarray, xy: np.ndarray, W: int = 524, H: int = 295) -> QImage:
    h, w = rgb.shape[:2]
    src = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    out = QImage(W, H, QImage.Format_RGB32)
    out.fill(QColor(theme.STAGE))
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    s = min(W / w, H / h)
    ox, oy = (W - w * s) / 2, (H - h * s) / 2
    p.drawImage(QRectF(ox, oy, w * s, h * s), src)
    if xy is not None:
        for a, b in DRAW_BONES:
            if np.isfinite(xy[[a, b]]).all():
                p.setPen(QPen(QColor(SIDE_COLOR[SIDE[a] if SIDE[a] == SIDE[b] else "C"]), 2.2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(float(ox + xy[a, 0] * s), float(oy + xy[a, 1] * s)),
                           QPointF(float(ox + xy[b, 0] * s), float(oy + xy[b, 1] * s)))
    p.end()
    return out


class BuildDialog(QDialog):
    """Runs the automatic first attempt with a live preview."""

    def __init__(self, videos: list, height_m: float, mode: str, parent=None):
        super().__init__(parent)
        from openkick.multiview.build import steps
        from openkick.ui.processing import StepRow
        self.session: Session | None = None
        self.setWindowTitle(tr("Reconstructing in 3D"))
        self.setModal(True)
        self.setFixedWidth(600)
        self.setStyleSheet(f"QDialog {{ background: {theme.CARD_2}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24)
        lay.setSpacing(18)
        title = QLabel(tr("Reconstructing in 3D"))
        title.setFont(theme.display(24))
        lay.addWidget(title)
        names = [Path(v).name for v in videos]
        sub = QLabel("  +  ".join(names) if len(names) == 2 else tr("{n} videos", n=len(names)))
        sub.setStyleSheet(f"color: {theme.FAINT};")
        lay.addWidget(sub)
        self.preview = QLabel()
        self.preview.setFixedSize(524, 295)
        self.preview.setStyleSheet(f"background: {theme.STAGE}; border-radius: 12px; border: 1px solid #222834;")
        self.preview.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.preview, 0, Qt.AlignHCenter)
        self.bar = QProgressBar()
        self.bar.setFixedHeight(8)
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.order = steps(len(videos))
        texts = [tr("Detect the skeleton in video {n}", n=k + 1) for k in range(len(videos))] + [
            tr("Check whether a camera was handheld"), tr("Synchronize the videos and locate the cameras")]
        self.steps = {k: StepRow(t) for k, t in zip(self.order, texts)}
        box = QVBoxLayout()
        box.setSpacing(10)
        for k in self.order:
            box.addLayout(self.steps[k])
        lay.addLayout(box)
        foot = QHBoxLayout()
        note = QLabel(tr("The first time, the detection model is downloaded (about 150 MB)."))
        note.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
        foot.addWidget(note, 1)
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self._cancel)
        foot.addWidget(self.cancel_btn)
        lay.addLayout(foot)
        self.worker = BuildWorker(videos, height_m, mode)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, step: str, i: int, n: int, img: QImage):
        k = self.order.index(step)
        for name in self.order[:k]:
            self.steps[name].set_state("done")
        self.steps[step].set_state("now", f"{i} / {n}" if step.startswith("detect") else "")
        self.bar.setValue(int((k + i / max(n, 1)) / len(self.order) * 1000))
        if not img.isNull():
            self.preview.setPixmap(QPixmap.fromImage(img))

    def _done(self, session):
        self.worker.wait()
        if session is None:
            super().reject()
            return
        self.session = session
        self.accept()

    def _failed(self, msg: str):
        self.worker.wait()
        QMessageBox.critical(self, tr("The reconstruction failed"), msg)
        super().reject()

    def _cancel(self):
        self.worker.stop()
        self.cancel_btn.setEnabled(False)

    def reject(self):
        if self.worker.isRunning():
            self._cancel()
            return
        super().reject()

    def closeEvent(self, e):
        if self.worker.isRunning():
            self._cancel()
            e.ignore()
        else:
            super().closeEvent(e)


def _close_button(slot, tip: str) -> QToolButton:
    b = QToolButton()
    b.setIcon(icons.icon("close", theme.TEXT_2, 18))
    b.setFixedSize(34, 34)
    b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet(f"QToolButton {{ background: {theme.CARD}; border: 1px solid {theme.BORDER}; }}"
                    f"QToolButton:hover {{ background: #7F1D1D; border-color: #B91C1C; }}")
    b.clicked.connect(slot)
    return b


def _is_video(path: str) -> bool:
    from openkick.ui.welcome import VIDEO_EXT
    return Path(path).suffix.lower() in VIDEO_EXT


class VideoCard(QFrame):
    """One chosen video: thumbnail, name and a button to take it out."""
    removed = Signal(str)

    def __init__(self, n: int, path: str, parent=None):
        super().__init__(parent)
        from openkick.ui.welcome import thumbnail
        self.path = path
        self.setFixedSize(196, 156)
        self.setStyleSheet(f"VideoCard {{ background: {theme.CARD}; border: 1px solid {theme.BORDER}; border-radius: 12px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(4)
        thumb = QLabel()
        thumb.setPixmap(thumbnail(path, 176, 99))
        thumb.setFixedSize(176, 99)
        lay.addWidget(thumb)
        name = Path(path).name
        row = QHBoxLayout()
        num_lb = QLabel(tr("Video {n}", n=n))
        num_lb.setStyleSheet("font-weight: 600; font-size: 12px;")
        row.addWidget(num_lb)
        row.addStretch(1)
        lay.addLayout(row)
        nm = QLabel(name if len(name) < 27 else name[:12] + "…" + name[-12:])
        nm.setToolTip(path)
        nm.setStyleSheet(f"color: {theme.FAINT}; font-size: 11px;")
        lay.addWidget(nm)
        x = QToolButton(self)
        x.setIcon(icons.icon("close", theme.TEXT, 14))
        x.setFixedSize(24, 24)
        x.move(self.width() - 32, 8)
        x.setToolTip(tr("Remove this video"))
        x.setCursor(Qt.PointingHandCursor)
        x.setStyleSheet("QToolButton { background: rgba(11, 13, 17, 215); border-radius: 12px; }"
                        "QToolButton:hover { background: #7F1D1D; }")
        x.clicked.connect(lambda: self.removed.emit(self.path))


class VideoPicker(QFrame):
    """Where the videos are dropped or chosen (two or more)."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from openkick.multiview.session import MAX_VIEWS
        self.max = MAX_VIEWS
        self.paths: list[str] = []
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(200)
        self._style(False)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(16, 16, 16, 14)
        self.lay.setSpacing(10)
        self.body = QWidget()
        self.body.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.lay.addWidget(self.body)
        self._rebuild()

    def _style(self, hot: bool):
        self.setStyleSheet(f"VideoPicker {{ background: {'#171D12' if hot else theme.CARD_2}; border: 1.5px dashed "
                           f"{theme.ACCENT if hot else theme.BORDER}; border-radius: 14px; }}")

    def _rebuild(self):
        self.body.hide()
        self.body.deleteLater()
        self.body = QWidget()
        self.lay.insertWidget(0, self.body)
        self.setCursor(Qt.ArrowCursor if self.paths else Qt.PointingHandCursor)
        v = QVBoxLayout(self.body)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        if not self.paths:
            v.addStretch(1)
            icon = QLabel()
            icon.setPixmap(icons.pixmap("upload", theme.ACCENT, 34))
            icon.setAlignment(Qt.AlignCenter)
            v.addWidget(icon)
            t = QLabel(tr("Drop the videos here"))
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet("font-size: 16px; font-weight: 600;")
            v.addWidget(t)
            sub = QLabel(tr("Two or more, all at once or one by one · or click to choose them"))
            sub.setAlignment(Qt.AlignCenter)
            sub.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
            v.addWidget(sub)
            v.addStretch(1)
            for w in (icon, t, sub):
                w.setAttribute(Qt.WA_TransparentForMouseEvents)
        else:
            grid = QGridLayout()
            grid.setSpacing(12)
            for i, path in enumerate(self.paths):
                card = VideoCard(i + 1, path)
                card.removed.connect(self.remove)
                grid.addWidget(card, i // 3, i % 3)
            if len(self.paths) < self.max:
                add = QPushButton(icons.icon("plus", theme.ACCENT, 22), "  " + tr("Add video"))
                add.setFixedSize(196, 156)
                add.setCursor(Qt.PointingHandCursor)
                add.setStyleSheet(f"QPushButton {{ background: transparent; border: 1.5px dashed {theme.BORDER}; border-radius: 12px;"
                                  f" color: {theme.MUTED}; }} QPushButton:hover {{ border-color: {theme.ACCENT}; color: {theme.TEXT}; }}")
                add.clicked.connect(self.browse)
                i = len(self.paths)
                grid.addWidget(add, i // 3, i % 3)
            wrap = QHBoxLayout()
            wrap.addStretch(1)
            wrap.addLayout(grid)
            wrap.addStretch(1)
            v.addLayout(wrap)
            hint = QLabel(tr("{n} videos · you can drop more here (up to {max})", n=len(self.paths), max=self.max)
                          if len(self.paths) < self.max else tr("{n} videos (the maximum)", n=len(self.paths)))
            hint.setAlignment(Qt.AlignCenter)
            hint.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
            hint.setAttribute(Qt.WA_TransparentForMouseEvents)
            v.addWidget(hint)
        self.changed.emit()

    def add(self, paths: list) -> int:
        """Add video files (others are ignored). Returns how many were added."""
        added = 0
        for path in paths:
            path = str(path)
            if not path or path in self.paths or not _is_video(path) or not Path(path).exists():
                continue
            if len(self.paths) >= self.max:
                break
            self.paths.append(path)
            added += 1
        if added:
            self._rebuild()
        return added

    def remove(self, path: str):
        if path in self.paths:
            self.paths.remove(path)
            self._rebuild()

    def clear(self):
        self.paths = []
        self._rebuild()

    def browse(self):
        st = QSettings()
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Choose the videos"), st.value("last_dir", ""), VIDEO_FILTER)
        if paths:
            st.setValue("last_dir", str(Path(paths[0]).parent))
            self.add(paths)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and not self.paths:        # with videos in, the "add" card does this
            self.browse()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._style(True)

    def dragLeaveEvent(self, _):
        self._style(False)

    def dropEvent(self, e):
        self._style(False)
        self.add([u.toLocalFile() for u in e.mimeData().urls()])
        e.acceptProposedAction()


class SetupPage(QWidget):
    start = Signal(list, float, str)
    openSession = Signal()
    openFile = Signal(str)            # a saved session dropped on the page
    closeRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        self.setAcceptDrops(True)                         # a drop anywhere on the page counts
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.setContentsMargins(14, 10, 14, 0)
        top.addStretch(1)
        top.addWidget(_close_button(self.closeRequested, tr("Close (Esc)")))
        root.addLayout(top)
        outer = QHBoxLayout()
        outer.addStretch(1)
        box = QWidget()
        box.setFixedWidth(680)
        outer.addWidget(box)
        outer.addStretch(1)
        root.addLayout(outer, 1)
        col = QVBoxLayout(box)
        col.setSpacing(16)
        col.addStretch(1)
        title = QLabel(tr("3D from several videos"))
        title.setFont(theme.display(30))
        title.setAlignment(Qt.AlignCenter)
        col.addWidget(title)
        intro = QLabel(tr("Record the same movement with two or more cameras at the same time, from different angles "
                          "(for example from the side and from the front). Nothing has to be calibrated: "
                          "OpenKick finds where the cameras were from the body itself."))
        intro.setWordWrap(True)
        intro.setAlignment(Qt.AlignCenter)
        intro.setStyleSheet(f"color: {theme.MUTED};")
        col.addWidget(intro)
        self.picker = VideoPicker()
        self.picker.changed.connect(self._refresh)
        col.addWidget(self.picker)
        opts = QHBoxLayout()
        opts.setSpacing(10)
        opts.addStretch(1)
        opts.addWidget(QLabel(tr("Height of the person")))
        self.height = QDoubleSpinBox()
        self.height.setRange(80, 230)
        self.height.setDecimals(0)
        self.height.setSuffix(" cm")
        self.height.setValue(float(QSettings().value("multiview/height_cm", 170)))
        self.height.setFixedWidth(100)
        self.height.setToolTip(tr("It gives the reconstruction its real size"))
        opts.addWidget(self.height)
        opts.addSpacing(16)
        opts.addWidget(QLabel(tr("Detection")))
        self.mode = QComboBox()
        for key, label in MODES:
            self.mode.addItem(tr(label), key)
        opts.addWidget(self.mode)
        opts.addStretch(1)
        col.addLayout(opts)
        more = QHBoxLayout()
        more.setSpacing(10)
        more.addStretch(1)
        more.addWidget(QLabel(tr("Activity")))
        self.activity = QComboBox()
        from openkick.core import templates as TPL
        from openkick.core.store import activity_available
        for key, label in ACTIVITIES:
            if key in ("", "other") or activity_available(key):
                self.activity.addItem(tr(label), key)
        for tpl in TPL.list_templates():                     # every other template: its angles in 3D
            if tpl.id not in ("cycling", "gait", "blank") and tpl.angles:
                self.activity.insertItem(self.activity.count() - 1, tpl.label, tpl.id)
        last = QSettings().value("multiview/activity", "")
        self.activity.setCurrentIndex(max(0, self.activity.findData(last)))
        self.activity.setToolTip(tr("What is analysed in the 3D: its cycles and phases. Automatic tells pedalling from walking."))
        more.addWidget(self.activity)
        more.addSpacing(16)
        from openkick.core.subject import Subject
        self.subject = Subject()
        more.addWidget(QLabel(tr("Patient")))
        self.patient = QComboBox()
        self.patient.setToolTip(tr("A patient saved in another analysis: their record, height and measured lengths "
                                   "come with them"))
        self.patient.activated.connect(self._pick_patient)
        more.addWidget(self.patient)
        self.subject_btn = QPushButton(icons.icon("user", theme.TEXT, 16), " " + tr("Patient record…"))
        self.subject_btn.setToolTip(tr("Name, height, weight and the lengths measured on the person (optional)"))
        self.subject_btn.clicked.connect(self._edit_subject)
        more.addWidget(self.subject_btn)
        more.addStretch(1)
        self.reload_patients()
        col.addLayout(more)
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.warn.setAlignment(Qt.AlignCenter)
        self.warn.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.warn.setStyleSheet(f"color: {theme.WARN_TEXT}; background: #2A2016; border-radius: 8px; padding: 10px;")
        col.addWidget(self.warn)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.open_btn = QPushButton(tr("Open a saved 3D session…"))
        self.open_btn.clicked.connect(self.openSession)
        btns.addWidget(self.open_btn)
        self.go = QPushButton(tr("Reconstruct in 3D"))
        self.go.setProperty("primary", True)
        self.go.setMinimumWidth(200)
        self.go.clicked.connect(self._start)
        btns.addWidget(self.go)
        btns.addStretch(1)
        col.addLayout(btns)
        note = QLabel(tr("Experimental. Checked against a motion-capture laboratory for squats, jumps, sit-to-stand "
                         "and walking with phones on tripods; other movements and cameras are not validated yet."))
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
        col.addWidget(note)
        col.addStretch(1)
        self._refresh()

    def _refresh(self):
        miss = multiview.missing()
        self.warn.setVisible(bool(miss))
        if miss:
            self.warn.setText(tr("This tool needs extra components that are not installed: {names}. "
                                 "Install them with:  pip install {names}", names=" ".join(miss)))
        n = len(self.picker.paths)
        self.go.setEnabled(not miss and n >= 2)
        self.go.setToolTip("" if n >= 2 else tr("Add at least two videos"))

    def showEvent(self, e):
        super().showEvent(e)
        self.reload_patients()                 # someone may have been saved since (in another analysis)

    def reload_patients(self):
        """The saved patients, to pick one for this analysis."""
        from openkick.core import patients
        from openkick.ui.subject import patient_label
        self.patient.blockSignals(True)
        self.patient.clear()
        self.patient.addItem(tr("New patient"), "")
        for row in patients.every():
            self.patient.addItem(patient_label(row["subject"]), row["id"])
        self.patient.setCurrentIndex(max(0, self.patient.findData(self.subject.id)) if self.subject.id else 0)
        self.patient.blockSignals(False)

    def _pick_patient(self, _=None):
        from openkick.core import patients
        from openkick.core.subject import Subject
        found = patients.get(self.patient.currentData() or "")
        self._take(found["subject"] if found else Subject())

    def _take(self, subject):
        """This record for the analysis: its height goes to the height box."""
        self.subject = subject
        if subject.height_m:
            self.height.setValue(round(subject.height_m * 100))
        self.subject_btn.setText(" " + (subject.name + " ✓" if subject.name else tr("Patient record…")))
        self.reload_patients()

    def _edit_subject(self):
        from openkick.ui.subject import edit_subject
        self.subject.height_m = self.subject.height_m or self.height.value() / 100
        new = edit_subject(self, self.subject, self.activity.currentData() or None)
        if new is not None:
            self._take(new)

    def _start(self):
        QSettings().setValue("multiview/height_cm", self.height.value())
        QSettings().setValue("multiview/activity", self.activity.currentData())
        self.subject.height_m = self.height.value() / 100
        self.start.emit(list(self.picker.paths), self.height.value() / 100, self.mode.currentData())

    def dragEnterEvent(self, e):
        self.picker.dragEnterEvent(e)

    def dragLeaveEvent(self, e):
        self.picker.dragLeaveEvent(e)

    def dropEvent(self, e):
        saved = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile().lower().endswith(".ok3d")]
        if saved:
            self.picker.dragLeaveEvent(e)
            e.acceptProposedAction()
            self.openFile.emit(saved[0])
            return
        self.picker.dropEvent(e)


# What the 3D can analyse: (template id, name). "" finds pedalling or walking by itself.
ACTIVITIES = [("", "Automatic"), ("cycling", "Cycling"), ("gait", "Gait"), ("other", "Other (correction only)")]
# the 3D smoothing: cut-off (Hz, 0 none) and its name
SMOOTHING = [(0.0, "Off (the points as triangulated)"), (12.0, "Light (12 Hz)"), (8.0, "Normal (8 Hz)"),
             (6.0, "Strong (6 Hz)")]
FEET = [("", "Free (as the cameras see them)"), ("rigid", "Rigid, in one piece (stiff sole)")]

# Messages of Session.read(), listed here so they are translated
SESSION_ERRORS = ("this is not an OpenKick 3D session, or the file is damaged",
                  "this session was saved with a newer version of OpenKick")


# ── side panels ────────────────────────────────────────────────────────


def _card() -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("Card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)
    return f, lay


def _section(text: str) -> QLabel:
    lb = QLabel(text.upper())
    lb.setObjectName("SectionTitle")
    return lb


def _scrolling(panel: QWidget) -> QVBoxLayout:
    panel.setObjectName("Panel")
    outer = QVBoxLayout(panel)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    outer.addWidget(scroll)
    body = QWidget()
    scroll.setWidget(body)
    lay = QVBoxLayout(body)
    lay.setContentsMargins(14, 14, 14, 14)
    lay.setSpacing(12)
    return lay


def _value_row(lay: QVBoxLayout, text: str, tip: str = "") -> QLabel:
    r = QHBoxLayout()
    lb = QLabel(text)
    lb.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
    lb.setToolTip(tip)
    val = QLabel()
    val.setFont(theme.mono(12))
    r.addWidget(lb, 1)
    r.addWidget(val)
    lay.addLayout(r)
    val.name_label = lb                    # so that the whole row can be hidden
    return val


class SidePanel(QWidget):
    """Step 1: quality of the reconstruction and the tools to correct it."""

    def __init__(self, ctl: MultiviewController, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        lay = _scrolling(self)

        card, c = _card()
        head = QHBoxLayout()
        self.dot = QLabel()
        self.dot.setFixedSize(12, 12)
        head.addWidget(self.dot)
        self.level = QLabel()
        self.level.setStyleSheet("font-weight: 600;")
        head.addWidget(self.level, 1)
        c.addLayout(head)
        self.rows = {}
        for key, text, tip in (
                ("mismatch", tr("The videos disagree by"), tr("Typical distance between where each camera puts the same point. Lower is better")),
                ("legs", tr("Leg bones change length by"), tr("A real bone keeps its length: the less it changes, the better the 3D")),
                ("done", tr("Points reconstructed"), tr("Points seen well enough by at least two cameras")),
                ("cams", tr("Distance between cameras"), tr("Between the first camera and the farthest one, estimated from the height of the person"))):
            self.rows[key] = _value_row(c, text, tip)
        lay.addWidget(card)

        # gait: only when the person walks
        self.gait_title = _section(tr("Gait"))
        lay.addWidget(self.gait_title)
        self.gait_card, c = _card()
        self.gait_now = QLabel()
        self.gait_now.setStyleSheet("font-weight: 600;")
        c.addWidget(self.gait_now)
        from openkick.core import gait as GC
        self.gait_rows = {key: _value_row(c, GC.name(key)) for key in ("cadence", "stance", "step_length", "step_width")}
        from openkick.core import cycling as CC
        self.cycling_rows = {key: _value_row(c, CC.name(key)) for key in ("cadence", "knee_bottom", "knee_top", "ankle_bottom",
                                                                          "foot_bottom")}
        self.gait_btn = QPushButton(tr("See the gait analysis"))
        self.gait_btn.setToolTip(tr("Every stride, left against right, and how the recording was checked"))
        self.gait_btn.clicked.connect(self._show_gait)
        c.addWidget(self.gait_btn)
        lay.addWidget(self.gait_card)

        lay.addWidget(_section(tr("Review")))
        card, c = _card()
        self.todo = QLabel()
        self.todo.setWordWrap(True)
        self.todo.setMinimumHeight(92)
        self.todo.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        c.addWidget(self.todo)
        self.next_btn = QPushButton(tr("Next point to check"))
        self.next_btn.setProperty("primary", True)
        self.next_btn.setToolTip(tr("Jump to the next doubtful point (Tab)"))
        self.next_btn.clicked.connect(ctl.next_problem)
        c.addWidget(self.next_btn)
        self.list_btn = QPushButton(tr("List of points to check…"))
        self.list_btn.setToolTip(tr("Every doubtful stretch, the worst first, with its exact frames and a picture"))
        self.list_btn.clicked.connect(self.show_review)
        c.addWidget(self.list_btn)
        self.review_win = None
        self.swap_btn = QPushButton(tr("Look for left/right confusions"))
        self.swap_btn.setToolTip(tr("Finds frames where the detector took one leg or arm for the other, and fixes them"))
        self.swap_btn.clicked.connect(self._fix_swaps)
        c.addWidget(self.swap_btn)
        lay.addWidget(card)

        lay.addWidget(_section(tr("Selected point")))
        card, c = _card()
        self.pt_name = QLabel()
        self.pt_name.setStyleSheet("font-weight: 600;")
        c.addWidget(self.pt_name)
        self.pt_info = QLabel()
        self.pt_info.setWordWrap(True)
        self.pt_info.setMinimumHeight(132)
        self.pt_info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.pt_info.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        c.addWidget(self.pt_info)
        self.revert_btn = QPushButton(tr("Back to the automatic detection"))
        self.revert_btn.clicked.connect(self._revert)
        c.addWidget(self.revert_btn)
        self.snap_btn = QPushButton(tr("Bring it where its bones say, in every video"))
        self.snap_btn.setToolTip(tr("Puts the point, in every video at once, where its thigh and shank have their usual "
                                    "length (the circle of the length guides). Ctrl+Z undoes it."))
        self.snap_btn.clicked.connect(self._snap)
        c.addWidget(self.snap_btn)
        self.learn_info = QLabel()
        self.learn_info.setWordWrap(True)
        self.learn_info.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        c.addWidget(self.learn_info)
        self.learn_btn = QPushButton(tr("Learn its path over the pedal stroke and apply it to every stroke"))
        self.learn_btn.setToolTip(tr("From your corrections of this point in this video over one whole pedal stroke, "
                                     "the program learns where the detector misplaces it at each moment of the stroke "
                                     "and corrects every other stroke the same way. Your points are not moved. "
                                     "Ctrl+Z undoes it."))
        self.learn_btn.clicked.connect(self._learn)
        c.addWidget(self.learn_btn)
        lay.addWidget(card)

        lay.addWidget(_section(tr("When I move a point")))
        card, c = _card()
        self.scope = QButtonGroup(self)
        for key, text, tip in (("frame", tr("Correct only this frame"), tr("The point moves only in the frame you are looking at")),
                               ("neighbors", tr("Also the neighboring frames"), tr("The correction fades out over the 3 frames before and after")),
                               ("stretch", tr("The whole doubtful stretch"),
                                tr("The same correction in every frame of the stretch marked as doubtful for this point, "
                                   "fading out over 3 frames on either side"))):
            rb = QRadioButton(text)
            rb.setToolTip(tip)
            rb.setChecked(key == ctl.scope)
            rb.toggled.connect(lambda on, k=key: on and setattr(ctl, "scope", k))
            self.scope.addButton(rb)
            c.addWidget(rb)
        self.follow = QCheckBox(tr("The other videos follow my correction"))
        self.follow.setToolTip(tr("When you drop a point, it moves in the other videos to where your correction puts it "
                                  "in 3D, as little as it takes, so they do not come apart. Points you placed by hand "
                                  "are never moved. The ones moved this way have a dotted ring."))
        self.follow.setChecked(ctl.follow)
        self.follow.toggled.connect(self._follow)
        c.addWidget(self.follow)
        lay.addWidget(card)

        lay.addWidget(_section(tr("Synchronization")))
        card, c = _card()
        self.sync = QLabel()
        self.sync.setWordWrap(True)
        self.sync.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        c.addWidget(self.sync)
        r = QHBoxLayout()
        self.sync_view = QComboBox()
        self.sync_view.setToolTip(tr("The video to move in time"))
        r.addWidget(self.sync_view, 1)
        for text, d, tip in (("−", -1, tr("A little earlier")), ("+", 1, tr("A little later"))):
            b = QPushButton(text)
            b.setFixedWidth(44)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, d=d: self._nudge(d))
            r.addWidget(b)
        c.addLayout(r)
        auto = QPushButton(tr("Synchronize again automatically"))
        auto.setToolTip(tr("Find the time offsets again from the movement"))
        auto.clicked.connect(self._resync)
        c.addWidget(auto)
        recal = QPushButton(tr("Place the cameras again"))
        recal.setToolTip(tr("Work out where the cameras are again, with your corrections. A correction alone "
                            "does not move them, so the other points stay where they are."))
        recal.clicked.connect(self._recalibrate)
        c.addWidget(recal)
        lay.addWidget(card)

        lay.addWidget(_section(tr("3D smoothing")))
        card, c = _card()
        self.smoothing = QComboBox()
        for hz, text in SMOOTHING:
            self.smoothing.addItem(tr(text), hz)
        self.smoothing.setToolTip(tr("Takes away the jumps of a frame or two and the tremor faster than the movement. "
                                     "The detection and your corrections are not changed, and the doubtful points "
                                     "are still found before smoothing."))
        self.smoothing.currentIndexChanged.connect(self._smoothing)
        c.addWidget(self.smoothing)
        hint = QLabel(tr("For fast gestures (a kick, a jump) use Light or Off: a stronger smoothing flattens their peaks."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        c.addWidget(hint)
        lay.addWidget(card)

        lay.addWidget(_section(tr("Feet")))
        card, c = _card()
        self.feet = QComboBox()
        for key, text in FEET:
            self.feet.addItem(tr(text), key)
        self.feet.setToolTip(tr("Rigid: heel, toes and ankle keep one shape and turn together about the ankle, like a "
                                "shoe with a stiff sole. The foot still turns as much as it does: only its shape stops "
                                "changing."))
        self.feet.currentIndexChanged.connect(self._feet)
        c.addWidget(self.feet)
        note = QLabel(tr("For cycling shoes or any stiff sole. With soft shoes the forefoot bends as it pushes, "
                         "and a rigid foot would not show it."))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        c.addWidget(note)
        lay.addWidget(card)
        lay.addStretch(1)

        ctl.changed.connect(self.refresh)
        ctl.rowChanged.connect(lambda _: self.refresh_point())
        ctl.rowChanged.connect(lambda _: self.refresh_gait())
        ctl.selectionChanged.connect(self.refresh_point)
        ctl.sessionLoaded.connect(self._loaded)

    def _loaded(self):
        s = self.ctl.session
        self.sync_view.clear()
        for i in range(1, len(s.views)):
            self.sync_view.addItem(tr("Video {n}", n=i + 1), i)
        self.smoothing.blockSignals(True)
        k = min(range(len(SMOOTHING)), key=lambda i: abs(SMOOTHING[i][0] - (s.smooth_hz or 0.0)))
        self.smoothing.setCurrentIndex(k)
        self.smoothing.blockSignals(False)
        self.feet.blockSignals(True)
        self.feet.setCurrentIndex(max(0, self.feet.findData(s.feet or "")))
        self.feet.blockSignals(False)
        self.refresh()

    def show_review(self):
        from openkick.ui.review import ReviewWindow
        if self.review_win is None:
            self.review_win = ReviewWindow(self.ctl, self.window())
        self.review_win.show()
        self.review_win.raise_()
        self.review_win.activateWindow()

    def _feet(self, _=None):
        s = self.ctl.session
        if not s:
            return
        s.feet = self.feet.currentData() or ""
        self.ctl.version += 1
        self.ctl.recompute(cameras=False)

    def _smoothing(self, _=None):
        s = self.ctl.session
        if not s:
            return
        s.smooth_hz = float(self.smoothing.currentData() or 0.0)
        self.ctl.version += 1
        self.ctl.recompute(cameras=False)

    def refresh(self):
        s = self.ctl.session
        if not s or not s.solved:
            return
        q = s.quality()
        text, color = LEVELS[q["level"]]
        self.level.setText(tr(text))
        self.dot.setStyleSheet(f"background: {color}; border-radius: 6px;")
        self.rows["mismatch"].setText(f"{num(q['mismatch_cm'])} cm")
        self.rows["legs"].setText(f"{num(q['legs_variation'], 0)} %")
        self.rows["done"].setText(f"{num(100 * q['reconstructed'], 0)} %")
        self.rows["cams"].setText(f"{num(s.scale * max(float(np.linalg.norm(t)) for _, t in s.cams), 1)} m")
        bad = int((s.frame_error() > PROBLEM_CM).sum())
        if q["problems"]:
            self.todo.setText(tr("{n} points in {f} frames worth checking: the videos disagree, a bone changes "
                                 "length or the point jumps, by more than {cm} cm.", n=q["problems"], f=bad, cm=num(PROBLEM_CM, 0))
                              + (" " + tr("{n} corrected by hand.", n=q["manual"]) if q["manual"] else ""))
        else:
            self.todo.setText(tr("Nothing to check: the videos agree in every frame."))
        self.next_btn.setEnabled(bool(q["problems"]))
        self.refresh_gait()
        self.sync.setText("\n".join(
            tr("Video {n} starts {s} s {when} video 1.", n=i + 1, s=num(abs(off), 3), when=tr("after") if off >= 0 else tr("before"))
            for i, off in enumerate(s.offsets) if i))
        self.refresh_point()

    def refresh_gait(self):
        from openkick.core import cycling as CC
        from openkick.core import gait as GC
        g = self.ctl.activity()
        cycling = self.ctl.activity_kind() == "cycling"
        if self.ctl.activity_kind() == "template":           # any other template: its angles now
            self._refresh_template()
            return
        self.gait_title.setVisible(g is not None)
        self.gait_card.setVisible(g is not None)
        if g is None:
            return
        self.gait_title.setText(tr("Cycling") if cycling else tr("Gait"))
        self.gait_btn.setText(report_label(self.ctl))
        self.gait_btn.setToolTip(tr("Every pedal stroke, both legs and how they vary") if cycling else
                                 tr("Every stride, left against right, and how the recording was checked"))
        for rows, show in ((self.gait_rows, not cycling), (self.cycling_rows, cycling)):
            for val in rows.values():
                val.setVisible(show)
                val.name_label.setVisible(show)
        row = self.ctl.row
        ph, sub = g.phase_at(row), g.subphase_at(row)
        t = self.ctl.session.t
        if ph and sub:
            self.gait_now.setText(f"{ph.name} · {sub.name} ({num(t[sub.start] - t[0], 2)}–{num(t[sub.end] - t[0], 2)} s)")
            self.gait_now.setStyleSheet(f"font-weight: 600; color: {sub.color};")
        else:
            self.gait_now.setText(tr("Outside the strokes") if cycling else tr("Outside the strides"))
            self.gait_now.setStyleSheet(f"font-weight: 600; color: {theme.MUTED};")
        if cycling:
            for key, val in self.cycling_rows.items():
                mid, sd = CC.typical(s.get(key) for s in g.strokes)
                val.setText(f"{CC.show(mid, key)} ± {sd:.1f}" if np.isfinite(mid) else "—")
            return
        for key, val in self.gait_rows.items():
            mid, _ = GC.typical(s.get(key) for s in g.strides)
            val.setText(GC.show(mid, key))

    def _refresh_template(self):
        from openkick.core import templates as TPL
        s = self.ctl.session
        tpl = TPL.find(s.template)
        self.gait_title.setVisible(tpl is not None)
        self.gait_card.setVisible(tpl is not None)
        if tpl is None:
            return
        self.gait_title.setText(tpl.label)
        self.gait_btn.setText(report_label(self.ctl))
        self.gait_btn.setToolTip(tr("Range of every angle of the template, in 3D"))
        for rows in (self.gait_rows, self.cycling_rows):
            for val in rows.values():
                val.setVisible(False)
                val.name_label.setVisible(False)
        from openkick.multiview.template3d import angles
        now = [f"{name}: {v[self.ctl.row]:.0f}°" for name, v in angles(s.X, s.t, tpl) if np.isfinite(v[self.ctl.row])]
        self.gait_now.setText("   ".join(now) or "—")
        self.gait_now.setStyleSheet("font-weight: 600;")

    def _show_gait(self):
        show_gait_report(self.ctl, self)

    def refresh_point(self):
        s, sel = self.ctl.session, self.ctl.selected
        if not s or not s.solved:
            return
        self.snap_btn.setEnabled(bool(sel) and any(g["joint"] == sel[1] for g in s.bone_guides(self.ctl.row)))
        self._learning(sel)
        if not sel:
            self.pt_name.setText(tr("None"))
            self.pt_info.setText(tr("Click a point of the skeleton in any video. Drag it to correct it."))
            self.revert_btn.setEnabled(False)
            return
        view, j = sel
        row = self.ctl.row
        f = self.ctl.frame(view)
        v = s.views[view]
        cm = s.err_cm[row, j]
        self.pt_name.setText(joint_name(j))
        if not np.isfinite(cm):
            info = tr("Not reconstructed here: fewer than two cameras see it well. Drag it to its place to use it.")
        else:
            info = tr("The videos disagree by {cm} cm.", cm=num(cm))
            if v.valid()[f, j] and not s.used[view, row, j]:
                info += " " + tr("This video is left out here: it puts the point {cm} cm away from the others.",
                                 cm=num(s.view_error_cm(view, row, j)))
            bone = s.bone_cm[row, j]
            if np.isfinite(bone) and bone > PROBLEM_CM:
                info += " " + tr("The bones that meet here are {cm} cm off their usual length.", cm=num(bone))
            jump = s.jump_cm[row, j]
            if np.isfinite(jump) and jump > PROBLEM_CM:
                info += " " + tr("It jumps {cm} cm away from the frames before and after.", cm=num(jump))
            if s.issue_cm[row, j] > PROBLEM_CM:
                info += " " + tr("If it sits on the body in every video, the cause is a fast movement between "
                                 "frames: dragging will not fix it.")
        if v.source[f, j] == SRC_MANUAL:
            info += " " + tr("Placed by hand in video {n}.", n=view + 1)
        self.pt_info.setText(info)
        self.revert_btn.setEnabled(bool(v.source[f, j] != 1))

    def _learning(self, sel):
        """How far the corrections of the selected point cover a pedal stroke (only when pedalling)."""
        s = self.ctl.session
        info = s.learning(*sel) if sel else None
        self.learn_info.setVisible(info is not None)
        self.learn_btn.setVisible(info is not None)
        if info is None:
            return
        done = len(info["parts"])
        if info["missing"]:
            names = []
            for k in info["missing"]:
                n = tr(STROKE_PARTS[k])
                if n not in names:
                    names.append(n)
            self.learn_info.setText(tr("Pedal stroke: corrected in {done} of {all} parts. To learn its path, correct it "
                                       "also {where}.", done=done, all=len(STROKE_PARTS), where=", ".join(names)))
        else:
            self.learn_info.setText(tr("Pedal stroke: corrected all the way round ({n} points by hand).",
                                       n=int(info["hand"].sum())))
        self.learn_btn.setEnabled(not info["missing"])

    def _learn(self):
        c = self.ctl
        s, sel = c.session, c.selected
        if not sel:
            return
        view, j = sel
        edit = s.learn_path(view, j)
        if edit is None:
            return
        followed = s.follow(view, edit, j) if c.follow else []
        c.push([edit] + followed, tr("Learn the path of {joint}", joint=joint_name(j)))
        L = s.last_learn
        c.message.emit(tr("{joint}: path learned from {hand} corrections and applied to {frames} frames (moved "
                          "{px} px on average; it passes {fit} px from your points). Ctrl+Z undoes it.",
                          joint=joint_name(j), hand=L["hand"], frames=L["frames"], px=num(L["moved_px"], 0),
                          fit=num(L["fit_px"], 0)))

    def _follow(self, on: bool):
        self.ctl.follow = on
        QSettings().setValue("multiview/follow", on)

    def _snap(self):
        c = self.ctl
        s, sel = c.session, c.selected
        if not sel:
            return
        j, row = sel[1], c.row
        g = next((g for g in s.bone_guides(row) if g["joint"] == j), None)
        if g is None:
            c.message.emit(tr("Its bones already have their usual length here"))
            return
        target = g["target"]
        edits = []
        # a video other than the first is read between two of its frames at this moment: moving the
        # nearest one may not be enough, so the point is placed again until the 3D lands on the target
        for _ in range(4):
            for v in range(len(s.views)):
                px = s.project(v, row, target)
                if np.isfinite(px).all():
                    f = s.frame_of(v, row)
                    here = s.views[v].xy[f, j].astype(float)
                    now = s.project(v, row, s.X_raw[row, j])
                    new = px + (here - now) if np.isfinite(here).all() and np.isfinite(now).all() else px
                    edits.append(s.move_point(v, f, j, (float(new[0]), float(new[1]))))
            s.update(recalibrate=False)
            if np.linalg.norm(s.X_raw[row, j] - target) < 0.005:
                break
        if edits:
            c.push(edits, tr("Bring {joint} where its bones say", joint=joint_name(j)))
            c.message.emit(tr("{joint} placed where its bones have their usual length, in every video (Ctrl+Z to undo)",
                              joint=joint_name(j)))

    def _revert(self):
        s, sel = self.ctl.session, self.ctl.selected
        if sel:
            f = self.ctl.frame(sel[0])
            self.ctl.push([s.revert_point(sel[0], f, sel[1])], tr("Revert {joint}", joint=joint_name(sel[1])))

    def _fix_swaps(self):
        edits, n = self.ctl.session.fix_swaps()
        if not n:
            self.ctl.message.emit(tr("No left/right confusions found"))
            return
        self.ctl.push(edits, tr("Exchange left and right"))
        self.ctl.message.emit(tr("Left and right exchanged in {n} frames (Ctrl+Z to undo)", n=n))

    def _nudge(self, direction: int):
        s = self.ctl.session
        i = self.sync_view.currentData() or 1
        s.offsets[i] += direction * float(np.median(np.diff(s.views[i].t))) / 4
        self.ctl.version += 1
        self.ctl.recompute()

    def _recalibrate(self):
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.ctl.version += 1
            self.ctl.recompute(cameras=True)
        finally:
            QApplication.restoreOverrideCursor()
        self.ctl.message.emit(tr("Cameras placed again with your corrections"))

    def _resync(self):
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.ctl.session.solve()
        except ValueError as e:
            QMessageBox.warning(self, APP_NAME, str(e))
        finally:
            QApplication.restoreOverrideCursor()
        self.ctl.version += 1
        self.ctl.recompute(cameras=False)


ANGLE_ROWS = (("Right knee", "knee_angle_r"), ("Left knee", "knee_angle_l"), ("Right hip (flexion)", "hip_flexion_r"),
              ("Left hip (flexion)", "hip_flexion_l"), ("Right ankle", "ankle_angle_r"), ("Left ankle", "ankle_angle_l"))


class OpensimPanel(QWidget):
    """Step 2: state of the OpenSim model, its joint angles and how to take the files away."""
    recompute = Signal()
    saveFiles = Signal()
    exportVideo = Signal(float, str)   # speed (1 real, 0.5 half, ...) and layout ("side" | "cameras")

    def __init__(self, ctl: MultiviewController, view: OsimView, parent=None):
        super().__init__(parent)
        self.ctl, self.view = ctl, view
        self.result: dict | None = None
        lay = _scrolling(self)

        card, c = _card()
        head = QHBoxLayout()
        self.dot = QLabel()
        self.dot.setFixedSize(12, 12)
        head.addWidget(self.dot)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: 600;")
        head.addWidget(self.title, 1)
        c.addLayout(head)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setMinimumHeight(72)
        self.info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        c.addWidget(self.info)
        self.again = QPushButton(tr("Compute the model again"))
        self.again.setToolTip(tr("Send the current 3D points to OpenSim: scaling and inverse kinematics"))
        self.again.clicked.connect(self.recompute)
        c.addWidget(self.again)
        lay.addWidget(card)

        lay.addWidget(_section(tr("Joint angles in this frame")))
        card, c = _card()
        self.angles = {name: _value_row(c, tr(text)) for text, name in ANGLE_ROWS}
        lay.addWidget(card)

        lay.addWidget(_section(tr("Take it to OpenSim")))
        card, c = _card()
        tip = QLabel(tr("Saves the scaled model (.osim), the movement (.mot), the markers (.trc) and the bone shapes "
                        "in one folder, ready to open in the OpenSim application."))
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        c.addWidget(tip)
        self.save_btn = QPushButton(icons.icon("folder", theme.TEXT, 16), " " + tr("Save the OpenSim files…"))
        self.save_btn.clicked.connect(self.saveFiles)
        c.addWidget(self.save_btn)
        lay.addWidget(card)

        lay.addWidget(_section(tr("Export as video")))
        card, c = _card()
        self.video_btns = []
        speeds = ((tr("Real speed"), 1.0), (tr("Slow motion, half speed"), 0.5), (tr("Slow motion, quarter speed"), 0.25))
        for layout, icon, title, what in (
                ("side", "cube", tr("3D points + OpenSim model"),
                 tr("The two halves you see here, side by side, from the angle you have now.")),
                ("cameras", "video", tr("Your videos + OpenSim model"),
                 tr("Each video above and, below it, the model seen from that camera."))):
            b = QPushButton(icons.icon(icon, theme.TEXT, 16), " " + title + "  ▾")
            b.setToolTip(what)
            menu = QMenu(b)
            theme.menu_title(menu, title)
            for text, speed in speeds:
                menu.addAction(text, lambda sp=speed, la=layout: self.exportVideo.emit(sp, la))
            b.setMenu(menu)
            c.addWidget(b)
            d = QLabel(what)
            d.setWordWrap(True)
            d.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
            c.addWidget(d)
            if layout == "side":
                c.addSpacing(6)
            self.video_btns.append(b)
        self.video_btn = self.video_btns[0]
        lay.addWidget(card)

        note = QLabel(tr("Checked against a laboratory: knee and ankle come out within about 3–6°. Hip flexion comes out "
                         "9–16° too high in squats, sit-to-stand and jumps, because the tilt of the pelvis cannot be "
                         "seen from two hip points."))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
        lay.addWidget(note)
        lay.addStretch(1)
        ctl.rowChanged.connect(lambda _: self.refresh_angles())

    def set_state(self, state: str, result: dict | None = None, detail: str = ""):
        """state: "none" | "busy" | "ready" | "stale" | "failed" | "missing"."""
        self.result = result if state in ("ready", "stale") else None
        color = {"ready": theme.OK, "stale": theme.WARN, "busy": theme.ACCENT, "failed": theme.DANGER,
                 "missing": theme.WARN}.get(state, theme.DIM)
        self.dot.setStyleSheet(f"background: {color}; border-radius: 6px;")
        self.title.setText({"none": tr("OpenSim model not computed yet"), "busy": tr("Computing the OpenSim model…"),
                            "ready": tr("OpenSim model ready"), "stale": tr("The model is out of date"),
                            "failed": tr("OpenSim could not build the model"),
                            "missing": tr("OpenSim is not installed")}[state])
        if state in ("ready", "stale") and result:
            text = tr("Model scaled to the person and moved by inverse kinematics. Its markers end up {cm} cm away "
                      "from the 3D points, on average.", cm=num(result["marker_rms_cm"]))
            if state == "stale":
                text = tr("You corrected points after it was computed. Compute it again to see the change.") + " " + text
        else:
            text = detail
        self.info.setText(text)
        self.again.setEnabled(state not in ("busy", "missing"))
        self.again.setProperty("primary", state in ("stale", "none", "failed"))
        self.again.style().unpolish(self.again)
        self.again.style().polish(self.again)
        self.save_btn.setEnabled(state in ("ready", "stale"))
        for b in self.video_btns:
            b.setEnabled(state == "ready")
        self.refresh_angles()

    def refresh_angles(self):
        f = self.view.frame() if self.result else None
        for name, lb in self.angles.items():
            v = self.view.scene.angle(name, f) if f is not None else float("nan")
            lb.setText(f"{num(v)}°" if np.isfinite(v) else "—")


class OpensimWorker(QThread):
    """Runs the OpenSim step in another process."""
    done = Signal(dict)

    def __init__(self, folder: Path, trc: Path, height_m: float):
        super().__init__()
        self.folder, self.trc, self.height = Path(folder), Path(trc), height_m

    def run(self):
        import json
        import subprocess
        import sys
        try:
            proc = subprocess.run([sys.executable, "-m", "openkick.multiview.opensim_job", str(self.folder), str(self.trc),
                                   str(self.height)], capture_output=True, text=True)
            f = self.folder / "result.json"
            res = json.loads(f.read_text()) if f.exists() else {"ok": False, "error": (proc.stderr or proc.stdout)[-1500:]}
        except Exception as e:  # noqa: BLE001 - reported to the user
            res = {"ok": False, "error": str(e)}
        self.done.emit(res)


# ── window ─────────────────────────────────────────────────────────────


class CanvasGrid(QWidget):
    """The videos side by side (two or three in a row; more go in two rows)."""

    def __init__(self, ctl: MultiviewController, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.canvases: list[ViewCanvas] = []
        self.setStyleSheet(f"CanvasGrid {{ background: {theme.LINE}; }}")
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)

    def rebuild(self, count: int):
        for cv in self.canvases:
            cv.detach()                    # before anything else: it must not hear about the next session
            cv.hide()
            cv.setParent(None)
            cv.deleteLater()
        self.canvases = [ViewCanvas(self.ctl, i) for i in range(count)]
        cols = count if count <= 3 else 2 if count == 4 else 3
        for i, cv in enumerate(self.canvases):
            self.grid.addWidget(cv, i // cols, i % cols)

    def show_only(self, view: int | None):
        """One video filling the place of all (None: all of them again)."""
        for i, cv in enumerate(self.canvases):
            cv.setVisible(view is None or i == view)


class MultiviewWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("3D from several videos") + " · " + APP_NAME)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(1500, 900)
        self.setMinimumSize(1000, 640)
        self.ctl = MultiviewController(self)
        self.playing = False
        self.phase = 1
        self.osim_version = -1                 # ctl.version the OpenSim model was computed from
        self.osim_result: dict | None = None
        self.osim_worker: OpensimWorker | None = None
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.setup = SetupPage()
        self.setup.start.connect(self.build)
        self.setup.openSession.connect(self.open_session)
        self.setup.openFile.connect(self.load_session)
        self.setup.closeRequested.connect(self.close)
        self.stack.addWidget(self.setup)
        self.stack.addWidget(self._editor())
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._tick)
        c = self.ctl
        c.rowChanged.connect(self._refresh_transport)
        c.sessionLoaded.connect(self._refresh_transport)
        c.changed.connect(self._changed)
        c.message.connect(self.hint.setText)
        c.focusChanged.connect(self._focus_changed)
        for seq, fn in (("Space", self.toggle_play), ("Right", lambda: c.step(1)), ("Left", lambda: c.step(-1)),
                        ("Shift+Right", lambda: c.step(10)), ("Shift+Left", lambda: c.step(-10)),
                        ("Home", lambda: c.set_row(0)), ("End", lambda: c.set_row(10 ** 9)),
                        ("Tab", lambda: c.next_problem() if self.phase == 1 else None),
                        ("G", lambda: self.shadow_btn.toggle() if c.focus is not None else None),
                        ("Ctrl+S", self.save), ("Ctrl+Shift+S", self.save_as), ("Ctrl+Z", c.undo.undo), ("Ctrl+Shift+Z", c.undo.redo), ("Ctrl+Y", c.undo.redo)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(lambda fn=fn: fn() if self.ctl.session else None)
        esc = QShortcut(QKeySequence("Esc"), self)
        esc.activated.connect(lambda: self.close() if self.stack.currentIndex() == 0 else c.set_focus(None))

    @property
    def canvases(self) -> list:
        return self.grid.canvases

    def _editor(self) -> QWidget:
        c = self.ctl
        page = QWidget()
        page.setObjectName("Page")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # top bar
        top = QWidget()
        top.setObjectName("Panel")
        top.setFixedHeight(54)
        t = QHBoxLayout(top)
        t.setContentsMargins(14, 0, 10, 0)
        t.setSpacing(8)
        back = QToolButton()
        back.setIcon(icons.icon("chevron_left", theme.TEXT_2, 18))
        back.setToolTip(tr("Choose other videos"))
        back.clicked.connect(self.back_to_setup)
        t.addWidget(back)
        title = QLabel(tr("3D from several videos"))
        title.setFont(theme.display(17, 600))
        t.addWidget(title)
        chip = QLabel(tr("EXPERIMENTAL"))
        chip.setStyleSheet(f"color: {theme.WARN_TEXT}; background: #2A2016; border-radius: 6px; padding: 0 8px; font-size: 11px; font-weight: 600;")
        chip.setFixedHeight(22)
        t.addWidget(chip, 0, Qt.AlignVCenter)
        self.name_lbl = QLabel()
        self.name_lbl.setStyleSheet(f"color: {theme.MUTED};")
        self.name_lbl.setMaximumWidth(260)
        t.addWidget(self.name_lbl)
        t.addSpacing(10)
        self.phase_btns = {}
        group = QButtonGroup(self)
        for k, text, tip in ((1, tr("1 · Correct the points"), tr("The videos with the skeleton: drag the points that are wrong")),
                             (2, tr("2 · OpenSim model"), tr("The 3D points next to the OpenSim model they produce"))):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(f"QPushButton {{ background: transparent; border: 1px solid {theme.BORDER}; min-height: 30px; padding: 0 14px; }}"
                            f"QPushButton:checked {{ background: #171D12; border-color: {theme.ACCENT}; color: {theme.ACCENT}; font-weight: 600; }}")
            b.clicked.connect(lambda _=False, k=k: self.set_phase(k))
            group.addButton(b)
            t.addWidget(b)
            self.phase_btns[k] = b
        self.phase_btns[1].setChecked(True)
        t.addStretch(1)
        for name, action, tip in (("undo", c.undo.undo, tr("Undo (Ctrl+Z)")), ("redo", c.undo.redo, tr("Redo (Ctrl+Y)"))):
            b = QToolButton()
            b.setIcon(icons.icon(name, theme.TEXT_2, 18))
            b.setToolTip(tip)
            b.setFixedSize(34, 34)
            b.clicked.connect(action)
            t.addWidget(b)
            setattr(self, f"{name}_btn", b)
        c.undo.canUndoChanged.connect(self.undo_btn.setEnabled)
        c.undo.canRedoChanged.connect(self.redo_btn.setEnabled)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.ghost_btn = QPushButton(icons.icon("cube", theme.TEXT, 16), " " + tr("3D result"))
        self.ghost_btn.setCheckable(True)
        self.ghost_btn.setChecked(True)
        self.ghost_btn.setToolTip(tr("White lines: the 3D skeleton seen from each camera. Where it leaves the body, something is wrong"))
        self.ghost_btn.toggled.connect(self._toggle_ghost)
        t.addWidget(self.ghost_btn)
        self.shadow_btn = QPushButton(icons.icon("eye", theme.TEXT, 16), " " + tr("Shadow"))
        self.shadow_btn.setCheckable(True)
        self.shadow_btn.setChecked(c.shadow)
        self.shadow_btn.setToolTip(tr("Faint skeletons of the previous (amber) and next (blue) frame, to line up a correction with them (G)"))
        self.shadow_btn.toggled.connect(c.set_shadow)
        self.shadow_btn.hide()
        t.addWidget(self.shadow_btn)
        self.lengths_btn = QPushButton(icons.icon("target", theme.TEXT, 16), " " + tr("Lengths"))
        self.lengths_btn.setCheckable(True)
        self.lengths_btn.setChecked(c.show_lengths)
        self.lengths_btn.setToolTip(tr("Thighs and shanks off their usual length by 5 % or more: in red what is too much, "
                                       "in dashed yellow what is missing, up to where the point should be"))
        self.lengths_btn.toggled.connect(self._toggle_lengths)
        t.addWidget(self.lengths_btn)
        patient = QPushButton(icons.icon("user", theme.TEXT, 16), " " + tr("Patient"))
        patient.setToolTip(tr("The patient's record: height, weight and the lengths measured on them"))
        patient.clicked.connect(self.edit_subject)
        t.addWidget(patient)
        save = QPushButton(icons.icon("save", theme.TEXT, 16), " " + tr("Save"))
        save.setToolTip(tr("Save the session with your corrections (Ctrl+S)"))
        save.clicked.connect(self.save)
        t.addWidget(save)
        more = QToolButton()
        more.setIcon(icons.icon("chevron_down", theme.TEXT_2, 16))
        more.setFixedSize(26, 34)
        more.setToolTip(tr("Save with another name"))
        more.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(more)
        menu.addAction(icons.icon("save", theme.TEXT_2, 16), tr("Save as…") + "    Ctrl+Shift+S", self.save_as)
        more.setMenu(menu)
        t.addWidget(more)
        from openkick.ui.report import report_menu
        notes = QPushButton(icons.icon("note", theme.TEXT, 16), " " + tr("Notes"))
        notes.setToolTip(tr("Your name and your observations and recommendations: kept with the session, they go "
                            "into its report"))
        notes.clicked.connect(self.edit_notes)
        t.addWidget(notes)
        report = QPushButton(icons.icon("pdf", theme.TEXT, 16), " " + tr("Report") + "  ▾")
        report.setToolTip(tr("Make the report of the analysis: PDF, plain text or JSON (the numbers, for research)"))
        report.setMenu(report_menu(self, self.export_report))
        t.addWidget(report)
        exp = QPushButton(icons.icon("export", theme.ON_ACCENT, 16), " " + tr("Export to OpenSim"))
        exp.setProperty("primary", True)
        exp.setToolTip(tr("Save the 3D points as an OpenSim marker file (.trc)"))
        exp.clicked.connect(self.export_trc)
        t.addWidget(exp)
        t.addSpacing(6)
        t.addWidget(_close_button(self.close, tr("Close this window")))
        lay.addWidget(top)
        # centre: one page per step
        self.pages = QStackedWidget()
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(2)
        split.setStyleSheet(f"QSplitter::handle {{ background: {theme.LINE}; }}")
        self.grid = CanvasGrid(c)
        split.addWidget(self.grid)
        right = QSplitter(Qt.Vertical)
        right.setHandleWidth(2)
        self.view3d = Skeleton3D(c)
        right.addWidget(self.view3d)
        self.panel = SidePanel(c)
        right.addWidget(self.panel)
        right.setSizes([380, 420])
        right.setMinimumWidth(300)
        right.setMaximumWidth(420)
        split.addWidget(right)
        split.setSizes([1120, 330])
        self.pages.addWidget(split)
        split2 = QSplitter(Qt.Horizontal)
        split2.setHandleWidth(2)
        split2.setStyleSheet(f"QSplitter::handle {{ background: {theme.LINE}; }}")
        orbit = Orbit(self)                               # the two 3D views turn together
        self.points3d = Skeleton3D(c, orbit, "3D POINTS")
        self.points3d.show_selection = False
        self.osim_view = OsimView(c, orbit)
        split2.addWidget(self.points3d)
        split2.addWidget(self.osim_view)
        self.osim_panel = OpensimPanel(c, self.osim_view)
        self.osim_panel.recompute.connect(self.start_opensim)
        self.osim_panel.saveFiles.connect(self.save_opensim_files)
        self.osim_panel.exportVideo.connect(self.export_video)
        self.osim_panel.setMinimumWidth(300)
        self.osim_panel.setMaximumWidth(420)
        split2.addWidget(self.osim_panel)
        split2.setSizes([560, 560, 330])
        self.pages.addWidget(split2)
        lay.addWidget(self.pages, 1)
        # hint
        self.hint = QLabel()
        self.hint.setFixedHeight(30)
        self.hint.setStyleSheet(f"background: {theme.PANEL}; color: {theme.MUTED}; padding: 0 14px; border-top: 1px solid {theme.LINE};")
        lay.addWidget(self.hint)
        # transport + timeline
        bottom = QWidget()
        bottom.setObjectName("Panel")
        b = QHBoxLayout(bottom)
        b.setContentsMargins(14, 0, 0, 0)
        b.setSpacing(4)

        def btn(name, tip, slot):
            w = QToolButton()
            w.setIcon(icons.icon(name, theme.TEXT_2, 18))
            w.setFixedSize(34, 34)
            w.setToolTip(tip)
            w.clicked.connect(slot)
            b.addWidget(w)

        btn("prev", tr("Previous frame (←)"), lambda: c.step(-1))
        self.play_btn = QToolButton()
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setToolTip(tr("Play / pause (Space)"))
        self.play_btn.setStyleSheet(f"QToolButton {{ background: {theme.ACCENT}; border-radius: 20px; }}"
                                    f"QToolButton:hover {{ background: {theme.ACCENT_HOVER}; }}")
        self.play_btn.clicked.connect(self.toggle_play)
        b.addWidget(self.play_btn)
        btn("next", tr("Next frame (→)"), lambda: c.step(1))
        b.addSpacing(8)
        self.time_lbl = QLabel()
        self.time_lbl.setFont(theme.mono(16, 500))
        self.time_lbl.setFixedWidth(150)
        b.addWidget(self.time_lbl)
        self.timeline = ErrorTimeline(c)
        b.addWidget(self.timeline, 1)
        lay.addWidget(bottom)
        self._play_icon()
        return page

    # session
    def build(self, videos: list, height_m: float, mode: str):
        dlg = BuildDialog(videos, height_m, mode, self)
        if dlg.exec() and dlg.session is not None:
            s = dlg.session
            from openkick.multiview.session import smoothing_for
            s.template = self.setup.activity.currentData() or None
            s.subject = self.setup.subject
            s.subject.height_m = s.subject.height_m or height_m
            s.smooth_hz = smoothing_for(s.template)
            s.update(recalibrate=False)
            self.set_session(s)

    def edit_subject(self):
        """The patient's record of the open session; measured lengths rescale the 3D."""
        s = self.ctl.session
        if not s:
            return
        from openkick.ui.subject import edit_subject
        new = edit_subject(self, s.subject, s.template)
        if new is None:
            return
        s.subject = new
        if new.height_m:
            s.height_m = new.height_m
        s.update(recalibrate=False)
        self.ctl.version += 1
        self.ctl.changed.emit()
        self.panel.refresh()
        self._refresh_name()

    def set_session(self, session: Session):
        self.stop()
        self.grid.rebuild(len(session.views))
        self.ctl.set_session(session)
        self.osim_version, self.osim_result = -1, None
        self.osim_view.set_scene(None, tr("The OpenSim model appears here."))
        self.osim_panel.set_state("none", detail=tr("It takes a few seconds."))
        self.stack.setCurrentIndex(1)
        self.set_phase(1)
        self.panel.refresh()
        self._refresh_name()
        QTimer.singleShot(0, self, self._focus_all)
        q = session.quality()
        self.hint.setText(tr("Drag any skeleton point onto the body where it is wrong. The white lines show the 3D result; "
                             "when the videos agree, they lie on the body in all of them.") if q["problems"] else
                          tr("Nothing to check. You can still drag any point to correct it."))

    def _focus_all(self):
        for cv in self.canvases:
            cv.focus_athlete()

    def _refresh_name(self):
        """The session's name in the top bar and the window title, with a dot while there are unsaved corrections."""
        s = self.ctl.session
        if not s:
            self.setWindowTitle(tr("3D from several videos") + " · " + APP_NAME)
            return
        name = s.path.stem if s.path else tr("not saved yet")
        mark = "  •" if self.ctl.dirty else ""
        self.name_lbl.setText(f"· {name}{mark}")
        self.name_lbl.setToolTip((str(s.path) if s.path else tr("This session is not in a file yet: Save keeps it."))
                                 + ("\n" + tr("There are corrections that are not saved.") if self.ctl.dirty else ""))
        self.setWindowTitle(f"{name}{' •' if self.ctl.dirty else ''} — {tr('3D from several videos')} · {APP_NAME}")

    def _base(self) -> Path:
        """Folder and name that exported files take: the session's if it is saved, else the first video's."""
        s = self.ctl.session
        return s.path.with_suffix("") if s.path else Path(s.views[0].meta.path).with_suffix("")

    def _remember(self, path):
        """Put the session in the start page's recent list."""
        add = getattr(self.parent(), "add_recent", None)
        if add:
            add(path)

    def open_session(self):
        st = QSettings()
        path, _ = QFileDialog.getOpenFileName(self, tr("Open a 3D session"), st.value("last_dir", ""),
                                              tr("OpenKick 3D session") + " (*.ok3d)")
        if path:
            self.load_session(path)

    def load_session(self, path: str) -> bool:
        """Open a saved session. If a video is no longer where it was, ask where it is."""
        if self.ctl.session and not self.confirm_discard():
            return False
        try:
            data = Session.read(path)
        except (ValueError, OSError) as e:
            QMessageBox.critical(self, APP_NAME, tr("Could not open the session:\n{error}", error=tr(str(e))))
            return False
        places, moved = [], False
        for i, info in enumerate(data["videos"]):
            where = info["path"]
            if where is None:
                QMessageBox.information(self, APP_NAME, tr("Video {n} of this session ({name}) is not where it was.\n"
                                                            "Show where it is now.", n=i + 1, name=info["name"]))
                where, _ = QFileDialog.getOpenFileName(self, tr("Where is {name}?", name=info["name"]),
                                                       str(Path(path).parent), VIDEO_FILTER)
                if not where:
                    return False
                moved = True
            if not Session.same_video(where, info["frames"]):
                QMessageBox.critical(self, APP_NAME, tr("{file} is not the video this session was made from "
                                                        "(it has a different number of frames).", file=Path(where).name))
                return False
            places.append(where)
        try:
            s = Session.load(path, videos=places)
        except Exception as e:  # noqa: BLE001 - reported to the user
            QMessageBox.critical(self, APP_NAME, tr("Could not open the session:\n{error}", error=str(e)))
            return False
        QSettings().setValue("last_dir", str(Path(path).parent))
        self.set_session(s)
        if moved:                       # the new places are worth saving
            self.ctl.version += 1
            self._refresh_name()
        self._remember(path)
        return True

    def save(self) -> bool:
        s = self.ctl.session
        if not s:
            return False
        if s.path is None:
            return self.save_as()
        return self._write(s.path)

    def edit_notes(self):
        """The evaluator's name and observations, kept with the session for its report."""
        s = self.ctl.session
        if not s:
            return
        from openkick.ui.report import NotesDialog
        dlg = NotesDialog(tr("Notes"), s.report_evaluator, s.report_notes, self)
        if not dlg.exec():
            return
        who, notes = dlg.values()
        if (who, notes) != (s.report_evaluator, s.report_notes):
            s.report_evaluator, s.report_notes = who, notes
            self.ctl.version += 1
            self._refresh_name()

    def export_report(self, fmt: str = "pdf"):
        """The report of the session in this format ("pdf", "txt" or "json")."""
        s = self.ctl.session
        if not s or not s.solved:
            return
        from openkick.ui.report import ask_path, option, session_report
        path = ask_path(self, fmt, s.path or Path(s.views[0].meta.path).with_name("informe_3d"))
        if not path:
            return
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            session_report(self, path, s.report_evaluator, s.report_notes, option("appendix"), option("pictures"),
                           option("anonymous"), option("series"))
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.hint.setText(tr("Report saved: {path}", path=path))

    def save_as(self) -> bool:
        s = self.ctl.session
        if not s:
            return False
        default = str(s.path or Path(s.views[0].meta.path).with_name("sesion_3d.ok3d"))
        path, _ = QFileDialog.getSaveFileName(self, tr("Save 3D session"), default, tr("OpenKick 3D session") + " (*.ok3d)")
        if not path:
            return False
        if not path.lower().endswith(".ok3d"):
            path += ".ok3d"
        return self._write(Path(path))

    def _write(self, path: Path) -> bool:
        try:
            self.ctl.session.save(path)
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, tr("The session could not be saved:\n{error}", error=str(e)))
            return False
        from openkick.core import patients
        patients.note_analysis(self.ctl.session.subject, path, "3D")
        self.ctl.saved_version = self.ctl.version
        QSettings().setValue("last_dir", str(Path(path).parent))
        self._remember(path)
        self._refresh_name()
        self.hint.setText(tr("Saved: {path}", path=str(path)))
        return True

    def export_trc(self):
        s = self.ctl.session
        if not s:
            return
        from openkick.multiview.export import export_trc
        base = self._base()
        path, _ = QFileDialog.getSaveFileName(self, tr("Export to OpenSim"), str(base.with_name(base.name + "_3d.trc")),
                                              tr("OpenSim markers") + " (*.trc)")
        if not path:
            return
        if not path.lower().endswith(".trc"):
            path += ".trc"
        info = export_trc(s, Path(path))
        QMessageBox.information(self, APP_NAME, tr(
            "3D points saved for OpenSim:\n{path}\n\n{m} markers · {n} frames · {r} per second, in metres.\n"
            "In OpenSim: scale the model with this file, then run Inverse Kinematics.",
            path=path, m=info["markers"], n=info["frames"], r=info["rate"]))

    # step 2: OpenSim
    def set_phase(self, k: int):
        self.phase = k
        self.phase_btns[k].setChecked(True)
        if k != 1:
            self.ctl.set_focus(None)
        self.pages.setCurrentIndex(k - 1)
        self.ghost_btn.setVisible(k == 1)
        if k == 2:
            self.hint.setText(tr("Left: the 3D points. Right: the OpenSim model they produce. Drag either one to turn both."))
            if self.osim_result is None and not self.osim_worker:
                self.start_opensim()
            else:
                self._osim_state()
        elif self.ctl.session:
            self.hint.setText("")

    def _focus_changed(self):
        """One video is edited on its own, or all of them are back."""
        view = self.ctl.focus
        self.stop()
        self.grid.show_only(view)
        self.shadow_btn.setVisible(view is not None and self.phase == 1)
        self._refresh_transport()
        if view is not None:
            self.hint.setText(tr("Video {n} on its own: go frame by frame (← →) and drag the points that are off. "
                                 "The 3D on the right follows. Esc returns to all the videos.", n=view + 1))
        else:
            self.hint.setText(tr("Back to all the videos: the 3D uses your corrections."))
            QTimer.singleShot(0, self, self._focus_all)

    def _osim_folder(self) -> Path:
        import hashlib

        from openkick.core.pose import data_dir
        key = hashlib.sha1("|".join(v.meta.path for v in self.ctl.session.views).encode()).hexdigest()[:16]
        return data_dir() / "cache" / "multiview" / "opensim" / key

    def start_opensim(self):
        s = self.ctl.session
        if not s or self.osim_worker:
            return
        import sys
        miss = multiview.missing(multiview.OPENSIM_REQUIREMENTS)
        if miss or getattr(sys, "frozen", False):
            text = tr("This step needs extra components that are not installed: {names}. "
                      "Install them with:  pip install {names}", names=" ".join(miss)) if miss else \
                tr("This step is not available in this version of the program yet.")
            self.osim_view.set_scene(None, text)
            self.osim_panel.set_state("missing", detail=text)
            return
        from openkick.multiview import opensim_job
        from openkick.multiview.export import export_trc
        folder = self._osim_folder()
        trc = opensim_job.prepare(folder) / "movement.trc"
        export_trc(s, trc)
        self.osim_result = None
        self.osim_view.set_scene(None, tr("Computing the OpenSim model…\nscaling it to the person and solving inverse kinematics"), busy=True)
        self.osim_panel.set_state("busy", detail=tr("It takes a few seconds."))
        self.osim_worker = OpensimWorker(folder, trc, s.height_m)
        self._osim_for = (s, self.ctl.version)
        self.osim_worker.done.connect(self._opensim_done)
        self.osim_worker.start()

    def _opensim_done(self, res: dict):
        self.osim_worker.wait()
        self.osim_worker = None
        session, version = self._osim_for
        if self.ctl.session is not session:        # another session was opened meanwhile: this result is not its
            if self.ctl.session and self.phase == 2:
                self.start_opensim()
            return
        if not res.get("ok"):
            last = [ln for ln in str(res.get("error", "")).strip().splitlines() if ln.strip()]
            detail = last[-1] if last else ""
            self.osim_view.set_scene(None, tr("OpenSim could not build the model."))
            self.osim_panel.set_state("failed", detail=detail)
            return
        try:
            from openkick.multiview.bones import BoneScene
            scene = BoneScene(res["baked"])
        except Exception as e:  # noqa: BLE001 - reported to the user
            self.osim_view.set_scene(None, tr("The model was computed, but it cannot be drawn on this computer."))
            self.osim_panel.set_state("failed", detail=str(e))
            return
        self.osim_result, self.osim_version = res, version
        self.osim_view.set_scene(scene)
        self._osim_state()

    def _osim_state(self):
        if self.osim_worker or self.osim_result is None:
            return
        self.osim_panel.set_state("ready" if self.osim_version == self.ctl.version else "stale", self.osim_result)

    def save_opensim_files(self):
        res, s = self.osim_result, self.ctl.session
        if not res or not s:
            return
        import shutil

        from openkick.multiview.opensim_job import geometry_dir
        base = self._base()
        folder = QFileDialog.getExistingDirectory(self, tr("Folder for the OpenSim files"), str(base.parent))
        if not folder:
            return
        stem = base.name
        out = Path(folder) / f"{stem}_opensim"
        out.mkdir(parents=True, exist_ok=True)
        for key, suffix in (("osim", ".osim"), ("mot", ".mot"), ("trc", ".trc")):
            shutil.copyfile(res[key], out / (stem + suffix))
        shutil.copytree(geometry_dir(), out / "Geometry", dirs_exist_ok=True)
        (out / "LEEME.txt").write_text(tr(
            "Files made by OpenKick for OpenSim\n\n{stem}.osim  the model, scaled to the person\n{stem}.mot   the movement "
            "(joint angles from inverse kinematics)\n{stem}.trc   the 3D points used as markers\nGeometry     the bone shapes\n\n"
            "In OpenSim: File > Open Model ({stem}.osim), then File > Load Motion ({stem}.mot).\n", stem=stem), encoding="utf-8")
        self.hint.setText(tr("OpenSim files saved in {path}", path=str(out)))

    def export_video(self, speed: float = 1.0, layout: str = "side", size: tuple | None = None) -> bool:
        """MP4 of the result. layout "side": the 3D points next to the OpenSim model, from the
        angle on screen. layout "cameras": every video with the model below it, seen from that camera."""
        s, scene = self.ctl.session, self.osim_view.scene
        if not s or scene is None or self.osim_version != self.ctl.version:
            return False
        from PySide6.QtWidgets import QApplication, QProgressDialog
        base = self._base()
        tag = ("_3d_opensim" if layout == "side" else "_videos_opensim") + ("" if speed == 1 else f"_x{speed:g}".replace(".", ","))
        path, _ = QFileDialog.getSaveFileName(self, tr("Export video"), str(base.with_name(base.name + tag + ".mp4")), "MP4 (*.mp4)")
        if not path:
            return False
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        self.stop()
        dlg = QProgressDialog(tr("Exporting the video…"), tr("Cancel"), 0, 100, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)

        def progress(done, total):
            dlg.setMaximum(total)
            dlg.setValue(done)
            QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            if layout == "side":
                n = write_comparison_video(path, s, self.ctl.world(), scene, self.osim_view.orbit, speed, size or (1920, 1080),
                                           name=base.name, progress=progress)
            else:
                n = write_camera_video(path, s, self.ctl.world(), scene, self.ctl.readers, speed, size,
                                       name=base.name, progress=progress)
        except Exception as e:  # noqa: BLE001 - reported to the user
            dlg.close()
            QMessageBox.critical(self, APP_NAME, tr("The video could not be exported:\n{error}", error=str(e)))
            return False
        finally:
            self.osim_view._key = None              # the model view draws itself again at its own size
            self.osim_view.update()
        dlg.setValue(dlg.maximum())
        dlg.close()
        if not n:
            Path(path).unlink(missing_ok=True)
            self.hint.setText(tr("Video export cancelled."))
            return False
        self.hint.setText(tr("Video saved: {path}", path=path))
        return True

    def _changed(self):
        self._refresh_transport()
        self._refresh_name()
        if self.phase == 2:
            self._osim_state()

    def confirm_discard(self) -> bool:
        s = self.ctl.session
        if not s or not self.ctl.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, tr("Save the corrections of this 3D session?"),
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def _leave(self):
        self.stop()
        self.osim_view.set_scene(None)
        self.ctl.close()
        self.stack.setCurrentIndex(0)
        self._refresh_name()

    def back_to_setup(self):
        if self.confirm_discard():
            self._leave()

    def closeEvent(self, e):
        if not self.confirm_discard():
            e.ignore()
            return
        self._leave()
        super().closeEvent(e)

    # transport
    def _toggle_ghost(self, on: bool):
        self.ctl.show_3d = on
        for cv in self.canvases:
            cv.update()

    def _toggle_lengths(self, on: bool):
        self.ctl.show_lengths = on
        QSettings().setValue("multiview/lengths", bool(on))
        for cv in self.canvases:
            cv.update()
        self.timeline._refresh()

    def _play_icon(self):
        self.play_btn.setIcon(icons.icon(icons.PAUSE if self.playing else icons.PLAY, theme.ON_ACCENT, 18, fill=True))

    def _refresh_transport(self, *_):
        s = self.ctl.session
        if not s or not s.frame_count:
            return
        c = self.ctl
        if c.focus is None:
            self.time_lbl.setText(f"{timecode(s.t[c.row] - s.t[0])}  {c.row + 1}/{s.frame_count}")
            self.time_lbl.setToolTip(tr("Step {n} of {total} of the 3D (one per frame of video 1)", n=c.row + 1, total=s.frame_count))
        else:                                        # one video: its own frames
            lo, hi = c.focus_range()
            v = s.views[c.focus]
            self.time_lbl.setText(f"{timecode(v.t[c.focus_frame] - v.t[lo])}  {c.focus_frame - lo + 1}/{hi - lo + 1}")
            self.time_lbl.setToolTip(tr("Frame {f} of video {v}: number {n} of the {total} that take part in the 3D",
                                        f=c.focus_frame + 1, v=c.focus + 1, n=c.focus_frame - lo + 1, total=hi - lo + 1))

    def _at_end(self) -> bool:
        c = self.ctl
        if c.focus is None:
            return c.row >= c.session.frame_count - 1
        return c.focus_frame >= c.focus_range()[1]

    def toggle_play(self):
        s = self.ctl.session
        if not s:
            return
        self.playing = not self.playing
        if self.playing and self._at_end():
            self.ctl.set_row(0)
        self._play_icon()
        self.ctl.set_playing(self.playing)
        if self.playing:
            self._schedule()
        else:
            self.timer.stop()

    def stop(self):
        if self.playing:
            self.toggle_play()

    def _schedule(self):
        c, s = self.ctl, self.ctl.session
        if self._at_end():
            self.stop()
            return
        t, i = (s.t, c.row) if c.focus is None else (s.views[c.focus].t, c.focus_frame)
        self.timer.start(max(1, int((t[i + 1] - t[i]) * 1000)))

    def _tick(self):
        if self.playing and self.ctl.session:
            self.ctl.step(1)
            self._schedule()
