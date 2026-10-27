"""Shared application state: the open project, the current frame, the selection and
undo history. Widgets observe its signals instead of talking to each other."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QUndoCommand, QUndoStack

from openkick.core import analysis as A
from openkick.core.project import Project
from openkick.core.video import VideoReader


class _PoseCommand(QUndoCommand):
    def __init__(self, ctl: "Controller", edit: A.PoseEdit, text: str):
        super().__init__(text)
        self.ctl, self.edit, self._first = ctl, edit, True

    def redo(self):
        if self._first:           # the edit is already applied when pushed
            self._first = False
            return
        self.edit.redo(self.ctl.project.pose)
        self.ctl.notify()

    def undo(self):
        self.edit.undo(self.ctl.project.pose)
        self.ctl.notify()


class _SnapshotCommand(QUndoCommand):
    """Undo for list-like project state (measurements, notes, phases, calibration)."""

    def __init__(self, ctl: "Controller", before: dict, after: dict, text: str):
        super().__init__(text)
        self.ctl, self.before, self.after, self._first = ctl, before, after, True

    def redo(self):
        if self._first:
            self._first = False
            return
        self.ctl._restore(self.after)

    def undo(self):
        self.ctl._restore(self.before)


class Controller(QObject):
    projectLoaded = Signal()
    projectClosed = Signal()
    frameChanged = Signal(int)
    changed = Signal()                 # project data changed
    selectionChanged = Signal()
    toolChanged = Signal(str)
    playingChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self.reader: VideoReader | None = None
        self.frame = 0
        self.selected_landmark: int | None = None
        self.selected_measure: str | None = None
        self.tool = "select"
        self.edit_scope = "frame"      # "frame" | "neighbors" | "phase"
        self.layers = {"skeleton": True, "angles": True, "trajectories": True, "distances": True,
                       "markers": True, "notes": True}
        self.undo = QUndoStack(self)
        self._qimage_cache: dict[int, QImage] = {}
        self.plugin_results: dict[str, any] = {}

    # ── project ────────────────────────────────────────────
    def set_project(self, project: Project):
        self.close_project(emit=False)
        self.project = project
        try:
            self.reader = VideoReader(project.video)
        except Exception:
            self.reader = None
        self.frame = 0
        self.selected_landmark = None
        self.selected_measure = None
        self.undo.clear()
        self._qimage_cache.clear()
        self.plugin_results.clear()
        if project.template and project.pose.has_data:
            self.run_template_plugin()
        self.projectLoaded.emit()
        self.frameChanged.emit(0)

    def close_project(self, emit=True):
        if self.reader:
            self.reader.close()
        self.reader = None
        self.project = None
        self.undo.clear()
        self._qimage_cache.clear()
        self.plugin_results.clear()
        if emit:
            self.projectClosed.emit()

    def run_template_plugin(self):
        p = self.project
        if not p or not p.pose.has_data or not p.template:
            return None
        from openkick.plugins import PluginContext, discover
        plugins, _ = discover()
        pl = next((x for x in plugins if x.id == p.template and getattr(x, "kind", "") == "analysis"), None)
        if not pl:
            return None
        try:
            res = pl.run(PluginContext(p))
            self.plugin_results[pl.id] = res
            self.changed.emit()
            return res
        except Exception:  # noqa: BLE001 - a failing plugin must not stop the editor, but it is logged
            import traceback

            from openkick.app import log_error
            log_error(f"plugin {pl.id}:\n{traceback.format_exc()}")
            return None

    @property
    def has_project(self) -> bool:
        return self.project is not None

    # ── frames ─────────────────────────────────────────────
    def set_frame(self, index: int):
        if not self.project:
            return
        index = max(0, min(int(index), self.project.frame_count - 1))
        if index != self.frame:
            self.frame = index
            self.frameChanged.emit(index)

    def step(self, delta: int):
        self.set_frame(self.frame + delta)

    def image(self, index: int | None = None) -> QImage:
        index = self.frame if index is None else index
        img = self._qimage_cache.get(index)
        if img is None:
            arr = self.reader.frame(index)
            h, w = arr.shape[:2]
            img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self._qimage_cache[index] = img
            if len(self._qimage_cache) > 24:
                self._qimage_cache.pop(next(iter(self._qimage_cache)))
        return img

    # ── selection / tools ──────────────────────────────────
    def select_landmark(self, lm: int | None):
        self.selected_landmark = lm
        if lm is not None:
            self.selected_measure = None
        self.selectionChanged.emit()

    def select_measure(self, mid: str | None):
        self.selected_measure = mid
        if mid is not None:
            self.selected_landmark = None
        self.selectionChanged.emit()

    def set_tool(self, tool: str):
        self.tool = tool
        self.toolChanged.emit(tool)

    # ── changes ────────────────────────────────────────────
    def notify(self):
        if self.project:
            self.project.touch()
        self.changed.emit()

    def push_pose_edit(self, edit: A.PoseEdit, text: str):
        self.undo.push(_PoseCommand(self, edit, text))
        self.notify()

    def _snapshot(self) -> dict:
        p = self.project
        return copy.deepcopy({"measurements": p.measurements, "notes": p.notes, "phases": p.phases,
                              "subphases": p.subphases, "tracks": p.tracks, "subject": p.subject, "calibration": p.calibration,
                              "template": p.template})

    def _restore(self, snap: dict):
        p = self.project
        s = copy.deepcopy(snap)
        p.measurements, p.notes, p.phases = s["measurements"], s["notes"], s["phases"]
        p.subphases = s.get("subphases", p.subphases)
        p.tracks = s.get("tracks", p.tracks)
        p.subject = s.get("subject", p.subject)
        p.calibration, p.template = s["calibration"], s["template"]
        if self.selected_measure and not p.measurement(self.selected_measure):
            self.selected_measure = None
        self.notify()
        self.selectionChanged.emit()

    def change(self, text: str, fn):
        """Run `fn()` that mutates measurements/notes/phases/calibration as one undoable step."""
        before = self._snapshot()
        result = fn()
        after = self._snapshot()
        self.undo.push(_SnapshotCommand(self, before, after, text))
        self.notify()
        return result

    def begin(self) -> dict:
        """Start a change made interactively (e.g. a drag); finish it with commit()."""
        return self._snapshot()

    def commit(self, before: dict, text: str):
        self.undo.push(_SnapshotCommand(self, before, self._snapshot(), text))
        self.notify()

    def replace_pose(self, pose, text: str):
        """Swap the whole pose (re-detection) as one undoable step."""
        old = self.project.pose
        n = pose.frame_count
        edit = A.PoseEdit(0, n - 1, old.xy.copy(), old.source.copy(), pose.xy.copy(), pose.source.copy())
        old.raw_xy, old.visibility, old.z = pose.raw_xy, pose.visibility, pose.z
        edit.redo(old)
        self.push_pose_edit(edit, text)

    def low_confidence_frames(self) -> list[int]:
        if not self.project:
            return []
        key = ("low", self.project.revision)
        if getattr(self, "_low_key", None) != key:
            self._low_key, self._low = key, A.low_confidence_frames(self.project)
        return self._low

    def manual_frames(self) -> np.ndarray:
        from openkick.core.project import SRC_MANUAL
        return np.nonzero((self.project.pose.source == SRC_MANUAL).any(axis=1))[0]
