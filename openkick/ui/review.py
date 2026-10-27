"""The doubtful stretches of a 3D session, worst first, to go through and correct one by one.

Each row is one point over a run of frames where the videos disagree, a bone changes length or
the point jumps. Selecting it shows that frame of the video that disagrees most, with the
detected point and where the 3D puts it; going to it lets the reviewer drag the point in the
3D window. What the reviewer finds fine is marked as such and kept with the session."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from openkick.i18n import tr
from openkick.ui import theme
from openkick.ui.multiview import joint_name, num, timecode

WHY = {"disagree": "the videos disagree", "bone": "a bone changes length", "jump": "it jumps"}
PREVIEW = 260            # side of the preview, in pixels


class ReviewWindow(QWidget):
    """The list of doubtful stretches of the open session, next to the 3D window."""

    def __init__(self, ctl, parent=None):
        super().__init__(parent, Qt.Window)
        self.ctl = ctl
        self.stretches: list[dict] = []
        self._revision = None
        self.setWindowTitle(tr("Points to check"))
        self.resize(1060, 720)
        self.setMinimumSize(900, 640)
        self.setObjectName("Page")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel(tr("Points to check"))
        title.setFont(theme.display(20))
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(QLabel(tr("Point")))
        self.joint = QComboBox()
        self.joint.currentIndexChanged.connect(self._fill)
        head.addWidget(self.joint)
        self.show_done = QCheckBox(tr("Show the ones marked as fine"))
        self.show_done.toggled.connect(self._fill)
        head.addWidget(self.show_done)
        close = QToolButton()
        close.setText("✕")
        close.setToolTip(tr("Close"))
        close.clicked.connect(self.close)
        head.addWidget(close)
        lay.addLayout(head)
        hint = QLabel(tr("Stretches where the 3D is doubtful, the worst first. Click one to see it; Go to it takes "
                         "you there to drag the point where it is wrong (with “The whole doubtful stretch” the "
                         "correction reaches every frame of it). Mark as fine what is right: it is kept with the "
                         "session and Next point to check skips it."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(hint)

        body = QHBoxLayout()
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([tr("Point"), tr("Video"), tr("Frames"), tr("Time"), tr("Worst"),
                                              tr("Why"), tr("State")])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._preview)
        self.table.cellDoubleClicked.connect(lambda *_: self.go())
        body.addWidget(self.table, 1)
        side = QVBoxLayout()
        self.picture = QLabel()
        self.picture.setFixedSize(PREVIEW, PREVIEW)
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setStyleSheet(f"background: {theme.PANEL}; border-radius: 10px;")
        side.addWidget(self.picture)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setFixedWidth(PREVIEW)
        self.caption.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.caption.setMinimumHeight(88)
        self.caption.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        side.addWidget(self.caption)
        self.go_btn = QPushButton(tr("Go to it"))
        self.go_btn.setProperty("primary", True)
        self.go_btn.setToolTip(tr("Its worst frame in the 3D window, with the point selected (double click does the same)"))
        self.go_btn.clicked.connect(self.go)
        side.addWidget(self.go_btn)
        self.solo_btn = QPushButton(tr("Edit it in its video"))
        self.solo_btn.setToolTip(tr("The video that disagrees most, full size and frame by frame"))
        self.solo_btn.clicked.connect(lambda: self.go(solo=True))
        side.addWidget(self.solo_btn)
        row = QHBoxLayout()
        self.fine_btn = QPushButton(tr("It is fine"))
        self.fine_btn.setToolTip(tr("The point is where it should be in every video: stop marking it"))
        self.fine_btn.clicked.connect(self.toggle_fine)
        row.addWidget(self.fine_btn, 1)
        self.next_btn = QPushButton(tr("Next"))
        self.next_btn.clicked.connect(self.next)
        row.addWidget(self.next_btn)
        side.addLayout(row)
        side.addStretch(1)
        body.addLayout(side)
        lay.addLayout(body, 1)
        self.count = QLabel()
        self.count.setStyleSheet(f"color: {theme.DIM};")
        lay.addWidget(self.count)
        ctl.changed.connect(self.refresh)
        self.refresh()

    # ── contents ──
    def refresh(self):
        s = self.ctl.session
        if s is None or not s.solved:
            self.stretches = []
            self._fill()
            return
        if self._revision == (id(s), s.revision):
            return
        self._revision = (id(s), s.revision)
        keep = self._current()
        self.stretches = s.stretches()
        joints = sorted({e["joint"] for e in self.stretches}, key=joint_name)
        chosen = self.joint.currentData()
        self.joint.blockSignals(True)
        self.joint.clear()
        self.joint.addItem(tr("All"), None)
        for j in joints:
            self.joint.addItem(joint_name(j), j)
        self.joint.setCurrentIndex(max(0, self.joint.findData(chosen)))
        self.joint.blockSignals(False)
        self._fill(keep=keep)

    def shown(self) -> list[dict]:
        j = self.joint.currentData()
        return [e for e in self.stretches if (j is None or e["joint"] == j) and (self.show_done.isChecked() or not e["reviewed"])]

    def _current(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        items = getattr(self, "_items", [])
        return items[rows[0].row()] if rows and rows[0].row() < len(items) else None

    def _fill(self, *_, keep: dict | None = None):
        s = self.ctl.session
        self._items = self.shown()
        self.table.setRowCount(len(self._items))
        for r, e in enumerate(self._items):
            v = e["view"]
            a, b = s.frame_of(v, e["first"]), s.frame_of(v, e["last"])
            cells = [joint_name(e["joint"]), tr("Video {n}", n=v + 1), f"{a}–{b}" if b > a else str(a),
                     timecode(s.t[e["worst"]] - s.t[0]), f"{num(e['cm'])} cm", tr(WHY[e["why"]]),
                     tr("fine") if e["reviewed"] else tr("to check")]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setToolTip(text)
                if c in (2, 3, 4):
                    it.setFont(theme.mono(12))
                if c == 4:
                    it.setForeground(QColor(theme.DANGER if e["cm"] > 12.0 else theme.WARN))
                self.table.setItem(r, c, it)
        total = sum(1 for e in self.stretches if not e["reviewed"])
        self.count.setText(tr("{n} stretches to check, {m} marked as fine", n=total, m=len(self.stretches) - total))
        pick = 0
        if keep is not None:
            near = [i for i, e in enumerate(self._items) if e["joint"] == keep["joint"]]
            if near:
                pick = min(near, key=lambda i: abs(self._items[i]["worst"] - keep["worst"]))
        if self._items:
            self.table.selectRow(pick)
        else:
            self.picture.setPixmap(QPixmap())
            self.picture.setText(tr("Nothing to check"))
            self.caption.setText("")
        for b in (self.go_btn, self.solo_btn, self.fine_btn, self.next_btn):
            b.setEnabled(bool(self._items))

    def _preview(self):
        e = self._current()
        s = self.ctl.session
        if e is None or s is None:
            return
        self.fine_btn.setText(tr("Mark it to check again") if e["reviewed"] else tr("It is fine"))
        v, row, j = e["view"], e["worst"], e["joint"]
        f = s.frame_of(v, row)
        self.caption.setText(tr("Video {n}, frame {f}: {what}.", n=v + 1, f=f, what=self.ctl.describe(row, j)) + " "
                             + tr("Orange: what the detector saw. White: where the 3D puts it."))
        try:
            rgb = self.ctl.readers[v].frame(f)
        except Exception:  # noqa: BLE001 - no picture (the video is missing): the list still works
            self.picture.setText(tr("No picture"))
            return
        seen = s.views[v].xy[f, j].astype(float)
        built = s.projection(v, row)[j]
        self.picture.setPixmap(crop(rgb, seen, built, PREVIEW))

    # ── actions ──
    def go(self, solo: bool = False):
        e = self._current()
        if e is None:
            return
        c = self.ctl
        if solo:
            c.set_focus(e["view"])
        c.set_row(e["worst"])
        c.select((e["view"] if c.focus is None else c.focus, e["joint"]))
        c.message.emit(c.describe(e["worst"], e["joint"]))

    def toggle_fine(self):
        e = self._current()
        if e is None:
            return
        self.ctl.session.set_reviewed(e["joint"], e["first"], e["last"], not e["reviewed"])
        self.ctl.version += 1
        self.ctl.changed.emit()

    def next(self):
        r = self.table.currentRow()
        if r + 1 < self.table.rowCount():
            self.table.selectRow(r + 1)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)


def crop(rgb: np.ndarray, seen, built, size: int) -> QPixmap:
    """A square of the frame around a point: the detected one (orange ring) and the 3D one
    (white dot), joined by a line."""
    h, w = rgb.shape[:2]
    pts = [p for p in (seen, built) if p is not None and np.isfinite(p).all()]
    centre = np.mean(pts, axis=0) if pts else np.array([w / 2, h / 2])
    gap = float(np.linalg.norm(pts[0] - pts[1])) if len(pts) == 2 else 0.0
    half = int(min(max(80.0, 1.6 * gap), min(w, h) / 2))
    x0, y0 = int(np.clip(centre[0] - half, 0, max(0, w - 2 * half))), int(np.clip(centre[1] - half, 0, max(0, h - 2 * half)))
    part = np.ascontiguousarray(rgb[y0:y0 + 2 * half, x0:x0 + 2 * half])
    img = QImage(part.data, part.shape[1], part.shape[0], 3 * part.shape[1], QImage.Format_RGB888).copy()
    out = QPixmap.fromImage(img.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    k = out.width() / max(1, part.shape[1])
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)

    def at(q):
        return QPointF((q[0] - x0) * k, (q[1] - y0) * k)
    if len(pts) == 2:
        p.setPen(QPen(QColor(255, 255, 255, 160), 1.5, Qt.DashLine))
        p.drawLine(at(seen), at(built))
    if seen is not None and np.isfinite(seen).all():
        p.setPen(QPen(QColor(theme.WARN), 3))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(at(seen).x() - 9, at(seen).y() - 9, 18, 18))
    if built is not None and np.isfinite(built).all():
        p.setPen(QPen(QColor(0, 0, 0, 180), 1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QRectF(at(built).x() - 5, at(built).y() - 5, 10, 10))
    p.end()
    return out
