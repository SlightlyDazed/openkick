"""Dialogs: plugin results, plugin manager, template builder, shortcuts, about."""

from __future__ import annotations

import csv
import math

from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from openkick import AUTHOR, __version__
from openkick.core import landmarks as L
from openkick.core import templates
from openkick.i18n import tr
from openkick.plugins import Result, user_dir
from openkick.ui import icons, theme


def _dialog_style(d: QDialog):
    d.setStyleSheet(f"QDialog {{ background: {theme.CARD_2}; }}")


def _title(text: str, size=22) -> QLabel:
    t = QLabel(text)
    t.setFont(theme.display(size))
    return t


# ── plugin results ─────────────────────────────────────────────────────


class PluginResultDialog(QDialog):
    def __init__(self, title: str, result: Result, parent=None):
        super().__init__(parent)
        self.result = result
        _dialog_style(self)
        self.setWindowTitle(title)
        self.resize(860, 560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(14)
        lay.addWidget(_title(result.title or title))
        if result.summary:
            s = QLabel(result.summary)
            s.setWordWrap(True)
            s.setStyleSheet(f"color: {theme.MUTED};")
            lay.addWidget(s)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(18)
        for t in result.tables:
            if t.title:
                h = QLabel(t.title)
                h.setStyleSheet("font-weight: 600;")
                bl.addWidget(h)
            tw = QTableWidget(len(t.rows), len(t.columns))
            tw.setHorizontalHeaderLabels(t.columns)
            tw.verticalHeader().hide()
            tw.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tw.setAlternatingRowColors(True)
            tw.setShowGrid(False)
            for r, row in enumerate(t.rows):
                for c, v in enumerate(row):
                    it = QTableWidgetItem(str(v))
                    if c > 0:
                        it.setFont(theme.mono(12))
                    tw.setItem(r, c, it)
            tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tw.setFixedHeight(46 + 36 * len(t.rows))
            bl.addWidget(tw)
        bl.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        copy = QPushButton(tr("Copy"))
        copy.clicked.connect(self._copy)
        row.addWidget(copy)
        save = QPushButton(tr("Save CSV…"))
        save.clicked.connect(self._save)
        row.addWidget(save)
        ok = QPushButton(tr("Close"))
        ok.setProperty("primary", True)
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        lay.addLayout(row)

    def _text(self, sep="\t"):
        out = []
        for t in self.result.tables:
            if t.title:
                out.append(t.title)
            out.append(sep.join(t.columns))
            out += [sep.join(str(v) for v in r) for r in t.rows]
            out.append("")
        return "\n".join(out)

    def _copy(self):
        QGuiApplication.clipboard().setText(self._text())

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Save CSV…"), "", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for t in self.result.tables:
                if t.title:
                    w.writerow([t.title])
                w.writerow(t.columns)
                w.writerows(t.rows)
                w.writerow([])


# ── plugin manager ─────────────────────────────────────────────────────


def disabled_plugins() -> set[str]:
    return set(QSettings().value("plugins/disabled", [], type=list) or [])


class PluginsDialog(QDialog):
    changed = Signal()

    def __init__(self, plugins, errors, parent=None):
        super().__init__(parent)
        _dialog_style(self)
        self.setWindowTitle(tr("Plugins"))
        self.resize(720, 600)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(14)
        lay.addWidget(_title(tr("Plugins")))
        s = QLabel(tr("Plugins add analyses and export formats. Built-in ones come with OpenKick; "
                      "yours live in the plugins folder. Only install plugins you trust: they run code on your computer."))
        s.setWordWrap(True)
        s.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(s)
        off = disabled_plugins()
        self.checks = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 8, 0)
        bl.setSpacing(10)
        for p in plugins:
            card = QFrame()
            card.setObjectName("Card")
            cl = QHBoxLayout(card)
            cl.setContentsMargins(14, 12, 14, 12)
            cl.setSpacing(12)
            ic = QLabel()
            ic.setFixedSize(34, 34)
            ic.setAlignment(Qt.AlignCenter)
            ic.setStyleSheet(f"background: {theme.ACTIVE}; border-radius: 9px;")
            ic.setPixmap(icons.pixmap("export" if p.kind == "export" else "chart", theme.ACCENT, 18))
            cl.addWidget(ic, 0, Qt.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(3)
            n = QLabel(p.label)
            n.setStyleSheet("font-weight: 600;")
            col.addWidget(n)
            d = QLabel(p.summary)
            d.setWordWrap(True)
            d.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
            col.addWidget(d)
            src = QLabel((tr("Built-in") if p.builtin else tr("Your plugins folder")) + f" · v{p.version}"
                         + (f" · {p.author}" if p.author else ""))
            src.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
            col.addWidget(src)
            cl.addLayout(col, 1)
            cb = QCheckBox(tr("Active"))
            cb.setChecked(p.id not in off)
            self.checks[p.id] = cb
            cl.addWidget(cb, 0, Qt.AlignTop)
            bl.addWidget(card)
        if errors:
            e = QLabel(tr("Some plugins could not be loaded:") + "\n\n" + "\n\n".join(errors))
            e.setWordWrap(True)
            e.setTextInteractionFlags(Qt.TextSelectableByMouse)
            e.setStyleSheet(f"color: {theme.WARN_TEXT}; font-family: '{theme.MONO}'; font-size: 11px;")
            bl.addWidget(e)
        bl.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)
        box = QFrame()
        box.setStyleSheet(f"QFrame {{ border: 1px dashed {theme.BORDER}; border-radius: 12px; }} QLabel {{ border: 0; }}")
        bx = QHBoxLayout(box)
        bx.setContentsMargins(14, 10, 10, 10)
        t = QLabel(tr("Do you code in Python? A plugin can add metrics, charts, exporters or other pose models."))
        t.setWordWrap(True)
        t.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
        bx.addWidget(t, 1)
        folder = QPushButton(icons.icon("folder", theme.TEXT, 16), " " + tr("Open plugins folder"))
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_dir()))))
        bx.addWidget(folder)
        lay.addWidget(box)
        row = QHBoxLayout()
        row.addStretch(1)
        ok = QPushButton(tr("Save"))
        ok.setProperty("primary", True)
        ok.clicked.connect(self._save)
        row.addWidget(ok)
        lay.addLayout(row)

    def _save(self):
        off = [pid for pid, cb in self.checks.items() if not cb.isChecked()]
        QSettings().setValue("plugins/disabled", off)
        self.changed.emit()
        self.accept()


# ── template builder ───────────────────────────────────────────────────


FIGURE = {  # front view, figure's right side on the viewer's left
    "nose": (90, 30), "left_shoulder": (120, 64), "right_shoulder": (60, 64),
    "left_elbow": (130, 110), "right_elbow": (50, 110), "left_wrist": (136, 152), "right_wrist": (44, 152),
    "left_hip": (106, 150), "right_hip": (74, 150), "left_knee": (108, 212), "right_knee": (72, 212),
    "left_ankle": (110, 272), "right_ankle": (70, 272), "left_foot_index": (122, 284), "right_foot_index": (58, 284),
}
FIG_BONES = [("right_shoulder", "left_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
             ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_hip"),
             ("left_shoulder", "left_hip"), ("right_hip", "left_hip"), ("right_hip", "right_knee"),
             ("right_knee", "right_ankle"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
             ("right_ankle", "right_foot_index"), ("left_ankle", "left_foot_index")]


class BodyPicker(QWidget):
    """Tap a joint to toggle its standard angle."""
    toggled = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected: set[str] = set()
        self.setFixedSize(180, 310)
        self.setCursor(Qt.PointingHandCursor)
        self.presets = {L.NAMES[j]: v for j, v in L.ANGLE_PRESETS.items()}

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#3A4352"), 3, Qt.SolidLine, Qt.RoundCap))
        for a, b in FIG_BONES:
            p.drawLine(QPointF(*FIGURE[a]), QPointF(*FIGURE[b]))
        p.drawLine(QPointF(90, 46), QPointF(90, 64))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(90, 30), 16, 16)
        for sel in self.selected:
            a, o, b = self.presets[sel]
            p.setPen(QPen(QColor(theme.ACCENT), 3.5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(*FIGURE[a]), QPointF(*FIGURE[o]))
            p.drawLine(QPointF(*FIGURE[o]), QPointF(*FIGURE[b]))
        for name in self.presets:
            c = QPointF(*FIGURE[name])
            if name in self.selected:
                p.setPen(QPen(QColor(theme.CARD_2), 2))
                p.setBrush(QColor(theme.ACCENT))
                p.drawEllipse(c, 7.5, 7.5)
            else:
                p.setPen(QPen(QColor("#5B6474"), 2))
                p.setBrush(QColor(theme.CARD_2))
                p.drawEllipse(c, 6, 6)
        p.setPen(QColor(theme.DIM))
        p.setFont(theme.font(11))
        p.drawText(QRectF(10, 292, 60, 16), Qt.AlignLeft, tr("Right"))
        p.drawText(QRectF(110, 292, 60, 16), Qt.AlignRight, tr("Left"))

    def mousePressEvent(self, e):
        pos = e.position()
        for name in self.presets:
            x, y = FIGURE[name]
            if math.hypot(pos.x() - x, pos.y() - y) < 13:
                self.toggled.emit(name)
                return


class TemplateEditor(QDialog):
    """No-code analysis template builder."""
    applyRequested = Signal(object)

    def __init__(self, base: templates.Template | None = None, can_apply: bool = False, parent=None):
        super().__init__(parent)
        _dialog_style(self)
        self.setWindowTitle(tr("Analysis template"))
        self.resize(980, 660)
        self.t = templates.Template(id="", name="") if base is None else templates.Template.from_dict(base.to_dict())
        if base is not None and base.builtin:
            self.t.id = ""                         # editing a built-in saves a copy
            self.t.name = tr("{name} (copy)", name=base.label)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(18)
        top = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(6)
        nl = QLabel(tr("Template name"))
        nl.setStyleSheet(f"color: {theme.MUTED}; font-size: 12px;")
        col.addWidget(nl)
        self.name = QLineEdit(self.t.label)
        self.name.setPlaceholderText(tr("e.g. Football kick · right leg"))
        self.name.setMinimumHeight(40)
        self.name.setStyleSheet("font-size: 17px; font-weight: 600;")
        col.addWidget(self.name)
        top.addLayout(col, 1)
        lay.addLayout(top)

        cols = QHBoxLayout()
        cols.setSpacing(18)
        # 1 · what to measure
        c1 = self._card("1", tr("What do you want to measure?"))
        hint = QLabel(tr("Tap a joint on the figure or tick it in the list."))
        hint.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px; border: 0;")
        c1.layout().addWidget(hint)
        row = QHBoxLayout()
        self.body = BodyPicker()
        self.body.toggled.connect(self._toggle_angle)
        row.addWidget(self.body, 0, Qt.AlignTop)
        self.measure_list = QListWidget()
        self.measure_list.setStyleSheet(f"QListWidget {{ background: transparent; border: 0; }}"
                                        f"QListWidget::item {{ padding: 6px 4px; border-radius: 6px; }}"
                                        f"QListWidget::item:hover {{ background: {theme.RAISED}; }}")
        self.measure_list.itemChanged.connect(self._list_changed)
        row.addWidget(self.measure_list, 1)
        c1.layout().addLayout(row)
        cols.addWidget(c1, 5)
        # 2 · phases, 3 · results
        right = QVBoxLayout()
        right.setSpacing(18)
        c2 = self._card("2", tr("Phases of the movement"))
        self.phases = QListWidget()
        self.phases.setDragDropMode(QAbstractItemView.InternalMove)
        self.phases.setStyleSheet(f"QListWidget {{ background: transparent; border: 0; }}"
                                  f"QListWidget::item {{ background: {theme.CARD}; border: 1px solid #222834; border-radius: 8px;"
                                  f" padding: 8px 10px; margin-bottom: 6px; }}")
        for ph in self.t.phase_names():
            self._add_phase(ph)
        c2.layout().addWidget(self.phases, 1)
        prow = QHBoxLayout()
        addp = QPushButton(icons.icon("plus", theme.ACCENT, 14), " " + tr("Add phase"))
        addp.setProperty("ghost", True)
        addp.clicked.connect(lambda: self._add_phase(tr("Phase {n}", n=self.phases.count() + 1), edit=True))
        prow.addWidget(addp)
        rem = QPushButton(tr("Remove selected"))
        rem.setProperty("ghost", True)
        rem.clicked.connect(lambda: [self.phases.takeItem(self.phases.row(i)) for i in self.phases.selectedItems()])
        prow.addWidget(rem)
        prow.addStretch(1)
        c2.layout().addLayout(prow)
        tip = QLabel(tr("Drag to reorder, double-click to rename. While analyzing, “Mark phase here” suggests these names in order."))
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {theme.DIM}; font-size: 12px; border: 0;")
        c2.layout().addWidget(tip)
        right.addWidget(c2, 1)
        c3 = self._card("3", tr("Results"))
        r3 = QLabel(tr("Every template gets range of motion, min/max per phase, angular velocity, "
                       "trajectory speed and phase durations in the Results page and the PDF report."))
        r3.setWordWrap(True)
        r3.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px; border: 0;")
        c3.layout().addWidget(r3)
        right.addWidget(c3)
        cols.addLayout(right, 4)
        lay.addLayout(cols, 1)

        foot = QHBoxLayout()
        note = QLabel(tr("Saved as a small file you can share with your classmates."))
        note.setStyleSheet(f"color: {theme.DIM}; font-size: 12px;")
        foot.addWidget(note, 1)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        foot.addWidget(cancel)
        if can_apply:
            ap = QPushButton(tr("Save and apply to this video"))
            ap.clicked.connect(lambda: self._save(apply=True))
            foot.addWidget(ap)
        save = QPushButton(tr("Save template"))
        save.setProperty("primary", True)
        save.clicked.connect(lambda: self._save(apply=False))
        foot.addWidget(save)
        lay.addLayout(foot)
        self._fill_measures()

    def _card(self, num: str, title: str) -> QFrame:
        f = QFrame()
        f.setObjectName("Card")
        f.setStyleSheet(f"QFrame#Card {{ background: {theme.PANEL}; border: 1px solid #1E232C; border-radius: 14px; }}")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        h = QHBoxLayout()
        n = QLabel(num)
        n.setFixedSize(24, 24)
        n.setAlignment(Qt.AlignCenter)
        n.setStyleSheet(f"background: {theme.ACTIVE}; color: {theme.ACCENT}; border-radius: 12px; font-size: 12px; border: 0;")
        h.addWidget(n)
        t = QLabel(title)
        t.setStyleSheet("font-size: 15px; font-weight: 600; border: 0;")
        h.addWidget(t, 1)
        lay.addLayout(h)
        return f

    def _fill_measures(self):
        self.measure_list.blockSignals(True)
        self.measure_list.clear()
        chosen = {tuple(a) for a in self.t.angles}
        for j, (a, o, b) in L.ANGLE_PRESETS.items():
            it = QListWidgetItem(tr("Angle · {name}", name=tr(L.DISPLAY[o])))
            it.setData(Qt.UserRole, ("angle", (a, o, b)))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if (a, o, b) in chosen else Qt.Unchecked)
            self.measure_list.addItem(it)
        for name in ("right_ankle", "left_ankle", "right_wrist", "left_wrist", "right_knee", "left_knee",
                     "right_hip", "left_hip", "right_foot_index", "left_foot_index", "nose"):
            it = QListWidgetItem(tr("Trajectory · {name}", name=tr(L.DISPLAY[name])))
            it.setData(Qt.UserRole, ("trajectory", name))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if name in self.t.trajectories else Qt.Unchecked)
            self.measure_list.addItem(it)
        self.measure_list.blockSignals(False)
        self.body.selected = {o for a, o, b in self.t.angles if o in self.body.presets}
        self.body.update()

    def _list_changed(self, _):
        angles, tracks = [], []
        for i in range(self.measure_list.count()):
            it = self.measure_list.item(i)
            if it.checkState() == Qt.Checked:
                kind, val = it.data(Qt.UserRole)
                (angles if kind == "angle" else tracks).append(list(val) if kind == "angle" else val)
        self.t.angles, self.t.trajectories = angles, tracks
        self.body.selected = {a[1] for a in angles}
        self.body.update()

    def _toggle_angle(self, vertex: str):
        preset = list(self.body.presets[vertex])
        if preset in self.t.angles:
            self.t.angles.remove(preset)
        else:
            self.t.angles.append(preset)
        self._fill_measures()

    def _add_phase(self, name: str, edit: bool = False):
        it = QListWidgetItem(name)
        it.setFlags(it.flags() | Qt.ItemIsEditable)
        self.phases.addItem(it)
        if edit:
            self.phases.editItem(it)

    def _save(self, apply: bool):
        name = self.name.text().strip()
        if not name:
            self.name.setFocus()
            self.name.setPlaceholderText(tr("Give the template a name"))
            return
        self.t.name = name
        self.t.phases = [self.phases.item(i).text().strip() for i in range(self.phases.count())
                         if self.phases.item(i).text().strip()]
        if not self.t.icon:
            self.t.icon = "M22 11a3 3 0 1 0 0-6a3 3 0 0 0 0 6M22 11v13M22 14l6 5M22 14l-6 5M22 24l3 9 1 9M22 24l-3 9-2 9"
        templates.save_user_template(self.t)
        if apply:
            self.applyRequested.emit(self.t)
        self.accept()


# ── shortcuts / about ──────────────────────────────────────────────────


SHORTCUTS = [
    ("Space", "Play / pause"), ("← / →", "Previous / next frame"), ("Shift + ← / →", "Jump 10 frames"),
    ("Home / End", "First / last frame"), ("Tab", "Next doubtful frame"),
    ("V", "Select and move points"), ("A", "Angle tool"), ("T", "Trajectory tool"), ("L", "Line and distance"),
    ("X", "Text note"), ("C", "Calibrate"), ("Z", "Magnifier"), ("F", "Mark phase here"),
    ("Mouse wheel", "Zoom"), ("Shift + drag / middle button", "Pan the video"), ("Double click", "Fit to window"),
    ("Ctrl + Z / Ctrl + Y", "Undo / redo"), ("Ctrl + S", "Save"), ("Ctrl + O", "Open video"),
    ("Ctrl + R", "Results"), ("Delete", "Delete the selected measurement"), ("Esc", "Cancel the current tool"),
    ("G", "Shadow of the previous and next frame"),
]


class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        _dialog_style(self)
        self.setWindowTitle(tr("Keyboard shortcuts"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(14)
        lay.addWidget(_title(tr("Keyboard shortcuts")))
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        for i, (k, d) in enumerate(SHORTCUTS):
            kl = QLabel(k)
            kl.setFont(theme.mono(12, 500))
            kl.setStyleSheet(f"background: {theme.RAISED}; border-radius: 6px; padding: 3px 8px;")
            dl = QLabel(tr(d))
            dl.setStyleSheet(f"color: {theme.TEXT_2};")
            grid.addWidget(kl, i % 11, (i // 11) * 2, Qt.AlignRight)
            grid.addWidget(dl, i % 11, (i // 11) * 2 + 1)
        lay.addLayout(grid)
        ok = QPushButton(tr("Close"))
        ok.setProperty("primary", True)
        ok.clicked.connect(self.accept)
        lay.addWidget(ok, 0, Qt.AlignRight)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        _dialog_style(self)
        self.setWindowTitle(tr("About OpenKick"))
        self.setFixedWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 22)
        lay.setSpacing(12)
        row = QHBoxLayout()
        lg = QLabel()
        lg.setPixmap(icons.logo(44))
        row.addWidget(lg)
        name = QLabel(f"<span style='color:{theme.MUTED}; font-weight:500'>open</span><b>kick</b>")
        name.setFont(theme.display(30))
        row.addWidget(name)
        row.addStretch(1)
        lay.addLayout(row)
        by = QLabel(tr("Developed by {name}", name=AUTHOR))
        by.setStyleSheet(f"color: {theme.ACCENT}; font-weight: 600;")
        lay.addWidget(by)
        for text in (tr("Version {v}", v=__version__),
                     tr("Open-source biomechanics video analysis with automatic pose detection. "
                        "Reads and writes Kinovea .kva files."),
                     tr("License: GNU GPL v3 or later. Pose detection: Google MediaPipe (Apache 2.0). "
                        "Fonts: IBM Plex and Space Grotesk (SIL OFL)."),
                     tr("3D: RTMPose and its person detector (OpenMMLab, through rtmlib, Apache 2.0), downloaded "
                        "from their official site. Their training datasets (Halpe, Human-Art and others) are "
                        "for non-commercial use only."),
                     tr("OpenKick is not a medical device. Its measurements help the professional's judgement and "
                        "do not replace it.")):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {theme.TEXT_2};")
            lay.addWidget(lbl)
        ok = QPushButton(tr("Close"))
        ok.setProperty("primary", True)
        ok.clicked.connect(self.accept)
        lay.addWidget(ok, 0, Qt.AlignRight)


# ── text notes ─────────────────────────────────────────────────────────

NOTE_COLORS = ["#E8EAED", "#C6F24E", "#FACC15", "#FF8A3D", "#FF5CA8", "#5EEAD4", "#60A5FA"]


class FadePreview(QWidget):
    """Opacity of the note over time: ramp in, plateau, ramp out."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self.values = (0.0, 1.0, 0.0, 0.5, 2.0)   # start, end, fade_in, fade_out, duration

    def paintEvent(self, _):
        from PySide6.QtGui import QPainterPath, QPolygonF
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        ts, te, fi, fo, dur = self.values
        w, h = self.width(), self.height()
        top, bot = 8, h - 14
        X = lambda t: max(0.0, min(w, t / max(dur, 1e-6) * w))  # noqa: E731
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawLine(QPointF(0, bot), QPointF(w, bot))
        poly = QPolygonF([QPointF(X(ts), bot), QPointF(X(ts + fi), top), QPointF(X(te), top),
                          QPointF(X(te + fo), bot)])
        c = QColor(theme.ACCENT)
        c.setAlpha(60)
        p.setBrush(c)
        p.setPen(QPen(QColor(theme.ACCENT), 1.6))
        p.drawPolygon(poly)
        p.setPen(QColor(theme.DIM))
        p.setFont(theme.mono(10))
        p.drawText(QRectF(0, bot + 1, 80, 13), Qt.AlignLeft, "0 s")
        p.drawText(QRectF(w - 80, bot + 1, 80, 13), Qt.AlignRight, f"{dur:.2f} s")


class NoteDialog(QDialog):
    """Create or edit a text note: content, color, visible interval and fades."""

    def __init__(self, video, note, current_frame: int, parent=None):
        from PySide6.QtWidgets import QButtonGroup, QComboBox, QDoubleSpinBox, QPlainTextEdit
        super().__init__(parent)
        _dialog_style(self)
        self.video, self.note, self.current = video, note, current_frame
        self.setWindowTitle(tr("Note"))
        self.setFixedWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(14)
        lay.addWidget(_title(tr("Text note"), 20))
        self.text = QPlainTextEdit(note.text)
        self.text.setPlaceholderText(tr("Write the note…"))
        self.text.setFixedHeight(84)
        lay.addWidget(self.text)

        row = QHBoxLayout()
        row.setSpacing(8)
        cl = QLabel(tr("Color"))
        cl.setStyleSheet(f"color: {theme.MUTED};")
        row.addWidget(cl)
        self.colors = QButtonGroup(self)
        for i, c in enumerate(NOTE_COLORS):
            b = QPushButton()
            b.setCheckable(True)
            b.setFixedSize(26, 26)
            b.setStyleSheet(f"QPushButton {{ background: {c}; border-radius: 13px; border: 2px solid {theme.CARD_2}; min-height: 0; padding: 0; }}"
                            f"QPushButton:checked {{ border: 2px solid {theme.TEXT}; }}")
            b.setChecked(c.upper() == note.color.upper())
            self.colors.addButton(b, i)
            row.addWidget(b)
        if self.colors.checkedId() < 0:
            self.colors.button(0).setChecked(True)
        row.addStretch(1)
        lay.addLayout(row)

        dur = video.duration
        step = 1.0 / max(video.fps, 1)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        def spin(value):
            s = QDoubleSpinBox()
            s.setRange(0, dur)
            s.setDecimals(2)
            s.setSingleStep(step)
            s.setSuffix(" s")
            s.setValue(value)
            s.setFont(theme.mono(13))
            s.setMinimumWidth(110)
            s.valueChanged.connect(self._update)
            return s

        def now_btn(target):
            b = QPushButton(tr("Current frame"))
            b.setToolTip(tr("Use the time of the frame you are on"))
            b.clicked.connect(lambda: target.setValue(video.time_of(current_frame)))
            return b

        self.start = spin(video.time_of(note.start))
        self.end = spin(video.time_of(note.end))
        for r, (label, sp) in enumerate(((tr("Visible from"), self.start), (tr("until"), self.end))):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {theme.MUTED};")
            grid.addWidget(lbl, r, 0)
            grid.addWidget(sp, r, 1)
            grid.addWidget(now_btn(sp), r, 2)
        whole = QPushButton(tr("Whole video"))
        whole.clicked.connect(lambda: (self.start.setValue(0), self.end.setValue(video.time_of(video.frame_count - 1))))
        grid.addWidget(whole, 0, 3)

        def fade_row(r, label, value):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {theme.MUTED};")
            grid.addWidget(lbl, r, 0)
            mode = QComboBox()
            mode.addItem(tr("At once"), False)
            mode.addItem(tr("Fading"), True)
            mode.setCurrentIndex(1 if value > 0 else 0)
            grid.addWidget(mode, r, 1)
            secs = QDoubleSpinBox()
            secs.setRange(0.05, 10)
            secs.setDecimals(2)
            secs.setSingleStep(0.1)
            secs.setSuffix(" s")
            secs.setValue(value if value > 0 else 0.5)
            secs.setFont(theme.mono(13))
            secs.setEnabled(value > 0)
            mode.currentIndexChanged.connect(lambda _: (secs.setEnabled(mode.currentData()), self._update()))
            secs.valueChanged.connect(self._update)
            grid.addWidget(secs, r, 2)
            return mode, secs

        self.in_mode, self.in_secs = fade_row(2, tr("Appears"), note.fade_in)
        self.out_mode, self.out_secs = fade_row(3, tr("Disappears"), note.fade_out)
        lay.addLayout(grid)
        self.preview = FadePreview()
        lay.addWidget(self.preview)
        self.summary = QLabel()
        self.summary.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        ok = QPushButton(tr("Save note"))
        ok.setProperty("primary", True)
        ok.clicked.connect(self._accept)
        row.addWidget(ok)
        lay.addLayout(row)
        self._update()

    def _fades(self):
        fi = self.in_secs.value() if self.in_mode.currentData() else 0.0
        fo = self.out_secs.value() if self.out_mode.currentData() else 0.0
        return fi, fo

    def _update(self, *_):
        ts, te = self.start.value(), self.end.value()
        if te < ts:
            te = ts
        fi, fo = self._fades()
        self.preview.values = (ts, te, fi, fo, self.video.duration)
        self.preview.update()
        txt = tr("Visible {a} – {b} s", a=f"{ts:.2f}", b=f"{te:.2f}")
        if fi:
            txt += " · " + tr("fades in over {s} s", s=f"{fi:.2f}")
        if fo:
            txt += " · " + tr("then fades out over {s} s", s=f"{fo:.2f}")
        else:
            txt += " · " + tr("then disappears at once")
        self.summary.setText(txt)

    def _accept(self):
        if not self.text.toPlainText().strip():
            self.text.setFocus()
            return
        self.accept()

    def values(self) -> dict:
        v = self.video
        s = v.index_at_time(self.start.value())
        e = max(s, v.index_at_time(self.end.value()))
        fi, fo = self._fades()
        return {"text": self.text.toPlainText().strip(), "color": NOTE_COLORS[max(0, self.colors.checkedId())],
                "start": s, "end": e, "fade_in": fi, "fade_out": fo}
