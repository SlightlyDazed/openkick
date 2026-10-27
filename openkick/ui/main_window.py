"""Main window: top bar with menus, the start page, the analysis workspace and results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QInputDialog,
                               QLabel, QMainWindow, QMenu, QMenuBar, QMessageBox, QProgressDialog,
                               QPushButton, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from openkick import APP_NAME, AUTHOR, __version__
from openkick.core import analysis as A
from openkick.core import templates
from openkick.core.project import SRC_AUTO, SRC_MANUAL, Pose, Project
from openkick.core.video import probe
from openkick.formats import csvexport, kva
from openkick.i18n import LANGUAGES, language, tr
from openkick.plugins import PluginContext, discover, user_dir
from openkick.core import patients
from openkick.ui import icons, theme
from openkick.ui.canvas import VideoCanvas
from openkick.ui.controller import Controller
from openkick.ui.dialogs import (AboutDialog, PluginResultDialog, PluginsDialog, ShortcutsDialog, TemplateEditor,
                                 disabled_plugins)
from openkick.ui.exporters import VideoExportWorker, write_pdf
from openkick.ui.panels import RightPanel
from openkick.ui.processing import ProcessingDialog
from openkick.ui.results import ResultsPage
from openkick.ui.timeline import Timeline, TransportBar, mark_phase
from openkick.ui.welcome import VIDEO_EXT, WelcomePage

VIDEO_FILTER = "Video (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv *.mpg *.mpeg *.MP4 *.MOV *.AVI)"

TOOLS = [
    ("select", "select", "Select and move points", "V"),
    ("angle", "angle", "Angle", "A"),
    ("trajectory", "track", "Trajectory", "T"),
    ("distance", "line", "Line and distance", "L"),
    ("marker", "target", "Marker / point", "P"),
    ("text", "text", "Text note", "X"),
    ("calibrate", "calib", "Calibrate", "C"),
    ("zoom", "zoom", "Magnifier", "Z"),
]
TOOL_HINTS = {
    "select": "Drag a joint to correct it · Shift + drag or middle button moves the video",
    "angle": "Click a joint (knee, hip, elbow, shoulder, ankle), or mark 3 points: end, vertex, end (Shift + click to start on a joint)",
    "trajectory": "Click a joint to follow its trajectory, or an object for a free point",
    "distance": "Mark both ends (they snap to the joints)",
    "marker": "Click on the body to attach a marker to that segment, or anywhere else for a free marker",
    "text": "Click where you want the note",
    "calibrate": "Mark the ends of a known length (e.g. 1 m on the floor)",
    "zoom": "Click to zoom in · Alt + click to zoom out",
}
TOOL_COLORS = {"select": theme.ACCENT, "angle": "#FF8A3D", "trajectory": "#FF5CA8", "distance": "#5EEAD4",
               "marker": "#60A5FA", "text": theme.MUTED, "calibrate": theme.ACCENT, "zoom": theme.MUTED}


class ElidedLabel(QLabel):
    """Label that shrinks with "…" instead of forcing the window wider."""

    def __init__(self, text=""):
        super().__init__()
        self._full = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)

    def setText(self, text):
        self._full = text
        self._elide()

    def _elide(self):
        fm = self.fontMetrics()
        super().setText(fm.elidedText(self._full, Qt.ElideRight, max(10, self.width())))
        self.setToolTip(self._full if fm.horizontalAdvance(self._full) > self.width() else "")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()


def hline(vertical=False) -> QFrame:
    f = QFrame()
    if vertical:
        f.setFixedSize(1, 24)
    else:
        f.setFixedHeight(1)
    f.setStyleSheet(f"background: {theme.LINE};")
    return f


class TopBar(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.setObjectName("Panel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(52)
        self.setStyleSheet(f"TopBar {{ background: {theme.PANEL}; border-bottom: 1px solid {theme.LINE}; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 14, 0)
        lay.setSpacing(14)
        logo = QLabel()
        logo.setPixmap(icons.logo(28))
        lay.addWidget(logo)
        self._project_mode = False
        word = QLabel(f"<span style='color:{theme.MUTED}; font-weight:500'>open</span><b>kick</b>")
        word.setFont(theme.display(18))
        self.word = word
        lay.addWidget(word)
        lay.addSpacing(6)
        self.menubar = QMenuBar()
        self.menubar.setNativeMenuBar(False)
        self.menubar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        lay.addWidget(self.menubar, 0, Qt.AlignVCenter)
        self.tabs = QWidget()
        tl = QHBoxLayout(self.tabs)
        tl.setContentsMargins(8, 0, 0, 0)
        tl.setSpacing(2)
        self.tab_group = QButtonGroup(self)
        for i, text in enumerate((tr("Video"), tr("Results"))):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setStyleSheet(f"QPushButton {{ background: transparent; border: 0; border-radius: 0; color: {theme.MUTED};"
                            f" min-height: 50px; padding: 0 12px; border-bottom: 2px solid transparent; }}"
                            f"QPushButton:checked {{ color: {theme.TEXT}; border-bottom-color: {theme.ACCENT}; font-weight: 600; }}"
                            f"QPushButton:hover:!checked {{ color: {theme.TEXT_2}; }}")
            self.tab_group.addButton(b, i)
            tl.addWidget(b)
        self.tab_group.button(0).setChecked(True)
        lay.addWidget(self.tabs)
        lay.addSpacing(8)
        self.title = ElidedLabel()
        self.title.setMinimumWidth(90)
        self.title.setStyleSheet("font-weight: 500;")
        self.title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.title, 2)
        self.status = QLabel()
        self.status.setStyleSheet(f"color: {theme.DIM};")
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.undo = QToolButton()
        self.undo.setIcon(icons.icon("undo", theme.MUTED, 18))
        self.undo.setFixedSize(36, 36)
        self.undo.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self.undo)
        self.redo = QToolButton()
        self.redo.setIcon(icons.icon("redo", theme.MUTED, 18))
        self.redo.setFixedSize(36, 36)
        self.redo.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self.redo)
        lay.addWidget(hline(True))
        self.detect = QPushButton(icons.icon("detect", theme.ACCENT, 16), " " + tr("Detect pose"))
        self.detect.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self.detect)
        self.export = QPushButton(icons.icon("export", theme.ON_ACCENT, 16), " " + tr("Export") + "  ")
        self.export.setProperty("primary", True)
        self.export.setFocusPolicy(Qt.NoFocus)
        lay.addWidget(self.export)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        """Progressively drop non-essential items so the bar never overlaps on small screens."""
        w = self.width()
        on = self._project_mode
        compact = w < 1400
        self.detect.setText("" if compact else " " + tr("Detect pose"))
        self.detect.setToolTip(tr("Detect pose") if compact else "")
        self.status.setVisible(on and w >= 1380)
        self.word.setVisible(w >= 1330 or not on)
        for b in (self.undo, self.redo):
            b.setVisible(on and w >= 1200)

    def set_project_mode(self, on: bool):
        self._project_mode = on
        for w in (self.tabs, self.title, self.status, self.undo, self.redo, self.detect, self.export):
            w.setVisible(on)
        self._fit()


class ContextBar(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        ctl = win.ctl
        self.setFixedHeight(44)
        self.setStyleSheet(f"ContextBar {{ border-bottom: 1px solid {theme.LINE_SOFT}; }}")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 12, 0)
        lay.setSpacing(12)
        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        lay.addWidget(self.dot)
        self.tool = QLabel()
        self.tool.setStyleSheet("font-weight: 500;")
        lay.addWidget(self.tool)
        self.hint = ElidedLabel()
        self.hint.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(self.hint, 1)
        self.layers_btn = QPushButton(icons.icon("eye", theme.MUTED, 14), " " + tr("Layers") + " ")
        self.layers_btn.setFocusPolicy(Qt.NoFocus)
        self.layers_btn.setStyleSheet(f"QPushButton {{ min-height: 26px; padding: 0 10px; border-radius: 13px; font-size: 12px;"
                                      f" color: {theme.TEXT_2}; background: {theme.CARD}; border: 1px solid {theme.BORDER}; }}")
        menu = QMenu(self)
        for key, label, color in (("skeleton", tr("Skeleton"), theme.RIGHT), ("angles", tr("Angles and distances"), "#FF8A3D"),
                                  ("trajectories", tr("Trajectories"), "#FF5CA8"), ("markers", tr("Markers"), "#60A5FA"),
                                  ("notes", tr("Notes"), theme.MUTED)):
            a = menu.addAction(icons.icon("M12 12m-5 0a5 5 0 1 0 10 0a5 5 0 1 0-10 0", color, 12, fill=True), label)
            a.setCheckable(True)
            a.setChecked(True)
            a.toggled.connect(lambda on, k=key: self._layer(k, on))
        menu.addSeparator()
        allp = menu.addAction(tr("All 33 MediaPipe points"))
        allp.setCheckable(True)
        allp.setToolTip(tr("Also show eyes, ears, mouth and finger points"))
        allp.toggled.connect(self._all_points)
        self.ghost_act = menu.addAction(tr("Shadow of the previous and next frame (G)"))
        self.ghost_act.setCheckable(True)
        self.ghost_act.setChecked(QSettings().value("ghost", True, type=bool))
        self.ghost_act.setToolTip(tr("Faint skeletons of the neighbouring frames, to line up a correction with them"))
        self.ghost_act.toggled.connect(lambda on: self.win.canvas.set_ghost(on))
        self.layers_btn.setMenu(menu)
        lay.addWidget(self.layers_btn)
        lay.addWidget(hline(True))
        self.calib = QPushButton()
        self.calib.setFocusPolicy(Qt.NoFocus)
        self.calib.setFont(theme.mono(12))
        self.calib.setStyleSheet(f"QPushButton {{ background: transparent; border: 0; color: {theme.MUTED}; min-height: 26px; padding: 0 6px; }}"
                                 f"QPushButton:hover {{ color: {theme.TEXT}; }}")
        self.calib.setToolTip(tr("Calibrate a known length to get real units (C)"))
        self.calib.clicked.connect(lambda: ctl.set_tool("calibrate"))
        lay.addWidget(self.calib)
        self.focus = QPushButton(icons.icon("fit", theme.MUTED, 14), " " + tr("Focus athlete"))
        self.focus.setFocusPolicy(Qt.NoFocus)
        self.focus.setToolTip(tr("Zoom to the athlete's movement"))
        self.focus.setStyleSheet(f"QPushButton {{ background: transparent; border: 0; color: {theme.MUTED}; min-height: 26px;"
                                 f" padding: 0 6px; font-size: 12px; }} QPushButton:hover {{ color: {theme.TEXT}; }}")
        self.focus.clicked.connect(lambda: win.canvas.focus_athlete())
        lay.addWidget(self.focus)
        self.zoom = QPushButton("100%")
        self.zoom.setFocusPolicy(Qt.NoFocus)
        self.zoom.setFont(theme.mono(12))
        self.zoom.setToolTip(tr("Fit to window"))
        self.zoom.setStyleSheet(self.calib.styleSheet())
        self.zoom.clicked.connect(lambda: win.canvas.fit())
        lay.addWidget(self.zoom)
        ctl.toolChanged.connect(self.set_tool)
        ctl.changed.connect(self.refresh)
        ctl.projectLoaded.connect(self.refresh)
        self.set_tool("select")

    def _all_points(self, on):
        self.win.canvas.opts.all_points = on
        self.win.canvas.update()

    def _layer(self, key, on):
        self.win.ctl.layers[key] = on
        if key == "angles":
            self.win.ctl.layers["distances"] = on
        self.win.canvas.update()

    def set_tool(self, tool: str):
        name = next(t[2] for t in TOOLS if t[0] == tool)
        self.tool.setText(tr(name))
        self.dot.setStyleSheet(f"background: {TOOL_COLORS[tool]}; border-radius: 4px;")
        self.hint.setText(tr(TOOL_HINTS[tool]))
        self._hint = tr(TOOL_HINTS[tool])

    def message(self, text: str):
        self.hint.setText(text or getattr(self, "_hint", ""))

    def refresh(self):
        p = self.win.ctl.project
        if not p:
            return
        self.calib.setText(f"1 m = {round(p.calibration.px_per_m)} px" if p.calibration else tr("Not calibrated"))
        self.zoom.setText(f"{self.win.canvas.zoom_percent}%")


class Workspace(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.setObjectName("Page")
        ctl = win.ctl
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        rail = QWidget()
        rail.setObjectName("Panel")
        rail.setAttribute(Qt.WA_StyledBackground, True)
        rail.setFixedWidth(62)
        rail.setStyleSheet(f"QWidget#Panel {{ background: {theme.PANEL}; border-right: 1px solid {theme.LINE}; }}")
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(9, 12, 9, 12)
        rl.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons = {}
        for i, (key, ic, name, sc) in enumerate(TOOLS):
            if key in ("angle", "calibrate"):
                rl.addSpacing(6)
                rl.addWidget(hline())
                rl.addSpacing(6)
            b = QToolButton()
            b.setObjectName("RailButton")
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setIcon(icons.icon(ic, theme.MUTED, 22, checked_color=theme.ACCENT))
            b.setIconSize(QSize(22, 22))
            b.setFixedSize(44, 44)
            b.setToolTip(f"{tr(name)}  ({sc})")
            self.tool_group.addButton(b, i)
            self.tool_buttons[key] = b
            rl.addWidget(b)
            if key == "text":
                ph = QToolButton()
                ph.setObjectName("RailButton")
                ph.setFocusPolicy(Qt.NoFocus)
                ph.setIcon(icons.icon("phase", theme.MUTED, 22))
                ph.setIconSize(QSize(22, 22))
                ph.setFixedSize(44, 44)
                ph.setToolTip(tr("Mark phase here") + "  (F)")
                ph.clicked.connect(lambda: mark_phase(win, ctl))
                rl.addWidget(ph)
        rl.addStretch(1)
        self.tool_buttons["select"].setChecked(True)
        self.tool_group.idClicked.connect(lambda i: ctl.set_tool(TOOLS[i][0]))
        ctl.toolChanged.connect(lambda t: self.tool_buttons[t].setChecked(True))
        lay.addWidget(rail)

        center = QVBoxLayout()
        center.setSpacing(0)
        self.context = ContextBar(win)
        center.addWidget(self.context)
        center.addWidget(win.canvas, 1)
        bottom = QWidget()
        bottom.setObjectName("Panel")
        bottom.setAttribute(Qt.WA_StyledBackground, True)
        bottom.setStyleSheet(f"QWidget#Panel {{ background: {theme.PANEL}; border-top: 1px solid {theme.LINE}; }}")
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(win.transport)
        sep = hline()
        sep.setStyleSheet(f"background: {theme.LINE_SOFT};")
        bl.addWidget(sep)
        bl.addWidget(win.timeline)
        center.addWidget(bottom)
        cw = QWidget()
        cw.setLayout(center)
        lay.addWidget(cw, 1)
        right = RightPanel(ctl)
        right.setStyleSheet(f"RightPanel {{ border-left: 1px solid {theme.LINE}; }}")
        right.phases.markPhase.connect(lambda: mark_phase(win, ctl))
        self.right = right
        lay.addWidget(right)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings()
        self.ctl = Controller(self)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(icons.icon("M0 0", theme.ACCENT))
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.pending_template: str | None = None
        self.processing: ProcessingDialog | None = None

        self.canvas = VideoCanvas(self.ctl)
        self.transport = TransportBar(self.ctl)
        self.timeline = Timeline(self.ctl)
        self.topbar = TopBar(self)
        self.setMenuWidget(self.topbar)
        self.stack = QStackedWidget()
        self.welcome = WelcomePage()
        self.workspace = Workspace(self)
        self.results = ResultsPage(self.ctl)
        for w in (self.welcome, self.workspace, self.results):
            self.stack.addWidget(w)
        self.setCentralWidget(self.stack)
        from PySide6.QtGui import QIcon
        self.setWindowIcon(QIcon(icons.logo(64)))

        self._actions()
        self._menus()
        self._wire()
        self.show_welcome()

    # ── setup ──────────────────────────────────────────────
    def show_models(self):
        """The first-start window: which detection models are here, and getting the others."""
        from openkick.ui.firstrun import FirstRunDialog
        FirstRunDialog(self).exec()

    def _act(self, text, slot, shortcut=None, icon=None, needs_project=True) -> QAction:
        a = QAction(tr(text), self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.WindowShortcut)
        if icon:
            a.setIcon(icons.icon(icon, theme.TEXT_2, 16))
        a.triggered.connect(slot)
        a.setData(needs_project)
        self.addAction(a)
        return a

    def _actions(self):
        c = self.ctl
        self.a_open = self._act("Open video…", lambda: self.open_video(), "Ctrl+O", "folder", False)
        self.a_open_any = self._act("Open project or KVA…", self.open_project_dialog, "Ctrl+Shift+O", None, False)
        self.a_save = self._act("Save", self.save, "Ctrl+S", "save")
        self.a_save_as = self._act("Save as…", self.save_as, "Ctrl+Shift+S")
        self.a_import_kva = self._act("Import KVA into this project…", self.import_kva)
        self.a_exp_kva = self._act("Kinovea (.kva)", self.export_kva, "Ctrl+E", "export")
        self.a_exp_csv = self._act("Measurements (CSV)", self.export_csv, None, "table")
        self.a_exp_lm = self._act("Landmarks (CSV, MediaPipe format)", self.export_landmarks)
        self.a_exp_video = self._act("Video with drawings (MP4)", self.export_video, None, "video")
        self.a_exp_pdf = self._act("PDF report", self.export_pdf, None, "pdf")
        self.a_home = self._act("Back to start", self.go_home, None, "home")
        self.a_quit = self._act("Quit", self.close, "Ctrl+Q", None, False)
        self.a_undo = c.undo.createUndoAction(self, tr("Undo"))
        self.a_undo.setShortcut(QKeySequence.Undo)
        self.a_undo.setIcon(icons.icon("undo", theme.MUTED, 18))
        self.a_redo = c.undo.createRedoAction(self, tr("Redo"))
        self.a_redo.setIcon(icons.icon("redo", theme.MUTED, 18))
        self.a_redo.setShortcuts([QKeySequence.Redo, QKeySequence("Ctrl+Y")])
        self.addAction(self.a_undo)
        self.addAction(self.a_redo)
        self.a_delete = self._act("Delete selected measurement", self.delete_selected, "Delete", "trash")
        self.a_revert = self._act("Revert selected joint to MediaPipe", self.revert_selected)
        self.a_fit = self._act("Fit to window", lambda: self.canvas.fit(), "0", "fit")
        self.a_zin = self._act("Zoom in", lambda: self.canvas.zoom_at(1.25), "+")
        self.a_zout = self._act("Zoom out", lambda: self.canvas.zoom_at(0.8), "-")
        self.a_results = self._act("Results", self.toggle_results, "Ctrl+R", "chart")
        self.a_tools = {}
        for key, ic, name, sc in TOOLS:
            self.a_tools[key] = self._act(name, lambda _=False, k=key: self.set_tool(k), sc, ic)
        self.a_phase = self._act("Mark phase here", lambda: mark_phase(self, c), "F", "flag")
        self.a_note = self._act("Add note at the center", self.add_note_center, None, "note")
        self.a_detect = self._act("Detect pose again…", self.redetect, None, "detect")
        self.a_smooth = self._act("Smooth trajectories (Butterworth 6 Hz)", self.smooth, None, "smooth")
        self.a_next_low = self._act("Next doubtful frame", self.workspace_next_low, None, None)
        self.a_new_tpl = self._act("New analysis template…", lambda: self.edit_template(None), None, "template", False)
        self.a_multiview = self._act("3D from several videos (experimental)…", self.open_multiview, None, "cube", False)
        self.a_subject = self._act("Patient record…", self.edit_subject, None, "user")
        self.a_compare = self._act("Compare analyses…", self.open_compare, None, "compare", False)
        self.a_patients = self._act("Patients…", self.open_patients, None, "user", False)
        self.a_plugins = self._act("Manage plugins…", self.manage_plugins, None, "plugin", False)
        self.a_plugin_dir = self._act("Open plugins folder", lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(user_dir()))), None, "folder", False)
        self.a_shortcuts = self._act("Keyboard shortcuts", lambda: ShortcutsDialog(self).exec(), "F1", "keyboard", False)
        self.a_about = self._act("About OpenKick", lambda: AboutDialog(self).exec(), None, "info", False)
        self.a_models = self._act("Detection models…", self.show_models, None, "detect", False)
        # keys that widgets would otherwise consume
        for seq, fn in (("Space", self.transport.toggle), ("Right", lambda: c.step(1)), ("Left", lambda: c.step(-1)),
                        ("Shift+Right", lambda: c.step(10)), ("Shift+Left", lambda: c.step(-10)),
                        ("Home", lambda: c.set_frame(0)), ("End", lambda: c.set_frame(10 ** 9)),
                        ("Tab", self.workspace_next_low), ("G", lambda: self.workspace.context.ghost_act.toggle())):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(lambda fn=fn: fn() if self.ctl.has_project and self.stack.currentWidget() is not self.welcome else None)

    def _menus(self):
        mb = self.topbar.menubar
        m = mb.addMenu(tr("File"))
        m.addAction(self.a_open)
        m.addAction(self.a_open_any)
        self.recent_menu = m.addMenu(tr("Recent"))
        m.addSeparator()
        m.addAction(self.a_save)
        m.addAction(self.a_save_as)
        m.addSeparator()
        m.addAction(self.a_import_kva)
        self.export_menu = m.addMenu(icons.icon("export", theme.TEXT_2, 16), tr("Export"))
        m.addSeparator()
        m.addAction(self.a_home)
        m.addAction(self.a_quit)
        m = mb.addMenu(tr("Edit"))
        m.addAction(self.a_undo)
        m.addAction(self.a_redo)
        m.addSeparator()
        m.addAction(self.a_delete)
        m.addAction(self.a_revert)
        m = mb.addMenu(tr("View"))
        m.addAction(self.a_results)
        m.addSeparator()
        m.addAction(self.a_fit)
        m.addAction(self.a_zin)
        m.addAction(self.a_zout)
        m = mb.addMenu(tr("Tools"))
        for key in self.a_tools:
            m.addAction(self.a_tools[key])
        m.addSeparator()
        m.addAction(self.a_phase)
        m.addAction(self.a_note)
        m = mb.addMenu(tr("Analysis"))
        m.addAction(self.a_detect)
        self.model_menu = m.addMenu(tr("Detection model"))
        grp = QActionGroup(self)
        cur = self.settings.value("model", "heavy")
        for key, label in (("heavy", "Accurate (heavy)"), ("full", "Balanced (full)"), ("lite", "Fast (lite)")):
            a = self.model_menu.addAction(tr(label))
            a.setCheckable(True)
            a.setChecked(key == cur)
            a.triggered.connect(lambda _=False, k=key: self.settings.setValue("model", k))
            grp.addAction(a)
        m.addAction(self.a_smooth)
        m.addAction(self.a_next_low)
        m.addSeparator()
        m.addAction(self.a_subject)
        self.template_menu = m.addMenu(icons.icon("template", theme.TEXT_2, 16), tr("Apply template"))
        m.addAction(self.a_new_tpl)
        m.addSeparator()
        m.addAction(self.a_multiview)
        m.addAction(self.a_compare)
        m.addAction(self.a_patients)
        self.plugins_menu = mb.addMenu(tr("Plugins"))
        m = mb.addMenu(tr("Help"))
        lang = m.addMenu(icons.icon("globe", theme.TEXT_2, 16), tr("Language"))
        grp = QActionGroup(self)
        for code, label in LANGUAGES.items():
            a = lang.addAction(label)
            a.setCheckable(True)
            a.setChecked(code == language())
            a.triggered.connect(lambda _=False, c=code: self.set_language(c))
            grp.addAction(a)
        m.addAction(self.a_shortcuts)
        m.addAction(self.a_models)
        m.addAction(self.a_about)
        self.export_menu.aboutToShow.connect(self._fill_export_menu)
        self.template_menu.aboutToShow.connect(self._fill_template_menu)
        self.plugins_menu.aboutToShow.connect(self._fill_plugins_menu)
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.topbar.export.clicked.connect(self._export_popup)
        self.topbar.detect.clicked.connect(self.redetect)
        self.topbar.undo.setDefaultAction(self.a_undo)
        self.topbar.redo.setDefaultAction(self.a_redo)
        self.topbar.undo.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.topbar.redo.setToolButtonStyle(Qt.ToolButtonIconOnly)
        mb.adjustSize()
        mb.setFixedWidth(mb.sizeHint().width() + 4)

    def _wire(self):
        c = self.ctl
        w = self.welcome
        w.openVideo.connect(lambda: self.open_video())
        w.openProject.connect(self.open_project_dialog)
        w.openPath.connect(self.open_path)
        w.templatePicked.connect(self._template_then_open)
        w.manageTemplates.connect(lambda: self.edit_template(None))
        w.navigate.connect(self._navigate)
        self.transport.markPhase.connect(lambda: mark_phase(self, c))
        self.transport.addNote.connect(self.add_note_center)
        self.transport.newTrack.connect(lambda kind: self.timeline._new_track(kind, None))
        self.canvas.message.connect(self.workspace.context.message)
        self.canvas.trackRequested.connect(self.track_marker)
        c.changed.connect(self._update_status)
        c.undo.cleanChanged.connect(self._on_clean_changed)
        c.changed.connect(self._update_zoom_label)
        self.topbar.tab_group.idClicked.connect(lambda i: self.show_results() if i == 1 else self.show_workspace())
        self.results.exportPdf.connect(self.export_pdf)
        self.results.exportCsv.connect(self.export_csv)
        self.results.exportKva.connect(self.export_kva)
        self.results.exportVideo.connect(self.export_video)
        zoom_timer = QTimer(self)
        zoom_timer.timeout.connect(self._update_zoom_label)
        zoom_timer.start(250)

    def _update_zoom_label(self, *_):
        self.workspace.context.zoom.setText(f"{self.canvas.zoom_percent}%")

    # ── pages ──────────────────────────────────────────────
    def show_welcome(self):
        self.welcome.set_recent(self.recent())
        self.welcome.reload_templates()
        self.stack.setCurrentWidget(self.welcome)
        self.topbar.set_project_mode(False)
        self._enable_actions(False)

    def show_workspace(self):
        self.stack.setCurrentWidget(self.workspace)
        self.topbar.set_project_mode(True)
        self.topbar.tab_group.button(0).setChecked(True)
        self._enable_actions(True)
        self.canvas.setFocus()
        self._update_status()

    def show_results(self):
        self.transport.stop()
        self.stack.setCurrentWidget(self.results)
        self.topbar.tab_group.button(1).setChecked(True)
        self.results.refresh()

    def toggle_results(self):
        if self.stack.currentWidget() is self.results:
            self.show_workspace()
        else:
            self.show_results()

    def _enable_actions(self, on: bool):
        for a in self.actions():
            if a.data() is True:
                a.setEnabled(on)
        self.export_menu.setEnabled(on)
        self.template_menu.setEnabled(on)

    def _on_clean_changed(self, *_):
        self._update_status()

    def _update_status(self, *_):
        p = self.ctl.project
        if not p:
            return
        self.topbar.title.setText(p.title)
        if self.processing and self.processing.isHidden() and self.processing.worker.isRunning():
            return
        if p.dirty:
            self.topbar.status.setText(f"<span style='color:{theme.WARN}'>●</span>&nbsp; {tr('Unsaved changes')}")
        else:
            self.topbar.status.setText(f"<span style='color:{theme.OK}'>●</span>&nbsp; {tr('Saved')}")
        self.setWindowTitle(f"{p.title}{' •' if p.dirty else ''} — {APP_NAME}")

    def set_tool(self, key):
        if self.stack.currentWidget() is not self.workspace:
            return
        self.canvas.cancel()
        self.ctl.set_tool(key)

    # ── recent files ───────────────────────────────────────
    def recent(self) -> list[str]:
        return list(self.settings.value("recent", [], type=list) or [])

    def add_recent(self, path):
        path = str(Path(path).resolve())
        r = [p for p in self.recent() if p != path]
        self.settings.setValue("recent", [path] + r[:14])

    def _fill_recent_menu(self):
        self.recent_menu.clear()
        for p in self.recent()[:10]:
            if Path(p).exists():
                self.recent_menu.addAction(Path(p).name, lambda p=p: self.open_path(p))
        if self.recent_menu.isEmpty():
            a = self.recent_menu.addAction(tr("Nothing yet"))
            a.setEnabled(False)

    # ── opening ────────────────────────────────────────────
    def confirm_discard(self) -> bool:
        p = self.ctl.project
        if not p or not p.dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setText(tr("Save the changes to “{name}”?", name=p.title))
        box.setInformativeText(tr("If you don't save, your corrections and measurements will be lost."))
        save = box.addButton(tr("Save"), QMessageBox.AcceptRole)
        box.addButton(tr("Don't save"), QMessageBox.DestructiveRole)
        cancel = box.addButton(tr("Cancel"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is cancel:
            return False
        if box.clickedButton() is save:
            return self.save()
        return True

    def _template_then_open(self, tid: str):
        self.pending_template = tid
        self.open_video()

    def open_video(self, path: str | None = None, template: str | None = None):
        if not self.confirm_discard():
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, tr("Open video"), self.settings.value("last_dir", ""), VIDEO_FILTER)
            if not path:
                self.pending_template = None
                return
        self.settings.setValue("last_dir", str(Path(path).parent))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            meta = probe(path)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, APP_NAME, tr("This video could not be opened:\n{err}", err=e))
            return
        QApplication.restoreOverrideCursor()
        if meta.frame_count < 2:
            QMessageBox.critical(self, APP_NAME, tr("The video has no readable frames."))
            return
        project = Project(meta)
        self.ctl.set_project(project)
        self.add_recent(path)
        self.show_workspace()
        tid = template or self.pending_template or self.settings.value("last_template", "kick")
        self.pending_template = None
        QTimer.singleShot(50, lambda: self.start_detection(tid))

    def open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open project or KVA"), self.settings.value("last_dir", ""),
                                              tr("OpenKick projects and Kinovea files") + " (*.okick *.kva *.ok3d)")
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        suffix = Path(path).suffix.lower()
        if suffix in VIDEO_EXT:
            self.open_video(path)
        elif suffix == ".okick":
            if not self.confirm_discard():
                return
            try:
                project = Project.load(path)
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, APP_NAME, tr("This project could not be opened:\n{err}", err=e))
                return
            if not Path(project.video.path).exists():
                vp = self._ask_video(tr("The project's video was not found. Where is it?"))
                if not vp:
                    return
                project.video.path = vp
            self.ctl.set_project(project)
            self.add_recent(path)
            self.show_workspace()
        elif suffix == ".kva":
            self.open_kva(path)
        elif suffix == ".ok3d":                      # a saved 3D session: it lives in the 3D window
            self.open_multiview()
            self.multiview_window.load_session(path)
        else:
            QMessageBox.information(self, APP_NAME, tr("OpenKick opens videos, .okick projects and .kva files."))

    def _ask_video(self, text: str) -> str | None:
        QMessageBox.information(self, APP_NAME, text)
        path, _ = QFileDialog.getOpenFileName(self, tr("Open video"), self.settings.value("last_dir", ""), VIDEO_FILTER)
        return path or None

    def open_kva(self, path: str):
        if not self.confirm_discard():
            return
        video = kva.find_video(path)
        if video is None:
            video = self._ask_video(tr("Select the video that goes with this .kva file."))
            if not video:
                return
        try:
            meta = probe(video)
            project = Project(meta)
            notes = kva.import_into(project, path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, tr("This KVA could not be imported:\n{err}", err=e))
            return
        project.saved_revision = project.revision
        self.ctl.set_project(project)
        self.add_recent(path)
        self.show_workspace()
        self._kva_summary(notes)
        if not project.pose.has_data:
            if QMessageBox.question(self, APP_NAME, tr("This KVA has no skeleton. Detect the pose with MediaPipe now?")) \
                    == QMessageBox.Yes:
                self.start_detection(None)

    def _kva_summary(self, notes):
        labels = {"skeleton": tr("skeleton in {n} frames"), "angles": tr("{n} angles"),
                  "trajectories": tr("{n} trajectories"), "phases": tr("{n} named keyframes as phases")}
        parts = []
        for n in notes:
            k, v = n.split(":")
            parts.append(labels[k].format(n=v))
        if parts:
            self.workspace.context.message(tr("Imported from Kinovea: {items}", items=", ".join(parts)))

    # ── detection ──────────────────────────────────────────
    def start_detection(self, template_id: str | None):
        p = self.ctl.project
        if not p:
            return
        if self.processing and self.processing.worker.isRunning():
            if self.processing.meta.path == p.video.path:
                self.processing.show()
                return
            self.stop_detection()               # it was for a video that is no longer open
        model = self.settings.value("model", "heavy")
        dlg = ProcessingDialog(p.video, template_id, model, self)
        dlg.finished_ok.connect(lambda res, tid, path=p.video.path: self._detected(res, tid, path))
        dlg.progressed.connect(self._bg_progress)
        dlg.finished.connect(lambda _: self._update_status())
        self.processing = dlg
        dlg.start()

    def _bg_progress(self, pct):
        if self.processing and self.processing.isHidden():
            self.topbar.status.setText(f"<span style='color:{theme.ACCENT}'>●</span>&nbsp; {tr('Detecting… {p}%', p=pct)}")

    def stop_detection(self):
        """Stop the detection running in the background: the video it was for is closed."""
        if self.processing and self.processing.worker.isRunning():
            self.processing.worker.stop()
            self.processing.worker.wait(3000)

    def _detected(self, res, template_id, path: str | None = None):
        p = self.ctl.project
        if p is None or (path is not None and path != p.video.path) or len(res.xy) != p.frame_count:
            return                              # its video was closed, or another one opened, meanwhile
        n = p.frame_count
        pose = Pose(n)
        pose.xy[:] = res.xy
        pose.raw_xy[:] = res.xy
        pose.visibility[:] = res.visibility
        pose.z[:] = res.z
        pose.source[:] = np.where(np.isfinite(res.xy[..., 0]), SRC_AUTO, 0)
        first = not p.pose.has_data
        if first:
            p.pose = pose
            self.ctl.undo.clear()
        else:
            manual = p.pose.source == SRC_MANUAL                      # keep hand corrections
            pose.xy[manual] = p.pose.xy[manual]
            pose.source[manual] = SRC_MANUAL
            self.ctl.replace_pose(pose, tr("Detect pose again"))
        t = templates.find(template_id)
        if t:
            templates.apply(p, t)
            self.settings.setValue("last_template", t.id)
            self.ctl.run_template_plugin()
        if first:
            p.touch()
            self.ctl.undo.clear()
        self.ctl.notify()
        self.ctl.projectLoaded.emit()
        self.ctl.frameChanged.emit(self.ctl.frame)
        if self.canvas.athlete_is_small():
            QTimer.singleShot(0, self.canvas.focus_athlete)
        missing = n - res.detected
        msg = tr("Skeleton detected in {d} of {n} frames", d=res.detected, n=n)
        if missing:
            msg += " · " + tr("check the frames without a person")
        self.workspace.context.message(msg)
        self._update_status()

    def redetect(self):
        if not self.ctl.project:
            return
        if self.ctl.project.pose.has_data:
            if QMessageBox.question(self, APP_NAME, tr("Run MediaPipe again on the whole video? "
                                                      "Your hand corrections are kept.")) != QMessageBox.Yes:
                return
        self.start_detection(self.ctl.project.template)

    def smooth(self):
        p = self.ctl.project
        if not p or not p.pose.has_data:
            return
        edit = A.smooth_pose(p.pose, p.video.fps)
        self.ctl.push_pose_edit(edit, tr("Smooth trajectories"))
        self.workspace.context.message(tr("Trajectories smoothed (Butterworth 6 Hz). Hand corrections were kept."))

    # ── saving ─────────────────────────────────────────────
    def save(self) -> bool:
        p = self.ctl.project
        if not p:
            return False
        if not p.path:
            return self.save_as()
        try:
            p.save(p.path)
        except OSError as e:
            QMessageBox.critical(self, APP_NAME, str(e))
            return False
        patients.note_analysis(p.subject, p.path, "2D")
        self.add_recent(p.path)
        self._update_status()
        return True

    def save_as(self) -> bool:
        p = self.ctl.project
        if not p:
            return False
        default = str(p.path or Path(p.video.path).with_suffix(".okick"))
        path, _ = QFileDialog.getSaveFileName(self, tr("Save project"), default, tr("OpenKick project") + " (*.okick)")
        if not path:
            return False
        if not path.lower().endswith(".okick"):
            path += ".okick"
        p.path = Path(path)
        return self.save()

    def _save_path(self, title, suffix, filt, default_name=None):
        p = self.ctl.project
        base = Path(p.path or p.video.path)
        default = str(base.with_name(default_name or base.stem + suffix))
        path, _ = QFileDialog.getSaveFileName(self, title, default, filt)
        if path and not path.lower().endswith(suffix.lower()):
            path += suffix
        return path

    # ── exports ────────────────────────────────────────────
    def _fill_export_menu(self):
        m = self.export_menu
        m.clear()
        for a in (self.a_exp_kva, self.a_exp_csv, self.a_exp_lm, self.a_exp_video, self.a_exp_pdf):
            m.addAction(a)
        plugins = [p for p in self._plugins() if p.kind == "export"]
        if plugins:
            m.addSeparator()
            for pl in plugins:
                m.addAction(icons.icon("plugin", theme.TEXT_2, 16), pl.label, lambda pl=pl: self.run_export_plugin(pl))

    def _export_popup(self):
        self._fill_export_menu()
        b = self.topbar.export
        self.export_menu.exec(b.mapToGlobal(b.rect().bottomLeft()))

    def export_kva(self):
        path = self._save_path(tr("Export to Kinovea"), ".kva", "Kinovea (*.kva)")
        if not path:
            return
        info = kva.export(self.ctl.project, path)
        msg = tr("KVA written: skeleton, {a} angles and {t} trajectories. Open the video in Kinovea and use "
                 "File › Load annotations.", a=info["angles"], t=info["trajectories"])
        if info["skipped"]:
            msg += "\n\n" + tr("Not exported (Kinovea format not supported yet): {items}", items=", ".join(info["skipped"]))
        QMessageBox.information(self, APP_NAME, msg)

    def export_csv(self):
        path = self._save_path(tr("Export measurements"), ".csv", "CSV (*.csv)",
                               Path(self.ctl.project.video.path).stem + "_mediciones.csv")
        if path:
            csvexport.export_measurements(self.ctl.project, path)
            self.workspace.context.message(tr("CSV written: {name}", name=Path(path).name))

    def export_landmarks(self):
        path = self._save_path(tr("Export landmarks"), ".csv", "CSV (*.csv)",
                               Path(self.ctl.project.video.path).stem + "_landmarks.csv")
        if path:
            csvexport.export_landmarks(self.ctl.project, path)
            self.workspace.context.message(tr("CSV written: {name}", name=Path(path).name))

    def export_pdf(self):
        path = self._save_path(tr("PDF report"), ".pdf", "PDF (*.pdf)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            write_pdf(self.ctl.project, self.results.chart, path)
        finally:
            QApplication.restoreOverrideCursor()
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def export_video(self):
        path = self._save_path(tr("Video with drawings"), ".mp4", "MP4 (*.mp4)",
                               Path(self.ctl.project.video.path).stem + "_openkick.mp4")
        if not path:
            return
        self.transport.stop()
        n = self.ctl.project.frame_count
        dlg = QProgressDialog(tr("Rendering the video with drawings…"), tr("Cancel"), 0, n, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        worker = VideoExportWorker(self.ctl.project, path, self.ctl.layers)
        worker.progress.connect(lambda i, _n: dlg.setValue(i))
        dlg.canceled.connect(worker.stop)

        def done(out):
            dlg.close()
            if out:
                self.workspace.context.message(tr("Video written: {name}", name=Path(out).name))
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(out).parent)))

        def failed(msg):
            dlg.close()
            QMessageBox.critical(self, APP_NAME, msg)
        worker.done.connect(done)
        worker.failed.connect(failed)
        self._video_worker = worker
        worker.start()

    def import_kva(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Import KVA"), self.settings.value("last_dir", ""), "Kinovea (*.kva)")
        if not path:
            return
        before = self.ctl.begin()
        edit_before = self.ctl.project.pose.xy.copy()
        notes = kva.import_into(self.ctl.project, path)
        self.ctl.commit(before, tr("Import KVA"))
        del edit_before
        self._kva_summary(notes)

    # ── editing ────────────────────────────────────────────
    def delete_selected(self):
        from openkick.ui.canvas import delete_measure
        mid = self.ctl.selected_measure
        if mid and self.ctl.project:
            delete_measure(self.ctl, self.ctl.project.measurement(mid))

    def revert_selected(self):
        from openkick.ui.canvas import revert_joint
        if self.ctl.selected_landmark is not None:
            revert_joint(self.ctl, self.ctl.selected_landmark)

    def add_note_center(self):
        from openkick.ui.canvas import new_note
        p = self.ctl.project
        if p:
            # top-left of the visible part of the video, whatever the zoom
            v = self.canvas.view()
            x, y = v.inv(max(v.ox, 0) + 24, max(v.oy, 0) + 56)
            new_note(self, self.ctl, (min(max(x, 0), p.video.width * 0.8), min(max(y, 0), p.video.height * 0.9)))

    def add_derived_marker(self, derived: str, name: str):
        from openkick.core.project import Measurement, PointRef
        p = self.ctl.project
        if not p:
            return
        m = Measurement(kind="point", name=name, color=p.next_color(), points=[PointRef(derived=derived)])
        self.ctl.change(tr("Add marker"), lambda: p.measurements.append(m))

    def add_midpoint_marker(self, a: str, b: str, name: str):
        from openkick.core import landmarks as L
        from openkick.core.project import Measurement, PointRef
        p = self.ctl.project
        if not p:
            return
        m = Measurement(kind="point", name=name, color=p.next_color(),
                        points=[PointRef(segment=(L.INDEX[a], L.INDEX[b], 0.5, 0.0))])
        self.ctl.change(tr("Add marker"), lambda: p.measurements.append(m))

    def track_marker(self, mid: str, direction: int):
        from PySide6.QtCore import QThread, Signal as _Sig
        from openkick.core.tracking import track_point
        p = self.ctl.project
        m = p.measurement(mid) if p else None
        if not m or m.points[0].kind != "free":
            return
        start = self.ctl.frame
        xy = m.points[0].free_at(start)
        if xy is None:
            return
        self.transport.stop()
        n = (p.frame_count - 1 - start) if direction > 0 else start
        if n <= 0:
            return
        dlg = QProgressDialog(tr("Tracking “{name}”…", name=m.name), tr("Cancel"), 0, n, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        stop = [False]
        dlg.canceled.connect(lambda: stop.__setitem__(0, True))

        class Worker(QThread):
            prog = _Sig(int)
            out = _Sig(object, object)

            def run(w):
                res, lost = track_point(p.video, start, xy, direction,
                                        on_progress=lambda i, _t: w.prog.emit(i),
                                        should_stop=lambda: stop[0])
                w.out.emit(res, lost)

        worker = Worker()
        worker.prog.connect(dlg.setValue)

        def done(res, lost):
            dlg.close()

            def apply():
                mm = p.measurement(mid)
                mm.points[0].keys.update(res)
            self.ctl.change(tr("Track marker"), apply)
            last = max(res) if direction > 0 else min(res)
            if lost is not None:
                self.ctl.set_frame(lost)
                self.workspace.context.message(tr("Tracking lost at frame {f}: place the marker there by hand "
                                                  "and track again", f=lost + 1))
            else:
                self.workspace.context.message(tr("Tracked {n} frames, up to frame {f}", n=len(res) - 1, f=last + 1))
        worker.out.connect(done)
        self._track_worker = worker
        worker.start()

    def workspace_next_low(self):
        self.workspace.right.measures.next_doubtful()

    # ── templates ──────────────────────────────────────────
    def _fill_template_menu(self):
        m = self.template_menu
        m.clear()
        for t in templates.list_templates():
            m.addAction(icons.icon("template", t.color, 16), t.label,
                        lambda t=t: self.ctl.change(tr("Apply template"),
                                                    lambda: (templates.apply(self.ctl.project, t), self.ctl.run_template_plugin())))
        m.addSeparator()
        cur = templates.find(self.ctl.project.template) if self.ctl.project else None
        if cur:
            m.addAction(tr("Edit “{name}”…", name=cur.label), lambda: self.edit_template(cur))

    def edit_template(self, base):
        dlg = TemplateEditor(base, can_apply=self.ctl.has_project, parent=self)
        dlg.applyRequested.connect(lambda t: self.ctl.change(tr("Apply template"),
                                                             lambda: (templates.apply(self.ctl.project, t), self.ctl.run_template_plugin())))
        dlg.exec()
        self.welcome.reload_templates()

    # ── plugins ────────────────────────────────────────────
    def _plugins(self):
        plugins, self._plugin_errors = discover()
        off = disabled_plugins()
        return [p for p in plugins if p.id not in off]

    def _fill_plugins_menu(self):
        m = self.plugins_menu
        m.clear()
        plugins = self._plugins()
        for pl in plugins:
            if pl.kind == "analysis":
                a = m.addAction(icons.icon("chart", theme.TEXT_2, 16), pl.label, lambda pl=pl: self.run_plugin(pl))
                a.setEnabled(self.ctl.has_project)
                a.setToolTip(pl.summary)
        exp = [p for p in plugins if p.kind == "export"]
        if exp:
            m.addSeparator()
            for pl in exp:
                a = m.addAction(icons.icon("export", theme.TEXT_2, 16), pl.label, lambda pl=pl: self.run_export_plugin(pl))
                a.setEnabled(self.ctl.has_project)
        m.addSeparator()
        m.addAction(self.a_plugins)
        m.addAction(self.a_plugin_dir)
        if self._plugin_errors:
            a = m.addAction(icons.icon("info", theme.WARN, 16), tr("{n} plugins failed to load", n=len(self._plugin_errors)))
            a.triggered.connect(self.manage_plugins)

    def _ctx(self):
        def ask(label, default):
            v, ok = QInputDialog.getDouble(self, APP_NAME, label, default, 0, 1e9, 2)
            return v if ok else None
        return PluginContext(self.ctl.project, ask)

    def run_plugin(self, pl):
        try:
            res = pl.run(self._ctx())
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, pl.label, tr("The plugin failed:\n{err}", err=e))
            return
        self.ctl.plugin_results[pl.id] = res
        PluginResultDialog(pl.label, res, self).exec()
        self.ctl.changed.emit()        # the plugin may have modified the project (e.g. auto-detected phases)

    def run_export_plugin(self, pl):
        path = self._save_path(pl.label, pl.suffix or "", pl.file_filter)
        if not path:
            return
        try:
            msg = pl.export(self._ctx(), Path(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, pl.label, tr("The plugin failed:\n{err}", err=e))
            return
        if msg:
            QMessageBox.information(self, pl.label, msg)

    def _navigate(self, key: str):
        """An entry of the start page's sidebar."""
        if key.startswith("language:"):
            self.set_language(key.split(":", 1)[1])
            return
        {"3d": self.open_multiview, "compare": self.open_compare, "patients": self.open_patients,
         "plugins": self.manage_plugins,
         "shortcuts": lambda: ShortcutsDialog(self).exec(),
         "about": lambda: AboutDialog(self).exec()}[key]()

    def edit_subject(self):
        """The patient's record of the open project."""
        p = self.ctl.project
        if not p:
            return
        from openkick.ui.subject import edit_subject
        new = edit_subject(self, p.subject, p.template)
        if new is not None:
            self.ctl.change(tr("Patient record"), lambda: setattr(self.ctl.project, "subject", new))

    def open_compare(self, paths=None):
        """Compare several analyses (2D projects or 3D sessions) side by side."""
        from openkick.ui.compare import CompareWindow
        if getattr(self, "compare_win", None) is None:
            self.compare_win = CompareWindow()
        self.compare_win.show()
        self.compare_win.raise_()
        self.compare_win.activateWindow()
        if paths:
            self.compare_win.add_files(paths)

    def open_patients(self):
        """The patients kept on this computer, their records and their analyses."""
        from openkick.ui.patients import PatientsWindow
        if getattr(self, "patients_win", None) is None:
            self.patients_win = PatientsWindow(open_path=self.open_path, compare=self.open_compare, parent=self)
        self.patients_win.reload()
        self.patients_win.show()
        self.patients_win.raise_()
        self.patients_win.activateWindow()

    def open_multiview(self):
        """The 3D tool lives in its own window (it handles several videos at once)."""
        from openkick.ui.multiview import MultiviewWindow
        self.transport.stop()
        if getattr(self, "multiview_window", None) is None:
            self.multiview_window = MultiviewWindow(self)
        self.multiview_window.show()
        self.multiview_window.raise_()
        self.multiview_window.activateWindow()

    def manage_plugins(self):
        from openkick.ui.store import StoreDialog
        dlg = StoreDialog(self)
        dlg.exec()
        if dlg.changed_any:                                  # what was installed or removed shows at once
            self._fill_plugins_menu()
            if hasattr(self, "welcome"):
                self.welcome.reload_templates()

    # ── misc ───────────────────────────────────────────────
    def set_language(self, code: str):
        self.settings.setValue("language", code)
        QMessageBox.information(self, APP_NAME, tr("The language will change the next time you open OpenKick."))

    def go_home(self):
        if not self.confirm_discard():
            return
        self.transport.stop()
        self.stop_detection()
        self.ctl.close_project()
        self.show_welcome()

    def closeEvent(self, e):
        if not self.confirm_discard():
            e.ignore()
            return
        self.stop_detection()
        e.accept()
