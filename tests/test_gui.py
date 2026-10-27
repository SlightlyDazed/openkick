"""Headless end-to-end test of the main window (no MediaPipe needed)."""
import os

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    QCoreApplication.setOrganizationName("OpenKickTest")
    QCoreApplication.setApplicationName("OpenKickTest")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from openkick.ui import theme
    theme.load_fonts()
    app.setStyleSheet(theme.STYLE)
    return app


def _mouse(widget, kind, pos, button=Qt.LeftButton, buttons=Qt.LeftButton, mods=Qt.NoModifier):
    e = QMouseEvent(kind, pos, widget.mapToGlobal(pos.toPoint()), button, buttons, mods)
    QtWidgets.QApplication.sendEvent(widget, e)


def test_workspace_flow(qapp, project, tmp_path):
    from openkick.core import templates
    from openkick.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1440, 900)
    w.show()
    c = w.ctl
    c.set_project(project)
    w.show_workspace()
    c.change("tpl", lambda: templates.apply(project, templates.find("kick")))
    assert len(w.workspace.right.measures.cards) == 3
    c.set_frame(20)
    qapp.processEvents()

    # drag the right knee on the canvas -> manual correction, undoable
    v = w.canvas.view()
    start = v.map(*project.pose.xy[20, 26])
    _mouse(w.canvas, QEvent.MouseButtonPress, start)
    _mouse(w.canvas, QEvent.MouseMove, start + QPointF(12, 8), Qt.NoButton)
    _mouse(w.canvas, QEvent.MouseButtonRelease, start + QPointF(12, 8), Qt.LeftButton, Qt.NoButton)
    assert list(c.manual_frames()) == [20]
    c.undo.undo()
    assert list(c.manual_frames()) == []
    c.undo.redo()

    # angle tool on the left elbow creates its preset angle
    c.set_tool("angle")
    el = w.canvas.view().map(*project.pose.xy[20, 13])
    _mouse(w.canvas, QEvent.MouseButtonPress, el)
    assert any(m.kind == "angle" and m.points[1].landmark == 13 for m in project.measurements)
    c.set_tool("select")

    for f, name in ((0, "A"), (30, "B"), (50, "C")):
        c.change("phase", lambda f=f, name=name: project.add_phase_at(f, name))
    w.show_results()
    qapp.processEvents()
    assert w.results.table.rowCount() == 4
    from openkick.ui.exporters import write_pdf
    write_pdf(project, w.results.chart, str(tmp_path / "r.pdf"))
    assert (tmp_path / "r.pdf").stat().st_size > 10_000
    project.saved_revision = project.revision
    w.close()
    w.deleteLater()
    qapp.processEvents()


def test_markers_and_note_dialog(qapp, project):
    from openkick.core import landmarks as L
    from openkick.core.project import TextNote
    from openkick.ui.dialogs import NoteDialog
    from openkick.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1440, 900)
    w.show()
    c = w.ctl
    c.set_project(project)
    w.show_workspace()
    qapp.processEvents()
    c.set_frame(10)
    c.set_tool("marker")
    v = w.canvas.view()
    hip, knee = project.pose.xy[10, L.INDEX["right_hip"]], project.pose.xy[10, L.INDEX["right_knee"]]
    on_thigh = v.map(*((hip + knee) / 2))
    _mouse(w.canvas, QEvent.MouseButtonPress, on_thigh)
    far = v.map(5.0, 5.0)
    _mouse(w.canvas, QEvent.MouseButtonPress, far)
    kinds = [m.points[0].kind for m in project.measurements if m.kind == "point"]
    assert kinds == ["segment", "free"]
    # a marker can be the vertex of an angle
    c.set_tool("angle")
    marker = project.measurements[0]
    mpos = v.map(*project.point(marker.points[0], 10))
    ank = v.map(*project.pose.xy[10, L.INDEX["right_ankle"]])
    _mouse(w.canvas, QEvent.MouseButtonPress, v.map(*hip), mods=Qt.ShiftModifier)   # Shift: custom angle
    _mouse(w.canvas, QEvent.MouseButtonPress, mpos)
    _mouse(w.canvas, QEvent.MouseButtonPress, ank)
    angle = project.measurements[-1]
    assert angle.kind == "angle" and angle.points[1].marker == marker.id
    # note dialog returns frames and fades
    note = TextNote("hola", 10, 10, start=5, end=20, fade_out=0.5)
    dlg = NoteDialog(project.video, note, 30)
    dlg.in_mode.setCurrentIndex(1)
    dlg.in_secs.setValue(0.3)
    vals = dlg.values()
    assert vals["start"] == 5 and vals["end"] == 20 and vals["fade_in"] == 0.3 and vals["fade_out"] == 0.5
    project.saved_revision = project.revision
    w.close()
    w.deleteLater()
    qapp.processEvents()


def test_shadow_of_neighbouring_frames(qapp, project):
    """The faint skeletons of the previous and next frame: shown while correcting, hidden while
    playing, with another tool, or when switched off."""
    from PySide6.QtGui import QColor

    from openkick.ui import overlay
    from openkick.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1440, 900)
    w.show()
    c = w.ctl
    c.set_project(project)
    w.show_workspace()
    act = w.workspace.context.ghost_act
    if not act.isChecked():
        act.toggle()
    c.set_frame(20)
    qapp.processEvents()

    def colors(img):
        seen = set()
        for x in range(0, img.width(), 2):
            for y in range(0, img.height(), 2):
                seen.add(img.pixelColor(x, y).name())
        return seen

    def has_shadows():
        seen = colors(w.canvas.grab().toImage())
        near = lambda hx: any(abs(QColor(s).red() - QColor(hx).red()) + abs(QColor(s).green() - QColor(hx).green())  # noqa: E731
                              + abs(QColor(s).blue() - QColor(hx).blue()) < 60 for s in seen)
        return near(overlay.GHOST_PREV), near(overlay.GHOST_NEXT)

    assert w.canvas.ghost_shown and has_shadows() == (True, True)
    c.set_frame(0)                                     # no previous frame at the start
    assert has_shadows() == (False, True)
    c.set_frame(project.frame_count - 1)               # no next frame at the end
    assert has_shadows() == (True, False)
    c.set_frame(20)
    c.set_tool("angle")                                # only while correcting
    assert not w.canvas.ghost_shown
    c.set_tool("select")
    c.playingChanged.emit(True)                        # not while the video plays
    assert not w.canvas.ghost_shown
    c.playingChanged.emit(False)
    act.toggle()                                       # Layers > shadow off
    assert not w.canvas.ghost and has_shadows() == (False, False)
    act.toggle()
    assert w.canvas.ghost_shown
    project.saved_revision = project.revision
    w.close()
    w.deleteLater()


def test_start_page_sidebar(qapp):
    """The sidebar of the start page reaches every part of the program, and shrinks to icons
    when the window is narrow."""
    from openkick.ui.main_window import MainWindow
    from openkick.ui.welcome import NAV, SideBar
    w = MainWindow()
    w.resize(1500, 900)
    w.show()
    qapp.processEvents()
    bar = w.welcome.sidebar
    keys = [key for _, entries in NAV for key, *_ in entries]
    assert list(bar.items) == keys and {"video", "3d", "project", "templates", "plugins"} <= set(keys)
    assert not bar.compact and bar.width() == SideBar.WIDTH
    calls = []
    w.open_multiview = lambda: calls.append("3d")
    w.manage_plugins = lambda: calls.append("plugins")
    w.set_language = lambda code: calls.append("language:" + code)
    w.welcome.openVideo.disconnect()
    w.welcome.openVideo.connect(lambda: calls.append("video"))
    w.welcome.openProject.disconnect()
    w.welcome.openProject.connect(lambda: calls.append("project"))
    w.welcome.manageTemplates.disconnect()
    w.welcome.manageTemplates.connect(lambda: calls.append("templates"))
    for key in ("video", "3d", "project", "templates", "plugins"):
        bar.items[key].click()
    w.welcome.navigate.emit("language:en")
    assert calls == ["video", "3d", "project", "templates", "plugins", "language:en"]
    w.resize(1100, 700)                                # narrow: icons only, the words go to the tooltip
    qapp.processEvents()
    assert bar.compact and bar.width() == SideBar.COMPACT and not bar.items["3d"].text.isVisible()
    assert bar.items["3d"].title in bar.items["3d"].toolTip()
    w.grab()
    w.resize(1500, 900)
    qapp.processEvents()
    assert not bar.compact and bar.items["3d"].text.isVisible()
    w.close()
    w.deleteLater()


def test_hovering_the_end_of_a_phase_that_lasts_to_the_last_frame(qapp, project):
    """A phase that runs to the end of the video: its end lies one frame past the last one."""
    from PySide6.QtCore import QPointF
    from openkick.ui.main_window import MainWindow
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w.ctl.set_project(project)
    w.show_workspace()
    project.add_phase_at(10, "A")
    assert project.phases[-1].end == project.frame_count - 1
    t = w.timeline
    t.resize(1200, t.height())
    x = t.x_of(project.frame_count)
    assert x == t.track_rect().right()
    y = t.rows()["phases"].center().y()
    for dx in (-3, 0, 3):                                 # used to raise IndexError
        t._boundary_at(QPointF(x + dx, y))
        _mouse(t, QEvent.MouseMove, QPointF(x + dx, y), Qt.NoButton, Qt.NoButton)
    project.saved_revision = project.revision
    w.hide()


@pytest.mark.skipif(__import__("sys").version_info >= (3, 12), reason="None cannot be freed from Python 3.12 on")
def test_painting_keeps_none_alive(qapp):
    """Some PySide6 releases (6.12.0) take a reference from None on every drawRect or fillRect:
    on Python 3.10 and 3.11 the app then aborts ("deallocating None") after a few thousand, which
    the 3D timeline reaches in seconds while playing."""
    import sys

    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    img = QImage(64, 64, QImage.Format_ARGB32)
    p = QPainter(img)
    before = sys.getrefcount(None)
    for _ in range(500):
        p.drawRect(QRectF(1, 2, 3, 4))
        p.fillRect(QRectF(1, 2, 3, 4), QColor(1, 2, 3))
    after = sys.getrefcount(None)
    p.end()
    assert after - before > -100, "this PySide6 release frees None by mistake: install another one"


def test_detection_result_for_a_video_no_longer_open_is_dropped(qapp, project):
    """Detection goes on in the background: by the time it ends, its video may have been closed or
    another one opened. Its result used to crash (no project) or land on the wrong video."""
    import numpy as np

    from openkick.core.pose import PoseResult
    from openkick.ui.main_window import MainWindow
    w = MainWindow()
    w.ctl.set_project(project)
    n = project.frame_count
    res = PoseResult(xy=np.full((n, 33, 2), 100.0, np.float32), visibility=np.ones((n, 33), np.float32),
                     z=np.zeros((n, 33), np.float32), detected=n)
    w._detected(res, "gait", "/elsewhere/another_video.mp4")        # another video was opened meanwhile
    assert not project.pose.has_data or not np.allclose(project.pose.xy, 100.0)
    project.saved_revision = project.revision
    w.go_home()
    w._detected(res, "gait", project.video.path)                     # its project was closed meanwhile
    assert w.ctl.project is None
    w.close()
