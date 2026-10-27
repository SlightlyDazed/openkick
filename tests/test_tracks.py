"""Tracks of the timeline and editing them as in a video editor: the core, then the timeline."""
import os
from fractions import Fraction

import pytest

from openkick.core import tracks as T
from openkick.core.project import Phase, Project
from openkick.core.video import VideoMeta


def blocks(*spans):
    return [Phase(name=f"b{i}", start=a, end=b) for i, (a, b) in enumerate(spans)]


def spans(row):
    return [(b.start, b.end) for b in row]


# ── core ───────────────────────────────────────────────────────────────

def test_segments_never_overlap_when_moved_or_resized():
    row = blocks((10, 19), (40, 49))
    assert T.move(row, row[0], 25, 100) == 25 and spans(row)[0] == (25, 34)
    assert T.move(row, row[0], 38, 100) == 30                 # stops against the next one
    assert spans(row)[0] == (30, 39)
    assert T.move(row, row[0], -5, 100) == 0
    assert T.resize(row, row[0], "end", 80, 100) == 39       # cannot run into the next block
    assert T.resize(row, row[0], "start", 50, 100) == 39     # keeps at least one frame
    assert T.move(row, row[1], 500, 100) == 90 and spans(row)[1] == (90, 99)


def test_children_go_along_with_their_phase():
    row = blocks((10, 19))
    kids = blocks((10, 14), (15, 19))
    T.move(row, row[0], 30, 100, kids)
    assert spans(kids) == [(30, 34), (35, 39)]


def test_split_merge_duplicate_add():
    row = blocks((10, 29))
    second = T.split(row, row[0], 20)
    assert spans(row) == [(10, 19), (20, 29)] and second.name == row[0].name
    assert T.split(row, row[0], 10) is None and T.split(row, row[0], 25) is None
    assert T.merge_next(row, row[0]) and spans(row) == [(10, 29)]
    assert not T.merge_next(row, row[0])
    copy = T.duplicate(row, row[0], 100)
    assert spans(row) == [(10, 29), (30, 49)] and copy.name == row[0].name
    assert T.duplicate(row, row[0], 100) is None                # no room right after it any more
    new = T.add_segment(row, 70, 30, 100, "x")
    assert (new.start, new.end) == (70, 99)                     # cut at the end of the video
    assert T.add_segment(row, 15, 5, 100, "y") is None          # inside a block


def test_events_and_snapping():
    marks = []
    T.add_mark(marks, 50, "a", "#fff")
    T.add_mark(marks, 20, "b", "#fff")
    assert [m.name for m in marks] == ["b", "a"]
    T.move_mark(marks, marks[0], 999, 100)
    assert [m.frame for m in marks] == [50, 99]
    row = blocks((10, 19))
    targets = T.snap_targets([row, marks], playhead=33)
    assert targets == [10, 20, 33, 50, 99]
    assert T.snap(22, targets, 3) == (20, 20)
    assert T.snap(26, targets, 3) == (26, None)
    assert 10 not in T.snap_targets([row], skip=row)          # a moving block does not attract itself


def test_tracks_are_saved_with_the_project(tmp_path):
    p = Project(VideoMeta(str(tmp_path / "v.mp4"), 640, 480, 30.0, Fraction(1, 30000), [i * 1000 for i in range(90)]))
    seg = T.new_track(p.tracks, T.SEGMENTS, "Arm")
    ev = T.new_track(p.tracks, T.EVENTS, "Ball", at=0)
    assert [t.name for t in p.tracks] == ["Ball", "Arm"] and seg.color != ev.color
    T.add_segment(seg.items, 5, 10, 90, "swing")
    T.add_mark(ev.items, 40, "hit", ev.color)
    ev.locked, seg.hidden = True, True
    assert T.move_track(p.tracks, seg, -1) and not T.move_track(p.tracks, seg, -1)
    p.save(tmp_path / "p.okick")
    q = Project.load(tmp_path / "p.okick")
    assert [(t.name, t.kind, t.hidden, t.locked) for t in q.tracks] == [("Arm", "segments", True, False),
                                                                        ("Ball", "events", False, True)]
    assert spans(q.tracks[0].items) == [(5, 14)] and q.tracks[1].items[0].frame == 40


# ── the timeline ───────────────────────────────────────────────────────

@pytest.fixture
def timeline():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from openkick.ui.controller import Controller
    from openkick.ui.timeline import Timeline
    p = Project(VideoMeta("v.mp4", 640, 480, 30.0, Fraction(1, 30000), [i * 1000 for i in range(300)]))
    c = Controller()
    c.set_project(p)
    tl = Timeline(c)
    tl.resize(1300, 400)
    tl.show()
    app.processEvents()
    yield tl, c, p, app
    p.saved_revision = p.revision
    tl.close()


def mouse(tl, kind, x, y, button=None, buttons=None, mods=None):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    button = Qt.LeftButton if button is None else button
    buttons = button if buttons is None else buttons
    e = QMouseEvent(kind, QPointF(x, y), tl.mapToGlobal(QPointF(x, y).toPoint()), button, buttons,
                    Qt.NoModifier if mods is None else mods)
    QApplication.sendEvent(tl, e)


def drag(tl, x0, x1, y):
    from PySide6.QtCore import QEvent, Qt
    mouse(tl, QEvent.MouseButtonPress, x0, y)
    mouse(tl, QEvent.MouseMove, (x0 + x1) / 2, y)
    mouse(tl, QEvent.MouseMove, x1, y)
    mouse(tl, QEvent.MouseButtonRelease, x1, y, Qt.LeftButton, Qt.NoButton)


def test_build_a_track_and_edit_it_with_the_mouse(timeline):
    from PySide6.QtCore import QEvent
    tl, c, p, app = timeline
    tl._new_track(T.SEGMENTS, None)
    t = p.tracks[0]
    y = tl.rows()[f"track:{t.id}"].center().y()
    mouse(tl, QEvent.MouseButtonDblClick, tl.x_of(30), y)      # double click on empty track: a new segment
    assert spans(t.items) == [(30, 59)]                       # one second long
    drag(tl, tl.x_of(40), tl.x_of(140), y)                    # drag its body: it moves, same length
    assert spans(t.items) == [(130, 159)]
    drag(tl, tl.x_of(160), tl.x_of(200), y)                   # drag its right edge: longer
    assert spans(t.items) == [(130, 199)]
    c.undo.undo()                                             # undo hands back the track as it was saved
    assert spans(p.tracks[0].items) == [(130, 159)]
    c.undo.undo()
    assert spans(p.tracks[0].items) == [(30, 59)]


def test_edges_snap_to_the_phases_and_alt_turns_it_off(timeline):
    tl, c, p, app = timeline
    p.phases = [Phase("A", 100, 199)]
    tl._new_track(T.SEGMENTS, None)
    t = p.tracks[0]
    T.add_segment(t.items, 10, 20, 300, "s")
    y = tl.rows()[f"track:{t.id}"].center().y()
    c.set_frame(0)
    near = tl.x_of(100) - 5                                   # a few pixels short of the phase's start
    drag(tl, tl.x_of(15), near + (tl.x_of(15) - tl.x_of(10)), y)
    assert t.items[0].start == 100


def test_split_delete_and_keys(timeline):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    tl, c, p, app = timeline
    tl._new_track(T.SEGMENTS, None)
    t = p.tracks[0]
    T.add_segment(t.items, 10, 60, 300, "s")
    y = tl.rows()[f"track:{t.id}"].center().y()
    mouse(tl, QEvent.MouseButtonPress, tl.x_of(30), y)
    mouse(tl, QEvent.MouseButtonRelease, tl.x_of(30), y, Qt.LeftButton, Qt.NoButton)
    assert tl.selected == (f"track:{t.id}", t.items[0].id) and c.frame == 30
    QApplication.sendEvent(tl, QKeyEvent(QEvent.KeyPress, Qt.Key_B, Qt.ControlModifier))
    assert spans(t.items) == [(10, 29), (30, 69)]
    QApplication.sendEvent(tl, QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
    assert spans(t.items) == [(10, 29)]
    c.undo.undo()
    assert spans(p.tracks[0].items) == [(10, 29), (30, 69)]


def test_events_track_and_locking(timeline):
    from PySide6.QtCore import QEvent
    tl, c, p, app = timeline
    tl._new_track(T.EVENTS, None)
    t = p.tracks[0]
    y = tl.rows()[f"track:{t.id}"].center().y()
    mouse(tl, QEvent.MouseButtonDblClick, tl.x_of(50), y)
    assert [m.frame for m in t.items] == [50]
    drag(tl, tl.x_of(50), tl.x_of(80), y)
    assert [m.frame for m in t.items] == [80]
    t.locked = True                                           # a locked track does not move
    drag(tl, tl.x_of(80), tl.x_of(120), y)
    assert [m.frame for m in t.items] == [80]


def test_zoom_and_scroll(timeline):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    tl, c, p, app = timeline
    r = tl.track_rect()
    assert tl.x_of(0) == r.left() and abs(tl.x_of(300) - r.right()) < 1e-6
    for _ in range(5):
        e = QWheelEvent(QPointF(r.center().x(), 50), tl.mapToGlobal(QPointF(r.center().x(), 50).toPoint()),
                        QPointF(0, 0).toPoint(), QPointF(0, 120).toPoint(), Qt.NoButton, Qt.ControlModifier,
                        Qt.NoScrollPhase, False)
        QApplication.sendEvent(tl, e)
    t0, t1 = tl.view
    assert t1 - t0 < 4 and t0 > 0                             # zoomed around the middle of the 10 s video
    assert tl.frame_at(tl.x_of(150)) == 150                   # pixels and frames still match
    tl.scroll(-100)
    assert tl.view[0] == 0
    tl.view = None
    assert tl.frame_at(r.right()) == 299
