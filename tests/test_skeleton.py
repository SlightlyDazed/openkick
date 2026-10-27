"""Correcting points: the cameras stay where they are, and a point can go where its bones say."""
import numpy as np
import pytest

from openkick.multiview import geometry as G
from openkick.multiview.export import from_world, to_world
from openkick.multiview.session import SRC_FOLLOW, SRC_MANUAL, observe
from tests.test_multiview import make, make_many

I = G.IDX


@pytest.fixture(autouse=True)
def english():
    from openkick import i18n
    before = i18n.language()
    i18n.set_language("en")
    yield
    i18n.set_language(before)


def test_back_from_opensim_axes():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.3, (50, 26, 3)) + np.array([0.0, 0.0, 3.0])
    W, info = to_world(X)
    assert np.allclose(from_world(W, info), X, atol=1e-9)


def test_a_correction_leaves_the_cameras_and_a_point_can_go_where_its_bones_say():
    from tests.test_multiview import _app, _on_real_file
    from tests.test_review import wrong_knee
    app = _app()
    from openkick.ui.multiview import MultiviewWindow
    s, _ = make()
    _on_real_file(s)
    s.solve()
    w = MultiviewWindow()
    w.show()
    w.set_session(s)
    app.processEvents()
    cams = [(R.copy(), t.copy()) for R, t in s.cams]
    v, j, row = s.views[1], I["right_knee"], 42
    f = s.frame_of(1, row)
    w.ctl.push([s.move_point(1, f, j, tuple(v.xy[f, j] + np.array([3.0, 0.0])))], "move")
    assert all(np.allclose(R, R2) and np.allclose(t, t2) for (R, t), (R2, t2) in zip(cams, s.cams))   # cameras stay
    wrong_knee(s, shift=(0.0, 90.0))
    w.ctl.recompute(cameras=False)
    w.ctl.set_row(row)
    guide = [g for g in s.bone_guides(row) if g["joint"] == j]
    if guide:
        w.ctl.select((1, j))
        app.processEvents()
        assert w.panel.snap_btn.isEnabled()
        w.panel._snap()
        assert not [g for g in s.bone_guides(row) if g["joint"] == j]     # its bones have their length again
        w.ctl.undo.undo()
        assert [g for g in s.bone_guides(row) if g["joint"] == j]
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()


def _corrected(view=1, rows=range(40, 46), shift=(45.0, -30.0)):
    """A session whose right knee is wrong in one video for a few frames, and where it really is."""
    from tests.test_review import wrong_knee
    s, _ = make()
    s.solve()
    j = I["right_knee"]
    truth = [v.xy[:, j].copy() for v in s.views]
    X_true = s.X_raw.copy()
    frames = wrong_knee(s, view=view, rows=rows, shift=shift)
    return s, j, truth, X_true, frames


def test_the_other_video_follows_a_correction_and_barely_moves():
    s, j, truth, X_true, frames = _corrected()
    f = frames[2]
    row = s.row_of(1, f)
    assert s.err_cm[row, j] > 3                                      # the videos disagree there
    other = s.views[0].xy[:, j].copy()
    edit = s.move_point(1, f, j, tuple(truth[1][f]), scope="stretch")   # the user puts it where it is
    followed = s.follow(1, edit, j)
    s.update(recalibrate=False)
    assert followed and all(e.view == 0 for e in followed)
    g = s.frame_of(0, row)
    assert s.views[0].source[g, j] == SRC_FOLLOW and s.views[1].source[f, j] == SRC_MANUAL
    assert np.linalg.norm(s.views[0].xy[g, j] - other[g]) < 3.0       # it was right: it hardly moves
    assert np.linalg.norm(s.X_raw[row, j] - X_true[row, j]) < 0.015  # the 3D point is back where it was
    assert s.err_cm[row, j] < 1.0                                    # and the videos agree
    for e in reversed([edit] + followed):                          # one undo takes everything back
        e.apply(s, new=False)
    assert np.array_equal(s.views[0].xy[:, j], other, equal_nan=True)


def test_a_correction_the_other_video_did_not_see_keeps_the_hand_placed_point():
    """The user moves the point somewhere the other video does not agree with: the other video
    moves onto what the correction says, the hand-placed point stays exactly where it was put."""
    s, _ = make()
    s.solve()
    j, row = I["left_knee"], 30
    f = s.frame_of(0, row)                                            # the first video: its frames are the rows
    new = s.views[0].xy[f, j] + np.array([25.0, 0.0])
    edit = s.move_point(0, f, j, tuple(new))
    followed = s.follow(0, edit, j)
    s.update(recalibrate=False)
    assert followed and len({g for e in followed for g in range(e.start, e.end + 1)}) >= 2   # both frames around
    assert np.allclose(s.views[0].xy[f, j], new)                       # the user's point is untouched
    seen = s.project(0, row, s.X_raw[row, j])
    assert np.linalg.norm(seen - new) < 2.0                          # the 3D point is on its ray
    assert s.err_cm[row, j] < 1.0


def test_a_point_placed_by_hand_is_never_moved_by_following():
    s, j, truth, X_true, frames = _corrected()
    f = frames[2]
    row = s.row_of(1, f)
    g = s.frame_of(0, row)
    s.move_point(0, g, j, tuple(truth[0][g]))                       # the user had placed it in video 1
    s.update(recalibrate=False)
    placed = s.views[0].xy[g, j].copy()
    edit = s.move_point(1, f, j, tuple(truth[1][f]))
    s.follow(1, edit, j)
    s.update(recalibrate=False)
    assert np.array_equal(s.views[0].xy[g, j], placed) and s.views[0].source[g, j] == SRC_MANUAL


def test_following_over_a_stretch_with_cameras_at_other_speeds():
    s, _ = make_many(4)                                             # 30, 30, 30 and 60 fps
    s.solve()
    j = I["right_ankle"]
    rows = range(50, 56)
    frames = sorted({s.frame_of(2, r) for r in rows})
    v = s.views[2]
    truth = v.xy[frames, j].copy()
    for f in frames:
        v.xy[f, j] = v.raw_xy[f, j] = v.xy[f, j] + np.array([0.0, 40.0], np.float32)
    s.update(recalibrate=False)
    edits = []
    for f, p in zip(frames, truth):
        e = s.move_point(2, f, j, tuple(p))
        edits += [e] + s.follow(2, e, j)
    s.update(recalibrate=False)
    assert {e.view for e in edits} == {0, 1, 2, 3}
    assert np.nanmax(s.err_cm[50:56, j]) < 2.0
    assert (s.views[3].source[:, j] == SRC_FOLLOW).sum() >= len(frames)   # the 60 fps video, every frame in between


def test_the_cameras_are_not_placed_with_points_that_follow():
    s, j, truth, X_true, frames = _corrected()
    f = frames[2]
    edit = s.move_point(1, f, j, tuple(truth[1][f]))
    s.follow(1, edit, j)
    o = observe(s.views, s.offsets)
    assert o["follow"].any() and (o["w"][o["follow"]] == 0).all()
    s.update(recalibrate=False)
    assert s.quality()["followed"] >= 1


def test_window_the_other_video_follows_the_drag_and_one_undo_takes_both_back():
    from PySide6 import QtWidgets
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from tests.test_multiview import _app, _on_real_file
    app = _app()
    from openkick.ui.multiview import MultiviewWindow
    s, _ = make()
    _on_real_file(s)
    s.solve()
    w = MultiviewWindow()
    w.resize(1400, 860)
    w.show()
    w.set_session(s)
    app.processEvents()
    c = w.ctl
    assert c.follow and w.panel.follow.isChecked()
    c.set_row(12)
    cv = w.canvases[0]
    j = I["right_knee"]
    f = c.frame(0)
    other = s.views[1].xy[:, j].copy()
    p0 = cv.map(cv.v.xy[f, j])
    p1 = QPointF(p0.x() + 20, p0.y() + 8)

    def send(kind, pos, buttons=Qt.LeftButton):
        QtWidgets.QApplication.sendEvent(cv, QMouseEvent(kind, pos, cv.mapToGlobal(pos.toPoint()), Qt.LeftButton, buttons, Qt.NoModifier))

    send(QEvent.MouseButtonPress, p0)
    send(QEvent.MouseMove, p1)
    send(QEvent.MouseButtonRelease, p1, Qt.NoButton)
    app.processEvents()
    assert c.undo.count() == 1 and s.views[0].source[f, j] == SRC_MANUAL
    assert (s.views[1].source[:, j] == SRC_FOLLOW).any() and s.err_cm[c.row, j] < 1.0
    w.grab()                                                        # draws the dotted rings
    c.undo.undo()
    assert np.array_equal(s.views[1].xy[:, j], other, equal_nan=True) and not (s.views[1].source[:, j] == SRC_FOLLOW).any()
    w.panel.follow.setChecked(False)
    assert not c.follow
    w.panel.follow.setChecked(True)
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()


# ── learning a point's path over the pedal stroke ──────────────────────

def pedalling(slide_px=18.0):
    """A rider filmed from the side (video 1) and from behind (video 2), cameras known. In video 1 the
    detector slides the right hip forward when the right leg goes up, as in a real session."""
    from fractions import Fraction

    from openkick.core.video import VideoMeta
    from openkick.multiview.session import Session, View
    from tests.test_cycling3d import rider
    X, t, _ = rider(seconds=12.0, fps=30.0)
    C = np.array([3.2, -0.5, 3.0])                                   # behind: looking along the bike
    z = np.array([-1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    R2 = np.stack([np.cross(y, z), y, z])
    cams = [(np.eye(3), np.zeros(3)), (R2, -R2 @ C)]
    views, true = [], []
    for k, (R, tt) in enumerate(cams):
        K = G.guess_intrinsics(1280, 720)
        Xc = X @ R.T + tt
        xy = Xc[..., :2] / Xc[..., 2:3] * K[0, 0] + K[:2, 2]
        true.append(xy.copy())
        if k == 0:
            hip, knee = I["right_hip"], I["right_knee"]
            ky = X[:, knee, 1]
            up = (ky.max() - ky) / np.ptp(ky)                              # 1 with the right knee highest (y is down)
            xy[:, hip, 0] += slide_px * up
        meta = VideoMeta("synthetic.mp4", 1280, 720, 30.0, Fraction(1, 1000), [round(1000 * x) for x in t])
        views.append(View(meta, t, xy, np.full(xy.shape[:2], 0.9), K=K))
    s = Session(views, 1.75)
    s.offsets, s.cams, s.base = [0.0, 0.0], cams, 1
    s.update(recalibrate=False)
    return s, true


def test_a_path_learned_over_one_stroke_corrects_every_stroke():
    s, true = pedalling()
    j = I["right_hip"]
    info = s.learning(0, j)
    assert info is not None and info["missing"] == list(range(8))
    assert s.learning(0, I["head"]) is None                          # a point with no side: no path
    turns = info["turns"]
    first = int(np.nanmin(turns)) + 2
    teach = [int(np.nanargmin(np.abs(turns - (first + q)))) for q in np.arange(10) / 10 + 0.05]
    for f in teach[:6]:
        s.move_point(0, f, j, tuple(true[0][f, j]))
    assert s.learning(0, j)["missing"] and s.learn_path(0, j) is None   # not the whole stroke yet
    for f in teach[6:]:
        s.move_point(0, f, j, tuple(true[0][f, j]))
    assert not s.learning(0, j)["missing"]
    v = s.views[0]
    before = np.linalg.norm(v.xy[:, j] - true[0][:, j], axis=1)
    edit = s.learn_path(0, j)
    assert edit is not None and s.last_learn["hand"] == 10
    after = np.linalg.norm(v.xy[:, j] - true[0][:, j], axis=1)
    other = np.abs(turns - (first + 5)) < 0.5                         # a stroke it did not learn from
    assert np.max(before[other]) > 15 and np.max(after[other]) < 3
    assert all(v.source[f, j] == SRC_MANUAL for f in teach)          # the user's points stay theirs
    from openkick.multiview.session import SRC_LEARNED
    assert (v.source[:, j] == SRC_LEARNED).sum() > 300
    s.update(recalibrate=False)
    L = np.linalg.norm(s.X_raw[:, j] - s.X_raw[:, I["right_knee"]], axis=1)
    assert np.nanstd(L) / np.nanmedian(L) < 0.01                      # the thigh keeps its length again
    edit.apply(s, new=False)                                         # undo
    assert np.allclose(v.xy[other, j], v.raw_xy[other, j])


def test_the_guides_use_the_tape_when_the_record_has_it():
    s, _ = pedalling(slide_px=0.0)
    from openkick.ui.multiview import bone_text
    usual = s.bone_lengths("typical")[("right_hip", "right_knee")]
    assert not s.tape_bones()
    s.subject.lengths["thigh_right"] = 100 * usual * 1.2              # the tape says it is longer
    assert ("right_hip", "right_knee") in s.tape_bones()
    k = [i for i, b in enumerate(G.LEG_BONES) if b == ("right_hip", "right_knee")][0]
    assert abs(np.nanmedian(s.bone_deviation()[:, k]) + 1 / 6) < 0.02     # 1/1.2 - 1
    g = s.bone_guides(10)
    assert any(G.HALPE26[x["joint"]].startswith("right") for x in g)
    assert "tape" in bone_text([("right_hip", "right_knee", -0.17, usual, usual * 1.2)], s.tape_bones())


def test_window_learning_the_path_from_the_panel():
    from tests.test_multiview import _app, _on_real_file
    app = _app()
    from openkick.ui.multiview import MultiviewWindow
    s, true = pedalling()
    _on_real_file(s, truncate=False)
    s.update(recalibrate=False)
    w = MultiviewWindow()
    w.show()
    w.set_session(s)
    app.processEvents()
    c, panel = w.ctl, w.panel
    j = I["right_hip"]
    c.select((0, j))
    app.processEvents()
    assert panel.learn_btn.isVisible() and not panel.learn_btn.isEnabled() and "8" in panel.learn_info.text()
    turns = s.learning(0, j)["turns"]
    first = int(np.nanmin(turns)) + 2
    for q in np.arange(10) / 10 + 0.05:
        f = int(np.nanargmin(np.abs(turns - (first + q))))
        c.push([s.move_point(0, f, j, tuple(true[0][f, j]))], "move")
    c.select((0, j))
    app.processEvents()
    assert panel.learn_btn.isEnabled() and "all the way round" in panel.learn_info.text()
    count = c.undo.count()
    panel._learn()
    assert c.undo.count() == count + 1
    from openkick.multiview.session import SRC_LEARNED
    assert (s.views[0].source[:, j] == SRC_LEARNED).any()
    w.grab()
    c.undo.undo()
    assert not (s.views[0].source[:, j] == SRC_LEARNED).any()
    c.select((0, I["head"]))
    app.processEvents()
    assert not panel.learn_btn.isVisible()
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()
