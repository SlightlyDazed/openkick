"""Going through the doubtful stretches of a 3D session, and keeping the leg bones' length."""
import numpy as np

from openkick.multiview import geometry as G
from openkick.multiview.session import SRC_MANUAL, SRC_SPREAD, Session
from tests.test_multiview import figure, knee_angle, make

I = G.IDX


def wrong_knee(s, view=1, rows=range(40, 46), shift=(45.0, -30.0)):
    """The detector misses the right knee of one video for a few frames."""
    v = s.views[view]
    frames = sorted({s.frame_of(view, r) for r in rows})
    for f in frames:
        v.xy[f, I["right_knee"]] = v.raw_xy[f, I["right_knee"]] = v.xy[f, I["right_knee"]] + np.array(shift, np.float32)
    s.update()
    return frames


def test_doubtful_stretches_and_marking_them_fine(tmp_path):
    s, _ = make()
    s.solve()
    wrong_knee(s)
    st = [e for e in s.stretches() if e["joint"] == I["right_knee"]]
    assert len(st) == 1
    e = st[0]
    assert e["first"] <= 40 and e["last"] >= 45 and e["first"] <= e["worst"] <= e["last"] and e["cm"] > 6
    assert e["why"] in ("disagree", "bone", "jump") and not e["reviewed"]
    assert s.stretch_at(42, I["right_knee"]) == (e["first"], e["last"]) and s.stretch_at(42, I["left_knee"]) is None
    before = s.quality()["problems"]
    s.set_reviewed(e["joint"], e["first"], e["last"])
    assert s.quality()["problems"] < before
    assert all(j != I["right_knee"] or not e["first"] <= r <= e["last"] for r, j, _ in s.problems())
    assert [x for x in s.stretches() if x["joint"] == I["right_knee"]][0]["reviewed"]
    s.save(tmp_path / "s.ok3d")
    t = Session.load(tmp_path / "s.ok3d", require_videos=False)
    assert t.reviewed == s.reviewed and [x for x in t.stretches() if x["joint"] == I["right_knee"]][0]["reviewed"]
    t.set_reviewed(e["joint"], e["first"], e["last"], on=False)
    assert not t.reviewed


def test_a_correction_reaches_the_whole_stretch():
    s, _ = make()
    s.solve()
    truth = np.array([knee_angle(figure(s.t[r])[None])[0] for r in range(40, 46)])
    clean = np.abs(knee_angle(s.X_raw)[40:46] - truth)
    frames = wrong_knee(s)
    v, j = s.views[1], I["right_knee"]
    mid = frames[len(frames) // 2]
    good = v.xy[mid, j] - np.array([45.0, -30.0], np.float32)
    edit = s.move_point(1, mid, j, tuple(good), scope="stretch")
    s.update()
    assert v.source[mid, j] == SRC_MANUAL
    assert all(v.source[f, j] == SRC_SPREAD for f in frames if f != mid)         # every frame of the stretch
    assert all(np.allclose(v.xy[f, j], v.raw_xy[f, j] - np.array([45.0, -30.0]), atol=0.5) for f in frames)
    # its edges, a frame either side, are within the stretch too: they move with it, and, if that
    # was wrong for them, the list shows them again
    assert np.abs(knee_angle(s.X_raw)[40:46] - truth).max() < clean.max() + 0.5     # as good as before the error
    edit.apply(s, new=False)                                                     # undo: all of it comes back
    assert all(v.source[f, j] != SRC_MANUAL for f in frames) and np.allclose(v.xy[mid, j], v.raw_xy[mid, j])


def test_review_window():
    from tests.test_multiview import _app, _on_real_file
    app = _app()
    from openkick.ui.multiview import MultiviewWindow
    s, _ = make()
    _on_real_file(s)
    s.solve()
    w = MultiviewWindow()
    w.show()
    w.set_session(s)
    frames = wrong_knee(s)
    w.ctl.recompute(cameras=False)
    w.panel.show_review()
    app.processEvents()
    r = w.panel.review_win
    knee = [k for k, e in enumerate(r._items) if e["joint"] == I["right_knee"]]
    assert r.table.rowCount() == len(r._items) >= 1 and knee
    r.table.selectRow(knee[0])
    r.go()
    e = r._items[knee[0]]
    assert w.ctl.row == e["worst"] and w.ctl.selected == (e["view"], I["right_knee"])
    assert not r.picture.pixmap().isNull()
    r.toggle_fine()
    app.processEvents()
    assert all(x["joint"] != I["right_knee"] for x in r._items) and w.ctl.dirty      # marked fine: off the list
    r.show_done.setChecked(True)
    assert any(x["joint"] == I["right_knee"] and x["reviewed"] for x in r._items)
    assert frames
    r.close()
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()


def test_rigid_feet_turn_but_keep_their_shape(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    t = np.arange(n) / 60
    X = np.full((n, 26, 3), np.nan)
    shape = {"heel": np.array([-0.06, 0.05, 0.0]), "toe": np.array([0.19, 0.06, 0.02]),
             "small_toe": np.array([0.16, 0.06, 0.07])}
    pitch = np.radians(25) * np.sin(2 * np.pi * 1.2 * t)                          # the ankle working
    c, s_ = np.cos(pitch), np.sin(pitch)
    R = np.zeros((n, 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s_, s_, c, 1.0
    for side in ("left", "right"):
        ankle = np.stack([0.2 * np.cos(2 * np.pi * 1.2 * t), 0.2 * np.sin(2 * np.pi * 1.2 * t), np.zeros(n)], 1)
        X[:, I[f"{side}_ankle"]] = ankle
        X[:, I[f"{side}_knee"]] = ankle + [0.05, -0.4, 0]
        for p, v in shape.items():
            X[:, I[f"{side}_{p}"]] = ankle + np.einsum("nij,j->ni", R, v)
    truth = X.copy()
    noisy = X.copy()
    noisy[:, I["right_toe"]] += rng.normal(0, 0.015, (n, 3))                     # the toe badly seen
    noisy[:, I["right_small_toe"]] += rng.normal(0, 0.02, (n, 3))
    Y = G.rigid_feet(noisy, np.ones((n, 26)))
    assert np.array_equal(Y[:, I["right_ankle"]], noisy[:, I["right_ankle"]])    # the leg is untouched
    L = np.linalg.norm(Y[:, I["right_heel"]] - Y[:, I["right_toe"]], axis=1)
    assert np.ptp(L) < 1e-6                                                       # one shape
    def pitch_of(Z):
        v = Z[:, I["right_toe"]] - Z[:, I["right_heel"]]
        return np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    err_free = np.abs(pitch_of(noisy) - pitch_of(truth))
    err_rigid = np.abs(pitch_of(Y) - pitch_of(truth))
    assert err_rigid.mean() < err_free.mean() and np.ptp(pitch_of(Y)) > 45        # it still turns 50 degrees
    assert np.abs(G.rigid_feet(truth, np.ones((n, 26))) - truth)[np.isfinite(truth)].max() < 1e-6


def test_rigid_feet_in_the_session(tmp_path):
    s, _ = make()
    s.solve()
    foot = [I["right_toe"], I["right_small_toe"], I["right_heel"]]
    s.feet = "rigid"
    s.update(recalibrate=False)
    L = np.linalg.norm(s.X[:, I["right_heel"]] - s.X[:, I["right_toe"]], axis=1)
    assert np.nanstd(L) < np.nanstd(np.linalg.norm(s.X_raw[:, I["right_heel"]] - s.X_raw[:, I["right_toe"]], axis=1))
    assert np.isnan(s.bone_cm[:, foot]).all()                                    # a foot's shape is not flagged any more
    s.save(tmp_path / "s.ok3d")
    assert Session.load(tmp_path / "s.ok3d", require_videos=False).feet == "rigid"


def test_where_a_point_should_be_for_its_bones():
    from openkick.multiview.session import _two_lengths
    hip, ankle = np.array([0.0, 0.0, 0.0]), np.array([0.0, -0.7, 0.0])
    knee = _two_lengths(hip, ankle, 0.42, 0.40, near=np.array([0.3, -0.3, 0.0]))
    assert abs(np.linalg.norm(knee - hip) - 0.42) < 1e-9 and abs(np.linalg.norm(knee - ankle) - 0.40) < 1e-9
    assert knee[0] > 0 and abs(knee[2]) < 1e-9                             # on the side it was
    s, _ = make()
    s.solve()
    row, knee_j = 40, I["right_knee"]
    s.X_raw[row, knee_j] += [0.0, 0.0, 0.0]
    hip_j, ankle_j = I["right_hip"], I["right_ankle"]
    usual = s.bone_lengths("typical")[("right_hip", "right_knee")]
    # the hip 20 % too far from the knee, in that frame only
    v = s.X_raw[row, hip_j] - s.X_raw[row, knee_j]
    s.X_raw[row, hip_j] = s.X_raw[row, knee_j] + v / np.linalg.norm(v) * usual * 1.2
    s.jump_cm[row, hip_j] = 30.0                                              # and it jumped: the one to move
    s._deviation_key = None
    assert abs(s.bone_deviation()[row, 2] - 0.2) < 0.01
    g = [x for x in s.bone_guides(row) if x["joint"] == hip_j]
    assert g and g[0]["long"] and abs(np.linalg.norm(g[0]["target"] - s.X_raw[row, knee_j]) - usual) < 1e-6
    assert not [x for x in s.bone_guides(row) if x["joint"] == hip_j and not x["long"]]
    assert all(abs(d) * 100 >= 5 for x in s.bone_guides(row) for _, _, d, _, _ in x["bones"])
    px = s.project(1, row, g[0]["target"])
    assert px.shape == (2,) and np.isfinite(px).all()


def test_length_guides_in_the_window():
    from tests.test_multiview import _app, _on_real_file
    app = _app()
    from openkick.ui.multiview import BONES_H, MultiviewWindow
    s, _ = make()
    _on_real_file(s)
    s.solve()
    w = MultiviewWindow()
    w.show()
    w.set_session(s)
    app.processEvents()
    wrong_knee(s, shift=(80.0, 40.0))
    w.ctl.recompute(cameras=False)
    w.lengths_btn.setChecked(True)
    tl = w.timeline
    assert "bones" in tl.lanes()
    w.ctl.set_row(42)
    for cv in w.canvases:
        cv.grab()                                                            # paints the guides
    tl.grab()
    lane = tl.lanes()["bones"]
    from PySide6.QtCore import QPointF
    x = lane.x() + (42 + 0.5) * lane.width() / tl.cells()["n"]
    assert tl.bone_tip(QPointF(x, lane.center().y())) is not None
    h = tl.height()
    w.lengths_btn.setChecked(False)
    assert "bones" not in tl.lanes() and tl.height() == h - BONES_H
    w.lengths_btn.setChecked(True)
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()
