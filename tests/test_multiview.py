"""3D from several videos: a synthetic person seen by known cameras. The solver gets only the
2D points and the person's height, and must recover time offsets, cameras and joint angles; the
corrections (drag, exchange left/right) must behave like the user expects."""
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from openkick.core.video import VideoMeta
from openkick.multiview import geometry as G
from openkick.multiview.session import PROBLEM_CM, SRC_MANUAL, Session, View

I = G.IDX
HEIGHT = 1.70


def figure(t: float) -> np.ndarray:
    """A person (26 points, metres; x right, y up, z forward) squatting, raising a knee, swinging
    the arms and drifting across the floor: enough variety to locate two cameras."""
    s = 0.5 - 0.5 * np.cos(2 * np.pi * t / 2.3)                 # squat depth 0..1
    lift = np.clip(np.sin(2 * np.pi * (t - 0.4) / 3.1), 0, 1)   # left knee raise
    P = np.zeros((26, 3))
    pelvis = np.array([0.25 * np.sin(t * 0.9), 0.93 - 0.30 * s, 0.30 * np.sin(t * 0.6 + 1)])
    lean = 0.35 * s
    up = np.array([0, np.cos(lean), np.sin(lean)])
    P[I["hip"]] = pelvis
    for side, sx in (("left", -1), ("right", 1)):
        hip = pelvis + np.array([0.10 * sx, 0, 0])
        ankle = np.array([pelvis[0] + 0.13 * sx, 0.08, pelvis[2]])          # feet slide: bones keep their length
        if side == "left":
            ankle = ankle + np.array([0, 0.25 * lift, 0.20 * lift])
        mid = (hip + ankle) / 2
        reach = np.linalg.norm(ankle - hip)
        bend = np.sqrt(max(0.43 ** 2 - (reach / 2) ** 2, 0.0))  # thigh = shank = 0.43 m
        axis = (ankle - hip) / reach
        fwd = np.array([0, 0, 1.0]) - axis[2] * axis                # bend forward, square to the leg
        knee = mid + fwd / np.linalg.norm(fwd) * bend
        P[I[f"{side}_hip"]], P[I[f"{side}_knee"]], P[I[f"{side}_ankle"]] = hip, knee, ankle
        P[I[f"{side}_heel"]] = ankle + np.array([0, -0.05, -0.05])
        P[I[f"{side}_toe"]] = ankle + np.array([0.02 * sx, -0.06, 0.17])
        P[I[f"{side}_small_toe"]] = ankle + np.array([0.07 * sx, -0.06, 0.13])
        shoulder = pelvis + up * 0.52 + np.array([0.19 * sx, 0, 0])
        swing = 0.9 * np.sin(2 * np.pi * t / 1.7 + (0 if sx > 0 else 2.0))
        elbow = shoulder + 0.29 * np.array([0.1 * sx, -np.cos(swing), np.sin(swing)])
        wrist = elbow + 0.26 * np.array([0, -np.cos(swing + 0.6), np.sin(swing + 0.6)])
        P[I[f"{side}_shoulder"]], P[I[f"{side}_elbow"]], P[I[f"{side}_wrist"]] = shoulder, elbow, wrist
        P[I[f"{side}_eye"]] = pelvis + up * 0.70 + np.array([0.03 * sx, 0, 0.08])
        P[I[f"{side}_ear"]] = pelvis + up * 0.69 + np.array([0.07 * sx, 0, 0])
    P[I["neck"]] = pelvis + up * 0.54
    P[I["nose"]] = pelvis + up * 0.67 + np.array([0, 0, 0.10])
    P[I["head"]] = pelvis + up * 0.78
    return P


def look_at(center, target=(0, 0.9, 0)):
    """Camera pose (x right, y down, z forward): x_cam = R X + t."""
    c, tg = np.asarray(center, float), np.asarray(target, float)
    z = (tg - c) / np.linalg.norm(tg - c)
    x = np.cross(np.array([0, -1.0, 0]), z)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return R, -R @ c


def camera_view(center, size, fps, t0, duration, noise, seed, drift=None):
    w, h = size
    K = G.guess_intrinsics(w, h)
    R, tt = look_at(center)
    n = int(duration * fps)
    t = np.arange(n) / fps
    rng = np.random.default_rng(seed)
    xy = np.empty((n, 26, 2))
    for i in range(n):
        Xc = figure(t[i] + t0) @ R.T + tt
        xy[i] = Xc[:, :2] / Xc[:, 2:3] * K[0, 0] + K[:2, 2]
    xy += rng.normal(0, noise, xy.shape)
    motion = None
    if drift is not None:                       # a handheld camera: the picture slides
        shift = np.stack([drift[0] * np.sin(t * 2.0), drift[1] * (1 - np.cos(t * 1.3))], axis=1)
        xy = xy - shift[:, None, :]
        motion = np.tile(np.eye(3), (n, 1, 1))
        motion[:, :2, 2] = shift
    meta = VideoMeta("synthetic.mp4", w, h, fps, Fraction(1, 1000), [round(1000 * x) for x in t])
    return View(meta, t, xy, np.full((n, 26), 0.9), motion), (R, tt)


def make(noise=0.7, offset=0.37, drift=None) -> tuple[Session, dict]:
    a, (Ra, ta) = camera_view((4.0, 1.3, 0.8), (848, 478), 30.0, 0.0, 5.0, noise, 1)
    b, (Rb, tb) = camera_view((-0.6, 1.1, 4.5), (720, 1280), 30.0, offset, 4.0, noise, 2, drift)
    R = Rb @ Ra.T
    return Session([a, b], HEIGHT), {"R": R, "t": tb - R @ ta, "offset": offset, "Ra": Ra, "ta": ta}


# further cameras: (position, picture size, frames per second, clock offset, seconds recorded)
EXTRA = [((2.8, 1.6, 3.6), (1280, 720), 30.0, -0.21, 4.4), ((-4.2, 1.2, -0.5), (720, 1280), 60.0, 0.12, 4.5),
         ((0.5, 2.4, -4.6), (848, 478), 25.0, 0.55, 3.9)]


def make_many(count=3, noise=0.7) -> tuple[Session, dict]:
    """The same person seen by `count` cameras with different pictures, speeds and start times."""
    a, (Ra, ta) = camera_view((4.0, 1.3, 0.8), (848, 478), 30.0, 0.0, 5.0, noise, 1)
    b, (Rb, tb) = camera_view((-0.6, 1.1, 4.5), (720, 1280), 30.0, 0.37, 4.0, noise, 2)
    views, poses, offsets = [a, b], [(Ra, ta), (Rb, tb)], [0.0, 0.37]
    for k, (pos, size, fps, off, dur) in enumerate(EXTRA[:count - 2]):
        v, pose = camera_view(pos, size, fps, off, dur, noise, 10 + k)
        views.append(v)
        poses.append(pose)
        offsets.append(off)
    rel = [(R @ Ra.T, t - R @ Ra.T @ ta) for R, t in poses]
    return Session(views, HEIGHT), {"cams": rel, "offsets": offsets}


def knee_angle(X, side="right"):
    a, o, b = X[:, I[f"{side}_hip"]], X[:, I[f"{side}_knee"]], X[:, I[f"{side}_ankle"]]
    v1, v2 = a - o, b - o
    c = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def rotation_error(R, R_true):
    return np.degrees(np.arccos(np.clip((np.trace(R @ R_true.T) - 1) / 2, -1, 1)))


@pytest.fixture(scope="module")
def solved():
    s, truth = make()
    s.solve()
    return s, truth


def test_solves_time_cameras_and_angles(solved):
    s, truth = solved
    assert abs(s.offset - truth["offset"]) < 0.012               # within a third of a frame
    assert rotation_error(s.R, truth["R"]) < 2.0
    true_dir = truth["t"] / np.linalg.norm(truth["t"])
    assert np.degrees(np.arccos(np.clip(s.d @ true_dir, -1, 1))) < 3.0
    assert abs(s.scale - np.linalg.norm(truth["t"])) / np.linalg.norm(truth["t"]) < 0.06   # metres from the height
    ref = np.array([knee_angle(figure(t)[None])[0] for t in s.t])
    for side in ("right", "left"):
        ref = np.array([knee_angle(figure(t)[None], side)[0] for t in s.t])
        assert np.nanmean(np.abs(knee_angle(s.X, side) - ref)) < 2.0
    q = s.quality()
    assert q["level"] == "good" and q["problems"] == 0 and q["reconstructed"] > 0.99


def test_wrong_point_is_flagged_and_dragging_fixes_it():
    s, _ = make()
    s.solve()
    row, j = 40, I["right_knee"]
    f = s.frame_of(1, row)
    v = s.views[1]
    good = v.xy[f, j].copy()
    v.xy[f, j] = v.raw_xy[f, j] = good + np.array([45.0, -30.0])   # the detector missed the knee in video 2
    s.update()
    # displaced mostly along the other camera's line of sight: the cameras barely disagree, yet it is caught
    assert s.err_cm[row, j] < PROBLEM_CM < s.jump_cm[row, j]
    assert [(r, k) for r, k, _ in s.problems()] == [(row, j)]
    assert s.next_problem(0) == row and s.worst_joint(row) == j
    truth = knee_angle(figure(s.t[row])[None])[0]
    before = knee_angle(s.X_raw)[row]
    assert abs(knee_angle(s.X)[row] - truth) < 2.0 < abs(before - truth)      # the smoothing already hides it
    edit = s.move_point(1, f, j, tuple(good))                      # the user drags it back onto the knee
    s.update()
    assert v.source[f, j] == SRC_MANUAL and s.issue_cm[row, j] < 3.0 and not s.problems()
    assert abs(knee_angle(s.X_raw)[row] - truth) < 2.0 < abs(before - knee_angle(s.X_raw)[row])
    edit.apply(s, new=False)                                       # undo
    s.update()
    assert (row, j) in {(r, k) for r, k, _ in s.problems()} and v.source[f, j] != SRC_MANUAL


def test_spreading_a_correction_to_neighbors_spares_hand_placed_points():
    s, _ = make()
    s.solve()
    v, j = s.views[0], I["left_wrist"]
    s.move_point(0, 51, j, tuple(v.xy[51, j] + 4))
    kept = v.xy[51, j].copy()
    start = v.xy[:, j].copy()
    s.move_point(0, 50, j, tuple(v.xy[50, j] + np.array([20.0, 0])), scope="neighbors", radius=3)
    moved = v.xy[:, j, 0] - start[:, 0]
    assert moved[50] == pytest.approx(20) and 0 < moved[47] < moved[48] < moved[49] < 20
    assert moved[46] == 0 and moved[54] == 0 and np.allclose(v.xy[51, j], kept)


def test_left_right_confusion_is_found_and_fixed():
    s, _ = make()
    s.solve()
    v = s.views[1]
    perm = G.swap_indices(("legs",))
    v.xy[30:45] = v.xy[30:45][:, perm]                             # the detector exchanged the legs in video 2
    s.update()
    assert s.quality()["problems"] > 20
    in_a, in_b = s.find_swaps()["legs"]
    frames = {s.frame_of(1, int(r)) for r in np.nonzero(in_b)[0]}
    assert not in_a.any() and set(range(31, 44)) <= frames <= set(range(29, 46))
    edits, n = s.fix_swaps()
    s.update()
    assert n >= 13 and s.quality()["problems"] <= 2
    for e in reversed(edits):
        e.apply(s, new=False)
    s.update()
    assert s.quality()["problems"] > 20
    s.swap_sides(1, 30, 44, "legs")                                # by hand: right-click > exchange
    s.update()
    assert s.quality()["problems"] == 0


def test_handheld_camera_is_compensated():
    s, truth = make(drift=(35.0, 22.0))
    assert s.views[1].handheld
    s.solve()
    assert abs(s.offset - truth["offset"]) < 0.012 and rotation_error(s.R, truth["R"]) < 2.0
    row = 30
    f = s.frame_of(1, row)
    seen = s.views[1].xy[f]
    assert np.nanmedian(np.linalg.norm(s.projection(1, row) - seen, axis=1)) < 3.0   # drawn on the moving picture


def test_epipolar_line_passes_through_the_same_point_in_the_other_video(solved):
    s, _ = solved
    row, j = 25, I["left_ankle"]
    a, b = s.epipolar_line(0, row, j)
    p = s.projection(1, row)[j]
    d = (b - a) / np.linalg.norm(b - a)
    assert abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) < 4.0


@pytest.fixture(scope="module")
def three():
    s, truth = make_many(3)
    s.solve()
    return s, truth


def test_three_cameras_are_located_on_one_scale(three):
    s, truth = three
    assert len(s.views) == 3 and len(s.cams) == 3
    for i in (1, 2):
        R_true, t_true = truth["cams"][i]
        assert abs(s.offsets[i] - truth["offsets"][i]) < 0.012
        assert rotation_error(s.cams[i][0], R_true) < 2.0
        # every camera in metres, in the same frame: position within 6 % of its true distance
        est = -s.cams[i][0].T @ s.cams[i][1] * s.scale
        true = -R_true.T @ t_true
        assert np.linalg.norm(est - true) < 0.06 * np.linalg.norm(true)
    for side in ("right", "left"):
        ref = np.array([knee_angle(figure(t)[None], side)[0] for t in s.t])
        assert np.nanmean(np.abs(knee_angle(s.X, side) - ref)) < 2.0
    q = s.quality()
    assert q["level"] == "good" and q["problems"] == 0 and q["reconstructed"] > 0.99
    for v in range(3):                                  # the 3D skeleton lands on every video
        f = s.frame_of(v, 40)
        assert np.nanmedian(np.linalg.norm(s.projection(v, 40) - s.views[v].xy[f], axis=1)) < 3.0


def test_five_cameras(three):
    s, truth = make_many(5)
    s.solve()
    assert max(rotation_error(s.cams[i][0], truth["cams"][i][0]) for i in range(1, 5)) < 2.0
    assert max(abs(s.offsets[i] - truth["offsets"][i]) for i in range(1, 5)) < 0.015
    assert s.quality()["level"] == "good"


def test_with_three_cameras_a_wrong_view_is_left_out_and_a_hidden_point_survives():
    s, _ = make_many(3)
    s.solve()
    row, j = 40, I["right_knee"]
    truth = knee_angle(figure(s.t[row])[None])[0]
    f = s.frame_of(1, row)
    v = s.views[1]
    v.xy[f, j] = v.raw_xy[f, j] = v.xy[f, j] + np.array([45.0, -30.0])        # video 2 misses the knee
    s.update()
    assert not s.used[1, row, j] and s.used[0, row, j] and s.used[2, row, j]  # the other two outvote it
    assert abs(knee_angle(s.X)[row] - truth) < 2.0 and s.view_error_cm(1, row, j) > 6.0 and s.worst_view(row, j) == 1
    # a point one camera does not see (hidden behind the body) is still reconstructed from the other two
    edit = s.discard_point(0, s.frame_of(0, row), I["left_wrist"])
    s.update()
    assert np.isfinite(s.X[row, I["left_wrist"]]).all() and not s.used[0, row, I["left_wrist"]]
    edit.apply(s, new=False)
    # dragging in one video moves the 3D point and draws its line in the other two
    s.update()
    for other in (0, 2):
        a, b = s.epipolar_line(1, row, I["left_ankle"], other)
        p = s.projection(other, row)[I["left_ankle"]]
        d = (b - a) / np.linalg.norm(b - a)
        assert abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) < 4.0


def test_left_right_confusion_with_three_cameras():
    s, _ = make_many(3)
    s.solve()
    perm = G.swap_indices(("legs",))
    s.views[2].xy[30:60] = s.views[2].xy[30:60][:, perm]             # video 3 exchanges the legs
    s.update()
    found = s.find_swaps()["legs"]
    assert not found[0].any() and not found[1].any() and found[2].sum() >= 10
    edits, n = s.fix_swaps()
    s.update()
    assert n >= 10 and s.quality()["problems"] <= 2


def test_session_round_trip(solved, tmp_path):
    s, _ = make()
    s.solve()
    s.move_point(0, 20, I["right_ankle"], tuple(s.views[0].xy[20, I["right_ankle"]] + 6))
    s.update()
    path = tmp_path / "s.ok3d"
    s.save(path)
    assert not s.modified
    with pytest.raises(FileNotFoundError):              # the videos of this made-up session do not exist
        Session.load(path)
    z = Session.load(path, require_videos=False)
    assert z.offset == pytest.approx(s.offset) and np.allclose(z.R, s.R)
    assert z.views[0].source[20, I["right_ankle"]] == SRC_MANUAL
    assert np.allclose(z.X, s.X, equal_nan=True, atol=1e-6) and z.height_m == HEIGHT
    many, _ = make_many(3)
    many.solve()
    many.save(tmp_path / "m.ok3d")
    z3 = Session.load(tmp_path / "m.ok3d", require_videos=False)
    assert len(z3.views) == 3 and z3.base == many.base and np.allclose(z3.X, many.X, equal_nan=True, atol=1e-6)


def test_export_to_opensim(solved, tmp_path):
    from openkick.multiview.export import MARKERS, export_trc
    s, _ = solved
    info = export_trc(s, tmp_path / "m.trc")
    lines = (tmp_path / "m.trc").read_text().splitlines()
    assert info["markers"] == len(MARKERS) == 22 and lines[2].split("\t")[4] == "m"
    row = lines[6].split("\t")
    assert len(row) == 2 + 3 * 22
    y = [float(l.split("\t")[2 + 3 * list(MARKERS).index("Head") + 1]) for l in lines[6:]]
    assert 1.2 < max(y) < 1.8 and min(y) > 0.7                    # Y is up, floor at zero, metres


def test_window_drag_and_undo(solved, tmp_path):
    """The window itself: drag a point in one video, see the 3D follow, undo."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from tests.conftest import VIDEO
    if not VIDEO.exists():
        pytest.skip("sample video not available")
    from openkick.core.video import probe
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from openkick.ui.multiview import MultiviewWindow
    s, _ = make()
    real = probe(VIDEO)
    for v in s.views:                           # any real file, so frames can be decoded
        v.meta = VideoMeta(real.path, v.meta.width, v.meta.height, real.fps, real.time_base, real.pts[:len(v.t)])
        v.t, v.xy, v.raw_xy = v.t[:len(real.pts)], v.xy[:len(real.pts)], v.raw_xy[:len(real.pts)]
        v.score, v.raw_score, v.source = v.score[:len(real.pts)], v.raw_score[:len(real.pts)], v.source[:len(real.pts)]
    s.solve()
    w = MultiviewWindow()
    w.resize(1400, 860)
    w.show()
    w.set_session(s)
    app.processEvents()
    c = w.ctl
    c.follow = False                            # the other video keeps its point: they disagree
    c.set_row(12)
    cv = w.canvases[1]
    j = I["right_knee"]
    f = c.frame(1)
    p0 = cv.map(cv.v.xy[f, j])
    p1 = QPointF(p0.x() + 30, p0.y() + 12)

    def send(kind, pos, buttons=Qt.LeftButton):
        QtWidgets.QApplication.sendEvent(cv, QMouseEvent(kind, pos, cv.mapToGlobal(pos.toPoint()), Qt.LeftButton, buttons, Qt.NoModifier))

    x_before = s.X[c.row, j].copy()
    send(QEvent.MouseButtonPress, p0)
    assert c.selected == (1, j) and c.dragging == (1, j)
    send(QEvent.MouseMove, p1)
    assert not np.allclose(s.X[c.row, j], x_before)               # the 3D point follows the hand
    send(QEvent.MouseButtonRelease, p1, Qt.NoButton)
    app.processEvents()
    assert c.dragging is None and c.undo.count() == 1 and cv.v.source[f, j] == SRC_MANUAL
    assert s.issue_cm[c.row, j] > PROBLEM_CM and c.session.next_problem(0) == c.row
    assert "cm" in c.describe(c.row, j)
    w.grab()
    c.undo.undo()
    assert cv.v.source[f, j] != SRC_MANUAL and np.allclose(s.X[c.row, j], x_before, atol=2e-3)
    c.next_problem()
    w.toggle_play()
    w.stop()
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()


def _app():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.setOrganizationName("OpenKickTest")
    QCoreApplication.setApplicationName("OpenKickTest")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _on_real_file(s, truncate=True):
    """Point every view at a real video file, so the window can show frames. With
    truncate=False the views keep all their frames (pictures beyond the file repeat its last one)."""
    from tests.conftest import VIDEO
    if not VIDEO.exists():
        pytest.skip("sample video not available")
    from openkick.core.video import probe
    real = probe(VIDEO)
    n = len(real.pts) if truncate else 10 ** 9
    for v in s.views:
        v.meta = VideoMeta(real.path, v.meta.width, v.meta.height, real.fps, real.time_base, real.pts[:len(v.t)])
        v.t, v.xy, v.raw_xy = v.t[:n], v.xy[:n], v.raw_xy[:n]
        v.score, v.raw_score, v.source = v.score[:n], v.raw_score[:n], v.source[:n]
    return s


def test_choosing_videos_by_dropping_them(tmp_path):
    """Any number of files can be dropped at once; only videos count, two are the minimum."""
    import shutil

    from tests.conftest import VIDEO
    if not VIDEO.exists():
        pytest.skip("sample video not available")
    app = _app()
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    from openkick.multiview.session import MAX_VIEWS
    from openkick.ui.multiview import SetupPage
    files = []
    for k in range(MAX_VIEWS + 1):
        f = tmp_path / f"cam{k}.mp4"
        shutil.copy(VIDEO, f)
        files.append(str(f))
    (tmp_path / "notes.txt").write_text("x")
    page = SetupPage()
    page.resize(1200, 800)
    page.show()
    started = []
    page.start.connect(lambda videos, height, mode: started.append((videos, height, mode)))

    def drop(paths):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        page.dropEvent(QDropEvent(QPointF(20, 20), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        app.processEvents()

    assert not page.go.isEnabled()
    drop([files[0], str(tmp_path / "notes.txt")])                 # one video and something that is not
    assert page.picker.paths == files[:1] and not page.go.isEnabled()
    drop([files[0], files[1], files[2]])                          # two more at once; the repeated one is ignored
    assert page.picker.paths == files[:3] and page.go.isEnabled()
    drop(files[3:])                                               # more than the maximum
    assert len(page.picker.paths) == MAX_VIEWS
    page.picker.remove(files[1])
    assert files[1] not in page.picker.paths and len(page.picker.paths) == MAX_VIEWS - 1
    page.height.setValue(181)
    page.go.click()
    assert started and started[0][0] == page.picker.paths and started[0][1] == pytest.approx(1.81)
    page.deleteLater()


def test_window_with_three_videos_and_close_button():
    app = _app()
    from PySide6.QtWidgets import QToolButton

    from openkick.ui.multiview import MultiviewWindow
    s, _ = make_many(3)
    _on_real_file(s)
    s.solve()
    w = MultiviewWindow()
    w.resize(1400, 860)
    w.show()
    closers = [b for b in w.setup.findChildren(QToolButton) if "Esc" in b.toolTip()]
    assert len(closers) == 1                                      # the start page can be closed
    w.set_session(s)
    app.processEvents()
    assert len(w.canvases) == 3 and w.stack.currentIndex() == 1 and w.phase == 1
    assert w.panel.sync_view.count() == 2                         # videos 2 and 3 can be moved in time
    w.grab()
    w.ctl.dragging = (1, I["left_knee"])                          # its line appears in the other two videos
    w.grab()
    w.ctl.dragging = None
    before = s.offsets[2]
    w.panel.sync_view.setCurrentIndex(1)
    w.panel._nudge(1)
    assert s.offsets[2] > before and w.ctl.version == 1
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.ctl.undo.clear()
    closers[0].click() if w.stack.currentIndex() == 0 else w.close()
    assert not w.isVisible()
    w.deleteLater()


def test_opensim_step_and_model_view(tmp_path, monkeypatch):
    """Step 2: the 3D points go through OpenSim (scaling + inverse kinematics) in another
    process, and the model is drawn next to them, turning together."""
    pytest.importorskip("opensim")
    pytest.importorskip("Pose2Sim")
    pytest.importorskip("vtk")
    import time
    app = _app()
    from openkick.ui import multiview as MV
    s, _ = make()
    _on_real_file(s)
    s.solve()
    w = MV.MultiviewWindow()
    w._osim_folder = lambda: tmp_path / "osim"
    w.resize(1400, 860)
    w.show()
    w.set_session(s)
    w.set_phase(2)
    assert w.osim_worker is not None and w.pages.currentIndex() == 1
    t0 = time.time()
    while w.osim_worker is not None and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.05)
    assert w.osim_result and w.osim_result["ok"], w.osim_panel.info.text()
    assert (tmp_path / "osim" / "kinematics").is_dir() and w.osim_view.scene is not None
    w.ctl.set_row(30)
    app.processEvents()
    img = w.osim_view.grab().toImage()
    bright = sum(img.pixelColor(x, y).lightness() > 150 for x in range(0, img.width(), 3) for y in range(0, img.height(), 3))
    assert bright > 100                                           # bones on the dark background
    # the model's knee follows the person's: the figure squats, so the angle moves by tens of degrees
    # (only the shape is compared: this stick figure is not built like a human skeleton; the angles
    # themselves are checked against a laboratory in validation/opensim_ik.py)
    f = w.osim_view.frame()
    knee = w.osim_view.scene.data[:, w.osim_view.scene.labels.index("knee_angle_r")]
    true = 180 - np.array([knee_angle(figure(t)[None])[0] for t in s.t[0] + w.osim_view.scene.times])
    assert np.ptp(knee) > 40 and np.corrcoef(knee, true)[0, 1] > 0.97
    assert f == w.osim_view.scene.frame_at(float(s.t[30] - s.t[0]))
    assert "°" in w.osim_panel.angles["knee_angle_r"].text() and w.osim_panel.save_btn.isEnabled()
    key = w.osim_view._key
    w.points3d.orbit.yaw += 40                                    # turning one view turns the other
    w.points3d.orbit.changed.emit()
    w.osim_view.grab()
    assert w.osim_view._key != key
    # the comparison as a video: both halves, the whole movement, at real and at half speed
    from PySide6.QtWidgets import QFileDialog

    from openkick.core.video import VideoReader, probe
    assert w.osim_panel.video_btn.isEnabled()
    for speed, name in ((1.0, "cmp.mp4"), (0.5, "slow")):
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, n=name, **k: (str(tmp_path / n), "")))
        assert w.export_video(speed, size=(960, 540))
    real, slow = probe(tmp_path / "cmp.mp4"), probe(tmp_path / "slow.mp4")             # the ending is added
    seconds = float(s.t[-1] - s.t[0])
    assert (real.width, real.height) == (960, 540) and abs(real.duration - seconds) < 0.15
    assert abs(slow.frame_count - 2 * real.frame_count) <= 2
    with VideoReader(real) as reader:
        a, b = reader.frame(0), reader.frame(real.frame_count // 2)
    half = real.width // 2
    assert (a[60:480, :half].max(axis=2) > 120).sum() > 300                           # the stick figure, left
    assert (a[60:480, half:].min(axis=2) > 150).sum() > 300                           # the bones, right
    assert np.abs(a.astype(int) - b).mean() > 0.3                                     # and it moves
    # the other layout: every video above, the model seen from that camera below
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "cams.mp4"), "")))
    assert w.export_video(1.0, "cameras", size=(800, 706))
    cams = probe(tmp_path / "cams.mp4")
    assert (cams.width, cams.height) == (800, 706) and abs(cams.frame_count - real.frame_count) <= 1
    with VideoReader(cams) as reader:
        pic = reader.frame(cams.frame_count // 2)
    top, bottom = pic[40:330], pic[380:680]
    for col in (slice(0, 400), slice(400, 800)):
        assert top[:, col].std() > 8                                                  # a real video picture above
        assert (bottom[:, col].min(axis=2) > 150).sum() > 150                         # bones below
        green = (bottom[:, col, 1].astype(int) - bottom[:, col, 2] > 90) & (bottom[:, col, 1] > 180)
        assert green.sum() > 30                                                       # and the green points
    dirs = MV.camera_directions(s)
    assert len(dirs) == 2 and all(abs(np.linalg.det(B) - 1) < 1e-6 and B[2][1] >= 0.15 for B in dirs)
    assert np.degrees(np.arccos(np.clip(dirs[0][2] @ dirs[1][2], -1, 1))) > 40        # two really different viewpoints
    assert w.osim_view.grab().width() == w.osim_view.width()                          # the window's view is intact
    j = I["right_knee"]
    w.ctl.push([s.move_point(0, w.ctl.frame(0), j, tuple(s.views[0].xy[w.ctl.frame(0), j] + 9))], "move")
    assert w.osim_panel.title.text() == MV.tr("The model is out of date")
    assert not w.osim_panel.video_btn.isEnabled() and not w.export_video(1.0)        # not while the model is out of date
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.ctl.undo.clear()
    w.close()
    w.deleteLater()


def test_editing_one_video_on_its_own():
    """A video of the 3D session can be opened alone, stepped through its own frames and
    corrected with the shadow of the neighbouring frames; the 3D follows, and going back to all
    the videos keeps the corrections."""
    app = _app()
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from openkick.ui.multiview import MultiviewWindow
    from openkick.ui.overlay import GHOST_NEXT, GHOST_PREV
    s, _ = make_many(4)                               # video 4 records twice as fast as video 1
    _on_real_file(s, truncate=False)
    s.solve()
    w = MultiviewWindow()
    w.resize(1500, 900)
    w.show()
    w.set_session(s)
    app.processEvents()
    c = w.ctl
    c.set_row(20)
    cv = w.canvases[3]

    def tints(img):
        seen = {img.pixelColor(x, y).name() for x in range(0, img.width(), 2) for y in range(0, img.height(), 2)}
        return GHOST_PREV.lower() in seen, GHOST_NEXT.lower() in seen

    assert not cv.shadow_shown and not w.shadow_btn.isVisible()
    cv.solo_btn.click()                               # "Edit this video"
    app.processEvents()
    assert c.focus == 3 and cv.isVisible() and not any(x.isVisible() for x in w.canvases[:3])
    assert w.shadow_btn.isVisible() and cv.shadow_shown
    f0 = c.focus_frame
    assert f0 == s.frame_of(3, 20)
    c.step(1)                                         # one frame of THIS video, half a row
    assert c.focus_frame == f0 + 1 and c.frame(3) == f0 + 1 and c.row in (20, 21)
    c.step(1)
    assert c.focus_frame == f0 + 2 and c.row == 21
    lo, hi = c.focus_range()
    c.step(10 ** 6)
    assert c.focus_frame == hi
    c.step(-10 ** 6)
    assert c.focus_frame == lo
    c.set_row(20)
    assert c.focus_frame == f0
    # the timeline now has one bar per frame of THIS video: clicking reaches every one of them,
    # also the ones that fall between two steps of the 3D (this video records twice as fast)
    tl = w.timeline
    assert tl.cells()["n"] == hi - lo + 1 > s.frame_count
    track, reached = tl.track(), set()
    for x in np.arange(track.x() + 0.2, track.right(), 0.5):
        pos = QPointF(float(x), 30.0)
        tl.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, pos, tl.mapToGlobal(pos.toPoint()), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        reached.add(c.focus_frame)
    assert reached == set(range(lo, hi + 1))
    from PySide6.QtWidgets import QToolButton
    nxt = [b for b in w.findChildren(QToolButton) if "→" in b.toolTip()][0]
    prv = [b for b in w.findChildren(QToolButton) if "←" in b.toolTip()][0]
    c.set_focus_frame(lo)
    walked = [c.focus_frame]
    for _ in range(hi - lo):
        nxt.click()
        walked.append(c.focus_frame)
    assert walked == list(range(lo, hi + 1))                      # "next" never skips a frame
    prv.click()
    assert c.focus_frame == hi - 1
    c.set_row(20)
    assert c.focus_frame == f0 and tl.cells()["current"] == f0 - lo
    # drag a knee in this frame
    j = I["right_knee"]
    f = c.focus_frame + 1
    c.step(1)
    p0 = cv.map(cv.v.xy[f, j])
    p1 = QPointF(p0.x() + 26, p0.y() - 14)

    def send(kind, pos, buttons=Qt.LeftButton):
        QtWidgets = __import__("PySide6.QtWidgets").QtWidgets
        QtWidgets.QApplication.sendEvent(cv, QMouseEvent(kind, pos, cv.mapToGlobal(pos.toPoint()), Qt.LeftButton, buttons, Qt.NoModifier))

    before = s.X[c.row, j].copy()
    send(QEvent.MouseButtonPress, p0)
    send(QEvent.MouseMove, p1)
    send(QEvent.MouseButtonRelease, p1, Qt.NoButton)
    app.processEvents()
    assert cv.v.source[f, j] == SRC_MANUAL and c.focus_frame == f and c.undo.count() == 1     # stays on the frame edited
    assert not np.allclose(s.X[c.row, j], before)
    prev, nxt = tints(cv.grab().toImage())
    assert prev and nxt                               # the neighbouring frames' skeletons are drawn
    w.shadow_btn.toggle()
    assert tints(cv.grab().toImage()) == (False, False) and not cv.shadow_shown
    w.shadow_btn.toggle()
    w.toggle_play()                                   # not while playing
    assert not cv.shadow_shown
    w.stop()
    cv.solo_btn.click()                               # "Back to all the videos"
    app.processEvents()
    assert c.focus is None and all(x.isVisible() for x in w.canvases) and not w.shadow_btn.isVisible()
    assert tl.cells()["n"] == s.frame_count                       # back to the steps of the 3D
    assert cv.v.source[f, j] == SRC_MANUAL and not cv.shadow_shown
    w.canvases[1].solo_btn.click()
    assert c.focus == 1
    w.set_phase(2) if False else c.set_focus(None)
    c.undo.undo()
    assert cv.v.source[f, j] != SRC_MANUAL
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    c.undo.clear()
    w.close()
    w.deleteLater()


def test_saving_and_reopening_sessions(tmp_path, monkeypatch):
    """The whole life of a session file in the window: save, save under another name, reopen
    over another session with a different number of videos, find a video that was moved. No
    step may raise, and nothing may be lost without asking."""
    import shutil
    import sys

    from tests.conftest import VIDEO
    if not VIDEO.exists():
        pytest.skip("sample video not available")
    app = _app()
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from openkick.ui.multiview import MultiviewWindow
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    folder = tmp_path / "work"
    folder.mkdir()
    videos = []
    for k in range(3):
        f = folder / f"cam{k}.mp4"
        shutil.copy(VIDEO, f)
        videos.append(str(f))

    def session(count):
        s, _ = make_many(count)
        _on_real_file(s)
        for v, path in zip(s.views, videos):
            v.meta.path = path
        s.solve()
        return s

    asked, answers = [], {}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (answers["save"], "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (answers["open"], "")))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: asked.append("save?") or answers["question"]))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: asked.append("info")))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: asked.append("error: " + str(a[2]))))

    w = MultiviewWindow()
    w.resize(1400, 860)
    w.show()
    c = w.ctl
    three = session(3)
    w.set_session(three)
    app.processEvents()
    assert "•" not in w.name_lbl.text() and not c.dirty and w.confirm_discard()      # nothing to lose yet

    # a correction makes it unsaved; Save asks for a name the first time
    j = I["left_knee"]
    c.push([three.move_point(0, c.frame(0), j, tuple(three.views[0].xy[c.frame(0), j] + 7))], "move")
    assert c.dirty and "•" in w.name_lbl.text()
    answers["save"] = str(folder / "first")                       # typed without the ending
    assert w.save() and (folder / "first.ok3d").exists() and not c.dirty
    assert "first" in w.name_lbl.text() and "•" not in w.name_lbl.text() and "first" in w.windowTitle()
    # Save again writes the same file without asking; Save as makes another one and keeps the first
    c.push([three.move_point(1, c.frame(1), j, tuple(three.views[1].xy[c.frame(1), j] + 5))], "move")
    answers["save"] = "must-not-be-used"
    assert w.save() and three.path == folder / "first.ok3d" and not (folder / "must-not-be-used.ok3d").exists()
    stamp = (folder / "first.ok3d").stat().st_mtime_ns
    answers["save"] = str(folder / "second.ok3d")
    assert w.save_as() and (folder / "second.ok3d").exists() and three.path == folder / "second.ok3d"
    assert "second" in w.name_lbl.text() and (folder / "first.ok3d").stat().st_mtime_ns == stamp
    assert w._base() == folder / "second"                         # exported files take the session's name

    # another session, with fewer videos, opened on top: the old canvases must stay silent
    two = session(2)
    w.back_to_setup()
    w.set_session(two)
    app.processEvents()
    assert len(w.canvases) == 2 and "•" not in w.name_lbl.text()
    for _ in range(3):
        c.step(1)
    w.toggle_play()
    app.processEvents()
    w.stop()
    c.next_problem()
    w.canvases[1].solo_btn.click()
    c.step(1)
    c.set_focus(None)
    app.processEvents()

    # reopen the saved one while this one has unsaved corrections: it asks first
    c.push([two.move_point(0, c.frame(0), j, tuple(two.views[0].xy[c.frame(0), j] + 4))], "move")
    asked.clear()
    answers["question"] = QMessageBox.Cancel
    assert not w.load_session(str(folder / "second.ok3d")) and asked == ["save?"] and c.session is two
    answers["question"] = QMessageBox.Discard
    assert w.load_session(str(folder / "second.ok3d"))
    app.processEvents()
    s = c.session
    assert len(s.views) == 3 and len(w.canvases) == 3 and s.path == folder / "second.ok3d" and not c.dirty
    assert s.views[0].source[three.frame_of(0, 0), j] == SRC_MANUAL or s.views[0].manual().any()   # the corrections came back
    for _ in range(3):
        c.step(1)
    app.processEvents()

    # the folder is moved as a whole: the session finds its videos by itself
    moved = tmp_path / "moved"
    w.back_to_setup()
    shutil.move(str(folder), str(moved))
    asked.clear()
    assert w.load_session(str(moved / "second.ok3d")) and asked == []
    assert all(Path(v.meta.path).parent == moved for v in c.session.views)
    # one video renamed: the program asks where it is, checks it is the same one, and wants to save the new place
    w.back_to_setup()
    (moved / "cam1.mp4").rename(moved / "renamed.mp4")
    answers["open"] = str(moved / "renamed.mp4")
    asked.clear()
    assert w.load_session(str(moved / "second.ok3d")) and asked == ["info"]
    assert Path(c.session.views[1].meta.path).name == "renamed.mp4" and c.dirty
    assert w.save() and not c.dirty
    w.back_to_setup()
    asked.clear()
    assert w.load_session(str(moved / "second.ok3d")) and asked == []           # found without asking now
    # cancelling the question, a missing video, or a wrong file: the session is not opened, with a message
    w.back_to_setup()
    (moved / "cam2.mp4").rename(moved / "gone.mp4")
    answers["open"] = ""
    assert not w.load_session(str(moved / "second.ok3d")) and c.session is None
    (moved / "bad.ok3d").write_text("x")
    asked.clear()
    assert not w.load_session(str(moved / "bad.ok3d")) and asked and asked[0].startswith("error")
    assert w.stack.currentIndex() == 0
    app.processEvents()
    w.close()
    w.deleteLater()
    app.processEvents()
    assert not errors, errors
