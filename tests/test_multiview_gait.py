"""Gait phases from the 3D reconstruction, on a made-up walker seen from anywhere."""
import numpy as np
import pytest

from openkick import i18n
from openkick.multiview import gait
from openkick.multiview.geometry import HALPE26, IDX

THIGH = SHANK = 0.45
REACH = 0.35               # the heel goes from this far ahead of the pelvis to this far behind it (m)
HALF_WIDTH = 0.10          # each foot this far from the middle, sideways
STANCE = 0.6


def rotation(yaw: float, pitch: float) -> np.ndarray:
    """A camera turned `yaw` round the vertical and tilted `pitch` down (radians)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    turn = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    tilt = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return tilt @ turn


def walker3d(seconds=8.0, fps=30.0, cadence=110.0, overground=False, yaw=0.6, pitch=0.25, noise=0.0):
    """Rows × 26 points × 3 in metres, in the axes of a camera (x right, y down, z away), and the
    times. The person walks along `forward`, which the camera sees from an angle."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    stride = 120.0 / cadence
    speed = 2 * REACH / (STANCE * stride) if overground else 0.0
    fwd, up, side = np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0])
    pelvis = (speed * t)[:, None] * fwd + 0.9 * up
    X = np.zeros((n, len(HALPE26), 3))
    for name, shift, lat in (("right", 0.0, HALF_WIDTH), ("left", 0.5, -HALF_WIDTH)):
        u = (t / stride + shift) % 1.0                            # 0 = initial contact
        stance = u < STANCE
        w = np.clip((u - STANCE) / (1 - STANCE), 0, 1)
        ahead = np.where(stance, REACH - 2 * REACH * u / STANCE, -REACH + 2 * REACH * (1 - np.cos(np.pi * w)) / 2)
        lift = np.where(stance, 0.0, 0.12 * np.sin(np.pi * w))
        hip = pelvis + lat * side
        ankle = (pelvis[:, 0] + ahead)[:, None] * fwd + (0.08 + lift)[:, None] * up + lat * side
        d = np.linalg.norm(ankle - hip, axis=1)
        v = (ankle - hip) / d[:, None]
        bend = np.cross(side, v)                                  # in the walking plane, forward of the leg
        bend *= np.sign(bend @ fwd)[:, None]
        knee = hip + v * (d / 2)[:, None] + bend * np.sqrt(np.maximum(THIGH ** 2 - (d / 2) ** 2, 0))[:, None]
        heel = ankle - 0.05 * fwd - 0.06 * up
        shoulder = pelvis + 0.5 * up + 0.04 * fwd + 1.8 * lat * side
        for key, value in (("hip", hip), ("knee", knee), ("ankle", ankle), ("heel", heel),
                           ("toe", heel + 0.24 * fwd), ("small_toe", heel + 0.2 * fwd + 0.04 * lat / HALF_WIDTH * side),
                           ("shoulder", shoulder), ("elbow", shoulder - 0.28 * up), ("wrist", shoulder - 0.52 * up),
                           ("eye", pelvis + 0.72 * up + 0.08 * fwd + 0.3 * lat * side),
                           ("ear", pelvis + 0.7 * up + 0.7 * lat * side)):
            X[:, IDX[f"{name}_{key}"]] = value
    X[:, IDX["hip"]] = pelvis
    X[:, IDX["neck"]] = pelvis + 0.55 * up + 0.04 * fwd
    X[:, IDX["head"]] = pelvis + 0.8 * up + 0.04 * fwd
    X[:, IDX["nose"]] = pelvis + 0.72 * up + 0.12 * fwd
    X = X @ rotation(yaw, pitch).T + np.array([0.0, 0.0, 4.0])
    if noise:
        X += np.random.default_rng(0).normal(0, noise, X.shape)
    return X, t


@pytest.fixture(autouse=True)
def english():
    before = i18n.language()
    i18n.set_language("en")
    yield
    i18n.set_language(before)


def rows(g, title):
    t = next(t for t in g.result.tables if t.title == title)
    return {r[0]: r for r in t.rows}


def number(cell: str) -> float:
    return float(str(cell).replace("±", " ").replace("°", " ").split()[0])


@pytest.mark.parametrize("yaw, pitch, overground, fps", [
    (0.0, 0.0, False, 30), (0.6, 0.25, True, 30), (np.pi / 2, 0.3, False, 60), (2.5, 0.1, True, 60)])
def test_strides_from_any_camera(yaw, pitch, overground, fps):
    X, t = walker3d(fps=fps, yaw=yaw, pitch=pitch, overground=overground, noise=0.005)
    g = gait.analyse(X, t)
    assert g is not None
    cycle = rows(g, "Gait cycle")
    assert abs(number(cycle["Cadence"][1]) - 110) < 2.0
    assert abs(number(cycle["Stance"][1]) - 100 * STANCE) < 4.0
    # at contact the other foot is half a stride on: REACH - 2 REACH 0.5 / STANCE behind the pelvis
    assert abs(number(cycle["Step length"][1]) - 2 * REACH * 0.5 / STANCE) < 0.04
    assert abs(number(cycle["Step width"][1]) - 200 * HALF_WIDTH) < 2.0
    if overground:
        assert abs(number(cycle["Stride length"][1]) - 2 * REACH / STANCE) < 0.05
    else:
        assert "Stride length" not in cycle
    sides = rows(g, "Left and right")
    assert abs(number(sides["Stance"][1]) - number(sides["Stance"][2])) < 3.0


def test_phases_and_subphases_follow_the_strides():
    from openkick.core import gait as G
    X, t = walker3d()
    g = gait.analyse(X, t)
    assert len(g.phases) == len(g.strides) >= 5
    assert len(g.subphases) == 8 * len(g.phases)               # Perry's eight phases, as in 2D
    names = [full for _, full, _, _, _ in G.PERRY]
    stride = 120 / 110
    for i, ph in enumerate(g.phases):
        eight = g.subphases[8 * i:8 * i + 8]
        assert [s.name for s in eight] == names
        assert eight[0].start == ph.start and eight[-1].end == ph.end
        assert all(a.end + 1 == b.start for a, b in zip(eight, eight[1:]))       # back to back, no gaps
        share = {s.name: (s.start - ph.start) / (ph.end + 1 - ph.start) for s in eight}
        assert abs(share["Initial swing"] - STANCE) < 0.06                   # toe off
        assert abs(share["Mid stance"] - (STANCE - 0.5)) < 0.06              # the other foot leaves
        assert abs(share["Pre-swing"] - 0.5) < 0.06                          # the other foot lands
        start = t[ph.start] % stride if g.side == "right" else (t[ph.start] + stride / 2) % stride
        assert min(start, stride - start) < 2.5 / 30
    assert g.phase_at(g.phases[1].start + 1) is g.phases[1]
    assert g.subphase_at(g.phases[1].end).name == "Terminal swing"


def test_no_walking():
    X, t = walker3d()
    still = np.repeat(X[:1], len(t), axis=0)
    assert gait.analyse(still, t) is None
    assert gait.analyse(np.full_like(X, np.nan), t) is None
    assert gait.analyse(X[:5], t[:5]) is None


def test_spanish():
    i18n.set_language("es")
    X, t = walker3d(overground=True)
    g = gait.analyse(X, t)
    assert g.phases[0].name == "Zancada 1" and g.subphases[5].name == "Balanceo inicial"
    for tb in g.result.tables:
        for text in [tb.title, *tb.columns, *(str(c) for r in tb.rows for c in r)]:
            assert not any(w in text for w in ("Stride", "Step", "Knee", "Feet", "steps", "Gait", "Check")), text
    assert "zancadas" in g.result.summary


def walking_session(count=3, noise=0.7):
    """The walker filmed by `count` cameras around it (2D points only, as the detector gives them),
    each starting to record at its own time."""
    from fractions import Fraction

    from openkick.core.video import VideoMeta
    from openkick.multiview import geometry as GEO
    from openkick.multiview.session import Session, View
    from tests.test_multiview import look_at
    X, t = walker3d(seconds=9, yaw=0, pitch=0)
    world = (X - np.array([0.0, 0.0, 4.0])) * np.array([1.0, -1.0, 1.0])         # y up, feet on the floor
    views = []
    for k, (center, size, start) in enumerate((((3.5, 1.2, 2.0), (848, 478), 0.0), ((-1.0, 1.1, 4.2), (720, 1280), 0.3),
                                               ((-3.8, 1.5, -1.5), (1280, 720), 0.17))[:count]):
        R, tt = look_at(center)
        K = GEO.guess_intrinsics(*size)
        keep = slice(int(start * 30), int((start + 8) * 30))
        Xc = world[keep] @ R.T + tt
        xy = Xc[..., :2] / Xc[..., 2:3] * K[0, 0] + K[:2, 2]
        xy += np.random.default_rng(k).normal(0, noise, xy.shape)
        own = t[keep] - t[keep][0]
        meta = VideoMeta(f"cam{k}.mp4", *size, 30.0, Fraction(1, 1000), [round(1000 * x) for x in own])
        views.append(View(meta, own, xy, np.full(xy.shape[:2], 0.9)))
    return Session(views, 1.75)


def test_gait_through_the_whole_3d_reconstruction():
    s = walking_session()
    s.solve()
    g = gait.analyse(s.X, s.t)
    assert g is not None and len(g.strides) >= 5
    cycle = rows(g, "Gait cycle")
    assert abs(number(cycle["Cadence"][1]) - 110) < 3.0
    assert abs(number(cycle["Stance"][1]) - 100 * STANCE) < 5.0
    assert abs(number(cycle["Step width"][1]) - 200 * HALF_WIDTH) < 4.0     # scaled from the height given


def test_window_shows_the_gait():
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    pytest.importorskip("PySide6.QtWidgets")
    from tests.test_multiview import _app, _on_real_file
    app = _app()
    from openkick.ui.multiview import MultiviewWindow
    s = walking_session(2)
    _on_real_file(s, truncate=False)
    s.solve()
    w = MultiviewWindow()
    w.resize(1400, 860)
    w.show()
    w.set_session(s)
    app.processEvents()
    g = w.ctl.gait()
    assert g is not None and w.ctl.gait() is g                    # worked out once per revision
    assert w.panel.gait_card.isVisibleTo(w.panel)
    w.ctl.set_row(g.subphases[5].start)
    assert "Stride 1" in w.panel.gait_now.text() and "Initial swing" in w.panel.gait_now.text()
    assert w.panel.gait_rows["cadence"].text().endswith("steps/min")

    # the timeline: strides and their stance and swing above the doubt bars, as in the 2D timeline
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    from openkick.ui.multiview import BARS_H, BONES_H, STANCE_H, STRIDES_H
    tl = w.timeline
    lanes = tl.lanes()
    assert {"strides", "phases", "bars"} <= set(lanes)
    assert tl.height() == BARS_H + STRIDES_H + STANCE_H + (BONES_H if w.ctl.show_lengths else 0)
    assert lanes["strides"].bottom() <= lanes["phases"].top() and lanes["phases"].bottom() <= lanes["bars"].top()

    def at(lane, ph):
        a, b = tl._span(tl.cells()["rows"], ph)
        return QPointF(lanes[lane].x() + (a + b) / 2 * lanes[lane].width() / tl.cells()["n"], lanes[lane].center().y())

    def send(kind, pos, button=Qt.NoButton):
        QApplication.sendEvent(tl, QMouseEvent(kind, pos, tl.mapToGlobal(pos.toPoint()), button, button, Qt.NoModifier))

    send(QEvent.MouseMove, at("phases", g.subphases[8 + 2]))
    assert tl.toolTip().startswith("Stride 2 · Mid stance · ")
    send(QEvent.MouseMove, at("strides", g.phases[1]))
    assert tl.toolTip().startswith("Stride 2 · ") and "stance" not in tl.toolTip()
    send(QEvent.MouseMove, lanes["bars"].center())
    assert tl.toolTip() == tl.tip
    w.ctl.set_row(0)
    send(QEvent.MouseButtonDblClick, at("strides", g.phases[1]), Qt.LeftButton)
    assert w.ctl.row == g.phases[1].start                        # double click: to the start of the stride
    send(QEvent.MouseButtonDblClick, at("phases", g.subphases[3]), Qt.LeftButton)
    assert w.ctl.row == g.subphases[3].start
    tl.grab()

    # with one video on its own, the blocks follow that video's frames
    w.ctl.set_focus(1)
    app.processEvents()
    cells = tl.cells()
    a, b = tl._span(cells["rows"], g.phases[1])
    assert 0 < a < b <= cells["n"]
    send(QEvent.MouseButtonDblClick, at("strides", g.phases[1]), Qt.LeftButton)
    assert cells["rows"][w.ctl.focus_frame - cells["first"]] >= g.phases[1].start
    tl.grab()
    w.ctl.set_focus(None)
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()


def test_no_gait_rows_when_nobody_walks():
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    pytest.importorskip("PySide6.QtWidgets")
    from tests.test_multiview import _app, _on_real_file, make
    app = _app()
    from openkick.ui.multiview import BARS_H, MultiviewWindow
    s, _ = make()                                                 # squats and a knee raise: no strides
    _on_real_file(s)
    s.solve()
    w = MultiviewWindow()
    w.resize(1400, 860)
    w.show()
    w.set_session(s)
    app.processEvents()
    assert w.ctl.gait() is None
    assert set(w.timeline.lanes()) == {"track", "bars"} and w.timeline.height() == BARS_H
    assert not w.panel.gait_card.isVisibleTo(w.panel)
    w.timeline.grab()
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()
