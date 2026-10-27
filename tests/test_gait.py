"""Gait plugin, on a made-up walker whose contacts and toe offs are known."""
from fractions import Fraction

import numpy as np
import pytest

from openkick import i18n
from openkick.core import gait as G
from openkick.core import landmarks as L
from openkick.core import templates
from openkick.core.project import SRC_AUTO, Calibration, Phase, Project
from openkick.core.video import VideoMeta
from openkick.plugins import PluginContext
from openkick.plugins.builtin import gait

THIGH = SHANK = 230.0
REACH = 180.0              # the heel goes from this far ahead of the pelvis to this far behind it
STANCE = 0.6


def walker(seconds=8.0, fps=30.0, cadence=110.0, near="right", forward=1.0, overground=False,
           noise=0.0, shoulders=0.0) -> Project:
    """A person walking seen from the side. Each foot slides back under the pelvis while on the
    ground (stance, the first 60 % of the stride) and swings forward through the air. Overground
    the pelvis moves on and the foot stays put while it is on the ground. `shoulders` (pixels) sets
    them apart across the picture, as a wide camera shows someone near the edges of its picture."""
    n = int(seconds * fps)
    stride = 120.0 / cadence
    speed = 2 * REACH / (STANCE * stride) if overground else 0.0
    width = int(400 + speed * seconds + 400) if overground else 1280
    p = Project(VideoMeta("walker.mp4", width, 900, fps, Fraction(1, 30000), [round(i * 30000 / fps) for i in range(n)]))
    t = np.arange(n) / fps
    start = 200.0 if forward > 0 else width - 200.0
    pelvis = np.stack([start + forward * speed * t, np.full(n, 380.0)], axis=1)
    xy = np.zeros((n, L.COUNT, 2))
    far = templates.mirrored(f"{near}_")[:-1]
    for side, shift, dx in ((near, 0.0, 0.0), (far, 0.5, 6.0)):
        u = (t / stride + shift) % 1.0                            # 0 = initial contact
        stance = u < STANCE
        w = np.clip((u - STANCE) / (1 - STANCE), 0, 1)
        ahead = np.where(stance, REACH - 2 * REACH * u / STANCE, -REACH + 2 * REACH * (1 - np.cos(np.pi * w)) / 2)
        lift = np.where(stance, 0.0, 70 * np.sin(np.pi * w))
        hip = pelvis + np.array([dx, 0.0])
        ankle = np.stack([pelvis[:, 0] + forward * ahead, np.full(n, 780.0) - lift], axis=1)
        d = np.linalg.norm(ankle - hip, axis=1)
        v = (ankle - hip) / d[:, None]
        normal = np.stack([-v[:, 1], v[:, 0]], axis=1) * -forward      # the knee bends forward
        knee = hip + v * (d / 2)[:, None] + normal * np.sqrt(np.maximum(THIGH ** 2 - (d / 2) ** 2, 0))[:, None]
        shoulder = hip + np.array([forward * 30.0 + (shoulders if side == near else -shoulders), -280.0])
        heel = ankle + np.array([-forward * 25.0, 20.0])
        points = {"shoulder": shoulder, "elbow": shoulder + np.array([0.0, 140.0]),
                  "wrist": shoulder + np.array([forward * 20.0, 260.0]), "hip": hip, "knee": knee,
                  "ankle": ankle, "heel": heel, "foot_index": heel + np.array([forward * 150.0, 0.0])}
        for name, value in points.items():
            xy[:, L.INDEX[f"{side}_{name}"]] = value
            p.pose.z[:, L.INDEX[f"{side}_{name}"]] = -0.2 if side == near else 0.2
    xy[:, :11] = pelvis[:, None] + np.array([forward * 60.0, -360.0])
    if noise:
        xy += np.random.default_rng(0).normal(0, noise, xy.shape)
    p.pose.xy[:] = xy
    p.pose.raw_xy[:] = xy
    p.pose.visibility[:] = 0.95
    p.pose.source[:] = SRC_AUTO
    return p


def table(result, title: str):
    return next(t for t in result.tables if t.title == i18n.tr(title))


def number(cell: str) -> float:
    return float(str(cell).replace("±", " ").replace("°", " ").split()[0])


@pytest.fixture(autouse=True)
def english():
    before = i18n.language()
    i18n.set_language("en")
    yield
    i18n.set_language(before)


@pytest.mark.parametrize("fps, cadence, forward, overground", [
    (30, 110, 1.0, False), (60, 100, -1.0, False), (30, 120, 1.0, True), (60, 105, -1.0, True)])
def test_strides_are_found_and_timed(fps, cadence, forward, overground):
    p = walker(fps=fps, cadence=cadence, forward=forward, overground=overground, noise=1.0)
    r = gait.PLUGINS[0].run(PluginContext(p))
    cycle = {row[0]: row for row in table(r, "Gait cycle").rows}
    assert abs(number(cycle["Cadence"][1]) - cadence) < 2.0
    assert abs(number(cycle["Stride time"][1]) - 120 / cadence) < 0.03
    assert abs(number(cycle["Stance"][1]) - 100 * STANCE) < 4.0
    strides = table(r, "Stride by stride").rows
    assert abs(len(strides) - 8 * cadence / 120) <= 2
    assert "Right side" in r.summary


def test_phases_and_subphases_are_proposed():
    p = walker(cadence=110, noise=1.0)
    assert not p.phases and not p.subphases
    r = gait.PLUGINS[0].run(PluginContext(p))
    strides = table(r, "Stride by stride").rows
    assert len(p.phases) == len(strides) >= 2
    assert [ph.name for ph in p.phases[:2]] == ["Stride 1", "Stride 2"]
    assert len(p.subphases) == 8 * len(p.phases)
    fps, stride = 30.0, 120 / 110
    names = [full for _, full, _, _, _ in G.PERRY]
    for i, ph in enumerate(p.phases):
        eight = p.subphases[8 * i:8 * i + 8]
        assert [s.name for s in eight] == names
        assert eight[0].start == ph.start and eight[-1].end == ph.end
        assert all(a.end + 1 == b.start for a, b in zip(eight, eight[1:]))       # back to back, no gaps
        assert abs((ph.start / fps) % stride) < 2 / fps or abs((ph.start / fps) % stride - stride) < 2 / fps
        share = {s.name: (s.start - ph.start) / (ph.end + 1 - ph.start) for s in eight}
        assert abs(share["Initial swing"] - STANCE) < 0.06                   # toe off
        assert abs(share["Mid stance"] - 0.1) < 0.06                         # the other foot leaves (half a stride later)
        assert abs(share["Pre-swing"] - 0.5) < 0.06                          # the other foot lands
    for a, b in zip(p.phases, p.phases[1:]):
        assert a.end < b.start
    assert "automatically" in r.summary
    p.phases[0].name = "mine"                                   # phases already there are kept
    gait.PLUGINS[0].run(PluginContext(p))
    assert p.phases[0].name == "mine"


def test_both_legs_and_stride_length():
    p = walker(cadence=110, overground=True)
    r = gait.PLUGINS[0].run(PluginContext(p))
    sides = {row[0]: row for row in table(r, "Left and right").rows}
    assert abs(number(sides["Stance"][1]) - number(sides["Stance"][2])) < 3.0
    assert "Stride length" not in {row[0] for row in table(r, "Gait cycle").rows}
    p.calibration = Calibration(px_per_m=300.0, a=(0, 0), b=(300, 0), length_m=1.0)
    cycle = {row[0]: row for row in table(gait.PLUGINS[0].run(PluginContext(p)), "Gait cycle").rows}
    expected = 2 * REACH / STANCE / 300.0                        # metres the pelvis covers in a stride
    assert abs(number(cycle["Stride length"][1]) - expected) < 0.05
    assert abs(number(cycle["Walking speed"][1]) - expected / (120 / 110)) < 0.05
    treadmill = walker(cadence=110)
    treadmill.calibration = p.calibration
    rows = {row[0] for row in table(gait.PLUGINS[0].run(PluginContext(treadmill)), "Gait cycle").rows}
    assert "Stride length" not in rows


def test_user_phases_are_compared():
    p = walker(seconds=10, cadence=110)
    n = p.frame_count
    p.phases = [Phase("before", 0, n // 2 - 1), Phase("after", n // 2, n - 1)]
    r = gait.PLUGINS[0].run(PluginContext(p))
    assert [ph.name for ph in p.phases] == ["before", "after"]
    assert {row[0] for row in table(r, "By phase").rows} == {"before", "after"}


def test_nothing_to_measure():
    empty = Project(walker().video)
    r = gait.PLUGINS[0].run(PluginContext(empty))
    assert "No strides" in r.summary and r.tables and not empty.phases


def test_every_text_has_a_spanish_translation():
    i18n.set_language("es")
    p = walker(cadence=110, overground=True)
    p.calibration = Calibration(px_per_m=300.0, a=(0, 0), b=(300, 0), length_m=1.0)
    r = gait.PLUGINS[0].run(PluginContext(p))
    english_words = ("Stride", "stride", "Knee", "Hip", "Check", "side", "Cadence", "Stance", "Swing",
                     "Trunk", "Gait", "Left", "Right", "Feet", "steps")
    for t in r.tables:
        for text in [t.title, *t.columns, *(str(c) for row in t.rows for c in row)]:
            assert not any(w in text for w in english_words), text
    assert "zancadas" in r.summary and p.phases[0].name == "Zancada 1"
    assert [s.name for s in p.subphases[:2]] == ["Contacto inicial", "Respuesta a la carga"]


def test_gait_template_auto_run():
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    from openkick.ui.controller import Controller
    from openkick.ui.results import ResultsPage
    from openkick.ui.timeline import Timeline
    p = walker(cadence=110, noise=1.0)
    templates.apply(p, templates.find("gait"))
    assert p.template == "gait"
    c = Controller()
    c.set_project(p)
    assert "gait" in c.plugin_results
    assert len(p.phases) >= 2 and len(p.subphases) == 8 * len(p.phases)
    rp = ResultsPage(c)
    rp.show()
    rp.refresh()
    assert rp.plugin_box.count() > 0
    rp.close()
    tl = Timeline(c)
    tl.resize(1000, 200)
    assert "subphases" in tl.rows()
    tl.repaint()
    tl.close()


def test_crossing_a_wide_picture_counts_as_side_on():
    """Seen a little from the front where they come in and from behind where they leave, the
    shoulders part (as in a wide action camera), but the feet still show their whole length."""
    p = walker(cadence=110, overground=True, shoulders=90.0, noise=1.0)
    r = gait.PLUGINS[0].run(PluginContext(p))
    check = {row[0]: row for row in table(r, "Recording check").rows}
    assert check["Filmed from the side"][2] == "OK"
    assert len(table(r, "Stride by stride").rows) >= 5 and len(p.phases) >= 5


def test_walking_towards_the_camera_gives_no_strides():
    """Head-on, the feet look short and stay under the body: nothing to time, so no phases."""
    p = walker(cadence=110)
    xy = p.pose.xy
    for side, dx in (("left", -40.0), ("right", 40.0)):
        hip_x = xy[:, L.INDEX[f"{side}_hip"], 0]
        for name in ("knee", "ankle", "heel"):
            xy[:, L.INDEX[f"{side}_{name}"], 0] = hip_x + dx
        xy[:, L.INDEX[f"{side}_foot_index"], 0] = hip_x + dx + 8.0
    r = gait.PLUGINS[0].run(PluginContext(p))
    assert "No strides" in r.summary and not p.phases and not p.subphases
    check = {row[0]: row for row in table(r, "Recording check").rows}
    assert check["Filmed from the side"][2] == "Review"


def test_the_eight_phases_follow_the_feet():
    """Events found from both feet, in order; the heel of this walker never lifts before toe off,
    so heel rise takes its usual place and says so."""
    p = walker(cadence=100, noise=1.0)
    r = gait.PLUGINS[0].run(PluginContext(p))
    rows = {row[0]: row for row in table(r, "The eight phases of the gait cycle").rows}
    assert set(rows) == {full for _, full, _, _, _ in G.PERRY}
    assert rows["Mid stance"][3] != "—" and rows["Pre-swing"][3] == "—"      # heel rise estimated, the rest seen
    assert "could not be seen" in r.summary
    share = {k: number(v[1]) for k, v in rows.items()}
    assert abs(sum(share.values()) - 100) < 1.5
    assert abs(share["Initial swing"] + share["Mid swing"] + share["Terminal swing"] - 100 * (1 - STANCE)) < 6


def test_events_out_of_order_take_their_usual_place():
    s = {"a": 0, "o": 60, "b": 100}
    n = 100
    flat = {"contacts": [], "offs": [], "heel_y": np.full(n, 500.0), "ankle_x": np.zeros(n), "knee_x": np.zeros(n)}
    found = G.gait_events(s, flat, flat, 1.0, 200.0)        # nothing to see: every event is the usual one
    ev = found["events"]
    order = [0] + [ev[k] for k in G.EVENTS] + [100]
    assert order == sorted(order) and len(set(order)) == len(order)
    assert set(found["estimated"]) >= {"ofo", "hr", "oic"}
    assert abs(ev["oic"] - 50) <= 1 and ev["to"] == 60


def test_dragging_a_phase_edge_on_the_timeline():
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from openkick.ui.controller import Controller
    from openkick.ui.timeline import Timeline
    p = walker(cadence=100, noise=1.0)
    gait.PLUGINS[0].run(PluginContext(p))
    c = Controller()
    c.set_project(p)
    tl = Timeline(c)
    tl.resize(1400, 300)
    tl.show()
    app.processEvents()
    sub = p.subphases[3]                                      # terminal stance of the first stride
    edge, y = sub.start, tl.rows()["subphases"].center().y()
    x = tl.x_of(edge)

    def mouse(kind, x, buttons=Qt.LeftButton):
        e = QMouseEvent(kind, QPointF(x, y), tl.mapToGlobal(QPointF(x, y).toPoint()), Qt.LeftButton, buttons, Qt.NoModifier)
        QApplication.sendEvent(tl, e)
    assert tl._boundary_at(QPointF(x, y)) == ("edge", edge)
    mouse(QEvent.MouseButtonPress, x)
    mouse(QEvent.MouseMove, tl.x_of(edge + 2))
    mouse(QEvent.MouseButtonRelease, tl.x_of(edge + 2), Qt.NoButton)
    assert sub.start == edge + 2 and p.subphases[2].end == edge + 1          # the neighbour follows
    c.undo.undo()
    assert p.subphases[3].start == edge
    tl.close()
