"""Cycling template and pedal stroke plugin, on a made-up rider whose angles are known."""
from fractions import Fraction

import numpy as np
import pytest

from openkick import i18n
from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core import templates
from openkick.core.project import SRC_AUTO, Calibration, Phase, Project
from openkick.core.video import VideoMeta
from openkick.plugins import PluginContext
from openkick.plugins.builtin import cycling

THIGH = SHANK = 200.0
CRANK = 80.0


def rider(seconds=30.0, fps=30.0, rpm=85.0, reach=300.0, near="right", noise=0.0) -> Project:
    """A rider seen from the side: the hip stays put and each ankle goes round a circle whose
    centre is `reach` pixels from the hip. `reach` may change along the clip (a saddle change)."""
    n = int(seconds * fps)
    p = Project(VideoMeta("rider.mp4", 1280, 720, fps, Fraction(1, 30000), [round(i * 30000 / fps) for i in range(n)]))
    t = np.arange(n) / fps
    reach = np.broadcast_to(np.asarray(reach, float), (n,))
    hip = np.array([500.0, 260.0])
    seat = np.radians(73)
    xy = np.zeros((n, L.COUNT, 2))
    for side, shift, dx in ((near, 0.0, 0.0), (templates.mirrored(f"{near}_")[:-1], np.pi, 8.0)):
        turn = 2 * np.pi * rpm / 60 * t + shift
        centre = hip + reach[:, None] * np.array([np.cos(seat), np.sin(seat)])
        ankle = centre + CRANK * np.stack([np.cos(turn), np.sin(turn)], axis=1)
        d = np.linalg.norm(ankle - hip, axis=1)
        u = (ankle - hip) / d[:, None]
        knee = hip + u * (d / 2)[:, None] + np.stack([u[:, 1], -u[:, 0]], axis=1) * np.sqrt(THIGH ** 2 - (d / 2) ** 2)[:, None]
        shoulder = hip + 260 * np.array([np.cos(np.radians(40)), -np.sin(np.radians(40))])
        elbow = shoulder + np.array([60.0, 130.0])
        heel = ankle + np.array([-30.0, 30.0])
        points = {"shoulder": shoulder, "elbow": elbow, "wrist": elbow + np.array([130.0, 60.0]), "hip": hip,
                  "knee": knee, "ankle": ankle, "heel": heel,
                  "foot_index": heel + 150 * np.array([np.cos(np.radians(15)), np.sin(np.radians(15))])}
        for name, v in points.items():
            xy[:, L.INDEX[f"{side}_{name}"]] = v + np.array([dx, 0.0])
            p.pose.z[:, L.INDEX[f"{side}_{name}"]] = -0.2 if side == near else 0.2
    xy[:, :11] = hip + np.array([240.0, -230.0])
    if noise:
        xy += np.random.default_rng(0).normal(0, noise, xy.shape)
    p.pose.xy[:] = xy
    p.pose.raw_xy[:] = xy
    p.pose.visibility[:] = 0.95
    p.pose.source[:] = SRC_AUTO
    return p


def knee_at(distance: float) -> float:
    """Knee angle when the ankle is `distance` pixels from the hip."""
    return float(np.degrees(np.arccos((THIGH ** 2 + SHANK ** 2 - distance ** 2) / (2 * THIGH * SHANK))))


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


@pytest.mark.parametrize("fps, rpm", [(30, 60), (30, 85), (30, 100), (60, 95)])
def test_strokes_and_angles_are_recovered(fps, rpm):
    p = rider(fps=fps, rpm=rpm, noise=1.0)
    r = cycling.PLUGINS[0].run(PluginContext(p))
    pos = {row[0]: row for row in table(r, "Position").rows}
    ped = {row[0]: row for row in table(r, "Pedaling").rows}
    assert abs(number(ped["Cadence"][1]) - rpm) < 0.5
    assert abs(number(pos["Knee, bottom"][1]) - knee_at(300 + CRANK)) < 1.0
    assert abs(number(pos["Knee, top"][1]) - knee_at(300 - CRANK)) < 1.0
    assert abs(number(pos["Trunk tilt"][1]) - 40) < 0.5
    assert abs(number(pos["Toes down, bottom *"][1]) - 15) < 1.0
    assert number(pos["Knee, bottom"][2].replace("±", "")) < 1.5      # a steady rider varies little
    strokes = table(r, "Stroke by stroke").rows
    assert abs(len(strokes) - 30 * rpm / 60) <= 2
    assert all(row[2] == "OK" for row in table(r, "Recording check").rows), table(r, "Recording check").rows
    assert "Right side" in r.summary


def test_the_side_turned_to_the_camera_is_the_one_measured():
    p = rider(near="left")
    assert A.facing_side(p) == "left"
    templates.apply(p, templates.find("cycling"))
    vertices = {L.NAMES[m.points[1].landmark] for m in p.measurements if m.kind == "angle"}
    assert vertices == {"left_knee", "left_hip", "left_ankle", "left_elbow", "left_shoulder"}
    assert [L.NAMES[m.points[0].landmark] for m in p.measurements if m.kind == "trajectory"] == ["left_ankle"]
    templates.apply(p, templates.find("cycling"))                 # applying it twice adds nothing
    assert len(p.measurements) == 6
    assert "Left side" in cycling.PLUGINS[0].run(PluginContext(p)).summary

    q = rider(near="right")
    templates.apply(q, templates.find("cycling"))
    assert all(L.NAMES[m.points[1].landmark].startswith("right_") for m in q.measurements if m.kind == "angle")
    kick = Project(q.video)                                       # the other templates keep the side they list
    kick.pose = rider(near="left").pose
    templates.apply(kick, templates.find("kick"))
    assert all(L.NAMES[m.points[1].landmark].startswith("right_") for m in kick.measurements if m.kind == "angle")


def test_phases_are_compared():
    """Second half of the clip with the saddle lower: the knee opens less at the bottom."""
    n = 1800
    reach = np.where(np.arange(n) < n // 2, 300.0, 291.0)
    p = rider(seconds=60, reach=reach, noise=1.0)
    p.phases = [Phase("high", 0, n // 2 - 1), Phase("low", n // 2, n - 1)]
    r = cycling.PLUGINS[0].run(PluginContext(p))
    by_phase = {row[0]: row for row in table(r, "By phase").rows}
    # column 3 is knee_bottom (column 0=phase, 1=strokes, 2=cadence, 3=knee_bottom)
    assert abs(number(by_phase["high"][3]) - knee_at(300 + CRANK)) < 1.0
    assert abs(number(by_phase["low"][3]) - knee_at(291 + CRANK)) < 1.0
    compared = next(t for t in r.tables if t.title == "low compared with high")
    rows = {row[0]: row for row in compared.rows}
    expected = knee_at(291 + CRANK) - knee_at(300 + CRANK)
    assert abs(float(rows["Knee, bottom"][3]) - expected) < 1.0
    assert rows["Knee, bottom"][4] == "Yes"
    assert rows["Trunk tilt"][4] == "No" and rows["Cadence"][4] == "No"
    assert all(len(row) == len(t.columns) for t in r.tables for row in t.rows)


def test_what_is_not_side_on_is_left_out():
    """Ten seconds from behind in the middle of the clip: no stroke is taken from there."""
    p = rider(seconds=30)
    a, b = 300, 599
    for side, x in (("left", 380.0), ("right", 620.0)):
        p.pose.xy[a:b + 1, L.INDEX[f"{side}_shoulder"], 0] = x
    assert not A.side_on_frames(p.pose, 30)[a + 15:b - 15].any()
    assert A.side_on_frames(p.pose, 30)[:a - 15].all()
    r = cycling.PLUGINS[0].run(PluginContext(p))
    times = [float(row[1].split()[0]) for row in table(r, "Stroke by stroke").rows]
    assert times and not any(a / 30 <= t <= b / 30 for t in times)
    check = {row[0]: row for row in table(r, "Recording check").rows}
    assert check["Filmed from the side"][2] == "Partial"
    assert "1 point to review" in r.summary


def test_recordings_that_cannot_be_measured_say_why():
    still = rider(seconds=10)
    still.pose.xy[:] = still.pose.xy[0]                           # nobody pedals
    r = cycling.PLUGINS[0].run(PluginContext(still))
    assert "No pedal strokes" in r.summary and [t.title for t in r.tables] == ["Recording check"]

    short = rider(seconds=5)                                      # too few strokes to tell a typical one
    check = {row[0]: row for row in table(cycling.PLUGINS[0].run(PluginContext(short)), "Recording check").rows}
    assert check["Pedal strokes"][2] == "Review"

    moving = rider(seconds=20)                                    # the rider crosses the picture
    moving.pose.xy[..., 0] += np.linspace(0, 400, moving.frame_count)[:, None]
    check = {row[0]: row for row in table(cycling.PLUGINS[0].run(PluginContext(moving)), "Recording check").rows}
    assert check["Rider stays in place"][2] == "Review"

    empty = Project(still.video)                                  # nothing detected at all
    r = cycling.PLUGINS[0].run(PluginContext(empty))
    assert "No pedal strokes" in r.summary and r.tables


def test_hip_travel_in_centimetres_once_calibrated():
    p = rider()
    names = [row[0] for row in table(cycling.PLUGINS[0].run(PluginContext(p)), "Position").rows]
    assert "Hip bounce (% trunk)" in names
    p.calibration = Calibration(px_per_m=500.0, a=(0, 0), b=(500, 0), length_m=1.0)
    names = [row[0] for row in table(cycling.PLUGINS[0].run(PluginContext(p)), "Position").rows]
    assert "Hip bounce (cm)" in names


def test_every_text_has_a_spanish_translation():
    i18n.set_language("es")
    n = 1800
    p = rider(seconds=60, reach=np.where(np.arange(n) < n // 2, 300.0, 291.0))
    p.phases = [Phase("alto", 0, n // 2 - 1), Phase("bajo", n // 2, n - 1)]
    r = cycling.PLUGINS[0].run(PluginContext(p))
    english_words = ("stroke", "Knee", "Hip", "Check", "Yes", "side", "Cadence", "compared",
                     "Trunk", "Toes", "bounce", "Position", "Pedaling", "Stability")
    for t in r.tables:
        for text in [t.title, *t.columns]:
            assert not any(w in text for w in english_words), text
    assert "pedaladas" in r.summary and templates.find("cycling").label == "Ciclismo"


def test_auto_phase_detection():
    """When no phases are set, the plugin proposes one per pedal stroke."""
    n = 1800
    p = rider(seconds=60, rpm=85, noise=1.0)
    assert not p.phases
    r = cycling.PLUGINS[0].run(PluginContext(p))
    strokes = table(r, "Stroke by stroke").rows
    assert len(p.phases) == len(strokes), "should have auto-detected one phase per pedal stroke"
    assert p.phases[0].start < p.phases[1].start
    assert "Phases" in r.summary or "Fases" in r.summary
    # running again with phases already set should NOT overwrite them
    p.phases[0].name = "custom"
    cycling.PLUGINS[0].run(PluginContext(p))
    assert p.phases[0].name == "custom"


def test_pedaling_stability_table():
    """The Pedaling table flags measures whose variation is too high."""
    p = rider(seconds=30, rpm=85, noise=1.0)
    r = cycling.PLUGINS[0].run(PluginContext(p))
    ped = table(r, "Pedaling")
    # a synthetic steady rider should have all OK
    verdicts = {row[0]: row[5] for row in ped.rows}
    assert verdicts["Cadence"] == "OK", verdicts
    assert len(ped.columns) == 6         # Measure, Typical, Variation, Lowest, Highest, Stability


def test_cycling_template_auto_run_and_results_page():
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    from openkick.ui.controller import Controller
    from openkick.ui.results import ResultsPage
    p = rider(seconds=60, rpm=85, noise=1.0)
    p.template = "cycling"
    c = Controller()
    c.set_project(p)
    assert len(p.phases) >= 1
    assert "cycling" in c.plugin_results
    rp = ResultsPage(c)
    rp.show()
    rp.refresh()
    assert rp.plugin_box.count() > 0
    rp.close()
    rp.deleteLater()


def test_cycling_auto_subphases_and_timeline():
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    from openkick.ui.controller import Controller
    from openkick.ui.timeline import Timeline
    p = rider(seconds=30, rpm=85, noise=1.0)
    p.template = "cycling"
    c = Controller()
    c.set_project(p)
    assert len(p.phases) >= 2
    assert len(p.subphases) == 2 * len(p.phases)
    # Check that subphases align with stroke phases
    for i, ph in enumerate(p.phases):
        rec = p.subphases[2 * i]
        prop = p.subphases[2 * i + 1]
        assert rec.start == ph.start
        assert rec.end < prop.start
        assert prop.end == ph.end
    tl = Timeline(c)
    tl.resize(1000, 200)
    rows = tl.rows()
    assert "groups" in rows
    assert "subphases" in rows
    assert rows["subphases"].top() >= rows["groups"].bottom()
    # Trigger paint to ensure no errors
    tl.repaint()
    tl.close()
    tl.deleteLater()
