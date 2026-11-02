"""Measures checked against how the literature defines them (review of 1 November 2026)."""
from fractions import Fraction

import numpy as np
import pytest

from openkick import i18n
from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core import templates
from openkick.core.project import SRC_AUTO, Measurement, PointRef, Project
from openkick.core.subject import Subject
from openkick.core.video import VideoMeta


@pytest.fixture(autouse=True)
def english():
    before = i18n.language()
    i18n.set_language("en")
    yield
    i18n.set_language(before)


def project(n=120, fps=60.0, template=None) -> Project:
    p = Project(VideoMeta("v.mp4", 1000, 900, fps, Fraction(1, 60000), [round(i * 60000 / fps) for i in range(n)]))
    p.pose.visibility[:] = 0.95
    p.pose.source[:] = SRC_AUTO
    if template:
        p.template = template
    return p


def test_fast_templates_keep_more_of_the_signal():
    assert A.cutoff(project()) == A.CUTOFF_HZ == 6.0
    assert A.cutoff(project(template="gait")) == 6.0
    assert A.cutoff(project(template="jump")) == 12.0
    assert A.cutoff(project(template="kick")) == 15.0
    assert A.cutoff(project(fps=30.0, template="kick")) == pytest.approx(13.5)      # below Nyquist


def test_acceleration_is_that_of_the_position_not_of_the_speed():
    """A point going round a circle at a steady speed: its speed does not change, but it
    accelerates towards the centre (v² / r)."""
    fps, n, r, turns = 60.0, 240, 200.0, 1.0           # one turn a second
    p = project(n=n, fps=fps)
    t = np.arange(n) / fps
    a = 2 * np.pi * turns * t
    p.pose.xy[:, L.INDEX["right_ankle"]] = np.stack([500 + r * np.cos(a), 450 + r * np.sin(a)], 1)
    p.pose.raw_xy[:] = p.pose.xy
    m = Measurement("trajectory", "ankle", [PointRef.at(L.INDEX["right_ankle"])])
    p.measurements.append(m)
    s = A.series(p, m)
    v = 2 * np.pi * r * turns
    mid = slice(40, -40)
    assert np.nanmedian(s.value[mid]) == pytest.approx(v, rel=0.03)
    assert np.nanmedian(s.accel[mid]) == pytest.approx(v ** 2 / r, rel=0.05)


def test_velocities_plugin_warns_when_the_video_is_too_slow_for_the_movement():
    from openkick.plugins import PluginContext
    from openkick.plugins.builtin import velocities
    p = project(n=90, fps=30.0)
    xy = np.random.default_rng(0).uniform(300, 600, (90, L.COUNT, 2))
    p.pose.xy[:] = xy
    p.pose.raw_xy[:] = xy
    templates.apply(p, templates.find("kick"))
    r = velocities.PLUGINS[0].run(PluginContext(p))
    assert "too fast for 30 frames per second" in r.summary and "below 14 Hz" in r.summary


def test_bmi_classes_are_for_adults():
    adult = Subject(birth="1990-01-01", height_m=1.70, mass_kg=80)
    assert adult.bmi_class == "Overweight"
    child = Subject(birth=f"{__import__('datetime').date.today().year - 12}-01-01", height_m=1.50, mass_kg=50)
    assert child.bmi is not None and child.bmi_class is None


def test_3d_scale_of_someone_who_never_stands_up():
    from openkick.multiview import geometry as G
    from openkick.multiview.geometry import HALPE26, IDX
    H = 1.70
    X = np.zeros((50, len(HALPE26), 3))
    for s, x in (("left", -0.1), ("right", 0.1)):
        X[:, IDX[f"{s}_hip"]] = (x, 0.0, 0.0)
        X[:, IDX[f"{s}_knee"]] = (x, 0.40, 0.10)      # thigh and shank in reconstruction units
        X[:, IDX[f"{s}_ankle"]] = (x, 0.795, 0.10)
        X[:, IDX[f"{s}_heel"]] = (x, 0.85, 0.12)
    legs = G.leg_length(X)
    standing = X.copy()
    standing[:, IDX["head"]] = (0.0, 0.85 - legs / G.LEGS_OF_HEIGHT, 0.0)       # head-heel = the height
    tall = np.linalg.norm(standing[0, IDX["head"]] - (standing[0, IDX["left_heel"]] + standing[0, IDX["right_heel"]]) / 2)
    assert G.scale_from_height(standing, H) == pytest.approx(H / tall, rel=1e-6)            # from the height
    seated = X.copy()
    seated[:, IDX["head"]] = (0.0, 0.85 - 0.8 * legs / G.LEGS_OF_HEIGHT, 0.0)   # never taller than 80 %
    assert G.scale_from_height(seated, H) == pytest.approx(G.LEGS_OF_HEIGHT * H / legs, rel=1e-6)


def test_trc_height_share():
    from openkick.plugins.builtin import opensim_trc
    assert opensim_trc.NOSE_TO_ANKLE == pytest.approx(0.876)
