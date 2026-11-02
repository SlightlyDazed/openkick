"""Clinical joint angles (core/joints.py), how projects use them, and the speed clean-up."""
from fractions import Fraction

import numpy as np
import pytest

from openkick.core import analysis as A
from openkick.core import joints as J
from openkick.core import landmarks as L
from openkick.core import templates
from openkick.core.project import SRC_AUTO, Measurement, PointRef, Project
from openkick.core.video import VideoMeta


def pose(n=1, forward=1.0, thigh=0.0, knee=0.0, foot=0.0, lean=0.0):
    """(n, 33, 2) image points of one standing side-on body; angles in degrees, + = towards the
    front: the thigh from the vertical, the knee bent, the foot turned up from square, the trunk
    leaning forward."""
    xy = np.full((n, L.COUNT, 2), np.nan)
    hip = np.array([500.0, 400.0])

    def towards(start, deg, length):         # deg from straight down, + = forward
        r = np.radians(deg)
        return start + length * np.array([forward * np.sin(r), np.cos(r)])
    k = towards(hip, thigh, 230)
    a = towards(k, thigh - knee, 230)
    shank = np.degrees(np.arctan2(forward * (a - k)[0], (a - k)[1]))
    heel = a + np.array([-forward * 25.0, 20.0])
    # the sole square to the shank, turned up by `foot`
    toe = heel + 150 * np.array([forward * np.cos(np.radians(shank + foot)), -np.sin(np.radians(shank + foot))])
    sh = hip + 280 * np.array([forward * np.sin(np.radians(lean)), -np.cos(np.radians(lean))])
    elbow = sh + np.array([0.0, 140.0])
    for side in ("left", "right"):
        for name, v in (("hip", hip), ("knee", k), ("ankle", a), ("heel", heel), ("foot_index", toe),
                        ("shoulder", sh), ("elbow", elbow), ("wrist", elbow + np.array([0.0, 120.0]))):
            xy[:, L.INDEX[f"{side}_{name}"]] = v
    return xy


@pytest.mark.parametrize("forward", [1.0, -1.0])
def test_signed_angles_read_from_zero(forward):
    F = np.full(1, forward)
    assert J.clinical(pose(forward=forward), "knee", "right", F)[0] == pytest.approx(0, abs=1e-6)
    assert J.clinical(pose(forward=forward, knee=40), "knee", "right", F)[0] == pytest.approx(40, abs=1e-6)
    assert J.clinical(pose(forward=forward, thigh=30), "hip", "right", F, "vertical")[0] == pytest.approx(30, abs=1e-6)
    # a thigh behind the body is extension: negative, where the angle inside the joint read 170 either way
    behind = pose(forward=forward, thigh=-10)
    assert J.clinical(behind, "hip", "right", F, "vertical")[0] == pytest.approx(-10, abs=1e-6)
    inside = A.angle_between(behind[:, L.INDEX["right_shoulder"]], behind[:, L.INDEX["right_hip"]],
                             behind[:, L.INDEX["right_knee"]])
    ahead = pose(forward=forward, thigh=10)
    assert inside[0] == pytest.approx(A.angle_between(ahead[:, L.INDEX["right_shoulder"]], ahead[:, L.INDEX["right_hip"]],
                                                      ahead[:, L.INDEX["right_knee"]])[0], abs=1e-6)
    assert J.clinical(pose(forward=forward), "ankle", "right", F)[0] == pytest.approx(0, abs=1e-6)
    assert J.clinical(pose(forward=forward, foot=10), "ankle", "right", F)[0] == pytest.approx(10, abs=1e-6)


def test_hip_against_the_pelvis_takes_three_quarters_of_the_lean():
    F = np.ones(1)
    xy = pose(lean=20)              # bent forward at the hips, thigh vertical
    assert J.trunk_lean(xy, F)[0] == pytest.approx(20, abs=1e-6)
    assert J.clinical(xy, "hip", "right", F, "vertical")[0] == pytest.approx(0, abs=1e-6)
    assert J.clinical(xy, "hip", "right", F, "pelvis")[0] == pytest.approx(15, abs=1e-6)


def test_facing_follows_a_turn():
    xy = np.concatenate([pose(40, forward=1.0), pose(40, forward=-1.0)])
    F = J.facing(xy, fps=30)
    assert (F[:30] == 1).all() and (F[50:] == -1).all()


def test_joint_of_knows_the_standard_angles_only():
    I = L.INDEX
    assert J.joint_of(I["right_hip"], I["right_knee"], I["right_ankle"]) == ("knee", "right")
    assert J.joint_of(I["left_ankle"], I["left_knee"], I["left_hip"]) == ("knee", "left")
    assert J.joint_of(I["left_knee"], I["left_ankle"], I["left_foot_index"]) == ("ankle", "left")
    assert J.joint_of(I["right_wrist"], I["right_knee"], I["right_ankle"]) is None


def project(xy, template=None) -> Project:
    n = len(xy)
    p = Project(VideoMeta("v.mp4", 1000, 900, 30.0, Fraction(1, 30000), [i * 1000 for i in range(n)]))
    p.pose.xy[:] = xy
    p.pose.raw_xy[:] = xy
    p.pose.visibility[:] = 0.95
    p.pose.source[:] = SRC_AUTO
    if template:
        templates.apply(p, templates.find(template))
    return p


def angle(p, a, o, b):
    m = Measurement("angle", "x", [PointRef.at(L.INDEX[a]), PointRef.at(L.INDEX[o]), PointRef.at(L.INDEX[b])])
    p.measurements.append(m)
    return A.series(p, m).value


def test_the_template_decides_the_convention():
    xy = pose(5, thigh=-10, knee=20)
    gait = project(xy, "gait")
    assert A.convention(gait) == ("clinical", "vertical")
    assert angle(gait, "right_hip", "right_knee", "right_ankle")[0] == pytest.approx(20, abs=1e-3)
    assert angle(gait, "right_shoulder", "right_hip", "right_knee")[0] == pytest.approx(-10, abs=1e-3)
    bike = project(xy, "cycling")
    assert A.convention(bike) == ("interior", "pelvis")
    assert angle(bike, "right_hip", "right_knee", "right_ankle")[0] == pytest.approx(160, abs=1e-3)
    none = project(xy)
    assert A.convention(none) == ("clinical", "pelvis")
    # an angle of the user's own is always the one inside the joint
    assert angle(none, "right_wrist", "right_knee", "right_ankle")[0] > 90


def test_template_conventions_survive_a_save(tmp_path, monkeypatch):
    monkeypatch.setattr(templates, "user_dir", lambda: tmp_path)
    t = templates.Template(id="mine", name="Mine", convention="interior", hip_reference="vertical")
    templates.save_user_template(t)
    back = templates.find("mine")
    assert (back.convention, back.hip_reference) == ("interior", "vertical")


def test_a_detector_jump_does_not_make_a_speed_peak():
    fps, n = 30.0, 90
    t = np.arange(n) / fps
    real = 40 * np.sin(2 * np.pi * t)                   # 1 Hz, up to 250 °/s
    jumped = real.copy()
    jumped[45] += 60
    clean = A.despike(jumped, fps, A.DESPIKE_FLOOR_DEG)
    assert np.isnan(clean[45]) and np.isfinite(np.delete(clean, 45)).all()
    assert np.isfinite(A.despike(real, fps, A.DESPIKE_FLOOR_DEG)).all()       # the real movement is kept
    rate = np.gradient(A.lowpass(clean, fps), t)
    assert np.nanmax(np.abs(rate)) < 300


def test_trajectory_speed_skips_a_jump():
    xy = pose(60)
    xy[:, :, 0] += np.arange(60)[:, None] * 3.0         # walking along at 90 px/s
    xy[30, L.INDEX["right_ankle"]] += np.array([80.0, 0.0])
    p = project(xy)
    m = Measurement("trajectory", "ankle", [PointRef.at(L.INDEX["right_ankle"])])
    p.measurements.append(m)
    speed = A.series(p, m).value
    assert np.nanmax(speed[5:-5]) < 130


def test_clinical_3d_angles():
    from openkick.multiview.geometry import HALPE26, IDX
    X = np.full((1, len(HALPE26), 3), np.nan)
    up = np.array([0.0, -1.0, 0.0])                     # camera axes: y down
    fwd = np.array([0.0, 0.0, -1.0])                    # facing the camera
    for side, x in (("right", -0.1), ("left", 0.1)):
        hip = np.array([x, 0.0, 3.0])
        r = np.radians(-10)                             # thigh 10° behind
        knee = hip + 0.45 * (np.sin(r) * fwd - np.cos(r) * up)
        ankle = knee - 0.45 * up
        heel = ankle - 0.06 * up - 0.05 * fwd
        for name, v in (("hip", hip), ("knee", knee), ("ankle", ankle), ("heel", heel), ("toe", heel + 0.22 * fwd),
                        ("shoulder", hip + 0.5 * up), ("elbow", hip + 0.22 * up), ("wrist", hip)):
            X[0, IDX[f"{side}_{name}"]] = v
    assert J.clinical_3d(X, IDX, "hip", "right", up, "vertical")[0] == pytest.approx(-10, abs=1e-6)
    assert J.clinical_3d(X, IDX, "knee", "left", up)[0] == pytest.approx(10, abs=1e-6)
    assert J.clinical_3d(X, IDX, "ankle", "right", up)[0] == pytest.approx(0, abs=1e-6)


def test_toe_off_delay_moves_only_the_toe_off():
    from openkick.core import gait as G
    t = np.arange(100) / 60.0
    contacts, offs = [(10, 10.0), (76, 76.0)], [(50, 50.0)]
    plain = G.pair(contacts, offs, t, np.ones(100, bool))[0]
    late = G.pair(contacts, offs, t, np.ones(100, bool), toe_off_delay=0.025)[0]
    assert (plain["a"], plain["b"]) == (late["a"], late["b"])
    assert late["fo"] - plain["fo"] == pytest.approx(1.5)
    assert late["stance"] > plain["stance"]


def test_shoulder_follows_the_hip_reference():
    F = np.ones(1)
    xy = pose(lean=20)                                  # the arm hangs straight down, the trunk leans
    assert J.clinical(xy, "shoulder", "right", F, "vertical")[0] == pytest.approx(0, abs=1e-6)
    assert J.clinical(xy, "shoulder", "right", F, "pelvis")[0] == pytest.approx(20, abs=1e-6)
