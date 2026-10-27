"""Patient record, plugin store, pedalling in 3D, comparing analyses, sound sync and resection."""
import numpy as np
import pytest

from openkick import i18n
from openkick.core import compare as K
from openkick.core import cycling as C
from openkick.core.subject import Subject, describe
from openkick.multiview import audio as AU
from openkick.multiview import geometry as G
from openkick.multiview.cycling import analyse, looks_like_pedalling

THIGH = SHANK = 0.44
CRANK = 0.17


@pytest.fixture(autouse=True)
def english():
    before = i18n.language()
    i18n.set_language("en")
    yield
    i18n.set_language(before)


# ── the patient's record ───────────────────────────────────────────────

def test_body_mass_index_and_leg_difference():
    s = Subject(name="A", height_m=1.70, mass_kg=72.25, lengths={"leg_left": 88.0, "leg_right": 89.5, "thigh_left": 44.0})
    assert abs(s.bmi - 25.0) < 1e-9 and s.bmi_class == "Overweight"
    assert Subject(height_m=1.70, mass_kg=60).bmi_class == "Normal weight"
    assert Subject(height_m=1.70, mass_kg=50).bmi_class == "Underweight"
    assert Subject(height_m=1.60, mass_kg=100).bmi_class == "Obesity class II"
    assert Subject(height_m=1.60, mass_kg=110).bmi_class == "Obesity class III"
    assert abs(s.leg_difference() - 1.5) < 1e-9
    assert s.scale_lengths() == {"left_thigh": 0.44}
    back = Subject.from_dict(s.to_dict())
    assert back == s and Subject.from_dict(None).empty and not s.empty
    rows = dict(describe(s))
    assert rows["Body mass index"].startswith("25.0") and "right longer" in rows["Leg length difference"]


def test_the_record_is_kept_in_the_2d_project(tmp_path):
    from fractions import Fraction

    from openkick.core.project import Project
    from openkick.core.video import VideoMeta
    p = Project(VideoMeta(str(tmp_path / "v.mp4"), 640, 480, 30.0, Fraction(1, 30000), [i * 1000 for i in range(30)]))
    p.subject = Subject(name="B", height_m=1.80, mass_kg=80, equipment={"Saddle height (cm)": "75"})
    p.save(tmp_path / "p.okick")
    q = Project.load(tmp_path / "p.okick")
    assert q.subject.name == "B" and q.subject.equipment == {"Saddle height (cm)": "75"}


# ── the plugin store ───────────────────────────────────────────────────

def test_removing_a_package_hides_its_template_and_plugin(tmp_path, monkeypatch):
    import openkick.core.store as ST
    monkeypatch.setattr(ST, "data_dir", lambda: tmp_path)
    from openkick.core import templates
    from openkick.plugins import discover
    assert ST.installed() == {p["id"] for p in ST.CATALOG}            # everything at first
    assert "cycling" in {t.id for t in templates.list_templates()}
    ST.set_installed("cycling", False)
    assert "cycling" not in ST.installed() and not ST.activity_available("cycling")
    assert "cycling" not in {t.id for t in templates.list_templates()}
    assert "cycling" not in {p.id for p in discover()[0]}
    ST.set_installed("base-rom", False)                                # the base cannot be removed
    assert "base-rom" in ST.installed()
    ST.set_installed("cycling", True)
    assert "cycling" in {t.id for t in templates.list_templates()}


# ── pedalling in 3D ────────────────────────────────────────────────────

def rider(seconds=20.0, fps=60.0, rpm=80.0, reach=0.62, toes=10.0) -> tuple[np.ndarray, np.ndarray, list]:
    """A rider in camera 1's axes (x right, y down, z away): the bike points along x, each foot goes
    round a crank circle in a vertical plane, the two half a turn apart. `toes`: how far the sole
    points down (degrees). Returns X (rows, 26, 3) in metres, t and the cameras (level)."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    X = np.full((n, 26, 3), np.nan)
    I = G.IDX
    for side, z, shift in (("left", 3.15, np.pi), ("right", 2.85, 0.0)):
        hip = np.array([0.0, -0.95, z])
        turn = 2 * np.pi * rpm / 60 * t + shift
        centre = hip + np.array([0.15, reach, 0.0])
        ankle = centre + CRANK * np.stack([np.cos(turn), np.sin(turn), np.zeros(n)], axis=1)
        d = np.linalg.norm(ankle - hip, axis=1)
        u = (ankle - hip) / d[:, None]
        fwd = np.stack([-u[:, 1], u[:, 0], np.zeros(n)], axis=1)          # in the plane, ahead of the line
        knee = hip + u * (d / 2)[:, None] + fwd * np.sqrt(THIGH ** 2 - (d / 2) ** 2)[:, None]
        a = np.radians(toes)
        heel = ankle + np.array([-0.06, 0.05, 0.0])
        toe = heel + 0.2 * np.array([np.cos(a), np.sin(a), 0.0])        # +y is down: the toes lower
        shoulder = hip + np.array([0.35, -0.42, 0.0])
        for name, v in (("hip", hip), ("knee", knee), ("ankle", ankle), ("heel", heel), ("toe", toe),
                        ("small_toe", toe), ("shoulder", shoulder), ("elbow", shoulder + [0.2, 0.25, 0]),
                        ("wrist", shoulder + [0.45, 0.3, 0])):
            X[:, I[f"{side}_{name}"]] = v
    X[:, I["head"]] = X[:, I["left_shoulder"]] * 0 + np.array([0.5, -1.6, 3.0])
    cams = [(np.eye(3), np.zeros(3)), (G.around(90)[0], np.zeros(3))]
    return X, t, cams


def knee_at(d):
    return float(np.degrees(np.arccos((THIGH ** 2 + SHANK ** 2 - d ** 2) / (2 * THIGH * SHANK))))


def test_pedal_strokes_and_angles_in_3d():
    X, t, cams = rider()
    c = analyse(X, t, cams)
    assert c is not None and set(c.by_side) == {"left", "right"}
    assert abs(len(c.strokes) - 20 * 80 / 60) <= 2
    dist = np.hypot(0.15, 0.62)
    for side in ("left", "right"):
        cyc = c.by_side[side]
        assert abs(C.typical(s["cadence"] for s in cyc)[0] - 80) < 1
        assert abs(C.typical(s["knee_bottom"] for s in cyc)[0] - knee_at(dist + CRANK)) < 1.0
        assert abs(C.typical(s["knee_top"] for s in cyc)[0] - knee_at(dist - CRANK)) < 1.0
        assert abs(C.typical(s["foot_bottom"] for s in cyc)[0] - 10) < 1.0     # toes down, from the cameras' vertical
        assert C.typical(s["knee_lateral"] for s in cyc)[0] < 0.5               # the knee stays in its plane
        assert C.typical(s["pelvis_rock"] for s in cyc)[0] < 0.5
    assert len(c.subphases) == 2 * len(c.phases)
    assert {p.name for p in c.subphases} == {"Recovery", "Propulsion"}
    assert looks_like_pedalling(X, cams)


def test_walking_is_not_taken_for_pedalling():
    from tests.test_multiview_gait import walking_session
    s = walking_session()
    s.solve()
    assert not looks_like_pedalling(s.X, s.cams)


# ── comparing ──────────────────────────────────────────────────────────

def test_comparing_two_conditions_leg_by_leg():
    def item(label, knee, toes, rows=30):
        cyc = [{"knee_bottom": knee + 0.3 * np.sin(k), "foot_bottom": toes + 0.2 * np.cos(k), "cadence": 70.0,
                "knee_lateral": 2.0} for k in range(rows)]
        return K.Analysis(path=f"{label}.ok3d", label=label, kind="3D", activity="cycling",
                          subject=Subject(height_m=1.70), cycles={"right": cyc})
    a, b = item("93", 140.0, 1.0), item("95", 140.4, 12.0)
    t = K.table([a, b], "right")
    rows = {r[0]: r for r in t.rows}
    assert t.columns == ["Measure", "93", "95", "95 − 93"]
    assert rows["Knee, bottom"][3] == "+0.4"                       # within the variation: no dot
    assert rows["Toes down, bottom *"][3].startswith("+11.0") and "●" in rows["Toes down, bottom *"][3]
    per = {r[0]: r for r in K.table([a, b], "right", per_height=True).rows}
    assert "% height" in per["Knee side-to-side travel"][1]        # lengths against the height
    three = K.table([a, b, item("97", 141.0, 20.0)], "right")
    assert len(three.columns) == 6                                  # more than two analyses


def test_curves_are_averaged_over_the_cycles():
    series = np.tile(np.concatenate([np.linspace(140, 70, 20), np.linspace(70, 140, 20)]), 5)
    cycles = [{"first": 40 * k, "last": 40 * k + 39} for k in range(4)]
    mean, sd, n = K._resample(series, cycles)
    assert n == 4 and len(mean) == K.SAMPLES and sd.max() < 2.0 and mean.min() < 75


# ── 3D: scale, sound, resection ────────────────────────────────────────

def test_measured_lengths_give_the_scale():
    from tests.test_multiview import make
    s, _ = make()
    s.solve()
    before = s.scale
    thigh = np.nanmedian(np.linalg.norm(s.X[:, G.IDX["left_hip"]] - s.X[:, G.IDX["left_knee"]], axis=1))
    s.subject = Subject(height_m=s.height_m, lengths={"thigh_left": 2 * thigh * 100})   # a tape says twice as long
    s.update(recalibrate=False)
    assert abs(s.scale / before - 2.0) < 0.05


def test_sound_lines_up_two_recordings():
    rng = np.random.default_rng(0)
    room = rng.normal(0, 1, AU.RATE * 20)
    claps = np.zeros_like(room)
    claps[rng.integers(0, len(room), 30)] = 25.0
    room += claps
    lag = 1.2345
    k = int(round(lag * AU.RATE))
    a = room[k:k + AU.RATE * 12]                     # phone A started recording 1.2345 s later than phone B
    b = room[:AU.RATE * 14] + rng.normal(0, 0.5, AU.RATE * 14)
    found, sure = AU.lag(a, b)
    assert abs(found - lag) < 1.5 / AU.RATE and sure >= AU.SURE
    _, unsure = AU.lag(rng.normal(0, 1, AU.RATE * 5), rng.normal(0, 1, AU.RATE * 5))
    assert unsure < AU.SURE


def test_a_third_camera_is_placed_from_the_first_ones_3d():
    rng = np.random.default_rng(1)
    X = rng.uniform([-0.5, -1.0, 3.0], [0.5, 1.0, 4.0], (400, 3))
    R, _ = G.around(70)
    t = np.array([0.3, -0.1, 0.5])
    x = G.reproject(X, R, t) + rng.normal(0, 0.5 / 1000, (400, 2))
    found = G.resect(X, x, np.ones(400, bool), focal=1000.0)
    assert found is not None
    Rf, tf = found
    assert np.degrees(np.arccos(np.clip((np.trace(Rf @ R.T) - 1) / 2, -1, 1))) < 0.5 and np.allclose(tf, t, atol=0.02)


# ── smoothing the 3D ───────────────────────────────────────────────────

def test_smoothing_takes_away_one_frame_jumps_and_keeps_the_movement():
    fps = 60.0
    t = np.arange(600) / fps
    truth = np.zeros((600, 26, 3))
    truth[:, :, 0] = 0.2 * np.sin(2 * np.pi * 1.2 * t)[:, None]          # a pedal stroke's pace, 20 cm
    X = truth + np.random.default_rng(0).normal(0, 0.004, truth.shape)   # 4 mm of tremor
    X[200, 14] += [0.0, 0.25, 0.0]                                       # a knee off by 25 cm for one frame
    X[300:302, 21] += [0.15, 0.0, 0.0]                                   # a toe for two frames
    X[400, 3] = np.nan                                                   # a point not reconstructed
    Y = G.smooth(X, fps, 8.0)
    assert np.linalg.norm(Y[200, 14] - truth[200, 14]) < 0.01
    assert np.linalg.norm(Y[300:302, 21] - truth[300:302, 21], axis=1).max() < 0.01
    ok = np.isfinite(Y).all(axis=2)
    assert not ok[400, 3] and ok.sum() == ok.size - 1
    assert abs(np.ptp(Y[50:550, 0, 0]) - np.ptp(truth[50:550, 0, 0])) < 0.01   # the movement keeps its size
    assert np.std(Y[50:550, 7] - truth[50:550, 7]) < np.std(X[50:550, 7] - truth[50:550, 7]) / 2


def test_smoothing_is_kept_with_the_session(tmp_path):
    from openkick.multiview.session import FAST_SMOOTH_HZ, SMOOTH_HZ, Session, smoothing_for
    from tests.test_multiview import make
    assert smoothing_for("kick") == FAST_SMOOTH_HZ and smoothing_for("cycling") == smoothing_for(None) == SMOOTH_HZ
    s, _ = make()
    s.solve()
    assert s.smooth_hz == SMOOTH_HZ and not np.allclose(s.X, s.X_raw, equal_nan=True)
    s.smooth_hz = 0.0
    s.update(recalibrate=False)
    assert np.array_equal(s.X, s.X_raw, equal_nan=True)                  # off: the points as triangulated
    s.smooth_hz = 12.0
    s.save(tmp_path / "s.ok3d")
    assert Session.load(tmp_path / "s.ok3d", require_videos=False).smooth_hz == 12.0


def test_smoothing_from_the_window():
    from tests.test_multiview import _app, _on_real_file, make
    app = _app()
    from openkick.ui.multiview import SMOOTHING, MultiviewWindow
    s, _ = make()
    _on_real_file(s)
    s.solve()
    w = MultiviewWindow()
    w.show()
    w.set_session(s)
    app.processEvents()
    box = w.panel.smoothing
    assert box.currentData() == s.smooth_hz
    box.setCurrentIndex([hz for hz, _ in SMOOTHING].index(0.0))
    assert s.smooth_hz == 0.0 and np.array_equal(s.X, s.X_raw, equal_nan=True) and w.ctl.dirty
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()
