import numpy as np

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core import templates
from openkick.core.project import SRC_MANUAL, Measurement, PointRef, Project


def test_probe_uses_real_timestamps(meta):
    assert meta.frame_count == 80
    assert meta.pts[:4] == [0, 2291, 5614, 8613]   # variable frame rate
    assert meta.ticks_per_frame == 3013


def test_angle_between_right_angle():
    a = np.array([[1.0, 0.0]])
    o = np.array([[0.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert np.isclose(A.angle_between(a, o, b)[0], 90.0)


def test_template_creates_attached_measurements(project):
    t = templates.find("kick")
    templates.apply(project, t)
    kinds = [m.kind for m in project.measurements]
    assert kinds.count("angle") == 2 and kinds.count("trajectory") == 1
    knee = project.measurements[0]
    assert [r.landmark for r in knee.points] == [L.INDEX["right_hip"], L.INDEX["right_knee"], L.INDEX["right_ankle"]]
    templates.apply(project, t)                      # applying twice does not duplicate
    assert len(project.measurements) == 3


def test_editing_a_joint_updates_its_angle(project):
    templates.apply(project, templates.find("kick"))
    knee = project.measurements[0]
    before = A.value_at(project, knee, 10)
    hip = project.pose.point(10, L.INDEX["right_hip"])
    edit = A.edit_point(project.pose, 10, L.INDEX["right_knee"], (hip[0] + 5, hip[1] + 80))
    project.touch()
    after = A.value_at(project, knee, 10)
    assert after != before
    assert project.pose.source[10, L.INDEX["right_knee"]] == SRC_MANUAL
    edit.undo(project.pose)
    project.touch()
    assert np.isclose(A.value_at(project, knee, 10), before)


def test_neighbor_scope_spreads_and_protects_manual(project):
    k = L.INDEX["right_knee"]
    A.edit_point(project.pose, 12, k, (1.0, 1.0))           # manual at 12
    orig14 = project.pose.xy[14, k].copy()
    A.edit_point(project.pose, 10, k, tuple(project.pose.xy[10, k] + 30), scope="neighbors", radius=3)
    assert tuple(project.pose.xy[12, k]) == (1.0, 1.0)      # hand correction untouched
    assert not np.allclose(project.pose.xy[11, k], project.pose.raw_xy[11, k])
    assert np.allclose(project.pose.xy[14, k], orig14)      # outside the radius


def test_phases_are_contiguous(project):
    project.add_phase_at(0, "A")
    project.add_phase_at(30, "B")
    project.add_phase_at(50, "C")
    a, b, c = project.phases
    assert (a.start, a.end, b.start, b.end, c.start, c.end) == (0, 29, 30, 49, 50, 79)
    project.add_phase_at(40, "B2")
    assert project.phase_at(45).name == "B2" and project.phase_at(39).name == "B"


def test_speed_uses_calibration(project):
    from openkick.core.project import Calibration
    m = Measurement(kind="trajectory", name="t", points=[PointRef.at(L.INDEX["right_ankle"])])
    project.measurements.append(m)
    px = A.series(project, m)
    assert px.unit == "px/s"
    project.calibration = Calibration(px_per_m=200.0, a=(0, 0), b=(200, 0), length_m=1.0)
    project.touch()
    ms = A.series(project, m)
    assert ms.unit == "m/s"
    assert np.allclose(ms.value[5:-5], px.value[5:-5] / 200.0, equal_nan=True)


def test_project_roundtrip(project, tmp_path):
    templates.apply(project, templates.find("kick"))
    project.add_phase_at(20, "Golpeo")
    project.measurements.append(Measurement(kind="distance", name="d",
                                            points=[PointRef.free(0, 10, 10), PointRef.at(0)]))
    f = tmp_path / "p.okick"
    project.save(f)
    q = Project.load(f)
    assert q.frame_count == project.frame_count
    assert np.allclose(q.pose.xy, project.pose.xy, atol=0.01, equal_nan=True)
    assert [m.name for m in q.measurements] == [m.name for m in project.measurements]
    assert q.measurements[-1].points[0].keys == {0: (10.0, 10.0)}
    assert q.phases[0].name == "Golpeo"


def test_builtin_plugins_run(project, tmp_path):
    from openkick.plugins import PluginContext, discover
    templates.apply(project, templates.find("kick"))
    project.add_phase_at(0, "A")
    project.add_phase_at(40, "B")
    plugins, errors = discover()
    assert {p.id for p in plugins} >= {"rom_report", "velocities", "opensim_trc"}
    ctx = PluginContext(project)
    for p in plugins:
        if p.kind == "analysis":
            r = p.run(ctx)
            assert r.tables and all(len(row) == len(t.columns) for t in r.tables for row in t.rows)
        else:
            out = tmp_path / f"x{p.suffix}"
            p.export(ctx, out)
            lines = out.read_text().splitlines()
            assert lines[0].startswith("PathFileType") and len(lines) == 6 + project.frame_count
