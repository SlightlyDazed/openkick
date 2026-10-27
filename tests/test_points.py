import numpy as np

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core.bodymodel import center_of_mass, nearest_segment
from openkick.core.project import Measurement, PointRef, Project, TextNote

K, H, AN = L.INDEX["right_knee"], L.INDEX["right_hip"], L.INDEX["right_ankle"]


def test_segment_point_follows_the_bone(project):
    f = 10
    pa, pb = project.pose.point(f, H), project.pose.point(f, K)
    p = (pa[0] + 0.4 * (pb[0] - pa[0]) + 7, pa[1] + 0.4 * (pb[1] - pa[1]) - 3)
    ref = PointRef.on_segment(H, K, pa, pb, p)
    assert np.allclose(project.point(ref, f), p, atol=1e-3)
    # move the knee: the attached point moves with the thigh
    project.pose.xy[f + 1, K] += (30, 10)
    q = project.point(ref, f + 1)
    arr = A._points(project, ref, project.frame_count)
    assert np.allclose(arr[f + 1], q) and not np.allclose(q, p, atol=1)


def test_center_of_mass_is_inside_the_body(project):
    com = center_of_mass(project.pose.xy[0].astype(np.float64))
    xy = project.pose.xy[0]
    assert np.all(com >= np.nanmin(xy, axis=0)) and np.all(com <= np.nanmax(xy, axis=0))
    ref = PointRef(derived="com:f")
    assert project.point(ref, 3) is not None
    assert np.isfinite(A._points(project, ref, project.frame_count)).all()


def test_marker_used_by_an_angle_propagates(project):
    marker = Measurement(kind="point", name="Trocánter", points=[PointRef.free(0, 400, 250)])
    project.measurements.append(marker)
    ang = Measurement(kind="angle", name="a", points=[PointRef(marker=marker.id), PointRef.at(K), PointRef.at(AN)])
    project.measurements.append(ang)
    v0 = A.value_at(project, ang, 5)
    marker.points[0].keys[0] = (300, 100)
    project.touch()
    assert A.value_at(project, ang, 5) != v0
    assert project.dependents(marker.id) == [ang]
    q = Project.from_dict if hasattr(Project, "from_dict") else None
    assert q is None or q


def test_nearest_segment():
    xy = np.full((33, 2), np.nan)
    xy[H] = (0, 0)
    xy[K] = (0, 100)
    a, b, d = nearest_segment(xy, (5, 50))
    assert (a, b) == (H, K) and abs(d - 5) < 1e-6


def test_note_fades(meta):
    n = TextNote("x", 0, 0, start=10, end=20, fade_in=0.2, fade_out=0.3)
    assert n.opacity(meta, 9) == 0
    assert 0 < n.opacity(meta, 11) < 1          # fading in
    assert n.opacity(meta, 18) == 1
    assert 0 < n.opacity(meta, 22) < 1          # fading out after the end
    assert n.opacity(meta, 40) == 0
    hard = TextNote("x", 0, 0, start=10, end=20)
    assert hard.opacity(meta, 20) == 1 and hard.opacity(meta, 21) == 0


def test_points_roundtrip(project, tmp_path):
    mk = Measurement(kind="point", name="m", points=[PointRef(segment=(H, K, 0.4, 0.05))])
    project.measurements += [mk, Measurement(kind="trajectory", name="t", points=[PointRef(marker=mk.id)]),
                             Measurement(kind="point", name="c", points=[PointRef(derived="com:m")])]
    project.notes.append(TextNote("hola", 1, 2, 3, 4, fade_out=0.5))
    project.save(tmp_path / "p.okick")
    q = Project.load(tmp_path / "p.okick")
    assert [r.kind for m in q.measurements for r in m.points] == ["segment", "marker", "derived"]
    assert q.notes[0].fade_out == 0.5
    assert np.allclose(A._points(q, q.measurements[1].points[0], q.frame_count),
                       A._points(project, project.measurements[1].points[0], project.frame_count), atol=0.05, equal_nan=True)


def test_optical_flow_tracks_a_static_textured_point(meta):
    from openkick.core.tracking import track_point
    pos, lost = track_point(meta, 0, (200.0, 440.0), stop_at=15)
    assert lost is None and len(pos) == 16
    d = np.linalg.norm(np.array(pos[15]) - np.array(pos[0]))
    assert d < 6       # the camera is fixed: grass does not move
