import xml.etree.ElementTree as ET

import numpy as np
import pytest

from openkick.core import analysis as A
from openkick.core import templates
from openkick.core.project import Project
from openkick.formats import kva

from .conftest import KINOVEA_KVA


def _root(path):
    return ET.fromstring(path.read_bytes()[3:])


def test_export_uses_kinovea_2023_structure(project, tmp_path):
    templates.apply(project, templates.find("kick"))
    project.add_phase_at(0, "Carrera")
    project.add_phase_at(40, "Golpeo")
    out = tmp_path / "x.kva"
    summary = kva.export(project, out)
    assert summary == {"angles": 2, "trajectories": 1, "skipped": [], "keyframes": 80}
    root = _root(out)
    ts = [int(k.findtext("Timestamp")) for k in root.iter("Keyframe")]
    assert ts[:3] == [0, 2291, 5614]                        # real pts, not frame * 3013
    angles = list(root.iter("Angle"))
    assert len(angles) == 2 and not list(root.iter("DrawingAngle"))
    td = {t.get("id"): t for t in root.iter("TrackableDrawing")}
    assert set(td) == {a.get("id") for a in angles}
    for t in td.values():
        assert [p.get("key") for p in t.iter("TrackablePoint")] == ["o", "a", "b"]
        assert len(list(t.iter("Frame"))) == 3 * 80
    tp = next(root.iter("TrackPoint"))
    assert tp.text.count(";") == 2 and tp.get("x") is None  # "x;y;t" text form
    names = [k.findtext("Name") for k in root.iter("Keyframe")]
    assert names[0] == "Carrera" and names[40] == "Golpeo"


def test_export_import_roundtrip(project, tmp_path):
    templates.apply(project, templates.find("kick"))
    project.add_phase_at(10, "Apoyo")
    out = tmp_path / "x.kva"
    kva.export(project, out)
    q = Project(project.video)
    notes = kva.import_into(q, out)
    assert "skeleton:80" in notes and "angles:2" in notes and "trajectories:1" in notes
    knee_p, knee_q = project.measurements[0], q.measurements[0]
    for f in (0, 33, 79):
        assert abs(A.value_at(project, knee_p, f) - A.value_at(q, knee_q, f)) < 0.2
    assert q.phases[0].name == "Apoyo" and q.phases[0].start == 10


def test_import_file_saved_by_kinovea(meta):
    if not KINOVEA_KVA.exists():
        pytest.skip("Kinovea sample not available")
    p = Project(meta)
    notes = kva.import_into(p, KINOVEA_KVA)
    assert "angles:1" in notes and "trajectories:1" in notes
    angle = next(m for m in p.measurements if m.kind == "angle")
    v = A.value_at(p, angle, 0)
    assert v is not None and 0 < v < 180
    assert np.isfinite(p.pose.xy[5, 26]).all()             # right knee from the OpenPose posture
