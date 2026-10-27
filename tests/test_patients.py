"""The patients kept on this computer, and picking one again in another analysis."""
import os

import pytest

from openkick import i18n
from openkick.core import patients as P
from openkick.core.subject import Subject
from tests.test_gui import qapp  # noqa: F401  (the same application for every window)


@pytest.fixture(autouse=True)
def english():
    before = i18n.language()
    i18n.set_language("en")
    yield
    i18n.set_language(before)


def test_remembering_patients(tmp_path):
    assert P.every() == [] and P.remember(Subject()) == Subject()            # an empty record is not kept
    a = P.remember(Subject(name="Sami", height_m=1.68, mass_kg=94.0, lengths={"thigh_left": 43.5}))
    assert a.id and P.get(a.id)["subject"] == a
    b = P.remember(Subject(name="Hilary", height_m=1.60))
    assert {p["id"] for p in P.every()} == {a.id, b.id} and P.every()[0]["id"] in (a.id, b.id)
    a.mass_kg = 92.0
    again = P.remember(a)                                                     # the same patient, brought up to date
    assert again.id == a.id and len(P.every()) == 2 and P.get(a.id)["subject"].mass_kg == 92.0
    P.note_analysis(again, tmp_path / "93.ok3d", "3D")
    P.note_analysis(again, tmp_path / "93.ok3d", "3D")                       # saved twice: listed once
    P.note_analysis(again, tmp_path / "95.ok3d", "3D")
    assert [x["path"] for x in P.get(a.id)["analyses"]] == [str((tmp_path / f"{n}.ok3d").resolve()) for n in (93, 95)]
    P.note_analysis(Subject(name="no id"), tmp_path / "x.okick", "2D")       # not on the list: nothing happens
    P.forget(b.id)
    assert [p["id"] for p in P.every()] == [a.id]
    P._file().write_text("{ not json", encoding="utf-8")
    assert P.every() == []                                                    # a broken file is an empty list


def test_picking_a_saved_patient_in_the_record(monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from openkick.ui import subject as U
    saved = P.remember(Subject(name="Sami", height_m=1.68, mass_kg=94.0, lengths={"thigh_left": 43.5},
                               equipment={"Saddle height (cm)": "93"}))
    d = U.SubjectDialog(None, "cycling")
    assert d.saved.count() == 2 and d.subject().empty
    d.saved.setCurrentIndex(d.saved.findData(saved.id))
    d._pick()
    got = d.subject()
    assert got.id == saved.id and got.name == "Sami" and got.lengths == {"thigh_left": 43.5}
    assert got.equipment == {"Saddle height (cm)": "93"} and "Crank length (mm)" in d.equipment
    d.mass.setValue(92.0)                                                     # today's weight
    d.saved.setCurrentIndex(0)
    d._pick()
    assert d.subject().empty and not d.subject().id                           # a new patient: an empty form
    d.close()
    # a record saved before the list, with the same name: linked, and nothing in it changes
    old = U.SubjectDialog(Subject(name="sami", mass_kg=93.0, equipment={"Saddle height (cm)": "95"}), "cycling")
    assert old.subject().id == saved.id and old.subject().mass_kg == 93.0
    assert old.subject().equipment == {"Saddle height (cm)": "95"}
    old.close()
    # saved with "keep it": the list has the new record, with its id
    monkeypatch.setattr(U.SubjectDialog, "exec", lambda self: (self.set_subject(Subject(name="New one")), 1)[1])
    s = U.edit_subject(None, None)
    assert s.id and P.get(s.id)["subject"].name == "New one"
    assert app is not None


def test_patients_window(tmp_path, monkeypatch):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from openkick.ui.patients import PatientsWindow
    a = P.remember(Subject(name="Sami", birth="2000-01-01", height_m=1.68, mass_kg=94.0))
    P.remember(Subject(name="Hilary", height_m=1.60))
    for n in (93, 95):
        (tmp_path / f"{n}.ok3d").write_text("{}")
        P.note_analysis(a, tmp_path / f"{n}.ok3d", "3D")
    opened, compared = [], []
    w = PatientsWindow(open_path=opened.append, compare=compared.append)
    w.show()
    app.processEvents()
    assert w.table.rowCount() == 2
    w.search.setText("sam")
    assert w.table.rowCount() == 1 and w.current_id() == a.id
    assert "Body mass index" in w.record.text() and w.analyses.count() == 2
    w.open_selected()
    assert opened == [str((tmp_path / "95.ok3d").resolve())]                 # the latest first
    w.compare_all()
    assert compared == [[str((tmp_path / f"{n}.ok3d").resolve()) for n in (93, 95)]]
    w.search.setText("")
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    w.table.selectRow([r["id"] for r in w._items].index(a.id))
    w.forget()
    assert [r["subject"].name for r in w.rows] == ["Hilary"]
    w.close()


def test_saving_an_analysis_lists_it_with_the_patient(qapp, project, tmp_path):
    from openkick.ui.main_window import MainWindow
    w = MainWindow()
    w.ctl.set_project(project)
    project.subject = P.remember(Subject(name="Sami", height_m=1.68))
    project.path = tmp_path / "sami.okick"
    assert w.save()
    assert P.get(project.subject.id)["analyses"][0]["path"] == str(project.path.resolve())
    w.close()
    w.deleteLater()


def test_picking_a_patient_on_the_3d_start_page(qapp):
    from openkick.ui.multiview import MultiviewWindow
    saved = P.remember(Subject(name="Sami", height_m=1.68, lengths={"thigh_left": 43.5}))
    w = MultiviewWindow()
    w.show()
    qapp.processEvents()
    page = w.setup
    assert [page.patient.itemData(i) for i in range(page.patient.count())] == ["", saved.id]
    page.patient.setCurrentIndex(1)
    page._pick_patient()
    assert page.subject.id == saved.id and page.subject.lengths == {"thigh_left": 43.5}
    assert page.height.value() == 168 and "Sami" in page.subject_btn.text()
    page.patient.setCurrentIndex(0)
    page._pick_patient()
    assert page.subject.empty
    w.close()
    w.deleteLater()
