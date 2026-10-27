"""The comparison window, the plugin store and the patient's record, headless."""
import json
import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from openkick import i18n  # noqa: E402
from openkick.core import compare as K  # noqa: E402
from openkick.core.subject import Subject  # noqa: E402
from tests.test_gui import qapp  # noqa: E402,F401  (the same application for every window)


@pytest.fixture(autouse=True)
def english():
    before = i18n.language()
    i18n.set_language("en")
    yield
    i18n.set_language(before)


def analysis(label, knee=140.0, toes=1.0, height=1.70):
    """A 3D cycling analysis with 30 strokes of each leg and its mean cycles."""
    cyc = [{"knee_bottom": knee + 0.3 * np.sin(k), "foot_bottom": toes + 0.2 * np.cos(k), "cadence": 70.0,
            "knee_lateral": 2.0, "first": 40 * k, "last": 40 * k + 39} for k in range(30)]
    series = np.tile(np.concatenate([np.linspace(knee, 70, 20), np.linspace(70, knee, 20)]), 30)
    curve = K._resample(series, cyc)
    return K.Analysis(path=f"{label}.ok3d", label=label, kind="3D", activity="cycling",
                      subject=Subject(name=label, height_m=height, mass_kg=65.0),
                      cycles={"right": cyc, "left": cyc},
                      curves={side: {"knee": curve, "hip": curve, "ankle": curve} for side in ("right", "left")})


def test_comparison_window_tables_charts_and_order(qapp):
    from openkick.ui.compare import AnalysisCard, CompareWindow, CycleChart
    w = CompareWindow()
    w.show()
    qapp.processEvents()
    assert w.tables == []                                     # fewer than two: only the hint
    w.items = [analysis("93"), analysis("95", 140.4, 12.0), analysis("97", 141.0, 20.0)]
    w.refresh()
    qapp.processEvents()
    titles = [t.title for t in w.tables]
    assert titles[0] == "Patients" and len(titles) == 3      # the people, then each leg
    assert w.tables[1].columns[1:4] == ["93", "95", "97"]
    assert len(w.body.widget().findChildren(CycleChart)) == 6  # knee, hip, ankle for each leg
    w.grab()                                                  # paints the cards, tables and charts
    w.leg.setCurrentIndex(w.leg.findData("right"))
    assert len(w.tables) == 2
    cards = w.cards_box.findChildren(AnalysisCard)
    w._move(cards[0], +1)                                    # the second one becomes the reference
    assert [i.label for i in w.items] == ["95", "93", "97"] and w.tables[1].columns[1] == "95"
    w._remove(w.cards_box.findChildren(AnalysisCard)[-1])
    assert [i.label for i in w.items] == ["95", "93"]
    w.close()
    w.deleteLater()


def test_a_comparison_is_saved_and_opened_again(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from openkick.ui.compare import CompareWindow
    folder = tmp_path / "bike"
    folder.mkdir()
    for name in ("93", "95"):
        (folder / f"{name}.ok3d").write_text("{}")
    read = []

    def fake_load(path, activity=None, label=None):
        read.append(path)
        from pathlib import Path
        a = analysis(label or Path(path).stem)
        a.path = str(path)
        return a
    monkeypatch.setattr(K, "load", fake_load)
    w = CompareWindow()
    w.add_files([str(folder / "93.ok3d"), str(folder / "95.ok3d")], ["Saddle 93", "Saddle 95"])
    w.people.setCurrentIndex(1)
    target = folder / "saddle"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    w.save_comparison()
    data = json.loads((folder / "saddle.okcmp").read_text(encoding="utf-8"))
    assert [r["relative"] for r in data["analyses"]] == ["93.ok3d", "95.ok3d"] and data["different_people"]
    moved = tmp_path / "elsewhere"                             # the folder is moved: the relative paths still find them
    folder.rename(moved)
    read.clear()
    v = CompareWindow()
    v.load_comparison(moved / "saddle.okcmp")
    assert [i.label for i in v.items] == ["Saddle 93", "Saddle 95"] and v.people.currentData() is True
    assert read == [str(moved / "93.ok3d"), str(moved / "95.ok3d")]
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "out"), "")))
    v.export_csv()
    text = (tmp_path / "out.csv").read_text(encoding="utf-8")
    assert "Patients" in text and "Saddle 95 − Saddle 93" in text
    for x in (w, v):
        x.close()
        x.deleteLater()


def test_store_installs_and_removes(qapp, tmp_path, monkeypatch):
    import openkick.core.store as ST
    from openkick.ui.store import PackageCard, StoreDialog
    monkeypatch.setattr(ST, "data_dir", lambda: tmp_path)
    d = StoreDialog()
    total = len(ST.CATALOG)
    assert d.count.text() == f"{total} of {total} packages installed"

    def card(pid):
        return next(c for c in d.scroll.widget().findChildren(PackageCard) if c.pkg["id"] == pid)
    remove = card("cycling").findChildren(QtWidgets.QPushButton)[0]
    assert remove.text() == "Remove"
    remove.click()
    assert "cycling" not in ST.installed() and d.changed_any
    assert d.count.text() == f"{total - 1} of {total} packages installed"
    assert card("cycling").findChildren(QtWidgets.QPushButton)[0].text() == "Install"
    assert not card("base-rom").findChildren(QtWidgets.QPushButton)[0].isEnabled()
    d.search.setText("perry")                                  # the gait package, by its description
    assert [c.pkg["id"] for c in d.scroll.widget().findChildren(PackageCard)] == ["gait"]
    d.search.setText("")
    d.chip_buttons["Strength"].click()
    assert [c.pkg["id"] for c in d.scroll.widget().findChildren(PackageCard)] == ["squat"]
    d.close()
    d.deleteLater()


def test_patient_record_dialog(qapp):
    from openkick.ui.subject import SubjectDialog
    s = Subject(name="Sami", sex="female", height_m=1.70, mass_kg=72.3,
                lengths={"leg_left": 88.0, "leg_right": 89.5, "thigh_right": 44.0},
                equipment={"Saddle height (cm)": "93"})
    d = SubjectDialog(s, "cycling")
    assert d.bmi.text() == "25.0 kg/m² · Overweight"
    assert "1.5 cm, right longer" in d.difference.text() and "±1 cm" in d.difference.text()
    assert "Crank length (mm)" in d.equipment                 # the bike's own items are offered
    d.equipment["Crank length (mm)"].setText("172.5")
    d.extra[0][0].setText("Insoles")
    d.extra[0][1].setText("none")
    d.lengths["leg_left"].setValue(89.5)
    out = d.subject()
    assert out.name == "Sami" and out.sex == "female" and abs(out.height_m - 1.70) < 1e-9
    assert abs(out.mass_kg - 72.3) < 1e-9 and out.lengths["thigh_right"] == 44.0
    assert out.equipment == {"Saddle height (cm)": "93", "Crank length (mm)": "172.5", "Insoles": "none"}
    assert d.difference.text() == "Both legs measure the same."
    d.mass.setValue(d.mass.minimum())                          # "—": not weighed
    assert d.subject().mass_kg is None and d.bmi.text() == "—"
    d.close()
    d.deleteLater()


def test_pdf_reports(qapp, tmp_path, monkeypatch):
    """The 3D session and the comparison as PDF, with the evaluator's observations kept."""
    from PySide6.QtWidgets import QFileDialog

    from openkick.ui import report as R
    from openkick.ui.compare import CompareWindow
    from openkick.ui.multiview import MultiviewWindow
    from tests.test_multiview import _on_real_file, make
    s, _ = make()
    _on_real_file(s)
    s.solve()
    s.subject = Subject(name="Sami", height_m=1.68, mass_kg=94.0)
    w = MultiviewWindow()
    w.show()
    w.set_session(s)
    qapp.processEvents()
    out = tmp_path / "s.pdf"
    R.session_report(w, str(out), "Lic. Prueba", "Todo en orden.\nSegunda línea.")
    assert out.read_bytes()[:4] == b"%PDF" and out.stat().st_size > 20_000          # with its pictures
    row = w.ctl.row
    # Notes: the evaluator and the observations, kept with the session
    def fill(self):
        self.evaluator.setText("Lic. Prueba")
        self.notes.setPlainText("Revisar el tobillo")
        return 1
    monkeypatch.setattr(R.NotesDialog, "exec", fill)
    w.edit_notes()
    assert (s.report_evaluator, s.report_notes) == ("Lic. Prueba", "Revisar el tobillo") and w.ctl.dirty
    # Report: each format straight away, from the button's menu
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "b"), "")))
    from PySide6.QtWidgets import QPushButton
    button = next(b for b in w.findChildren(QPushButton) if b.menu() is not None and b.text().strip().startswith("Report"))
    titles = [a.text() for a in button.menu().actions() if a.text()]
    assert titles[:3] == ["PDF report…", "Report as text (.txt)…", "Data for research (.json)…"]
    for fmt in ("pdf", "txt", "json"):
        w.export_report(fmt)
        assert (tmp_path / f"b.{fmt}").exists()
    assert "Revisar el tobillo" in (tmp_path / "b.txt").read_text(encoding="utf-8") and w.ctl.row == row
    s.save(tmp_path / "s.ok3d")
    from openkick.multiview.session import Session
    back = Session.load(tmp_path / "s.ok3d", require_videos=False)
    assert (back.report_evaluator, back.report_notes) == ("Lic. Prueba", "Revisar el tobillo")
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()
    c = CompareWindow()
    c.items = [analysis("93"), analysis("95", 140.4, 12.0)]
    c.refresh()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "cmp"), "")))
    c.edit_notes()
    c.export_report("pdf")
    assert (tmp_path / "cmp.pdf").read_bytes()[:4] == b"%PDF" and c.notes == "Revisar el tobillo"
    c.export_report("json")
    assert (tmp_path / "cmp.json").exists()
    c.close()
    c.deleteLater()


def test_text_and_json_reports(qapp, tmp_path):
    """The same reports as plain text and as JSON (numbers unrounded, null for what is missing),
    with the patient left out on request."""
    import json as J

    from openkick.core import report_data as D
    from openkick.ui import report as R
    from openkick.ui.compare import CompareWindow
    from openkick.ui.multiview import MultiviewWindow
    from tests.test_multiview import _on_real_file, make
    assert D.clean({"a": np.float64(1.5), "b": [np.nan, np.int64(2)], "c": np.array([1.0, np.inf])}) == \
        {"a": 1.5, "b": [None, 2], "c": [1.0, None]}
    s, _ = make()
    _on_real_file(s)
    s.solve()
    s.subject = Subject(name="Sami", birth="2000-01-01", height_m=1.68, mass_kg=94.0, notes="private")
    w = MultiviewWindow()
    w.show()
    w.set_session(s)
    qapp.processEvents()
    R.session_report(w, str(tmp_path / "s.txt"), "Lic. Prueba", "Bien.")
    text = (tmp_path / "s.txt").read_text(encoding="utf-8")
    assert "Sami" in text and "How far it can be trusted" in text and "Lic. Prueba" in text and "----" in text
    R.session_report(w, str(tmp_path / "s.json"), "Lic. Prueba", "Bien.", anonymous=True, series=True)
    data = J.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert data["format"] == "openkick-3d-report" and data["observations"] == "Bien."
    assert "name" not in data["patient"] and "notes" not in data["patient"] and data["patient"]["age_years"] >= 25
    assert data["patient"]["bmi_kg_m2"] > 33 and len(data["session"]["videos"]) == 2
    assert data["quality"]["level"] in ("good", "fair", "poor") and len(data["series"]["t_s"]) == len(s.t)
    s.saved_revision = s.revision
    w.ctl.saved_version = w.ctl.version
    w.close()
    w.deleteLater()

    c = CompareWindow()
    c.items = [analysis("93"), analysis("95", 140.4, 12.0)]
    c.refresh()
    R.comparison_report(c, str(tmp_path / "c.json"))
    data = J.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    rows = {r["key"]: r for r in data["legs"]["right"]}
    toes = rows["foot_bottom"]
    assert toes["unit"] == "°" and toes["values"][0]["n"] == 30
    assert abs(toes["differences"][0]["value"] - 11.0) < 0.2 and toes["differences"][0]["beyond_variation"]
    assert not rows["knee_bottom"]["differences"][0]["beyond_variation"]
    assert len(data["mean_cycle"]["93"]["right"]["knee"]["mean"]) == K.SAMPLES
    R.comparison_report(c, str(tmp_path / "c.txt"), anonymous=True)
    text = (tmp_path / "c.txt").read_text(encoding="utf-8")
    assert "Toes down, bottom" in text and "93" in text and "●" in text
    c.close()
    c.deleteLater()
