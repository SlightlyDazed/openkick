"""Every language: complete, with the same placeholders as English, and the windows open in it."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from openkick import i18n

ROOT = Path(__file__).resolve().parents[1]
CODES = [c for c in i18n.LANGUAGES if c != "en"]


@pytest.fixture(scope="module")
def strings():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "extract_strings.py")], capture_output=True, text=True,
                         cwd=ROOT, env={"PYTHONPATH": str(ROOT), "QT_QPA_PLATFORM": "offscreen", "PATH": ""})
    return json.loads(out.stdout)


@pytest.fixture(autouse=True)
def back_to_spanish():
    yield
    i18n.set_language("es")


@pytest.mark.parametrize("code", CODES)
def test_every_text_is_translated_with_its_placeholders(strings, code):
    cat = json.loads((ROOT / "openkick" / "locale" / f"{code}.json").read_text(encoding="utf-8"))
    missing = [s for s in strings if s not in cat]
    assert not missing, missing[:5]
    ph = lambda s: sorted(re.findall(r"\{[a-z_]*\}", s))  # noqa: E731
    wrong = [s for s in strings if ph(s) != ph(cat[s]) or s.count("\n") != cat[s].count("\n")]
    assert not wrong, wrong[:5]


def test_numbers_templates_and_the_computer_language(monkeypatch):
    from openkick.ui.multiview import num
    i18n.set_language("de")
    assert num(1.5) == "1,5"
    i18n.set_language("ja")
    assert num(1.5) == "1.5"
    assert i18n.pick({"es": "Sentadilla", "en": "Squat"}) == "スクワット"     # from the catalog
    assert i18n.pick({"es": "Sentadilla", "en": "Something new"}) == "Something new"
    from PySide6.QtCore import QLocale
    for system, expected in (("de_DE", "de"), ("pt_PT", "pt_BR"), ("zh_CN", "zh_CN"), ("nl_NL", "es"), ("en_GB", "en")):
        monkeypatch.setattr(QLocale, "system", staticmethod(lambda s=system: QLocale(s)))
        assert i18n.system_language() == expected, system


@pytest.mark.parametrize("code", ["de", "ru", "ja", "zh_CN"])
def test_the_windows_open_in(code):
    from tests.test_multiview import _app
    app = _app()
    i18n.set_language(code)
    from openkick.ui.main_window import MainWindow
    from openkick.ui.multiview import MultiviewWindow
    w = MainWindow()
    w.show()
    app.processEvents()
    w.grab()
    m = MultiviewWindow()
    m.show()
    app.processEvents()
    m.grab()
    assert w.a_about.text() == i18n.tr("About OpenKick") != "About OpenKick"
    m.close()
    w.close()
