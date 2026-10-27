"""The window of the first start: what is downloaded, once, and that it does not open again."""
import pytest


@pytest.fixture
def app():
    from tests.test_multiview import _app
    from PySide6.QtCore import QSettings
    a = _app()
    QSettings().remove("firstrun/done")
    yield a
    QSettings().remove("firstrun/done")


def test_the_first_start_offers_the_models_and_then_stays_away(app, monkeypatch):
    from openkick.core import models
    from openkick.ui import firstrun
    got = []
    state = {"2d": False, "3d": False}
    monkeypatch.setattr(models, "components", lambda: [
        {"key": k, "title": models.TITLES[k], "mb": 1, "ready": state[k]} for k in ("2d", "3d")])

    def fake(key, progress=None, cancelled=None):
        for i in range(3):
            progress(0, 1, i + 1, 3)
        got.append(key)
        state[key] = True
        return True
    monkeypatch.setattr(models, "download", fake)
    assert firstrun.needed()
    d = firstrun.FirstRunDialog()
    assert d.get.isVisible() is False or d.get.isEnabled()
    d._download()
    d.worker.wait()
    app.processEvents()
    assert got == ["2d", "3d"] and not firstrun.needed() and not d.get.isVisible()
    d2 = firstrun.FirstRunDialog()                                   # everything here: only "Start"
    assert not d2.get.isVisible() if d2.isVisible() else d2.get.isHidden()


def test_a_failed_download_says_why_and_leaves_it_for_later(app, monkeypatch):
    from openkick.core import models
    from openkick.ui import firstrun
    monkeypatch.setattr(models, "components", lambda: [{"key": "3d", "title": models.TITLES["3d"], "mb": 1, "ready": False}])

    def fail(key, progress=None, cancelled=None):
        raise OSError("no internet")
    monkeypatch.setattr(models, "download", fail)
    d = firstrun.FirstRunDialog()
    d._download()
    d.worker.wait()
    app.processEvents()
    assert "no internet" in d.status.text() and d.get.isEnabled()
    d._close()
    assert not firstrun.needed()                                     # it does not insist at every start


def test_the_real_components_are_listed():
    from openkick.core import models
    keys = [c["key"] for c in models.components()]
    assert keys[0] == "2d" and all(isinstance(c["ready"], bool) for c in models.components())
