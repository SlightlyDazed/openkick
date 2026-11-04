"""Installing a plugin from the ZIP GitHub gives with Code › Download ZIP."""

import json
import zipfile

import pytest

from openkick.plugins import install as INS

PLUGIN = '''
from openkick.plugins import AnalysisPlugin, Result

class Hello(AnalysisPlugin):
    id = "hello"
    name = "Hello"

    def run(self, ctx):
        return Result(summary="hi")

PLUGINS = [Hello()]
'''


def manifest(**over):
    m = {"id": "hello-plugin", "name": {"en": "Hello", "es": "Hola"}, "version": "1.2.0", "author": "Someone",
         "openkick": ">=0.1.0", "license": "GPL-3.0-or-later", "templates": ["templates/hello.json"]}
    m.update(over)
    return m


def github_zip(path, man=None, extra=None):
    """A ZIP laid out as GitHub's: everything inside <repo>-main/."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("hello-plugin-main/plugin.json", json.dumps(man or manifest()))
        z.writestr("hello-plugin-main/plugin.py", PLUGIN)
        z.writestr("hello-plugin-main/README.md", "# Hello")
        z.writestr("hello-plugin-main/templates/hello.json", json.dumps({"id": "hello", "name": "Hello"}))
        z.writestr("hello-plugin-main/tests/test_plugin.py", "def test(): pass")
        z.writestr("hello-plugin-main/.github/workflows/test.yml", "on: push")
        for name, text in (extra or {}).items():
            z.writestr(name, text)
    return path


@pytest.fixture
def folders(tmp_path, monkeypatch):
    plugins, tpl = tmp_path / "plugins", tmp_path / "templates"
    plugins.mkdir()
    tpl.mkdir()
    monkeypatch.setattr(INS, "user_dir", lambda: plugins)
    from openkick.core import templates
    monkeypatch.setattr(templates, "user_dir", lambda: tpl)
    return plugins, tpl


def test_installs_and_loads(folders, tmp_path, monkeypatch):
    plugins, tpl = folders
    man = INS.install_zip(github_zip(tmp_path / "hello.zip"))
    dest = plugins / "hello-plugin"
    assert man["version"] == "1.2.0" and (dest / "plugin.py").exists() and (dest / "README.md").exists()
    assert not (dest / "tests").exists() and not (dest / ".github").exists()       # what the user does not need
    assert (tpl / "hello.json").exists()                                            # the template it brings
    import openkick.plugins as P
    monkeypatch.setattr(P, "user_dir", lambda: plugins)
    found, errors = P.discover()
    assert not errors and any(p.id == "hello" for p in found)


def test_already_installed_asks_and_replaces(folders, tmp_path):
    plugins, _ = folders
    INS.install_zip(github_zip(tmp_path / "a.zip"))
    newer = github_zip(tmp_path / "b.zip", manifest(version="2.0.0"))
    with pytest.raises(INS.AlreadyInstalled) as e:
        INS.install_zip(newer)
    assert e.value.installed["version"] == "1.2.0" and e.value.manifest["version"] == "2.0.0"
    assert INS.install_zip(newer, replace=True)["version"] == "2.0.0"
    assert json.loads((plugins / "hello-plugin" / "plugin.json").read_text())["version"] == "2.0.0"


@pytest.mark.parametrize("man, why", [
    (manifest(openkick=">=99.0"), "needs OpenKick"),
    (manifest(id="Bad Id"), "id"),
    (manifest(id="cycling"), "belongs to a plugin of OpenKick"),
    ({k: v for k, v in manifest().items() if k != "version"}, "missing version"),
    (manifest(templates=["templates/none.json"]), "not in the ZIP"),
])
def test_refuses_what_it_cannot_use(folders, tmp_path, man, why):
    with pytest.raises(INS.InstallError, match=why):
        INS.install_zip(github_zip(tmp_path / "x.zip", man))
    assert not any(folders[0].iterdir())                 # nothing left half-installed


def test_refuses_paths_outside_and_non_plugins(folders, tmp_path):
    with pytest.raises(INS.InstallError, match="unsafe"):
        INS.install_zip(github_zip(tmp_path / "evil.zip", extra={"hello-plugin-main/../../evil.py": "x"}))
    assert not (tmp_path / "evil.py").exists()
    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as z:
        z.writestr("photos/a.txt", "x")
    with pytest.raises(INS.InstallError, match="no plugin.json"):
        INS.install_zip(plain)
    assert not any(folders[0].iterdir())


def test_versions():
    assert INS.compatible(">=0.1.0", "0.1.0") and INS.compatible(">=0.1", "0.2.3")
    assert not INS.compatible(">=0.2.0", "0.1.9") and INS.compatible("0.1.0", "0.1.0")
