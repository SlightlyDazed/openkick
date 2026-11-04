"""Installing a plugin from a ZIP file: the one GitHub gives with "Code › Download ZIP", or a release.

The ZIP holds the plugin's repository: a plugin.json (what it is), its plugin.py and whatever
else the author keeps there (README, tests, examples). GitHub puts it all inside one folder named
after the repository; plugin.json is looked for there or at the top. What is copied to the
plugins folder is the repository without its tests and its GitHub files; templates the plugin
brings (listed in plugin.json) also go to the templates folder.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from openkick import __version__
from openkick.plugins import user_dir

MANIFEST = "plugin.json"
ENTRY = "plugin.py"
REQUIRED = ("id", "name", "version", "author", "openkick")
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,48}$")
LEFT_OUT = {"tests", ".github", ".git", "__pycache__", ".pytest_cache"}     # top folders not installed
MAX_BYTES = 300 * 1024 * 1024          # a ZIP that unpacks to more than this is refused


class InstallError(Exception):
    """The ZIP is not a plugin OpenKick can install; the message says why."""


class AlreadyInstalled(InstallError):
    def __init__(self, manifest: dict, installed: dict | None):
        super().__init__(manifest["id"])
        self.manifest, self.installed = manifest, installed or {}


def version_tuple(text: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(text))
    return tuple(int(p) for p in parts[:3]) + (0,) * (3 - len(parts[:3]))


def compatible(spec: str, version: str = __version__) -> bool:
    """Does this OpenKick satisfy the plugin's "openkick" field? Only a lowest version (">=0.2")
    or an exact one ("0.2.1") is understood."""
    spec = str(spec).strip()
    if spec.startswith(">="):
        return version_tuple(version) >= version_tuple(spec[2:])
    return version_tuple(version) == version_tuple(spec)


def _manifest(zf: zipfile.ZipFile) -> tuple[str, dict]:
    """(folder inside the ZIP that holds plugin.json, its content)."""
    found = [n for n in zf.namelist() if PurePosixPath(n).name == MANIFEST and len(PurePosixPath(n).parts) <= 2]
    if not found:
        raise InstallError("no plugin.json")
    name = min(found, key=lambda n: len(PurePosixPath(n).parts))
    try:
        data = json.loads(zf.read(name).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise InstallError(f"plugin.json: {e}") from e
    prefix = str(PurePosixPath(name).parent)
    return ("" if prefix == "." else prefix + "/"), data


def check(manifest: dict) -> None:
    missing = [k for k in REQUIRED if not manifest.get(k)]
    if missing:
        raise InstallError("plugin.json: missing " + ", ".join(missing))
    if not ID.match(str(manifest["id"])):
        raise InstallError(f"plugin.json: id \"{manifest['id']}\" must be lower case letters, digits, - or _")
    from openkick.core.store import CATALOG
    if manifest["id"] in {p["id"] for p in CATALOG}:
        raise InstallError(f"plugin.json: id \"{manifest['id']}\" belongs to a plugin of OpenKick")
    if not compatible(manifest["openkick"]):
        raise InstallError(f"needs OpenKick {manifest['openkick']} (this is {__version__})")


def installed_manifest(plugin_id: str) -> dict | None:
    try:
        return json.loads((user_dir() / plugin_id / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def install_zip(path: str | Path, replace: bool = False) -> dict:
    """Install the plugin in a ZIP; returns its plugin.json with "folder" added. Raises
    AlreadyInstalled when it is there already (call again with replace=True) and InstallError
    when the ZIP is not a plugin this OpenKick can use."""
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise InstallError(str(e)) from e
    with zf:
        prefix, manifest = _manifest(zf)
        check(manifest)
        if f"{prefix}{ENTRY}" not in zf.namelist():
            raise InstallError(f"no {ENTRY} next to plugin.json")
        if sum(i.file_size for i in zf.infolist()) > MAX_BYTES:
            raise InstallError("the ZIP is too large")
        dest = user_dir() / manifest["id"]
        if dest.exists() and not replace:
            raise AlreadyInstalled(manifest, installed_manifest(manifest["id"]))
        work = Path(tempfile.mkdtemp(prefix=".installing-", dir=user_dir()))
        try:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.startswith(prefix):
                    continue
                rel = PurePosixPath(info.filename[len(prefix):])
                if not rel.parts or rel.parts[0] in LEFT_OUT:
                    continue
                if rel.is_absolute() or ".." in rel.parts:
                    raise InstallError(f"unsafe path in the ZIP: {info.filename}")
                out = work.joinpath(*rel.parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            templates = _templates(work, manifest)
            if dest.exists():
                shutil.rmtree(dest)
            work.rename(dest)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    for t in templates:
        from openkick.core.templates import user_dir as template_dir
        shutil.copy(dest / t, template_dir() / Path(t).name)
    return dict(manifest, folder=str(dest))


def _templates(folder: Path, manifest: dict) -> list[str]:
    """The templates the plugin brings (paths inside it), each checked to be a template file."""
    out = []
    for t in manifest.get("templates") or []:
        p = PurePosixPath(t)
        f = folder.joinpath(*p.parts)
        if p.is_absolute() or ".." in p.parts or f.suffix != ".json" or not f.is_file():
            raise InstallError(f"plugin.json: template {t} is not in the ZIP")
        try:
            if not json.loads(f.read_text(encoding="utf-8")).get("id"):
                raise ValueError("no id")
        except ValueError as e:
            raise InstallError(f"template {t}: {e}") from e
        out.append(str(p))
    return out
