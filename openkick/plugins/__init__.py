"""Plugin API.

A plugin is a Python file (or a folder with plugin.py) placed in the user's
plugins folder (menu Plugins > Open plugins folder). It must define a list
called PLUGINS with instances of AnalysisPlugin or ExportPlugin subclasses.

Minimal example (my_plugin.py):

    from openkick.plugins import AnalysisPlugin, Result, Table

    class MaxKneeFlexion(AnalysisPlugin):
        id = "max_knee_flexion"
        name = {"es": "Flexión máxima de rodilla", "en": "Max knee flexion"}
        description = "Minimum knee angle in each phase."

        def run(self, ctx):
            knee = ctx.find_angle("right_knee")
            rows = []
            for ph in ctx.project.phases:
                s = ctx.stats(knee, ph)
                rows.append([ph.name, ctx.fmt(s.minimum, "°")])
            return Result(tables=[Table(["Phase", "Min"], rows)])

    PLUGINS = [MaxKneeFlexion()]

Plugins get read access to everything through `ctx` (project, pose arrays,
series, statistics) and must not modify the project unless they say so.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openkick.core import analysis as A
from openkick.core import landmarks as L
from openkick.core.pose import data_dir
from openkick.core.project import Measurement, Phase, Project
from openkick.i18n import pick, tr


@dataclass
class Table:
    columns: list[str]
    rows: list[list]
    title: str = ""


@dataclass
class Result:
    title: str = ""
    summary: str = ""
    tables: list[Table] = field(default_factory=list)


class PluginContext:
    """What a plugin can see and use."""

    def __init__(self, project: Project, ask: Callable[[str, float], float | None] | None = None):
        self.project = project
        self.analysis = A
        self.landmarks = L
        self._ask = ask

    # helpers
    def series(self, m: Measurement):
        return A.series(self.project, m)

    def stats(self, m: Measurement, phase: Phase | None = None):
        if phase is None:
            return A.stats(self.project, m)
        return A.phase_stats(self.project, m, phase)

    def times(self):
        return A.times(self.project)

    def fmt(self, v, unit: str) -> str:
        return A.format_value(v, unit)

    def find_angle(self, vertex: str) -> Measurement | None:
        idx = L.INDEX[vertex]
        return next((m for m in self.project.measurements
                     if m.kind == "angle" and m.points[1].landmark == idx), None)

    def ask_number(self, label: str, default: float) -> float | None:
        """Ask the user for a number (returns default when running without a GUI)."""
        return self._ask(label, default) if self._ask else default


class Plugin:
    id = ""
    name: str | dict = ""
    description: str | dict = ""
    version = "0.1"
    author = ""
    builtin = False

    @property
    def label(self) -> str:
        return pick(self.name)

    @property
    def summary(self) -> str:
        return pick(self.description)


class AnalysisPlugin(Plugin):
    kind = "analysis"

    def run(self, ctx: PluginContext) -> Result:
        raise NotImplementedError


class ExportPlugin(Plugin):
    kind = "export"
    file_filter = "All files (*)"
    suffix = ""

    def export(self, ctx: PluginContext, path: Path) -> str:
        """Write the file; return a short message for the user."""
        raise NotImplementedError


def user_dir() -> Path:
    d = data_dir() / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover() -> tuple[list[Plugin], list[str]]:
    """Built-in plugins first, then the user's. Returns (plugins, load errors)."""
    plugins: list[Plugin] = []
    errors: list[str] = []
    from openkick.plugins.builtin import BUILTIN_MODULES
    from openkick.core.store import plugin_available
    for modname in BUILTIN_MODULES:
        mod = importlib.import_module(f"openkick.plugins.builtin.{modname}")
        for p in getattr(mod, "PLUGINS", []):
            p.builtin = True
            if plugin_available(p.id):                       # its package may have been removed in the store
                plugins.append(p)
    for entry in sorted(user_dir().iterdir()):
        path = entry / "plugin.py" if entry.is_dir() else entry
        if path.suffix != ".py" or not path.exists() or path.name.startswith("_"):
            continue
        try:
            mod = _load_module(path, f"openkick_user_plugin_{entry.stem}")
            found = getattr(mod, "PLUGINS", [])
            if not found:
                errors.append(f"{entry.name}: {tr('no PLUGINS list found')}")
            plugins.extend(found)
        except Exception:
            errors.append(f"{entry.name}:\n{traceback.format_exc(limit=3)}")
    return plugins, errors
