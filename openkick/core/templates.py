"""Analysis templates: reusable, shareable, no-code analysis definitions.

A template is a small JSON file listing which angles and trajectories to
measure and which phases the movement has. Built-in templates live in
openkick/templates/, user templates in <data dir>/templates/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from openkick.core import landmarks as L
from openkick.core.pose import data_dir
from openkick.core.project import Measurement, PointRef, Project
from openkick.i18n import pick, tr

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "templates"


def user_dir() -> Path:
    d = data_dir() / "templates"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Template:
    id: str
    name: str | dict
    description: str | dict = ""
    angles: list[list[str]] = field(default_factory=list)       # [proximal, vertex, distal]
    trajectories: list[str] = field(default_factory=list)
    phases: list[str | dict] = field(default_factory=list)
    color: str = "#C6F24E"
    icon: str = ""                                              # SVG path in a 48x48 box
    side: str = ""                                              # "camera": measure the side turned to the camera
    builtin: bool = False
    path: Path | None = None

    @property
    def label(self) -> str:
        return pick(self.name)

    @property
    def summary(self) -> str:
        return pick(self.description)

    def phase_names(self) -> list[str]:
        return [pick(p) for p in self.phases]

    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "description": self.description, "angles": self.angles,
             "trajectories": self.trajectories, "phases": self.phases, "color": self.color, "icon": self.icon}
        if self.side:
            d["side"] = self.side
        return d

    @classmethod
    def from_dict(cls, d: dict, builtin=False, path=None) -> "Template":
        return cls(id=d["id"], name=d["name"], description=d.get("description", ""),
                   angles=d.get("angles", []), trajectories=d.get("trajectories", []),
                   phases=d.get("phases", []), color=d.get("color", "#C6F24E"), icon=d.get("icon", ""),
                   side=d.get("side", ""), builtin=builtin, path=path)


def list_templates() -> list[Template]:
    from openkick.core.store import template_available
    out = []
    for f in sorted(BUILTIN_DIR.glob("*.json")):
        t = Template.from_dict(json.loads(f.read_text(encoding="utf-8")), builtin=True, path=f)
        if t.id == "blank" or template_available(t.id):        # a removed package takes its template away
            out.append(t)
    out.sort(key=lambda t: json.loads(t.path.read_text(encoding="utf-8")).get("order", 99))
    for f in sorted(user_dir().glob("*.json")):
        try:
            out.append(Template.from_dict(json.loads(f.read_text(encoding="utf-8")), path=f))
        except (ValueError, KeyError):
            continue
    return out


def find(template_id: str | None) -> Template | None:
    if not template_id:
        return None
    return next((t for t in list_templates() if t.id == template_id), None)


def save_user_template(t: Template) -> Path:
    if not t.id:
        t.id = re.sub(r"[^a-z0-9]+", "-", t.label.lower()).strip("-") or "template"
    path = user_dir() / f"{t.id}.json"
    path.write_text(json.dumps(t.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    t.path = path
    return path


def angle_name(vertex: str) -> str:
    return tr(L.DISPLAY[vertex])


def mirrored(name: str) -> str:
    """The same point on the other side of the body."""
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def apply(project: Project, t: Template):
    """Add the template's measurements (skipping ones that already exist).
    A template with side "camera" lists one side of the body and is applied to whichever side
    the video shows."""
    angles, trajectories = t.angles, t.trajectories
    if t.side == "camera" and angles:
        from openkick.core.analysis import facing_side
        if not angles[0][1].startswith(facing_side(project)):
            angles = [[mirrored(n) for n in a] for a in angles]
            trajectories = [mirrored(n) for n in trajectories]
    existing = {(m.kind, tuple(p.landmark for p in m.points)) for m in project.measurements}
    for a, o, b in angles:
        key = ("angle", (L.INDEX[a], L.INDEX[o], L.INDEX[b]))
        if key in existing:
            continue
        project.measurements.append(Measurement(
            kind="angle", name=angle_name(o), color=project.next_color(),
            points=[PointRef.at(L.INDEX[a]), PointRef.at(L.INDEX[o]), PointRef.at(L.INDEX[b])]))
    for p in trajectories:
        key = ("trajectory", (L.INDEX[p],))
        if key in existing:
            continue
        project.measurements.append(Measurement(
            kind="trajectory", name=tr(L.DISPLAY[p]), color=project.next_color(),
            points=[PointRef.at(L.INDEX[p])]))
    project.template = t.id
    project.touch()
