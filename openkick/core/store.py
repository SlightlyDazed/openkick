"""The plugin store: packages of analyses that can be installed and removed.

A package brings what one kind of analysis needs: its plugin (the tables), its template (the
angles and phases to measure) and, where there is one, its 3D analysis. For now every package
ships with OpenKick and "installing" one only shows it again; the catalogue is laid out so that
it can later come from an online repository (each package with its own version and files).

What is installed is kept in <data dir>/store.json; with no such file, everything is.
"""

from __future__ import annotations

import json

from openkick.core.pose import data_dir

# id, name, category, what it is for, the plugins and templates it brings, the 3D activity it
# brings, whether it is part of the base (cannot be removed), its icon
CATALOG = [
    {"id": "base-rom", "name": {"es": "Informe de rango de movimiento", "en": "Range of motion report"},
     "category": "Analysis", "description": {"es": "Tabla de ROM por articulación y fase.",
                                             "en": "ROM table per joint and phase."},
     "plugins": ["rom_report"], "templates": [], "activity": None, "base": True, "icon": "chart"},
    {"id": "base-speed", "name": {"es": "Velocidades y aceleraciones", "en": "Velocities and accelerations"},
     "category": "Analysis", "description": {"es": "Velocidad lineal y angular máxima por fase.",
                                             "en": "Peak linear and angular velocity per phase."},
     "plugins": ["velocities"], "templates": [], "activity": None, "base": True, "icon": "chart"},
    {"id": "base-opensim", "name": {"es": "Exportar a OpenSim", "en": "Export to OpenSim"},
     "category": "Export", "description": {"es": "Archivo .trc con los puntos para OpenSim.",
                                           "en": ".trc file with the points, for OpenSim."},
     "plugins": ["opensim_trc"], "templates": [], "activity": None, "base": True, "icon": "export"},
    {"id": "cycling", "name": {"es": "Ciclismo", "en": "Cycling"}, "category": "Sport",
     "description": {"es": "Cada pedalada en 2D y 3D: rodilla, cadera, tobillo y postura, cuánto varían, ambas "
                           "piernas en 3D, y la comparación entre ajustes de la bici.",
                     "en": "Every pedal stroke in 2D and 3D: knee, hip, ankle and posture, how much they vary, "
                           "both legs in 3D, and the comparison between bike settings."},
     "plugins": ["cycling"], "templates": ["cycling"], "activity": "cycling", "base": False, "icon": "bike"},
    {"id": "gait", "name": {"es": "Marcha", "en": "Gait"}, "category": "Clinical",
     "description": {"es": "Zancadas con sus ocho fases (Perry), ritmo, longitudes de paso y la comparación "
                           "izquierda/derecha, en 2D y 3D.",
                     "en": "Strides with their eight phases (Perry), pace, step lengths and left against right, "
                           "in 2D and 3D."},
     "plugins": ["gait"], "templates": ["gait"], "activity": "gait", "base": False, "icon": "walk"},
    {"id": "kick", "name": {"es": "Patada", "en": "Kick"}, "category": "Sport",
     "description": {"es": "Plantilla de patada: rodilla y cadera de la pierna de golpeo, cuatro fases.",
                     "en": "Kick template: knee and hip of the kicking leg, four phases."},
     "plugins": [], "templates": ["kick"], "activity": None, "base": False, "icon": "template"},
    {"id": "squat", "name": {"es": "Sentadilla", "en": "Squat"}, "category": "Strength",
     "description": {"es": "Plantilla de sentadilla: rodilla, cadera y tobillo, descenso y ascenso.",
                     "en": "Squat template: knee, hip and ankle, descent and ascent."},
     "plugins": [], "templates": ["squat"], "activity": None, "base": False, "icon": "template"},
    {"id": "jump", "name": {"es": "Salto vertical", "en": "Vertical jump"}, "category": "Sport",
     "description": {"es": "Plantilla de salto: contramovimiento, impulso, vuelo y aterrizaje.",
                     "en": "Jump template: countermovement, push-off, flight and landing."},
     "plugins": [], "templates": ["jump"], "activity": None, "base": False, "icon": "template"},
]
CATEGORIES = ["Sport", "Clinical", "Strength", "Analysis", "Export"]
VERSION = "0.1"


def _file():
    return data_dir() / "store.json"


def installed() -> set[str]:
    """Ids of the installed packages (all of them until the user removes one)."""
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
        removed = set(data.get("removed", []))
    except (OSError, ValueError):
        removed = set()
    return {p["id"] for p in CATALOG if p["base"] or p["id"] not in removed}


def set_installed(package_id: str, on: bool):
    pkg = next((p for p in CATALOG if p["id"] == package_id), None)
    if pkg is None or pkg["base"]:
        return
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    removed = set(data.get("removed", []))
    (removed.discard if on else removed.add)(package_id)
    data["removed"] = sorted(removed)
    _file().write_text(json.dumps(data, indent=1), encoding="utf-8")


def _owner(kind: str, item: str):
    return next((p for p in CATALOG if item in p[kind]), None)


def plugin_available(plugin_id: str) -> bool:
    """Is this built-in plugin's package installed? (Plugins no package claims always are.)"""
    p = _owner("plugins", plugin_id)
    return p is None or p["id"] in installed()


def template_available(template_id: str) -> bool:
    p = _owner("templates", template_id)
    return p is None or p["id"] in installed()


def activity_available(activity: str) -> bool:
    p = next((p for p in CATALOG if p["activity"] == activity), None)
    return p is None or p["id"] in installed()
