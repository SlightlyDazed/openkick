"""Collect every user-visible English source string (for locale/*.json)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ast
import glob
import json
import sys

strings = []


def add(s):
    if isinstance(s, str) and s and s not in strings:
        strings.append(s)


for f in sorted(glob.glob("openkick/**/*.py", recursive=True)):
    tree = ast.parse(open(f, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in ("tr", "_act"):
                add(node.args[0].value)

from openkick.core.landmarks import DISPLAY  # noqa: E402
from openkick.ui.dialogs import SHORTCUTS  # noqa: E402
from openkick.ui.main_window import TOOL_HINTS, TOOLS  # noqa: E402
from openkick.ui.panels import PointTab  # noqa: E402

for v in DISPLAY.values():
    add(v)
for t in TOOLS:
    add(t[2])
for v in TOOL_HINTS.values():
    add(v)
for _, a, b in PointTab.SCOPES:
    add(a)
    add(b)
for _, d in SHORTCUTS:
    add(d)
for s in ("Accurate (heavy)", "Balanced (full)", "Fast (lite)", "segment"):
    add(s)
from openkick.core.bodymodel import DERIVED, SEGMENTS  # noqa: E402

for v in list(SEGMENTS.values()) + list(DERIVED.values()):
    add(v)

from openkick.ui import multiview as MV  # noqa: E402

for v in (list(MV.DISPLAY.values()) + [t for t, _ in MV.LEVELS.values()] + [label for _, label in MV.MODES]
          + [text for text, _ in MV.ANGLE_ROWS] + ["3D VIEW", "3D POINTS"] + list(MV.SESSION_ERRORS)):
    add(v)

from openkick.ui.welcome import NAV  # noqa: E402

for section, entries in NAV:
    add(section)
    for _, _, title, hint, _ in entries:
        add(title)
        add(hint)

from openkick.plugins.builtin import cycling  # noqa: E402

for _, name, _ in cycling.MEASURES:
    add(name)
add(cycling.HIP_TRAVEL_CM)

from openkick.core import gait as _gait  # noqa: E402

for _, full, short, _, _ in _gait.PERRY:
    add(full)
    add(short)

from openkick.core import cycling as _cyc  # noqa: E402
from openkick.core import store as _store  # noqa: E402
from openkick.core import subject as _subj  # noqa: E402
from openkick.ui import compare as _cmp  # noqa: E402
from openkick.ui import report as _report  # noqa: E402
from openkick.ui import review as _rev  # noqa: E402
from openkick.ui import subject as _usubj  # noqa: E402

for _n, _u in _cyc.MEASURES.values():
    add(_n)
for _k, _n, _w in _subj.LENGTHS + _subj.WIDTHS:
    add(_n)
    add(_w)
for _, _n in _subj.BMI_CLASSES:
    add(_n)
for _c in _store.CATEGORIES:
    add(_c)
for _, _n in MV.ACTIVITIES + MV.SMOOTHING + MV.FEET:
    add(_n)
for _names in _usubj.EQUIPMENT.values():
    for _n in _names:
        add(_n)
for _n in (_report.LIMITS, _report.STAR, _report.DOT) + tuple(v[0] for v in _report.FORMATS.values()) + tuple(v[0] for v in _report.OPTIONS.values()):
    add(_n)
from openkick.multiview.session import STROKE_PARTS as _parts  # noqa: E402
from openkick.core.models import TITLES as _titles  # noqa: E402
for _n in _titles.values():
    add(_n)
for _n in _parts + ("{bone} {cm} cm · tape {usual} · {pct} %",):
    add(_n)
for _n in _rev.WHY.values():
    add(_n)
for _, _n in _cmp.CURVES:
    add(_n)
for _s in ("Female", "Male", "Other", "Left", "Right", "All", "height", "Both", "{n} years", "Name", "Age", "Sex",
           "Height", "Weight", "Body mass index", "Dominant side", "Activity", "Reason for consultation",
           "Pain (0–10)", "Leg length difference", "Notes", "L", "R", "left", "right", "none",
           "{d} cm, {side} longer"):
    add(_s)

if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else None
    if lang:
        cat = json.load(open(f"openkick/locale/{lang}.json", encoding="utf-8"))
        missing = [s for s in strings if s not in cat]
        print(f"{len(strings)} strings, {len(missing)} missing in {lang}")
        for s in missing:
            print(repr(s))
    else:
        print(json.dumps(strings, ensure_ascii=False, indent=0))
