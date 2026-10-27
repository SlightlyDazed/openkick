"""Any analysis template in 3D: its angles measured on the reconstruction.

Cycling and gait have their own 3D analysis, with cycles and phases. Every other template (the
user's own included) gets its angles in 3D: the real angle of the limbs, not their shadow on one
picture, frame by frame, and their range over the clip.
"""

from __future__ import annotations

import warnings

import numpy as np

from openkick.core.analysis import angle_between, lowpass
from openkick.i18n import tr
from openkick.multiview.geometry import IDX

# the 2D skeleton's names that the 3D one calls otherwise
RENAME = {"foot_index": "toe"}


def _point(name: str) -> int | None:
    side, _, part = name.partition("_")
    key = f"{side}_{RENAME.get(part, part)}" if side in ("left", "right") else name
    return IDX.get(key)


def angles(X: np.ndarray, t: np.ndarray, template) -> list[tuple[str, np.ndarray]]:
    """(name, per-row angle in degrees) of each angle of the template that the 3D skeleton has."""
    from openkick.core.templates import angle_name
    fps = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 30.0
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a, o, b in template.angles:
            ia, io, ib = _point(a), _point(o), _point(b)
            if None in (ia, io, ib):
                continue
            out.append((angle_name(o), lowpass(angle_between(X[:, ia], X[:, io], X[:, ib]), fps)))
    return out


def report(X: np.ndarray, t: np.ndarray, template):
    from openkick.plugins import Result, Table
    rows = []
    for name, v in angles(X, t, template):
        ok = np.isfinite(v)
        if not ok.any():
            continue
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        i0, i1 = int(np.nanargmin(v)), int(np.nanargmax(v))
        rows.append([name, f"{lo:.1f}°", f"{hi:.1f}°", f"{hi - lo:.1f}°", f"{np.nanmean(v):.1f}°",
                     f"{t[i0] - t[0]:.2f} s", f"{t[i1] - t[0]:.2f} s"])
    table = Table([tr("Angle"), tr("Min"), tr("Max"), tr("ROM"), tr("Mean"), tr("Min at"), tr("Max at")], rows,
                  tr("Angles in 3D"))
    return Result(title=tr("“{name}” in 3D", name=template.label), tables=[table],
                  summary=tr("The template's angles measured on the 3D reconstruction. Angles are measured inside "
                             "the joint, so 180° is a straight limb."))
