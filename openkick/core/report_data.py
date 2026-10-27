"""The numbers of a report, for a spreadsheet or a script (JSON): unrounded, with their units.

Measure keys and names are the program's own (English, stable between versions and languages);
"label" carries the name in the language of the report. NaN becomes null. With `anonymous`, the
patient's name, date of birth, notes and id are left out (the age stays)."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from openkick.core import cycling as C
from openkick.i18n import tr

FORMAT_VERSION = 1
ABOUT = {
    "angles": "degrees, measured inside the joint: 180 is a straight limb",
    "lengths": "cm; with different people, '% height' is the length as a share of each one's height",
    "median, spread": "typical value over the cycles and how much it varies from one to the next: "
                      "1.4826 x the median absolute deviation (a standard deviation that wrong cycles do not inflate)",
    "beyond_variation": "the difference is larger than twice the larger spread, and at least 1 unit",
    "less_reliable": "measures resting on the points the detector places least reliably (the foot)",
    "cycles": "one per pedal stroke or stride; a, b, top, first, last are rows (frames of video 1 inside the "
              "time all videos share), seconds is its duration",
    "mean_cycle": "101 samples from 0 to 100 % of the cycle: mean and standard deviation over n cycles",
    "series": "per-frame values; t_s is the time from the first row",
}


def clean(obj):
    """Plain JSON values: numpy numbers to floats and ints, NaN and infinities to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def patient(subject, anonymous: bool = False) -> dict:
    if subject is None:
        return {}
    d = subject.to_dict()
    if anonymous:
        for k in ("name", "birth", "notes", "id"):
            d.pop(k, None)
    else:
        d.pop("id", None)
    if subject.age is not None:
        d["age_years"] = subject.age
    if subject.bmi is not None:
        d["bmi_kg_m2"] = subject.bmi
        d["bmi_class"] = subject.bmi_class
    return d


def header(kind: str, evaluator: str, notes: str) -> dict:
    from openkick import __version__
    return {"format": f"openkick-{kind}", "version": FORMAT_VERSION, "openkick": __version__,
            "generated": datetime.now().isoformat(timespec="seconds"), "about": ABOUT, "evaluator": evaluator or None,
            "observations": notes or None}


def _named(name: str) -> dict:
    """A measure's name without its star, and whether it had one."""
    return {"name": name.rstrip(" *"), "label": tr(name).rstrip(" *"), "less_reliable": name.endswith("*")}


def measures(activity: str):
    """(module with MEASURES, keys in the order of the tables) of an activity."""
    from openkick.core import compare as K
    return K._measures(activity)


def typical(cycles: list[dict], activity: str) -> dict:
    """key -> {"name", "label", "unit", "median", "spread", "n"} over the cycles of one leg."""
    M, keys = measures(activity)
    out = {}
    for key in keys:
        name, unit = M.MEASURES[key]
        vals = [c.get(key) for c in cycles]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        mid, spread = C.typical(vals)
        out[key] = {**_named(name), "unit": unit, "median": mid, "spread": spread, "n": len(vals)}
    return out


def curves(item) -> dict:
    """side -> joint -> {"mean": [...], "sd": [...], "n"} from 0 to 100 % of the cycle."""
    out = {}
    for side, by_joint in (item.curves or {}).items():
        for joint, curve in (by_joint or {}).items():
            if curve is None:
                continue
            mean, sd, n = curve
            out.setdefault(side, {})[joint] = {"mean": mean, "sd": sd, "n": n}
    return out


def session(s, kind: str | None, activity, result, item, evaluator: str = "", notes: str = "",
            anonymous: bool = False, series: dict | None = None, angles: list | None = None) -> dict:
    """A 3D session: how it was recorded and rebuilt, how far it can be trusted, and what was found.
    `activity`: its Cycling or Gait analysis (or None); `result`: the tables of what was found;
    `item`: its compare.Analysis (cycles and mean curves); `series`: per-frame values to include;
    `angles`: a template's [(name, per-frame angle)]."""
    q = s.quality()
    stretches = s.stretches()
    d = header("3d-report", evaluator, notes)
    d["session"] = {
        "name": s.path.stem if s.path else None, "file": str(s.path) if s.path else None,
        "activity": kind, "template": s.template,
        "settings": {"smoothing_hz": s.smooth_hz or 0.0, "feet": s.feet or "free"},
        "videos": [{"file": str(v.meta.path), "width": v.meta.width, "height": v.meta.height, "fps": v.meta.fps,
                    "frames": v.frame_count, "offset_s": (s.offsets[i] if s.offsets else None),
                    "handheld": bool(v.handheld)} for i, v in enumerate(s.views)],
        "rows": int(len(s.t)), "duration_s": float(s.t[-1] - s.t[0]) if len(s.t) > 1 else 0.0,
    }
    d["patient"] = patient(s.subject, anonymous)
    d["quality"] = {"level": q["level"], "videos_disagree_cm": q["mismatch_cm"],
                    "videos_disagree_p90_cm": q["mismatch_p90_cm"], "leg_bones_variation_pct": q["legs_variation"],
                    "bones_variation_pct": q["bones"], "reconstructed": q["reconstructed"],
                    "corrected_by_hand": q["manual"],
                    "doubtful_stretches": sum(1 for e in stretches if not e["reviewed"]),
                    "reviewed_stretches": sum(1 for e in stretches if e["reviewed"]),
                    "doubtful_legs": [side for side, ok in (item.reliable.items() if item else []) if not ok]}
    found = {}
    if kind in ("cycling", "gait") and item is not None:
        found["cycles"] = dict(item.cycles)
        found["typical"] = {side: typical(cyc, kind) for side, cyc in item.cycles.items() if cyc}
        found["mean_cycle"] = curves(item)
    if angles:
        found["angles"] = {name: {"unit": "°", "min": np.nanmin(v), "max": np.nanmax(v), "rom": np.nanmax(v) - np.nanmin(v),
                                  "mean": np.nanmean(v)} for name, v in angles if np.isfinite(v).any()}
    if result is not None:
        found["summary"] = result.summary
        found["tables"] = [{"title": t.title, "columns": t.columns, "rows": t.rows} for t in result.tables]
    d["found"] = found
    if series:
        d["series"] = series
    return clean(d)


def comparison(items, sides, per_height: bool, evaluator: str = "", notes: str = "", anonymous: bool = False) -> dict:
    """Several analyses side by side: each one's typical values, the differences from the first,
    and whether each difference is beyond what changes from one cycle to the next."""
    from openkick.core import compare as K
    d = header("comparison", evaluator, notes)
    d["different_people"] = per_height
    d["analyses"] = [{"label": i.label, "kind": i.kind, "activity": i.activity, "file": i.path,
                      "patient": patient(i.subject, anonymous), "doubtful_legs": [s for s, ok in i.reliable.items() if not ok],
                      "note": i.note or None} for i in items]
    if not items:
        return clean(d)
    M, keys = measures(items[0].activity)
    legs = {}
    for side in sides:
        rows = []
        for key in keys:
            name, unit = M.MEASURES[key]
            typ = [C.typical(K._value(c, key, unit, i, per_height) for c in (i.cycles.get(side) or [])) for i in items]
            if not any(np.isfinite(m) for m, _ in typ):
                continue
            ns = [sum(1 for c in (i.cycles.get(side) or []) if np.isfinite(K._value(c, key, unit, i, per_height))) for i in items]
            a, sa = typ[0]
            rows.append({"key": key, **_named(name),
                         "unit": "% height" if per_height and unit in K.LENGTH_UNITS else unit,
                         "values": [{"label": i.label, "median": m, "spread": sd, "n": n} for i, (m, sd), n in zip(items, typ, ns)],
                         "differences": [{"label": i.label, "minus": items[0].label, "value": b - a,
                                          "beyond_variation": bool(np.isfinite(a) and np.isfinite(b) and C.beyond(a, sa, b, sb))}
                                         for i, (b, sb) in zip(items[1:], typ[1:])]})
        legs[side] = rows
    d["legs"] = legs
    d["mean_cycle"] = {i.label: curves(i) for i in items}
    return clean(d)
