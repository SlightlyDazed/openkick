"""Gait events and strides, shared by the 2D gait plugin and the 3D reconstruction.

Initial contact and toe off are found from where the heel and the toes are against the pelvis,
along the direction of walking: the heel is furthest ahead of it when the foot lands and the
toes are furthest behind it when the foot leaves the ground (Zeni et al., 2008). It works
overground and on a treadmill, and needs only the position of each point along that direction:
pixels in a side-on video, metres in 3D."""

from __future__ import annotations

import warnings

import numpy as np

from openkick.i18n import tr

STRIDE_S = (0.6, 2.5)     # a walking stride lasts about a second; outside this it is not one stride
STANCE = (0.45, 0.80)     # share of the stride with the foot on the ground (about 0.6 when healthy)
TRAVEL = 0.5              # the pelvis moves at least this fraction of the leg in a stride: walking
#                           overground (on a treadmill it stays put and there is no stride length)
COLOR_STANCE = "#10B981"
COLOR_SWING = "#2563EB"

# What can be reported for each stride: key, name, unit. Angles are measured inside the joint, so
# 180° is a straight limb; "*" marks what the detector places least reliably.
MEASURES = {
    "stride_time": ("Stride time", "s"),
    "cadence": ("Cadence", "steps/min"),
    "stance": ("Stance", "%"),
    "swing": ("Swing", "%"),
    "knee_contact": ("Knee at contact", "°"),
    "knee_swing": ("Knee, swing", "°"),              # most bent during the swing
    "knee_rom": ("Knee range", "°"),
    "hip_rom": ("Hip range", "°"),
    "ankle_rom": ("Ankle range *", "°"),
    "trunk": ("Trunk lean", "°"),                    # forward of the vertical
    "step_length": ("Step length", "m"),             # from the other heel to this one, both on the ground
    "step_width": ("Step width", "cm"),
    "stride_length": ("Stride length", "m"),
    "speed": ("Walking speed", "m/s"),
}
BY_SIDE = ("stride_time", "stance", "knee_contact", "knee_swing", "knee_rom", "hip_rom", "step_length")
SHOWN = ("stride_time", "cadence", "stance", "knee_contact", "knee_swing")


def decimals(unit: str) -> int:
    return 2 if unit in ("s", "m", "m/s") else 1


def unit(key: str) -> str:
    return MEASURES[key][1]


def name(key: str) -> str:
    return tr(MEASURES[key][0])


def show(v, key: str) -> str:
    u = unit(key)
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.1f}°" if u == "°" else f"{v:.{decimals(u)}f} {tr(u)}"


def typical(values) -> tuple[float, float]:
    """Median and the usual distance from it (a standard deviation that one bad stride cannot inflate)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return float("nan"), float("nan")
    mid = float(np.median(v))
    return mid, float(1.4826 * np.median(np.abs(v - mid)))


def vertex(y: np.ndarray, i: int) -> float:
    """Where the peak near sample `i` really is: it falls between two frames, so it is read off
    the parabola through three samples."""
    if i <= 0 or i >= len(y) - 1 or not np.isfinite(y[i - 1:i + 2]).all():
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    bend = a - 2 * b + c
    return float(i) if abs(bend) < 1e-9 else i + 0.5 * (a - c) / bend


def events(rel: np.ndarray, fps: float, leg: float) -> list[tuple[int, float]]:
    """The peaks of one foot point's position against the pelvis (positive ahead for the heel,
    behind for the toes): (nearest frame, frame with its fraction)."""
    from scipy.signal import find_peaks
    seen = np.isfinite(rel)
    if seen.sum() < fps / 2:
        return []
    frames = np.arange(len(rel))
    filled = np.interp(frames, frames[seen], rel[seen])
    peaks, _ = find_peaks(filled, distance=max(2, int(0.4 * fps)), prominence=0.2 * leg)
    return [(int(i), vertex(filled, int(i))) for i in peaks if seen[i]]


def pair(contacts: list, offs: list, t: np.ndarray, seen: np.ndarray) -> list[dict]:
    """Strides of one foot: from one initial contact to the next, with one toe off in between.
    Each is {"a", "fa", "b", "fb", "o", "fo", "seconds", "stance"} (frames within the arrays given,
    the f* ones with their fraction; stance as a share of the stride)."""
    frames = np.arange(len(t))
    gaps = np.diff([a for a, _ in contacts])
    out = []
    for n, ((a, fa), (b, fb)) in enumerate(zip(contacts[:-1], contacts[1:])):
        usual = float(np.median(gaps[max(0, n - 3):n + 4]))       # the pace may change along the clip
        seconds = float(np.interp(fb, frames, t) - np.interp(fa, frames, t))
        inside = [(o, fo) for o, fo in offs if a < o < b]
        if (not STRIDE_S[0] <= seconds <= STRIDE_S[1] or not 0.6 * usual <= b - a <= 1.6 * usual
                or len(inside) != 1 or seen[a:b + 1].mean() < 0.8):
            continue
        o, fo = inside[0]
        stance = (float(np.interp(fo, frames, t)) - float(np.interp(fa, frames, t))) / seconds
        if STANCE[0] <= stance <= STANCE[1]:
            out.append({"a": a, "fa": fa, "b": b, "fb": fb, "o": o, "fo": fo, "seconds": seconds, "stance": stance})
    return out


def joint_measures(s: dict, knee: np.ndarray, hip: np.ndarray, ankle: np.ndarray, trunk: np.ndarray) -> dict:
    """The angles of one stride (`s` from pair()), from per-frame series of the stride's leg."""
    a, b, o = s["a"], s["b"], s["o"]
    sl = slice(a, b + 1)
    frames = np.arange(len(knee))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"stride_time": s["seconds"], "cadence": 120.0 / s["seconds"],
                "stance": 100 * s["stance"], "swing": 100 * (1 - s["stance"]),
                "knee_contact": float(np.interp(s["fa"], frames, knee)),
                "knee_swing": float(np.nanmin(knee[o:b + 1])),
                "knee_rom": float(np.nanmax(knee[sl]) - np.nanmin(knee[sl])),
                "hip_rom": float(np.nanmax(hip[sl]) - np.nanmin(hip[sl])),
                "ankle_rom": float(np.nanmax(ankle[sl]) - np.nanmin(ankle[sl])),
                "trunk": float(np.nanmean(trunk[sl]))}


# The eight phases of the gait cycle (Perry and Burnfield, Rancho Los Amigos): key, name, short
# name for narrow blocks, colour (greens with the foot on the ground, blues in the air) and where
# it usually starts, as a share of the stride. The usual starts stand in for an event that cannot
# be found, and say what to expect: they shift with walking speed and from one person to another.
PERRY = [
    ("ic", "Initial contact", "IC", "#065F46", 0.00),
    ("lr", "Loading response", "LR", "#047857", 0.02),
    ("mst", "Mid stance", "MSt", "#059669", 0.12),
    ("tst", "Terminal stance", "TSt", "#10B981", 0.31),
    ("psw", "Pre-swing", "PSw", "#34D399", 0.50),
    ("isw", "Initial swing", "ISw", "#1D4ED8", 0.62),
    ("msw", "Mid swing", "MSw", "#2563EB", 0.75),
    ("tsw", "Terminal swing", "TSw", "#3B82F6", 0.87),
]
# the event that starts each phase after initial contact, in order along the stride
EVENTS = ("lr", "ofo", "hr", "oic", "to", "fa", "tv")
HEEL_RISE = 0.015         # the heel is up once it is this fraction of the leg above where it rested
#                           (it peels off slowly: a higher mark finds it late, once the other foot is down)


def short_name(name: str) -> str:
    """Short name of a gait phase (any other name: its first letter), for narrow timeline blocks."""
    for _, full, short, _, _ in PERRY:
        if name in (full, tr(full)):
            return tr(short)
    return name[:1].upper()


def _first(cond: np.ndarray, start: int, stop: int) -> int | None:
    """First frame in [start, stop) where `cond` holds."""
    hit = np.flatnonzero(cond[start:stop])
    return int(start + hit[0]) if len(hit) else None


def gait_events(s: dict, own: dict, other: dict, forward: float, leg: float) -> dict:
    """The seven events that split one stride (`s` from pair(), frames within the arrays given)
    into Perry's eight phases. `own` and `other` hold each foot's per-frame "heel_y" (image
    vertical, down positive), "ankle_x" and "knee_x" along the picture, plus "contacts" and
    "offs" (frames). `forward` is +1 walking to the right of the picture, -1 to the left.

    lr   end of initial contact: 2 % of the stride after it (contact is an instant)
    ofo  opposite foot off: the other foot's toe off, ends the first double support
    hr   heel rise: this heel lifts off the floor
    oic  opposite initial contact: the other heel lands, begins the second double support
    to   toe off: this foot leaves the ground
    fa   feet adjacent: the swinging ankle draws level with the other one
    tv   tibia vertical: the swinging shank passes the vertical, ankle under the knee

    Returns {"events": {key: frame}, "estimated": [keys]}: an event that cannot be found, or that
    is out of order, takes its usual place between its neighbours and is listed as estimated."""
    a, o, b = int(s["a"]), int(s["o"]), int(s["b"])
    n = b - a
    ev = {"lr": a + max(1, round(0.02 * n)), "to": o}
    ev["ofo"] = next((f for f in other["offs"] if a < f < o), None)
    ev["oic"] = next((f for f in other["contacts"] if (ev["ofo"] or a) < f < o), None)
    heel = own["heel_y"]
    lo = ev["ofo"] or a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rest = np.nanmax(heel[a:o]) if np.isfinite(heel[a:o]).any() else np.nan     # the heel flat on the floor
    with np.errstate(invalid="ignore"):
        ev["hr"] = _first(rest - heel > HEEL_RISE * leg, lo, o) if np.isfinite(rest) else None
    with np.errstate(invalid="ignore"):
        ev["fa"] = _first(forward * (own["ankle_x"] - other["ankle_x"]) >= 0, o, b)
        ev["tv"] = _first(forward * (own["ankle_x"] - own["knee_x"]) >= 0, ev["fa"] or o, b)
    usual = {key: start for key, _, _, _, start in PERRY}
    usual_at = {"lr": usual["lr"], "ofo": usual["mst"], "hr": usual["tst"], "oic": usual["psw"],
                "to": usual["isw"], "fa": usual["msw"], "tv": usual["tsw"]}
    # Events are checked from the most to the least reliable: the end of contact and toe off (what
    # the stride is built on), then the other foot's, then the swinging leg's, heel rise last (it
    # peels off slowly and is the hardest to see). One that falls outside the ones already kept
    # is dropped, and every dropped event takes its usual place between its kept neighbours.
    kept = {"lr": ev["lr"], "to": ev["to"]}

    def room(key):
        i = EVENTS.index(key)
        prev = max([a] + [kept[k] for k in EVENTS[:i] if k in kept])
        nxt = min([b] + [kept[k] for k in EVENTS[i + 1:] if k in kept])
        return prev, nxt
    for key in ("ofo", "oic", "fa", "tv", "hr"):
        prev, nxt = room(key)
        if ev[key] is not None and prev < ev[key] < nxt:
            kept[key] = ev[key]
    estimated = [k for k in EVENTS if k not in kept]
    for key in estimated:
        prev, nxt = room(key)
        kept[key] = min(nxt - 1, max(prev + 1, a + round(usual_at[key] * n)))
    ev = kept
    # every phase keeps at least one frame, so that all eight show on the timeline and can be moved
    order = [a] + [ev[k] for k in EVENTS] + [b]
    for i in range(1, len(order) - 1):
        order[i] = max(order[i], order[i - 1] + 1)
    for i in range(len(order) - 2, 0, -1):
        order[i] = min(order[i], order[i + 1] - 1)
    ev = dict(zip(EVENTS, order[1:-1]))
    return {"events": ev, "estimated": estimated}


def phases(strides: list[dict], colors: list[str]) -> tuple[list, list]:
    """One phase per stride ("first" to "last" frame, toe off at "off"), each split into Perry's
    eight phases when the stride has its "events" (gait_events()), else into stance and swing."""
    from openkick.core.project import Phase
    main, sub = [], []
    for i, s in enumerate(strides, start=1):
        first, last = int(s["first"]), int(s["last"])
        main.append(Phase(name=tr("Stride {n}", n=i), start=first, end=last, color=colors[(i - 1) % len(colors)]))
        ev = s.get("events")
        if ev:
            starts = [first] + [int(ev[k]) for k in EVENTS] + [last + 1]
            for (_, full, _, color, _), a, b in zip(PERRY, starts[:-1], starts[1:]):
                if b > a:
                    sub.append(Phase(name=tr(full), start=a, end=b - 1, color=color))
            continue
        off = max(first + 1, min(last, int(s["off"])))
        sub.append(Phase(name=tr("Stance"), start=first, end=off - 1, color=COLOR_STANCE))
        sub.append(Phase(name=tr("Swing"), start=off, end=last, color=COLOR_SWING))
    return main, sub


def phase_table(strides: list[dict]) -> "Table | None":
    """How long each of the eight phases lasts, as a share of the stride, against the usual values."""
    from openkick.plugins import Table
    done = [s for s in strides if s.get("events")]
    if not done:
        return None
    rows = []
    for k, (key, full, _, _, start) in enumerate(PERRY):
        end = PERRY[k + 1][4] if k + 1 < len(PERRY) else 1.0
        shares, guessed = [], 0
        for s in done:
            starts = [s["first"]] + [s["events"][e] for e in EVENTS] + [s["last"] + 1]
            shares.append(100 * (starts[k + 1] - starts[k]) / (s["last"] + 1 - s["first"]))
            nxt = EVENTS[k] if k < len(EVENTS) else None
            guessed += bool(nxt and nxt in s["estimated"])
        mid, sd = typical(shares)
        rows.append([tr(full), f"{mid:.1f} % ± {sd:.1f}", f"{100 * (end - start):.0f} %",
                     f"{guessed} / {len(done)}" if guessed else "—"])
    return Table([tr("Phase"), tr("Share of the stride"), tr("Usual"), tr("End estimated")], rows,
                 tr("The eight phases of the gait cycle"))


# ── tables ─────────────────────────────────────────────────────────────


def spread(key: str, value: float) -> str:
    return f"± {value:.{decimals(unit(key))}f}"


def cell(group: list[dict], key: str) -> str:
    mid, sd = typical(s.get(key) for s in group)
    return f"{show(mid, key)} {spread(key, sd)}"


def cycle_table(strides: list[dict], keys) -> "Table":
    """Typical stride and how much it changes from one to the next. Keys with no value are left out."""
    from openkick.plugins import Table
    rows = []
    for key in keys:
        v = [s[key] for s in strides if key in s and np.isfinite(s[key])]
        if not v:
            continue
        mid, sd = typical(v)
        rows.append([name(key), show(mid, key), spread(key, sd), show(min(v), key), show(max(v), key)])
    return Table([tr("Measure"), tr("Typical stride"), tr("From stride to stride"), tr("Lowest"), tr("Highest")],
                 rows, tr("Gait cycle"))


def sides_table(every: list[dict]) -> "Table | None":
    """Left leg against right leg, when both have at least two strides."""
    from openkick.plugins import Table
    left = [s for s in every if s["side"] == "left"]
    right = [s for s in every if s["side"] == "right"]
    if len(left) < 2 or len(right) < 2:
        return None
    rows = []
    for key in BY_SIDE:
        (a, _), (b, _) = typical(s.get(key) for s in left), typical(s.get(key) for s in right)
        if not (np.isfinite(a) or np.isfinite(b)):
            continue
        d = decimals(unit(key))
        diff = f"{round(b - a, d) + 0.0:+.{d}f}" if np.isfinite(a) and np.isfinite(b) else "—"
        rows.append([name(key), show(a, key), show(b, key), diff])
    return Table([tr("Measure"), tr("Left"), tr("Right"), tr("Difference")], rows, tr("Left and right"))


def stride_table(strides: list[dict], phase_of=None) -> "Table":
    """One row per stride; `phase_of(stride)` names the phase it falls in, when there are phases."""
    from openkick.plugins import Table
    rows = []
    for n, s in enumerate(strides, start=1):
        rows.append([n, f"{s['time']:.2f} s"] + ([phase_of(s)] if phase_of else []) + [show(s[k], k) for k in SHOWN])
    return Table([tr("Stride"), tr("At")] + ([tr("Phase")] if phase_of else []) + [name(k) for k in SHOWN],
                 rows, tr("Stride by stride"))
