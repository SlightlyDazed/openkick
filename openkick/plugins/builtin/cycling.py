"""Pedal stroke analysis: a rider filmed from the side, on a trainer.

Finds every pedal stroke on the side of the body turned to the camera, measures the usual
bike-fit angles in each one and tells how much they change from stroke to stroke. When the
project has phases (one per saddle setting, say) it compares them. It measures; it does not
prescribe a position.
"""

import warnings

import numpy as np

from openkick.i18n import tr
from openkick.plugins import AnalysisPlugin, Result, Table

MIN_STROKES = 10          # fewer strokes than this do not show what a typical one looks like
MIN_COMPARE = 5           # strokes a phase needs to be compared with another
MIN_STRETCH_S = 2.0       # side-on stretches shorter than this are skipped
MAX_DRIFT = 0.35          # the hip may wander this fraction of the trunk length; beyond it, the rider or
#                           the camera is travelling (hand-held cameras on a trainer stay under 0.3)
JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel", "foot_index")

# What is reported for each stroke: key, name, unit. "Bottom" and "top" are the two ends of the
# stroke (leg most stretched, leg most bent); "*" marks what the detector places least reliably.
# The names are short because they are also column headings.
MEASURES = [
    ("cadence", "Cadence", "rpm"),
    ("knee_bottom", "Knee, bottom", "°"),
    ("knee_top", "Knee, top", "°"),
    ("hip_top", "Hip, top", "°"),
    ("trunk", "Trunk tilt", "°"),                    # above the horizontal
    ("shoulder", "Shoulder", "°"),
    ("elbow", "Elbow *", "°"),
    ("foot_bottom", "Toes down, bottom *", "°"),     # sole against the horizontal, positive with the toes lower
    ("ankle_travel", "Ankle travel *", "°"),
    ("hip_travel", "Hip bounce (% trunk)", "%"),     # up-and-down travel of the hip against the trunk length
]
HIP_TRAVEL_CM = "Hip bounce (cm)"                     # its name once the project has a calibration
BY_PHASE = ("cadence", "knee_bottom", "knee_top", "hip_top", "trunk")


def _runs(mask: np.ndarray, shortest: int) -> list[tuple[int, int]]:
    """(first, last) frame of every run of True at least `shortest` frames long."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2]) if b - a >= shortest]


def _vertex(y: np.ndarray, i: int) -> tuple[float, float]:
    """Where the extreme near sample `i` really is and its value: the top and the bottom of a
    stroke fall between two frames, so they are read off the parabola through three samples."""
    if i <= 0 or i >= len(y) - 1 or not np.isfinite(y[i - 1:i + 2]).all():
        return float(i), float(y[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    bend = a - 2 * b + c
    if abs(bend) < 1e-9:
        return float(i), float(b)
    shift = 0.5 * (a - c) / bend
    return i + shift, float(b - 0.25 * (a - c) * shift)


def _descent(y: np.ndarray, start: int, stop: int, level: float) -> float | None:
    """Where the curve falls through `level` between two samples (a fraction of a frame).
    The knee closes fast there, so this instant is known far better than that of the flat
    bottom of the stroke: it is what the cadence is timed with."""
    for f in range(start, min(stop, len(y) - 1)):
        if y[f] >= level > y[f + 1]:
            return f + (y[f] - level) / (y[f] - y[f + 1])
    return None


def _typical(values) -> tuple[float, float]:
    """Median and the usual distance from it (a standard deviation that one bad stroke cannot inflate)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return float("nan"), float("nan")
    mid = float(np.median(v))
    return mid, float(1.4826 * np.median(np.abs(v - mid)))


def _strokes(ctx, first: int, last: int) -> list[dict]:
    """Every pedal stroke of one side-on stretch, from one bottom of the stroke to the next."""
    from scipy.signal import find_peaks
    A, index, p = ctx.analysis, ctx.landmarks.INDEX, ctx.project
    fps = p.video.fps
    t = ctx.times()[first:last + 1]
    side = A.facing_side(p, first, last)
    xy = p.pose.xy[first:last + 1].astype(np.float64)
    P = {k: xy[:, index[f"{side}_{k}"]] for k in JOINTS}

    def low(v):
        return A.lowpass(v, fps)
    knee = low(A.angle_between(P["hip"], P["knee"], P["ankle"]))
    seen = np.isfinite(knee)
    if seen.sum() < fps:
        return []
    frames = np.arange(len(knee))
    filled = np.interp(frames, frames[seen], knee[seen])
    swing = np.percentile(filled, 95) - np.percentile(filled, 5)
    bottoms, _ = find_peaks(filled, distance=max(2, int(0.3 * fps)), prominence=max(10.0, 0.3 * swing))
    if len(bottoms) < 2:
        return []
    hip = low(A.angle_between(P["shoulder"], P["hip"], P["knee"]))
    ankle = low(A.angle_between(P["knee"], P["ankle"], P["foot_index"]))
    shoulder = A.angle_between(P["elbow"], P["shoulder"], P["hip"])
    elbow = A.angle_between(P["shoulder"], P["elbow"], P["wrist"])
    up = P["shoulder"] - P["hip"]
    trunk = np.degrees(np.arctan2(-up[:, 1], np.abs(up[:, 0])))
    sole = P["foot_index"] - P["heel"]
    foot = low(np.degrees(np.arctan2(sole[:, 1], np.abs(sole[:, 0]))))
    hip_y = low(P["hip"][:, 1])
    size = float(np.nanmedian(np.linalg.norm(up, axis=1)))            # trunk length, pixels
    gaps = np.diff(bottoms)
    tops = [i + int(np.argmin(filled[i:j + 1])) for i, j in zip(bottoms[:-1], bottoms[1:])]
    tops.append(bottoms[-1] + int(np.argmin(filled[bottoms[-1]:bottoms[-1] + int(gaps[-1]) + 1])))
    halfway = [_descent(filled, b, k, (filled[b] + filled[k]) / 2) for b, k in zip(bottoms, tops)]
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for n, (i, j) in enumerate(zip(bottoms[:-1], bottoms[1:])):
            usual = float(np.median(gaps[max(0, n - 3):n + 4]))     # the cadence may change along the clip
            if not 0.6 * usual <= j - i <= 1.6 * usual or seen[i:j + 1].mean() < 0.8:
                continue
            s = slice(i, j + 1)
            at_i, knee_bottom = _vertex(filled, i)
            at_j, _ = _vertex(filled, j)
            a, b = halfway[n], halfway[n + 1]
            if a is not None and b is not None and 0.6 * (j - i) <= b - a <= 1.6 * (j - i):
                at_i, at_j = a, b
            _, knee_top = _vertex(filled, tops[n])
            hip_s = np.where(np.isfinite(hip[s]), hip[s], np.inf)
            _, hip_top = _vertex(hip, i + int(np.argmin(hip_s)))
            seconds = float(np.interp(at_j, frames, t) - np.interp(at_i, frames, t))
            out.append({
                "side": side, "first": first + int(i), "last": first + int(j), "time": float(t[i]), "size": size,
                "top": first + int(tops[n]),
                "cadence": 60.0 / seconds if seconds > 0 else float("nan"),
                "knee_bottom": knee_bottom, "knee_top": knee_top, "hip_top": hip_top,
                "trunk": float(np.nanmean(trunk[s])), "shoulder": float(np.nanmean(shoulder[s])),
                "elbow": float(np.nanmean(elbow[s])), "foot_bottom": float(foot[i]),
                "ankle_travel": float(np.nanmax(ankle[s]) - np.nanmin(ankle[s])),
                "hip_px": float(np.nanmax(hip_y[s]) - np.nanmin(hip_y[s])),
            })
    return out


def _show(v: float, unit: str) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.1f}°" if unit == "°" else f"{v:.1f} {unit}"


COLOR_RECOVERY = "#2563EB"
COLOR_PROPULSION = "#10B981"


def _auto_phases(strokes: list[dict], project, colors: list[str]) -> list:
    """Propose one phase per pedal stroke, synchronized with each pedaling cycle."""
    from openkick.core.project import Phase
    if not strokes:
        return []
    phases = []
    for i, s in enumerate(strokes, start=1):
        phases.append(Phase(
            name=tr("Stroke {n}", n=i),
            start=int(s["first"]),
            end=int(s["last"]),
            color=colors[(i - 1) % len(colors)],
        ))
    return phases


def _auto_subphases(strokes: list[dict], project) -> list:
    """Propose sub-phases (Recovery and Propulsion) for each pedal stroke."""
    from openkick.core.project import Phase
    if not strokes:
        return []
    subphases = []
    for s in strokes:
        first = int(s["first"])
        last = int(s["last"])
        top = int(s.get("top", first + (last - first) // 2))
        top = max(first + 1, min(last - 1, top))
        subphases.append(Phase(
            name=tr("Recovery"),
            start=first,
            end=top,
            color=COLOR_RECOVERY,
        ))
        subphases.append(Phase(
            name=tr("Propulsion"),
            start=top + 1,
            end=last,
            color=COLOR_PROPULSION,
        ))
    return subphases


class PedalStroke(AnalysisPlugin):
    id = "cycling"
    name = {"es": "Análisis de pedalada (ciclismo)", "en": "Pedal stroke analysis (cycling)"}
    description = {"es": "Mide cada pedalada de un video de perfil: rodilla, cadera, postura, cuánto varían "
                         "y en qué cambian de una fase a otra.",
                   "en": "Measures every pedal stroke of a side-on video: knee, hip, posture, how much "
                         "they vary and how they change from one phase to another."}

    def run(self, ctx):
        from openkick.core.project import PHASE_COLORS, Phase
        p, A = ctx.project, ctx.analysis
        fps = p.video.fps
        side_on = A.side_on_frames(p.pose, fps)
        stretches = _runs(side_on, int(MIN_STRETCH_S * fps))
        strokes = [s for a, b in stretches for s in _strokes(ctx, a, b)]
        cm = p.calibration is not None
        for s in strokes:       # the hip's travel: centimetres with a calibration, else against the trunk
            s["hip_travel"] = 100 * s["hip_px"] / (p.calibration.px_per_m if cm else s["size"])
        units = {k: u for k, _, u in MEASURES}
        names = {k: tr(n) for k, n, _ in MEASURES}
        if cm:
            units["hip_travel"], names["hip_travel"] = "cm", tr(HIP_TRAVEL_CM)
        checks, advice = self._checks(ctx, stretches, strokes)
        if len(advice) == 1:
            advice.insert(0, tr("Recording check, 1 point to review:"))
        elif advice:
            advice.insert(0, tr("Recording check, {n} points to review:", n=len(advice)))
        if len(strokes) < 2:
            return Result(title=self.label, tables=[checks], summary="\n".join([tr(
                "No pedal strokes were found. This analysis needs a rider filmed from the side, pedalling "
                "on a trainer, with the whole leg in the picture.")] + advice))

        # ── auto-detect phases and sub-phases when needed ──
        if len(strokes) >= 2:
            if not p.phases:
                proposed = _auto_phases(strokes, p, PHASE_COLORS)
                if proposed:
                    p.phases = proposed
            if not p.subphases:
                proposed_sub = _auto_subphases(strokes, p)
                if proposed_sub:
                    p.subphases = proposed_sub

        def cell(group, key):
            mid, spread = _typical(s[key] for s in group)
            return f"{_show(mid, units[key])} ± {spread:.1f}"

        # ── 1. Position: the bike-fit angles ──
        POSITION = ("knee_bottom", "knee_top", "hip_top", "trunk", "shoulder", "elbow",
                     "foot_bottom", "ankle_travel", "hip_travel")
        rows = []
        for key in POSITION:
            v = [s[key] for s in strokes if np.isfinite(s[key])]
            mid, spread = _typical(v)
            rows.append([names[key], _show(mid, units[key]), f"± {spread:.1f}",
                         _show(min(v, default=np.nan), units[key]), _show(max(v, default=np.nan), units[key])])
        tables = [Table([tr("Measure"), tr("Typical stroke"), tr("From stroke to stroke"), tr("Lowest"), tr("Highest")],
                        rows, tr("Position"))]

        # ── 2. Pedaling: cadence and stability ──
        rpm_mid, rpm_spread = _typical(s["cadence"] for s in strokes)
        STABILITY_KEYS = ("cadence", "knee_bottom", "knee_top", "hip_top", "trunk", "hip_travel")
        stab_rows = []
        for key in STABILITY_KEYS:
            v = [s[key] for s in strokes if np.isfinite(s[key])]
            mid, spread = _typical(v)
            lo, hi = min(v, default=np.nan), max(v, default=np.nan)
            # a spread above 5 % of the median (or 2° for angles) is flagged
            threshold = max(0.05 * abs(mid), 2.0) if units[key] == "°" else 0.05 * abs(mid) if abs(mid) > 0 else 1.0
            verdict = tr("OK") if spread <= threshold else tr("Review")
            stab_rows.append([names[key], _show(mid, units[key]), f"± {spread:.1f}",
                              _show(lo, units[key]), _show(hi, units[key]), verdict])
        tables.append(Table([tr("Measure"), tr("Typical"), tr("Variation"), tr("Lowest"), tr("Highest"), tr("Stability")],
                            stab_rows, tr("Pedaling")))

        # ── 3. Phases: comparison ──
        groups = [(ph, [s for s in strokes if ph.start <= s["first"] and s["last"] <= ph.end]) for ph in p.phases]
        if groups:
            tables.append(Table([tr("Phase"), tr("Strokes")] + [names[k] for k in BY_PHASE],
                                [[ph.name, len(g)] + [cell(g, k) if g else "—" for k in BY_PHASE] for ph, g in groups],
                                tr("By phase")))
        enough = [(ph, g) for ph, g in groups if len(g) >= MIN_COMPARE]
        for ph, g in enough[1:]:
            ref, base = enough[0]
            rows = []
            for key, _, _ in MEASURES:
                (a, sa), (b, sb) = _typical(s[key] for s in base), _typical(s[key] for s in g)
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                clear = abs(b - a) > max(2 * max(sa, sb), 1.0)
                rows.append([names[key], _show(a, units[key]), _show(b, units[key]), f"{b - a:+.1f}",
                             tr("Yes") if clear else tr("No")])
            tables.append(Table([tr("Measure"), ref.name, ph.name, tr("Difference"), tr("Beyond the variation?")],
                                rows, tr("{b} compared with {a}", a=ref.name, b=ph.name)))

        # ── 4. Subphases: Recovery vs Propulsion ──
        if strokes:
            rec_times, rec_pcts = [], []
            prop_times, prop_pcts = [], []
            for s in strokes:
                total_f = s["last"] - s["first"]
                if total_f > 0:
                    rec_f = s["top"] - s["first"]
                    prop_f = s["last"] - s["top"]
                    rec_times.append(rec_f / fps)
                    rec_pcts.append(100.0 * rec_f / total_f)
                    prop_times.append(prop_f / fps)
                    prop_pcts.append(100.0 * prop_f / total_f)
            
            rmid_t, _ = _typical(rec_times)
            rmid_p, rspread_p = _typical(rec_pcts)
            pmid_t, _ = _typical(prop_times)
            pmid_p, pspread_p = _typical(prop_pcts)

            subphase_rows = [
                [tr("Recovery"), f"{rmid_t:.2f} s", f"{rmid_p:.1f} %", f"± {rspread_p:.1f} %"],
                [tr("Propulsion"), f"{pmid_t:.2f} s", f"{pmid_p:.1f} %", f"± {pspread_p:.1f} %"]
            ]
            tables.append(Table([tr("Subphase"), tr("Duration"), tr("Percentage"), tr("Variation")],
                                subphase_rows, tr("Pedal Subphases")))

        # ── 5. Recording check ──
        tables.append(checks)

        # ── 6. Stroke by stroke ──
        shown = ("cadence", "knee_bottom", "knee_top", "hip_top", "foot_bottom")
        rows = []
        for n, s in enumerate(strokes, start=1):
            ph = next((q.name for q in p.phases if q.start <= s["first"] and s["last"] <= q.end), "—")
            rows.append([n, f"{s['time']:.2f} s"] + ([ph] if p.phases else []) + [_show(s[k], units[k]) for k in shown])
        tables.append(Table([tr("Stroke"), tr("At")] + ([tr("Phase")] if p.phases else []) + [names[k] for k in shown],
                            rows, tr("Stroke by stroke")))

        sides = {s["side"] for s in strokes}
        side = tr("Left and right sides") if len(sides) > 1 else tr("Left side") if "left" in sides else tr("Right side")
        summary = [
            tr("{side} · {s} s measured · {n} pedal strokes at {rpm} rpm.", side=side, n=len(strokes),
               s=f"{sum(b - a + 1 for a, b in stretches) / fps:.0f}", rpm=f"{rpm_mid:.0f}"),
            tr("Bottom and top are the two ends of the stroke: leg most stretched and leg most bent. Angles are "
               "measured inside the joint, so 180° is a straight limb. The measures marked * are the ones the "
               "detector places least reliably.")]
        if p.phases:
            summary.append(tr("Phases were detected automatically. You can adjust them in the Phases tab."))
        return Result(title=self.label, summary="\n".join(summary + advice), tables=tables)

    def _checks(self, ctx, stretches, strokes) -> tuple[Table, list[str]]:
        """Is this recording fit to be measured? One row for each thing looked at, and what to do
        about the ones that are not right."""
        p, index = ctx.project, ctx.landmarks.INDEX
        fps = p.video.fps
        total = p.frame_count / fps if fps else 0.0
        measured = sum(b - a + 1 for a, b in stretches) / fps if fps else 0.0
        ok, partial, review = tr("OK"), tr("Partial"), tr("Review")
        rows, advice = [], []

        def add(what, found, verdict, to_do):
            rows.append([what, found, verdict])
            if verdict != ok:
                advice.append(f"• {what}: {to_do}")

        share = measured / total if total else 0.0
        add(tr("Filmed from the side"), tr("{a} s of {b} s", a=f"{measured:.0f}", b=f"{total:.0f}"),
            ok if share >= 0.8 else partial if share >= 0.2 else review,
            tr("only the side-on parts are measured.") if share >= 0.2 else
            tr("put the camera level with the hip, square to the bike."))
        if stretches:
            w, h = p.video.width, p.video.height
            fine, sure, moved = [], [], []
            for a, b in stretches:
                side = ctx.analysis.facing_side(p, a, b)
                pts = p.pose.xy[a:b + 1][:, [index[f"{side}_{k}"] for k in JOINTS]].astype(np.float64)
                with np.errstate(invalid="ignore"):
                    fine.append(((pts[..., 0] > 0) & (pts[..., 0] < w - 1) &
                                 (pts[..., 1] > 0) & (pts[..., 1] < h - 1)).all(axis=1))
                leg = [index[f"{side}_{k}"] for k in ("hip", "knee", "ankle")]
                sure.append((p.pose.visibility[a:b + 1][:, leg] >= 0.5).all(axis=1))
                hip = pts[:, JOINTS.index("hip")]
                trunk = np.nanmedian(np.linalg.norm(pts[:, JOINTS.index("shoulder")] - hip, axis=1))
                step = max(1, int(fps))           # where the hip is, second by second
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    spots = np.array([np.nanmean(hip[i:i + step], axis=0) for i in range(0, len(hip) - step // 2, step)])
                    if len(spots) > 1 and trunk > 0:
                        moved.append(float(np.max(np.nanmax(spots, axis=0) - np.nanmin(spots, axis=0)) / trunk))
            inside, seen = float(np.concatenate(fine).mean()), float(np.concatenate(sure).mean())
            add(tr("Body inside the picture"), f"{inside * 100:.0f} %", ok if inside >= 0.95 else review,
                tr("move the camera back until shoulder, hands and feet stay inside the picture."))
            add(tr("Leg detected with confidence"), f"{seen * 100:.0f} %", ok if seen >= 0.9 else review,
                tr("correct the doubtful frames (Tab jumps to the next one), or film with more light and tighter clothes."))
            if moved and np.isfinite(max(moved)):
                add(tr("Rider stays in place"), tr("{n} % of the trunk", n=f"{max(moved) * 100:.0f}"),
                    ok if max(moved) <= MAX_DRIFT else review,
                    tr("use a trainer and a tripod, so that the rider does not travel across the picture."))
        add(tr("Pedal strokes"), str(len(strokes)), ok if len(strokes) >= MIN_STROKES else review,
            tr("record at least 20 seconds of steady pedalling."))
        if len(strokes) >= 2:
            rpm, spread = _typical(s["cadence"] for s in strokes)
            add(tr("Steady cadence"), f"{rpm:.0f} rpm ± {spread:.1f}", ok if spread <= 0.05 * rpm else review,
                tr("hold one cadence while recording."))
            per = fps * 60 / rpm if rpm > 0 else 0.0
            add(tr("Frames in each stroke"), f"{per:.0f}", ok if per >= 16 else review,
                tr("record at 60 frames per second, or pedal more slowly."))
        return Table([tr("Check"), tr("Found"), tr("Verdict")], rows, tr("Recording check")), advice


PLUGINS = [PedalStroke()]
