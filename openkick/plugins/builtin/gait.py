"""Gait analysis: a person walking, filmed from the side, overground or on a treadmill.

Finds the two events of every step, initial contact and toe off (see openkick.core.gait). Every
stride of the leg turned to the camera becomes a phase, split into stance and swing, as the
cycling plugin does with pedal strokes. It measures; it does not diagnose. The same analysis in
3D, from several videos, is in openkick.multiview.gait.
"""

import warnings

import numpy as np

from openkick.core import gait as G
from openkick.core import joints as J
from openkick.i18n import tr
from openkick.plugins import AnalysisPlugin, Result, Table
from openkick.plugins.builtin.cycling import _runs

MIN_STRIDES = 3           # a walk across the picture seldom holds more than three or four
MIN_STRETCH_S = 1.5       # side-on stretches shorter than this are skipped
FOOT_SIDE_ON = 0.18       # heel to toes at least this long across the picture, against the leg: the feet
#                           are seen from the side (0.23-0.31 crossing a wide picture; about 0.06 head-on)
STRIDE_SPREAD = 0.15      # longest minus shortest stride, against the median: walking varies 2-4 % from
#                           stride to stride (CV), so more than this is the recording, not the walk
SIZE_CHANGE = 0.10        # the leg's length in the picture changing more than this between strides
JOINTS = ("shoulder", "hip", "knee", "ankle", "heel", "foot_index")
KEYS = ("stride_time", "cadence", "stance", "swing", "knee_contact", "knee_swing", "knee_rom", "hip_rom",
        "hip_extension", "ankle_rom", "trunk", "stride_length", "speed")
BY_PHASE = ("stride_time", "cadence", "stance", "knee_swing", "knee_rom")


def _other(side: str) -> str:
    return "left" if side == "right" else "right"


def _side_on(ctx) -> np.ndarray:
    """True where the feet are seen from the side, which is what the gait events need. Not the
    shoulders, as for a rider on a trainer: someone crossing a wide picture is seen a little from
    the front where they come in and from behind where they leave, and the shoulders part, while
    the feet still show their whole length. Flickers shorter than half a second are ignored."""
    from scipy.ndimage import median_filter
    index, p = ctx.landmarks.INDEX, ctx.project
    xy = p.pose.xy.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        leg = np.nanmedian(np.concatenate([np.linalg.norm(xy[:, index[f"{s}_hip"]] - xy[:, index[f"{s}_ankle"]], axis=1)
                                           for s in ("left", "right")]))
        foot = np.nanmean([np.abs(xy[:, index[f"{s}_foot_index"], 0] - xy[:, index[f"{s}_heel"], 0])
                           for s in ("left", "right")], axis=0) / leg
    seen = np.isfinite(foot)
    if not seen.any():
        return seen
    size = max(3, int(p.video.fps / 2) | 1)
    return (median_filter(np.where(seen, foot, 0.0), size=size, mode="nearest") >= FOOT_SIDE_ON) & seen


def _forward(ctx, first: int, last: int) -> float:
    """+1 when the person walks towards the right of the picture, -1 towards the left: the toes
    point the way they walk, on a treadmill too."""
    index = ctx.landmarks.INDEX
    xy = ctx.project.pose.xy[first:last + 1].astype(np.float64)
    d = np.concatenate([xy[:, index[f"{s}_foot_index"], 0] - xy[:, index[f"{s}_heel"], 0] for s in ("left", "right")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(d)
    return -1.0 if np.isfinite(m) and m < 0 else 1.0


def _strides(ctx, first: int, last: int, side: str, forward: float) -> list[dict]:
    """Every stride of one foot in one side-on stretch, from one initial contact to the next."""
    A, index, p = ctx.analysis, ctx.landmarks.INDEX, ctx.project
    fps = p.video.fps
    t = ctx.times()[first:last + 1]
    xy = p.pose.xy[first:last + 1].astype(np.float64)
    P = {k: xy[:, index[f"{side}_{k}"]] for k in JOINTS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pelvis = np.nanmean([xy[:, index["left_hip"], 0], xy[:, index["right_hip"], 0]], axis=0)
        leg = float(np.nanmedian(np.linalg.norm(P["hip"] - P["ankle"], axis=1)))     # pixels
    if not np.isfinite(leg) or leg <= 0:
        return []

    def low(v):
        return A.lowpass(v, fps)
    feet = {}
    for s in (side, _other(side)):           # this foot's strides need the other foot's events too
        heel, toe = xy[:, index[f"{s}_heel"]], xy[:, index[f"{s}_foot_index"]]
        feet[s] = {"contacts": [f for f, _ in G.events(low(forward * (heel[:, 0] - pelvis)), fps, leg)],
                   "offs": [int(round(f + G.TOE_OFF_DELAY_2D * fps))
                            for _, f in G.events(low(-forward * (toe[:, 0] - pelvis)), fps, leg)],
                   "heel_y": low(heel[:, 1]), "ankle_x": low(xy[:, index[f"{s}_ankle"], 0]),
                   "knee_x": low(xy[:, index[f"{s}_knee"], 0])}
    contacts = G.events(low(forward * (P["heel"][:, 0] - pelvis)), fps, leg)
    offs = G.events(low(-forward * (P["foot_index"][:, 0] - pelvis)), fps, leg)
    if len(contacts) < 2 or not offs:
        return []
    knee, hip, ankle = (low(A.despike(J.clinical(xy, j, side, forward, "vertical"), fps, A.DESPIKE_FLOOR_DEG))
                        for j in ("knee", "hip", "ankle"))
    up = P["shoulder"] - P["hip"]
    trunk = np.degrees(np.arctan2(forward * up[:, 0], -up[:, 1]))
    pelvis = low(pelvis)
    frames = np.arange(len(knee))
    legs = np.linalg.norm(P["hip"] - P["ankle"], axis=1)      # the leg's size in the picture, for _checks
    cal = p.calibration
    out = []
    for s in G.pair(contacts, offs, t, np.isfinite(knee), G.TOE_OFF_DELAY_2D):
        travel = abs(float(np.interp(s["fb"], frames, pelvis) - np.interp(s["fa"], frames, pelvis)))
        length = travel / cal.px_per_m if cal and travel >= G.TRAVEL * leg else float("nan")
        found = G.gait_events(s, feet[side], feet[_other(side)], forward, leg)
        out.append({"side": side, "first": first + s["a"], "last": first + s["b"] - 1, "off": first + s["o"],
                    "time": float(t[s["a"]]), **G.joint_measures(s, knee, hip, ankle, trunk),
                    "stride_length": length, "speed": length / s["seconds"],
                    "leg_px": float(np.nanmedian(legs[s["a"]:s["b"] + 1])),
                    "events": {k: first + f for k, f in found["events"].items()}, "estimated": found["estimated"]})
    return out


class GaitCycle(AnalysisPlugin):
    id = "gait"
    name = {"es": "Análisis de la marcha", "en": "Gait analysis"}
    description = {"es": "Encuentra cada zancada de un video de perfil, la divide en apoyo y balanceo y "
                         "mide rodilla, cadera y ritmo.",
                   "en": "Finds every stride of a side-on video, splits it into stance and swing and "
                         "measures knee, hip and pace."}

    def run(self, ctx):
        from openkick.core.project import PHASE_COLORS
        p, A = ctx.project, ctx.analysis
        fps = p.video.fps
        stretches = _runs(_side_on(ctx), int(MIN_STRETCH_S * fps))
        strides, every = [], []         # the leg turned to the camera; both legs
        for a, b in stretches:
            near, forward = A.facing_side(p, a, b), _forward(ctx, a, b)
            mine = _strides(ctx, a, b, near, forward)
            strides += mine
            every += mine + _strides(ctx, a, b, _other(near), forward)
        checks, advice = self._checks(ctx, stretches, strides)
        if len(advice) == 1:
            advice.insert(0, tr("Recording check, 1 point to review:"))
        elif advice:
            advice.insert(0, tr("Recording check, {n} points to review:", n=len(advice)))
        if len(strides) < 2:
            return Result(title=self.label, tables=[checks], summary="\n".join([tr(
                "No strides were found. This analysis needs a person walking, filmed from the side, "
                "with both feet in the picture.")] + advice))

        # ── phases and sub-phases, when the project has none yet ──
        main, sub = G.phases(strides, PHASE_COLORS)
        created = not p.phases
        if created:
            p.phases = main
        if not p.subphases:
            p.subphases = sub

        tables = [G.cycle_table(strides, KEYS)]
        eight = G.phase_table(strides)
        if eight:
            tables.append(eight)
        sides = G.sides_table(every)
        if sides:
            tables.append(sides)
        # phases set by the user (several strides each): comparison
        groups = [(ph, [s for s in strides if ph.start <= s["first"] and s["last"] <= ph.end]) for ph in p.phases]
        if any(len(g) >= 2 for _, g in groups):
            tables.append(Table([tr("Phase"), tr("Strides")] + [G.name(k) for k in BY_PHASE],
                                [[ph.name, len(g)] + [G.cell(g, k) if g else "—" for k in BY_PHASE] for ph, g in groups],
                                tr("By phase")))
        checks.detail = not advice
        tables.append(checks)
        strides_table = G.stride_table(strides, lambda s: next(
            (q.name for q in p.phases if q.start <= s["first"] and s["last"] <= q.end), "—"))
        strides_table.detail = True
        tables.append(strides_table)

        sides = {s["side"] for s in strides}
        side = tr("Left and right sides") if len(sides) > 1 else tr("Left side") if "left" in sides else tr("Right side")
        cadence, _ = G.typical(s["cadence"] for s in strides)
        stance, _ = G.typical(s["stance"] for s in strides)
        summary = [
            tr("{side} · {n} strides · {c} steps/min · stance {s} % of the stride.", side=side, n=len(strides),
               c=f"{cadence:.0f}", s=f"{stance:.0f}"),
            tr("Initial contact and toe off are found from where the heel and the toes are against the pelvis "
               "(Zeni et al., 2008). Angles are clinical ones: 0° is a straight knee and a thigh in line with the "
               "body, positive is flexion and a negative hip is behind the body. The measures marked * are the "
               "ones the detector places least reliably.")]
        guessed = sum(len(s.get("estimated", [])) for s in strides)
        summary.append(tr("Each stride is split into the eight phases of Perry and Burnfield. You can drag their "
                          "edges on the timeline."))
        if guessed:
            summary.append(tr("{n} phase edges could not be seen in the video and sit where they usually are: "
                              "check them on the timeline.", n=guessed))
        if created:
            summary.append(tr("Phases were detected automatically. You can adjust them in the Phases tab."))
        return Result(title=self.label, summary="\n".join(summary + advice), tables=tables)

    def _checks(self, ctx, stretches, strides) -> tuple[Table, list[str]]:
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
            tr("put the camera level with the hip, square to the walkway."))
        if stretches:
            w, h = p.video.width, p.video.height
            feet = [index[f"{s}_{k}"] for s in ("left", "right") for k in ("ankle", "heel", "foot_index")]
            frames = np.concatenate([np.arange(a, b + 1) for a, b in stretches])
            pts = p.pose.xy[frames][:, feet].astype(np.float64)
            with np.errstate(invalid="ignore"):
                inside = float(((pts[..., 0] > 0) & (pts[..., 0] < w - 1) &
                                (pts[..., 1] > 0) & (pts[..., 1] < h - 1)).all(axis=1).mean())
            seen = float((p.pose.visibility[frames][:, feet] >= 0.5).all(axis=1).mean())
            add(tr("Feet inside the picture"), f"{inside * 100:.0f} %", ok if inside >= 0.95 else review,
                tr("move the camera back until both feet stay inside the picture."))
            add(tr("Feet detected with confidence"), f"{seen * 100:.0f} %", ok if seen >= 0.8 else review,
                tr("correct the doubtful frames (Tab jumps to the next one), or film with more light and tighter clothes."))
        add(tr("Strides"), str(len(strides)), ok if len(strides) >= MIN_STRIDES else review,
            tr("record at least {n} strides: film from further away, or on a treadmill.", n=MIN_STRIDES))
        lengths = [s["stride_length"] for s in strides if np.isfinite(s["stride_length"])]
        if len(lengths) >= 2:
            # one calibration holds on one line: someone walking nearer to or further from the
            # camera changes size in the picture, and their strides with it
            lo, hi, mid = min(lengths), max(lengths), float(np.median(lengths))
            sizes = [s["leg_px"] for s in strides if np.isfinite(s["stride_length"]) and np.isfinite(s["leg_px"])]
            change = (max(sizes) - min(sizes)) / float(np.median(sizes)) if len(sizes) >= 2 else 0.0
            steady = (hi - lo) <= STRIDE_SPREAD * mid
            add(tr("Stride length steady"), f"{lo:.2f}–{hi:.2f} m", ok if steady else review,
                tr("the person's size in the picture changes {n} % from one stride to another: they come nearer to "
                   "or go away from the camera, and a calibration only holds on one line. Put the camera square to "
                   "the walkway and the calibration reference on the line walked.", n=f"{100 * change:.0f}")
                if change > SIZE_CHANGE else
                tr("the strides differ more than walking does. Check that the calibration reference lies on the "
                   "line walked and that the camera is square to it."))
        if len(strides) >= 2:
            mid, _ = G.typical(s["stride_time"] for s in strides)
            per = fps * mid if np.isfinite(mid) else 0.0
            add(tr("Frames in each stride"), f"{per:.0f}", ok if per >= 25 else review,
                tr("record at 60 frames per second."))
        return Table([tr("Check"), tr("Found"), tr("Verdict")], rows, tr("Recording check")), advice


PLUGINS = [GaitCycle()]
