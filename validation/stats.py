"""Paper-style statistics from the results already computed (no video is processed here).

    python -m validation.stats          ->  validation/results/study_summary.txt

- Error as mean absolute error (MAE) with a 95 % confidence interval. The interval comes from
  resampling PARTICIPANTS (bootstrap), not recordings: the recordings of one person are not
  independent, so treating them as such would make the interval too narrow.
- Agreement in the Bland-Altman way: bias and 95 % limits of agreement over all frames.
- Reliability of what a clinician reads off a curve: range of motion and peak of each recording,
  OpenKick against the laboratory, as an intraclass correlation ICC(A,1) (two-way, absolute
  agreement, single measurement) and the mean difference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from validation import opencap as V

OUT = Path(__file__).resolve().parent / "results"
KINDS = ("squats", "STS", "DJ", "walking")
LABEL = {"squats": "squat", "STS": "sit-to-stand", "DJ": "drop jump", "walking": "walking", "all": "all"}
RNG = np.random.default_rng(2026)


def boot_ci(per_subject: dict, n: int = 5000) -> tuple[float, float, float]:
    """Mean of the participants' means, and its 95 % interval resampling participants."""
    means = np.array([np.mean(v) for v in per_subject.values() if len(v)])
    if len(means) < 2:
        return (float(means.mean()) if len(means) else float("nan")), float("nan"), float("nan")
    draws = RNG.choice(means, size=(n, len(means)), replace=True).mean(axis=1)
    return float(means.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def icc_a1(a, b) -> float:
    """ICC(A,1): do two methods give the same number for each recording? 1 = identical."""
    Y = np.column_stack([a, b]).astype(float)
    Y = Y[np.isfinite(Y).all(axis=1)]
    n, k = Y.shape
    if n < 3:
        return float("nan")
    grand = Y.mean()
    msr = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse)))


def by_subject(rows: list, value) -> dict:
    out = {}
    for r in rows:
        v = value(r)
        if v is not None and np.isfinite(v):
            out.setdefault(r["subject"], []).append(v)
    return out


def fmt_ci(t: tuple) -> str:
    return f"{t[0]:5.1f} [{t[1]:4.1f}, {t[2]:4.1f}]"


def pooled_agreement(tag: str, rows: list, joint: str) -> tuple[float, float, float, int]:
    """Bias and limits of agreement over every frame of the given recordings."""
    errs = []
    for r in rows:
        f = V.CACHE / "study" / f"{tag}_{r['subject']}_{r['trial']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for side in "rl":
            if f"{joint}_{side}_est" in z.files:
                e = z[f"{joint}_{side}_est"] - z[f"{joint}_{side}_ref"]
                errs.append(e[np.isfinite(e)])
    if not errs:
        return float("nan"), float("nan"), float("nan"), 0
    e = np.concatenate(errs)
    sd = float(np.std(e, ddof=1))
    return float(e.mean()), float(e.mean() - 1.96 * sd), float(e.mean() + 1.96 * sd), len(e)


def table_3d(rows: list, joints, tag: str | None = "base") -> list[str]:
    out = [f"{'movement':14s} {'joint':9s} {'n':>3s}  {'MAE° [95 % CI]':>20s}  {'bias°':>6s}  {'limits of agreement°':>22s}  {'r':>5s}  "
           f"{'ROM diff°':>10s} {'ICC ROM':>8s} {'ICC peak':>9s}"]
    for kind in KINDS + ("all",):
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        for j in joints:
            have = [r for r in rs if j in r]
            if not have:
                continue
            ci = boot_ci(by_subject(have, lambda r: r[j]["mae"]))
            bias, lo, hi, _ = pooled_agreement(tag, have, j) if tag else (np.mean([r[j]["bias"] for r in have]), np.nan, np.nan, 0)
            rom = np.mean([r[j]["rom_est"] - r[j]["rom_ref"] for r in have])
            out.append(f"{LABEL[kind]:14s} {j:9s} {len(have):3d}  {fmt_ci(ci):>20s}  {bias:+6.1f}  "
                       f"{(f'{lo:+6.1f} to {hi:+5.1f}' if np.isfinite(lo) else '—'):>22s}  {np.mean([r[j]['r'] for r in have]):5.2f}  "
                       f"{rom:+10.1f} {icc_a1([r[j]['rom_est'] for r in have], [r[j]['rom_ref'] for r in have]):8.2f} "
                       f"{icc_a1([r[j]['peak_est'] for r in have], [r[j]['peak_ref'] for r in have]) if 'peak_est' in have[0][j] else float('nan'):9.2f}")
        out.append("")
    return out


def load_study(name: str):
    f = OUT / f"study_{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def section_2d() -> list[str]:
    out = ["", "2D, ONE CAMERA (side views Cam0 and Cam4). MAE° [95 % CI over participants]; ICC of the range of motion"]
    engines = ["mediapipe", "rtmpose-lightweight", "rtmpose-balanced", "rtmpose-performance"]
    for joint in ("knee", "hip", "ankle"):
        out.append(f"\n  {joint}")
        out.append(f"  {'engine':22s}" + "".join(f"{LABEL[k]:>26s}" for k in KINDS))
        for e in engines:
            line = f"  {e:22s}"
            for k in KINDS:
                f = OUT / f"{e}_{k}.csv"
                if not f.exists():
                    line += f"{'—':>26s}"
                    continue
                rows = [r for r in csv.DictReader(open(f)) if r["joint"] == f"{joint}_aligned"]
                per = {}
                for r in rows:
                    per.setdefault(r["subject"], []).append(float(r["mae"]))
                icc = icc_a1([float(r["rom_est"]) for r in rows], [float(r["rom_ref"]) for r in rows])
                line += f"{fmt_ci(boot_ci(per)):>19s} {icc:5.2f} "
            out.append(line)
    return out


def section_opensim() -> list[str]:
    f = OUT / "opensim_ik_Cam0-Cam4.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text())
    out = ["", "OPENSIM JOINT ANGLES (two cameras -> OpenKick -> OpenSim) against the laboratory's OpenSim angles.",
           "MAE° [95 % CI over participants], bias°, ICC of the range of motion",
           f"{'movement':14s} {'':9s}" + "".join(f"{c:>34s}" for c in ("knee", "hip flexion", "hip adduction", "hip rotation", "ankle"))]
    for who in ("openkick", "opencap"):
        for kind in KINDS + ("all",):
            rs = [r for r in rows if (kind == "all" or r["kind"] == kind) and r[who]]
            line = f"{LABEL[kind]:14s} {who:9s}"
            for c in ("knee", "hip_flexion", "hip_adduction", "hip_rotation", "ankle"):
                ci = boot_ci(by_subject(rs, lambda r: r[who][c]["mae"]))
                icc = icc_a1([r[who][c]["rom_est"] for r in rs], [r[who][c]["rom_ref"] for r in rs])
                line += f"{fmt_ci(ci):>19s} {np.mean([r[who][c]['bias'] for r in rs]):+5.1f} {icc:5.2f}   "
            out.append(line)
        out.append("")
    return out


def section_fps() -> list[str]:
    base, low = load_study("baseline"), load_study("fps30")
    if not base or not low:
        return []
    out = ["", "FRAME RATE: the same recordings at 60 and at 30 frames per second (every other frame). MAE° knee / hip / ankle",
           f"{'movement':14s} {'60 fps':>22s} {'30 fps':>22s}   {'camera rotation error° 60 / 30':>32s}   {'sync error ms (median |.|) 60 / 30':>36s}"]
    for kind in KINDS + ("all",):
        a = [r for r in base if kind == "all" or r["kind"] == kind]
        b = [r for r in low if kind == "all" or r["kind"] == kind]
        g = lambda rs: " / ".join(f"{np.mean([r[j]['mae'] for r in rs]):4.1f}" for j in ("knee", "hip", "ankle"))  # noqa: E731
        out.append(f"{LABEL[kind]:14s} {g(a):>22s} {g(b):>22s}   {np.mean([r['rotation_deg'] for r in a]):14.1f} / {np.mean([r['rotation_deg'] for r in b]):4.1f}"
                   f"            {np.median(np.abs([r['offset_err_ms'] for r in a])):14.0f} / {np.median(np.abs([r['offset_err_ms'] for r in b])):3.0f}")
    return out


def section_pairs() -> list[str]:
    rows = load_study("pairs")
    if not rows:
        return []
    out = ["", "CAMERA PAIRS (squat and walking). MAE° knee / hip / ankle: board calibration (ceiling) vs OpenKick without calibration",
           f"{'pair':12s} {'angle':>6s} {'distance':>9s}  {'movement':10s} {'n':>3s} {'ceiling':>20s} {'OpenKick':>20s} {'camera rot. err°':>17s} {'sync err ms':>12s} {'good':>6s}"]
    pairs = sorted({tuple(r["cams"]) for r in rows}, key=lambda c: np.mean([r["angle_between_cameras_deg"] for r in rows if tuple(r["cams"]) == c]))
    for cams in pairs:
        for kind in ("squats", "walking"):
            rs = [r for r in rows if tuple(r["cams"]) == cams and r["kind"] == kind]
            if not rs:
                continue
            g = lambda n: " / ".join(f"{np.nanmean([r[n][j]['mae'] for r in rs]):4.1f}" for j in ("knee", "hip", "ankle"))  # noqa: E731
            out.append(f"{'+'.join(cams):12s} {np.mean([r['angle_between_cameras_deg'] for r in rs]):5.0f}° {np.mean([r['baseline_true_m'] for r in rs]):7.1f} m  "
                       f"{LABEL[kind]:10s} {len(rs):3d} {g('ceiling'):>20s} {g('body'):>20s} {np.mean([r['body']['rotation_deg'] for r in rs]):17.1f} "
                       f"{np.median(np.abs([r['body']['offset_error_ms'] for r in rs])):12.0f} {sum(r['body']['level'] == 'good' for r in rs):3d}/{len(rs)}")
    return out


def section_holdout() -> list[str]:
    h = load_study("holdout")
    if not h:
        return []
    out = ["", "HOLD-OUT. Settings chosen on participants " + ", ".join(s[7:] for s in h["tune_subjects"])
           + "; results reported on participants " + ", ".join(s[7:] for s in h["test_subjects"]) + ".",
           "Each setting tried around its default, on the tuning participants only (mean of knee, hip and ankle MAE°):"]
    for name, values in h["sweep"].items():
        out.append(f"  {name:13s} " + "   ".join(f"{v}: {d['score']:.2f}" for v, d in values.items()) + f"   -> chosen {h['chosen'][name]}")
    g = lambda d: " / ".join(f"{d[j]:4.1f}" for j in ("knee", "hip", "ankle"))  # noqa: E731
    out.append(f"\n{'MAE° knee / hip / ankle':34s}" + "".join(f"{LABEL[k]:>20s}" for k in KINDS + ("all",)))
    for key, label in (("tune_defaults", "tuning participants, defaults"), ("test_defaults", "TEST participants, defaults"),
                       ("test_chosen", "TEST participants, chosen")):
        if key in h:
            out.append(f"{label:34s}" + "".join(f"{g(h[key][k]):>20s}" for k in KINDS + ("all",)))
    return out


def main():
    lines = []
    base = load_study("baseline")
    if base:
        lines += ["3D, TWO CAMERAS (Cam0 + Cam4), OpenKick without calibration, angles from the 3D points, against the laboratory markers.",
                  f"{len(base)} recordings, {len({r['subject'] for r in base})} participants.", ""]
        lines += table_3d(base, ("knee", "hip", "ankle", "elbow", "shoulder"))
    lines += section_opensim() + section_2d() + section_fps() + section_pairs() + section_holdout()
    text = "\n".join(lines) + "\n"
    (OUT / "study_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
