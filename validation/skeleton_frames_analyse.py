"""Where, frame by frame, the three skeletons differ from the laboratory, and what kind of error it is.

python frames_analyse.py <folder of .npz from frames_collect.py> <folder for the figures>
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from openkick.multiview.geometry import IDX  # noqa: E402

F, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
MODES = ("detected", "model", "measured")
LABEL = {"detected": "1 detección", "model": "2 modelo", "measured": "3 medidas", "opencap": "OpenCap 2 cám."}
COLOR = {"detected": "#1f77b4", "model": "#d62728", "measured": "#ff7f0e", "opencap": "#2ca02c", "lab": "black"}
JOINTS = ("knee", "hip", "ankle")
JNAME = {"knee": "rodilla", "hip": "cadera", "ankle": "tobillo"}
SIDES = ("r", "l")
BIG = 10.0                     # degrees: a frame this far off (after its trial's offset) is "a big error"

trials = []
for p in sorted(F.glob("*.npz")):
    kind, subj, tr = p.stem.split("_")
    trials.append((kind, subj, tr, dict(np.load(p))))
print(len(trials), "trials:", ", ".join(f"{k} {s}" for k, s, _, _ in trials))


def runs(mask, t):
    """Durations (s) of the stretches where mask is on."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append(t[min(i, len(t) - 1)] - t[start] + (t[1] - t[0]))
            start = None
    return out


def decompose(est, ref, t, max_lag=0.15):
    """The error, and what is left of it once a fixed offset, an offset plus a gain, or a delay is
    taken out."""
    ok = np.isfinite(est) & np.isfinite(ref)
    if ok.sum() < 30:
        return None
    e = est[ok] - ref[ok]
    bias = float(e.mean())
    gain, icpt = np.polyfit(ref[ok], est[ok], 1)
    best = (float(np.mean(np.abs(e - bias))), 0.0)
    for L in np.arange(-max_lag, max_lag + 1e-9, 1 / 120):
        r2 = np.interp(t - L, t[ok], ref[ok], left=np.nan, right=np.nan)
        ok2 = ok & np.isfinite(r2)
        if ok2.sum() > 0.8 * ok.sum():
            e2 = est[ok2] - r2[ok2]
            best = min(best, (float(np.mean(np.abs(e2 - e2.mean()))), float(L)))
    full = np.where(ok, est - ref, np.nan)
    big = np.abs(full - bias) > BIG
    lo, hi = np.percentile(ref[ok], [2, 98])
    elo, ehi = np.percentile(est[ok], [2, 98])
    return {"mae": float(np.mean(np.abs(e))), "bias": bias, "mae_nobias": float(np.mean(np.abs(e - bias))),
            "gain": float(gain), "mae_linear": float(np.mean(np.abs(est[ok] - (icpt + gain * ref[ok])))),
            "lag_s": best[1], "mae_lag": best[0], "r": float(np.corrcoef(est[ok], ref[ok])[0, 1]),
            "p95": float(np.percentile(np.abs(e), 95)), "big_frac": float(big[ok].mean()),
            "big_runs": runs(big & ok, t), "rom_ref": float(hi - lo), "rom_est": float(ehi - elo),
            "e": full}


def table(rows, keys, title):
    print("\n" + title)
    print("  " + " ".join(f"{k:>11s}" for k in keys))
    for name, vals in rows:
        print(f"  {name:28s}" + " ".join(f"{v:11.2f}" if isinstance(v, float) else f"{v:>11}" for v in vals))


# ── 1. geometric angles (what OpenKick reports) against the laboratory markers ─────────────
R = defaultdict(list)          # (kind, mode, joint) -> per trial-side results
for kind, subj, tr, z in trials:
    for mode in MODES:
        for j in JOINTS:
            for sd in SIDES:
                d = decompose(z[f"{mode}_{j}_{sd}"], z[f"ref_{j}_{sd}"], z["t"])
                if d:
                    d.update(subj=subj, side=sd, ref=z[f"ref_{j}_{sd}"], t=z["t"], z=z)
                    R[(kind, mode, j)].append(d)

for kind in ("squats", "walking"):
    rows = []
    for j in JOINTS:
        for mode in MODES:
            L = R[(kind, mode, j)]
            if not L:
                continue
            biases = np.array([d["bias"] for d in L])
            # a fixed correction learned on the other people (leave one person out)
            loo = []
            for d in L:
                others = np.array([o["bias"] for o in L if o["subj"] != d["subj"]])
                loo.append(float(np.nanmean(np.abs(d["e"] - others.mean()))))
            rows.append((f"{JNAME[j]:8s} {LABEL[mode]}", [
                np.mean([d["mae"] for d in L]), float(np.mean(biases)), float(np.std(biases)),
                np.mean([d["mae_nobias"] for d in L]), float(np.mean(loo)),
                np.mean([d["gain"] for d in L]), np.mean([d["mae_linear"] for d in L]),
                np.mean([d["lag_s"] for d in L]) * 1000, np.mean([d["mae_lag"] for d in L]),
                np.mean([d["r"] for d in L]), 100 * np.mean([d["big_frac"] for d in L]),
                float(np.median(sum((d["big_runs"] for d in L), []) or [0.0])) * 1000]))
    table(rows, ["error", "desfase", "desf.DE", "sin_desf", "corr.fija", "ganancia", "sin_lineal", "retraso_ms",
                 "sin_retr", "r", "%>10°", "racha_ms"], f"[{kind}] ÁNGULOS GEOMÉTRICOS vs marcadores del laboratorio (grados)")

# per person, the offset of each skeleton (is it the same for everybody?)
for kind in ("squats", "walking"):
    print(f"\n[{kind}] desfase por persona (media de las dos piernas), grados")
    for j in JOINTS:
        for mode in MODES:
            by = defaultdict(list)
            for d in R[(kind, mode, j)]:
                by[d["subj"]].append(d["bias"])
            print(f"  {JNAME[j]:8s} {LABEL[mode]:13s} " + " ".join(f"{np.mean(v):+6.1f}" for _, v in sorted(by.items(), key=lambda kv: int(kv[0][7:]))))

# ── 2. where in the movement ─────────────────────────────────────────────────
EDGES = np.array([0, 20, 40, 60, 80, 100, 140])
print("\n[squats] error con signo según cuánto se dobla la rodilla (laboratorio), grados; media por persona")
sq_bins = {}
for j in JOINTS:
    for mode in MODES:
        per = []
        for d in R[("squats", mode, j)]:
            knee = d["z"][f"ref_knee_{d['side']}"]
            per.append([np.nanmean(d["e"][(knee >= a) & (knee < b)]) if ((knee >= a) & (knee < b)).sum() > 5 else np.nan
                        for a, b in zip(EDGES[:-1], EDGES[1:])])
        m = np.nanmean(np.array(per), axis=0)
        sq_bins[(mode, j)] = m
        print(f"  {JNAME[j]:8s} {LABEL[mode]:13s} " + " ".join(f"{a:3d}-{b:<3d}:{v:+6.1f}" for a, b, v in zip(EDGES[:-1], EDGES[1:], m)))

print("\n[squats] error con signo al bajar / abajo-arriba quieto / al subir, grados")
for j in JOINTS:
    for mode in MODES:
        acc = {"bajando": [], "quieto": [], "subiendo": []}
        for d in R[("squats", mode, j)]:
            knee = d["z"][f"ref_knee_{d['side']}"]
            v = np.gradient(knee, d["t"])
            for name, mk in (("bajando", v > 20), ("quieto", np.abs(v) <= 20), ("subiendo", v < -20)):
                if (mk & np.isfinite(d["e"])).sum() > 5:
                    acc[name].append(np.nanmean(d["e"][mk]))
        print(f"  {JNAME[j]:8s} {LABEL[mode]:13s} " + " ".join(f"{k}: {np.mean(v):+6.1f}" for k, v in acc.items()))


def phases(z, side):
    """Frames with the foot on the ground (heel or toe moving slower than 0.5 m/s across the floor,
    laboratory markers) and frames with it in the air. The clips hold about one stride, too little
    to cut whole cycles."""
    S = "r" if side == "r" else "L"
    slow, known = np.zeros(len(z["t"]), bool), np.ones(len(z["t"]), bool)
    for mk in (f"lab_{S}_calc", f"lab_{S}_toe"):
        P = z[mk].copy()
        P[:, 1] = 0
        v = np.linalg.norm(np.gradient(P, z["t"], axis=0), axis=1)
        slow |= v < 0.5
        known &= np.isfinite(v)
    return slow & known, ~slow & known


WEDGES = np.array([0, 10, 20, 30, 45, 75])


def walking_split(L, knee_key):
    """Signed error in stance, in swing, and by knee flexion; per trial-side then averaged."""
    st, sw, bins = [], [], []
    for d in L:
        on, off = phases(d["z"], d["side"])
        e = d["e"]
        if (on & np.isfinite(e)).sum() > 5:
            st.append(np.nanmean(e[on]))
        if (off & np.isfinite(e)).sum() > 5:
            sw.append(np.nanmean(e[off]))
        knee = d["z"][knee_key.format(side=d["side"])]
        bins.append([np.nanmean(e[(knee >= a) & (knee < b)]) if ((knee >= a) & (knee < b) & np.isfinite(e)).sum() > 5
                     else np.nan for a, b in zip(WEDGES[:-1], WEDGES[1:])])
    return np.mean(st), np.mean(sw), np.nanmean(np.array(bins), axis=0)


print("\n[walking] error con signo con el pie en el suelo (apoyo) y en el aire (balanceo), y según la flexión de rodilla (lab.), grados")
for j in JOINTS:
    for mode in MODES:
        a_, b_, m = walking_split(R[("walking", mode, j)], "ref_knee_{side}")
        print(f"  {JNAME[j]:8s} {LABEL[mode]:13s} apoyo {a_:+6.1f}  balanceo {b_:+6.1f}   "
              + " ".join(f"{x}-{y}:{v:+6.1f}" for x, y, v in zip(WEDGES[:-1], WEDGES[1:], m)))


# ── 3. OpenSim angles against the laboratory's OpenSim (the same definitions) ──────────
COORD = {"knee": "knee_angle", "hip": "hip_flexion", "ankle": "ankle_angle"}
IKR = defaultdict(list)
for kind, subj, tr, z in trials:
    t = z["t"]
    lo, hi = (t - z["lag"])[[0, -1]]
    for j, c in COORD.items():
        for sd in SIDES:
            ref = z[f"labik_{c}_{sd}"]
            for mode in ("model", "measured"):
                est = np.interp(t, z[f"{mode}_ik_t"], z[f"{mode}_ik_{c}_{sd}"], left=np.nan, right=np.nan)
                d = decompose(est, ref, t)
                if d:
                    d.update(subj=subj)
                    IKR[(kind, mode, j)].append(d)
            if "opencap_t" in z:
                tl = z["labik_t"]
                w = (tl >= lo) & (tl <= hi)
                oc = np.interp(tl[w], z["opencap_t"], z[f"opencap_{c}_{sd}"], left=np.nan, right=np.nan)
                d = decompose(oc, z[f"labikraw_{c}_{sd}"][w], tl[w])
                if d:
                    d.update(subj=subj)
                    IKR[(kind, "opencap", j)].append(d)
for kind, subj, tr, z in trials:
    t = z["t"]
    lo, hi = (t - z["lag"])[[0, -1]]
    ref = z["labik_pelvis_tilt"]
    for mode in ("model", "measured"):
        est = np.interp(t, z[f"{mode}_ik_t"], z[f"{mode}_ik_pelvis_tilt"], left=np.nan, right=np.nan)
        d = decompose(est, ref, t)
        if d:
            d.update(subj=subj)
            IKR[(kind, mode, "pelvis")].append(d)
    if "opencap_t" in z:
        tl = z["labik_t"]
        w = (tl >= lo) & (tl <= hi)
        d = decompose(np.interp(tl[w], z["opencap_t"], z["opencap_pelvis_tilt"]), z["labikraw_pelvis_tilt"][w], tl[w])
        if d:
            d.update(subj=subj)
            IKR[(kind, "opencap", "pelvis")].append(d)
JNAME["pelvis"] = "pelvis(inclin.)"
for kind in ("squats", "walking"):
    rows = []
    for j in JOINTS + ("pelvis",):
        for mode in ("model", "measured", "opencap"):
            L = IKR[(kind, mode, j)]
            if L:
                b = np.array([d["bias"] for d in L])
                rows.append((f"{JNAME[j]:8s} {LABEL[mode]}", [np.mean([d["mae"] for d in L]), float(b.mean()), float(b.std()),
                             np.mean([d["mae_nobias"] for d in L]), np.mean([d["gain"] for d in L]),
                             np.mean([d["lag_s"] for d in L]) * 1000, np.mean([d["r"] for d in L])]))
    table(rows, ["error", "desfase", "desf.DE", "sin_desf", "ganancia", "retraso_ms", "r"],
          f"[{kind}] ÁNGULOS DE OPENSIM vs OpenSim del laboratorio (grados)")

# ── 3b. what OpenKick shows (geometric angles) against the laboratory's OpenSim ─────────
# the knee is the same angle in both (flexion, 0 = straight); the ankle only up to a constant
# (its geometric angle is measured between the shank and the foot line), so for the ankle only what
# is left without the offset counts. The laboratory's own marker geometry is scored the same way.
GK = defaultdict(list)
for kind, subj, tr, z in trials:
    for j in ("knee", "ankle"):
        for sd in SIDES:
            ref = z[f"labik_{COORD[j]}_{sd}"]
            for mode in MODES + ("lab",):
                est = z[f"ref_{j}_{sd}"] if mode == "lab" else z[f"{mode}_{j}_{sd}"]
                if j == "ankle":
                    est = -est                       # the angle shank-foot closes as the foot lifts (dorsiflexion)
                d = decompose(est, ref, z["t"])
                if d:
                    d.update(subj=subj, side=sd, z=z)
                    GK[(kind, mode, j)].append(d)
LABEL["lab"] = "marcadores lab."
for kind in ("squats", "walking"):
    rows = []
    for j in ("knee", "ankle"):
        for mode in ("lab",) + MODES:
            L = GK[(kind, mode, j)]
            if L:
                b = np.array([d["bias"] for d in L])
                rows.append((f"{JNAME[j]:8s} {LABEL[mode]}", [np.mean([d["mae"] for d in L]), float(b.mean()), float(b.std()),
                             np.mean([d["mae_nobias"] for d in L]), np.mean([d["gain"] for d in L]),
                             np.mean([d["mae_linear"] for d in L]), np.mean([d["r"] for d in L])]))
    table(rows, ["error", "desfase", "desf.DE", "sin_desf", "ganancia", "sin_lineal", "r"],
          f"[{kind}] LO QUE MUESTRA OPENKICK (geométrico) vs OpenSim del laboratorio (grados; tobillo: vale 'sin_desf')")
    if kind == "squats":
        print("  rodilla, error con signo según la flexión (OpenSim lab.):")
        for mode in ("lab",) + MODES:
            per = []
            for d in GK[(kind, mode, "knee")]:
                knee = d["z"][f"labik_knee_angle_{d['side']}"]
                per.append([np.nanmean(d["e"][(knee >= a) & (knee < b)]) if ((knee >= a) & (knee < b)).sum() > 5 else np.nan
                            for a, b in zip(EDGES[:-1], EDGES[1:])])
            if per:
                m = np.nanmean(np.array(per), axis=0)
                print(f"    {LABEL[mode]:15s} " + " ".join(f"{a:3d}-{b:<3d}:{v:+6.1f}" for a, b, v in zip(EDGES[:-1], EDGES[1:], m)))

# ── 4. where each point is, against the laboratory ───────────────────────────────
PAIRS = [("right_hip", "R_HJC"), ("left_hip", "L_HJC"), ("right_knee", "r_knee"), ("left_knee", "L_knee"),
         ("right_ankle", "r_ankle"), ("left_ankle", "L_ankle"), ("right_shoulder", "R_Shoulder"),
         ("left_shoulder", "L_Shoulder"), ("right_heel", "r_calc"), ("left_heel", "L_calc"),
         ("right_toe", "r_toe"), ("left_toe", "L_toe")]
FIT = PAIRS[:8]


def similarity(P, Q):
    """Scale, rotation and translation that best put P onto Q (Umeyama)."""
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    A, B = P - pc, Q - qc
    U, S, Vt = np.linalg.svd(A.T @ B / len(P))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(axis=1).mean()
    return s, Rm, qc - s * Rm @ pc


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


POS = defaultdict(list)        # (kind, mode, joint) -> per trial-side (mean ap, vertical-along-bone, lateral), sd
for kind, subj, tr, z in trials:
    lab = {k: z[f"lab_{k}"] for _, k in PAIRS}
    pel_ml = unit(lab["R_HJC"] - lab["L_HJC"])
    foot = (lab["r_toe"] - lab["r_calc"]) + (lab["L_toe"] - lab["L_calc"])
    fwd = np.nanmean(foot, axis=0)
    fwd[1] = 0
    fwd = fwd / np.linalg.norm(fwd)
    for mode in MODES:
        X = z[f"{mode}_X"]
        P = np.concatenate([X[:, IDX[a]] for a, _ in FIT])
        Q = np.concatenate([lab[b] for _, b in FIT])
        ok = np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1)
        s, Rm, tt = similarity(P[ok], Q[ok])
        for a, b in PAIRS:
            E = (s * X[:, IDX[a]] @ Rm.T + tt) - lab[b]
            side = "r" if a.startswith("right") else "l"
            S = "R" if side == "r" else "L"
            sl = "r" if side == "r" else "L"
            part = a.split("_", 1)[1]
            if part == "hip":
                along = unit(lab[f"{S}_Shoulder"] - lab[f"{S}_HJC"])
            elif part == "knee":
                along = unit(lab[f"{S}_HJC"] - lab[f"{sl}_knee"])
            elif part == "ankle":
                along = unit(lab[f"{sl}_knee"] - lab[f"{sl}_ankle"])
            elif part == "shoulder":
                along = unit(lab[f"{S}_Shoulder"] - lab[f"{S}_HJC"])
            else:
                along = unit(np.cross(pel_ml, np.broadcast_to(fwd, pel_ml.shape)))       # the vertical, for the foot
            ml = unit(pel_ml - (pel_ml * along).sum(axis=1, keepdims=True) * along)
            ap = unit(np.cross(along, ml))
            sign = np.sign(np.nanmean((ap * fwd).sum(axis=1)))
            comp = np.stack([sign * (E * ap).sum(axis=1), (E * along).sum(axis=1),
                             (1 if side == "r" else -1) * (E * ml).sum(axis=1)], axis=1) * 100
            POS[(kind, mode, part)].append((np.nanmean(comp, axis=0), np.nanstd(comp, axis=0), s))
for kind in ("squats", "walking"):
    print(f"\n[{kind}] DÓNDE ESTÁ CADA PUNTO respecto al laboratorio, cm  (media fija ± lo que varía de fotograma a fotograma)")
    print("  + delante / + hacia arriba a lo largo del hueso / + hacia fuera;  "
          "ojo: la rodilla y el tobillo del laboratorio son marcadores por fuera de la articulación")
    for part in ("shoulder", "hip", "knee", "ankle", "heel", "toe"):
        for mode in MODES:
            L = POS[(kind, mode, part)]
            if L:
                m = np.mean([x[0] for x in L], axis=0)
                v = np.mean([x[1] for x in L], axis=0)
                print(f"  {part:8s} {LABEL[mode]:13s} delante {m[0]:+5.1f} ±{v[0]:3.1f}   arriba {m[1]:+5.1f} ±{v[1]:3.1f}   fuera {m[2]:+5.1f} ±{v[2]:3.1f}")

# bone lengths, frame by frame
print("\n[todas] largo de los huesos: laboratorio vs cada opción (cm, media por persona) y cuánto varía (DE, cm)")
for kind in ("squats", "walking"):
    for bone, (a, b, la, lb) in (("muslo", ("right_hip", "right_knee", "R_HJC", "r_knee")),
                                 ("pierna", ("right_knee", "right_ankle", "r_knee", "r_ankle"))):
        labL = [np.linalg.norm(z[f"lab_{la}"] - z[f"lab_{lb}"], axis=1) * 100 for k, _, _, z in trials if k == kind]
        line = f"  {kind:8s} {bone:7s} lab {np.mean([np.nanmean(x) for x in labL]):5.1f} (±{np.mean([np.nanstd(x) for x in labL]):3.1f})"
        for mode in MODES:
            L = [np.linalg.norm(z[f"{mode}_X"][:, IDX[a]] - z[f"{mode}_X"][:, IDX[b]], axis=1) * 100
                 for k, _, _, z in trials if k == kind]
            diff = [np.nanmean(x) - np.nanmean(y) for x, y in zip(L, labL)]
            line += f" | {LABEL[mode]} {np.mean([np.nanmean(x) for x in L]):5.1f} (±{np.mean([np.nanstd(x) for x in L]):3.1f}, dif {np.mean(diff):+4.1f})"
        print(line)

# ── figures ──────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9})


def example(kind, subj_pick=None):
    pick = [x for x in trials if x[0] == kind and x[1] == subj_pick] or [x for x in trials if x[0] == kind]
    if not pick:
        return
    pick = pick[0]
    _, subj, tr, z = pick
    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2]})
    t = z["t"] - z["t"][0]
    for i, j in enumerate(JOINTS):
        ax[i, 0].plot(t, z[f"ref_{j}_r"], color="black", lw=2, label="laboratorio (marcadores)")
        if j == "knee":
            ax[i, 0].plot(t, z["labik_knee_angle_r"], color="black", lw=1.5, ls="--", label="laboratorio (OpenSim)")
        for mode in MODES:
            ax[i, 0].plot(t, z[f"{mode}_{j}_r"], color=COLOR[mode], lw=1, label=LABEL[mode])
            ax[i, 1].plot(t, z[f"{mode}_{j}_r"] - z[f"ref_{j}_r"], color=COLOR[mode], lw=1)
        ax[i, 1].axhline(0, color="black", lw=0.8)
        ax[i, 0].set_ylabel(f"{JNAME[j]} der. (°)")
        ax[i, 1].set_ylabel("error (°)")
    ax[0, 0].legend(loc="best", fontsize=8)
    ax[2, 0].set_xlabel("s")
    ax[2, 1].set_xlabel("s")
    fig.suptitle(f"{'Sentadilla' if kind == 'squats' else 'Marcha'} · {subj}: ángulos (izq.) y error fotograma a fotograma "
                 "contra los marcadores del laboratorio (der.)")
    fig.tight_layout()
    fig.savefig(OUT / f"ejemplo_{kind}_{subj}.png", dpi=110)
    plt.close(fig)


example("squats", "subject5")
example("walking", "subject5")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
mid = (EDGES[:-1] + EDGES[1:]) / 2
for i, j in enumerate(JOINTS):
    for mode in MODES:
        ax[i].plot(mid, sq_bins[(mode, j)], "o-", color=COLOR[mode], label=LABEL[mode])
    ax[i].axhline(0, color="black", lw=0.8)
    ax[i].set_title(JNAME[j])
    ax[i].set_xlabel("flexión de rodilla del laboratorio (°)")
ax[0].set_ylabel("error con signo (°)")
ax[0].legend(fontsize=8)
fig.suptitle("Sentadillas: el error según lo profundo de la sentadilla")
fig.tight_layout()
fig.savefig(OUT / "sentadilla_por_profundidad.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharey="row")
for r_, kind in enumerate(("squats", "walking")):
    for i, j in enumerate(JOINTS):
        for k, mode in enumerate(MODES):
            b = [d["bias"] for d in R[(kind, mode, j)]]
            ax[r_, i].scatter(np.full(len(b), k) + np.random.default_rng(0).uniform(-0.12, 0.12, len(b)), b,
                              color=COLOR[mode], s=14)
            ax[r_, i].hlines(np.mean(b), k - 0.3, k + 0.3, color=COLOR[mode])
        ax[r_, i].axhline(0, color="black", lw=0.8)
        ax[r_, i].set_xticks(range(3), [LABEL[m] for m in MODES])
        ax[r_, i].set_title(f"{'sentadilla' if kind == 'squats' else 'marcha'} · {JNAME[j]}")
    ax[r_, 0].set_ylabel("desfase de cada pierna (°)")
fig.suptitle("¿Es el mismo desfase para todos? Cada punto es una pierna de una persona")
fig.tight_layout()
fig.savefig(OUT / "desfase_por_persona.png", dpi=110)
plt.close(fig)
print("\nfiguras en", OUT)


# ── the knee OpenKick shows against the laboratory's OpenSim: by depth ──
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
wmid = (WEDGES[:-1] + WEDGES[1:]) / 2
print("\n[walking] rodilla que muestra OpenKick vs OpenSim del laboratorio: apoyo / balanceo / por flexión (grados)")
for mode in ("lab",) + MODES:
    per = []
    for d in GK[("squats", mode, "knee")]:
        knee = d["z"][f"labik_knee_angle_{d['side']}"]
        per.append([np.nanmean(d["e"][(knee >= a) & (knee < b)]) if ((knee >= a) & (knee < b)).sum() > 5 else np.nan
                    for a, b in zip(EDGES[:-1], EDGES[1:])])
    ls = "--" if mode == "lab" else "-"
    if per:
        ax[0].plot(mid, np.nanmean(np.array(per), axis=0), "o-", color=COLOR[mode], label=LABEL[mode], ls=ls)
    a_, b_, m = walking_split(GK[("walking", mode, "knee")], "labik_knee_angle_{side}")
    ax[1].plot(wmid, m, "o-", color=COLOR[mode], label=LABEL[mode], ls=ls)
    print(f"  {LABEL[mode]:15s} apoyo {a_:+6.1f}  balanceo {b_:+6.1f}   "
          + " ".join(f"{x}-{y}:{v:+6.1f}" for x, y, v in zip(WEDGES[:-1], WEDGES[1:], m)))
for a_ in ax:
    a_.axhline(0, color="black", lw=0.8)
    a_.set_ylabel("error con signo (°)")
    a_.set_xlabel("flexión de rodilla, OpenSim del laboratorio (°)")
ax[0].set_title("Sentadilla")
ax[1].set_title("Marcha")
ax[0].legend(fontsize=8)
fig.suptitle("La rodilla que muestra OpenKick contra la del OpenSim del laboratorio (el patrón de oro)")
fig.tight_layout()
fig.savefig(OUT / "rodilla_vs_opensim_lab.png", dpi=110)
plt.close(fig)
