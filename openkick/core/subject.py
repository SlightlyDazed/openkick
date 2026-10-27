"""The person being analysed: who they are, their size and the lengths measured on them.

Every analysis (a 2D project or a 3D session) carries one of these. Everything is optional:
whatever is filled in is shown in the reports and used where it helps. The lengths measured
with a tape on the person give the reconstruction its true scale (a rider on a bike never
stands up straight, so their height alone gives it badly), and let people of different size be
compared.

The landmarks are those the skeleton's points sit on, so that a measured length and a
reconstructed one mean the same thing:
  thigh  greater trochanter to the knee joint line, on the outer side
  shank  knee joint line to the tip of the outer ankle bone (lateral malleolus)
  leg    anterior superior iliac spine (ASIS) to the inner ankle bone (medial malleolus):
         the usual tape measure of leg length, done on both sides to compare them
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

# (key, name, what it is measured between); centimetres, left and right where it applies
LENGTHS = [
    ("thigh", "Thigh", "greater trochanter → knee joint line (outer side)"),
    ("shank", "Shank", "knee joint line → outer ankle bone (lateral malleolus)"),
    ("leg", "Leg length", "ASIS → inner ankle bone (medial malleolus)"),
    ("foot", "Foot", "heel → tip of the longest toe"),
    ("upper_arm", "Upper arm", "acromion → outer elbow (lateral epicondyle)"),
    ("forearm", "Forearm", "outer elbow → wrist (radial styloid)"),
]
WIDTHS = [
    ("shoulders", "Shoulder width", "acromion to acromion"),
    ("hips", "Hip width", "ASIS to ASIS"),
    ("inseam", "Inseam", "crotch to the floor, barefoot"),
]
SIDES = ("left", "right")

# WHO adult classes of the body mass index (kg/m²)
BMI_CLASSES = [(18.5, "Underweight"), (25.0, "Normal weight"), (30.0, "Overweight"),
               (35.0, "Obesity class I"), (40.0, "Obesity class II"), (float("inf"), "Obesity class III")]
LEG_DIFFERENCE_NOTE = 1.0   # cm: from about here a difference is worth a closer look (a tape is ±1 cm itself)


@dataclass
class Subject:
    name: str = ""
    sex: str = ""                      # "female", "male", "other" or ""
    birth: str = ""                    # ISO date (YYYY-MM-DD)
    height_m: float | None = None
    mass_kg: float | None = None
    dominant: str = ""                 # "left", "right" or ""
    activity: str = ""                 # sport or activity, level
    reason: str = ""                   # reason for consultation, diagnosis
    pain: float | None = None          # 0..10 (numeric rating scale)
    pain_where: str = ""
    notes: str = ""
    lengths: dict = field(default_factory=dict)   # "thigh_left": cm ...; "shoulders": cm ...
    equipment: dict = field(default_factory=dict)  # free key -> value (saddle height, crank length...)
    id: str = ""                       # the same person in the list of patients (core/patients.py)

    # ── derived ────────────────────────────────────────
    @property
    def age(self) -> int | None:
        try:
            b = date.fromisoformat(self.birth)
        except ValueError:
            return None
        today = date.today()
        return today.year - b.year - ((today.month, today.day) < (b.month, b.day))

    @property
    def bmi(self) -> float | None:
        if not self.height_m or not self.mass_kg or self.height_m <= 0:
            return None
        return self.mass_kg / self.height_m ** 2

    @property
    def bmi_class(self) -> str | None:
        b = self.bmi
        if b is None:
            return None
        return next(name for top, name in BMI_CLASSES if b < top)

    def length(self, key: str, side: str | None = None) -> float | None:
        """A measured length in centimetres (one side, or the mean of both)."""
        if side:
            return self.lengths.get(f"{key}_{side}") or None
        if key in dict((k, n) for k, n, _ in WIDTHS):
            return self.lengths.get(key) or None
        both = [v for v in (self.lengths.get(f"{key}_{s}") for s in SIDES) if v]
        return sum(both) / len(both) if both else None

    def leg_difference(self) -> float | None:
        """Right leg minus left leg, cm (from the tape measures of leg length)."""
        r, l = self.lengths.get("leg_right"), self.lengths.get("leg_left")
        return (r - l) if r and l else None

    def scale_lengths(self) -> dict:
        """Measured thigh and shank of each side, in metres, for scaling a reconstruction."""
        out = {}
        for key in ("thigh", "shank"):
            for s in SIDES:
                v = self.lengths.get(f"{key}_{s}")
                if v:
                    out[f"{s}_{key}"] = v / 100
        return out

    @property
    def empty(self) -> bool:
        return not any([self.name, self.sex, self.birth, self.height_m, self.mass_kg, self.dominant, self.activity,
                        self.reason, self.pain is not None, self.pain_where, self.notes, self.lengths, self.equipment])

    # ── files ──────────────────────────────────────────
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in ("", None, {}, [])}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Subject":
        if not d:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


def describe(s: Subject, tr=lambda t, **k: t.format(**k) if k else t) -> list[tuple[str, str]]:
    """(label, value) rows for a report, only what is filled in."""
    rows = []
    if s.name:
        rows.append((tr("Name"), s.name))
    if s.age is not None:
        rows.append((tr("Age"), tr("{n} years", n=s.age)))
    if s.sex:
        rows.append((tr("Sex"), tr({"female": "Female", "male": "Male", "other": "Other"}.get(s.sex, s.sex))))
    if s.height_m:
        rows.append((tr("Height"), f"{s.height_m * 100:.1f} cm"))
    if s.mass_kg:
        rows.append((tr("Weight"), f"{s.mass_kg:.1f} kg"))
    if s.bmi is not None:
        rows.append((tr("Body mass index"), f"{s.bmi:.1f} kg/m² · {tr(s.bmi_class)}"))
    if s.dominant:
        rows.append((tr("Dominant side"), tr("left" if s.dominant == "left" else "right").capitalize()))
    if s.activity:
        rows.append((tr("Activity"), s.activity))
    if s.reason:
        rows.append((tr("Reason for consultation"), s.reason))
    if s.pain is not None:
        rows.append((tr("Pain (0–10)"), f"{s.pain:g}" + (f" · {s.pain_where}" if s.pain_where else "")))
    for key, name, _ in LENGTHS:
        l, r = s.lengths.get(f"{key}_left"), s.lengths.get(f"{key}_right")
        if l or r:
            rows.append((tr(name), f"{tr('L')} {l:.1f} · {tr('R')} {r:.1f} cm" if l and r else
                         f"{(l or r):.1f} cm ({tr('left') if l else tr('right')})"))
    for key, name, _ in WIDTHS:
        if s.lengths.get(key):
            rows.append((tr(name), f"{s.lengths[key]:.1f} cm"))
    d = s.leg_difference()
    if d is not None:
        longer = tr("right") if d > 0 else tr("left")
        rows.append((tr("Leg length difference"), tr("{d} cm, {side} longer", d=f"{abs(d):.1f}", side=longer)
                     if abs(d) >= 0.05 else tr("none")))
    for k, v in s.equipment.items():
        rows.append((k, str(v)))
    if s.notes:
        rows.append((tr("Notes"), s.notes))
    return rows
