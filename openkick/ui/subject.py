"""The patient's record: who they are, their size and what was measured on them with a tape.
The same dialog for a 2D project and a 3D session. A patient saved before can be picked again:
the record starts from their latest one, to be brought up to date (today's weight...)."""

from __future__ import annotations

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateEdit, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QScrollArea, QTabWidget, QVBoxLayout, QWidget)

from openkick.core import patients as P
from openkick.core import subject as S
from openkick.i18n import tr
from openkick.ui import theme

# what is usually worth writing down about the equipment, by template
EQUIPMENT = {
    "cycling": ["Saddle height (cm)", "Saddle setback (cm)", "Crank length (mm)", "Handlebar reach (cm)",
                "Handlebar drop (cm)", "Frame size", "Cleats or pedals"],
    "gait": ["Footwear", "Orthotics", "Walking aid"],
}


def _spin(lo: float, hi: float, decimals: int, suffix: str) -> QDoubleSpinBox:
    """A number that may be left empty: its lowest value reads "—" and means "not measured"."""
    b = QDoubleSpinBox()
    b.setRange(lo, hi)
    b.setDecimals(decimals)
    b.setSuffix(suffix)
    b.setSpecialValueText("—")
    b.setValue(lo)
    b.setMinimumWidth(96)
    return b


def _value(b: QDoubleSpinBox) -> float | None:
    return None if b.value() <= b.minimum() else round(b.value(), b.decimals())


def _set(b: QDoubleSpinBox, v):
    b.setValue(b.minimum() if v in (None, "") else float(v))


class SubjectDialog(QDialog):
    def __init__(self, subject: S.Subject | None, template: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Patient record"))
        self.resize(640, 660)
        self.template = template
        s = subject or S.Subject()
        self.patient_id = s.id
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        title = QLabel(tr("Patient record"))
        title.setFont(theme.display(22))
        lay.addWidget(title)
        hint = QLabel(tr("Everything is optional. What you fill in goes into the reports; the lengths measured "
                         "with a tape also give the 3D its real size."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(hint)
        pick = QHBoxLayout()
        pick.addWidget(QLabel(tr("Saved patient")))
        self.saved = QComboBox()
        self.saved.setToolTip(tr("Start from a patient saved in another analysis: their latest record fills the form"))
        self.saved.addItem(tr("New patient"), "")
        for row in P.every():
            self.saved.addItem(patient_label(row["subject"]), row["id"])
        self.saved.setCurrentIndex(max(0, self.saved.findData(s.id)) if s.id else 0)
        self.saved.activated.connect(self._pick)
        pick.addWidget(self.saved, 1)
        lay.addLayout(pick)
        tabs = QTabWidget()
        lay.addWidget(tabs, 1)

        # ── the person ──
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignRight)
        self.name = QLineEdit()
        self.name.setPlaceholderText(tr("Name or code"))
        form.addRow(tr("Name"), self.name)
        self.sex = QComboBox()
        for key, label in (("", "—"), ("female", "Female"), ("male", "Male"), ("other", "Other")):
            self.sex.addItem(tr(label), key)
        form.addRow(tr("Sex"), self.sex)
        birth_row = QHBoxLayout()
        self.has_birth = QCheckBox(tr("Known"))
        self.birth = QDateEdit()
        self.birth.setCalendarPopup(True)
        self.birth.setDisplayFormat("dd/MM/yyyy")
        self.has_birth.toggled.connect(self.birth.setEnabled)
        self.age = QLabel()
        self.age.setStyleSheet(f"color: {theme.MUTED};")
        for w in (self.has_birth, self.birth, self.age):
            birth_row.addWidget(w)
        birth_row.addStretch(1)
        form.addRow(tr("Date of birth"), birth_row)
        self.height = _spin(0, 230, 1, " cm")
        form.addRow(tr("Height"), self.height)
        self.mass = _spin(0, 250, 1, " kg")
        form.addRow(tr("Weight"), self.mass)
        self.bmi = QLabel()
        form.addRow(tr("Body mass index"), self.bmi)
        self.dominant = QComboBox()
        for key in ("", "right", "left"):
            self.dominant.addItem(tr(key).capitalize() if key else "—", key)       # the whole word, not "R."
        form.addRow(tr("Dominant side"), self.dominant)
        self.activity = QLineEdit()
        self.activity.setPlaceholderText(tr("Sport or activity, and level"))
        form.addRow(tr("Activity"), self.activity)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText(tr("Why they came, diagnosis"))
        form.addRow(tr("Reason for consultation"), self.reason)
        pain_row = QHBoxLayout()
        self.pain = _spin(-1, 10, 0, "")
        self.pain.setMinimum(-1)
        self.pain_where = QLineEdit()
        self.pain_where.setPlaceholderText(tr("Where"))
        pain_row.addWidget(self.pain)
        pain_row.addWidget(self.pain_where, 1)
        form.addRow(tr("Pain (0–10)"), pain_row)
        self.notes = QPlainTextEdit()
        self.notes.setFixedHeight(70)
        form.addRow(tr("Notes"), self.notes)
        tabs.addTab(page, tr("Person"))

        # ── lengths ──
        page = QWidget()
        grid = QGridLayout(page)
        grid.setHorizontalSpacing(10)
        grid.addWidget(QLabel(tr("Left")), 0, 1, Qt.AlignCenter)
        grid.addWidget(QLabel(tr("Right")), 0, 2, Qt.AlignCenter)
        self.lengths = {}
        r = 1
        for key, label, where in S.LENGTHS:
            name = QLabel(tr(label))
            name.setToolTip(tr(where))
            grid.addWidget(name, r, 0)
            for c, side in ((1, "left"), (2, "right")):
                b = _spin(0, 150, 1, " cm")
                b.setToolTip(tr(where))
                b.valueChanged.connect(self._update)
                self.lengths[f"{key}_{side}"] = b
                grid.addWidget(b, r, c)
            how = QLabel(tr(where))
            how.setStyleSheet(f"color: {theme.FAINT}; font-size: 11px;")
            grid.addWidget(how, r, 3)
            r += 1
        for key, label, where in S.WIDTHS:
            grid.addWidget(QLabel(tr(label)), r, 0)
            b = _spin(0, 150, 1, " cm")
            self.lengths[key] = b
            grid.addWidget(b, r, 1)
            how = QLabel(tr(where))
            how.setStyleSheet(f"color: {theme.FAINT}; font-size: 11px;")
            grid.addWidget(how, r, 3)
            r += 1
        self.difference = QLabel()
        self.difference.setWordWrap(True)
        grid.addWidget(self.difference, r, 0, 1, 4)
        grid.setRowStretch(r + 1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        tabs.addTab(scroll, tr("Measured lengths"))

        # ── equipment ──
        page = QWidget()
        self.equipment_form = QFormLayout(page)
        self.equipment, self.extra = {}, []
        tabs.addTab(page, tr("Equipment"))

        for w in (self.height, self.mass):
            w.valueChanged.connect(self._update)
        self.birth.dateChanged.connect(self._update)
        self.has_birth.toggled.connect(self._update)
        self.keep = QCheckBox(tr("Keep it in my patients, to pick it in other analyses"))
        self.keep.setChecked(True)
        lay.addWidget(self.keep)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Save"))
        buttons.button(QDialogButtonBox.Ok).setProperty("primary", True)
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.set_subject(s)
        if not s.id and s.name.strip():
            # a record from before the list (or from another computer): the one patient of that name
            same = [row for row in P.every() if row["subject"].name.strip().lower() == s.name.strip().lower()]
            if len(same) == 1:
                self.patient_id = same[0]["id"]
                self.saved.setCurrentIndex(max(0, self.saved.findData(self.patient_id)))

    def set_subject(self, s: S.Subject):
        """Fill the form with a record."""
        self.patient_id = s.id
        self.name.setText(s.name)
        self.sex.setCurrentIndex(max(0, self.sex.findData(s.sex)))
        d = QDate.fromString(s.birth, "yyyy-MM-dd")
        self.birth.setDate(d if d.isValid() else QDate(1995, 1, 1))
        self.has_birth.setChecked(d.isValid())
        self.birth.setEnabled(d.isValid())
        _set(self.height, s.height_m * 100 if s.height_m else None)
        _set(self.mass, s.mass_kg)
        self.dominant.setCurrentIndex(max(0, self.dominant.findData(s.dominant)))
        self.activity.setText(s.activity)
        self.reason.setText(s.reason)
        _set(self.pain, s.pain)
        self.pain_where.setText(s.pain_where)
        self.notes.setPlainText(s.notes)
        for key, b in self.lengths.items():
            _set(b, s.lengths.get(key))
        self._build_equipment(s.equipment)
        self._update()

    def _build_equipment(self, values: dict):
        form = self.equipment_form
        while form.rowCount():
            form.removeRow(0)
        self.equipment = {}
        names = [tr(n) for n in EQUIPMENT.get(self.template or "", [])]
        for k in names + [k for k in values if k not in names]:
            e = QLineEdit(str(values.get(k, "")))
            self.equipment[k] = e
            form.addRow(k, e)
        self.extra = [(QLineEdit(), QLineEdit()) for _ in range(3)]
        for a, b in self.extra:
            a.setPlaceholderText(tr("Other item"))
            row = QHBoxLayout()
            row.addWidget(a)
            row.addWidget(b, 1)
            form.addRow(row)

    def _pick(self, _=None):
        pid = self.saved.currentData()
        found = P.get(pid) if pid else None
        self.set_subject(found["subject"] if found else S.Subject())

    def _update(self, *_):
        s = self.subject()
        self.age.setText(tr("{n} years", n=s.age) if s.age is not None else "")
        if s.bmi is None:
            self.bmi.setText("—")
            self.bmi.setStyleSheet(f"color: {theme.MUTED};")
        else:
            self.bmi.setText(f"{s.bmi:.1f} kg/m² · {tr(s.bmi_class)}")
            self.bmi.setStyleSheet("font-weight: 600;")
        d = s.leg_difference()
        if d is None:
            self.difference.setText("")
        else:
            text = tr("Leg length difference: {d} cm, {side} longer.", d=f"{abs(d):.1f}",
                      side=tr("right") if d > 0 else tr("left")) if abs(d) >= 0.05 else tr("Both legs measure the same.")
            if abs(d) >= S.LEG_DIFFERENCE_NOTE:
                text += " " + tr("A tape measure is about ±1 cm itself: confirm it with another method before acting on it.")
            self.difference.setText(text)

    def subject(self) -> S.Subject:
        lengths = {k: v for k, v in ((k, _value(b)) for k, b in self.lengths.items()) if v}
        equipment = {k: e.text().strip() for k, e in self.equipment.items() if e.text().strip()}
        for a, b in self.extra:
            if a.text().strip() and b.text().strip():
                equipment[a.text().strip()] = b.text().strip()
        h = _value(self.height)
        pain = None if self.pain.value() < 0 else self.pain.value()
        return S.Subject(id=self.patient_id or "", name=self.name.text().strip(), sex=self.sex.currentData(),
                         birth=self.birth.date().toString("yyyy-MM-dd") if self.has_birth.isChecked() else "",
                         height_m=h / 100 if h else None, mass_kg=_value(self.mass), dominant=self.dominant.currentData(),
                         activity=self.activity.text().strip(), reason=self.reason.text().strip(), pain=pain,
                         pain_where=self.pain_where.text().strip(), notes=self.notes.toPlainText().strip(),
                         lengths=lengths, equipment=equipment)


def patient_label(s: S.Subject) -> str:
    """A patient in a list: name, age and height."""
    bits = [s.name or tr("No name")]
    if s.age is not None:
        bits.append(tr("{n} years", n=s.age))
    if s.height_m:
        bits.append(f"{s.height_m * 100:.0f} cm")
    return " · ".join(bits)


def edit_subject(parent, subject: S.Subject | None, template: str | None = None) -> S.Subject | None:
    """The record as the user left it (None if cancelled). Unless they unticked it, it is also
    kept in their patients, which gives a new patient its id."""
    dlg = SubjectDialog(subject, template, parent)
    if dlg.exec() != QDialog.Accepted:
        return None
    s = dlg.subject()
    return P.remember(s) if dlg.keep.isChecked() else s
