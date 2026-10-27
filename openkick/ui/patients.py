"""The patients kept on this computer: their latest record and the analyses they are in.

From here a patient's record is brought up to date, one of their analyses opened, or all of them
sent to the comparison window. A new analysis picks the patient in its own patient record."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from openkick.core import patients as P
from openkick.core.subject import describe
from openkick.i18n import tr
from openkick.ui import theme


def _num(v, digits=1) -> str:
    return f"{v:.{digits}f}" if v is not None else "—"


class PatientsWindow(QWidget):
    """`open_path(path)` opens an analysis; `compare(paths)` sends several to the comparison."""

    def __init__(self, open_path=None, compare=None, parent=None):
        super().__init__(parent, Qt.Window)
        self.open_path, self.compare = open_path, compare
        self.rows: list[dict] = []
        self.setWindowTitle(tr("Patients"))
        self.setObjectName("Page")
        self.resize(1080, 680)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel(tr("Patients"))
        title.setFont(theme.display(22))
        head.addWidget(title)
        head.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Search…"))
        self.search.setFixedWidth(260)
        self.search.textChanged.connect(self._fill)
        head.addWidget(self.search)
        close = QToolButton()
        close.setText("✕")
        close.setToolTip(tr("Close"))
        close.clicked.connect(self.close)
        head.addWidget(close)
        lay.addLayout(head)
        hint = QLabel(tr("Every patient record saved in an analysis is kept here, on this computer only. Each "
                         "analysis keeps the record as it was that day; this list holds the latest, to pick it in "
                         "the next analysis (Patient record › Saved patient)."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(hint)

        body = QHBoxLayout()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([tr("Name"), tr("Age"), tr("Sex"), tr("Height"), tr("Weight"),
                                              tr("BMI"), tr("Analyses"), tr("Updated")])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setMinimumSectionSize(48)
        self.table.itemSelectionChanged.connect(self._show)
        self.table.cellDoubleClicked.connect(lambda *_: self.edit())
        body.addWidget(self.table, 3)
        side = QVBoxLayout()
        self.record = QLabel()
        self.record.setWordWrap(True)
        self.record.setTextFormat(Qt.RichText)
        self.record.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.record.setMinimumWidth(340)
        box = QScrollArea()
        box.setWidgetResizable(True)
        box.setFrameShape(QFrame.NoFrame)
        box.setWidget(self.record)
        side.addWidget(box, 2)
        side.addWidget(QLabel(tr("Analyses")))
        self.analyses = QListWidget()
        self.analyses.itemDoubleClicked.connect(lambda *_: self.open_selected())
        side.addWidget(self.analyses, 1)
        row = QHBoxLayout()
        self.open_btn = QPushButton(tr("Open"))
        self.open_btn.setToolTip(tr("Open the selected analysis (double click does the same)"))
        self.open_btn.clicked.connect(self.open_selected)
        row.addWidget(self.open_btn)
        self.compare_btn = QPushButton(tr("Compare them"))
        self.compare_btn.setToolTip(tr("Every analysis of this patient side by side"))
        self.compare_btn.clicked.connect(self.compare_all)
        row.addWidget(self.compare_btn)
        side.addLayout(row)
        row = QHBoxLayout()
        self.edit_btn = QPushButton(tr("Edit the record…"))
        self.edit_btn.setProperty("primary", True)
        self.edit_btn.clicked.connect(self.edit)
        row.addWidget(self.edit_btn)
        self.forget_btn = QPushButton(tr("Take off the list"))
        self.forget_btn.setToolTip(tr("The patient leaves this list; their analyses are not touched"))
        self.forget_btn.clicked.connect(self.forget)
        row.addWidget(self.forget_btn)
        side.addLayout(row)
        new = QPushButton(tr("New patient…"))
        new.clicked.connect(self.new)
        side.addWidget(new)
        body.addLayout(side, 2)
        lay.addLayout(body, 1)
        self.count = QLabel()
        self.count.setStyleSheet(f"color: {theme.DIM};")
        lay.addWidget(self.count)
        self.reload()

    # ── contents ──
    def reload(self, keep: str | None = None):
        keep = keep or self.current_id()
        self.rows = P.every()
        self._fill(keep=keep)

    def shown(self) -> list[dict]:
        q = self.search.text().strip().lower()
        if not q:
            return self.rows
        return [r for r in self.rows if q in " ".join([r["subject"].name, r["subject"].activity,
                                                       r["subject"].reason, r["subject"].notes]).lower()]

    def current_id(self) -> str | None:
        r = self.table.currentRow()
        items = getattr(self, "_items", [])
        return items[r]["id"] if 0 <= r < len(items) else None

    def current(self) -> dict | None:
        pid = self.current_id()
        return next((r for r in self.rows if r["id"] == pid), None)

    def _fill(self, *_, keep: str | None = None):
        self._items = self.shown()
        self.table.setRowCount(len(self._items))
        sexes = {"female": tr("Female"), "male": tr("Male"), "other": tr("Other")}
        for i, r in enumerate(self._items):
            s = r["subject"]
            bmi = f"{s.bmi:.1f}" if s.bmi else "—"
            cells = [s.name or tr("No name"), str(s.age) if s.age is not None else "—", sexes.get(s.sex, "—"),
                     f"{_num(s.height_m * 100 if s.height_m else None, 0)} cm" if s.height_m else "—",
                     f"{_num(s.mass_kg)} kg" if s.mass_kg else "—", bmi, str(len(r["analyses"])),
                     r["updated"][:10]]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c in (1, 3, 4, 5, 6, 7):
                    it.setFont(theme.mono(12))
                self.table.setItem(i, c, it)
        self.count.setText(tr("{n} patients", n=len(self.rows)))
        pick = next((i for i, r in enumerate(self._items) if r["id"] == keep), 0)
        if self._items:
            self.table.selectRow(pick)
        else:
            self.record.setText(tr("No patients yet: fill in a patient record in an analysis and save it.")
                                if not self.rows else "")
            self.analyses.clear()
        self._show()

    def _show(self):
        r = self.current()
        for b in (self.open_btn, self.compare_btn, self.edit_btn, self.forget_btn):
            b.setEnabled(r is not None)
        if r is None:
            return
        rows = describe(r["subject"], tr)
        self.record.setText("<table cellspacing=3>" + "".join(
            f"<tr><td style='color:{theme.MUTED}; padding-right:10px'>{k}</td><td>{v}</td></tr>" for k, v in rows)
            + "</table>")
        self.analyses.clear()
        latest_first = [a for _, a in sorted(enumerate(r["analyses"]), key=lambda x: (x[1].get("date", ""), x[0]),
                                                reverse=True)]
        for a in latest_first:
            path = Path(a["path"])
            there = path.exists()
            it = QListWidgetItem(f"{a.get('kind', '')} · {path.name} · {a.get('date', '').replace('T', ' ')[:16]}"
                                 + ("" if there else "  (" + tr("not found") + ")"))
            it.setToolTip(str(path))
            it.setData(Qt.UserRole, str(path))
            if not there:
                it.setForeground(QColor(theme.DIM))
            self.analyses.addItem(it)
        if self.analyses.count():
            self.analyses.setCurrentRow(0)
        self.compare_btn.setEnabled(sum(Path(a["path"]).exists() for a in r["analyses"]) >= 2 and self.compare is not None)
        self.open_btn.setEnabled(self.analyses.count() > 0 and self.open_path is not None)

    # ── actions ──
    def edit(self):
        r = self.current()
        if r is None:
            return
        from openkick.ui.subject import edit_subject
        s = edit_subject(self, r["subject"])
        if s is not None:
            self.reload(keep=s.id or r["id"])

    def new(self):
        from openkick.ui.subject import edit_subject
        s = edit_subject(self, None)
        if s is not None:
            self.reload(keep=s.id)

    def forget(self):
        r = self.current()
        if r is None:
            return
        answer = QMessageBox.question(self, tr("Patients"), tr("Take {name} off the list? Their analyses are not "
                                                               "touched.", name=r["subject"].name or tr("No name")))
        if answer == QMessageBox.Yes:
            P.forget(r["id"])
            self.reload()

    def open_selected(self):
        it = self.analyses.currentItem()
        if it is None or self.open_path is None:
            return
        path = it.data(Qt.UserRole)
        if not Path(path).exists():
            QMessageBox.warning(self, tr("Patients"), tr("The file is not there any more:\n{path}", path=path))
            return
        self.open_path(path)

    def compare_all(self):
        r = self.current()
        if r is None or self.compare is None:
            return
        paths = [a["path"] for a in sorted(r["analyses"], key=lambda a: a.get("date", "")) if Path(a["path"]).exists()]
        if len(paths) >= 2:
            self.compare(paths)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)
