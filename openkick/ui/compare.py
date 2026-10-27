"""The comparison window: several analyses (2D projects or 3D sessions) side by side.

Add the files (or drop them on the window), give each a name ("Saddle 93", "Before"...), and
read the tables leg by leg and the mean cycle of each analysis drawn over the others. The first
analysis is the reference the others are compared with."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from openkick.core import compare as K
from openkick.i18n import tr
from openkick.ui import icons, theme

COLORS = ["#38BDF8", "#F59E0B", "#A78BFA", "#34D399", "#F472B6", "#F87171", "#FACC15", "#60A5FA"]
CURVES = [("knee", "Knee"), ("hip", "Hip"), ("ankle", "Ankle")]
FILES = "OpenKick (*.okick *.ok3d)"


class CycleChart(QWidget):
    """The mean cycle of each analysis (line) and its spread (band), from 0 to 100 % of the cycle."""

    # (background, title, axis labels, grid, legend) on screen and on paper
    SCREEN = (theme.PANEL, theme.TEXT, theme.DIM, theme.LINE_SOFT, theme.TEXT_2)
    PAPER = ("#FFFFFF", "#1B1F24", "#5B6474", "#E4E7EC", "#3D4550")

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.items: list[tuple[str, str, tuple]] = []     # (label, colour, (mean, sd, n))
        self.colors = self.SCREEN
        self.setMinimumHeight(240)

    def set_items(self, items):
        self.items = [i for i in items if i[2] is not None]
        self.update()

    def image(self, width: int, height: int) -> QImage:
        """The chart on white, for a report."""
        self.colors = self.PAPER
        self.resize(width, height)
        img = QImage(width, height, QImage.Format_ARGB32)
        img.fill(QColor("#FFFFFF"))
        self.render(img)
        self.colors = self.SCREEN
        return img

    def paintEvent(self, _):
        bg, ink, dim, grid, legend = self.colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(bg))
        p.setPen(QColor(ink))
        p.setFont(theme.font(12, 600))
        p.drawText(QRectF(12, 6, self.width() - 24, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        r = QRectF(48, 52, self.width() - 64, self.height() - 76)
        if not self.items or r.width() < 50:
            p.setPen(QColor(dim))
            p.drawText(r, Qt.AlignCenter, tr("No cycles to draw"))
            return
        lo = min(float(np.nanmin(m - s)) for _, _, (m, s, _) in self.items)
        hi = max(float(np.nanmax(m + s)) for _, _, (m, s, _) in self.items)
        pad = max(2.0, 0.08 * (hi - lo))
        lo, hi = lo - pad, hi + pad
        p.setFont(theme.mono(10))
        for k in range(5):
            v = lo + (hi - lo) * k / 4
            y = r.bottom() - (v - lo) / (hi - lo) * r.height()
            p.setPen(QPen(QColor(grid), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(dim))
            p.drawText(QRectF(0, y - 8, 44, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:.0f}°")
        for pct in (0, 25, 50, 75, 100):
            x = r.left() + pct / 100 * r.width()
            p.drawText(QRectF(x - 20, r.bottom() + 4, 40, 14), Qt.AlignCenter, f"{pct} %")
        xs = np.linspace(r.left(), r.right(), K.SAMPLES)

        def ys(v):
            return r.bottom() - (v - lo) / (hi - lo) * r.height()
        for label, color, (m, s, n) in self.items:
            band = QPainterPath()
            band.moveTo(xs[0], ys(m[0] + s[0]))
            for x, v in zip(xs[1:], (m + s)[1:]):
                band.lineTo(x, ys(v))
            for x, v in zip(xs[::-1], (m - s)[::-1]):
                band.lineTo(x, ys(v))
            band.closeSubpath()
            c = QColor(color)
            c.setAlpha(45)
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawPath(band)
            line = QPainterPath()
            line.moveTo(xs[0], ys(m[0]))
            for x, v in zip(xs[1:], m[1:]):
                line.lineTo(x, ys(v))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(color), 2))
            p.drawPath(line)
        # legend, under the title
        p.setFont(theme.font(11))
        fm = QFontMetricsF(p.font())
        x = 12.0
        for label, color, (m, s, n) in self.items:
            text = f"{label} (n={n})"
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(x, 31, 10, 10), 2, 2)
            p.setPen(QColor(legend))
            p.drawText(QPointF(x + 14, 40), text)
            x += 26 + fm.horizontalAdvance(text)


class AnalysisCard(QFrame):
    removed = Signal(object)
    moved = Signal(object, int)
    renamed = Signal()

    def __init__(self, item: K.Analysis, color: str, parent=None):
        super().__init__(parent)
        self.item = item
        self.setObjectName("Card")
        self.setStyleSheet(f"QFrame#Card {{ background: {theme.CARD}; border: 1px solid {theme.BORDER}; "
                           f"border-left: 4px solid {color}; border-radius: 10px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 8, 10)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.name = QLineEdit(item.label)
        self.name.setToolTip(tr("Name of this analysis in the tables and charts"))
        self.name.editingFinished.connect(self._rename)
        top.addWidget(self.name, 1)
        for text, step, tip in (("◀", -1, tr("Move left (the first one is the reference)")), ("▶", 1, tr("Move right"))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, st=step: self.moved.emit(self, st))
            top.addWidget(b)
        x = QToolButton()
        x.setText("✕")
        x.setToolTip(tr("Remove from the comparison"))
        x.clicked.connect(lambda: self.removed.emit(self))
        top.addWidget(x)
        lay.addLayout(top)
        what = tr("Cycling") if item.activity == "cycling" else tr("Gait") if item.activity == "gait" else "—"
        counts = ", ".join(f"{tr('Left') if s == 'left' else tr('Right')} {len(c)}" for s, c in item.cycles.items() if c)
        info = QLabel(f"{item.kind} · {what} · {counts or tr('no cycles')}\n{Path(item.path).name}")
        info.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px; border: 0;")
        lay.addWidget(info)
        who = getattr(item.subject, "name", "") if item.subject is not None else ""
        if who:
            w = QLabel(who)
            w.setStyleSheet(f"color: {theme.TEXT_2}; font-size: 11px; border: 0;")
            lay.addWidget(w)
        self.setFixedWidth(260)

    def _rename(self):
        text = self.name.text().strip()
        if text and text != self.item.label:
            self.item.label = text
            self.renamed.emit()


class CompareWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Compare analyses") + " · OpenKick")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        self.items: list[K.Analysis] = []
        self.path: Path | None = None
        self.notes = ""                     # the evaluator's observations, for the report
        self.evaluator = ""
        root = QWidget()
        root.setObjectName("Page")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(10)

        bar = QHBoxLayout()
        title = QLabel(tr("Compare analyses"))
        title.setFont(theme.display(22))
        bar.addWidget(title)
        bar.addStretch(1)
        add = QPushButton(icons.icon("plus", theme.ON_ACCENT, 16), " " + tr("Add analyses…"))
        add.setProperty("primary", True)
        add.setToolTip(tr("2D projects (.okick) or 3D sessions (.ok3d); you can also drop them on this window"))
        add.clicked.connect(self.browse)
        bar.addWidget(add)
        bar.addSpacing(12)
        bar.addWidget(QLabel(tr("People")))
        self.people = QComboBox()
        self.people.addItem(tr("The same person"), False)
        self.people.addItem(tr("Different people"), True)
        self.people.setToolTip(tr("With different people, lengths are shown as a share of each one's height"))
        self.people.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.people)
        bar.addWidget(QLabel(tr("Leg")))
        self.leg = QComboBox()
        for key in ("both", "right", "left"):
            self.leg.addItem(tr("Both") if key == "both" else tr(key).capitalize(), key)
        self.leg.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.leg)
        bar.addSpacing(12)
        for text, slot, tip in ((tr("Open…"), self.open_comparison, tr("Open a saved comparison (.okcmp)")),
                                (tr("Save…"), self.save_comparison, tr("Save which analyses are compared and their names")),
                                (tr("Export CSV…"), self.export_csv, tr("Every table, for a spreadsheet")),
                                (tr("Notes…"), self.edit_notes, tr("Your name and your observations and recommendations: "
                                                                   "kept with the comparison, they go into its report"))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        from openkick.ui.report import report_menu
        report = QPushButton(tr("Report") + "  ▾")
        report.setToolTip(tr("Make the report of the comparison: PDF, plain text or JSON (the numbers, for research)"))
        report.setMenu(report_menu(self, self.export_report, ("anonymous",)))
        bar.addWidget(report)
        close = QToolButton()
        close.setText("✕")
        close.setToolTip(tr("Close"))
        close.clicked.connect(self.close)
        bar.addWidget(close)
        lay.addLayout(bar)

        cards = QScrollArea()
        cards.setWidgetResizable(True)
        cards.setFixedHeight(118)
        cards.setFrameShape(QFrame.NoFrame)
        self.cards_box = QWidget()
        self.cards = QHBoxLayout(self.cards_box)
        self.cards.setContentsMargins(0, 0, 0, 0)
        self.cards.setSpacing(10)
        self.cards.addStretch(1)
        cards.setWidget(self.cards_box)
        lay.addWidget(cards)

        self.body = QScrollArea()
        self.body.setWidgetResizable(True)
        self.body.setFrameShape(QFrame.NoFrame)
        lay.addWidget(self.body, 1)
        self.tables = []
        self.refresh()

    # ── adding and removing ──
    def browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, tr("Add analyses"), "", FILES)
        if paths:
            self.add_files(paths)

    def add_files(self, paths, labels=None):
        bad = []
        for k, path in enumerate(paths):
            try:
                label = labels[k] if labels and k < len(labels) else None
                activity = self.items[0].activity if self.items else None
                self.items.append(K.load(path, activity=activity, label=label))
            except Exception as e:  # noqa: BLE001 - one unreadable file must not stop the others
                bad.append(f"{Path(path).name}: {e}")
        if bad:
            QMessageBox.warning(self, tr("Compare analyses"), tr("These files could not be read:") + "\n" + "\n".join(bad))
        self.refresh()

    def _remove(self, card):
        self.items.remove(card.item)
        self.refresh()

    def _move(self, card, step):
        i = self.items.index(card.item)
        j = max(0, min(len(self.items) - 1, i + step))
        self.items[i], self.items[j] = self.items[j], self.items[i]
        self.refresh()

    def dragEnterEvent(self, e):
        if any(u.toLocalFile().lower().endswith((".okick", ".ok3d", ".okcmp")) for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls()]
        cmp = [f for f in files if f.lower().endswith(".okcmp")]
        if cmp:
            self.load_comparison(cmp[0])
        else:
            self.add_files([f for f in files if f.lower().endswith((".okick", ".ok3d"))])
        e.acceptProposedAction()

    # ── showing ──
    def legs(self) -> list[str]:
        choice = self.leg.currentData()
        if choice != "both":
            return [choice]
        present = {s for i in self.items for s in i.sides()}
        return [s for s in ("right", "left") if s in present]

    def refresh(self, *_):
        while self.cards.count() > 1:
            w = self.cards.takeAt(0).widget()
            if w:
                w.deleteLater()
        for k, item in enumerate(self.items):
            card = AnalysisCard(item, COLORS[k % len(COLORS)])
            card.removed.connect(self._remove)
            card.moved.connect(self._move)
            card.renamed.connect(self.refresh)
            self.cards.insertWidget(self.cards.count() - 1, card)
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)
        self.tables = []
        if len(self.items) < 2:
            msg = QLabel(tr("Add two or more analyses to compare them: the same person in different conditions "
                            "(two saddle heights, before and after) or different people. 2D projects and 3D "
                            "sessions can be mixed; each one is compared leg against leg."))
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet(f"color: {theme.MUTED}; font-size: 14px; padding: 40px;")
            col.addWidget(msg)
            col.addStretch(1)
            self.body.setWidget(page)
            return
        kinds = {i.activity for i in self.items}
        lines = K.summary(self.items)
        if len(kinds) > 1:
            lines.insert(0, tr("These analyses are of different activities: only what they share can be compared."))
        lines.append(tr("● the difference is more than what changes from one cycle to the next. The first analysis "
                        "is the reference."))
        s = QLabel("\n".join(lines))
        s.setWordWrap(True)
        s.setStyleSheet(f"color: {theme.MUTED};")
        col.addWidget(s)
        per_height = bool(self.people.currentData())
        who = K.subjects_table(self.items)
        if who is not None:
            col.addWidget(self._table(who))
        for side in self.legs():
            t = K.table(self.items, side, per_height)
            col.addWidget(self._table(t))
            if self.items[0].activity == "cycling":
                charts = QHBoxLayout()
                for key, label in CURVES:
                    ch = CycleChart(f"{tr(label)} · {t.title}")
                    ch.set_items([(i.label, COLORS[k % len(COLORS)], (i.curves.get(side) or {}).get(key))
                                  for k, i in enumerate(self.items)])
                    charts.addWidget(ch)
                col.addLayout(charts)
        col.addStretch(1)
        self.body.setWidget(page)

    def _table(self, t) -> QWidget:
        self.tables.append(t)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        h = QLabel(t.title)
        h.setStyleSheet("font-weight: 600; font-size: 14px;")
        lay.addWidget(h)
        tw = QTableWidget(len(t.rows), len(t.columns))
        tw.setHorizontalHeaderLabels(t.columns)
        tw.verticalHeader().hide()
        tw.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tw.setAlternatingRowColors(True)
        tw.setShowGrid(False)
        for r, row in enumerate(t.rows):
            for c, v in enumerate(row):
                it = QTableWidgetItem(str(v))
                it.setToolTip(str(v))
                if c > 0:
                    it.setFont(theme.mono(12))
                    it.setTextAlignment(Qt.AlignCenter)       # under its column's name
                if "●" in str(v):
                    it.setForeground(QColor(theme.ACCENT))
                tw.setItem(r, c, it)
        tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tw.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tw.resizeRowsToContents()
        rows = sum(tw.rowHeight(r) for r in range(tw.rowCount()))
        tw.setFixedHeight(tw.horizontalHeader().sizeHint().height() + rows + 2 * tw.frameWidth() + 4)
        tw.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(tw)
        return box

    # ── files ──
    def save_comparison(self):
        if len(self.items) < 1:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Save the comparison"), str(self.path or "comparacion.okcmp"),
                                              "OpenKick (*.okcmp)")
        if not path:
            return
        if not path.lower().endswith(".okcmp"):
            path += ".okcmp"
        folder = Path(path).resolve().parent
        rows = []
        for i in self.items:
            full = Path(i.path).resolve()
            try:
                rel = str(full.relative_to(folder))
            except ValueError:
                rel = None
            rows.append({"path": str(full), "relative": rel, "label": i.label})
        Path(path).write_text(json.dumps({"format": "openkick-comparison", "version": 1, "analyses": rows,
                                          "different_people": bool(self.people.currentData()), "notes": self.notes,
                                          "evaluator": self.evaluator},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
        self.path = Path(path)

    def open_comparison(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open a comparison"), "", "OpenKick (*.okcmp)")
        if path:
            self.load_comparison(path)

    def load_comparison(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        folder = Path(path).resolve().parent
        paths, labels = [], []
        for row in data.get("analyses", []):
            here = folder / row["relative"] if row.get("relative") else None
            paths.append(str(here if here and here.exists() else row["path"]))
            labels.append(row.get("label"))
        self.items = []
        self.notes = data.get("notes", "")
        self.evaluator = data.get("evaluator", "")
        self.people.setCurrentIndex(1 if data.get("different_people") else 0)
        self.path = Path(path)
        self.add_files(paths, labels)

    def export_csv(self):
        if not self.tables:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Export CSV…"), "comparacion.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for t in self.tables:
                w.writerow([t.title])
                w.writerow(t.columns)
                w.writerows(t.rows)
                w.writerow([])

    def edit_notes(self):
        from openkick.ui.report import NotesDialog
        dlg = NotesDialog(tr("Notes"), self.evaluator, self.notes, self)
        if dlg.exec():
            self.evaluator, self.notes = dlg.values()

    def export_report(self, fmt: str = "pdf"):
        """The report of the comparison in this format ("pdf", "txt" or "json")."""
        if len(self.items) < 2:
            return
        from openkick.ui.report import ask_path, comparison_report, option
        path = ask_path(self, fmt, self.path or "comparacion")
        if not path:
            return
        try:
            comparison_report(self, path, self.evaluator, self.notes, option("anonymous"))
        except OSError as e:
            QMessageBox.critical(self, tr("Compare analyses"), str(e))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)
