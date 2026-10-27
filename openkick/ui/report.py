"""Reports of an open 3D session and of a comparison of analyses: PDF, plain text or JSON.

Both carry the patient's record, what was found, how far it can be trusted, and what the person
who reviewed it concludes (their name and observations, written just before exporting). The PDF
and the text are the same report (the text without pictures); the JSON has the numbers
unrounded, for a spreadsheet or a script (core/report_data.py)."""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QMenu,
                               QPlainTextEdit, QVBoxLayout, QWidget)

from openkick.core import compare as K
from openkick.core import report_data as D
from openkick.core.subject import Subject, describe
from openkick.i18n import tr
from openkick.ui import icons, theme
from openkick.ui.exporters import CSS, _table, print_pdf

MORE_CSS = """
.notes { background: #F4F6F8; padding: 8pt; }
.warn { color: #9A3412; }
.small { color: #5B6474; font-size: 8.5pt; }
"""
LIMITS = ("The 3D analysis is experimental. Knee, hip and ankle were checked against a motion capture laboratory "
          "in squats and walking (3 to 9 degrees of mean error); cycling has not been checked against one. "
          "Read the numbers as what changes between recordings made the same way, and check in the videos "
          "whatever looks odd.")
STAR = "* What the detector places least reliably (the foot)."
DOT = "● The difference is larger than what changes from one cycle to the next. The first analysis is the reference."
# format -> (menu entry, file filter)
FORMATS = {"pdf": ("PDF report…", "PDF (*.pdf)"), "txt": ("Report as text (.txt)…", "{text} (*.txt)"),
           "json": ("Data for research (.json)…", "JSON (*.json)")}
# what a report includes: key -> (menu entry, on unless the user turned it off)
OPTIONS = {"appendix": ("Include every pedal stroke or stride, as an appendix", True),
           "pictures": ("Include pictures of one moment in every video and in 3D (PDF)", True),
           "anonymous": ("Without what identifies the patient (name, date of birth, notes)", False),
           "series": ("JSON: also the values frame by frame", False)}


def option(key: str) -> bool:
    v = QSettings().value(f"report/{key}", OPTIONS[key][1])
    return v in (True, "true", "1", 1)


def report_menu(parent, export, options=tuple(OPTIONS)) -> QMenu:
    """The Report button's menu: each format makes the report at once; below, what it includes
    (remembered)."""
    m = QMenu(parent)
    for fmt, (label, _) in FORMATS.items():
        m.addAction(icons.icon({"pdf": "pdf", "txt": "text", "json": "table"}[fmt], theme.TEXT_2, 16), tr(label),
                    lambda f=fmt: export(f))
    m.addSeparator()
    for key in options:
        a = m.addAction(tr(OPTIONS[key][0]))
        a.setCheckable(True)
        a.setChecked(option(key))
        a.toggled.connect(lambda on, k=key: QSettings().setValue(f"report/{k}", bool(on)))
    return m


def ask_path(parent, fmt: str, default) -> str | None:
    """Where to save a report in this format."""
    from PySide6.QtWidgets import QFileDialog
    label, filt = FORMATS[fmt]
    path, _ = QFileDialog.getSaveFileName(parent, tr(label).rstrip("…"), str(Path(default).with_suffix("." + fmt)),
                                          filt.format(text=tr("Text")))
    if not path:
        return None
    return path if path.lower().endswith("." + fmt) else path + "." + fmt


class NotesDialog(QDialog):
    """The evaluator's name and observations: kept with the analysis, printed in its report."""

    def __init__(self, title: str, evaluator: str = "", notes: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        head = QLabel(title)
        head.setFont(theme.display(20))
        lay.addWidget(head)
        hint = QLabel(tr("Who evaluated this analysis and what they conclude. It is kept with it and goes into its "
                         "report (the Report button)."))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(hint)
        form = QFormLayout()
        self.evaluator = QLineEdit(evaluator or str(QSettings().value("report/evaluator", "") or ""))
        self.evaluator.setPlaceholderText(tr("Name, profession, registration number"))
        form.addRow(tr("Evaluator"), self.evaluator)
        lay.addLayout(form)
        lay.addWidget(QLabel(tr("Observations and recommendations")))
        self.notes = QPlainTextEdit(notes)
        self.notes.setPlaceholderText(tr("What you conclude from the analysis: it goes into the report as you write it."))
        lay.addWidget(self.notes, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Save"))
        buttons.button(QDialogButtonBox.Ok).setProperty("primary", True)
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def values(self) -> tuple[str, str]:
        who = self.evaluator.text().strip()
        if who:
            QSettings().setValue("report/evaluator", who)          # the next analysis starts with it
        return who, self.notes.toPlainText().strip()


# ── a report, in sections ──────────────────────────────────────────────

class Report:
    """What a report says, in order: written out as HTML (for the PDF) or as plain text."""

    def __init__(self, title: str):
        self.title = title
        self.items: list[tuple] = []
        self.images: dict = {}

    def add(self, kind: str, *args):
        self.items.append((kind, *args))

    def picture(self, name: str, img, width: int):
        self.images[name] = img
        self.add("image", name, width)

    def html(self, doc: QTextDocument) -> str:
        out, row = [], []

        def flush():
            if row:
                out.append("<p>" + " ".join(row) + "</p>")       # pictures in their own paragraph
                row.clear()
        e = html.escape
        for kind, *a in self.items:
            if kind == "image":
                doc.addResource(QTextDocument.ImageResource, QUrl(a[0]), self.images[a[0]])
                row.append(f"<img src='{a[0]}' width='{a[1]}'>")
                continue
            flush()
            if kind in ("h1", "h2", "h3"):
                out.append(f"<{kind}>{e(a[0])}</{kind}>")
            elif kind == "meta":
                out.append(f"<p class='meta'>{e(a[0])}</p>")
            elif kind == "p":
                out.append(f"<p class='{a[1]}'>{e(a[0])}</p>")
            elif kind == "table":
                out.append(_table(a[0], a[1], a[2]))
            elif kind == "notes":
                out.append("<div class='notes'>" + "<br>".join(e(line) for line in a[0].splitlines()) + "</div>")
            elif kind == "foot":
                out.append(f"<p class='foot'>{e(a[0])}</p>")
        flush()
        return "<body>" + "".join(out) + "</body>"

    def text(self) -> str:
        out = []
        for kind, *a in self.items:
            if kind == "h1":
                out += [a[0], "=" * len(a[0]), ""]
            elif kind == "h2":
                out += ["", a[0], "-" * len(a[0])]
            elif kind == "h3":
                out += ["", a[0]]
            elif kind in ("meta", "p", "notes", "foot"):
                out += [a[0], ""]
            elif kind == "table":
                out += [text_table(a[0], a[1]), ""]
        return "\n".join(out).rstrip() + "\n"

    def save(self, path: str):
        if path.lower().endswith(".txt"):
            Path(path).write_text(self.text(), encoding="utf-8")
            return
        doc = QTextDocument()
        doc.setDefaultStyleSheet(CSS + MORE_CSS)
        doc.setHtml(self.html(doc))
        print_pdf(doc, path, self.title)


def text_table(columns: list, rows: list) -> str:
    """Columns lined up with spaces, for a plain text file."""
    cells = [[str(c) for c in columns]] + [[str(c) for c in r] for r in rows]
    width = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(len(columns))]
    lines = ["  ".join(c.ljust(width[i]) for i, c in enumerate(r)).rstrip() for r in cells]
    lines.insert(1, "  ".join("-" * w for w in width))
    return "\n".join(lines)


def _who(subject, anonymous: bool):
    if subject is None or not anonymous:
        return subject
    s = Subject.from_dict(subject.to_dict())
    s.name, s.notes = "", ""             # the age stays (reports show the age, not the date of birth)
    return s


def _observations(r: Report, evaluator: str, notes: str):
    if not evaluator and not notes:
        return
    r.add("h2", tr("Observations and recommendations"))
    if notes:
        r.add("notes", notes)
    if evaluator:
        r.add("p", f"{tr('Evaluator')}: {evaluator}", "")


def _foot(r: Report):
    from openkick import AUTHOR, __version__
    r.add("foot", f"{tr('Generated with OpenKick {v} · openkick project', v=__version__)} · "
                  f"{tr('Developed by {name}', name=AUTHOR)} · {date.today().isoformat()}")


def _charts(r: Report, prefix: str, lines: list, width: int = 640):
    """Knee, hip and ankle over the cycle, one under the other: `lines` [(label, colour, {"knee": curve...})]."""
    from openkick.ui.compare import CURVES, CycleChart
    for key, label in CURVES:
        items = [(name, color, curves.get(key)) for name, color, curves in lines if curves and curves.get(key)]
        if not items:
            continue
        chart = CycleChart(tr(label))
        chart.set_items(items)
        r.picture(f"{prefix}_{key}.png", chart.image(900, 300), width)
        r.add("break")


# ── a 3D session ───────────────────────────────────────────────────────

def _representative_row(ctl) -> int:
    """A moment worth showing: the bottom of a typical pedal stroke, the start of a stride, else
    the middle of the clip."""
    s = ctl.session
    kind = ctl.activity_kind()
    if kind == "cycling":
        c = ctl.cycling()
        if c is not None and c.strokes:
            st = c.strokes[len(c.strokes) // 2]
            knee = c.series.get(c.side, {}).get("knee")
            a, b = int(st.get("first", st.get("a", 0))), int(st.get("last", st.get("b", 0)))
            if knee is not None and b > a and np.isfinite(knee[a:b + 1]).any():
                return a + int(np.nanargmax(knee[a:b + 1]))
            return a
    if kind == "gait":
        g = ctl.gait()
        if g is not None and g.phases:
            return int(g.phases[len(g.phases) // 2].start)
    return s.frame_count // 2


def _activity(ctl):
    """(kind, the activity's analysis, its Result, its compare.Analysis) of the open session."""
    s = ctl.session
    kind = ctl.activity_kind()
    name = s.path.stem if s.path else tr("3D session")
    obj = ctl.cycling() if kind == "cycling" else ctl.gait() if kind == "gait" else None
    res = obj.result if obj is not None else ctl.template_result() if kind == "template" else None
    item = K.from_session(s, str(s.path or ""), name, kind) if kind in ("cycling", "gait") else None
    return kind, obj, res, item


def _template_angles(s):
    from openkick.core import templates
    from openkick.multiview.template3d import angles
    t = templates.find(s.template) if s.template else None
    return angles(s.X, s.t, t) if t else []


def session_report(window, path: str, evaluator: str = "", notes: str = "", appendix: bool = True,
                   pictures: bool = True, anonymous: bool = False, series: bool = False):
    """The report of the 3D session open in `window` (a MultiviewWindow), as the ending of `path`
    says: .pdf, .txt or .json. `series`: in the JSON, the per-frame values too."""
    ctl = window.ctl
    s = ctl.session
    kind, obj, res, item = _activity(ctl)
    if path.lower().endswith(".json"):
        found = _template_angles(s) if kind == "template" else None
        frames = None
        if series:
            frames = {"t_s": s.t - s.t[0]}
            if kind == "cycling" and obj is not None:
                frames.update({side: obj.series[side] for side in ("left", "right") if obj.series.get(side)})
            elif kind == "template":
                frames.update(dict(found))
        data = D.session(s, kind, obj, res, item, evaluator, notes, anonymous, frames, found)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    session_document(window, kind, res, item, evaluator, notes, appendix,
                     pictures and path.lower().endswith(".pdf"), anonymous).save(path)


def session_document(window, kind, res, item, evaluator, notes, appendix, pictures, anonymous) -> Report:
    from openkick.ui.multiview import FEET, LEVELS, SMOOTHING, num
    ctl = window.ctl
    s = ctl.session
    name = s.path.stem if s.path else tr("3D session")
    r = Report(f"{tr('3D analysis')} · {name}")
    what = {"cycling": tr("Cycling"), "gait": tr("Gait"), "template": tr("Template angles")}.get(kind, tr("3D reconstruction"))
    smooth = next((t for hz, t in SMOOTHING if abs(hz - (s.smooth_hz or 0.0)) < 1e-6), SMOOTHING[0][1])
    settings = [f"{tr('3D smoothing')}: {tr(smooth)}",
                f"{tr('Feet')}: {tr(dict(FEET).get(s.feet or '', FEET[0][1]))}"]
    r.add("h1", f"{tr('3D analysis')}: {name}")
    r.add("meta", f"{what} · {tr('{n} videos', n=len(s.views))} · {date.today().isoformat()}")
    who = _who(s.subject, anonymous)
    if who and not who.empty:
        r.add("h2", tr("Patient"))
        r.add("table", [tr("Item"), tr("Value")], describe(who, tr), 2)
    _observations(r, evaluator, notes)

    rows = [[tr("Video {n}", n=i + 1), Path(v.meta.path).name, f"{v.meta.width}×{v.meta.height}", f"{v.meta.fps:.2f}",
             f"{s.offsets[i]:+.3f} s" if s.offsets else "—", tr("handheld, compensated") if v.handheld else tr("still")]
            for i, v in enumerate(s.views)]
    r.add("h2", tr("Recording"))
    r.add("table", [tr("Video"), tr("File"), tr("Size"), tr("fps"), tr("Offset"), tr("Camera")], rows, 2)
    r.add("p", f"{tr('Settings')}: {' · '.join(settings)}", "small")
    q = s.quality()
    doubtful = [side for side, ok in (item.reliable.items() if item else []) if not ok]
    pending = sum(1 for e in s.stretches() if not e["reviewed"])
    r.add("h2", tr("How far it can be trusted"))
    r.add("table", [tr("Item"), tr("Value")],
          [[tr("Overall"), tr(LEVELS[q["level"]][0])], [tr("The videos disagree"), f"{num(q['mismatch_cm'])} cm"],
           [tr("The leg bones change length"), f"{num(q['legs_variation'], 0)} %"],
           [tr("Points reconstructed"), f"{num(100 * q['reconstructed'], 0)} %"],
           [tr("Points corrected by hand"), str(q["manual"])], [tr("Doubtful stretches left to check"), str(pending)]], 1)
    if doubtful:
        legs = " / ".join(tr("left") if d == "left" else tr("right") for d in doubtful)
        r.add("p", tr("The 3D of the {side} leg is doubtful: its thigh or shank changes length by more than 10 %. "
                      "Read its numbers with care.", side=legs), "warn")

    later = []
    r.add("h2", tr("What was found"))
    if res is not None:
        if res.summary:
            r.add("p", res.summary, "")
        main = []
        for tb in res.tables:
            (later if tb.title in (tr("Stroke by stroke"), tr("Stride by stride")) else main).append(tb)
        for tb in main:
            r.add("h3", tb.title)
            r.add("table", tb.columns, tb.rows, 1)
        if pictures and kind == "cycling" and item is not None and item.curves:
            from openkick.ui.compare import COLORS
            r.add("h2", tr("The mean pedal stroke"))
            _charts(r, "cycle", [(tr("Right leg"), COLORS[0], item.curves.get("right")),
                                 (tr("Left leg"), COLORS[1], item.curves.get("left"))])
            r.add("p", tr("From 0 to 100 % of the stroke; the band is how much it varies from one stroke to the next."), "small")
        if any("*" in str(row[0]) for tb in main for row in tb.rows):
            r.add("p", tr(STAR), "small")
    else:
        r.add("p", tr("No activity analysed in this session: only its reconstruction."), "")

    if pictures:
        row = _representative_row(ctl)
        keep = ctl.row
        ctl.set_row(row)
        QApplication.processEvents()
        r.add("h2", tr("One moment, in every video and in 3D"))
        for i, cv in enumerate(window.canvases):
            buttons = [b for b in cv.findChildren(QWidget) if b.isVisible()]   # its own buttons stay out
            for b in buttons:
                b.hide()
            r.picture(f"view{i}.png", cv.grab().toImage(), 300)
            for b in buttons:
                b.show()
        r.picture("view3d.png", window.view3d.grab().toImage(), 300)
        ctl.set_row(keep)
        r.add("p", tr("At {t} s. Coloured skeleton: what the detector saw; white lines: the 3D seen from that camera.",
                      t=num(s.t[row] - s.t[0], 2)), "small")
    if appendix:
        for tb in later:
            r.add("h2", tr("Appendix") + " · " + tb.title)
            r.add("table", tb.columns, tb.rows, 1)
    r.add("h2", tr("Limits"))
    r.add("p", tr(LIMITS), "small")
    _foot(r)
    return r


# ── a comparison ───────────────────────────────────────────────────────

def comparison_report(window, path: str, evaluator: str = "", notes: str = "", anonymous: bool = False):
    """The report of what the comparison window (CompareWindow) shows: .pdf, .txt or .json."""
    items = window.items
    per_height = bool(window.people.currentData())
    if path.lower().endswith(".json"):
        data = D.comparison(items, window.legs(), per_height, evaluator, notes, anonymous)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    from openkick.ui.compare import COLORS
    labels = [i.label for i in items]
    r = Report(f"{tr('Comparison')} · {' · '.join(labels)}")
    r.add("h1", f"{tr('Comparison')}: {' · '.join(labels)}")
    r.add("meta", f"{tr('Different people') if per_height else tr('The same person')} · {date.today().isoformat()}")
    people = [_who(i.subject, anonymous) for i in items]
    rows = [[i.label, i.kind, tr({"cycling": "Cycling", "gait": "Gait"}.get(i.activity, i.activity or "—")),
             Path(i.path).name, (p.name if p and p.name else "—")] for i, p in zip(items, people)]
    r.add("h2", tr("Analyses"))
    r.add("table", [tr("Name"), tr("Kind"), tr("Activity"), tr("File"), tr("Patient")], rows, 9)
    who = K.subjects_table([K.Analysis(i.path, i.label, i.kind, i.activity, p) for i, p in zip(items, people)])
    if who is not None:
        r.add("h2", who.title)
        r.add("table", who.columns, who.rows, 1)
    _observations(r, evaluator, notes)
    lines = K.summary(items)
    if len({i.activity for i in items}) > 1:
        lines.insert(0, tr("These analyses are of different activities: only what they share can be compared."))
    if lines:
        r.add("h2", tr("What was found"))
        for x in lines:
            r.add("p", x, "")
    pictures = path.lower().endswith(".pdf")
    for side in window.legs():
        t = K.table(items, side, per_height)
        r.add("h3", t.title)
        r.add("table", t.columns, t.rows, 1)
        if pictures and items and items[0].activity == "cycling":
            _charts(r, f"cmp_{side}", [(i.label, COLORS[k % len(COLORS)], (i.curves.get(side) or {}))
                                       for k, i in enumerate(items)])
    r.add("p", tr(DOT), "small")
    r.add("p", tr(STAR), "small")
    if any(i.kind == "3D" for i in items):
        r.add("h2", tr("Limits"))
        r.add("p", tr(LIMITS), "small")
    _foot(r)
    r.save(path)
