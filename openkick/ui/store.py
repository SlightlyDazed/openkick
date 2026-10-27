"""The plugin store: every package of analyses, installed or not, with what it brings.

For now every package ships with OpenKick, so installing one only shows it again and removing
one hides it; the page is the one an online store will use when the packages come from a
repository."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from openkick.core import store as ST
from openkick.i18n import pick, tr
from openkick.ui import icons, theme


class PackageCard(QFrame):
    def __init__(self, pkg: dict, on: bool, toggle, parent=None):
        super().__init__(parent)
        self.pkg = pkg
        self.setObjectName("Card")
        self.setStyleSheet(f"QFrame#Card {{ background: {theme.CARD}; border: 1px solid {theme.BORDER}; border-radius: 12px; }}"
                           " QLabel { border: 0; }")
        self.setFixedHeight(184)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        top = QHBoxLayout()
        ic = QLabel()
        ic.setFixedSize(38, 38)
        ic.setAlignment(Qt.AlignCenter)
        ic.setStyleSheet(f"background: {theme.ACTIVE}; border-radius: 10px;")
        ic.setPixmap(icons.pixmap(pkg["icon"], theme.ACCENT, 20))
        top.addWidget(ic)
        names = QVBoxLayout()
        n = QLabel(pick(pkg["name"]))
        n.setStyleSheet("font-weight: 600; font-size: 14px;")
        names.addWidget(n)
        c = QLabel(tr(pkg["category"]) + f" · v{ST.VERSION} · OpenKick")
        c.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        names.addWidget(c)
        top.addLayout(names, 1)
        lay.addLayout(top)
        d = QLabel(pick(pkg["description"]))
        d.setWordWrap(True)
        d.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px;")
        lay.addWidget(d, 1)
        brings = []
        if pkg["plugins"]:
            brings.append(tr("analysis"))
        if pkg["templates"]:
            brings.append(tr("template"))
        if pkg["activity"]:
            brings.append(tr("3D"))
        row = QHBoxLayout()
        b = QLabel(tr("Includes: {what}", what=", ".join(brings)) if brings else "")
        b.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        row.addWidget(b, 1)
        if pkg["base"]:
            btn = QPushButton(tr("Part of OpenKick"))
            btn.setEnabled(False)
        else:
            btn = QPushButton(tr("Remove") if on else tr("Install"))
            btn.setProperty("primary", not on)
            btn.clicked.connect(lambda: toggle(pkg["id"], not on))
        row.addWidget(btn)
        lay.addLayout(row)


class StoreDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Plugin store"))
        self.resize(980, 720)
        self.changed_any = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 18)
        lay.setSpacing(12)
        head = QHBoxLayout()
        t = QLabel(tr("Plugin store"))
        t.setFont(theme.display(26))
        head.addWidget(t)
        head.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("Search…"))
        self.search.setFixedWidth(240)
        self.search.textChanged.connect(self.refresh)
        head.addWidget(self.search)
        lay.addLayout(head)
        note = QLabel(tr("Each package brings one kind of analysis: its tables, its template and, where there is "
                         "one, its 3D. For now they all come with OpenKick: removing one hides it and installing "
                         "it brings it back. Later they will be downloaded from an online store."))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED};")
        lay.addWidget(note)
        chips = QHBoxLayout()
        self.category = None
        self.chip_buttons = {}
        for key in [None] + ST.CATEGORIES:
            b = QPushButton(tr("All") if key is None else tr(key))
            b.setCheckable(True)
            b.setChecked(key is None)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(f"QPushButton {{ background: transparent; border: 1px solid {theme.BORDER}; min-height: 30px; padding: 0 14px; }}"
                            f"QPushButton:checked {{ background: #171D12; border-color: {theme.ACCENT}; color: {theme.ACCENT}; font-weight: 600; }}")
            b.clicked.connect(lambda _=False, k=key: self._pick(k))
            self.chip_buttons[key] = b
            chips.addWidget(b)
        chips.addStretch(1)
        lay.addLayout(chips)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        lay.addWidget(self.scroll, 1)
        foot = QHBoxLayout()
        self.count = QLabel()
        self.count.setStyleSheet(f"color: {theme.DIM};")
        foot.addWidget(self.count, 1)
        close = QPushButton(tr("Close"))
        close.setProperty("primary", True)
        close.clicked.connect(self.accept)
        foot.addWidget(close)
        lay.addLayout(foot)
        self.refresh()

    def _pick(self, key):
        self.category = key
        for k, b in self.chip_buttons.items():
            b.setChecked(k == key)
        self.refresh()

    def _toggle(self, pid: str, on: bool):
        ST.set_installed(pid, on)
        self.changed_any = True
        self.refresh()

    def refresh(self, *_):
        on = ST.installed()
        q = self.search.text().strip().lower()
        page = QWidget()
        grid = QGridLayout(page)
        grid.setSpacing(12)
        shown = 0
        for pkg in ST.CATALOG:
            if self.category and pkg["category"] != self.category:
                continue
            text = (pick(pkg["name"]) + " " + pick(pkg["description"])).lower()
            if q and q not in text:
                continue
            grid.addWidget(PackageCard(pkg, pkg["id"] in on, self._toggle), shown // 3, shown % 3)
            shown += 1
        grid.setRowStretch(shown // 3 + 1, 1)
        self.scroll.setWidget(page)
        self.count.setText(tr("{a} of {b} packages installed", a=len(on), b=len(ST.CATALOG)))
