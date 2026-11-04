"""The plugin store: every package of analyses, installed or not, with what it brings.

For now every package ships with OpenKick, so installing one only shows it again and removing
one hides it; the page is the one an online store will use when the packages come from a
repository."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from openkick.core import store as ST
from openkick.i18n import pick, tr
from openkick.ui import icons, theme


class PackageCard(QFrame):
    def __init__(self, pkg: dict, on: bool, toggle, parent=None):
        super().__init__(parent)
        self.pkg = pkg
        self.setObjectName("Card")
        self.setStyleSheet(f"QFrame#Card {{ background: {theme.CARD}; border: 1px solid {theme.BORDER}; border-radius: 6px; }}"
                           " QLabel { border: 0; }")
        self.setFixedHeight(184)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        top = QHBoxLayout()
        ic = QLabel()
        ic.setFixedSize(38, 38)
        ic.setAlignment(Qt.AlignCenter)
        ic.setStyleSheet(f"background: {theme.ACTIVE}; border-radius: 6px;")
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


COMMUNITY_URL = "https://slightlydazed.github.io/openkick/"     # the index of the community's plugins


def install_plugin_zip(parent, path: str | None = None) -> bool:
    """Ask for a plugin's ZIP (the one GitHub gives with Code › Download ZIP) and install it.
    True when something was installed."""
    from openkick.plugins import install as INS
    if path is None:
        path, _ = QFileDialog.getOpenFileName(parent, tr("Install plugin (.zip)…"), "", tr("Plugin (*.zip)"))
        if not path:
            return False
    try:
        try:
            man = INS.install_zip(path)
        except INS.AlreadyInstalled as e:
            old = e.installed.get("version", "?")
            ask = QMessageBox.question(parent, tr("Install plugin (.zip)…"),
                                       tr("{name} is already installed (version {old}). Replace it with version {new}?",
                                          name=pick(e.manifest["name"]), old=old, new=e.manifest["version"]))
            if ask != QMessageBox.Yes:
                return False
            man = INS.install_zip(path, replace=True)
    except INS.InstallError as e:
        QMessageBox.warning(parent, tr("Install plugin (.zip)…"), tr("This ZIP cannot be installed: {why}", why=e))
        return False
    QMessageBox.information(parent, tr("Install plugin (.zip)…"),
                            tr("{name} {version} is installed. It is in the Plugins menu. A plugin runs its own code: "
                               "use only those you trust.", name=pick(man["name"]), version=man["version"]))
    return True


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
        web = QPushButton(icons.icon("globe", theme.TEXT, 16), " " + tr("Community plugins"))
        web.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(COMMUNITY_URL)))
        head.addWidget(web)
        self.zip_btn = QPushButton(icons.icon("plus", theme.TEXT, 16), " " + tr("Install plugin (.zip)…"))
        self.zip_btn.clicked.connect(self._install_zip)
        head.addWidget(self.zip_btn)
        lay.addLayout(head)
        note = QLabel(tr("Each package brings one kind of analysis: its tables, its template and, where there is "
                         "one, its 3D. These come with OpenKick: removing one hides it and installing it brings it "
                         "back. Plugins made by others are on the community page: download their ZIP and install it here."))
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
                            f"QPushButton:checked {{ background: {theme.SELECT}; border-color: {theme.ACCENT}; color: {theme.ACCENT}; font-weight: 600; }}")
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

    def _install_zip(self):
        if install_plugin_zip(self):
            self.changed_any = True

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
