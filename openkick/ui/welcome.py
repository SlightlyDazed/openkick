"""Start page: a sidebar with everything the program can do, open a video (drag & drop),
pick a template, reopen recent work."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from openkick import AUTHOR, __version__
from openkick.core import templates
from openkick.core.pose import data_dir
from openkick.i18n import tr
from openkick.ui import icons, theme

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".wmv", ".mpg", ".mpeg"}


def thumbnail(path: str, w: int = 96, h: int = 54) -> QPixmap:
    """First frame of a video (cached on disk); a neutral tile when unavailable."""
    pm = QPixmap(w * 2, h * 2)
    pm.fill(QColor("#2B3037"))
    src = Path(path)
    if src.suffix.lower() in (".okick", ".kva", ".ok3d"):
        try:
            if src.suffix.lower() == ".ok3d":
                from openkick.multiview.session import Session
                src = Path(Session.read(src)["videos"][0]["path"] or src)
            elif src.suffix.lower() == ".okick":
                import json
                v = json.loads(src.read_text(encoding="utf-8"))["video"]["path"]
                src = (src.parent / v) if not Path(v).is_absolute() else Path(v)
            else:
                from openkick.formats.kva import find_video
                src = find_video(src) or src
        except Exception:  # noqa: BLE001
            pass
    if src.suffix.lower() in VIDEO_EXT and src.exists():
        cache = data_dir() / "thumbs"
        cache.mkdir(exist_ok=True)
        key = hashlib.sha1(f"{src.resolve()}{src.stat().st_mtime}".encode()).hexdigest()[:16]
        f = cache / f"{key}.jpg"
        if not f.exists():
            try:
                import av
                with av.open(str(src)) as c:
                    arr = next(c.decode(c.streams.video[0])).to_ndarray(format="rgb24")
                qi = QImage(arr.data, arr.shape[1], arr.shape[0], 3 * arr.shape[1], QImage.Format_RGB888)
                qi.scaled(w * 4, h * 4, Qt.KeepAspectRatio, Qt.SmoothTransformation).save(str(f), "JPG", 85)
            except Exception:  # noqa: BLE001
                return pm
        img = QImage(str(f))
        if not img.isNull():
            scaled = img.scaled(w * 2, h * 2, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (scaled.width() - w * 2) // 2
            y = (scaled.height() - h * 2) // 2
            pm = QPixmap.fromImage(scaled.copy(x, y, w * 2, h * 2))
    pm.setDevicePixelRatio(2)
    return pm


class DropZone(QFrame):
    openVideo = Signal()
    openProject = Signal()
    dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(250)
        self._hover = False
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(16)
        ic = QLabel()
        ic.setFixedSize(56, 56)
        ic.setAlignment(Qt.AlignCenter)
        ic.setStyleSheet(f"background: {theme.RAISED}; border-radius: 16px;")
        ic.setPixmap(icons.pixmap("upload", theme.ACCENT, 28))
        lay.addWidget(ic, 0, Qt.AlignHCenter)
        t = QLabel(tr("Drop a video here"))
        t.setStyleSheet("font-size: 18px; font-weight: 600;")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        s = QLabel(tr("MP4, MOV or AVI · also .okick projects and Kinovea .kva annotations"))
        s.setObjectName("Faint")
        s.setStyleSheet(f"color: {theme.FAINT}; font-size: 13px;")
        s.setAlignment(Qt.AlignCenter)
        lay.addWidget(s)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch(1)
        b1 = QPushButton(tr("Open video"))
        b1.setProperty("primary", True)
        b1.setMinimumHeight(44)
        b1.clicked.connect(self.openVideo)
        row.addWidget(b1)
        b2 = QPushButton(tr("Open project or KVA"))
        b2.setMinimumHeight(44)
        b2.clicked.connect(self.openProject)
        row.addWidget(b2)
        row.addStretch(1)
        lay.addLayout(row)
        self._style()

    def _style(self):
        border = theme.ACCENT if self._hover else "#353D4B"
        bg = "#131A0E" if self._hover else theme.PANEL
        self.setStyleSheet(f"DropZone {{ border: 2px dashed {border}; border-radius: 18px; background: {bg}; }}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover = True
            self._style()

    def dragLeaveEvent(self, e):
        self._hover = False
        self._style()

    def dropEvent(self, e):
        self._hover = False
        self._style()
        urls = e.mimeData().urls()
        if urls:
            self.dropped.emit(urls[0].toLocalFile())


class TemplateCard(QToolButton):
    def __init__(self, t: templates.Template, parent=None):
        super().__init__(parent)
        self.t = t
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(tr("Open a video and analyze it with this template"))
        self.setStyleSheet(f"QToolButton {{ background: {theme.CARD_2}; border: 1px solid #222834; border-radius: 14px; }}"
                           f"QToolButton:hover {{ border-color: {t.color}; background: #151A21; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        ic = QLabel()
        ic.setFixedSize(56, 56)
        ic.setAlignment(Qt.AlignCenter)
        ic.setStyleSheet(f"background: {theme.RAISED}; border-radius: 12px; border: 0;")
        ic.setPixmap(icons.template_icon(t.icon or "M24 14v20M14 24h20", t.color, 44))
        lay.addWidget(ic)
        n = QLabel(t.label)
        n.setStyleSheet("font-size: 15px; font-weight: 600; border: 0; background: transparent;")
        lay.addWidget(n)
        d = QLabel(t.summary)
        d.setWordWrap(True)
        d.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px; border: 0; background: transparent;")
        lay.addWidget(d)
        lay.addStretch(1)
        for w in (ic, n, d):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)


class RecentItem(QToolButton):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(76)
        self.setToolTip(path)
        self.setStyleSheet(f"QToolButton {{ background: {theme.PANEL}; border: 1px solid #1E232C; border-radius: 12px; }}"
                           f"QToolButton:hover {{ border-color: {theme.BORDER}; background: #12161C; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 10, 12, 10)
        lay.setSpacing(14)
        th = QLabel()
        th.setFixedSize(96, 54)
        th.setStyleSheet("border-radius: 8px; border: 0;")
        th.setPixmap(thumbnail(path))
        lay.addWidget(th)
        col = QVBoxLayout()
        col.setSpacing(3)
        p = Path(path)
        name = QLabel(p.name)
        name.setStyleSheet("font-weight: 500; border: 0; background: transparent;")
        name.setText(name.fontMetrics().elidedText(p.name, Qt.ElideMiddle, 170))
        col.addWidget(name)
        try:
            import datetime
            ts = datetime.datetime.fromtimestamp(p.stat().st_mtime)
            when = ts.strftime("%d/%m/%Y %H:%M")
        except OSError:
            when = tr("not found")
        w = QLabel(when)
        w.setStyleSheet(f"color: {theme.FAINT}; font-size: 12px; border: 0; background: transparent;")
        col.addWidget(w)
        lay.addLayout(col, 1)
        ext = QLabel(p.suffix.lstrip(".").upper()[:5] or "—")
        ext.setFont(theme.mono(11))
        ext.setStyleSheet(f"background: {theme.RAISED}; color: {theme.MUTED}; border-radius: 6px; padding: 3px 8px; border: 0;")
        lay.addWidget(ext)
        for x in (th, name, w, ext):
            x.setAttribute(Qt.WA_TransparentForMouseEvents)


# The sidebar of the start page: (section, [(key, icon, title, what it is for, badge)])
NAV = [
    ("Analyze", [
        ("video", "video", "2D analysis", "One video: angles, trajectories, phases", ""),
        ("3d", "cube", "3D analysis", "Two or more videos, up to OpenSim", "BETA"),
        ("compare", "compare", "Comparative analysis", "Several patients or conditions side by side", "NEW"),
        ("patients", "user", "Patients", "Their records and analyses, to pick them again", "NEW"),
    ]),
    ("Open", [
        ("project", "folder", "Project or Kinovea file", "Continue a saved analysis", ""),
    ]),
    ("Customize", [
        ("templates", "template", "Templates", "Your own analysis, without programming", ""),
        ("plugins", "store", "Plugin store", "Install and remove analyses", ""),
    ]),
    ("Help", [
        ("shortcuts", "keyboard", "Keyboard shortcuts", "", ""),
        ("language", "globe", "Language", "", ""),
        ("about", "info", "About OpenKick", "", ""),
    ]),
]


class NavItem(QPushButton):
    """One entry of the sidebar: icon, title and, below it, what it is for."""

    def __init__(self, icon: str, title: str, hint: str = "", badge: str = "", parent=None):
        super().__init__(parent)
        self.title, self.hint = title, hint
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        h = 56 if hint else 38
        self.setStyleSheet(f"QPushButton {{ background: transparent; border: 1px solid transparent; border-radius: 10px;"
                           f" min-height: {h}px; max-height: {h}px; padding: 0; text-align: left; }}"
                           f"QPushButton:hover {{ background: {theme.RAISED}; }} QPushButton:pressed {{ background: {theme.ACTIVE}; }}"
                           f"QPushButton:focus {{ border-color: {theme.BORDER}; }}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(12)
        ic = QLabel()
        ic.setFixedSize(34, 34)
        ic.setAlignment(Qt.AlignCenter)
        if hint:
            ic.setStyleSheet(f"background: {theme.CARD}; border: 1px solid {theme.LINE}; border-radius: 9px;")
        ic.setPixmap(icons.pixmap(icon, theme.ACCENT if hint else theme.MUTED, 18))
        lay.addWidget(ic)
        self.text = QWidget()
        self.text.setStyleSheet("background: transparent;")
        col = QVBoxLayout(self.text)
        col.setSpacing(2)
        col.setContentsMargins(0, 0, 0, 0)
        col.addStretch(1)
        row = QHBoxLayout()
        row.setSpacing(6)
        t = QLabel(title)
        t.setStyleSheet(f"font-weight: {600 if hint else 400}; color: {theme.TEXT if hint else theme.TEXT_2};")
        row.addWidget(t)
        if badge:
            b = QLabel(badge)
            b.setFixedHeight(16)
            b.setStyleSheet(f"color: {theme.WARN_TEXT}; background: #2A2016; border-radius: 4px; padding: 0 5px; font-size: 9px; font-weight: 700;")
            row.addWidget(b)
        row.addStretch(1)
        col.addLayout(row)
        if hint:
            hl = QLabel(hint)
            hl.setStyleSheet(f"color: {theme.FAINT}; font-size: 11px;")
            col.addWidget(hl)
        col.addStretch(1)
        lay.addWidget(self.text, 1)
        for w in [ic, self.text] + self.text.findChildren(QLabel):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.set_compact(False)

    def set_compact(self, on: bool):
        """Icon only (the words go to the tooltip) for narrow windows."""
        self.text.setVisible(not on)
        self.setToolTip((self.title + (" · " + self.hint if self.hint else "")) if on else self.hint)


class SideBar(QFrame):
    """Everything the program can do, in one place, so nobody has to hunt through the menus."""
    picked = Signal(str)           # one of the NAV keys, or "language:<code>"
    WIDTH, COMPACT = 272, 68

    def __init__(self, parent=None):
        super().__init__(parent)
        from openkick.i18n import LANGUAGES, language
        self.setStyleSheet(f"SideBar {{ background: {theme.PANEL}; border-right: 1px solid {theme.LINE}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 18, 8, 14)
        lay.setSpacing(2)
        self.items: dict[str, NavItem] = {}
        self.heads: list[QLabel] = []
        for i, (section, entries) in enumerate(NAV):
            if i:
                lay.addSpacing(14)
            head = QLabel(tr(section).upper())
            head.setObjectName("SectionTitle")
            head.setContentsMargins(10, 0, 0, 4)
            lay.addWidget(head)
            self.heads.append(head)
            for key, icon, title, hint, badge in entries:
                item = NavItem(icon, tr(title), tr(hint) if hint else "", badge)
                if key == "language":
                    menu = QMenu(item)
                    for code, label in LANGUAGES.items():
                        a = menu.addAction(label)
                        a.setCheckable(True)
                        a.setChecked(code == language())
                        a.triggered.connect(lambda _=False, c=code: self.picked.emit(f"language:{c}"))
                    item.clicked.connect(lambda _=False, it=item, m=menu: m.exec(it.mapToGlobal(it.rect().topRight())))
                else:
                    item.clicked.connect(lambda _=False, k=key: self.picked.emit(k))
                lay.addWidget(item)
                self.items[key] = item
        lay.addStretch(1)
        self.compact = None
        self.set_compact(False)

    def set_compact(self, on: bool):
        if on == self.compact:
            return
        self.compact = on
        self.setFixedWidth(self.COMPACT if on else self.WIDTH)
        for h in self.heads:
            h.setVisible(not on)
        for item in self.items.values():
            item.set_compact(on)


class _Forward:
    """Lets a keyed signal be emitted like a plain one."""

    def __init__(self, signal, key: str):
        self.signal, self.key = signal, key

    def emit(self):
        self.signal.emit(self.key)


class WelcomePage(QWidget):
    openVideo = Signal()
    openProject = Signal()
    openPath = Signal(str)
    templatePicked = Signal(str)
    manageTemplates = Signal()
    navigate = Signal(str)         # sidebar entries the main window handles: "3d", "plugins", "shortcuts", ...

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        from PySide6.QtWidgets import QScrollArea
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        self.sidebar = SideBar()
        self.sidebar.picked.connect(self._picked)
        middle.addWidget(self.sidebar)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        middle.addWidget(scroll, 1)
        outer.addLayout(middle, 1)
        page = QWidget()
        page.setObjectName("Page")
        scroll.setWidget(page)
        body = QHBoxLayout(page)
        body.setContentsMargins(44, 44, 44, 24)
        body.setSpacing(40)

        left = QVBoxLayout()
        left.setSpacing(36)
        hero = QVBoxLayout()
        hero.setSpacing(14)
        h1 = QLabel(tr("Analyze the movement,\nnot the software."))
        h1.setFont(theme.display(44))
        hero.addWidget(h1)
        sub = QLabel(tr("Load a video: OpenKick detects the skeleton with MediaPipe, measures angles and "
                        "trajectories frame by frame, and you only fix what needs fixing."))
        sub.setWordWrap(True)
        sub.setMaximumWidth(640)
        sub.setStyleSheet(f"color: {theme.MUTED}; font-size: 16px; line-height: 150%;")
        hero.addWidget(sub)
        left.addLayout(hero)
        self.drop = DropZone()
        self.drop.openVideo.connect(self.openVideo)
        self.drop.openProject.connect(self.openProject)
        self.drop.dropped.connect(self.openPath)
        left.addWidget(self.drop)
        tbox = QVBoxLayout()
        tbox.setSpacing(14)
        head = QHBoxLayout()
        h2 = QLabel(tr("Start with an analysis template"))
        h2.setStyleSheet("font-size: 15px; font-weight: 600;")
        head.addWidget(h2)
        head.addStretch(1)
        more = QPushButton(tr("Create your own"))
        more.setProperty("ghost", True)
        more.clicked.connect(self.manageTemplates)
        head.addWidget(more)
        tbox.addLayout(head)
        self.tgrid = QGridLayout()
        self.tgrid.setSpacing(12)
        tbox.addLayout(self.tgrid)
        left.addLayout(tbox)
        left.addStretch(1)
        body.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(12)
        h2 = QLabel(tr("Recent"))
        h2.setStyleSheet("font-size: 15px; font-weight: 600;")
        right.addWidget(h2)
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(10)
        right.addLayout(self.recent_box)
        right.addStretch(1)
        rw = QWidget()
        rw.setFixedWidth(340)
        rw.setLayout(right)
        body.addWidget(rw)

        foot = QFrame()
        foot.setFixedHeight(44)
        foot.setStyleSheet(f"QFrame {{ border-top: 1px solid {theme.LINE_SOFT}; }} QLabel {{ color: {theme.DIM}; font-size: 12px; border: 0; }}")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(64, 0, 64, 0)
        fl.setSpacing(16)
        for t in (f"OpenKick {__version__}", "·", tr("Free software (GPLv3)"), "·", tr("Kinovea compatible (.kva)")):
            fl.addWidget(QLabel(t))
        fl.addStretch(1)
        fl.addWidget(QLabel(tr("Developed by {name}", name=AUTHOR)))
        fl.addWidget(QLabel("·"))
        fl.addWidget(QLabel(tr("Pose detection: MediaPipe")))
        outer.addWidget(foot)
        self.reload_templates()

    def resizeEvent(self, e):
        self.sidebar.set_compact(self.width() < 1280)      # narrow window: icons only, so the page keeps its room
        super().resizeEvent(e)

    def _picked(self, key: str):
        {"video": self.openVideo, "project": self.openProject, "templates": self.manageTemplates}.get(
            key, _Forward(self.navigate, key)).emit()

    def reload_templates(self):
        while self.tgrid.count():
            w = self.tgrid.takeAt(0).widget()
            if w:
                w.hide()
                w.deleteLater()
        ts = templates.list_templates()
        for i, t in enumerate(ts[:10]):
            card = TemplateCard(t)
            card.clicked.connect(lambda _=False, tid=t.id: self.templatePicked.emit(tid))
            self.tgrid.addWidget(card, i // 5, i % 5)

    def set_recent(self, paths: list[str]):
        while self.recent_box.count():
            w = self.recent_box.takeAt(0).widget()
            if w:
                w.hide()
                w.deleteLater()
        shown = 0
        for p in paths:
            if not Path(p).exists():
                continue
            item = RecentItem(p)
            item.clicked.connect(lambda _=False, p=p: self.openPath.emit(p))
            self.recent_box.addWidget(item)
            shown += 1
            if shown >= 6:
                break
        if not shown:
            lbl = QLabel(tr("Your projects will appear here."))
            lbl.setStyleSheet(f"color: {theme.FAINT};")
            self.recent_box.addWidget(lbl)
