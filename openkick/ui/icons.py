"""Line icons drawn from SVG path data (24x24 grid, 1.8 px strokes)."""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from openkick.ui import theme

PATHS = {
    "select": "M5 3l14 7-6 2-2 6z",
    "angle": "M4 20h16M4 20L15 5M10 20a6 6 0 0 0-2.4-4.8",
    "track": "M3 19c4-2 5-8 9-9s6-4 9-6M3 19h.01",
    "line": "M3 17L17 3l4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2",
    "text": "M5 6V4h14v2M12 4v16M9 20h6",
    "phase": "M5 21V4M5 4h11l-2 4 2 4H5",
    "calib": "M3 12h18M3 8v8M21 8v8M8 10v4M13 10v4M17 10v4",
    "zoom": "M11 18a7 7 0 1 0 0-14a7 7 0 0 0 0 14M21 21l-5-5",
    "fit": "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
    "undo": "M9 14L4 9l5-5M4 9h11a5 5 0 0 1 0 10h-3",
    "redo": "M15 14l5-5-5-5M20 9H9a5 5 0 0 0 0 10h3",
    "detect": "M4 20L15 9M13 3l1.2 2.3L16.5 6.5l-2.3 1.2L13 10l-1.2-2.3L9.5 6.5l2.3-1.2zM19 11l.8 1.7 1.7.8-1.7.8L19 16l-.8-1.7-1.7-.8 1.7-.8z",
    "export": "M12 3v12M7 10l5 5 5-5M4 21h16",
    "upload": "M12 16V4M7 9l5-5 5 5M4 20h16",
    "chevron_down": "M6 9l6 6 6-6",
    "chevron_left": "M15 6l-6 6 6 6",
    "start": "M18 5v14L9 12zM6 5v14",
    "prev": "M15 6l-6 6 6 6",
    "next": "M9 6l6 6-6 6",
    "end": "M6 5v14l9-7zM18 5v14",
    "plus": "M12 5v14M5 12h14",
    "user": "M12 11a4 4 0 1 0 0-8a4 4 0 0 0 0 8M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1",
    "bike": "M6 18a3.5 3.5 0 1 0 0-7a3.5 3.5 0 0 0 0 7M18 18a3.5 3.5 0 1 0 0-7a3.5 3.5 0 0 0 0 7M6 14.5l4-7h5l3 7M10 7.5h4M9 14.5h6l-3-5M14 4h2",
    "walk": "M13 5a2 2 0 1 0 0-4a2 2 0 0 0 0 4M10 22l2-7 3 3v4M8 10l3-3 3 1 2 3h3M12 15l-1-5",
    "compare": "M4 4h7v16H4zM13 4h7v16h-7zM7 9h1M7 13h1M16 11h1M16 15h1",
    "store": "M4 9l1.5-5h13L20 9M4 9h16v11H4zM4 9a2.7 2.7 0 0 0 5.3 0a2.7 2.7 0 0 0 5.4 0a2.7 2.7 0 0 0 5.3 0M10 20v-5h4v5",
    "link": "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    "template": "M4 5h6v6H4zM14 5h6v6h-6zM4 15h6v6H4zM14 18h6M17 15v6",
    "lock": "M6 11h12v10H6zM8 11V7a4 4 0 0 1 8 0v4",
    "eye": "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6a3 3 0 0 0 0 6",
    "eye_off": "M3 3l18 18M10.6 5.1A9.8 9.8 0 0 1 12 5c6 0 10 7 10 7a17 17 0 0 1-3.2 3.9M6.6 6.6C3.9 8.3 2 12 2 12s4 7 10 7a9.6 9.6 0 0 0 5.4-1.6M9.9 9.9a3 3 0 0 0 4.2 4.2",
    "trash": "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
    "globe": "M12 21a9 9 0 1 0 0-18a9 9 0 0 0 0 18M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18",
    "check": "M5 12l5 5 9-10",
    "close": "M6 6l12 12M18 6L6 18",
    "folder": "M3 6h6l2 2h10v11H3z",
    "save": "M5 3h11l3 3v15H5zM8 3v5h7V3M8 14h8v7H8",
    "chart": "M4 20V4M4 20h16M8 16l4-5 3 3 5-7",
    "table": "M4 5h16v14H4zM4 10h16M10 5v14",
    "pdf": "M6 3h9l4 4v14H6zM14 3v5h5M9 13h6M9 17h6",
    "video": "M4 5h12v14H4zM16 10l5-3v10l-5-3",
    "plugin": "M4 20V4M4 20h16M8 16l4-5 3 3 5-7",
    "speed": "M3 17l6-6 4 4 8-8M15 7h6v6",
    "people": "M9 11a3 3 0 1 0 0-6a3 3 0 0 0 0 6M3 20a6 6 0 0 1 12 0M17 11a3 3 0 1 0 0-6M21 20a6 6 0 0 0-4-5.6",
    "cube": "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
    "info": "M12 21a9 9 0 1 0 0-18a9 9 0 0 0 0 18M12 11v5M12 8h.01",
    "keyboard": "M3 6h18v12H3zM7 10h.01M11 10h.01M15 10h.01M7 14h10",
    "wand": "M4 20L15 9M13 3l1.2 2.3L16.5 6.5l-2.3 1.2L13 10l-1.2-2.3L9.5 6.5l2.3-1.2z",
    "smooth": "M3 16c3 0 3-8 6-8s3 8 6 8 3-8 6-8",
    "home": "M3 11l9-7 9 7M5 10v10h14V10",
    "flag": "M5 21V4M5 4h11l-2 4 2 4H5",
    "note": "M5 6V4h14v2M12 4v16M9 20h6",
    "dots": "M5 12h.01M12 12h.01M19 12h.01",
    "code": "M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16",
    "target": "M12 18a6 6 0 1 0 0-12a6 6 0 0 0 0 12M12 2v5M12 17v5M2 12h5M17 12h5",
    "com": "M12 21a9 9 0 1 0 0-18a9 9 0 0 0 0 18M12 3v18M3 12h18",
}

PLAY = "M8 5l12 7-12 7z"
PAUSE = "M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"


def _svg(path: str, color: str, stroke: float = 1.8, fill: bool = False) -> bytes:
    if fill:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
                f'<path d="{path}" fill="{color}"/></svg>').encode()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>').encode()


@lru_cache(maxsize=512)
def pixmap(name_or_path: str, color: str = theme.MUTED, size: int = 20, stroke: float = 1.8,
           fill: bool = False, ratio: float = 2.0) -> QPixmap:
    path = PATHS.get(name_or_path, name_or_path)
    renderer = QSvgRenderer(QByteArray(_svg(path, color, stroke, fill)))
    px = int(size * ratio)
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    renderer.render(p, QRectF(0, 0, px, px))
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(ratio)
    return pm


def icon(name: str, color: str = theme.MUTED, size: int = 20, checked_color: str | None = None,
         stroke: float = 1.8, fill: bool = False) -> QIcon:
    ic = QIcon()
    ic.addPixmap(pixmap(name, color, size, stroke, fill), QIcon.Normal, QIcon.Off)
    ic.addPixmap(pixmap(name, theme.DIM, size, stroke, fill), QIcon.Disabled, QIcon.Off)
    if checked_color:
        ic.addPixmap(pixmap(name, checked_color, size, stroke, fill), QIcon.Normal, QIcon.On)
    return ic


def logo(size: int = 28) -> QPixmap:
    """Lime rounded square with the kicking-figure glyph."""
    ratio = 2.0
    px = int(size * ratio)
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(theme.qcolor(theme.ACCENT))
    p.drawRoundedRect(QRectF(0, 0, px, px), px * 0.28, px * 0.28)
    glyph = QSvgRenderer(QByteArray(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#0F1115" '
        b'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="4.5" r="2"/>'
        b'<path d="M9 8v6l-4 6M9 14l6-1 4-5"/></svg>'))
    m = px * 0.18
    glyph.render(p, QRectF(m, m, px - 2 * m, px - 2 * m))
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(ratio)
    return pm


def template_icon(path: str, color: str, size: int = 44) -> QPixmap:
    ratio = 2.0
    px = int(size * ratio)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none" stroke="{color}" '
           f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>').encode()
    img = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(svg)).render(p, QRectF(0, 0, px, px))
    p.end()
    pm = QPixmap.fromImage(img)
    pm.setDevicePixelRatio(ratio)
    return pm


ICON_SIZE = QSize(20, 20)
