"""Render the OpenKick logo to openkick/assets/openkick.ico (PNG-compressed, multi-size) and .png."""
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

GLYPH = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#0F1115" '
         b'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="4.5" r="2"/>'
         b'<path d="M9 8v6l-4 6M9 14l6-1 4-5"/></svg>')


def render(px: int) -> bytes:
    img = QImage(px, px, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#C6F24E"))
    p.drawRoundedRect(QRectF(0, 0, px, px), px * 0.26, px * 0.26)
    m = px * 0.17
    QSvgRenderer(QByteArray(GLYPH)).render(p, QRectF(m, m, px - 2 * m, px - 2 * m))
    p.end()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def main():
    QGuiApplication(sys.argv[:1])
    sizes = [16, 24, 32, 48, 64, 128, 256]
    pngs = [render(s) for s in sizes]
    out = Path(__file__).resolve().parents[1] / "openkick" / "assets"
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, data = b"", b""
    for s, png in zip(sizes, pngs):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    (out / "openkick.ico").write_bytes(header + entries + data)
    (out / "openkick.png").write_bytes(render(512))
    print("icon written")


if __name__ == "__main__":
    main()
