"""Automatic pose detection: background worker + progress dialog."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QVBoxLayout)

from openkick.core import landmarks as L
from openkick.core import templates
from openkick.core.pose import PoseDetector, PoseResult, ensure_model
from openkick.core.video import VideoMeta
from openkick.i18n import tr
from openkick.ui import icons, theme

PREVIEW_W, PREVIEW_H = 524, 295


class DetectionWorker(QThread):
    stage = Signal(str)                      # "model" | "detect"
    modelProgress = Signal(int, int)
    progress = Signal(int, int, QImage)
    done = Signal(object)                    # PoseResult | None (cancelled)
    failed = Signal(str)

    def __init__(self, meta: VideoMeta, model: str = "heavy", enhance: bool = True):
        super().__init__()
        self.meta, self.model, self.enhance = meta, model, enhance
        self._stop = False
        self._last_emit = 0.0

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.stage.emit("model")
            path = ensure_model(self.model, lambda d, t: self.modelProgress.emit(d, t))
            self.stage.emit("detect")
            det = PoseDetector(path, enhance=self.enhance)
            res = det.run(self.meta, self._on_frame, lambda: self._stop)
            self.done.emit(res)
        except Exception as e:  # noqa: BLE001 - reported to the user
            self.failed.emit(str(e))

    def _on_frame(self, i, n, rgb, xy):
        now = time.monotonic()
        if now - self._last_emit < 0.05 and i < n - 1:
            return
        self._last_emit = now
        self.progress.emit(i, n, _preview(rgb, xy))


def _preview(rgb: np.ndarray, xy: np.ndarray | None) -> QImage:
    h, w = rgb.shape[:2]
    src = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    out = QImage(PREVIEW_W, PREVIEW_H, QImage.Format_RGB32)
    out.fill(QColor(theme.STAGE))
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    s = min(PREVIEW_W / w, PREVIEW_H / h)
    ox, oy = (PREVIEW_W - w * s) / 2, (PREVIEW_H - h * s) / 2
    p.drawImage(QRectF(ox, oy, w * s, h * s), src)
    if xy is not None:
        colors = {"R": theme.RIGHT, "L": theme.LEFT, "C": theme.CENTER}
        for a, b in L.BONES:
            if np.isfinite(xy[[a, b]]).all():
                sa, sb = L.side(a), L.side(b)
                p.setPen(QPen(QColor(colors[sa if sa == sb else "C"]), 2.2, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(QPointF(float(ox + xy[a, 0] * s), float(oy + xy[a, 1] * s)),
                           QPointF(float(ox + xy[b, 0] * s), float(oy + xy[b, 1] * s)))
        p.setPen(Qt.NoPen)
        for j in L.JOINTS:
            if np.isfinite(xy[j]).all():
                p.setBrush(QColor(colors[L.side(j)]))
                p.drawEllipse(QPointF(float(ox + xy[j, 0] * s), float(oy + xy[j, 1] * s)), 3.0, 3.0)
    p.end()
    return out


class StepRow(QHBoxLayout):
    def __init__(self, text: str):
        super().__init__()
        self.setSpacing(12)
        self.mark = QLabel()
        self.mark.setFixedSize(22, 22)
        self.addWidget(self.mark)
        self.label = QLabel(text)
        self.addWidget(self.label, 1)
        self.meta = QLabel()
        self.meta.setFont(theme.mono(12))
        self.meta.setStyleSheet(f"color: {theme.FAINT};")
        self.addWidget(self.meta)
        self.set_state("todo")

    def set_state(self, state: str, meta: str = ""):
        if state == "done":
            self.mark.setStyleSheet("background: #1F3A1C; border-radius: 11px;")
            self.mark.setPixmap(icons.pixmap("check", "#86EFAC", 13, 3))
            self.mark.setAlignment(Qt.AlignCenter)
        elif state == "now":
            self.mark.setPixmap(QPixmap())
            self.mark.setStyleSheet(f"border: 2px solid {theme.ACCENT}; border-radius: 11px; background: qradialgradient("
                                    f"cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 {theme.ACCENT}, stop:0.36 {theme.ACCENT}, stop:0.42 transparent);")
        else:
            self.mark.setPixmap(QPixmap())
            self.mark.setStyleSheet(f"border: 2px solid #353D4B; border-radius: 11px;")
        self.label.setStyleSheet(f"color: {theme.FAINT if state == 'todo' else theme.TEXT};")
        self.meta.setText(meta)


class ProcessingDialog(QDialog):
    """Runs detection with live preview. Emits `finished_ok(result, template_id)`."""
    finished_ok = Signal(object, object)
    progressed = Signal(int)

    def __init__(self, meta: VideoMeta, template_id: str | None, model: str, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.setWindowTitle(tr("Detecting the pose"))
        self.setModal(True)
        self.setFixedWidth(600)
        self.setStyleSheet(f"QDialog {{ background: {theme.CARD_2}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 24)
        lay.setSpacing(20)
        title = QLabel(tr("Detecting the pose"))
        title.setFont(theme.display(24))
        lay.addWidget(title)
        from pathlib import Path
        sub = QLabel(f"{Path(meta.path).name} · {meta.width} × {meta.height} · "
                     + tr("{n} frames", n=meta.frame_count))
        sub.setStyleSheet(f"color: {theme.FAINT};")
        lay.addWidget(sub)
        self.preview = QLabel()
        self.preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview.setStyleSheet(f"background: {theme.STAGE}; border-radius: 12px; border: 1px solid #222834;")
        self.preview.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.preview, 0, Qt.AlignHCenter)
        row = QHBoxLayout()
        self.pct = QLabel("0 %")
        self.pct.setStyleSheet("font-weight: 500;")
        row.addWidget(self.pct)
        row.addStretch(1)
        self.eta = QLabel("")
        self.eta.setStyleSheet(f"color: {theme.FAINT};")
        row.addWidget(self.eta)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setFixedHeight(8)
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.steps = [StepRow(tr("Prepare the MediaPipe model")),
                      StepRow(tr("Read the video and the real time of each frame")),
                      StepRow(tr("Detect the skeleton (MediaPipe, {model} model)",
                                 model={"heavy": tr("accurate"), "full": tr("balanced"), "lite": tr("fast")}[model])),
                      StepRow("")]
        box = QVBoxLayout()
        box.setSpacing(12)
        for s in self.steps:
            box.addLayout(s)
        lay.addLayout(box)
        self.steps[1].set_state("done", tr("{n} frames", n=meta.frame_count))
        sep = QFrame()
        sep.setObjectName("Separator")
        lay.addWidget(sep)
        foot = QHBoxLayout()
        tl = QLabel(tr("Template"))
        tl.setStyleSheet(f"color: {theme.MUTED};")
        foot.addWidget(tl)
        self.tpl = QComboBox()
        for t in templates.list_templates():
            self.tpl.addItem(icons.icon("template", t.color, 16), t.label, t.id)
        i = self.tpl.findData(template_id or "blank")
        self.tpl.setCurrentIndex(max(0, i))
        self.tpl.currentIndexChanged.connect(self._tpl_text)
        foot.addWidget(self.tpl)
        foot.addStretch(1)
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self._cancel)
        foot.addWidget(self.cancel_btn)
        self.bg_btn = QPushButton(tr("Keep working in the background"))
        self.bg_btn.setProperty("primary", True)
        self.bg_btn.clicked.connect(self.hide)
        foot.addWidget(self.bg_btn)
        lay.addLayout(foot)
        self._tpl_text()
        self.t0 = time.monotonic()
        self.worker = DetectionWorker(meta, model)
        self.worker.stage.connect(self._stage)
        self.worker.modelProgress.connect(self._model_progress)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.result_ok = False

    def start(self):
        self.worker.start()
        self.show()

    def _tpl_text(self, *_):
        t = templates.find(self.tpl.currentData())
        if t and (t.angles or t.trajectories or t.phases):
            txt = tr("Apply the “{name}” template: {a} angles, {t} trajectories, {p} phases",
                     name=t.label, a=len(t.angles), t=len(t.trajectories), p=len(t.phases))
        else:
            txt = tr("No template: you add the measurements")
        self.steps[3].label.setText(txt)

    def _stage(self, s):
        if s == "model":
            self.steps[0].set_state("now")
        else:
            self.steps[0].set_state("done")
            self.steps[2].set_state("now")
            self.t0 = time.monotonic()

    def _model_progress(self, done, total):
        if total:
            self.steps[0].set_state("now", f"{done / total * 100:.0f} %")

    def _progress(self, i, n, img):
        self.preview.setPixmap(QPixmap.fromImage(img))
        frac = (i + 1) / max(n, 1)
        self.bar.setValue(int(frac * 1000))
        self.pct.setText(f"{frac * 100:.0f} %")
        self.steps[2].set_state("now", f"{i + 1} / {n}")
        el = time.monotonic() - self.t0
        if frac > 0.02:
            rest = el / frac - el
            self.eta.setText(tr("≈ {s} s left", s=max(1, round(rest))))
        self.progressed.emit(int(frac * 100))

    def _done(self, res: PoseResult | None):
        self.worker.wait()
        if res is None:
            self.reject()
            return
        self.steps[2].set_state("done", f"{res.detected} / {self.meta.frame_count}")
        self.steps[3].set_state("done")
        self.result_ok = True
        self.finished_ok.emit(res, self.tpl.currentData())
        self.accept()

    def _failed(self, msg):
        from PySide6.QtWidgets import QMessageBox
        self.worker.wait()
        QMessageBox.critical(self, tr("Detection failed"), msg)
        self.reject()

    def _cancel(self):
        self.worker.stop()
        self.cancel_btn.setEnabled(False)

    def closeEvent(self, e):
        if self.worker.isRunning():
            self.hide()
            e.ignore()
        else:
            super().closeEvent(e)

    def reject(self):
        if self.worker.isRunning() and not self.worker._stop:
            self.hide()               # Esc = keep going in the background
            return
        super().reject()
