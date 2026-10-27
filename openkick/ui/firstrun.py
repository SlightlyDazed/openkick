"""The window of the first start: what OpenKick downloads, why the first start is slow, and what
it is not (a medical device). Shown once (QSettings "firstrun/done"); from the Help menu again."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from openkick.i18n import tr
from openkick.ui import theme

KEY = "firstrun/done"


def needed() -> bool:
    """Whether the window should open by itself at this start."""
    return not QSettings().value(KEY, False, type=bool)


class _Download(QThread):
    progress = Signal(str, int, int, int, int)
    done = Signal(bool, str)

    def __init__(self, keys: list):
        super().__init__()
        self.keys, self._stop = keys, False

    def stop(self):
        self._stop = True

    def run(self):
        from openkick.core import models
        try:
            for key in self.keys:
                if not models.download(key, lambda f, n, d, t, key=key: self.progress.emit(key, f, n, d, t),
                                       lambda: self._stop):
                    self.done.emit(False, "")
                    return
            self.done.emit(True, "")
        except Exception as e:  # noqa: BLE001 - no internet, a proxy...: told to the user
            self.done.emit(False, str(e))


class FirstRunDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from openkick.core import models
        self.setWindowTitle(tr("Welcome to OpenKick"))
        self.setFixedWidth(600)
        self.setStyleSheet(f"QDialog {{ background: {theme.CARD_2}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 26, 28, 22)
        lay.setSpacing(12)
        title = QLabel(tr("Welcome to OpenKick"))
        title.setFont(theme.display(24))
        lay.addWidget(title)

        def para(text, color=theme.TEXT_2):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {color};")
            lay.addWidget(lbl)
            return lbl
        para(tr("The first start is slower: Windows checks the program's files with its antivirus. "
                "Next time it opens faster."))
        para(tr("To find the body in the videos, OpenKick uses detection models. They are downloaded once "
                "and then everything works without internet:"))
        self.items = models.components()
        self.rows = {}
        for c in self.items:
            state = tr("already on this computer") if c["ready"] else tr("about {mb} MB to download", mb=c["mb"])
            self.rows[c["key"]] = para(f"•  {tr(c['title'])} — {state}", theme.TEXT if not c["ready"] else theme.MUTED)
        if any(c["key"] == "3d" for c in self.items):
            para(tr("The 3D models come from OpenMMLab (RTMPose). They were trained with datasets that are "
                    "for non-commercial use only; OpenKick downloads them from their official site and does "
                    "not include them."), theme.FAINT)
        para(tr("OpenKick is not a medical device. Its measurements help the professional's judgement and "
                "do not replace it."), theme.FAINT)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        lay.addWidget(self.bar)
        self.status = para("", theme.MUTED)
        self.status.setVisible(False)
        row = QHBoxLayout()
        row.addStretch(1)
        self.later = QPushButton(tr("Later (when first needed)"))
        self.later.clicked.connect(self._close)
        row.addWidget(self.later)
        self.get = QPushButton(tr("Download now"))
        self.get.setProperty("primary", True)
        self.get.clicked.connect(self._download)
        row.addWidget(self.get)
        lay.addLayout(row)
        self.worker = None
        todo = [c for c in self.items if not c["ready"]]
        if not todo:
            self.get.setVisible(False)
            self.later.setText(tr("Start"))
            self.later.setProperty("primary", True)

    def _download(self):
        keys = [c["key"] for c in self.items if not c["ready"]]
        self.get.setEnabled(False)
        self.later.setText(tr("Cancel"))
        self.bar.setVisible(True)
        self.status.setVisible(True)
        self.worker = _Download(keys)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.start()

    def _progress(self, key, f, n, done, total):
        name = next(tr(c["title"]) for c in self.items if c["key"] == key)
        part = f" ({f + 1}/{n})" if n > 1 else ""
        self.status.setText(f"{name}{part}: {done / 1e6:.0f} / {total / 1e6:.0f} MB" if total else name)
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(done)

    def _done(self, ok: bool, error: str):
        w, self.worker = self.worker, None
        if w:
            w.wait()
        if ok:
            QSettings().setValue(KEY, True)
            self.status.setText(tr("Done. OpenKick can now analyse videos without internet."))
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
            for c in self.items:
                self.rows[c["key"]].setStyleSheet(f"color: {theme.MUTED};")
            self.later.setText(tr("Start"))
            self.later.setProperty("primary", True)
            self.later.style().polish(self.later)
            self.get.setVisible(False)
            return
        self.get.setEnabled(True)
        self.later.setText(tr("Later (when first needed)"))
        self.status.setText(tr("Could not download: {why}. It will be tried again the first time it is needed.",
                               why=error) if error else tr("Download cancelled."))

    def _close(self):
        if self.worker:
            self.worker.stop()
            return
        QSettings().setValue(KEY, True)
        self.accept()

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        QSettings().setValue(KEY, True)
        super().closeEvent(e)
