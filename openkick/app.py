"""Application entry point: `openkick` or `python -m openkick [file]`."""

from __future__ import annotations

import sys


def log_error(text: str):
    """Add an error, with the time, to <data dir>/openkick.log."""
    import datetime

    from openkick.core.pose import data_dir
    try:
        with open(data_dir() / "openkick.log", "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n{text}")
    except OSError:
        pass


def _install_crash_log():
    """Write unexpected errors to <data dir>/openkick.log and tell the user (a windowed .exe
    has no console, so without this a crash would be silent)."""
    import traceback

    from openkick.core.pose import data_dir
    log = data_dir() / "openkick.log"

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        log_error(text)
        sys.__stderr__ and sys.__stderr__.write(text)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance():
                QMessageBox.critical(None, "OpenKick", f"{value}\n\n{log}")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


def main(argv: list[str] | None = None):
    argv = sys.argv if argv is None else argv
    _install_crash_log()
    from PySide6.QtCore import QCoreApplication, QSettings, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setOrganizationName("OpenKick")
    QCoreApplication.setApplicationName("OpenKick")
    from openkick import __version__
    QCoreApplication.setApplicationVersion(__version__)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setStyle("Fusion")
    from pathlib import Path

    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "openkick.ico")))

    from openkick import i18n
    lang = QSettings().value("language", "") or i18n.system_language()   # first start: the computer's language
    if lang not in i18n.LANGUAGES:
        lang = "es"
    i18n.set_language(lang)
    # Qt's own texts (file dialogs, standard buttons) in the same language
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
    QLocale.setDefault(QLocale(lang))
    qt_tr = QTranslator(app)
    if qt_tr.load(f"qtbase_{lang}", QLibraryInfo.path(QLibraryInfo.TranslationsPath)):
        app.installTranslator(qt_tr)

    from openkick.ui import theme
    theme.load_fonts()
    app.setPalette(theme.palette())
    app.setFont(theme.font(13))
    app.setStyleSheet(theme.STYLE)

    if "--selftest" in argv:
        i = argv.index("--selftest")
        from openkick.selftest import run
        if len(argv) < i + 3:
            print("usage: openkick --selftest <video> <output folder>")
            return 2
        return run(argv[i + 1], argv[i + 2])

    from openkick.ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    from openkick.ui import firstrun
    if firstrun.needed():                # what is downloaded and why the first start is slow
        QTimer.singleShot(300, win.show_models)
    files = [a for a in argv[1:] if not a.startswith("-")]
    if files:
        QTimer.singleShot(0, lambda: win.open_path(files[0]))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
