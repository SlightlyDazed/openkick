# PyInstaller spec for OpenKick (Windows / Linux / macOS).
#   pyinstaller packaging/openkick.spec --noconfirm
# Put pose_landmarker_heavy.task in packaging/models/ to ship the model (works offline).
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
PKG = ROOT / "openkick"

datas = [
    (str(PKG / "assets"), "openkick/assets"),
    (str(PKG / "templates"), "openkick/templates"),
    (str(PKG / "locale"), "openkick/locale"),
]
models = ROOT / "packaging" / "models"
if models.exists():
    datas.append((str(models), "openkick/assets/models"))
datas += collect_data_files("mediapipe", includes=["**/*.binarypb", "**/*.tflite", "**/*.txt", "**/*.pbtxt"])
binaries = collect_dynamic_libs("mediapipe")
# 3D: RTMPose through rtmlib and onnxruntime. Their models are not shipped (their training data is for
# non-commercial use): they are downloaded from OpenMMLab the first time (see openkick/core/models.py).
# OpenSim and Pose2Sim are not included: the OpenSim step is switched off in this build.
try:
    import onnxruntime  # noqa: F401
    binaries += collect_dynamic_libs("onnxruntime")
    three_d = ["rtmlib", "onnxruntime", "openkick.multiview.detect", "openkick.multiview.build", "openkick.ui.multiview"]
except ImportError:
    three_d = []

hiddenimports = [
    "mediapipe.tasks.python", "mediapipe.tasks.python.vision", "mediapipe.tasks.python.core",
    "openkick.plugins.builtin.rom_report", "openkick.plugins.builtin.velocities",
    "openkick.plugins.builtin.opensim_trc", "openkick.selftest", "scipy.signal", "PySide6.QtSvg",
] + three_d

excludes = ["tkinter", "jax", "jaxlib", "IPython", "pandas", "PyQt5", "PyQt6",
            "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
            "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization"]

a = Analysis([str(ROOT / "packaging" / "openkick_launcher.py")], pathex=[str(ROOT)], binaries=binaries,
             datas=datas, hiddenimports=hiddenimports, excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="OpenKick", console=False,
          icon=str(PKG / "assets" / "openkick.ico"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="OpenKick", upx=False)
