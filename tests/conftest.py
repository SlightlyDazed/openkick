import os
from pathlib import Path

import numpy as np
import pytest

# the sample videos live outside the repository: <OpenKick>/datos/muestras/biomeca (or OPENKICK_SAMPLES)
_HOME = Path(__file__).resolve().parents[2]
SAMPLES = Path(os.environ.get("OPENKICK_SAMPLES", next(
    (p for p in (_HOME / "datos" / "muestras" / "biomeca", _HOME / "biomeca") if p.exists()), _HOME / "datos" / "muestras" / "biomeca")))
VIDEO = SAMPLES / "document_5019272118395734678.mp4"
KINOVEA_KVA = SAMPLES / "document_5019272118395734678_conangulo_contrayectoria.kva"


@pytest.fixture
def meta():
    if not VIDEO.exists():
        pytest.skip("sample video not available")
    from openkick.core.video import probe
    return probe(VIDEO)


@pytest.fixture
def project(meta):
    """Project with a synthetic but plausible pose (no MediaPipe needed)."""
    from openkick.core import landmarks as L
    from openkick.core.project import SRC_AUTO, Project
    p = Project(meta)
    n = meta.frame_count
    rng = np.random.default_rng(0)
    base = rng.uniform([300, 100], [550, 430], size=(L.COUNT, 2)).astype(np.float32)
    t = np.arange(n)
    p.pose.xy[:] = base[None] + np.stack([np.sin(t / 7.0) * 10, np.cos(t / 9.0) * 8], axis=-1)[:, None, :]
    p.pose.raw_xy[:] = p.pose.xy
    p.pose.visibility[:] = 0.95
    p.pose.source[:] = SRC_AUTO
    return p


@pytest.fixture(autouse=True)
def no_blocking_dialogs(monkeypatch):
    """A message box nobody answers would hang the test run for ever: make it fail instead.
    Tests that expect a dialog replace these themselves."""
    try:
        from PySide6.QtWidgets import QMessageBox
    except ImportError:
        return

    def unexpected(kind):
        def raiser(*args, **kwargs):
            text = next((a for a in args if isinstance(a, str) and len(a) > 12), "")
            raise AssertionError(f"unexpected {kind} dialog: {text}")
        return staticmethod(raiser)
    for kind in ("question", "information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, unexpected(kind))


@pytest.fixture(autouse=True)
def own_patients(tmp_path, monkeypatch):
    """Each test has its own list of patients: never the user's."""
    from openkick.core import patients
    monkeypatch.setattr(patients, "_file", lambda: tmp_path / "patients.json")
