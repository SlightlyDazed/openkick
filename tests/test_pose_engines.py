"""Choosing the 2D detector: RTMPose's points in MediaPipe's layout, the choice kept in the project,
and the detection window offering every detector."""
import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def test_rtmpose_points_go_where_mediapipe_has_them():
    from openkick.core import landmarks as L
    from openkick.core.pose import from_halpe26
    from openkick.multiview.geometry import IDX
    xy26 = np.arange(2 * 26 * 2, dtype=np.float32).reshape(2, 26, 2)
    score = np.full((2, 26), 0.8, np.float32)
    score[1, IDX["left_knee"]] = 0.2                       # far away: kept, but flagged to check
    score[1, IDX["right_knee"]] = 1.3                      # scores above 1 are read as 1
    xy26[0, IDX["right_heel"]] = np.nan                    # not found in this frame
    xy, vis = from_halpe26(xy26, score)
    assert xy.shape == (2, L.COUNT, 2)
    assert np.array_equal(xy[:, L.INDEX["left_knee"]], xy26[:, IDX["left_knee"]])
    assert np.array_equal(xy[:, L.INDEX["right_foot_index"]], xy26[:, IDX["right_toe"]])     # the big toe
    for name in ("left_thumb", "right_pinky", "mouth_left", "left_eye_inner"):             # RTMPose has none
        assert np.isnan(xy[:, L.INDEX[name]]).all() and (vis[:, L.INDEX[name]] == 0).all()
    assert vis[1, L.INDEX["left_knee"]] == pytest.approx(0.2) and np.isfinite(xy[1, L.INDEX["left_knee"]]).all()
    assert vis[1, L.INDEX["right_knee"]] == 1.0
    assert np.isnan(xy[0, L.INDEX["right_heel"]]).all() and vis[0, L.INDEX["right_heel"]] == 0


def test_the_detector_is_saved_with_the_pose():
    from openkick.core.project import Pose
    p = Pose(3)
    p.detector = "rtmpose-balanced"
    assert Pose.from_dict(p.to_dict()).detector == "rtmpose-balanced"
    old = p.to_dict()
    del old["detector"]                                    # projects from before the choice
    assert Pose.from_dict(old).detector == ""


def test_the_detection_window_offers_every_detector(meta, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication
    from openkick.core.pose import ENGINES
    from openkick.ui import processing
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(processing, "rtmpose_available", lambda: False)
    dlg = processing.ProcessingDialog(meta, "gait", "heavy")
    keys = [dlg.engine.itemData(i) for i in range(dlg.engine.count())]
    assert keys == list(ENGINES)
    assert dlg.engine.currentData() == "heavy"
    items = dlg.engine.model()
    for i, k in enumerate(keys):                           # without rtmlib RTMPose cannot be picked
        assert items.item(i).isEnabled() == (not k.startswith("rtmpose-"))
    assert "MB" in dlg.engine.itemText(0)
    assert dlg.engine_hint.text()


def test_picking_another_detector_starts_again_with_it(meta, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication
    from openkick.ui import processing
    QApplication.instance() or QApplication([])
    started = []

    class Worker(processing.DetectionWorker):              # records what would run, runs nothing
        def start(self):
            started.append(self.model)

        def wait(self, *a):
            return True
    monkeypatch.setattr(processing, "DetectionWorker", Worker)
    monkeypatch.setattr(processing, "rtmpose_available", lambda: True)
    dlg = processing.ProcessingDialog(meta, "gait", "heavy")
    changed = []
    dlg.modelChanged.connect(changed.append)
    dlg.start()
    dlg.engine.setCurrentIndex(dlg.engine.findData("rtmpose-balanced"))
    assert started == ["heavy", "rtmpose-balanced"]
    assert changed == ["rtmpose-balanced"]
    assert dlg.worker.model == "rtmpose-balanced"
    assert "RTMPose" in dlg.steps[2].label.text()
    dlg.hide()
