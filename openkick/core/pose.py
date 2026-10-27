"""Automatic 2D pose detection with MediaPipe Pose Landmarker (BlazePose, 33 points)."""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from openkick.core.landmarks import COUNT
from openkick.core.video import VideoMeta, iter_frames

MODELS = {
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def data_dir() -> Path:
    """Per-user folder for models, plugins and templates."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        d = base / "OpenKick"
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "OpenKick"
    else:
        d = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "openkick"
    d.mkdir(parents=True, exist_ok=True)
    return d


BUNDLED_MODELS = Path(__file__).resolve().parent.parent / "assets" / "models"


def model_path(kind: str = "heavy") -> Path:
    """Model shipped with the installer if present, else the per-user download location."""
    bundled = BUNDLED_MODELS / f"pose_landmarker_{kind}.task"
    if bundled.exists():
        return bundled
    return data_dir() / "models" / f"pose_landmarker_{kind}.task"


def ensure_model(kind: str = "heavy", progress: Callable[[int, int], None] | None = None) -> Path:
    """Return the model path, downloading it the first time."""
    path = model_path(kind)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(MODELS[kind], timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 16):
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    tmp.replace(path)
    return path


@dataclass
class PoseResult:
    xy: np.ndarray          # (N, 33, 2) float32, pixels; NaN where not detected
    visibility: np.ndarray  # (N, 33) float32, 0..1
    z: np.ndarray           # (N, 33) float32, MediaPipe relative depth
    detected: int           # frames with a person


def _enhance(rgb: np.ndarray) -> np.ndarray:
    """Local contrast (CLAHE) + light sharpening: helps MediaPipe with dim or blurry footage."""
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return np.clip(cv2.filter2D(out, -1, kernel), 0, 255).astype(np.uint8)


class PoseDetector:
    """Runs MediaPipe over a whole video in VIDEO mode (temporal tracking between frames)."""

    def __init__(self, model: str | Path, min_confidence: float = 0.5, enhance: bool = True):
        self.model = str(model)
        self.min_confidence = min_confidence
        self.enhance = enhance

    def run(self, meta: VideoMeta,
            on_frame: Callable[[int, int, np.ndarray, np.ndarray | None], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> PoseResult | None:
        """Detect every frame. `on_frame(i, n, rgb, xy_or_None)` reports progress.
        Returns None if cancelled."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        n = meta.frame_count
        xy = np.full((n, COUNT, 2), np.nan, np.float32)
        vis = np.zeros((n, COUNT), np.float32)
        z = np.full((n, COUNT), np.nan, np.float32)
        detected = 0

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        last_ms = -1
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            for i, rgb in iter_frames(meta):
                if should_stop and should_stop():
                    return None
                h, w = rgb.shape[:2]
                img = _enhance(rgb) if self.enhance else rgb
                ms = int(round(meta.time_of(i) * 1000))
                if ms <= last_ms:            # MediaPipe needs strictly increasing timestamps
                    ms = last_ms + 1
                last_ms = ms
                res = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)), ms)
                frame_xy = None
                if res.pose_landmarks:
                    detected += 1
                    for j, lm in enumerate(res.pose_landmarks[0]):
                        xy[i, j] = (lm.x * w, lm.y * h)
                        vis[i, j] = lm.visibility if lm.visibility is not None else 1.0
                        z[i, j] = lm.z
                    frame_xy = xy[i]
                if on_frame:
                    on_frame(i, n, rgb, frame_xy)
        return PoseResult(xy=xy, visibility=vis, z=z, detected=detected)
