"""Automatic 2D pose detection: MediaPipe Pose Landmarker (BlazePose, 33 points) or RTMPose (26
points, body and feet, through rtmlib), whose points are placed in MediaPipe's 33."""

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

# The detectors to choose from, lightest first: key -> (label, download in MB, what it is good for).
# Against the OpenCap lab (docs/VALIDATION.md, 2D, one side camera): in gait RTMPose balanced and
# performance put the knee 4.4–4.6° off against MediaPipe accurate's 9.3°; in squats and jumps
# MediaPipe has the better hip; RTMPose performance the better ankle everywhere. CPU speed on a
# 12-thread machine: RTMPose lightweight 48 frames/s, balanced 26, performance 8, MediaPipe accurate 25.
ENGINES = {
    "lite": ("MediaPipe · fast", 6, "The lightest, for slow computers. Less precise."),
    "full": ("MediaPipe · balanced", 9, "Between fast and accurate."),
    "heavy": ("MediaPipe · accurate", 31, "The best hip in squats and jumps."),
    "rtmpose-lightweight": ("RTMPose · fast", 43, "Twice as fast as MediaPipe accurate, and better in gait."),
    "rtmpose-balanced": ("RTMPose · balanced", 157,
                         "The best in gait (knee about 4° off against 9° with MediaPipe), as fast as MediaPipe accurate."),
    "rtmpose-performance": ("RTMPose · accurate", 596, "The best ankle. About three times slower: for powerful computers."),
}
RTMPOSE_NOTE = "RTMPose does not find the hands, the mouth or the corners of the eyes."


def is_rtmpose(model: str) -> bool:
    return model.startswith("rtmpose-")


def rtmpose_available() -> bool:
    """RTMPose needs rtmlib and onnxruntime (the 3D extras)."""
    try:
        import onnxruntime  # noqa: F401
        import rtmlib  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


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
    detector: str = ""      # the ENGINES key that found it


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


# RTMPose's 26 points (HALPE-26) placed in MediaPipe's 33; the hands, the mouth and the corners of
# the eyes stay empty, and the toes are the big toes.
FROM_HALPE = {"nose": "nose", "left_eye": "left_eye", "right_eye": "right_eye", "left_ear": "left_ear",
              "right_ear": "right_ear", "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
              "left_elbow": "left_elbow", "right_elbow": "right_elbow", "left_wrist": "left_wrist",
              "right_wrist": "right_wrist", "left_hip": "left_hip", "right_hip": "right_hip",
              "left_knee": "left_knee", "right_knee": "right_knee", "left_ankle": "left_ankle",
              "right_ankle": "right_ankle", "left_heel": "left_heel", "right_heel": "right_heel",
              "left_foot_index": "left_toe", "right_foot_index": "right_toe"}


def from_halpe26(xy26: np.ndarray, score26: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(N, 26, 2) RTMPose points (and their (N, 26) scores) -> (N, 33, 2) points and (N, 33) visibility.
    Every point is kept, as MediaPipe does, and its score is its visibility: RTMPose gives 0.7–0.9
    to a person seen well and 0.2–0.4 to one far away, so the app's 0.5 for "check this point" fits."""
    from openkick.core import landmarks as L
    from openkick.multiview.geometry import IDX
    xy26 = np.asarray(xy26, np.float32)
    n = len(xy26)
    xy = np.full((n, COUNT, 2), np.nan, np.float32)
    vis = np.zeros((n, COUNT), np.float32)
    for mp_name, halpe in FROM_HALPE.items():
        j, k = L.INDEX[mp_name], IDX[halpe]
        xy[:, j] = xy26[:, k]
        if score26 is not None:
            vis[:, j] = np.clip(np.asarray(score26, np.float32)[:, k], 0, 1)
    vis[~np.isfinite(xy[..., 0])] = 0
    return xy, vis


class RTMPoseDetector:
    """Runs RTMPose (body and feet) over a whole video, the way the 3D tool does (and shares its cache)."""

    def __init__(self, mode: str = "balanced"):
        self.mode = mode

    def run(self, meta: VideoMeta,
            on_frame: Callable[[int, int, np.ndarray, np.ndarray | None], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> PoseResult | None:
        from openkick.multiview.detect import detect

        def progress(i, n, rgb, xy26):
            if on_frame:
                found = np.isfinite(xy26).all()
                on_frame(i, n, rgb, from_halpe26(xy26[None])[0][0] if found else None)
        out = detect(meta, self.mode, progress, should_stop)
        if out is None:
            return None
        _, xy26, score = out
        xy, vis = from_halpe26(xy26, score)
        detected = int(np.isfinite(xy[..., 0]).any(axis=1).sum())
        return PoseResult(xy=xy, visibility=vis, z=np.full(vis.shape, np.nan, np.float32), detected=detected,
                          detector=f"rtmpose-{self.mode}")
