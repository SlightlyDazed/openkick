"""Pose engines under test. Each returns frame times and named 2D points in pixels:
right/left _shoulder, _hip, _knee, _ankle, _heel, _toe (NaN where not detected)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

NAMES = ["shoulder", "hip", "knee", "ankle", "heel", "toe"]


def run(engine: str, video: Path):
    if engine.startswith("mediapipe"):
        return _mediapipe(engine, video)
    if engine.startswith("rtmpose"):
        return _rtmpose(engine, video)
    raise ValueError(f"unknown engine {engine}")


def _times(meta) -> np.ndarray:
    return np.array([meta.time_of(i) for i in range(meta.frame_count)])


def _mediapipe(engine: str, video: Path):
    """OpenKick's current detector, exactly as the app runs it."""
    from openkick.core import landmarks as L
    from openkick.core.pose import PoseDetector, ensure_model
    from openkick.core.video import probe
    kind = engine.split("-")[1] if "-" in engine else "heavy"
    meta = probe(video)
    res = PoseDetector(ensure_model(kind)).run(meta)
    src = {"shoulder": "shoulder", "hip": "hip", "knee": "knee", "ankle": "ankle", "heel": "heel",
           "toe": "foot_index"}
    points = {f"{side}_{n}": res.xy[:, L.INDEX[f"{side}_{src[n]}"], :].astype(float)
              for side in ("right", "left") for n in NAMES}
    return _times(meta), points


# HALPE-26 (body + feet) keypoint order used by RTMPose "BodyWithFeet"
HALPE = {"left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12, "left_knee": 13,
         "right_knee": 14, "left_ankle": 15, "right_ankle": 16, "left_toe": 20, "right_toe": 21,
         "left_heel": 24, "right_heel": 25}


DET_EVERY = 10   # frames between person detections; in between, the box follows the last pose
_MODELS: dict = {}


def _rtmpose(engine: str, video: Path, min_score: float = 0.3):
    """RTMPose (HALPE-26) through rtmlib + ONNX Runtime, as an app would run it: the person
    detector runs every DET_EVERY frames and the box is carried by the previous pose.
    Keypoints are the raw per-frame output (no temporal smoothing)."""
    import cv2
    from rtmlib import BodyWithFeet, PoseTracker

    from openkick.core.video import iter_frames, probe
    mode = engine.split("-")[1] if "-" in engine else "balanced"
    model = _MODELS.get(mode)
    if model is None:      # loading the ONNX models takes seconds: keep them between videos
        model = _MODELS[mode] = PoseTracker(BodyWithFeet, det_frequency=DET_EVERY, tracking=False, mode=mode,
                                            backend="onnxruntime", device="cpu", to_openpose=False)
    model.reset()
    meta = probe(video)
    n = meta.frame_count
    xy = np.full((n, 26, 2), np.nan)
    prev = None
    for i, rgb in iter_frames(meta):
        kpts, scores = model(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if kpts is None or len(kpts) == 0:
            continue
        kpts, scores = np.asarray(kpts), np.asarray(scores)
        centres = kpts[:, [11, 12]].mean(axis=1)
        if prev is None:   # the subject: the largest person in the first frame, then the same one
            size = np.ptp(kpts[:, :, 1], axis=1)
            k = int(np.argmax(size))
        else:
            k = int(np.argmin(np.linalg.norm(centres - prev, axis=1)))
        prev = centres[k]
        p = kpts[k].astype(float)
        p[scores[k] < min_score] = np.nan
        xy[i] = p
    points = {name: xy[:, idx, :] for name, idx in HALPE.items()}
    return _times(meta), points
