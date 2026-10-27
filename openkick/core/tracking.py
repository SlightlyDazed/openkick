"""Automatic tracking of a free point (ball, marker on the floor...) with
pyramidal Lucas-Kanade optical flow and a forward-backward consistency check,
so tracking stops where it becomes unreliable instead of drifting silently."""

from __future__ import annotations

from typing import Callable

import numpy as np

from openkick.core.video import VideoMeta, VideoReader


def track_point(meta: VideoMeta, start: int, xy: tuple[float, float], direction: int = 1,
                stop_at: int | None = None, on_progress: Callable[[int, int], None] | None = None,
                should_stop: Callable[[], bool] | None = None, max_error: float = 1.5):
    """Follow `xy` from frame `start` frame by frame. Returns (positions {frame: (x, y)}, lost_frame or None)."""
    import cv2
    n = meta.frame_count
    end = (n - 1 if direction > 0 else 0) if stop_at is None else stop_at
    win = max(15, int(round(min(meta.width, meta.height) * 0.045)) | 1)
    params = dict(winSize=(win, win), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    out = {start: (float(xy[0]), float(xy[1]))}
    frames = list(range(start + direction, end + direction, direction))
    with VideoReader(meta, cache_size=4) as reader:
        prev = cv2.cvtColor(reader.frame(start), cv2.COLOR_RGB2GRAY)
        p0 = np.array([[xy]], np.float32)
        for k, f in enumerate(frames):
            if should_stop and should_stop():
                return out, None
            cur = cv2.cvtColor(reader.frame(f), cv2.COLOR_RGB2GRAY)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **params)
            if p1 is None or not st[0][0]:
                return out, f
            back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **params)
            err = float(np.linalg.norm(back[0][0] - p0[0][0])) if back is not None else 1e9
            x, y = map(float, p1[0][0])
            if not st2[0][0] or err > max_error or not (0 <= x < meta.width and 0 <= y < meta.height):
                return out, f
            out[f] = (x, y)
            prev, p0 = cur, p1
            if on_progress:
                on_progress(k + 1, len(frames))
    return out, None
