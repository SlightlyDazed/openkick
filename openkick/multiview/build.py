"""From the video files to a solved session (the automatic first attempt)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from openkick.core.video import probe
from openkick.multiview import detect as D
from openkick.multiview.session import MAX_VIEWS, Session, View


def steps(count: int) -> list[str]:
    """Names of the stages reported while building a session of `count` videos."""
    return [f"detect_{k}" for k in range(count)] + ["motion", "solve"]


def build(videos: list, height_m: float, mode: str = "balanced", progress: Callable | None = None,
          cancelled: Callable[[], bool] | None = None) -> Session | None:
    """`progress(step, done, total, rgb=None, xy=None)` reports every stage. None if cancelled."""
    if not 2 <= len(videos) <= MAX_VIEWS:
        raise ValueError(f"between 2 and {MAX_VIEWS} videos are needed")

    def say(step, i, n, rgb=None, xy=None):
        if progress:
            progress(step, i, n, rgb, xy)
    found = []
    for k, path in enumerate(videos):
        meta = probe(path)
        if meta.frame_count < 10:
            raise ValueError(f"{Path(path).name}: the video is too short")
        res = D.detect(meta, mode, lambda i, n, rgb, xy, k=k: say(f"detect_{k}", i + 1, n, rgb, xy), cancelled)
        if res is None:
            return None
        say(f"detect_{k}", meta.frame_count, meta.frame_count)
        found.append((meta, *res))
    views = []
    for k, (meta, t, xy, score) in enumerate(found):
        T = D.camera_motion(meta, xy, progress=lambda i, n, k=k: say("motion", k * n + i + 1, len(found) * n), cancelled=cancelled)
        if T is None:
            return None
        views.append(View(meta, t, xy, score, T if D.is_handheld(T, meta) else None))
    say("motion", 1, 1)
    s = Session(views, height_m)
    s.solve(progress=lambda i, n: say("solve", i, n))
    say("solve", 1, 1)
    return s
