"""Frame-accurate video access with PyAV (FFmpeg).

Every frame is addressed by its index and carries its REAL presentation
timestamp (pts). Phone and messaging-app videos often have a variable frame
rate, so `index * (1 / fps)` is not a valid time base: always use `pts`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    time_base: Fraction
    pts: list[int] = field(default_factory=list)
    rotation: int = 0          # degrees applied to decoded frames to display them upright

    @property
    def frame_count(self) -> int:
        return len(self.pts)

    @property
    def first_pts(self) -> int:
        return self.pts[0] if self.pts else 0

    def time_of(self, index: int) -> float:
        """Seconds from the first frame."""
        return float((self.pts[index] - self.first_pts) * self.time_base)

    @property
    def duration(self) -> float:
        if len(self.pts) < 2:
            return 0.0
        # Include the display time of the last frame.
        return self.time_of(len(self.pts) - 1) + 1.0 / max(self.fps, 1e-6)

    def index_at_time(self, seconds: float) -> int:
        """Nearest frame index for a time in seconds."""
        if not self.pts:
            return 0
        target = self.first_pts + seconds / float(self.time_base)
        arr = np.asarray(self.pts)
        return int(np.abs(arr - target).argmin())

    def index_of_pts(self, pts: int) -> int:
        arr = np.asarray(self.pts)
        return int(np.abs(arr - pts).argmin())

    @property
    def ticks_per_frame(self) -> int:
        """Average timestamp increment per frame (Kinovea's AverageTimeStampsPerFrame)."""
        if len(self.pts) >= 2:
            return max(1, round((self.pts[-1] - self.pts[0]) / (len(self.pts) - 1)))
        return max(1, round(1 / (float(self.time_base) * max(self.fps, 1e-6))))

    def to_dict(self) -> dict:
        return {
            "path": self.path, "width": self.width, "height": self.height, "fps": self.fps,
            "time_base": [self.time_base.numerator, self.time_base.denominator],
            "pts": self.pts, "rotation": self.rotation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VideoMeta":
        return cls(path=d["path"], width=d["width"], height=d["height"], fps=d["fps"],
                   time_base=Fraction(*d["time_base"]), pts=list(d["pts"]),
                   rotation=d.get("rotation", 0))


def _rotate(img: np.ndarray, rotation: int) -> np.ndarray:
    k = round(rotation / 90) % 4
    if k:
        img = np.rot90(img, k)
    return np.ascontiguousarray(img)


def probe(path: str | Path) -> VideoMeta:
    """Read size, frame rate and the real timestamp of every frame (no decoding)."""
    path = str(path)
    with av.open(path) as container:
        stream = container.streams.video[0]
        pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None and p.size > 0)
    with av.open(path) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        rotation = int(getattr(first, "rotation", 0) or 0)
        width, height = first.width, first.height
        if round(rotation / 90) % 2:
            width, height = height, width
        time_base = stream.time_base or Fraction(1, 90000)
        rate = stream.average_rate or stream.guessed_rate or 30
        if not pts:  # containers without packet timestamps: synthesize them
            n = stream.frames or 0
            step = round(1 / (float(time_base) * float(rate)))
            pts = [i * step for i in range(n)]
    return VideoMeta(path=path, width=width, height=height, fps=float(rate),
                     time_base=Fraction(time_base), pts=pts, rotation=rotation)


class VideoReader:
    """Random access to frames by index, optimized for sequential playback.

    Not thread-safe: use one reader per thread.
    """

    def __init__(self, meta: VideoMeta, cache_size: int = 48):
        self.meta = meta
        self._container = av.open(meta.path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._decoder = None
        self._last_index = -2
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._pts_to_index = {p: i for i, p in enumerate(meta.pts)}

    def close(self):
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def frame(self, index: int) -> np.ndarray:
        """RGB uint8 array (H, W, 3) of the frame at `index`."""
        index = max(0, min(index, self.meta.frame_count - 1))
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if index != self._last_index + 1 or self._decoder is None:
            self._seek(index)
        img = self._decode_until(index)
        self._put(index, img)
        return img

    def _seek(self, index: int):
        target = self.meta.pts[index]
        self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
        self._decoder = self._container.decode(self._stream)
        self._last_index = -2

    def _decode_until(self, index: int) -> np.ndarray:
        target = self.meta.pts[index]
        last = None
        for frame in self._decoder:
            if frame.pts is None:
                continue
            img = _rotate(frame.to_ndarray(format="rgb24"), self.meta.rotation)
            i = self._pts_to_index.get(frame.pts)
            if i is not None and i != index:
                self._put(i, img)
            last = img
            if frame.pts >= target:
                self._last_index = index
                return img
        # End of stream before reaching the target (broken tail): return what we have.
        self._decoder = None
        if last is None:
            return np.zeros((self.meta.height, self.meta.width, 3), np.uint8)
        return last

    def _put(self, index: int, img: np.ndarray):
        self._cache[index] = img
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def iter_frames(meta: VideoMeta):
    """Stream (index, rgb) for every frame in presentation order without holding them all."""
    pts_to_index = {p: i for i, p in enumerate(meta.pts)}
    with av.open(meta.path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            i = pts_to_index.get(frame.pts)
            if i is None:
                continue
            yield i, _rotate(frame.to_ndarray(format="rgb24"), meta.rotation)
