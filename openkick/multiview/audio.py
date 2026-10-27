"""Synchronizing videos by their sound.

Phones filming the same moment hear the same room: the shared sound lines the videos up to a
few milliseconds, whatever the person does. Movement alone cannot always do it: a repeated
movement (pedalling, walking on a treadmill) looks the same one turn later, and the videos can be
lined up one whole turn off.

The two soundtracks are compared with GCC-PHAT (generalized cross-correlation with phase
transform), which ignores how loud each microphone is and keeps only when things happen.
"""

from __future__ import annotations

import numpy as np

RATE = 8000               # Hz: enough for the timing, light to compare
LONGEST_S = 90.0          # only the first minute and a half is compared
SURE = 8.0                # the best match must stand this many standard deviations above the rest


def soundtrack(path: str, rate: int = RATE, longest_s: float = LONGEST_S):
    """Mono sound of a video file and when it starts in the file's own clock (seconds), or None."""
    import av
    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                return None
            stream = c.streams.audio[0]
            start = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
            resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            chunks, total = [], 0
            for frame in c.decode(stream):
                for out in resampler.resample(frame):
                    a = out.to_ndarray().reshape(-1)
                    chunks.append(a)
                    total += len(a)
                if total >= longest_s * rate:
                    break
    except Exception:  # noqa: BLE001 - a file whose sound cannot be read is synchronized by movement instead
        return None
    if not chunks:
        return None
    x = np.concatenate(chunks)[: int(longest_s * rate)].astype(np.float64)
    if len(x) < rate or not np.isfinite(x).all() or np.std(x) < 1e-6:
        return None
    return x, start


def video_start(path: str) -> float:
    """When the first video frame is shown, in the file's own clock (seconds)."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        pts = [p.pts for p in c.demux(s) if p.pts is not None]
        return float(min(pts) * s.time_base) if pts else 0.0


def lag(a: np.ndarray, b: np.ndarray, rate: int = RATE, most_s: float | None = None) -> tuple[float, float]:
    """Seconds L for which a(t) sounds like b(t + L), and how sure the match is (the peak against
    the spread of the rest; above SURE it can be trusted)."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    A, B = np.fft.rfft(a - a.mean(), n), np.fft.rfft(b - b.mean(), n)
    cross = np.conj(A) * B
    cross /= np.abs(cross) + 1e-12
    cc = np.fft.irfft(cross, n)
    cc = np.concatenate([cc[-(len(a) - 1):], cc[:len(b)]])       # lags from -(len(a)-1) to len(b)-1
    lags = np.arange(-(len(a) - 1), len(b))
    if most_s is not None:
        keep = np.abs(lags) <= most_s * rate
        cc, lags = cc[keep], lags[keep]
    k = int(np.argmax(cc))
    peak = cc[k]
    rest = np.delete(cc, slice(max(0, k - rate // 50), k + rate // 50))   # all but the peak's own 20 ms
    sure = float((peak - np.mean(rest)) / (np.std(rest) + 1e-12))
    # sub-sample peak from the parabola through its neighbours
    shift = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        bend = y0 - 2 * y1 + y2
        if abs(bend) > 1e-12:
            shift = 0.5 * (y0 - y2) / bend
    return float((lags[k] + shift) / rate), sure


def offset(path_a: str, path_b: str, most_s: float | None = None) -> tuple[float | None, float]:
    """Seconds to add to video B's clock (from its first frame) to match video A's, from their
    sound, and how sure that is. (None, 0) when either has no usable sound."""
    sa, sb = soundtrack(path_a), soundtrack(path_b)
    if sa is None or sb is None:
        return None, 0.0
    (xa, aa), (xb, ab) = sa, sb
    L, sure = lag(xa, xb, most_s=most_s)
    # the same instant: aa + t + ca = ab + t + L + cb  ->  cb - ca = aa - ab - L
    # a frame shown at B's clock tb is at A's clock tb + (vb - va) + (cb - ca)
    va, vb = video_start(path_a), video_start(path_b)
    return float(vb - va + aa - ab - L), sure
