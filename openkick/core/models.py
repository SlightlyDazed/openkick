"""The detection models OpenKick needs, whether they are on this computer, and getting them.

They are downloaded once and then work without internet: the 2D one from Google (MediaPipe), the
3D ones from OpenMMLab (RTMPose and its person detector, through rtmlib, which keeps them in its
own folder). The first-run window (ui/firstrun.py) offers to get them all at once; otherwise
each is fetched the first time it is needed.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable

THREE_D_MODES = ("balanced", "lightweight")     # Balanced (the default) and Fast
TITLES = {"2d": "2D analysis: body detection (Google MediaPipe)",
          "3d": "3D analysis: body and feet detection (OpenMMLab RTMPose)"}


def _rtm_urls() -> list[str]:
    """The RTMPose and person-detector downloads of the 3D modes (empty if rtmlib is not installed)."""
    try:
        from rtmlib import BodyWithFeet
    except Exception:  # noqa: BLE001 - without the 3D extras there is nothing to get
        return []
    return [BodyWithFeet.MODE[m][k] for m in THREE_D_MODES for k in ("det", "pose")]


def _rtm_dir() -> Path:
    from rtmlib.tools.file import _get_rtmhub_dir
    return Path(_get_rtmhub_dir()) / "checkpoints"


def _rtm_ready(url: str) -> bool:
    name = Path(url.split("?")[0]).name
    return (_rtm_dir() / (name.split(".")[0] + ".onnx")).exists()


def components() -> list[dict]:
    """What there is to get: key, title, approximate size (MB), whether it is already here."""
    from openkick.core.pose import model_path
    out = [{"key": "2d", "title": TITLES["2d"], "mb": 30,
            "ready": model_path("heavy").exists()}]
    urls = _rtm_urls()
    if urls:
        out.append({"key": "3d", "title": TITLES["3d"], "mb": 200,
                    "ready": all(_rtm_ready(u) for u in urls)})
    return out


def missing() -> list[dict]:
    return [c for c in components() if not c["ready"]]


def _fetch(url: str, dst: Path, progress: Callable[[int, int], None] | None, cancelled: Callable[[], bool] | None) -> bool:
    """Download url to dst (through a .part file). False if cancelled."""
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 16):
            if cancelled and cancelled():
                out.close()
                tmp.unlink(missing_ok=True)
                return False
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    tmp.replace(dst)
    return True


def download(key: str, progress: Callable[[int, int, int, int], None] | None = None,
             cancelled: Callable[[], bool] | None = None) -> bool:
    """Get one component. `progress(file, files, done bytes, total bytes)`. False if cancelled."""
    if key == "2d":
        from openkick.core.pose import ensure_model
        ensure_model("heavy", (lambda d, t: progress(0, 1, d, t)) if progress else None)
        return True
    from rtmlib.tools.file import download_checkpoint
    urls = [u for u in _rtm_urls() if not _rtm_ready(u)]
    folder = _rtm_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls):
        name = Path(url.split("?")[0]).name
        try:
            ok = _fetch(url, folder / name, (lambda d, t, i=i: progress(i, len(urls), d, t)) if progress else None, cancelled)
        except OSError:
            ok = True                    # rtmlib tries its mirror below
        if not ok:
            return False
        download_checkpoint(url)         # unpacks it where rtmlib looks (or downloads it from its mirror)
    return True
