"""Headless end-to-end check of an installation:  OpenKick --selftest <video> <out_folder>

Runs detection, applies the kick template, marks phases and writes a project, a
Kinovea .kva, a PDF report and a screenshot, plus selftest.txt with the result."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


def run(video: str, out: str) -> int:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    def log(msg):
        report.append(msg)
        (out_dir / "selftest.txt").write_text("\n".join(report), encoding="utf-8")

    try:
        from openkick import AUTHOR, __version__
        log(f"OpenKick {__version__} — {AUTHOR}")
        from openkick.core import templates
        from openkick.core.pose import PoseDetector, ensure_model
        from openkick.core.project import SRC_AUTO, Project
        from openkick.core.video import probe
        from openkick.formats import kva
        import numpy as np
        t0 = time.time()
        meta = probe(video)
        log(f"video: {meta.frame_count} frames, {meta.width}x{meta.height}, {meta.fps:.2f} fps")
        res = PoseDetector(ensure_model("heavy")).run(meta)
        log(f"pose: {res.detected}/{meta.frame_count} frames in {time.time() - t0:.1f} s")
        p = Project(meta)
        p.pose.xy[:] = res.xy
        p.pose.raw_xy[:] = res.xy
        p.pose.visibility[:] = res.visibility
        p.pose.z[:] = res.z
        p.pose.source[:] = np.where(np.isfinite(res.xy[..., 0]), SRC_AUTO, 0)
        templates.apply(p, templates.find("kick"))
        n = meta.frame_count
        for i, name in enumerate(templates.find("kick").phase_names()):
            p.add_phase_at(i * n // 4, name)
        p.save(out_dir / "selftest.okick")
        info = kva.export(p, out_dir / "selftest.kva")
        log(f"kva: {info}")
        from PySide6.QtWidgets import QApplication
        from openkick.ui.main_window import MainWindow
        from openkick.ui.exporters import write_pdf
        w = MainWindow()
        w.resize(1440, 900)
        w.show()
        w.ctl.set_project(p)
        w.show_workspace()
        w.ctl.set_frame(n // 2)
        QApplication.processEvents()
        w.canvas.focus_athlete()
        QApplication.processEvents()
        w.grab().save(str(out_dir / "selftest.png"))
        write_pdf(p, w.results.chart, str(out_dir / "selftest.pdf"))
        log("pdf and screenshot written")
        p.saved_revision = p.revision
        w.close()
        try:
            import rtmlib  # noqa: F401
        except ImportError:
            log("3d: not installed")
        else:                                       # the 3D detector: a few frames of the same video
            from openkick.core.video import VideoMeta
            from openkick.multiview.detect import detect
            short = VideoMeta(meta.path, meta.width, meta.height, meta.fps, meta.time_base, meta.pts[:30], meta.rotation)
            t1 = time.time()
            _, xy3, _ = detect(short, "balanced", use_cache=False)
            log(f"3d: {int(np.isfinite(xy3[:, 0, 0]).sum())}/30 frames with a person in {time.time() - t1:.1f} s")
        log("OK")
        return 0
    except Exception:  # noqa: BLE001
        log("FAILED\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2]))
