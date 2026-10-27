"""Project model: video + pose data + measurements + annotations + phases.

Design rule: the skeleton is the single source of truth. Angles, trajectories
and distances reference skeleton landmarks, so correcting a joint once updates
every measurement that uses it. Free points (not on the body) store keyed
positions per frame and interpolate between keys.
"""

from __future__ import annotations

import bisect
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openkick import __version__
from openkick.core import landmarks as L
from openkick.core.video import VideoMeta

FORMAT_VERSION = 1

# Per-landmark data origin
SRC_NONE, SRC_AUTO, SRC_MANUAL, SRC_SMOOTHED = 0, 1, 2, 3

LOW_CONFIDENCE = 0.5

MEASURE_COLORS = ["#FF8A3D", "#FACC15", "#FF5CA8", "#5EEAD4", "#A78BFA", "#60A5FA", "#F87171", "#C6F24E"]
PHASE_COLORS = ["#2E4763", "#4A3C7A", "#8A4B17", "#1D5A52", "#6B2D4A", "#3F4F1C", "#5A4A1E", "#28446B"]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class PointRef:
    """A point used by a measurement. Exactly one of these defines it:

    landmark  a MediaPipe skeleton point (index)
    segment   a point rigidly attached to a body segment: (a, b, along, across) means
              A + along·(B−A) + across·perp(B−A), so it moves and scales with the bone
    derived   a computed point, e.g. "com:m" / "com:f" (whole-body center of mass)
    marker    the id of a "point" measurement (a user marker), so edits propagate
    keys      free point: positions keyed per frame, linearly interpolated in between
    """
    landmark: int | None = None
    keys: dict[int, tuple[float, float]] = field(default_factory=dict)   # frame -> (x, y)
    segment: tuple[int, int, float, float] | None = None
    derived: str | None = None
    marker: str | None = None

    @property
    def is_landmark(self) -> bool:
        return self.landmark is not None

    @property
    def kind(self) -> str:
        if self.landmark is not None:
            return "landmark"
        if self.segment is not None:
            return "segment"
        if self.derived is not None:
            return "derived"
        if self.marker is not None:
            return "marker"
        return "free"

    @property
    def follows_body(self) -> bool:
        return self.kind in ("landmark", "segment", "derived")

    def free_at(self, frame: int) -> tuple[float, float] | None:
        if not self.keys:
            return None
        frames = sorted(self.keys)
        if frame <= frames[0]:
            return self.keys[frames[0]]
        if frame >= frames[-1]:
            return self.keys[frames[-1]]
        j = bisect.bisect_right(frames, frame)
        f0, f1 = frames[j - 1], frames[j]
        if f0 == frame:
            return self.keys[f0]
        t = (frame - f0) / (f1 - f0)
        (x0, y0), (x1, y1) = self.keys[f0], self.keys[f1]
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    def to_dict(self) -> dict:
        if self.landmark is not None:
            return {"landmark": L.NAMES[self.landmark]}
        if self.segment is not None:
            a, b, al, ac = self.segment
            return {"segment": [L.NAMES[a], L.NAMES[b], round(al, 5), round(ac, 5)]}
        if self.derived is not None:
            return {"derived": self.derived}
        if self.marker is not None:
            return {"marker": self.marker}
        return {"keys": {str(k): [round(v[0], 2), round(v[1], 2)] for k, v in sorted(self.keys.items())}}

    @classmethod
    def from_dict(cls, d: dict) -> "PointRef":
        if "landmark" in d:
            lm = d["landmark"]
            return cls(landmark=L.INDEX[lm] if isinstance(lm, str) else int(lm))
        if "segment" in d:
            a, b, al, ac = d["segment"]
            idx = lambda v: L.INDEX[v] if isinstance(v, str) else int(v)  # noqa: E731
            return cls(segment=(idx(a), idx(b), float(al), float(ac)))
        if "derived" in d:
            return cls(derived=d["derived"])
        if "marker" in d:
            return cls(marker=d["marker"])
        return cls(keys={int(k): (float(v[0]), float(v[1])) for k, v in d.get("keys", {}).items()})

    @classmethod
    def at(cls, lm: int) -> "PointRef":
        return cls(landmark=lm)

    @classmethod
    def free(cls, frame: int, x: float, y: float) -> "PointRef":
        return cls(keys={frame: (x, y)})

    @classmethod
    def on_segment(cls, a: int, b: int, pa, pb, p) -> "PointRef":
        """Attach point `p` to the segment a→b given their current positions."""
        ux, uy = pb[0] - pa[0], pb[1] - pa[1]
        l2 = ux * ux + uy * uy or 1.0
        dx, dy = p[0] - pa[0], p[1] - pa[1]
        along = (dx * ux + dy * uy) / l2
        across = (dx * -uy + dy * ux) / l2
        return cls(segment=(a, b, along, across))


@dataclass
class Measurement:
    kind: str                      # "angle" | "trajectory" | "distance" | "point"
    name: str
    points: list[PointRef]         # angle: [A, vertex, B]; trajectory/point: [P]; distance: [P, Q]
    color: str = "#FF8A3D"
    visible: bool = True
    opacity: float = 1.0           # 0.0 = fully transparent, 1.0 = fully opaque (for trajectory overlay)
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict:
        d = {"id": self.id, "kind": self.kind, "name": self.name, "color": self.color,
             "visible": self.visible, "points": [p.to_dict() for p in self.points]}
        if self.opacity < 1.0:
            d["opacity"] = round(self.opacity, 2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Measurement":
        return cls(kind=d["kind"], name=d["name"], color=d.get("color", "#FF8A3D"),
                   visible=d.get("visible", True), opacity=d.get("opacity", 1.0),
                   id=d.get("id", new_id()),
                   points=[PointRef.from_dict(p) for p in d["points"]])


@dataclass
class TextNote:
    text: str
    x: float
    y: float
    start: int                     # first frame where it is fully visible
    end: int                       # last frame where it is fully visible (inclusive)
    color: str = "#E8EAED"
    fade_in: float = 0.0           # seconds to fade in after `start` (0 = appears at once)
    fade_out: float = 0.0          # seconds to fade out after `end` (0 = disappears at once)
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "x": round(self.x, 2), "y": round(self.y, 2),
                "start": self.start, "end": self.end, "color": self.color,
                "fade_in": self.fade_in, "fade_out": self.fade_out}

    @classmethod
    def from_dict(cls, d: dict) -> "TextNote":
        return cls(text=d["text"], x=d["x"], y=d["y"], start=d["start"], end=d["end"],
                   color=d.get("color", "#E8EAED"), fade_in=d.get("fade_in", 0.0),
                   fade_out=d.get("fade_out", 0.0), id=d.get("id", new_id()))

    def opacity(self, video: "VideoMeta", frame: int) -> float:
        """0..1 visibility of the note at `frame`, including fades."""
        t = video.time_of(frame)
        ts, te = video.time_of(self.start), video.time_of(self.end)
        if t < ts:
            return 0.0
        a = 1.0
        if self.fade_in > 0 and t < ts + self.fade_in:
            a = (t - ts) / self.fade_in + 1e-3
        if t > te:
            if self.fade_out <= 0 or t >= te + self.fade_out:
                return 0.0
            a = min(a, 1 - (t - te) / self.fade_out)
        return max(0.0, min(1.0, a))


@dataclass
class Phase:
    name: str
    start: int                     # first frame
    end: int                       # last frame (inclusive)
    color: str = PHASE_COLORS[0]
    id: str = field(default_factory=new_id)
    kind: str = ""                 # what an analysis made it for ("stride-right", "stance", "swing"); "" = the user's

    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "start": self.start, "end": self.end, "color": self.color}
        if self.kind:
            d["kind"] = self.kind
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Phase":
        return cls(name=d["name"], start=d["start"], end=d["end"],
                   color=d.get("color", PHASE_COLORS[0]), id=d.get("id", new_id()), kind=d.get("kind", ""))


@dataclass
class Calibration:
    px_per_m: float
    a: tuple[float, float]
    b: tuple[float, float]
    length_m: float

    def to_dict(self) -> dict:
        return {"px_per_m": self.px_per_m, "a": list(self.a), "b": list(self.b), "length_m": self.length_m}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(px_per_m=d["px_per_m"], a=tuple(d["a"]), b=tuple(d["b"]), length_m=d["length_m"])


class Pose:
    """Per-frame landmarks. `raw_xy` keeps MediaPipe's output to allow reverting edits."""

    def __init__(self, n: int):
        self.xy = np.full((n, L.COUNT, 2), np.nan, np.float32)
        self.raw_xy = self.xy.copy()
        self.visibility = np.zeros((n, L.COUNT), np.float32)
        self.z = np.full((n, L.COUNT), np.nan, np.float32)
        self.source = np.zeros((n, L.COUNT), np.uint8)

    @property
    def frame_count(self) -> int:
        return self.xy.shape[0]

    @property
    def has_data(self) -> bool:
        return bool(np.isfinite(self.xy).any())

    def point(self, frame: int, lm: int) -> tuple[float, float] | None:
        x, y = self.xy[frame, lm]
        if np.isnan(x):
            return None
        return float(x), float(y)

    def to_dict(self) -> dict:
        def arr(a, nd=2):
            a = np.round(a.astype(np.float64), nd)
            return np.where(np.isnan(a), None, a).tolist()
        return {"xy": arr(self.xy), "raw_xy": arr(self.raw_xy), "visibility": arr(self.visibility, 3),
                "z": arr(self.z, 4), "source": self.source.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        def arr(v):
            return np.array(v, dtype=np.float64).astype(np.float32)
        xy = arr(d["xy"])
        p = cls(xy.shape[0])
        p.xy = xy
        p.raw_xy = arr(d.get("raw_xy", d["xy"]))
        p.visibility = arr(d["visibility"])
        p.z = arr(d["z"])
        p.source = np.array(d["source"], np.uint8)
        return p


class Project:
    def __init__(self, video: VideoMeta):
        self.video = video
        self.pose = Pose(video.frame_count)
        self.measurements: list[Measurement] = []
        self.notes: list[TextNote] = []
        self.phases: list[Phase] = []
        self.subphases: list[Phase] = []
        self.tracks: list = []              # the user's own rows of the timeline (openkick.core.tracks.Track)
        from openkick.core.subject import Subject
        self.subject = Subject()            # the patient's record
        self.calibration: Calibration | None = None
        self.template: str | None = None
        self.path: Path | None = None
        self.revision = 0          # bumped on every change; analysis caches key on it
        self.saved_revision = 0

    # ── state ──────────────────────────────────────────────
    @property
    def frame_count(self) -> int:
        return self.video.frame_count

    @property
    def dirty(self) -> bool:
        return self.revision != self.saved_revision

    def touch(self):
        self.revision += 1

    @property
    def title(self) -> str:
        if self.path:
            return self.path.stem
        return Path(self.video.path).stem

    # ── geometry ───────────────────────────────────────────
    def point(self, ref: PointRef, frame: int, _depth: int = 0) -> tuple[float, float] | None:
        k = ref.kind
        if k == "landmark":
            return self.pose.point(frame, ref.landmark)
        if k == "segment":
            a, b, al, ac = ref.segment
            pa, pb = self.pose.point(frame, a), self.pose.point(frame, b)
            if pa is None or pb is None:
                return None
            ux, uy = pb[0] - pa[0], pb[1] - pa[1]
            return (pa[0] + al * ux - ac * uy, pa[1] + al * uy + ac * ux)
        if k == "derived":
            from openkick.core.bodymodel import derived_point
            return derived_point(ref.derived, self.pose.xy[frame])
        if k == "marker":
            m = self.measurement(ref.marker)
            if m is None or _depth > 4:
                return None
            return self.point(m.points[0], frame, _depth + 1)
        return ref.free_at(frame)

    def dependents(self, mid: str) -> list["Measurement"]:
        """Measurements that use the marker `mid`."""
        return [m for m in self.measurements if any(r.marker == mid for r in m.points)]

    def measurement(self, mid: str) -> Measurement | None:
        return next((m for m in self.measurements if m.id == mid), None)

    def next_color(self) -> str:
        used = {m.color for m in self.measurements}
        for c in MEASURE_COLORS:
            if c not in used:
                return c
        return MEASURE_COLORS[len(self.measurements) % len(MEASURE_COLORS)]

    def phase_at(self, frame: int) -> Phase | None:
        return next((p for p in self.phases if p.start <= frame <= p.end), None)

    def subphase_at(self, frame: int) -> Phase | None:
        return next((p for p in self.subphases if p.start <= frame <= p.end), None)

    def sort_phases(self):
        self.phases.sort(key=lambda p: p.start)

    def sort_subphases(self):
        self.subphases.sort(key=lambda p: p.start)

    def add_phase_at(self, frame: int, name: str) -> Phase:
        """Start a new phase at `frame`; it lasts until the next phase or the end of the video.
        A phase that contained `frame` is cut there."""
        end = self.frame_count - 1
        for p in self.phases:
            if p.start > frame:
                end = min(end, p.start - 1)
        for p in self.phases:
            if p.start < frame <= p.end:
                p.end = frame - 1
        color = PHASE_COLORS[len(self.phases) % len(PHASE_COLORS)]
        phase = Phase(name=name, start=frame, end=end, color=color)
        self.phases = [p for p in self.phases if not (p.start == frame)]
        self.phases.append(phase)
        self.sort_phases()
        self.touch()
        return phase

    # ── persistence ────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "format": "openkick-project", "format_version": FORMAT_VERSION, "producer": f"OpenKick {__version__}",
            "video": self.video.to_dict(), "pose": self.pose.to_dict(),
            "measurements": [m.to_dict() for m in self.measurements],
            "notes": [n.to_dict() for n in self.notes],
            "phases": [p.to_dict() for p in self.phases],
            "subphases": [p.to_dict() for p in self.subphases],
            "tracks": [t.to_dict() for t in self.tracks],
            "subject": self.subject.to_dict(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "template": self.template,
        }

    def save(self, path: str | Path):
        path = Path(path)
        data = self.to_dict()
        # Store the video path relative to the project when possible (portable folders).
        try:
            data["video"]["path"] = str(Path(self.video.path).resolve().relative_to(path.resolve().parent))
        except ValueError:
            data["video"]["path"] = str(Path(self.video.path).resolve())
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)
        self.path = path
        self.saved_revision = self.revision

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "openkick-project":
            raise ValueError("Not an OpenKick project")
        video = VideoMeta.from_dict(data["video"])
        vp = Path(video.path)
        if not vp.is_absolute():
            video.path = str((path.parent / vp).resolve())
        p = cls(video)
        p.pose = Pose.from_dict(data["pose"])
        p.measurements = [Measurement.from_dict(m) for m in data.get("measurements", [])]
        p.notes = [TextNote.from_dict(n) for n in data.get("notes", [])]
        p.phases = [Phase.from_dict(ph) for ph in data.get("phases", [])]
        p.subphases = [Phase.from_dict(sph) for sph in data.get("subphases", [])]
        from openkick.core.tracks import Track
        p.tracks = [Track.from_dict(t) for t in data.get("tracks", [])]
        from openkick.core.subject import Subject
        p.subject = Subject.from_dict(data.get("subject"))
        cal = data.get("calibration")
        p.calibration = Calibration.from_dict(cal) if cal else None
        p.template = data.get("template")
        p.path = path
        return p
