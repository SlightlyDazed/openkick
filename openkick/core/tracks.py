"""Tracks of the timeline and how blocks on them are edited, as in a video editor.

A track is a row of the timeline. Two come with every project: the phases (strides, pedal
strokes, the user's own phases) and the sub-phases inside them (stance and swing, the eight
phases of gait...). The user adds more, above or below: a track of segments (stretches of time
with a name, like the phases) or a track of events (single instants: a heel strike, the ball
hit, "it hurt here").

Everything here works on plain lists of blocks, so the 2D and the 3D timelines share it. A
segment is a Phase (first and last frame, both inside it); an event is a Mark. Blocks on one
track never overlap. Nothing here knows about the screen: pixels become frames before calling in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openkick.core.project import PHASE_COLORS, Phase, new_id

SEGMENTS, EVENTS = "segments", "events"
TRACK_COLORS = ["#8B5CF6", "#EC4899", "#F59E0B", "#14B8A6", "#EF4444", "#3B82F6", "#84CC16", "#F97316"]


@dataclass
class Mark:
    """One instant on an events track."""
    name: str
    frame: int
    color: str = TRACK_COLORS[0]
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "frame": self.frame, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "Mark":
        return cls(name=d["name"], frame=int(d["frame"]), color=d.get("color", TRACK_COLORS[0]), id=d.get("id", new_id()))


@dataclass
class Track:
    """A row the user made. `items` are Phases (segments) or Marks (events), sorted in time."""
    name: str
    kind: str = SEGMENTS
    color: str = TRACK_COLORS[0]
    items: list = field(default_factory=list)
    hidden: bool = False
    locked: bool = False
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "color": self.color, "hidden": self.hidden,
                "locked": self.locked, "items": [i.to_dict() for i in self.items]}

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        kind = d.get("kind", SEGMENTS)
        load = Phase.from_dict if kind == SEGMENTS else Mark.from_dict
        return cls(name=d["name"], kind=kind, color=d.get("color", TRACK_COLORS[0]), hidden=d.get("hidden", False),
                   locked=d.get("locked", False), id=d.get("id", new_id()), items=[load(i) for i in d.get("items", [])])

    def sort(self):
        self.items.sort(key=lambda b: b.start if self.kind == SEGMENTS else b.frame)


# ── tracks ─────────────────────────────────────────────────────────────

def new_track(tracks: list, kind: str, name: str, at: int | None = None) -> Track:
    """Add an empty track at position `at` of the user's tracks (None: at the end)."""
    used = {t.color for t in tracks}
    color = next((c for c in TRACK_COLORS if c not in used), TRACK_COLORS[len(tracks) % len(TRACK_COLORS)])
    t = Track(name=name, kind=kind, color=color)
    tracks.insert(len(tracks) if at is None else max(0, min(at, len(tracks))), t)
    return t


def move_track(tracks: list, track: Track, step: int) -> bool:
    """One place up (-1) or down (+1). False when it is already at that end."""
    i = tracks.index(track)
    j = i + step
    if not 0 <= j < len(tracks):
        return False
    tracks[i], tracks[j] = tracks[j], tracks[i]
    return True


# ── segments ───────────────────────────────────────────────────────────

def _room(blocks: list, block, frame_count: int) -> tuple[int, int]:
    """First and last frame `block` may reach without running into its neighbours."""
    lo, hi = 0, frame_count - 1
    for b in blocks:
        if b is block:
            continue
        if b.end < block.start:
            lo = max(lo, b.end + 1)
        elif b.start > block.end:
            hi = min(hi, b.start - 1)
    return lo, hi


def free_span(blocks: list, frame: int, frame_count: int) -> tuple[int, int] | None:
    """The empty stretch around `frame` on a track (None if a block covers it)."""
    if any(b.start <= frame <= b.end for b in blocks):
        return None
    lo = max([b.end + 1 for b in blocks if b.end < frame], default=0)
    hi = min([b.start - 1 for b in blocks if b.start > frame], default=frame_count - 1)
    return lo, hi


def add_segment(blocks: list, frame: int, length: int, frame_count: int, name: str, color: str | None = None) -> Phase | None:
    """A new segment from `frame` on, `length` frames long or up to the next block. None if
    `frame` is already inside a block."""
    span = free_span(blocks, frame, frame_count)
    if span is None:
        return None
    end = min(span[1], frame + max(1, length) - 1)
    ph = Phase(name=name, start=frame, end=end, color=color or PHASE_COLORS[len(blocks) % len(PHASE_COLORS)])
    blocks.append(ph)
    blocks.sort(key=lambda b: b.start)
    return ph


def move(blocks: list, block, start: int, frame_count: int, children: list | None = None) -> int:
    """Slide `block` so that it starts at `start`, keeping its length and stopping at its
    neighbours. `children` (sub-phases inside it) go along. Returns where it starts now."""
    lo, hi = _room(blocks, block, frame_count)
    length = block.end - block.start
    new = max(lo, min(hi - length, int(start)))
    shift = new - block.start
    block.start, block.end = new, new + length
    for c in children or []:
        c.start += shift
        c.end += shift
    return new


def split(blocks: list, block, frame: int) -> Phase | None:
    """Cut `block` in two: the second part starts at `frame`. None when `frame` is not inside it
    (past its first frame)."""
    if not block.start < frame <= block.end:
        return None
    second = Phase(name=block.name, start=frame, end=block.end, color=block.color, kind=block.kind)
    block.end = frame - 1
    blocks.insert(blocks.index(block) + 1, second)
    return second


def merge_next(blocks: list, block) -> bool:
    """Join `block` with the block right after it (and the gap between them, if any)."""
    after = sorted((b for b in blocks if b.start > block.end), key=lambda b: b.start)
    if not after:
        return False
    block.end = after[0].end
    blocks.remove(after[0])
    return True


def duplicate(blocks: list, block, frame_count: int) -> Phase | None:
    """A copy of `block` right after it, as long as it fits in the room there."""
    length = block.end - block.start + 1
    span = free_span(blocks, block.end + 1, frame_count) if block.end + 1 < frame_count else None
    if span is None:
        return None
    copy = Phase(name=block.name, start=span[0], end=min(span[1], span[0] + length - 1), color=block.color)
    blocks.append(copy)
    blocks.sort(key=lambda b: b.start)
    return copy


def resize(blocks: list, block, side: str, frame: int, frame_count: int) -> int:
    """Move the first ("start") or last ("end") frame of one block of a user track, without
    passing its neighbours and keeping at least one frame. Returns the frame it ended at."""
    lo, hi = _room(blocks, block, frame_count)
    if side == "start":
        block.start = max(lo, min(block.end, int(frame)))
        return block.start
    block.end = max(block.start, min(hi, int(frame)))
    return block.end


# ── events ─────────────────────────────────────────────────────────────

def add_mark(marks: list, frame: int, name: str, color: str) -> Mark:
    m = Mark(name=name, frame=int(frame), color=color)
    marks.append(m)
    marks.sort(key=lambda k: k.frame)
    return m


def move_mark(marks: list, mark: Mark, frame: int, frame_count: int) -> int:
    mark.frame = max(0, min(frame_count - 1, int(frame)))
    marks.sort(key=lambda k: k.frame)
    return mark.frame


# ── snapping ───────────────────────────────────────────────────────────

def snap_targets(rows: list, playhead: int | None = None, skip=()) -> list[int]:
    """Frames an edge or an event can snap to: where any block starts or ends (by the first frame
    after it), every event and the playhead. `rows` are lists of blocks or marks; `skip` holds the
    blocks being moved, which do not attract themselves."""
    out = set()
    skip = {id(b) for b in skip}
    for row in rows:
        for b in row:
            if id(b) in skip:
                continue
            if isinstance(b, Mark):
                out.add(b.frame)
            else:
                out.update((b.start, b.end + 1))
    if playhead is not None:
        out.add(int(playhead))
    return sorted(out)


def snap(frame: int, targets: list[int], reach: float) -> tuple[int, int | None]:
    """The nearest target within `reach` frames, or `frame` itself. Returns (frame, target hit)."""
    best = min(targets, key=lambda f: abs(f - frame), default=None)
    if best is not None and abs(best - frame) <= reach:
        return best, best
    return frame, None
