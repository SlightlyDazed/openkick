"""Dragging the edges of phases on a timeline, shared by the 2D and the 3D timelines.

A timeline has two rows of blocks: the phases (strides, pedal strokes, the user's own) and the
sub-phases inside them (stance and swing...). An edge is where a block starts or where one ends;
it is named by the first frame after it. Dragging it moves every block of both rows that starts
or ends there: an edge shared by two neighbours stays shared, and a stride and its stance keep
starting at the same initial contact."""

from __future__ import annotations


def edges(blocks) -> list[int]:
    """Every edge of one row, by the first frame after it, each once."""
    return sorted({b.start for b in blocks} | {b.end + 1 for b in blocks})


def drag(phases, subphases, edge: int, to: int, frame_count: int) -> int:
    """Move the edge at frame `edge` to `to`, in both rows: every block that starts there starts
    at the new place and every block that ends just before it ends just before the new place.
    The move is cut short so that each of those keeps at least one frame and none runs into the
    blocks next to it. Changes the blocks in place; returns where the edge ended up."""
    starts = [b for b in [*phases, *subphases] if b.start == edge]
    ends = [b for b in [*phases, *subphases] if b.end == edge - 1]
    if not starts and not ends:
        return edge
    lo, hi = 0, frame_count
    for b in starts:
        hi = min(hi, b.end)                       # it keeps at least its last frame
    for b in ends:
        lo = max(lo, b.start + 1)                 # and this one its first
    moving = {id(b) for b in starts + ends}
    for row in (phases, subphases):
        for b in row:
            if id(b) in moving:
                continue
            if b.end < edge:                      # wholly before the edge: do not run into it
                lo = max(lo, b.end + 1)
            elif b.start >= edge:                 # wholly after it
                hi = min(hi, b.start)
    f = max(lo, min(hi, int(to)))
    for b in starts:
        b.start = f
    for b in ends:
        b.end = f - 1
    return f
