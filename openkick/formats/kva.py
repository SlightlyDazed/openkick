"""Kinovea annotation files (.kva, Kinovea 2023 / FormatVersion 2.0).

Format notes learned from files saved by Kinovea 2023.1.2 itself:
- Times are the video stream's real presentation timestamps (pts), in the
  stream time base (usually 1/90000 s). Never `frame * ticks`.
- Skeleton: one <GenericPosture> (OpenPose BODY_25 tool) per keyframe.
- Angle: ONE <Angle> in a keyframe; its motion lives in
  <Trackability><TrackableDrawing id=angle id><TrackablePoint key="o|a|b">
  <Timeline><Frame time location source/>. "o" is the vertex.
- Track: <TrackPointList Count="N"><TrackPoint>x;y;t</TrackPoint> as text.
  The old attribute form (x= y= t=) is silently ignored by Kinovea 2023.
- Unknown elements are ignored by Kinovea, so a partial file still loads.
"""

from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from openkick.core import landmarks as L
from openkick.core.project import SRC_AUTO, Measurement, Phase, PointRef, Project, PHASE_COLORS
from openkick.core.video import VideoMeta

OPENPOSE_TOOL = "a9ce7c83-4120-4400-8b20-b6f034213522"


def _uid() -> str:
    return str(uuid.uuid4())


def _argb(hex_color: str, alpha: int = 255) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{alpha};{r};{g};{b}"


def _rgb(hex_color: str) -> str:
    return _argb(hex_color).split(";", 1)[1]


def _hex(argb: str, default: str = "#FF8A3D") -> str:
    try:
        parts = [int(float(v)) for v in argb.split(";")]
        r, g, b = parts[-3:]
        return f"#{r:02X}{g:02X}{b:02X}"
    except (ValueError, IndexError):
        return default


def _num(v: float) -> str:
    return f"{round(float(v), 2)}"


def _tracker_params(v: VideoMeta, indent: str) -> str:
    sw = f"{round(v.width * 0.2)};{round(v.height * 0.2)}"
    bw = f"{round(v.width * 0.05)};{round(v.height * 0.05)}"
    return (f"{indent}<TrackerParameters>\n"
            f"{indent}  <SimilarityThreshold>0.5</SimilarityThreshold>\n"
            f"{indent}  <TemplateUpdateThreshold>0.8</TemplateUpdateThreshold>\n"
            f"{indent}  <RefinementNeighborhood>1</RefinementNeighborhood>\n"
            f"{indent}  <SearchWindow>{sw}</SearchWindow>\n"
            f"{indent}  <BlockWindow>{bw}</BlockWindow>\n"
            f"{indent}</TrackerParameters>")


def _body25(project: Project, frame: int) -> list[tuple[float, float]] | None:
    xy = project.pose.xy[frame]
    if not np.isfinite(xy).any():
        return None
    pts = []
    for names in L.BODY25:
        vals = [xy[L.INDEX[n]] for n in names]
        vals = [v for v in vals if np.isfinite(v).all()]
        if not vals:
            # Kinovea cannot store gaps: fall back to the mid-hip, or the image origin.
            mid = xy[[L.INDEX["left_hip"], L.INDEX["right_hip"]]]
            mid = mid[np.isfinite(mid).all(axis=1)]
            pts.append(tuple(mid.mean(axis=0)) if len(mid) else (0.0, 0.0))
        else:
            pts.append(tuple(np.mean(vals, axis=0)))
    return pts


# ── export ─────────────────────────────────────────────────────────────


def export(project: Project, path: str | Path, skeleton_color: str = "#2DD4BF") -> dict:
    """Write a Kinovea 2023 .kva. Returns a summary of what was written."""
    v = project.video
    n = project.frame_count
    ts = v.pts
    first, last = ts[0], ts[-1]
    phase_start = {p.start: p for p in project.phases}

    drawings: dict[int, list[str]] = {}
    for f in range(n):
        pts = _body25(project, f)
        if pts is None:
            continue
        pos = "\n".join(f"            <Point>{_num(x)};{_num(y)}</Point>" for x, y in pts)
        drawings.setdefault(f, []).append(f"""        <GenericPosture id="{_uid()}" name="OpenPose">
          <ToolId>{OPENPOSE_TOOL}</ToolId>
          <Positions>
{pos}
          </Positions>
          <DrawingStyle>
            <Color Key="line color"><Value>{_argb(skeleton_color)}</Value></Color>
          </DrawingStyle>
          <InfosFading>
            <Enabled>true</Enabled>
            <UseDefault>false</UseDefault>
            <AlwaysVisible>false</AlwaysVisible>
            <MasterFactor>1</MasterFactor>
            <OpaqueFrames>1</OpaqueFrames>
            <Frames>0</Frames>
          </InfosFading>
        </GenericPosture>""")

    trackability, tracks = [], []
    written = {"angles": 0, "trajectories": 0, "skipped": []}
    for m in project.measurements:
        samples = []
        for f in range(n):
            ps = [project.point(r, f) for r in m.points]
            if all(p is not None for p in ps):
                samples.append((f, ps))
        if not samples:
            continue
        if m.kind == "angle":
            aid = _uid()
            f0, (A, O, B) = samples[0]
            drawings.setdefault(f0, []).append(f"""        <Angle id="{aid}" name="{escape(m.name, {'"': '&quot;'})}">
          <PointO>{_num(O[0])};{_num(O[1])}</PointO>
          <PointA>{_num(B[0])};{_num(B[1])}</PointA>
          <PointB>{_num(A[0])};{_num(A[1])}</PointB>
          <Signed>false</Signed>
          <CCW>true</CCW>
          <Supplementary>false</Supplementary>
          <DrawingStyle>
            <Color Key="line color">
              <Value>{_argb(m.color)}</Value>
            </Color>
            <FontSize Key="font size">
              <Value>12</Value>
            </FontSize>
          </DrawingStyle>
          <InfosFading>
            <Enabled>true</Enabled>
            <UseDefault>false</UseDefault>
            <AlwaysVisible>true</AlwaysVisible>
            <MasterFactor>1</MasterFactor>
            <OpaqueFrames>1</OpaqueFrames>
            <Frames>20</Frames>
          </InfosFading>
        </Angle>""")
            blocks = []
            for key, idx in (("o", 1), ("a", 2), ("b", 0)):
                frames = "\n".join(
                    f'          <Frame time="{ts[f]}" location="{_num(ps[idx][0])};{_num(ps[idx][1])}" source="Manual" />'
                    for f, ps in samples)
                p0 = samples[0][1][idx]
                blocks.append(f"""      <TrackablePoint key="{key}">
{_tracker_params(v, '        ')}
        <NonTrackingValue>{_num(p0[0])};{_num(p0[1])}</NonTrackingValue>
        <CurrentValue>{_num(p0[0])};{_num(p0[1])}</CurrentValue>
        <Timeline>
{frames}
        </Timeline>
      </TrackablePoint>""")
            trackability.append(f'    <TrackableDrawing id="{aid}">\n' + "\n".join(blocks) + "\n    </TrackableDrawing>")
            written["angles"] += 1
        elif m.kind in ("trajectory", "point"):
            f0, (P0,) = samples[0]
            pts = "\n".join(f"        <TrackPoint>{_num(p[0][0])};{_num(p[0][1])};{ts[f]}</TrackPoint>" for f, p in samples)
            name = escape(m.name, {'"': '&quot;'})
            tracks.append(f"""    <Track id="{_uid()}" name="{name}">
      <TimePosition>{ts[f0]}</TimePosition>
      <Visible>0</Visible>
      <Invisible>-1</Invisible>
      <ExtraData>Name</ExtraData>
      <Marker>Cross</Marker>
{_tracker_params(v, '      ')}
      <TrackPointList Count="{len(samples)}">
{pts}
      </TrackPointList>
      <MainLabel Text="{name}">
        <SpacePosition>{_num(P0[0] - 20)};{_num(P0[1] - 50)}</SpacePosition>
        <TimePosition>{ts[f0]}</TimePosition>
      </MainLabel>
      <KeyframeLabelList Count="0">
      </KeyframeLabelList>
      <SeeFuture>true</SeeFuture>
      <ShowTrackLabel>true</ShowTrackLabel>
      <ShowKeyframeLabels>false</ShowKeyframeLabels>
      <UseKeyframeColors>false</UseKeyframeColors>
      <IsInteractiveTrack>true</IsInteractiveTrack>
      <ShowRotationCircle>false</ShowRotationCircle>
      <DrawingStyle>
        <Color Key="color">
          <Value>{_argb(m.color)}</Value>
        </Color>
        <LineSize Key="line size">
          <Value>3</Value>
        </LineSize>
        <TrackShape Key="track shape">
          <Value>Solid;false</Value>
        </TrackShape>
      </DrawingStyle>
    </Track>""")
            written["trajectories"] += 1
        else:
            written["skipped"].append(m.name)

    # Keyframes: every frame with drawings + every phase start (named after the phase).
    kf_frames = sorted(set(drawings) | set(phase_start))
    kfs = []
    for f in kf_frames:
        ph = phase_start.get(f)
        name = escape(ph.name) if ph else f"{v.time_of(f):.2f}"
        color = _rgb(ph.color) if ph else "70;130;180"
        body = "\n".join(drawings.get(f, []))
        kfs.append(f"""    <Keyframe id="{_uid()}">
      <Timestamp>{ts[f]}</Timestamp>
      <Name>{name}</Name>
      <Color>{color}</Color>
      <Drawings>
{body}
      </Drawings>
    </Keyframe>""")

    tracks_xml = ("\n  <Tracks>\n" + "\n".join(tracks) + "\n  </Tracks>") if tracks else ""
    track_xml = ("\n  <Trackability>\n" + "\n".join(trackability) + "\n  </Trackability>") if trackability \
        else "\n  <Trackability />"
    content = f"""﻿<?xml version="1.0" encoding="utf-8"?>
<KinoveaVideoAnalysis>
  <FormatVersion>2.0</FormatVersion>
  <Producer>Kinovea.2023.1.2</Producer>
  <OriginalFilename>{escape(Path(v.path).stem)}</OriginalFilename>
  <FullPath>{escape(str(Path(v.path).resolve()))}</FullPath>
  <ImageSize>{v.width};{v.height}</ImageSize>
  <Aspect>Auto</Aspect>
  <Rotation>Rotate0</Rotation>
  <Mirror>false</Mirror>
  <Demosaicing>None</Demosaicing>
  <Deinterlacing>false</Deinterlacing>
  <AverageTimeStampsPerFrame>{v.ticks_per_frame}</AverageTimeStampsPerFrame>
  <CaptureFramerate>{round(v.fps)}</CaptureFramerate>
  <UserFramerate>{v.fps}</UserFramerate>
  <FirstTimeStamp>{first}</FirstTimeStamp>
  <SelectionStart>{first}</SelectionStart>
  <SelectionEnd>{last}</SelectionEnd>
  <TimeOrigin>{first}</TimeOrigin>
  <Calibration>
    <CalibrationLine>
      <Length>100</Length>
      <Segment>
        <A>{v.width // 2};{v.height // 2}</A>
        <B>{v.width // 2 + 100};{v.height // 2}</B>
      </Segment>
      <Origin>0;100</Origin>
      <Axis>LineHorizontal</Axis>
    </CalibrationLine>
    <CalibrationDrawingId>00000000-0000-0000-0000-000000000000</CalibrationDrawingId>
    <Unit Abbreviation="px">Pixels</Unit>
  </Calibration>
  <Keyframes>
{chr(10).join(kfs)}
  </Keyframes>{tracks_xml}{track_xml}
</KinoveaVideoAnalysis>
"""
    Path(path).write_text(content, encoding="utf-8")
    written["keyframes"] = len(kfs)
    return written


# ── import ─────────────────────────────────────────────────────────────


def read_header(path: str | Path) -> dict:
    """Video reference stored in a .kva (to locate the video before importing)."""
    root = _parse(path)
    return {"original": root.findtext("OriginalFilename", ""), "full_path": root.findtext("FullPath", "")}


def find_video(kva_path: str | Path) -> Path | None:
    kva_path = Path(kva_path)
    h = read_header(kva_path)
    candidates = []
    if h["full_path"]:
        fp = h["full_path"]
        candidates.append(Path(fp))
        # Windows path stored by Kinovea: try the file name next to the .kva
        candidates.append(kva_path.parent / re.split(r"[\\/]", fp)[-1])
    stem = h["original"] or kva_path.stem
    for ext in (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".webm"):
        candidates.append(kva_path.parent / f"{stem}{ext}")
    return next((c for c in candidates if c.exists()), None)


def _parse(path: str | Path) -> ET.Element:
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return ET.fromstring(data)


def _xy(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    try:
        x, y = text.split(";")[:2]
        return float(x), float(y)
    except ValueError:
        return None


def import_into(project: Project, path: str | Path) -> list[str]:
    """Load skeletons, angles, tracks and named keyframes from a .kva into `project`.
    Returns human-readable notes about what was imported."""
    root = _parse(path)
    v = project.video
    n = project.frame_count
    kva_first = int(float(root.findtext("FirstTimeStamp", "0") or 0))
    kva_tpf = float(root.findtext("AverageTimeStampsPerFrame", "0") or 0)
    same_base = kva_tpf <= 0 or abs(kva_tpf - v.ticks_per_frame) / v.ticks_per_frame < 0.05

    def frame_of(t: str | int | float) -> int:
        t = float(t)
        if same_base:
            return v.index_of_pts(int(round(t)))
        return max(0, min(n - 1, round((t - kva_first) / kva_tpf)))

    notes = []
    body25_to_lm = {}
    for k, names in enumerate(L.BODY25):
        if len(names) == 1 and L.INDEX[names[0]] not in body25_to_lm.values():
            body25_to_lm[k] = L.INDEX[names[0]]

    postures = 0
    named = []
    angle_nodes = {}
    for kf in root.iter("Keyframe"):
        f = frame_of(kf.findtext("Timestamp", "0"))
        name = (kf.findtext("Name") or "").strip()
        if name and not re.fullmatch(r"[\d.,:]+", name):
            named.append((f, name, kf.findtext("Color", "")))
        drawings = kf.find("Drawings")
        if drawings is None:
            continue
        for d in drawings:
            if d.tag == "GenericPosture" and d.findtext("ToolId") == OPENPOSE_TOOL:
                pts = [_xy(p.text) for p in d.iter("Point")]
                if len(pts) != 25:
                    continue
                for k, lm in body25_to_lm.items():
                    if pts[k] is not None:
                        project.pose.xy[f, lm] = pts[k]
                        project.pose.raw_xy[f, lm] = pts[k]
                        project.pose.visibility[f, lm] = 1.0
                        project.pose.source[f, lm] = SRC_AUTO
                postures += 1
            elif d.tag == "Angle":
                angle_nodes[d.get("id")] = (f, d)

    if postures:
        notes.append(f"skeleton:{postures}")

    trackable = {td.get("id"): td for td in root.iter("TrackableDrawing")}
    for aid, (f, d) in angle_nodes.items():
        refs = {}
        td = trackable.get(aid)
        for key, tag in (("o", "PointO"), ("a", "PointA"), ("b", "PointB")):
            keys = {}
            if td is not None:
                tp = next((p for p in td.iter("TrackablePoint") if p.get("key") == key), None)
                if tp is not None:
                    for fr in tp.iter("Frame"):
                        loc = _xy(fr.get("location"))
                        if loc:
                            keys[frame_of(fr.get("time", "0"))] = loc
            if not keys:
                p = _xy(d.findtext(tag))
                if p:
                    keys[f] = p
            refs[key] = PointRef(keys=keys)
        color = _hex(d.findtext("DrawingStyle/Color/Value", "") or "")
        project.measurements.append(Measurement(kind="angle", name=d.get("name", "Angle"), color=color,
                                                points=[refs["b"], refs["o"], refs["a"]]))
    if angle_nodes:
        notes.append(f"angles:{len(angle_nodes)}")

    tracks = 0
    for tr in root.iter("Track"):
        keys = {}
        for tp in tr.iter("TrackPoint"):
            if tp.text and tp.text.strip():
                parts = tp.text.strip().split(";")
                if len(parts) >= 3:
                    keys[frame_of(parts[2])] = (float(parts[0]), float(parts[1]))
            elif tp.get("x") is not None:
                keys[frame_of(tp.get("t", "0"))] = (float(tp.get("x")), float(tp.get("y")))
        if not keys:
            continue
        label = tr.get("name") or tr.findtext("MainLabel") or (tr.find("MainLabel").get("Text") if tr.find("MainLabel") is not None else "") or "Track"
        color = _hex(tr.findtext("DrawingStyle/Color/Value", "") or "", "#FF5CA8")
        project.measurements.append(Measurement(kind="trajectory", name=label.strip() or "Track", color=color,
                                                points=[PointRef(keys=keys)]))
        tracks += 1
    if tracks:
        notes.append(f"trajectories:{tracks}")

    if named:
        named.sort()
        for i, (f, name, color) in enumerate(named):
            end = named[i + 1][0] - 1 if i + 1 < len(named) else n - 1
            c = PHASE_COLORS[i % len(PHASE_COLORS)]
            if color and color != "70;130;180":
                c = _hex("255;" + color, c)
            project.phases.append(Phase(name=name, start=f, end=max(f, end), color=c))
        notes.append(f"phases:{len(named)}")
    project.touch()
    return notes
