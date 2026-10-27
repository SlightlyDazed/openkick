"""The patients kept on this computer, to pick them again in the next analysis.

Every patient record saved in an analysis is remembered here: the latest version of it and the
analyses it is in (2D projects and 3D sessions). Each analysis keeps its own copy of the record
as it was that day (the weight then, the saddle height then); this list only holds the latest,
to start the next analysis from. It lives in <data dir>/patients.json and never leaves the
computer.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from openkick.core.pose import data_dir
from openkick.core.subject import Subject


def _file() -> Path:
    return data_dir() / "patients.json"


def _read() -> list[dict]:
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [p for p in data.get("patients", []) if isinstance(p, dict) and p.get("id")]


def _write(rows: list[dict]):
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"format": "openkick-patients", "version": 1, "patients": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(f)                       # all or nothing: a crash never leaves half a list


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def every() -> list[dict]:
    """Each patient: {"id", "subject" (Subject), "updated", "analyses": [{"path", "kind", "date"}]},
    the most recently changed first."""
    rows = _read()
    out = [{"id": r["id"], "subject": Subject.from_dict(r.get("subject")), "updated": r.get("updated", ""),
            "analyses": list(r.get("analyses", []))} for r in rows]
    return sorted(out, key=lambda r: r["updated"], reverse=True)


def get(patient_id: str) -> dict | None:
    return next((p for p in every() if p["id"] == patient_id), None)


def remember(subject: Subject) -> Subject:
    """Keep this record as the patient's latest. A record without an id becomes a new patient
    (the returned copy carries its id); an empty one is not kept."""
    if subject is None or subject.empty:
        return subject
    s = Subject.from_dict(subject.to_dict())
    s.id = s.id or uuid.uuid4().hex[:12]
    rows = _read()
    row = next((r for r in rows if r["id"] == s.id), None)
    if row is None:
        rows.append({"id": s.id, "subject": s.to_dict(), "updated": _now(), "analyses": []})
    else:
        row.update(subject=s.to_dict(), updated=_now())
    _write(rows)
    return s


def note_analysis(subject: Subject, path, kind: str):
    """An analysis with this patient was saved at `path` ("2D" or "3D"). Never raises: a list
    that cannot be written must not stop the analysis itself being saved."""
    if subject is None or not subject.id:
        return
    try:
        _note(subject, path, kind)
    except OSError:
        pass


def _note(subject: Subject, path, kind: str):
    rows = _read()
    row = next((r for r in rows if r["id"] == subject.id), None)
    if row is None:
        return
    full = str(Path(path).resolve())
    done = [a for a in row.get("analyses", []) if a.get("path") != full]
    row["analyses"] = done + [{"path": full, "kind": kind, "date": _now()}]
    _write(rows)


def forget(patient_id: str):
    """Off the list (the analyses themselves are not touched)."""
    _write([r for r in _read() if r["id"] != patient_id])

