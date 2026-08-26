#!/usr/bin/env python3
"""Parse Clova Note's simple speaker/timestamp export for evaluation.

Only timing and pseudonymous speaker IDs are retained.  Transcript text and
the source speaker labels are deliberately never emitted by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_TIME = re.compile(r"(?<!\d)(?P<h>\d{1,2}):(?P<m>[0-5]\d)(?::(?P<s>[0-5]\d))?(?!\d)")
_HEADER = re.compile(r"^\s*(?P<label>.+?)\s+(?P<time>\d{1,2}:[0-5]\d(?::[0-5]\d)?)\s*$")


def _seconds(value: str) -> float:
    parts = [int(x) for x in value.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def _header(line: str) -> tuple[str, float] | None:
    match = _HEADER.match(line)
    if not match:
        return None
    label = match.group("label").strip()
    # Date/metadata rows can end in a time too.  A genuine header has one
    # timestamp and a non-numeric label.
    if not label or _TIME.search(label) or re.search(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", label):
        return None
    return label, _seconds(match.group("time"))


def parse_clova_text(text: str, *, source_name: str | None = None) -> dict[str, Any]:
    """Return a JSON-safe, text-free timing reference.

    A turn ends at the next turn's start.  The final turn ends at the largest
    metadata timestamp before the first speaker header when one is available;
    otherwise its end is ``None``.
    """
    lines = text.splitlines()
    candidates: list[tuple[str, float]] = []
    preamble_times: list[float] = []
    first_header_line: int | None = None
    for index, line in enumerate(lines):
        parsed = _header(line)
        if parsed is None:
            if first_header_line is None:
                preamble_times.extend(_seconds(m.group(0)) for m in _TIME.finditer(line))
            continue
        # Require a following nonblank line: this avoids mistaking a title or
        # date row for a speaker header.
        if not any(part.strip() for part in lines[index + 1 :]):
            continue
        if first_header_line is None:
            first_header_line = index
        candidates.append(parsed)
    if not candidates:
        raise ValueError("no Clova speaker timestamp headers found")

    labels: dict[str, str] = {}
    def pseudonym(label: str) -> str:
        if label not in labels:
            labels[label] = f"REF_{len(labels):02d}"
        return labels[label]

    declared_end = max(preamble_times) if preamble_times else None
    # Clova's Korean export commonly writes duration as ``52분 51초`` rather
    # than a second clock value (the other clock is the recording time).
    preamble = "\n".join(lines[: first_header_line or 0])
    duration_match = re.search(r"(?:(\d+)시간\s*)?(?:(\d+)분\s*)?(?:(\d+)초)", preamble)
    if duration_match and (duration_match.group(1) or duration_match.group(2) or duration_match.group(3)):
        declared_end = float((int(duration_match.group(1) or 0) * 3600)
                             + (int(duration_match.group(2) or 0) * 60)
                             + int(duration_match.group(3) or 0))
    turns: list[dict[str, Any]] = []
    for pos, (label, start) in enumerate(candidates):
        next_start = candidates[pos + 1][1] if pos + 1 < len(candidates) else declared_end
        end = next_start if next_start is None or next_start >= start else None
        turns.append({"speaker_id": pseudonym(label), "start_sec": start, "end_sec": end})

    durations: dict[str, float] = {}
    counts: dict[str, int] = {}
    for turn in turns:
        sid = turn["speaker_id"]
        counts[sid] = counts.get(sid, 0) + 1
        if turn["end_sec"] is not None:
            durations[sid] = durations.get(sid, 0.0) + turn["end_sec"] - turn["start_sec"]
    return {
        "schema": "clova_reference_timing_v2",
        # Never retain the input filename: it can contain a person, customer,
        # or meeting name.  A content-derived opaque ID is enough to bind an
        # evaluation artifact to its source without exposing that metadata.
        "source_id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "speaker_ids": list(labels.values()),
        "turns": turns,
        "stats": {"turn_count": len(turns), "duration_sec": declared_end,
                  "speakers": {sid: {"turn_count": counts[sid], "duration_sec": durations.get(sid, 0.0)}
                               for sid in labels.values()}},
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a Clova Note transcript without emitting text")
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path, help="write pseudonymous timing JSON")
    args = parser.parse_args(argv)
    try:
        result = parse_clova_text(args.input.read_text(encoding="utf-8"), source_name=str(args.input))
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json_path:
        args.json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = result["stats"]
    print(f"turns={stats['turn_count']} speakers={len(result['speaker_ids'])} duration_sec={stats['duration_sec']}")
    for sid, item in stats["speakers"].items():
        print(f"{sid}: turns={item['turn_count']} duration_sec={item['duration_sec']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
