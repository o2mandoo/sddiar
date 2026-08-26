#!/usr/bin/env python3
"""Build a deterministic private blind diarization annotation pack.

The command emits only redacted hash/count evidence on stdout.  Audio clips,
manifest, and blank labels remain below the caller-selected private root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sddiar.blind_annotation import BlindAnnotationError, build_blind_annotation_pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline private blind annotation pack")
    parser.add_argument("source", type=Path, help="absolute PCM16 mono WAV")
    parser.add_argument("--reference", "--timing-reference", dest="reference", required=True, type=Path, help="timing-only reference JSON/RTTM")
    parser.add_argument("--system", "--system-output", dest="system", required=True, type=Path, help="system timing/disagreement JSON/RTTM")
    parser.add_argument("--output-root", required=True, type=Path, help="fresh final target below .private/blind-annotation")
    parser.add_argument("--repo-root", type=Path, help="repository root used for the canonical private-root check")
    parser.add_argument("--presentation-nonce", type=str, help="secret deterministic presentation nonce (not printed or stored)")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        result = build_blind_annotation_pack(
            args.source,
            args.output_root,
            reference_path=args.reference,
            system_path=args.system,
            seed=args.seed,
            clip_seconds=args.clip_seconds,
            repo_root=args.repo_root,
            presentation_nonce=args.presentation_nonce,
        )
    except BlindAnnotationError as exc:
        parser.error(str(exc))
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
