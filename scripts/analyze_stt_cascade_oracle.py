#!/usr/bin/env python3
"""Print a redacted, default-off STT cascade oracle curve.

Input is JSON containing ``segments`` (or an array) with source-time bounds
and scalar errors/redacted character counts.  The command never prints the
input filename, segment identifiers, or transcript text.  It performs no
model loading, subprocess execution, or network access.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sddiar.stt_cascade_experimental import (  # noqa: E402
    DEFAULT_BUDGETS,
    MAX_CLI_INPUT_BYTES,
    DEFAULT_MAX_SEGMENTS,
    DEFAULT_MAX_SOLVER_STATES,
    DEFAULT_MAX_TOTAL_DURATION_US,
    SttCascadeContractError,
    analyze_stt_cascade_oracle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON input file, or '-' for stdin")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)), help="comma-separated budget percentages")
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SEGMENTS)
    parser.add_argument("--max-total-duration-us", type=int, default=DEFAULT_MAX_TOTAL_DURATION_US)
    parser.add_argument("--max-solver-states", type=int, default=DEFAULT_MAX_SOLVER_STATES)
    args = parser.parse_args(argv)
    try:
        budgets = tuple(int(item.strip()) for item in args.budgets.split(",") if item.strip())
        if args.input == "-":
            raw = sys.stdin.buffer.read(MAX_CLI_INPUT_BYTES + 1)
        else:
            path = Path(args.input)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CLI_INPUT_BYTES:
                raise SttCascadeContractError("input file is invalid or exceeds the resource bound")
            raw = path.read_bytes()
        if len(raw) > MAX_CLI_INPUT_BYTES:
            raise SttCascadeContractError("input exceeds the resource bound")
        value = json.loads(raw)
        report = analyze_stt_cascade_oracle(value, budgets=budgets, max_segments=args.max_segments,
                                            max_total_duration_us=args.max_total_duration_us,
                                            max_solver_states=args.max_solver_states)
        json.dump(report, sys.stdout, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        sys.stdout.write("\n")
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError, SttCascadeContractError):
        # Never echo a caller-supplied path or malformed source payload.
        print("stt cascade oracle input rejected", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
