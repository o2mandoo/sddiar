#!/usr/bin/env python3
"""Validate a local independent annotation JSONL manifest.

Only the redacted aggregate report is printed.  The validator performs no
network access and writes no files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sddiar.annotation_intake import validate_annotation_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="local dataset manifest.jsonl")
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="directory containing audio/, rttm/, and uem/ (default: manifest parent)",
    )
    args = parser.parse_args(argv)
    report = validate_annotation_dataset(args.manifest, dataset_root=args.dataset_root)
    print(json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
