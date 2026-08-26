#!/usr/bin/env python3
"""CLI validator for an assembled offline sddiar release."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from sddiar.release import TARGETS, scan_zero_network_policy, validate_release_layout

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--target", action="append", choices=TARGETS)
    parser.add_argument("--production", action="store_true", help="apply the signed production evidence gates")
    parser.add_argument("--mode", choices=("development", "production"))
    parser.add_argument("--scan", action="store_true", help="scan release Python sources for network primitives")
    parser.add_argument("--scan-source", action="append", type=Path, default=[],
                        help="additional local source directory/file to scan for network primitives")
    args = parser.parse_args()
    selected = tuple(args.target or TARGETS)
    mode = args.mode or ("production" if args.production else "development")
    if args.production and args.mode and args.mode != "production":
        parser.error("--production conflicts with --mode development")
    report = validate_release_layout(
        args.release_root,
        targets=selected,
        mode=mode,
    )
    issues = list(report.issues)
    scan_paths = list(args.scan_source)
    if args.scan:
        scan_paths.append(args.release_root / "src")
    if scan_paths:
        issues.extend(scan_zero_network_policy(scan_paths))
    payload = {"ok": not issues, "mode": report.mode, "root": str(report.root), "targets": selected, "verified_files": report.verified_files, "issues": [i.__dict__ for i in issues]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not issues else 1

if __name__ == "__main__": raise SystemExit(main())
