"""Offline-only CLI for validating and inspecting local P6 job artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from .release import TARGETS, scan_zero_network_policy, validate_release_layout
from .service import atomic_publish, read_json
from .worker import LocalJobStore
from .onnx_diarization import LocalOnnxDiarizationConfig, LocalOnnxDiarizer

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sddiar")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("path")
    status = sub.add_parser("status"); status.add_argument("root"); status.add_argument("job_id")
    release = sub.add_parser("verify-release")
    release.add_argument("release_root")
    release.add_argument("--target", action="append", choices=TARGETS)
    release.add_argument("--scan-source", action="append", default=[])
    release.add_argument("--production", action="store_true", help="apply the signed production evidence gates")
    release.add_argument("--mode", choices=("development", "production"))
    diarize = sub.add_parser("diarize", help="run the development-only local CPU ONNX diarizer")
    diarize.add_argument("audio")
    diarize.add_argument("--silero-model", required=True)
    diarize.add_argument("--silero-sha256", required=True)
    diarize.add_argument("--wespeaker-model", required=True)
    diarize.add_argument("--wespeaker-sha256", required=True)
    diarize.add_argument("--threads", type=int, default=1)
    diarize.add_argument("--assignment-distance-limit", type=float)
    diarize.add_argument("--silero-temporal-postprocess", action="store_true")
    diarize.add_argument("--auto-gain-normalization", action="store_true")
    diarize.add_argument("--output", help="optional local JSON output path")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            value = read_json(args.path)
            if not isinstance(value, dict): raise ValueError("JSON root must be an object")
            print(json.dumps({"valid": True, "keys": sorted(value)}, ensure_ascii=False)); return 0
        if args.command == "verify-release":
            mode = args.mode or ("production" if args.production else "development")
            if args.production and args.mode and args.mode != "production":
                parser.error("--production conflicts with --mode development")
            report = validate_release_layout(
                args.release_root,
                targets=tuple(args.target or TARGETS),
                mode=mode,
            )
            issues = list(report.issues)
            if args.scan_source:
                issues.extend(scan_zero_network_policy([Path(path) for path in args.scan_source]))
            print(json.dumps({"ok": not issues, "mode": report.mode, "verified_files": report.verified_files,
                              "issues": [issue.__dict__ for issue in issues]}, ensure_ascii=False))
            return 0 if not issues else 1
        if args.command == "diarize":
            result = LocalOnnxDiarizer(
                args.silero_model,
                args.wespeaker_model,
                silero_sha256=args.silero_sha256,
                wespeaker_sha256=args.wespeaker_sha256,
                config=LocalOnnxDiarizationConfig(
                    assignment_distance_limit=args.assignment_distance_limit,
                    silero_temporal_postprocess=args.silero_temporal_postprocess,
                    auto_gain_normalization=args.auto_gain_normalization,
                ),
                threads=args.threads,
            ).process(args.audio)
            payload = result.to_json()
            if args.output:
                atomic_publish(args.output, result.to_dict())
            else:
                print(payload)
            return 0
        job = LocalJobStore(args.root).status(args.job_id)
        print(json.dumps({"job_id": job.job_id, "status": job.status.value, "attempts": job.attempts,
                          "result_path": job.result_path, "error_code": job.error_code}, ensure_ascii=False)); return 0
    except Exception as exc:
        print(json.dumps({"valid": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
