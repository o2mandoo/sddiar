#!/usr/bin/env python3
"""Run the offline Korean diarization benchmark and print redacted JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sddiar.evaluation_io import EvaluationIOError, evaluate_korean_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("prediction_manifest", type=Path)
    parser.add_argument("--corpus-lock", type=Path, required=True)
    parser.add_argument(
        "--split", required=True,
        choices=("CALIBRATION", "DEVELOPMENT_HOLDOUT", "RELEASE_HOLDOUT"),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    args = parser.parse_args(argv)
    try:
        result = evaluate_korean_benchmark(
            args.reference_manifest,
            args.prediction_manifest,
            corpus_lock=args.corpus_lock,
            split=args.split,
            dataset_root=args.dataset_root,
            prediction_root=args.prediction_root,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    except EvaluationIOError as exc:
        print(json.dumps({"ok": False, "error": exc.as_dict()}, ensure_ascii=False,
                         sort_keys=True, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
