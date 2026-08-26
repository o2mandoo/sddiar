#!/usr/bin/env python3
"""Evaluate development-only FP32/INT8 pyannote segmentation evidence parity.

Each model runs in its own subprocess so process peak RSS is comparable.  The
approved recording is inspected at deterministic [0, 60s] and [0, 300s]
segments.  No transcript, speaker label, clustering, or diarization threshold
is used.  The emitted metadata can select exactly one artifact for later
*annotated* SCD/OSD calibration, never for direct tracklet splitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sddiar.pyannote_segmentation_runtime import PyannoteSegmentationOnnxRuntime  # noqa: E402


FP32_MODEL = ROOT / "artifacts/dev/models/pyannote_segmentation3/model.onnx"
FP32_SHA256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
INT8_MODEL = ROOT / "artifacts/dev/models/pyannote_segmentation3/model.int8.onnx"
INT8_SHA256 = "d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d"
LICENSE_FILE = ROOT / "artifacts/dev/models/pyannote_segmentation3/LICENSE"
README_FILE = ROOT / "artifacts/dev/models/pyannote_segmentation3/README.md"
EXPORT_SCRIPT = ROOT / "artifacts/dev/models/pyannote_segmentation3/export-onnx.py"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/dev/models/pyannote_segmentation3/development_evidence_candidate.json"
)
SEGMENTS = ((0, 60_000_000), (0, 300_000_000))
GATES = {
    "argmax_agreement_min": 0.990,
    "speech_mae_max": 0.010,
    "speech_abs_error_p95_max": 0.030,
    "overlap_mae_max": 0.010,
    "overlap_abs_error_p95_max": 0.030,
    "change_mae_max": 0.020,
    "change_abs_error_p95_max": 0.080,
    "diagnostic_event_f1_min": 0.900,
    "diagnostic_event_collar_us": 250_000,
}


class ParityEvaluationError(RuntimeError):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256_path(path) != expected:
        raise ParityEvaluationError(f"{label} SHA-256 mismatch")


def percentile(values: Sequence[float], quantile: float) -> float:
    import numpy as np

    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def _compact_result(result: Any, trace: Any) -> dict[str, Any]:
    return {
        "summary": result.to_dict(include_frames=False),
        "frames": [
            [
                frame.frame_index,
                frame.center_us,
                frame.speech_probability,
                frame.overlap_probability,
                frame.speaker_change_evidence,
            ]
            for frame in result.frames
        ],
        "events": [[event.time_us, event.evidence] for event in result.diagnostic_change_events],
        "window_argmax": list(trace.window_argmax),
        "frames_per_window": trace.frames_per_window,
        "permutation_sha256": trace.permutation_sha256,
    }


def _worker(args: argparse.Namespace) -> int:
    require_hash(args.model, args.model_sha256, "worker model")
    require_hash(args.audio, args.audio_sha256, "worker approved audio")
    runtime = PyannoteSegmentationOnnxRuntime(
        args.model,
        expected_sha256=args.model_sha256,
    )
    results = []
    for start_us, end_us in SEGMENTS:
        result, trace = runtime.process_wav_with_trace(
            args.audio,
            segment_start_us=start_us,
            segment_end_us=end_us,
        )
        results.append(
            {
                "segment_start_us": start_us,
                "segment_end_us": end_us,
                **_compact_result(result, trace),
            }
        )
    payload = {
        "model_sha256": args.model_sha256,
        "model_bytes": args.model.stat().st_size,
        "model_variant": runtime.model_variant,
        "metadata": dict(sorted(runtime.metadata.items())),
        "segments": results,
        "peak_rss_mb": max(result["summary"]["peak_rss_mb"] for result in results),
        "total_elapsed_wall_sec": sum(result["summary"]["elapsed_wall_sec"] for result in results),
        "total_model_inference_wall_sec": sum(
            result["summary"]["model_inference_wall_sec"] for result in results
        ),
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


def _run_model_worker(
    *,
    model: Path,
    model_sha256: str,
    audio: Path,
    audio_sha256: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(SRC),
            "ORT_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            str(model),
            "--model-sha256",
            model_sha256,
            "--audio",
            str(audio),
            "--audio-sha256",
            audio_sha256,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ParityEvaluationError(
            f"isolated model worker failed ({completed.returncode}): {completed.stderr[-1000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ParityEvaluationError("isolated worker emitted invalid JSON") from exc


def _match_events(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    *,
    collar_us: int,
) -> dict[str, Any]:
    available = set(range(len(candidate)))
    matches: list[int] = []
    for reference_event in reference:
        time_us = int(reference_event[0])
        choices = sorted(
            (
                (abs(time_us - int(candidate[index][0])), index)
                for index in available
                if abs(time_us - int(candidate[index][0])) <= collar_us
            ),
            key=lambda item: (item[0], item[1]),
        )
        if choices:
            drift, index = choices[0]
            available.remove(index)
            matches.append(drift)
    tp = len(matches)
    precision = tp / max(1, len(candidate))
    recall = tp / max(1, len(reference))
    if not reference and not candidate:
        precision = recall = 1.0
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "reference_event_count": len(reference),
        "candidate_event_count": len(candidate),
        "matched_event_count": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time_abs_error_p95_us": percentile(matches, 0.95) if matches else None,
        "collar_us": collar_us,
    }


def _segment_metrics(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    reference_argmax = np.asarray(reference["window_argmax"], dtype=np.int16)
    candidate_argmax = np.asarray(candidate["window_argmax"], dtype=np.int16)
    if reference_argmax.shape != candidate_argmax.shape or reference_argmax.size == 0:
        raise ParityEvaluationError("window argmax traces are not comparable")
    reference_frames = np.asarray(reference["frames"], dtype=np.float64)
    candidate_frames = np.asarray(candidate["frames"], dtype=np.float64)
    if reference_frames.shape != candidate_frames.shape or reference_frames.shape[1] != 5:
        raise ParityEvaluationError("aggregated frame evidence is not comparable")
    if not np.array_equal(reference_frames[:, :2], candidate_frames[:, :2]):
        raise ParityEvaluationError("source-time evidence frame identities differ")
    speech_error = np.abs(reference_frames[:, 2] - candidate_frames[:, 2])
    overlap_error = np.abs(reference_frames[:, 3] - candidate_frames[:, 3])
    change_error = np.abs(reference_frames[:, 4] - candidate_frames[:, 4])
    events = _match_events(
        reference["events"],
        candidate["events"],
        collar_us=int(GATES["diagnostic_event_collar_us"]),
    )
    metrics = {
        "argmax_count": int(reference_argmax.size),
        "argmax_agreement": float(np.mean(reference_argmax == candidate_argmax)),
        "frame_count": int(reference_frames.shape[0]),
        "speech_mae": float(np.mean(speech_error)),
        "speech_abs_error_p95": percentile(speech_error, 0.95),
        "speech_abs_error_max": float(np.max(speech_error)),
        "overlap_mae": float(np.mean(overlap_error)),
        "overlap_abs_error_p95": percentile(overlap_error, 0.95),
        "overlap_abs_error_max": float(np.max(overlap_error)),
        "change_mae": float(np.mean(change_error)),
        "change_abs_error_p95": percentile(change_error, 0.95),
        "change_abs_error_max": float(np.max(change_error)),
        "diagnostic_events": events,
    }
    checks = {
        "argmax_agreement": metrics["argmax_agreement"] >= GATES["argmax_agreement_min"],
        "speech_mae": metrics["speech_mae"] <= GATES["speech_mae_max"],
        "speech_abs_error_p95": (
            metrics["speech_abs_error_p95"] <= GATES["speech_abs_error_p95_max"]
        ),
        "overlap_mae": metrics["overlap_mae"] <= GATES["overlap_mae_max"],
        "overlap_abs_error_p95": (
            metrics["overlap_abs_error_p95"] <= GATES["overlap_abs_error_p95_max"]
        ),
        "change_mae": metrics["change_mae"] <= GATES["change_mae_max"],
        "change_abs_error_p95": (
            metrics["change_abs_error_p95"] <= GATES["change_abs_error_p95_max"]
        ),
        "diagnostic_event_f1": events["f1"] >= GATES["diagnostic_event_f1_min"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def select_evidence_candidate(*, int8_parity_passed: bool, int8_not_slower: bool) -> dict[str, Any]:
    """Keep FP32 as the reference lane when the converted INT8 model fails."""
    if int8_parity_passed and int8_not_slower:
        return {
            "selected_artifact": "int8",
            "status": "INT8_EVIDENCE_CANDIDATE_PARITY_PASS",
            "eligible_for_later_annotated_scd_osd_calibration": True,
        }
    if int8_parity_passed:
        return {
            "selected_artifact": "fp32",
            "status": "FP32_EVIDENCE_CANDIDATE_INT8_NOT_FASTER",
            "eligible_for_later_annotated_scd_osd_calibration": True,
        }
    return {
        "selected_artifact": "fp32",
        "status": "FP32_EVIDENCE_CANDIDATE_INT8_REJECTED",
        "eligible_for_later_annotated_scd_osd_calibration": True,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require_hash(args.fp32_model, args.fp32_sha256, "FP32 model")
    require_hash(args.int8_model, args.int8_sha256, "INT8 model")
    require_hash(args.audio, args.audio_sha256, "approved audio")
    fp32 = _run_model_worker(
        model=args.fp32_model,
        model_sha256=args.fp32_sha256,
        audio=args.audio,
        audio_sha256=args.audio_sha256,
    )
    int8 = _run_model_worker(
        model=args.int8_model,
        model_sha256=args.int8_sha256,
        audio=args.audio,
        audio_sha256=args.audio_sha256,
    )
    comparisons = []
    for reference, candidate in zip(fp32["segments"], int8["segments"]):
        if (
            reference["segment_start_us"],
            reference["segment_end_us"],
        ) != (
            candidate["segment_start_us"],
            candidate["segment_end_us"],
        ):
            raise ParityEvaluationError("worker segment identities differ")
        comparisons.append(
            {
                "segment_start_us": reference["segment_start_us"],
                "segment_end_us": reference["segment_end_us"],
                **_segment_metrics(reference, candidate),
                "cpu_cost": {
                    "fp32": {
                        "elapsed_wall_sec": reference["summary"]["elapsed_wall_sec"],
                        "model_inference_wall_sec": reference["summary"][
                            "model_inference_wall_sec"
                        ],
                        "peak_rss_mb": reference["summary"]["peak_rss_mb"],
                    },
                    "int8": {
                        "elapsed_wall_sec": candidate["summary"]["elapsed_wall_sec"],
                        "model_inference_wall_sec": candidate["summary"][
                            "model_inference_wall_sec"
                        ],
                        "peak_rss_mb": candidate["summary"]["peak_rss_mb"],
                    },
                },
            }
        )
    parity_passed = all(comparison["passed"] for comparison in comparisons)
    int8_not_slower = (
        int8["total_model_inference_wall_sec"] <= fp32["total_model_inference_wall_sec"]
    )
    selection = select_evidence_candidate(
        int8_parity_passed=parity_passed,
        int8_not_slower=int8_not_slower,
    )
    selected = selection["selected_artifact"]
    provenance = {
        "bundle_relative_path": "models/pyannote_segmentation3",
        "source": "https://huggingface.co/pyannote/segmentation-3.0",
        "conversion_maintainer": "k2-fsa/sherpa-onnx",
        "license": "MIT",
        "license_path": "models/pyannote_segmentation3/LICENSE",
        "license_sha256": sha256_path(LICENSE_FILE),
        "readme_sha256": sha256_path(README_FILE),
        "export_script_sha256": sha256_path(EXPORT_SCRIPT),
        "fp32_sha256": args.fp32_sha256,
        "fp32_bytes": args.fp32_model.stat().st_size,
        "int8_sha256": args.int8_sha256,
        "int8_bytes": args.int8_model.stat().st_size,
        "int8_conversion": "onnxruntime.quantization.quantize_dynamic weight_type=QUInt8",
    }
    report = {
        "schema": "sddiar_pyannote_segmentation_parity_v1",
        "candidate_id": (
            f"pyannote-segmentation3-{selected}-development-evidence-20260826"
        ),
        "status": selection["status"],
        "production_approved": False,
        "runtime_eligible_for_default_pipeline": False,
        "eligible_for_later_annotated_scd_osd_calibration": selection[
            "eligible_for_later_annotated_scd_osd_calibration"
        ],
        "selected_artifact": selected,
        "selection_policy": (
            "select INT8 only when all quality parity gates pass and isolated model inference is no slower; "
            "otherwise retain FP32 as the reference evidence candidate and reject INT8"
        ),
        "candidate_conditions": [
            "FP32 eligibility is limited to later annotated SCD/OSD calibration.",
            "Model access terms, MIT notice preservation, weight provenance, and internal redistribution approval remain mandatory.",
            "Default-pipeline runtime, tracklet splitting, and overlap assignment remain prohibited.",
        ],
        "approved_audio_sha256": args.audio_sha256,
        "source_path": "omitted",
        "speaker_labels_used": False,
        "segments": [
            {"start_us": start, "end_us": end, "duration_sec": (end - start) / 1_000_000}
            for start, end in SEGMENTS
        ],
        "gates": GATES,
        "comparisons": comparisons,
        "parity_passed": parity_passed,
        "cpu_cost": {
            "fp32": {
                "total_elapsed_wall_sec": fp32["total_elapsed_wall_sec"],
                "total_model_inference_wall_sec": fp32["total_model_inference_wall_sec"],
                "peak_rss_mb": fp32["peak_rss_mb"],
            },
            "int8": {
                "total_elapsed_wall_sec": int8["total_elapsed_wall_sec"],
                "total_model_inference_wall_sec": int8["total_model_inference_wall_sec"],
                "peak_rss_mb": int8["peak_rss_mb"],
                "inference_wall_ratio_vs_fp32": (
                    int8["total_model_inference_wall_sec"]
                    / max(1e-12, fp32["total_model_inference_wall_sec"])
                ),
            },
            "threads": 1,
            "execution_provider": "CPUExecutionProvider",
            "runtime_network_imports": "none; project static zero-network scan required",
        },
        "model_metadata": {
            "fp32": fp32["metadata"],
            "int8": int8["metadata"],
        },
        "provenance": provenance,
        "evidence_contract": {
            "sample_rate_hz": 16000,
            "window_samples": 160000,
            "window_shift_samples": 16000,
            "receptive_field_samples": 991,
            "receptive_field_shift_samples": 270,
            "powerset_classes": 7,
            "local_speakers": 3,
            "max_active_speakers": 2,
            "outputs": ["speech_probability", "overlap_probability", "speaker_change_evidence"],
            "local_speaker_permutation_aligned": True,
            "edge_padding": "right-zero-pad one bounded tail window; padded-only frames excluded",
            "approved_for_tracklet_split": False,
            "approved_for_overlap_assignment": False,
            "calibration_status": "UNCALIBRATED",
        },
        "limits": [
            "This is model parity on one approved recording, not annotated SCD/OSD accuracy.",
            "No PyTorch-versus-ONNX parity claim was made or tested.",
            "Diagnostic event agreement uses uncalibrated evidence local maxima and cannot justify tracklet cuts.",
            "A fresh file-level Korean annotated calibration/holdout split is required before any SCD/OSD use.",
            "Intel Xeon Gold 6230R cgroup-v1 performance remains unverified.",
        ],
    }
    _atomic_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pyannote segmentation FP32/INT8 evidence parity")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--fp32-model", type=Path, default=FP32_MODEL)
    parser.add_argument("--fp32-sha256", default=FP32_SHA256)
    parser.add_argument("--int8-model", type=Path, default=INT8_MODEL)
    parser.add_argument("--int8-sha256", default=INT8_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if args.model is None or args.model_sha256 is None:
            raise SystemExit("worker requires --model and --model-sha256")
        return _worker(args)
    report = evaluate(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_artifact": report["selected_artifact"],
                "parity_passed": report["parity_passed"],
                "comparisons": report["comparisons"],
                "cpu_cost": report["cpu_cost"],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0 if report["parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
