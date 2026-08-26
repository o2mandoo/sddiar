#!/usr/bin/env python3
"""Run the development diarization benchmark inside an enforced 1-CPU cgroup.

The result contains aggregate timing, cgroup quota/throttling counters, hashes,
and pseudonymous quality metrics only.  It never stores input paths, audio,
transcript text, embeddings, or centroids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_onnx_diarization_experiment import run_experiment  # noqa: E402
from sddiar.runtime_env import delta_cpu_snapshots, read_cpu_snapshot  # noqa: E402
from sddiar.service import atomic_publish  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if len(expected) != 64 or _sha256(path) != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch")


def _delta_dict(delta: Any) -> dict[str, Any]:
    stat = delta.cpu_stat
    return {
        "cgroup_version": delta.cgroup_version,
        "effective_cpu_equivalent": delta.effective_cpu_equivalent,
        "cpuset_cpu_count": delta.cpuset_cpu_count,
        "usage_usec": stat.usage_usec if stat else None,
        "user_usec": stat.user_usec if stat else None,
        "system_usec": stat.system_usec if stat else None,
        "nr_periods": stat.nr_periods if stat else None,
        "nr_throttled": stat.nr_throttled if stat else None,
        "throttled_usec": stat.throttled_usec if stat else None,
        "counter_reset_detected": stat.reset_detected if stat else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline 1-CPU quota diarization benchmark")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--silero-model", required=True, type=Path)
    parser.add_argument("--silero-sha256", required=True)
    parser.add_argument("--wespeaker-model", required=True, type=Path)
    parser.add_argument("--wespeaker-sha256", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--calibrate-assignment", action="store_true")
    parser.add_argument("--silero-temporal-postprocess", action="store_true")
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--embedding-batch-regions", type=int, default=1)
    parser.add_argument("--exact-length-batching", action="store_true")
    parser.add_argument("--batch-buffer-regions", type=int, default=32)
    args = parser.parse_args(argv)

    if args.threads != 1:
        raise ValueError("the 1-CPU benchmark requires exactly one ORT thread")
    _require_hash(args.silero_model, args.silero_sha256, "Silero")
    _require_hash(args.wespeaker_model, args.wespeaker_sha256, "WeSpeaker")

    before = read_cpu_snapshot()
    cpu_equivalent = before.effective_cpu_equivalent
    if cpu_equivalent is None or not math.isclose(cpu_equivalent, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"benchmark requires exactly 1.00 CPU-equivalent; detected {cpu_equivalent!r}")

    result = run_experiment(
        args.audio,
        args.silero_model,
        args.wespeaker_model,
        args.reference,
        threads=1,
        silero_temporal_postprocess=args.silero_temporal_postprocess,
        max_duration_sec=args.max_duration_sec,
        h2_complexity_penalty=0.0,
        h2_min_cost_gain=0.02,
        max_tracklet_sec=3.0,
        subsegment_windows=False,
        embedding_batch_regions=args.embedding_batch_regions,
        exact_length_batching=args.exact_length_batching,
        batch_buffer_regions=args.batch_buffer_regions,
        calibrate_assignment=args.calibrate_assignment,
    )
    after = read_cpu_snapshot()
    cgroup = _delta_dict(delta_cpu_snapshots(before, after))
    elapsed = float(result["elapsed_wall_sec"])
    usage_usec = cgroup.get("usage_usec")
    cgroup["quota_utilization"] = (
        float(usage_usec) / max(1.0, elapsed * 1_000_000.0 * cpu_equivalent)
        if usage_usec is not None else None
    )
    throttled_usec = cgroup.get("throttled_usec")
    cgroup["throttled_wall_ratio"] = (
        float(throttled_usec) / max(1.0, elapsed * 1_000_000.0)
        if throttled_usec is not None else None
    )
    result["cpu_limit"] = {
        **cgroup,
        "quota_us": after.quota_us,
        "period_us": after.period_us,
        "visible_logical_cpu_count": os.cpu_count(),
    }
    result["execution_environment"] = {
        "platform": sys.platform,
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MKL_DYNAMIC",
                "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "KMP_BLOCKTIME",
                "ORT_DISABLE_TELEMETRY",
            )
        },
        "network_policy": "docker-network-none",
        "container_root": "read-only",
    }
    atomic_publish(args.output, result)
    print(json.dumps({
        "decision": result["decision"],
        "rtf": result["rtf"],
        "peak_rss_mb": result["peak_rss_mb"],
        "stage_timings": result["stage_timings"],
        "cpu_limit": result["cpu_limit"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
