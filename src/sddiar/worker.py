"""Typed, local, idempotent job state machine for P6."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import ContractValidationError, ModelPackError, OfflinePolicyViolation
from .service import atomic_publish, idempotency_key, read_json


class JobStatus(str, Enum):
    RECEIVED = "RECEIVED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"


def _now() -> str: return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class LocalJob:
    job_id: str
    idempotency_key: str
    source_ref: str
    profile_id: str
    options: Mapping[str, Any]
    pipeline_version: str = ""
    model_pack_id: str = ""
    calibration_profile_id: str = ""
    stt_backend_version: str = ""
    status: JobStatus = JobStatus.RECEIVED
    attempts: int = 0
    result_path: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str = ""
    updated_at: str = ""


def classify_retry(exc: BaseException) -> bool:
    """Return true only for infrastructure/transient failures."""
    if isinstance(exc, (ContractValidationError, ModelPackError, OfflinePolicyViolation, ValueError, TypeError, KeyError)):
        return False
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


class LocalJobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root); self.jobs = self.root / "jobs"; self.results = self.root / "results"
        self.jobs.mkdir(parents=True, exist_ok=True); self.results.mkdir(parents=True, exist_ok=True)
    def _path(self, job_id: str) -> Path: return self.jobs / f"{job_id}.json"
    def save(self, job: LocalJob) -> LocalJob:
        atomic_publish(self._path(job.job_id), {**asdict(job), "status": job.status.value})
        return job
    def get(self, job_id: str) -> LocalJob:
        data = read_json(self._path(job_id)); data["status"] = JobStatus(data["status"]); return LocalJob(**data)
    def find_idempotency(self, key: str) -> LocalJob | None:
        for path in self.jobs.glob("*.json"):
            try:
                job = self.get(path.stem)
                if job.idempotency_key == key: return job
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return None
    def list(self) -> tuple[LocalJob, ...]: return tuple(self.get(p.stem) for p in sorted(self.jobs.glob("*.json")))


class LocalWorker:
    def __init__(self, store: LocalJobStore, max_attempts: int = 3):
        if max_attempts < 1: raise ValueError("max_attempts must be positive")
        self.store, self.max_attempts = store, max_attempts
    def submit(self, *, job_id: str, source_ref: str, source_sha256: str, profile_id: str,
               options: Mapping[str, Any] | None = None, pipeline_version: str = "",
               model_pack_id: str = "", calibration_profile_id: str = "",
               stt_backend_version: str = "") -> LocalJob:
        options = dict(options or {})
        key = idempotency_key(
            source_sha256, profile_id, options,
            pipeline_version=pipeline_version, model_pack_id=model_pack_id,
            calibration_profile_id=calibration_profile_id,
            stt_backend_version=stt_backend_version,
        )
        existing = self.store.find_idempotency(key)
        if existing: return existing
        now = _now()
        return self.store.save(LocalJob(
            job_id, key, source_ref, profile_id, options,
            pipeline_version, model_pack_id, calibration_profile_id,
            stt_backend_version, created_at=now, updated_at=now,
        ))
    def run(self, job_id: str, handler: Callable[[LocalJob], Any]) -> LocalJob:
        job = self.store.get(job_id)
        if job.status == JobStatus.SUCCEEDED: return job
        if job.status == JobStatus.FAILED_TERMINAL: return job
        running = self.store.save(replace(job, status=JobStatus.RUNNING, attempts=job.attempts + 1, updated_at=_now()))
        try:
            result = handler(running)
            output = self.store.results / f"{job_id}.json"
            atomic_publish(output, result)
            return self.store.save(replace(running, status=JobStatus.SUCCEEDED, result_path=str(output), error_code=None, error_message=None, updated_at=_now()))
        except BaseException as exc:
            retry = classify_retry(exc) and running.attempts < self.max_attempts
            status = JobStatus.FAILED_RETRYABLE if retry else JobStatus.FAILED_TERMINAL
            return self.store.save(replace(running, status=status, error_code=getattr(exc, "code", type(exc).__name__), error_message=str(exc), updated_at=_now()))
    def status(self, job_id: str) -> LocalJob: return self.store.get(job_id)
