"""Local service-boundary adapters.

The reference service is deliberately transport-free: production callers must
inject a GenOS implementation, while local tests can use the filesystem store.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


SUMMARY_BLOCKED = frozenset({"REVIEW_REQUIRED", "UNSUPPORTED", "PASS_WITH_UNATTRIBUTED"})


class SummaryPolicyError(ValueError):
    """Raised when a speaker-aware summary is unsafe for the quality result."""


@dataclass(frozen=True, slots=True)
class SummaryPolicyDecision:
    mode: str
    allowed: bool
    reason_codes: tuple[str, ...] = ()


class SummaryPolicyAdapter:
    """Translate quality status into an explicit summary authorization."""

    def decide(self, quality: Any, requested_mode: str = "AUTO") -> SummaryPolicyDecision:
        status = quality.get("status") if isinstance(quality, Mapping) else getattr(quality, "status", None)
        declared = quality.get("summary_mode") if isinstance(quality, Mapping) else getattr(quality, "summary_mode", None)
        requested = requested_mode.upper()
        if requested not in {"AUTO", "SPEAKER_AWARE", "SPEAKER_NEUTRAL", "MANUAL_REVIEW"}:
            raise SummaryPolicyError(f"unknown summary mode: {requested_mode}")
        if status in SUMMARY_BLOCKED:
            if requested == "SPEAKER_AWARE" or declared == "SPEAKER_AWARE":
                raise SummaryPolicyError(f"speaker-aware summary blocked for {status}")
            mode = "MANUAL_REVIEW" if status == "REVIEW_REQUIRED" else "SPEAKER_NEUTRAL"
            # A caller may explicitly choose the safer neutral/manual path;
            # only speaker-aware output is forbidden by this adapter.
            if requested not in {"AUTO", "SPEAKER_NEUTRAL", "MANUAL_REVIEW"}:
                raise SummaryPolicyError(f"summary mode {requested} blocked for {status}")
            return SummaryPolicyDecision(mode if requested == "AUTO" else requested, True, (f"SUMMARY_{status}",))
        mode = declared or "SPEAKER_AWARE"
        if requested != "AUTO": mode = requested
        return SummaryPolicyDecision(mode, mode in {"SPEAKER_AWARE", "SPEAKER_NEUTRAL", "MANUAL_REVIEW"})

    def authorize(self, quality: Any, requested_mode: str = "AUTO") -> str:
        return self.decide(quality, requested_mode).mode


@runtime_checkable
class GenOSService(Protocol):
    """Caller-injected production boundary; no network implementation here."""
    def submit(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def status(self, job_id: str) -> Mapping[str, Any]: ...


class GenOSNotConfigured(RuntimeError):
    pass


class GenOSServiceStub:
    def submit(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise GenOSNotConfigured("inject a GenOSService implementation; local adapter never calls the network")
    def status(self, job_id: str) -> Mapping[str, Any]:
        raise GenOSNotConfigured("inject a GenOSService implementation; local adapter never calls the network")


def idempotency_key(
    source_sha256: str,
    profile_id: str,
    options: Mapping[str, Any] | None = None,
    *,
    pipeline_version: str = "",
    model_pack_id: str = "",
    calibration_profile_id: str = "",
    stt_backend_version: str = "",
) -> str:
    """Stable key for the full SDD idempotency contract.

    Callers that have not selected a model/STT version may leave the optional
    fields empty during local reference testing. Production adapters must pass
    every version so a changed model or calibration never reuses old output.
    """
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character SHA-256")
    payload = json.dumps({
        "source_sha256": source_sha256.lower(), "profile_id": profile_id,
        "pipeline_version": pipeline_version, "model_pack_id": model_pack_id,
        "calibration_profile_id": calibration_profile_id,
        "stt_backend_version": stt_backend_version, "options": options or {},
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_publish(path: str | os.PathLike[str], value: Any) -> Path:
    """Publish JSON using same-directory temp file and atomic replace."""
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
    return target


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
