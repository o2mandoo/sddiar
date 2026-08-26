"""Fail-closed, offline RNNoise preprocessing experiment.

This module is deliberately not wired into the default diarization path.  It
prepares one job-scoped PCM16-mono 16 kHz WAV so the *same* enhanced waveform
can be consumed by VAD and region-based embedding reads.  Only caller-supplied,
hash-verified local executables are invoked, always as argv sequences.

The upstream ``rnnoise_demo`` executable consumes raw native-endian PCM at
48 kHz.  RNNoise emits the previous delayed frame; the demo discards the first
all-zero warm-up output.  The adapter therefore appends one zero *input* frame
to flush the last real/padded frame, then trims the padded tail to the exact
source sample count without shifting the signal.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
import threading
import wave
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol

from .errors import SDDiarError
from .media import DecodedAudioChunk, WavPcmAccessor, WavPcmDecoder
from .offline import reject_url


RNNOISE_SOURCE_REPOSITORY = "https://gitlab.xiph.org/xiph/rnnoise.git"
RNNOISE_SOURCE_COMMIT = "70f1d256acd4b34a572f999a05c87bf00b67730d"
RNNOISE_MODEL_TAR_SHA256 = "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"
RNNOISE_LICENSE_SPDX = "BSD-3-Clause"
RNNOISE_SAMPLE_RATE_HZ = 48_000
SOURCE_SAMPLE_RATE_HZ = 16_000
RNNOISE_FRAME_SAMPLES = 480
RNNOISE_DEMO_WARMUP_OUTPUT_SAMPLES = 480
RNNOISE_DEMO_FLUSH_INPUT_SAMPLES = 480
_SHA256_LENGTH = 64
_COPY_CHUNK_BYTES = 1 << 20
_GLOBAL_JOB_GATE = threading.BoundedSemaphore(1)
_RECEIPT_VALIDATION_TOKEN = object()


class RNNoiseExperimentalError(SDDiarError):
    """Base failure for the non-production enhancement lane."""

    code = "RNNOISE_EXPERIMENT_FAILED"


class RNNoiseConfigurationError(RNNoiseExperimentalError):
    code = "RNNOISE_CONFIGURATION_INVALID"


class RNNoiseArtifactError(RNNoiseExperimentalError):
    code = "RNNOISE_ARTIFACT_INVALID"


class RNNoiseExecutionError(RNNoiseExperimentalError):
    code = "RNNOISE_EXECUTION_FAILED"


class RNNoiseTimebaseError(RNNoiseExperimentalError):
    code = "RNNOISE_TIMEBASE_INVARIANT_VIOLATION"


class RNNoiseBusyError(RNNoiseExperimentalError):
    code = "RNNOISE_JOB_BUSY"


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_COPY_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _duration_us(frame_count: int, sample_rate_hz: int) -> int:
    return (frame_count * 1_000_000 + sample_rate_hz // 2) // sample_rate_hz


def _frozen_map(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class RNNoiseEnhancementPolicy:
    """Bounded policy; omission is exactly equivalent to disabled."""

    enabled: bool = False
    max_source_bytes: int = 512 * 1024 * 1024
    max_duration_us: int = 4 * 60 * 60 * 1_000_000
    max_workspace_bytes: int = 6 * 1024 * 1024 * 1024
    max_output_bytes: int = 512 * 1024 * 1024 + 4096
    max_native_binary_bytes: int = 64 * 1024 * 1024
    stage_timeout_seconds: float = 1_800.0
    queue_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise RNNoiseConfigurationError("RNNoise enabled must be boolean")
        for name in (
            "max_source_bytes",
            "max_duration_us",
            "max_workspace_bytes",
            "max_output_bytes",
            "max_native_binary_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise RNNoiseConfigurationError("RNNoise byte and duration limits must be positive integers")
        for name in ("stage_timeout_seconds", "queue_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RNNoiseConfigurationError("RNNoise timeouts must be finite and positive")

    def public_identity(self) -> Mapping[str, Any]:
        return _frozen_map(
            {
                "policy_version": "rnnoise-experimental-v1",
                "enabled": self.enabled,
                "source_sample_rate_hz": SOURCE_SAMPLE_RATE_HZ,
                "rnnoise_sample_rate_hz": RNNOISE_SAMPLE_RATE_HZ,
                "rnnoise_frame_samples": RNNOISE_FRAME_SAMPLES,
                "demo_warmup_output_samples_48k": RNNOISE_DEMO_WARMUP_OUTPUT_SAMPLES,
                "demo_flush_input_samples_48k": RNNOISE_DEMO_FLUSH_INPUT_SAMPLES,
                "max_source_bytes": self.max_source_bytes,
                "max_duration_us": self.max_duration_us,
                "max_workspace_bytes": self.max_workspace_bytes,
                "max_output_bytes": self.max_output_bytes,
                "max_native_binary_bytes": self.max_native_binary_bytes,
                "stage_timeout_seconds": float(self.stage_timeout_seconds),
                "queue_timeout_seconds": float(self.queue_timeout_seconds),
            }
        )


@dataclass(frozen=True, slots=True)
class NativeInvocation:
    """Structured subprocess request; command strings are not accepted."""

    stage: str
    argv: tuple[str, ...]
    cwd: Path
    expected_output: Path
    expected_output_bytes: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if self.stage not in {"UPSAMPLE_48K", "RNNOISE_48K", "DOWNSAMPLE_16K"}:
            raise RNNoiseConfigurationError("unknown RNNoise native stage")
        if not self.argv or not all(isinstance(item, str) and item for item in self.argv):
            raise RNNoiseConfigurationError("native invocation requires a non-empty argv tuple")
        if not Path(self.argv[0]).is_absolute():
            raise RNNoiseConfigurationError("native executable path must be absolute")
        if type(self.expected_output_bytes) is not int or self.expected_output_bytes < 0:
            raise RNNoiseConfigurationError("invalid native output bound")


@dataclass(frozen=True, slots=True)
class NativeRunOutcome:
    stage: str
    returncode: int


class NativeRunner(Protocol):
    def run(self, invocation: NativeInvocation) -> NativeRunOutcome: ...


class SubprocessArgvRunner:
    """Minimal-env runner with no shell, stdin, output capture, or PATH lookup."""

    @staticmethod
    def _environment(cwd: Path) -> dict[str, str]:
        environment = {
            "LC_ALL": "C",
            "LANG": "C",
        }
        if os.name == "nt":
            for key in ("SYSTEMROOT", "WINDIR"):
                value = os.environ.get(key)
                if value:
                    environment[key] = value
            environment["TEMP"] = str(cwd)
            environment["TMP"] = str(cwd)
        else:
            environment["TMPDIR"] = str(cwd)
        return environment

    def run(self, invocation: NativeInvocation) -> NativeRunOutcome:
        try:
            completed = subprocess.run(
                list(invocation.argv),
                cwd=str(invocation.cwd),
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=invocation.timeout_seconds,
                env=self._environment(invocation.cwd),
                close_fds=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RNNoiseExecutionError("RNNoise native stage timed out") from exc
        except OSError as exc:
            raise RNNoiseExecutionError("RNNoise native stage could not start") from exc
        return NativeRunOutcome(invocation.stage, int(completed.returncode))


@dataclass(frozen=True, slots=True)
class RNNoiseReceipt:
    """Allowlisted receipt containing no paths, argv, logs, PCM, or transcript."""

    status: str
    source_sha256_prefix: str | None
    output_sha256_prefix: str | None
    source_frame_count: int | None
    output_frame_count: int | None
    duration_us: int | None
    rnnoise_binary_sha256: str | None
    resampler_binary_sha256: str | None
    build_attestation_sha256: str | None
    timebase_proof_sha256: str | None
    runner_kind: str
    policy_sha256: str
    stage_outcomes: tuple[NativeRunOutcome, ...] = ()
    _validation_token: object | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        applied = (
            self.status == "EXPERIMENTAL_APPLIED"
            and self._validation_token is _RECEIPT_VALIDATION_TOKEN
        )
        return {
            "schema_version": "1.0",
            "kind": "rnnoise-experimental-receipt",
            "experimental": True,
            "production_approved": False,
            "default_enabled": False,
            "status": self.status,
            "source_id": self.source_sha256_prefix,
            "output_id": self.output_sha256_prefix,
            "source_frame_count": self.source_frame_count,
            "output_frame_count": self.output_frame_count,
            "duration_us": self.duration_us,
            "sample_rates_hz": {"source": SOURCE_SAMPLE_RATE_HZ, "rnnoise": RNNOISE_SAMPLE_RATE_HZ},
            "timebase": {
                "public_mapping": "SAMPLE_COUNT_PRESERVED_LATENCY_NOT_INDEPENDENTLY_VERIFIED"
                if applied and self.timebase_proof_sha256 is not None
                else "NOT_APPLIED",
                "sample_count_preserved": self.source_frame_count == self.output_frame_count
                if self.source_frame_count is not None
                else None,
                "discarded_demo_warmup_output_samples_48k": RNNOISE_DEMO_WARMUP_OUTPUT_SAMPLES
                if applied
                else 0,
                "flush_input_samples_48k": RNNOISE_DEMO_FLUSH_INPUT_SAMPLES if applied else 0,
                "compensation": "APPEND_ZERO_FLUSH_FRAME_THEN_EXACT_TRIM" if applied else "NOT_APPLIED",
                "unrecoverable_initial_us": 0,
                "evidence_status": "CALLER_HASH_BOUND_STRUCTURAL_RECORD"
                if applied and self.timebase_proof_sha256 is not None
                else "NOT_APPLIED",
                "timebase_proof_sha256": self.timebase_proof_sha256,
                "production_authority": False,
            },
            "artifacts": {
                "rnnoise_binary_sha256": self.rnnoise_binary_sha256,
                "resampler_binary_sha256": self.resampler_binary_sha256,
                "build_attestation_sha256": self.build_attestation_sha256,
                "lineage_status": "CALLER_HASH_BOUND_STRUCTURAL_ATTESTATION"
                if applied and self.build_attestation_sha256 is not None
                else "NOT_APPLIED",
                "required_rnnoise_source_commit": RNNOISE_SOURCE_COMMIT,
                "required_rnnoise_source_repository": RNNOISE_SOURCE_REPOSITORY,
                "required_rnnoise_model_tar_sha256": RNNOISE_MODEL_TAR_SHA256,
                "required_source_license_spdx": RNNOISE_LICENSE_SPDX,
            },
            "policy_sha256": self.policy_sha256,
            "stages": [
                {"stage": outcome.stage, "returncode": outcome.returncode}
                for outcome in self.stage_outcomes
            ],
            "execution_policy": {
                "python_adapter_network_or_download": False,
                "native_child_egress": "NOT_VERIFIED_REQUIRES_EXTERNAL_SANDBOX" if applied else "NOT_APPLIED",
                "native_child_process_tree_confinement": "NOT_VERIFIED_REQUIRES_EXTERNAL_SANDBOX"
                if applied
                else "NOT_APPLIED",
                "orchestrator_invocation_shape": "argv_tuple",
                "runner_kind": self.runner_kind,
                "default_runner_shell": False
                if applied and self.runner_kind == "DEFAULT_SUBPROCESS_ARGV"
                else ("NOT_VERIFIED" if applied else "NOT_APPLIED"),
                "runner_execution_internals": "NO_SHELL_MINIMAL_ENV"
                if applied and self.runner_kind == "DEFAULT_SUBPROCESS_ARGV"
                else ("INJECTED_NOT_VERIFIED" if applied else "NOT_APPLIED"),
                "job_concurrency": 1,
                "temporary_output_retained": False,
            },
            "redaction": {
                "source_path": "omitted",
                "artifact_paths": "omitted",
                "temporary_paths": "omitted",
                "argv": "omitted",
                "stdout_stderr": "omitted",
                "audio_samples": "omitted",
                "transcript": "omitted",
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )


@dataclass(frozen=True, slots=True)
class PreparedRNNoiseAudio:
    """Internal job-scoped waveform plus the original source identity."""

    _path: Path
    source_sha256: str | None
    output_sha256: str | None
    frame_count: int | None
    duration_us: int | None
    enhanced: bool
    source_time_authorized: bool
    receipt: RNNoiseReceipt

    @property
    def local_path(self) -> Path:
        """Internal-only path.  Never copy this value into a public result."""

        return self._path

    def iter_chunks(self, *, frames_per_chunk: int = 240_000) -> Iterator[DecodedAudioChunk]:
        return WavPcmDecoder().iter_decode_chunks(self._path, frames_per_chunk=frames_per_chunk)

    def iter_chunks_numpy(self, *, frames_per_chunk: int = 240_000) -> Iterator[DecodedAudioChunk]:
        return WavPcmDecoder().iter_decode_chunks_numpy(self._path, frames_per_chunk=frames_per_chunk)

    def read_mono_samples(self, start_us: int, end_us: int) -> tuple[float, ...]:
        return WavPcmAccessor(self._path).read_mono_samples(start_us, end_us)

    def read_mono_samples_numpy(self, start_us: int, end_us: int) -> Any:
        return WavPcmAccessor(self._path).read_mono_samples_numpy(start_us, end_us)


def _policy_sha256(
    policy: RNNoiseEnhancementPolicy,
    rnnoise_sha: str | None,
    resampler_sha: str | None,
    build_attestation_sha: str | None = None,
    timebase_proof_sha: str | None = None,
    runner_kind: str = "DISABLED",
) -> str:
    payload = {
        **dict(policy.public_identity()),
        "rnnoise_binary_sha256": rnnoise_sha,
        "resampler_binary_sha256": resampler_sha,
        "build_attestation_sha256": build_attestation_sha,
        "timebase_proof_sha256": timebase_proof_sha,
        "runner_kind": runner_kind,
        "required_rnnoise_source_commit": RNNOISE_SOURCE_COMMIT,
        "required_rnnoise_source_repository": RNNOISE_SOURCE_REPOSITORY,
        "required_rnnoise_model_tar_sha256": RNNOISE_MODEL_TAR_SHA256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _secure_root(value: str | os.PathLike[str] | None, *, label: str) -> Path:
    if value is None:
        raise RNNoiseConfigurationError(f"explicit {label} root is required when RNNoise is enabled")
    reject_url(value)
    supplied = Path(value)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise RNNoiseConfigurationError(f"{label} root must be an absolute non-symlink directory")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RNNoiseConfigurationError(f"{label} root is unavailable") from exc
    if not resolved.is_dir():
        raise RNNoiseConfigurationError(f"{label} root is not a directory")
    return resolved


def _reject_symlink_components(root: Path, supplied: Path) -> None:
    # Find the caller's lexical spelling of the approved root.  This avoids
    # rejecting OS-managed aliases above it (for example macOS /var), while a
    # symlink introduced anywhere *below* that trust root remains visible.
    lexical_root: Path | None = None
    for candidate in (supplied.parent, *supplied.parents):
        try:
            if candidate.resolve(strict=True) == root:
                lexical_root = candidate
                break
        except OSError:
            continue
    if lexical_root is None:
        raise RNNoiseArtifactError("RNNoise path has no approved lexical root")
    relative = supplied.relative_to(lexical_root)
    current = lexical_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RNNoiseArtifactError("RNNoise paths may not contain symlinks")


def _secure_regular_file(value: str | os.PathLike[str] | None, *, root: Path, role: str) -> Path:
    if value is None:
        raise RNNoiseArtifactError(f"explicit local {role} is required")
    reject_url(value)
    supplied = Path(value)
    if not supplied.is_absolute():
        raise RNNoiseArtifactError(f"{role} path must be absolute")
    if ".." in supplied.parts:
        raise RNNoiseArtifactError(f"{role} path may not contain traversal")
    if supplied.is_symlink():
        raise RNNoiseArtifactError(f"local {role} may not be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RNNoiseArtifactError(f"local {role} is unavailable") from exc
    if resolved != root and root not in resolved.parents:
        raise RNNoiseArtifactError(f"local {role} escapes its approved root")
    _reject_symlink_components(root, supplied)
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise RNNoiseArtifactError(f"local {role} is unavailable") from exc
    if not stat.S_ISREG(mode):
        raise RNNoiseArtifactError(f"local {role} must be a regular file")
    return resolved


def _verified_executable(
    value: str | os.PathLike[str] | None,
    expected_sha256: str | None,
    *,
    root: Path,
    role: str,
) -> Path:
    if not _valid_sha256(expected_sha256):
        raise RNNoiseArtifactError(f"{role} requires an exact lowercase SHA-256")
    path = _secure_regular_file(value, root=root, role=role)
    if os.name != "nt" and not os.access(path, os.X_OK):
        raise RNNoiseArtifactError(f"local {role} is not executable")
    if _sha256(path) != expected_sha256:
        raise RNNoiseArtifactError(f"local {role} SHA-256 mismatch")
    return path


def _verified_json_artifact(
    value: str | os.PathLike[str] | None,
    expected_sha256: str | None,
    *,
    root: Path,
    role: str,
) -> tuple[Path, Mapping[str, Any]]:
    if not _valid_sha256(expected_sha256):
        raise RNNoiseArtifactError(f"{role} requires an exact lowercase SHA-256")
    path = _secure_regular_file(value, root=root, role=role)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > 1024 * 1024:
                raise RNNoiseArtifactError(f"{role} exceeds its JSON byte bound")
            raw = stream.read(1024 * 1024 + 1)
    except RNNoiseArtifactError:
        raise
    except OSError as exc:
        raise RNNoiseArtifactError(f"{role} could not be opened safely") from exc
    if len(raw) != info.st_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RNNoiseArtifactError(f"{role} SHA-256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RNNoiseArtifactError(f"{role} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RNNoiseArtifactError(f"{role} must contain a JSON object")
    return path, payload


def _canonical_payload_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _descriptor_valid(value: Any, *, expected_sha256: str | None = None) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("logical_name"), str)
        and bool(value.get("logical_name"))
        and Path(value["logical_name"]).name == value["logical_name"]
        and "\\" not in value["logical_name"]
        and _valid_sha256(value.get("sha256"))
        and (expected_sha256 is None or value.get("sha256") == expected_sha256)
        and type(value.get("bytes")) is int
        and value.get("bytes") > 0
    )


def _expected_build_commands(target_id: str) -> list[list[str]]:
    configure = ["sh", "./configure", "--host=x86_64-w64-mingw32"] if target_id == "windows-x86_64" else ["./configure"]
    configure.extend(
        [
            "--disable-doc",
            "--enable-examples",
            "--disable-x86-rtcd",
            "--disable-shared",
            "--enable-static",
        ]
    )
    binary = "rnnoise_demo.exe" if target_id == "windows-x86_64" else "rnnoise_demo"
    return [["autoreconf", "-isf"], configure, ["make", "-j1", f"examples/{binary}"]]


def _validate_build_attestation(payload: Mapping[str, Any], *, rnnoise_sha256: str) -> None:
    source = payload.get("source")
    model = payload.get("model")
    target = payload.get("target")
    configuration = payload.get("configuration")
    native = payload.get("native_binary")
    validation = payload.get("validation")
    submodules = payload.get("submodules")
    toolchain = payload.get("toolchain")
    host = payload.get("host")
    build_log = payload.get("build_log")
    dependency_report = payload.get("dependency_report")
    integrity = payload.get("integrity")
    system = platform.system()
    machine = platform.machine().lower()
    runtime_target = {
        ("Windows", "amd64"): "windows-x86_64",
        ("Windows", "x86_64"): "windows-x86_64",
        ("Linux", "x86_64"): "linux-x86_64",
        ("Linux", "amd64"): "linux-x86_64",
        ("Linux", "aarch64"): "linux-aarch64",
        ("Linux", "arm64"): "linux-aarch64",
        ("Darwin", "x86_64"): "macos-x86_64",
        ("Darwin", "amd64"): "macos-x86_64",
        ("Darwin", "arm64"): "macos-arm64",
        ("Darwin", "aarch64"): "macos-arm64",
    }.get((system, machine))
    if runtime_target is None:
        raise RNNoiseArtifactError("RNNoise runtime target is unsupported")
    target_roles = {
        "windows-x86_64": ["c_compiler", "mingw_w64", "posix_shell", "autoconf", "automake", "libtool", "make"],
        "linux-x86_64": ["c_compiler", "sysroot", "autoconf", "automake", "libtool", "make"],
        "linux-aarch64": ["c_compiler", "sysroot", "autoconf", "automake", "libtool", "make"],
        "macos-x86_64": ["c_compiler", "macos_sdk", "autoconf", "automake", "libtool", "make"],
        "macos-arm64": ["c_compiler", "macos_sdk", "autoconf", "automake", "libtool", "make"],
    }[runtime_target]
    source_archive = source.get("source_archive") if isinstance(source, Mapping) else None
    model_archive = model.get("archive") if isinstance(model, Mapping) else None
    model_files = model.get("staged_files") if isinstance(model, Mapping) else None
    submodule_records = submodules.get("records") if isinstance(submodules, Mapping) else None
    native_name = "rnnoise_demo.exe" if runtime_target == "windows-x86_64" else "rnnoise_demo"
    model_names = {
        record.get("relative_path")
        for record in model_files
        if isinstance(model_files, list)
        and isinstance(record, Mapping)
        and _valid_sha256(record.get("sha256"))
        and type(record.get("bytes")) is int
        and record.get("bytes") > 0
    } if isinstance(model_files, list) else set()
    submodule_records_valid = isinstance(submodule_records, list) and all(
        isinstance(record, Mapping)
        and record.get("state") == " "
        and isinstance(record.get("commit"), str)
        and len(record["commit"]) == 40
        and all(character in "0123456789abcdef" for character in record["commit"])
        and isinstance(record.get("path"), str)
        and bool(record.get("path"))
        and "\\" not in record["path"]
        and not PurePath(record["path"]).is_absolute()
        and ".." not in PurePath(record["path"]).parts
        for record in submodule_records
    )
    integrity_inputs = {
        key: payload.get(key)
        for key in ("source", "submodules", "model", "target", "host", "toolchain", "configuration")
    }
    integrity_outputs = {
        key: payload.get(key) for key in ("native_binary", "build_log", "dependency_report")
    }
    inputs_sha = _canonical_payload_sha256(integrity_inputs)
    outputs_sha = _canonical_payload_sha256(integrity_outputs)
    statement_sha = _canonical_payload_sha256(
        {
            "schema_version": payload.get("schema_version"),
            "kind": payload.get("kind"),
            "inputs_sha256": inputs_sha,
            "outputs_sha256": outputs_sha,
            "validation": validation,
        }
    )
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("kind") != "rnnoise-offline-build-attestation"
        or payload.get("experimental") is not True
        or payload.get("default_enabled") is not False
        or payload.get("production_approved") is not False
        or not isinstance(source, Mapping)
        or source.get("repository") != RNNOISE_SOURCE_REPOSITORY
        or source.get("expected_commit") != RNNOISE_SOURCE_COMMIT
        or source.get("checkout_commit") != RNNOISE_SOURCE_COMMIT
        or source.get("license_spdx") != RNNOISE_LICENSE_SPDX
        or not _descriptor_valid(source_archive)
        or not _valid_sha256(source.get("tracked_tree_sha256"))
        or not isinstance(submodules, Mapping)
        or not submodule_records_valid
        or submodules.get("count") != len(submodule_records)
        or submodules.get("canonical_sha256") != _canonical_payload_sha256(submodule_records)
        or not isinstance(model, Mapping)
        or model.get("expected_sha256") != RNNOISE_MODEL_TAR_SHA256
        or model.get("model_version") != RNNOISE_MODEL_TAR_SHA256
        or not _descriptor_valid(model_archive, expected_sha256=RNNOISE_MODEL_TAR_SHA256)
        or model.get("imported_offline") is not True
        or not {"src/rnnoise_data.c", "src/rnnoise_data.h"}.issubset(model_names)
        or not isinstance(target, Mapping)
        or target.get("id") != runtime_target
        or target.get("endianness") != "little"
        or not isinstance(host, Mapping)
        or host.get("system") != system
        or str(host.get("machine", "")).lower() != machine
        or host.get("matches_target") is not True
        or not isinstance(toolchain, Mapping)
        or not _descriptor_valid(toolchain.get("manifest"))
        or not _valid_sha256(toolchain.get("manifest_payload_sha256"))
        or toolchain.get("required_roles") != target_roles
        or not isinstance(configuration, Mapping)
        or configuration.get("commands") != _expected_build_commands(runtime_target)
        or configuration.get("autogen_executed") is not False
        or configuration.get("download_model_executed") is not False
        or configuration.get("build_network_required_state") != "disabled"
        or configuration.get("x86_rtcd") is not False
        or configuration.get("compile_time_vectorization")
        != "target_compiler_default_not_scalar_claim"
        or configuration.get("jobs") != 1
        or not _descriptor_valid(native, expected_sha256=rnnoise_sha256)
        or native.get("logical_name") != native_name
        or not _descriptor_valid(build_log)
        or not _descriptor_valid(dependency_report)
        or not isinstance(validation, Mapping)
        or validation.get("single_target_native_build_recorded") is not True
        or validation.get("binary_functional_smoke") != "not_run"
        or validation.get("four_platform_native_validation") != "not_run"
        or validation.get("xeon_validation") != "not_run"
        or validation.get("independent_review") != "not_run"
        or validation.get("release_authority") != "none"
        or not isinstance(integrity, Mapping)
        or integrity.get("build_inputs_sha256") != inputs_sha
        or integrity.get("build_outputs_sha256") != outputs_sha
        or integrity.get("statement_sha256") != statement_sha
        or integrity.get("signature_status") != "unsigned"
        or integrity.get("verification_scope") != "hash_bound_structure_only"
        or integrity.get("cryptographic_authenticity") != "not_verified"
    ):
        raise RNNoiseArtifactError("RNNoise build attestation is incomplete or mismatched")


_TIMEBASE_FRAME_CASES = (1, 159, 160, 161, 479, 480, 481, 16_003)


def _validate_timebase_proof(
    payload: Mapping[str, Any], *, rnnoise_sha256: str, resampler_sha256: str
) -> None:
    artifacts = payload.get("artifacts")
    contract = payload.get("contract")
    review = payload.get("review")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("kind") != "rnnoise-resampler-timebase-proof"
        or payload.get("experimental") is not True
        or payload.get("production_approved") is not False
        or not isinstance(artifacts, Mapping)
        or artifacts.get("rnnoise_binary_sha256") != rnnoise_sha256
        or artifacts.get("resampler_binary_sha256") != resampler_sha256
        or not isinstance(contract, Mapping)
        or contract.get("argv_contract_version") != "raw-s16le-exact-v1"
        or contract.get("source_sample_rate_hz") != SOURCE_SAMPLE_RATE_HZ
        or contract.get("rnnoise_sample_rate_hz") != RNNOISE_SAMPLE_RATE_HZ
        or contract.get("demo_warmup_output_samples_48k") != RNNOISE_DEMO_WARMUP_OUTPUT_SAMPLES
        or contract.get("demo_flush_input_samples_48k") != RNNOISE_DEMO_FLUSH_INPUT_SAMPLES
        or type(contract.get("roundtrip_impulse_peak_shift_samples_16k")) is not int
        or contract.get("roundtrip_impulse_peak_shift_samples_16k") != 0
        or type(contract.get("first_marker_shift_samples_16k")) is not int
        or contract.get("first_marker_shift_samples_16k") != 0
        or type(contract.get("last_marker_shift_samples_16k")) is not int
        or contract.get("last_marker_shift_samples_16k") != 0
        or contract.get("exact_duration_frame_cases") != list(_TIMEBASE_FRAME_CASES)
        or contract.get("all_frame_cases_exact") is not True
        or not isinstance(review, Mapping)
        or review.get("authority") != "CALLER_HASH_BOUND_EXPERIMENTAL"
        or review.get("release_authority") != "none"
    ):
        raise RNNoiseTimebaseError("RNNoise timebase proof is incomplete or mismatched")


def _copy_exact(source: Path, destination: Path, *, offset: int, byte_count: int) -> None:
    remaining = byte_count
    with source.open("rb") as reader, destination.open("xb") as writer:
        reader.seek(offset)
        while remaining:
            block = reader.read(min(_COPY_CHUNK_BYTES, remaining))
            if not block:
                raise RNNoiseExecutionError("audio source changed during bounded copy")
            writer.write(block)
            remaining -= len(block)
    if os.name != "nt":
        destination.chmod(0o600)


def _copy_verified_source_snapshot(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> tuple[int, int, int, int]:
    """Copy/hash one no-follow descriptor so later reads cannot race the path."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as reader, destination.open("xb") as writer:
            before = os.fstat(reader.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > max_bytes:
                raise RNNoiseArtifactError("audio source exceeds the bounded regular-file contract")
            copied = 0
            while True:
                block = reader.read(_COPY_CHUNK_BYTES)
                if not block:
                    break
                copied += len(block)
                if copied > max_bytes:
                    raise RNNoiseConfigurationError("audio source exceeds the RNNoise byte limit")
                digest.update(block)
                writer.write(block)
            after = os.fstat(reader.fileno())
    except (RNNoiseArtifactError, RNNoiseConfigurationError):
        raise
    except OSError as exc:
        raise RNNoiseArtifactError("audio source could not be snapshotted safely") from exc
    if (
        copied != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or digest.hexdigest() != expected_sha256
    ):
        raise RNNoiseArtifactError("audio source SHA-256 or descriptor identity mismatch")
    if os.name != "nt":
        destination.chmod(0o600)
    return before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns


def _copy_bytes(source: Path, destination: Path, byte_count: int) -> None:
    remaining = byte_count
    with source.open("rb") as reader, destination.open("xb") as writer:
        while remaining:
            block = reader.read(min(_COPY_CHUNK_BYTES, remaining))
            if not block:
                raise RNNoiseExecutionError("native output ended before its declared size")
            writer.write(block)
            remaining -= len(block)
    if os.name != "nt":
        destination.chmod(0o600)


def _write_pcm16_wav(raw_path: Path, output_path: Path, *, frame_count: int) -> None:
    with wave.open(str(output_path), "wb") as target, raw_path.open("rb") as source:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SOURCE_SAMPLE_RATE_HZ)
        remaining = frame_count * 2
        while remaining:
            block = source.read(min(_COPY_CHUNK_BYTES, remaining))
            if not block:
                raise RNNoiseExecutionError("downsampled PCM ended before its declared size")
            target.writeframesraw(block)
            remaining -= len(block)
        if source.read(1):
            raise RNNoiseTimebaseError("downsampled PCM exceeds the exact source frame count")
    if os.name != "nt":
        output_path.chmod(0o600)


def _regular_size(path: Path, expected_bytes: int, *, stage: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RNNoiseExecutionError(f"{stage} did not produce its required output") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise RNNoiseExecutionError(f"{stage} output is not a regular file")
    if info.st_size != expected_bytes:
        raise RNNoiseTimebaseError(f"{stage} output sample count mismatch")


def _seal_native_output(path: Path, expected_bytes: int, *, stage: str) -> None:
    _regular_size(path, expected_bytes, stage=stage)
    if os.name != "nt":
        try:
            path.chmod(0o600)
            mode = path.lstat().st_mode
        except OSError as exc:
            raise RNNoiseExecutionError(f"{stage} output permissions could not be sealed") from exc
        if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600:
            raise RNNoiseExecutionError(f"{stage} output permissions are not private")


def _audit_workspace(root: Path, allowed_names: set[str], max_bytes: int) -> int:
    total = 0
    for entry in root.iterdir():
        info = entry.lstat()
        if entry.name not in allowed_names or not stat.S_ISREG(info.st_mode) or entry.is_symlink():
            raise RNNoiseExecutionError("RNNoise workspace contains an unexpected entry")
        total += info.st_size
        if total > max_bytes:
            raise RNNoiseExecutionError("RNNoise workspace byte limit exceeded")
    return total


class ExperimentalRNNoisePreprocessor:
    """Prepare one exact-length waveform; never silently fall back when enabled."""

    def __init__(
        self,
        *,
        policy: RNNoiseEnhancementPolicy | None = None,
        rnnoise_binary: str | os.PathLike[str] | None = None,
        rnnoise_binary_sha256: str | None = None,
        resampler_binary: str | os.PathLike[str] | None = None,
        resampler_binary_sha256: str | None = None,
        rnnoise_build_attestation: str | os.PathLike[str] | None = None,
        rnnoise_build_attestation_sha256: str | None = None,
        timebase_proof: str | os.PathLike[str] | None = None,
        timebase_proof_sha256: str | None = None,
        artifact_root: str | os.PathLike[str] | None = None,
        input_root: str | os.PathLike[str] | None = None,
        work_root: str | os.PathLike[str] | None = None,
        runner: NativeRunner | None = None,
    ) -> None:
        self.policy = policy or RNNoiseEnhancementPolicy()
        self._rnnoise_binary = rnnoise_binary
        self._rnnoise_sha256 = rnnoise_binary_sha256
        self._resampler_binary = resampler_binary
        self._resampler_sha256 = resampler_binary_sha256
        self._build_attestation = rnnoise_build_attestation
        self._build_attestation_sha256 = rnnoise_build_attestation_sha256
        self._timebase_proof = timebase_proof
        self._timebase_proof_sha256 = timebase_proof_sha256
        self._artifact_root = artifact_root
        self._input_root = input_root
        self._work_root = work_root
        self._runner = runner or SubprocessArgvRunner()
        self._runner_kind = "DEFAULT_SUBPROCESS_ARGV" if runner is None else "INJECTED_NOT_VERIFIED"

    @contextmanager
    def prepare(
        self,
        source: str | os.PathLike[str],
        *,
        expected_source_sha256: str | None = None,
        expected_duration_us: int | None = None,
    ) -> Iterator[PreparedRNNoiseAudio]:
        """Yield a source or enhanced waveform and clean any derived audio.

        Disabled mode deliberately performs no root, binary, hash, or workspace
        I/O.  Enabled mode holds a process-wide one-job gate until the caller
        leaves the context, ensuring the private prepared waveform cannot lead
        to overlapping enhancement jobs in this process.
        """

        if not self.policy.enabled:
            prefix = expected_source_sha256[:16] if _valid_sha256(expected_source_sha256) else None
            receipt = RNNoiseReceipt(
                status="DISABLED",
                source_sha256_prefix=prefix,
                output_sha256_prefix=prefix,
                source_frame_count=None,
                output_frame_count=None,
                duration_us=expected_duration_us if type(expected_duration_us) is int else None,
                rnnoise_binary_sha256=None,
                resampler_binary_sha256=None,
                build_attestation_sha256=None,
                timebase_proof_sha256=None,
                runner_kind="DISABLED",
                policy_sha256=_policy_sha256(self.policy, None, None, None, None, "DISABLED"),
            )
            yield PreparedRNNoiseAudio(
                Path(source),
                expected_source_sha256 if _valid_sha256(expected_source_sha256) else None,
                expected_source_sha256 if _valid_sha256(expected_source_sha256) else None,
                None,
                receipt.duration_us,
                False,
                True,
                receipt,
            )
            return

        acquired = _GLOBAL_JOB_GATE.acquire(timeout=float(self.policy.queue_timeout_seconds))
        if not acquired:
            raise RNNoiseBusyError("RNNoise one-job gate is busy")
        try:
            with self._prepare_enabled(
                source,
                expected_source_sha256=expected_source_sha256,
                expected_duration_us=expected_duration_us,
            ) as prepared:
                yield prepared
        finally:
            _GLOBAL_JOB_GATE.release()

    @contextmanager
    def _prepare_enabled(
        self,
        source: str | os.PathLike[str],
        *,
        expected_source_sha256: str | None,
        expected_duration_us: int | None,
    ) -> Iterator[PreparedRNNoiseAudio]:
        artifact_root = _secure_root(self._artifact_root, label="artifact")
        input_root = _secure_root(self._input_root, label="input")
        work_root = _secure_root(self._work_root, label="work")
        if not _valid_sha256(expected_source_sha256):
            raise RNNoiseArtifactError("enabled RNNoise requires the exact source SHA-256")
        if type(expected_duration_us) is not int or expected_duration_us <= 0:
            raise RNNoiseTimebaseError("enabled RNNoise requires an exact positive source duration")
        if sys.byteorder != "little":
            raise RNNoiseConfigurationError("RNNoise raw PCM experiment requires a little-endian target")

        source_path = _secure_regular_file(source, root=input_root, role="audio source")
        source_info = source_path.stat()
        if source_info.st_size > self.policy.max_source_bytes:
            raise RNNoiseConfigurationError("audio source exceeds the RNNoise byte limit")
        source_identity = (source_info.st_dev, source_info.st_ino, source_info.st_size, source_info.st_mtime_ns)
        source_sha256 = _sha256(source_path)
        if source_sha256 != expected_source_sha256:
            raise RNNoiseArtifactError("audio source SHA-256 mismatch")

        accessor = WavPcmAccessor(source_path)
        layout = accessor.layout
        if (
            layout.sample_width_bytes != 2
            or layout.channel_count != 1
            or layout.sample_rate_hz != SOURCE_SAMPLE_RATE_HZ
        ):
            raise RNNoiseTimebaseError("RNNoise experiment accepts only PCM16 mono 16 kHz WAV")
        frame_count = layout.frame_count
        duration_us = _duration_us(frame_count, SOURCE_SAMPLE_RATE_HZ)
        if duration_us != expected_duration_us or duration_us > self.policy.max_duration_us:
            raise RNNoiseTimebaseError("source duration does not match the exact PCM timebase")

        source_pcm_bytes = frame_count * 2
        upsampled_samples = frame_count * 3
        padded_samples = ((upsampled_samples + RNNOISE_FRAME_SAMPLES - 1) // RNNOISE_FRAME_SAMPLES) * RNNOISE_FRAME_SAMPLES
        demo_input_samples = padded_samples + RNNOISE_DEMO_FLUSH_INPUT_SAMPLES
        # The demo discards only the zero delayed warm-up output.  The appended
        # zero frame causes the final padded signal frame to be emitted.
        rnnoise_output_samples = demo_input_samples - RNNOISE_DEMO_WARMUP_OUTPUT_SAMPLES
        # Exact peak if each no-longer-needed intermediate is removed at the
        # next stage boundary.  The containing volume still needs an external
        # quota to stop a malicious native binary before its postcondition is
        # checked; the adapter itself never retains all stages simultaneously.
        signal_workspace_bytes = max(
            source_info.st_size + source_pcm_bytes,
            source_pcm_bytes + upsampled_samples * 2,
            source_pcm_bytes + upsampled_samples * 2 + demo_input_samples * 2,
            demo_input_samples * 2 + rnnoise_output_samples * 2,
            rnnoise_output_samples * 2 + upsampled_samples * 2,
            upsampled_samples * 2 + source_pcm_bytes,
            source_pcm_bytes * 2 + 4096,
        )
        if source_pcm_bytes + 4096 > self.policy.max_output_bytes:
            raise RNNoiseConfigurationError("RNNoise output estimate exceeds the configured bound")

        rnnoise_path = _verified_executable(
            self._rnnoise_binary,
            self._rnnoise_sha256,
            root=artifact_root,
            role="RNNoise executable",
        )
        resampler_path = _verified_executable(
            self._resampler_binary,
            self._resampler_sha256,
            root=artifact_root,
            role="resampler executable",
        )
        _, build_attestation = _verified_json_artifact(
            self._build_attestation,
            self._build_attestation_sha256,
            root=artifact_root,
            role="RNNoise build attestation",
        )
        _validate_build_attestation(build_attestation, rnnoise_sha256=self._rnnoise_sha256)
        _, timebase_proof = _verified_json_artifact(
            self._timebase_proof,
            self._timebase_proof_sha256,
            root=artifact_root,
            role="RNNoise timebase proof",
        )
        _validate_timebase_proof(
            timebase_proof,
            rnnoise_sha256=self._rnnoise_sha256,
            resampler_sha256=self._resampler_sha256,
        )
        rnnoise_binary_bytes = rnnoise_path.stat().st_size
        resampler_binary_bytes = resampler_path.stat().st_size
        if (
            rnnoise_binary_bytes > self.policy.max_native_binary_bytes
            or resampler_binary_bytes > self.policy.max_native_binary_bytes
        ):
            raise RNNoiseConfigurationError("RNNoise native executable exceeds the configured bound")
        expected_workspace_bytes = signal_workspace_bytes + rnnoise_binary_bytes + resampler_binary_bytes
        if expected_workspace_bytes > self.policy.max_workspace_bytes:
            raise RNNoiseConfigurationError("RNNoise workspace estimate exceeds the configured bound")

        outcomes: list[NativeRunOutcome] = []
        with tempfile.TemporaryDirectory(prefix="sddiar-rnnoise-", dir=work_root) as directory:
            job_root = Path(directory)
            if os.name != "nt":
                job_root.chmod(0o700)
            raw16 = job_root / "source-16k.s16le"
            raw48 = job_root / "source-48k.s16le"
            padded48 = job_root / "source-48k-padded.s16le"
            denoised48 = job_root / "denoised-48k.s16le"
            aligned48 = job_root / "denoised-48k-aligned.s16le"
            output16 = job_root / "denoised-16k.s16le"
            prepared_wav = job_root / "prepared-16k.wav"
            source_snapshot = job_root / "verified-source.wav"
            local_rnnoise = job_root / ("rnnoise-local.exe" if os.name == "nt" else "rnnoise-local")
            local_resampler = job_root / ("resampler-local.exe" if os.name == "nt" else "resampler-local")
            allowed: set[str] = set()

            snapshot_identity = _copy_verified_source_snapshot(
                source_path,
                source_snapshot,
                expected_sha256=expected_source_sha256,
                max_bytes=self.policy.max_source_bytes,
            )
            if snapshot_identity != source_identity:
                raise RNNoiseArtifactError("audio source identity changed before private snapshot")
            snapshot_layout = WavPcmAccessor(source_snapshot).layout
            if snapshot_layout != layout:
                raise RNNoiseTimebaseError("audio source layout changed before private snapshot")
            allowed.add(source_snapshot.name)

            # Execute private, re-hashed copies so replacement of an artifact
            # path between verification and exec cannot run different bytes.
            _copy_exact(rnnoise_path, local_rnnoise, offset=0, byte_count=rnnoise_binary_bytes)
            _copy_exact(resampler_path, local_resampler, offset=0, byte_count=resampler_binary_bytes)
            if os.name != "nt":
                local_rnnoise.chmod(0o500)
                local_resampler.chmod(0o500)
            allowed.update((local_rnnoise.name, local_resampler.name))
            _verified_executable(local_rnnoise, self._rnnoise_sha256, root=job_root, role="private RNNoise executable")
            _verified_executable(local_resampler, self._resampler_sha256, root=job_root, role="private resampler executable")
            _verified_executable(self._rnnoise_binary, self._rnnoise_sha256, root=artifact_root, role="RNNoise executable")
            _verified_executable(self._resampler_binary, self._resampler_sha256, root=artifact_root, role="resampler executable")
            _audit_workspace(job_root, allowed, self.policy.max_workspace_bytes)

            _copy_exact(source_snapshot, raw16, offset=layout.data_offset, byte_count=source_pcm_bytes)
            allowed.add(raw16.name)
            _regular_size(raw16, source_pcm_bytes, stage="SOURCE_PCM")
            source_snapshot.unlink()
            allowed.remove(source_snapshot.name)
            _audit_workspace(job_root, allowed, self.policy.max_workspace_bytes)

            upsample = NativeInvocation(
                "UPSAMPLE_48K",
                (
                    str(local_resampler),
                    "raw-s16le",
                    "--input",
                    str(raw16),
                    "--output",
                    str(raw48),
                    "--input-rate-hz",
                    str(SOURCE_SAMPLE_RATE_HZ),
                    "--output-rate-hz",
                    str(RNNOISE_SAMPLE_RATE_HZ),
                    "--channels",
                    "1",
                    "--exact-output-samples",
                    str(upsampled_samples),
                ),
                job_root,
                raw48,
                upsampled_samples * 2,
                float(self.policy.stage_timeout_seconds),
            )
            _verified_executable(local_resampler, self._resampler_sha256, root=job_root, role="private resampler executable")
            outcome = self._runner.run(upsample)
            if outcome.stage != upsample.stage or type(outcome.returncode) is not int or outcome.returncode != 0:
                raise RNNoiseExecutionError("RNNoise upsample stage failed")
            outcomes.append(outcome)
            allowed.add(raw48.name)
            _seal_native_output(raw48, upsample.expected_output_bytes, stage=upsample.stage)
            _verified_executable(local_resampler, self._resampler_sha256, root=job_root, role="private resampler executable")
            _verified_executable(self._resampler_binary, self._resampler_sha256, root=artifact_root, role="resampler executable")

            _copy_bytes(raw48, padded48, upsampled_samples * 2)
            if demo_input_samples > upsampled_samples:
                with padded48.open("ab") as stream:
                    stream.write(b"\x00" * ((demo_input_samples - upsampled_samples) * 2))
            allowed.add(padded48.name)
            _regular_size(padded48, demo_input_samples * 2, stage="RNNOISE_FRAME_PAD_AND_FLUSH")
            _audit_workspace(job_root, allowed, self.policy.max_workspace_bytes)
            raw16.unlink()
            raw48.unlink()
            allowed.remove(raw16.name)
            allowed.remove(raw48.name)

            rnnoise = NativeInvocation(
                "RNNOISE_48K",
                (str(local_rnnoise), str(padded48), str(denoised48)),
                job_root,
                denoised48,
                rnnoise_output_samples * 2,
                float(self.policy.stage_timeout_seconds),
            )
            _verified_executable(local_rnnoise, self._rnnoise_sha256, root=job_root, role="private RNNoise executable")
            outcome = self._runner.run(rnnoise)
            if outcome.stage != rnnoise.stage or type(outcome.returncode) is not int or outcome.returncode != 0:
                raise RNNoiseExecutionError("RNNoise native stage failed")
            outcomes.append(outcome)
            allowed.add(denoised48.name)
            _seal_native_output(denoised48, rnnoise.expected_output_bytes, stage=rnnoise.stage)
            _verified_executable(local_rnnoise, self._rnnoise_sha256, root=job_root, role="private RNNoise executable")
            _verified_executable(self._rnnoise_binary, self._rnnoise_sha256, root=artifact_root, role="RNNoise executable")
            padded48.unlink()
            allowed.remove(padded48.name)

            _copy_bytes(denoised48, aligned48, upsampled_samples * 2)
            allowed.add(aligned48.name)
            _regular_size(aligned48, upsampled_samples * 2, stage="RNNOISE_FLUSH_TRIM")
            denoised48.unlink()
            allowed.remove(denoised48.name)
            _audit_workspace(job_root, allowed, self.policy.max_workspace_bytes)

            downsample = NativeInvocation(
                "DOWNSAMPLE_16K",
                (
                    str(local_resampler),
                    "raw-s16le",
                    "--input",
                    str(aligned48),
                    "--output",
                    str(output16),
                    "--input-rate-hz",
                    str(RNNOISE_SAMPLE_RATE_HZ),
                    "--output-rate-hz",
                    str(SOURCE_SAMPLE_RATE_HZ),
                    "--channels",
                    "1",
                    "--exact-output-samples",
                    str(frame_count),
                ),
                job_root,
                output16,
                source_pcm_bytes,
                float(self.policy.stage_timeout_seconds),
            )
            _verified_executable(local_resampler, self._resampler_sha256, root=job_root, role="private resampler executable")
            outcome = self._runner.run(downsample)
            if outcome.stage != downsample.stage or type(outcome.returncode) is not int or outcome.returncode != 0:
                raise RNNoiseExecutionError("RNNoise downsample stage failed")
            outcomes.append(outcome)
            allowed.add(output16.name)
            _seal_native_output(output16, downsample.expected_output_bytes, stage=downsample.stage)
            _verified_executable(local_resampler, self._resampler_sha256, root=job_root, role="private resampler executable")
            _verified_executable(self._resampler_binary, self._resampler_sha256, root=artifact_root, role="resampler executable")
            aligned48.unlink()
            allowed.remove(aligned48.name)

            _write_pcm16_wav(output16, prepared_wav, frame_count=frame_count)
            allowed.add(prepared_wav.name)
            if prepared_wav.stat().st_size > self.policy.max_output_bytes:
                raise RNNoiseExecutionError("RNNoise prepared WAV exceeds the output byte limit")
            output16.unlink()
            allowed.remove(output16.name)
            prepared_layout = WavPcmAccessor(prepared_wav).layout
            if (
                prepared_layout.frame_count != frame_count
                or prepared_layout.sample_rate_hz != SOURCE_SAMPLE_RATE_HZ
                or prepared_layout.channel_count != 1
                or prepared_layout.sample_width_bytes != 2
                or _duration_us(prepared_layout.frame_count, prepared_layout.sample_rate_hz) != duration_us
            ):
                raise RNNoiseTimebaseError("prepared RNNoise WAV changed the source timebase")
            _audit_workspace(job_root, allowed, self.policy.max_workspace_bytes)

            _verified_executable(self._rnnoise_binary, self._rnnoise_sha256, root=artifact_root, role="RNNoise executable")
            _verified_executable(self._resampler_binary, self._resampler_sha256, root=artifact_root, role="resampler executable")
            _verified_json_artifact(
                self._build_attestation,
                self._build_attestation_sha256,
                root=artifact_root,
                role="RNNoise build attestation",
            )
            _verified_json_artifact(
                self._timebase_proof,
                self._timebase_proof_sha256,
                root=artifact_root,
                role="RNNoise timebase proof",
            )
            output_sha256 = _sha256(prepared_wav)
            receipt = RNNoiseReceipt(
                status="EXPERIMENTAL_APPLIED",
                source_sha256_prefix=source_sha256[:16],
                output_sha256_prefix=output_sha256[:16],
                source_frame_count=frame_count,
                output_frame_count=prepared_layout.frame_count,
                duration_us=duration_us,
                rnnoise_binary_sha256=self._rnnoise_sha256,
                resampler_binary_sha256=self._resampler_sha256,
                build_attestation_sha256=self._build_attestation_sha256,
                timebase_proof_sha256=self._timebase_proof_sha256,
                runner_kind=self._runner_kind,
                policy_sha256=_policy_sha256(
                    self.policy,
                    self._rnnoise_sha256,
                    self._resampler_sha256,
                    self._build_attestation_sha256,
                    self._timebase_proof_sha256,
                    self._runner_kind,
                ),
                stage_outcomes=tuple(outcomes),
                _validation_token=_RECEIPT_VALIDATION_TOKEN,
            )
            yield PreparedRNNoiseAudio(
                prepared_wav,
                source_sha256,
                output_sha256,
                frame_count,
                duration_us,
                True,
                False,
                receipt,
            )


__all__ = [
    "RNNOISE_SOURCE_COMMIT",
    "RNNOISE_SOURCE_REPOSITORY",
    "RNNOISE_MODEL_TAR_SHA256",
    "RNNOISE_LICENSE_SPDX",
    "RNNoiseEnhancementPolicy",
    "RNNoiseReceipt",
    "PreparedRNNoiseAudio",
    "NativeInvocation",
    "NativeRunOutcome",
    "NativeRunner",
    "SubprocessArgvRunner",
    "ExperimentalRNNoisePreprocessor",
    "RNNoiseExperimentalError",
    "RNNoiseConfigurationError",
    "RNNoiseArtifactError",
    "RNNoiseExecutionError",
    "RNNoiseTimebaseError",
    "RNNoiseBusyError",
]
