"""Experimental, fail-closed local ``whisper.cpp`` transcript adapter.

This module is intentionally not selected by :mod:`production_orchestrator`.
The caller must construct a ``VerifiedLocalSttIdentity`` first; that identity
is the authority for the executable and model bytes.  No model discovery,
download, cache lookup, PATH lookup, or alternate engine is provided here.

The adapter consumes whisper.cpp's full ``-ojf`` JSON document and returns only
source-time :class:`~sddiar.contracts.Word` objects.  Transcript text is never
put in an exception, invocation receipt, or public configuration object.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .contracts import (
    MAX_TIME_US,
    AudioSourceMetadata,
    Word,
    WordProvenance,
    WordTimeline,
    deterministic_id,
)
from .errors import ContractValidationError, ModelHashMismatch, SDDiarError
from .production_orchestrator import VerifiedLocalSttIdentity


# b4938 is the upstream build label used for the pinned v1.9.3 release.
WHISPER_CPP_BACKEND_ID = "whisper.cpp"
WHISPER_CPP_BACKEND_VERSION = "v1.9.3-b4938"
WHISPER_CPP_COMMIT = "371b5a7561823ab2bb32142d2751e35e7534727b"
WHISPER_CPP_DEFAULT_ENABLED = False

_ALLOWED_SPECIAL_TOKENS = frozenset(
    {
        "<|startoftranscript|>",
        "<|endoftext|>",
        "<|translate|>",
        "<|transcribe|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        "<|ko|>",
        "<|lang_id|>",
    }
)
_PUNCTUATION = frozenset(".,!?;:%)]}〉》」』】〕］、。，！？；：…—–-~·\"'\u3001\u3002")
_JOB_GATE = threading.BoundedSemaphore(1)


class WhisperCppError(SDDiarError):
    """Base class for safe whisper.cpp adapter failures."""


class WhisperCppConfigurationError(WhisperCppError, ContractValidationError):
    """The adapter configuration or sealed identity is not supported."""


class WhisperCppBusyError(WhisperCppError):
    """The one-job local decoder slot was not available in time."""


class WhisperCppExecutionError(WhisperCppError):
    """The pinned executable did not complete successfully."""


class WhisperCppTimeoutError(WhisperCppExecutionError):
    """The bounded decoder timeout expired."""


class WhisperCppOutputError(WhisperCppError):
    """The decoder output was absent or exceeded its bound."""


class WhisperCppJsonError(WhisperCppError):
    """The decoder did not emit a complete, valid JSON document."""


class WhisperCppTimestampError(WhisperCppError):
    """Token offsets cannot be represented as ordered source microseconds."""


@dataclass(frozen=True, slots=True)
class WhisperCppConfig:
    """Bounded decoding policy; no paths or transcript content are retained."""

    timeout_seconds: float = 300.0
    queue_timeout_seconds: float = 30.0
    max_output_bytes: int = 8 * 1024 * 1024
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    no_fallback: bool = True
    dtw_preset: str = "base"

    def __post_init__(self) -> None:
        for name in ("timeout_seconds", "queue_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise WhisperCppConfigurationError("whisper.cpp timeout is invalid")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise WhisperCppConfigurationError("whisper.cpp timeout is invalid")
        if type(self.max_output_bytes) is not int or self.max_output_bytes <= 0:
            raise WhisperCppConfigurationError("whisper.cpp output bound is invalid")
        if type(self.beam_size) is not int or self.beam_size <= 0:
            raise WhisperCppConfigurationError("whisper.cpp beam policy is invalid")
        if type(self.best_of) is not int or self.best_of <= 0:
            raise WhisperCppConfigurationError("whisper.cpp best-of policy is invalid")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise WhisperCppConfigurationError("whisper.cpp temperature policy is invalid")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) < 0:
            raise WhisperCppConfigurationError("whisper.cpp temperature policy is invalid")
        if type(self.no_fallback) is not bool or not self.no_fallback:
            raise WhisperCppConfigurationError("whisper.cpp fallback is prohibited")
        if self.dtw_preset not in {"base", "small", "large.v3.turbo"}:
            raise WhisperCppConfigurationError("whisper.cpp DTW preset is invalid")

    def public_identity(self) -> Mapping[str, Any]:
        """Return only scalar policy identity; never paths or transcript text."""

        # ``dtw_preset`` is a decoder setting, not calibration authority.  A
        # release caller must bind this value together with the selected model
        # hash in its signed calibration profile before claiming timestamps.

        return {
            "backend_id": WHISPER_CPP_BACKEND_ID,
            "backend_version": WHISPER_CPP_BACKEND_VERSION,
            "upstream_commit": WHISPER_CPP_COMMIT,
            "default_enabled": WHISPER_CPP_DEFAULT_ENABLED,
            "threads": 1,
            "processors": 1,
            "no_gpu": True,
            "language": "ko",
            "json_full": True,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": float(self.temperature),
            "no_fallback": self.no_fallback,
            "dtw_preset": self.dtw_preset,
            "timestamps_calibrated": False,
            "max_output_bytes": self.max_output_bytes,
            "timeout_seconds": float(self.timeout_seconds),
        }


@dataclass(frozen=True, slots=True)
class WhisperCppInvocation:
    """Internal runner request.  ``output_path`` is job-scoped and not a receipt."""

    argv: tuple[str, ...]
    cwd: Path
    output_path: Path
    timeout_seconds: float
    max_output_bytes: int


@dataclass(frozen=True, slots=True)
class WhisperCppRunResult:
    returncode: int


class WhisperCppRunner(Protocol):
    def run(self, invocation: WhisperCppInvocation) -> WhisperCppRunResult | None: ...


class SubprocessArgvRunner:
    """Run one absolute executable with a minimal environment and no streams."""

    @staticmethod
    def _environment(cwd: Path) -> dict[str, str]:
        # Deliberately omit PATH, proxy, home, locale/user config, and all
        # inherited application settings.  The executable is argv[0].
        environment = {"LC_ALL": "C", "LANG": "C", "TMPDIR": str(cwd)}
        if os.name == "nt" or sys.platform.startswith("win"):
            for key in ("SYSTEMROOT", "WINDIR"):
                value = os.environ.get(key)
                if value:
                    environment[key] = value
            environment["TEMP"] = str(cwd)
            environment["TMP"] = str(cwd)
        return environment

    def run(self, invocation: WhisperCppInvocation) -> WhisperCppRunResult:
        # Import lazily: importing the package must remain possible on hosts
        # where Python's platform-specific subprocess helpers are unavailable.
        import subprocess

        try:
            completed = subprocess.run(
                list(invocation.argv),
                cwd=str(invocation.cwd),
                shell=False,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._environment(invocation.cwd),
                timeout=invocation.timeout_seconds,
                close_fds=True,
            )
        except subprocess.TimeoutExpired:
            raise WhisperCppTimeoutError("whisper.cpp timed out") from None
        except (OSError, ValueError):
            raise WhisperCppExecutionError("whisper.cpp could not start") from None
        return WhisperCppRunResult(int(completed.returncode))


def _strict_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result

        def reject_constant(value: str) -> Any:
            raise ValueError("non-finite JSON number")

        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        raise WhisperCppJsonError("whisper.cpp JSON is invalid") from None


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WhisperCppJsonError("whisper.cpp JSON shape is invalid")
    return value


def _offset_us(value: Any, duration_us: int) -> int:
    # whisper.cpp JSON offsets are integer milliseconds.  Converting before
    # range checks guarantees the returned Word contract is source microseconds.
    if type(value) is not int or value < 0:
        raise WhisperCppTimestampError("whisper.cpp token timestamp is invalid")
    if value > MAX_TIME_US // 1_000:
        raise WhisperCppTimestampError("whisper.cpp token timestamp is out of range")
    result = value * 1_000
    if result > duration_us or result > MAX_TIME_US:
        raise WhisperCppTimestampError("whisper.cpp token timestamp is out of range")
    return result


def _is_special(text: str) -> bool:
    return (
        (text.startswith("<|") and text.endswith("|>"))
        or text == "[_BEG_]"
        or (text.startswith("[_TT_") and text.endswith("]") and text[5:-1].isdigit())
        or (text.startswith("[_") and text.endswith("]"))
    )


def _allowlisted_special(text: str) -> bool:
    return text in _ALLOWED_SPECIAL_TOKENS or text == "[_BEG_]" or (
        text.startswith("[_TT_") and text.endswith("]") and text[5:-1].isdigit()
    )


def _is_punctuation(text: str) -> bool:
    return bool(text) and all(character in _PUNCTUATION for character in text)


def _timeline_from_json(payload: Any, source: AudioSourceMetadata) -> WordTimeline:
    root = _mapping(payload)
    segments = root.get("transcription")
    if not isinstance(segments, list):
        raise WhisperCppJsonError("whisper.cpp transcription is missing")

    # Each item is (token text, start_us, end_us, opaque segment id).
    tokens: list[tuple[str, int, int, str]] = []
    previous_start = -1
    previous_end = -1
    for segment_number, raw_segment in enumerate(segments):
        segment = _mapping(raw_segment)
        raw_tokens = segment.get("tokens")
        if not isinstance(raw_tokens, list):
            raise WhisperCppJsonError("whisper.cpp token offsets are missing")
        chunk_id = f"whispercpp-segment-{segment_number:06d}"
        for raw_token in raw_tokens:
            token = _mapping(raw_token)
            raw_text = token.get("text")
            if not isinstance(raw_text, str) or not raw_text:
                raise WhisperCppJsonError("whisper.cpp token text is invalid")
            marker_text = raw_text.strip()
            if _is_special(marker_text):
                if not _allowlisted_special(marker_text):
                    raise WhisperCppJsonError("whisper.cpp emitted a disallowed special token")
                continue
            offsets = _mapping(token.get("offsets"))
            start_us = _offset_us(offsets.get("from"), source.duration_us)
            end_us = _offset_us(offsets.get("to"), source.duration_us)
            if start_us > end_us:
                raise WhisperCppTimestampError("whisper.cpp token timestamp range is invalid")
            if start_us < previous_start or start_us < previous_end:
                raise WhisperCppTimestampError("whisper.cpp token timestamps are nonmonotonic")
            previous_start, previous_end = start_us, end_us
            tokens.append((raw_text, start_us, end_us, chunk_id))

    words: list[Word] = []
    provenance: dict[str, WordProvenance] = {}
    current_text = ""
    current_start = 0
    current_end = 0
    current_chunks: list[str] = []
    pending_boundary = False

    def flush() -> None:
        nonlocal current_text, current_start, current_end, current_chunks
        if not current_text:
            current_chunks = []
            return
        inferred_time = current_start == current_end
        if inferred_time:
            if source.duration_us <= 0:
                raise WhisperCppTimestampError("zero-duration word has no bounded source time")
            if current_start < source.duration_us:
                current_end = current_start + 1
            else:
                current_start = source.duration_us - 1
                current_end = source.duration_us
        ordinal = len(words)
        word_id = deterministic_id(
            source.audio_sha256,
            "whispercpp-word",
            current_start,
            current_end,
            ordinal,
            schema_version="1.0",
            pipeline_version="whispercpp-v1",
        )
        chunk_ids = tuple(dict.fromkeys(current_chunks))
        word = Word(
            word_id,
            current_start,
            current_end,
            current_text,
            source_chunk_id=chunk_ids[0] if chunk_ids else None,
        )
        words.append(word)
        provenance[word_id] = WordProvenance(
            word_id,
            crosses_timewarp_boundary=inferred_time,
            source_chunk_ids=chunk_ids,
        )
        current_text = ""
        current_start = 0
        current_end = 0
        current_chunks = []

    for raw_text, start_us, end_us, chunk_id in tokens:
        leading_space = raw_text[0].isspace()
        trailing_space = raw_text[-1].isspace()
        core = raw_text.strip()
        if not core:
            pending_boundary = True
            continue
        # Punctuation is attached to the preceding eojeol, even when the
        # tokenizer emitted it as a leading-space piece.
        if current_text and (pending_boundary or leading_space) and not _is_punctuation(core):
            flush()
        if not current_text:
            current_start = start_us
        current_text += core
        current_end = end_us
        current_chunks.append(chunk_id)
        pending_boundary = trailing_space
    flush()
    return WordTimeline(tuple(words), provenance)


class WhisperCppBackend:
    """Concrete local engine for the pinned whisper.cpp runtime.

    It is an explicit experimental object and is never instantiated by the
    production orchestrator's default path.  ``runner`` is injectable solely
    to permit deterministic synthetic tests without a model or native binary.
    """

    experimental = True
    default_selected = False

    def __init__(
        self,
        identity: VerifiedLocalSttIdentity,
        config: WhisperCppConfig | None = None,
        *,
        runner: WhisperCppRunner | Callable[[WhisperCppInvocation], Any] | None = None,
    ) -> None:
        if type(identity) is not VerifiedLocalSttIdentity:
            raise WhisperCppConfigurationError("whisper.cpp identity is invalid")
        if identity.backend_id != WHISPER_CPP_BACKEND_ID or identity.backend_version != WHISPER_CPP_BACKEND_VERSION:
            raise WhisperCppConfigurationError("whisper.cpp runtime identity is unsupported")
        try:
            identity.assert_artifacts_unchanged()
        except ModelHashMismatch:
            raise WhisperCppConfigurationError("whisper.cpp local artifacts changed") from None
        self._identity = identity
        self.config = config or WhisperCppConfig()
        self._runner = runner or SubprocessArgvRunner()

    @property
    def identity(self) -> VerifiedLocalSttIdentity:
        return self._identity

    def transcribe(
        self,
        canonical_audio_path: Path,
        source: AudioSourceMetadata,
    ) -> WordTimeline:
        try:
            self._identity.assert_artifacts_unchanged()
        except ModelHashMismatch:
            raise WhisperCppConfigurationError("whisper.cpp local artifacts changed") from None
        audio_path = Path(canonical_audio_path)
        if audio_path.is_symlink() or not audio_path.is_file():
            raise WhisperCppConfigurationError("whisper.cpp audio input is invalid")
        if not _JOB_GATE.acquire(timeout=float(self.config.queue_timeout_seconds)):
            raise WhisperCppBusyError("whisper.cpp decoder is busy")
        try:
            return self._run_job(audio_path, source)
        finally:
            _JOB_GATE.release()

    def _run_job(
        self,
        audio_path: Path,
        source: AudioSourceMetadata,
    ) -> WordTimeline:
        identity = self._identity
        import tempfile

        with tempfile.TemporaryDirectory(prefix="sddiar-whispercpp-") as temp_name:
            cwd = Path(temp_name)
            output_prefix = cwd / "transcription"
            output_path = output_prefix.with_suffix(".json")
            argv = (
                str(identity.engine_artifact.path),
                "-m", str(identity.model_artifact.path),
                "-f", str(audio_path),
                "-ojf", "-of", str(output_prefix),
                "-t", "1", "-p", "1", "-ng", "-l", "ko",
                "-bs", str(self.config.beam_size),
                "-bo", str(self.config.best_of),
                "-tp", str(self.config.temperature),
                "-nf", "-dtw", self.config.dtw_preset,
            )
            invocation = WhisperCppInvocation(
                argv, cwd, output_path, float(self.config.timeout_seconds), self.config.max_output_bytes
            )
            try:
                if callable(getattr(self._runner, "run", None)):
                    result = self._runner.run(invocation)  # type: ignore[union-attr]
                elif callable(self._runner):
                    result = self._runner(invocation)  # type: ignore[operator]
                else:
                    raise WhisperCppConfigurationError("whisper.cpp runner is invalid")
            except WhisperCppError:
                raise
            except Exception:
                raise WhisperCppExecutionError("whisper.cpp execution failed") from None
            if result is not None:
                returncode = getattr(result, "returncode", result if isinstance(result, int) else 0)
                if type(returncode) is not int or returncode != 0:
                    raise WhisperCppExecutionError("whisper.cpp returned a failure status")
            try:
                if output_path.is_symlink() or not output_path.is_file():
                    raise WhisperCppOutputError("whisper.cpp JSON output is missing")
                if output_path.stat().st_size > self.config.max_output_bytes:
                    raise WhisperCppOutputError("whisper.cpp JSON output is too large")
                raw = output_path.read_bytes()
            except WhisperCppOutputError:
                raise
            except (OSError, ValueError):
                raise WhisperCppOutputError("whisper.cpp JSON output is unreadable") from None
            timeline = _timeline_from_json(_strict_json(raw), source)
            try:
                identity.assert_artifacts_unchanged()
            except ModelHashMismatch:
                raise WhisperCppConfigurationError("whisper.cpp local artifacts changed") from None
            return timeline


__all__ = [
    "WHISPER_CPP_BACKEND_ID",
    "WHISPER_CPP_BACKEND_VERSION",
    "WHISPER_CPP_COMMIT",
    "WHISPER_CPP_DEFAULT_ENABLED",
    "WhisperCppBackend",
    "WhisperCppBusyError",
    "WhisperCppConfig",
    "WhisperCppConfigurationError",
    "WhisperCppError",
    "WhisperCppExecutionError",
    "WhisperCppInvocation",
    "WhisperCppJsonError",
    "WhisperCppOutputError",
    "WhisperCppRunResult",
    "WhisperCppRunner",
    "WhisperCppTimeoutError",
    "WhisperCppTimestampError",
    "SubprocessArgvRunner",
]
