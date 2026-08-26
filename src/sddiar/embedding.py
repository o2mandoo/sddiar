"""Offline speaker-embedding backends.

This module deliberately contains no model download or implicit dependency
installation.  The ONNX backend can only be constructed from an artifact that
has already passed :mod:`sddiar.model_pack` verification.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import EmbeddingRegion, EmbeddingResult
from .errors import ContractValidationError, ModelHashMismatch, ModelNotFound, ModelRuntimeIncompatible
from .model_pack import VerifiedArtifact
from .ort_cpu import create_ort_session


@dataclass(frozen=True, slots=True)
class EmbeddingInputContract:
    """The inspected, release-locked model I/O contract."""

    input_name: str
    output_name: str
    sample_rate_hz: int = 16000
    feature_bins: int = 80
    embedding_dimension: int = 256
    input_rank: int = 3
    output_rank: int = 2

    def __post_init__(self) -> None:
        if not self.input_name or not self.output_name:
            raise ContractValidationError("ONNX input/output names are required")
        if any(type(v) is not int or v <= 0 for v in (self.sample_rate_hz, self.feature_bins, self.embedding_dimension)):
            raise ContractValidationError("invalid embedding input contract")
        if self.input_rank != 3 or self.output_rank != 2:
            raise ContractValidationError("only [batch, frames, bins] -> [batch, dimension] is supported")


@runtime_checkable
class SpeakerEmbeddingBackend(Protocol):
    model_id: str
    model_hash: str
    input_contract: EmbeddingInputContract

    def embed(self, regions: Sequence[EmbeddingRegion]) -> Sequence[EmbeddingResult]: ...


def _unit(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ContractValidationError("embedding vector is empty")
    out = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in out):
        raise ContractValidationError("embedding vector contains non-finite value")
    norm = math.sqrt(sum(x * x for x in out))
    if norm <= 0:
        raise ContractValidationError("embedding vector is zero")
    return tuple(x / norm for x in out)


class DeterministicFixtureEmbeddingBackend:
    """Dependency-free backend for tests and pipeline seam verification."""

    def __init__(self, dimension: int = 4, model_id: str = "fixture-embedding-v1") -> None:
        if type(dimension) is not int or dimension <= 0:
            raise ContractValidationError("dimension must be positive")
        self.model_id = model_id
        self.model_hash = hashlib.sha256(model_id.encode()).hexdigest()
        self.input_contract = EmbeddingInputContract("fixture_features", "fixture_embedding", embedding_dimension=dimension)
        self._dimension = dimension

    def embed(self, regions: Sequence[EmbeddingRegion]) -> tuple[EmbeddingResult, ...]:
        if not isinstance(regions, Sequence):
            raise ContractValidationError("regions must be a sequence")
        result: list[EmbeddingResult] = []
        for region in regions:
            if not isinstance(region, EmbeddingRegion):
                raise ContractValidationError("regions must contain EmbeddingRegion")
            # Hash-derived vectors make IDs stable across processes and runs,
            # while still yielding distinct fixture speakers in normal tests.
            digest = hashlib.sha256(region.tracklet_id.encode()).digest()
            raw = [((digest[i % len(digest)] / 127.5) - 1.0) for i in range(self._dimension)]
            vector = _unit(raw)
            result.append(EmbeddingResult(region.embedding_region_id, region.tracklet_id, True, vector,
                                          dimension=self._dimension, valid_window_count=1,
                                          clean_window_coverage=region.speech_coverage_ratio,
                                          intra_window_consistency=1.0, quality=region.speech_coverage_ratio,
                                          model_pack_id=self.model_id, model_hash=self.model_hash))
        return tuple(result)


class OnnxRuntimeSpeakerEmbeddingBackend:
    """Strict local ONNX Runtime adapter; all intake and runtime checks fail closed."""

    def __init__(self, artifact: VerifiedArtifact, input_contract: EmbeddingInputContract,
                 *, model_id: str = "", runtime_contract: Mapping[str, Any] | None = None,
                 session: Any | None = None) -> None:
        if not isinstance(artifact, VerifiedArtifact) or artifact.role not in ("embedding", "speaker_embedding"):
            raise ModelRuntimeIncompatible("embedding backend requires a verified embedding artifact")
        path = Path(artifact.path)
        if not path.is_file():
            raise ModelNotFound("verified ONNX artifact is missing")
        if path.suffix.lower() != ".onnx":
            raise ModelRuntimeIncompatible("embedding artifact must be ONNX")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise ModelHashMismatch("verified ONNX artifact hash changed")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise ModelRuntimeIncompatible("numpy is required; no installation is attempted") from exc
        self.input_contract = input_contract
        if session is None:
            try:
                session = create_ort_session(path)
            except Exception as exc:
                raise ModelRuntimeIncompatible("ONNX session creation failed") from exc
        self._session = session
        providers = tuple(self._session.get_providers())
        if providers != ("CPUExecutionProvider",):
            raise ModelRuntimeIncompatible("ONNX session is not CPU-only")
        self._np = np
        self.model_id = model_id or artifact.file_id
        self.model_hash = artifact.sha256
        self._validate_io(runtime_contract or {})

    def _validate_io(self, runtime_contract: Mapping[str, Any]) -> None:
        ins, outs = self._session.get_inputs(), self._session.get_outputs()
        c = self.input_contract
        if len(ins) != 1 or len(outs) != 1 or ins[0].name != c.input_name or outs[0].name != c.output_name:
            raise ModelRuntimeIncompatible("ONNX input/output name contract mismatch")
        if len(ins[0].shape) != c.input_rank or len(outs[0].shape) != c.output_rank:
            raise ModelRuntimeIncompatible("ONNX input/output rank contract mismatch")
        for key, actual in (("input_name", ins[0].name), ("output_name", outs[0].name)):
            if key in runtime_contract and runtime_contract[key] != actual:
                raise ModelRuntimeIncompatible(f"runtime contract {key} mismatch")
        if "embedding_dimension" in runtime_contract and runtime_contract["embedding_dimension"] != c.embedding_dimension:
            raise ModelRuntimeIncompatible("runtime embedding dimension mismatch")

    def embed(self, regions: Sequence[EmbeddingRegion]) -> tuple[EmbeddingResult, ...]:
        if not isinstance(regions, Sequence) or not regions:
            raise ContractValidationError("regions must be a non-empty sequence")
        raise ModelRuntimeIncompatible("audio feature extraction is not part of this adapter; provide approved feature seam")
