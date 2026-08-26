"""Canonical, content-aware public result serialization.

Internal audio buffers, embeddings, and centroids are intentionally rejected at
this boundary. Public transcript text is allowed only when it is part of a
`PipelineResult` supplied by the caller.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import EmbeddingResult, PipelineResult, SpeakerHypothesis, SpeakerState
from .errors import ResultSchemaValidationFailed


_INTERNAL_TYPES = (EmbeddingResult, SpeakerHypothesis, SpeakerState)


def _public_value(value: Any) -> Any:
    if isinstance(value, _INTERNAL_TYPES):
        raise ResultSchemaValidationFailed("internal audio/embedding state cannot be serialized")
    if is_dataclass(value):
        return {field.name: _public_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_public_value(item) for item in value]
    if isinstance(value, Path):
        raise ResultSchemaValidationFailed("local filesystem paths cannot be serialized into a public result")
    if isinstance(value, float) and not math.isfinite(value):
        raise ResultSchemaValidationFailed("non-finite result value")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ResultSchemaValidationFailed(f"unsupported public result type: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Serialize with deterministic key ordering and no NaN/Infinity values."""

    try:
        return json.dumps(
            _public_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, ResultSchemaValidationFailed) as exc:
        if isinstance(exc, ResultSchemaValidationFailed):
            raise
        raise ResultSchemaValidationFailed(str(exc)) from exc


class ResultSerializer:
    def serialize(self, result: PipelineResult) -> bytes:
        if not isinstance(result, PipelineResult):
            raise ResultSchemaValidationFailed("ResultSerializer requires PipelineResult")
        return canonical_json(result)

    def to_mapping(self, result: PipelineResult) -> dict[str, Any]:
        return json.loads(self.serialize(result).decode("utf-8"))
