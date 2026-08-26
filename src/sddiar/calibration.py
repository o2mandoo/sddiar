"""Immutable, fail-safe calibration profile loading and binding.

``CalibrationProfile`` remains a data-only, backwards-compatible loader. It
is *not* authority for a quality PASS. Production callers must use an injected
signature verifier to create a ``VerifiedCalibrationBinding`` that is bound to
the current model hashes, source sample rate, and configuration hash.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol


class CalibrationError(ValueError):
    """Malformed or incompatible calibration data."""

    code = "Q_CALIBRATION_INVALID"


class CalibrationMissingError(CalibrationError):
    code = "Q_CALIBRATION_MISSING"


class CalibrationMismatchError(CalibrationError):
    code = "Q_CALIBRATION_MISMATCH"


class CalibrationSignatureError(CalibrationError):
    code = "Q_CALIBRATION_SIGNATURE_INVALID"


class CalibrationProvenanceError(CalibrationError):
    code = "Q_CALIBRATION_PROVENANCE_INCOMPLETE"


class CalibrationUnboundError(CalibrationError):
    code = "Q_CALIBRATION_UNBOUND"


class CalibrationSignatureVerifier(Protocol):
    """Offline-capable signature verification seam.

    Production supplies an audited verifier. Tests can inject a deterministic
    implementation without adding crypto or network dependencies to the core.
    """

    trust_level: str

    def verify(self, payload: bytes, signature: bytes, key_id: str) -> bool: ...


class DigestCalibrationSignatureVerifier:
    """Deterministic development verifier; not a production signature scheme."""

    trust_level = "DEVELOPMENT"

    def __init__(self, key: bytes):
        self.key = bytes(key)

    def verify(self, payload: bytes, signature: bytes, key_id: str) -> bool:
        expected = hashlib.sha256(self.key + payload).digest()
        return signature in (
            expected,
            expected.hex().encode("ascii"),
            base64.b64encode(expected),
        )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _mapping_value(data: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Immutable calibration data.

    Legacy/incomplete profiles can still be loaded for inspection and
    development tooling. ``CalibrationProfileVerifier`` enforces the strict
    release schema before a profile can influence a speaker-aware PASS.
    """

    profile_id: str
    model_hashes: Mapping[str, str]
    source_sample_rates: tuple[int, ...]
    thresholds: Mapping[str, float] = field(default_factory=dict)
    source: str | None = None
    schema_version: str = "1"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset_manifest_hash: str = ""
    scorer_hash: str = ""
    config_hash: str = ""
    approver: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    signer_key_id: str = ""
    signature: str = ""
    calibration_version: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise CalibrationError("profile_id is required")
        try:
            raw_hashes = dict(self.model_hashes)
            raw_rates = tuple(self.source_sample_rates)
            raw_thresholds = dict(self.thresholds)
            metadata = dict(self.metadata)
            provenance = dict(self.provenance)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("invalid calibration profile fields") from exc
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in raw_hashes.items()):
            raise CalibrationError("model hash names and values must be strings")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in raw_rates):
            raise CalibrationError("source sample rates must be integers")
        if any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            for key, value in raw_thresholds.items()
        ):
            raise CalibrationError("threshold names and values must be numeric")
        if any(not isinstance(key, str) for key in provenance):
            raise CalibrationError("provenance keys must be strings")
        hashes = {key: value.lower() for key, value in raw_hashes.items()}
        rates = tuple(sorted(set(raw_rates)))
        thresholds = {key: float(value) for key, value in raw_thresholds.items()}
        if not rates or any(x <= 0 for x in rates):
            raise CalibrationError("at least one positive source sample rate is required")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "model_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "source_sample_rates", rates)
        object.__setattr__(self, "thresholds", MappingProxyType(thresholds))
        object.__setattr__(self, "metadata", _freeze(metadata))
        object.__setattr__(self, "provenance", _freeze(provenance))
        object.__setattr__(self, "dataset_manifest_hash", self.dataset_manifest_hash.lower() if isinstance(self.dataset_manifest_hash, str) else "")
        object.__setattr__(self, "scorer_hash", self.scorer_hash.lower() if isinstance(self.scorer_hash, str) else "")
        object.__setattr__(self, "config_hash", self.config_hash.lower() if isinstance(self.config_hash, str) else "")
        object.__setattr__(self, "approver", self.approver.strip() if isinstance(self.approver, str) else "")
        object.__setattr__(self, "signer_key_id", self.signer_key_id.strip() if isinstance(self.signer_key_id, str) else "")
        object.__setattr__(self, "signature", self.signature.strip() if isinstance(self.signature, str) else "")
        object.__setattr__(self, "calibration_version", self.calibration_version.strip() if isinstance(self.calibration_version, str) else "")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str | None = None) -> "CalibrationProfile":
        if not isinstance(data, Mapping):
            raise CalibrationError("calibration profile must be an object")
        rates = data.get("source_sample_rates", data.get("source_rates", data.get("sample_rates", ())))
        hashes = data.get("model_hashes", data.get("models", {}))
        if isinstance(hashes, str):
            hashes = {"default": hashes}
        integrity = data.get("integrity", {})
        if not isinstance(integrity, Mapping):
            integrity = {}
        return cls(
            profile_id=data.get("profile_id", ""),
            model_hashes=hashes or {},
            source_sample_rates=tuple(rates or ()),
            thresholds=data.get("thresholds", {}),
            source=source or data.get("source"),
            schema_version=str(data.get("schema_version", "1")),
            metadata=data.get("metadata", {}),
            dataset_manifest_hash=_mapping_value(
                data, "dataset_manifest_hash", "dataset_manifest_sha256", "dataset_hash"
            ),
            scorer_hash=_mapping_value(data, "scorer_hash", "scorer_sha256"),
            config_hash=_mapping_value(data, "config_hash", "config_sha256"),
            approver=_mapping_value(data, "approver", "approval_id"),
            provenance=data.get("provenance", {}),
            signer_key_id=_mapping_value(
                data, "signer_key_id", default=integrity.get("signer_key_id", "")
            ),
            signature=_mapping_value(
                data, "signature", default=integrity.get("signature", "")
            ),
            calibration_version=data.get("calibration_version", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        p = Path(path)
        try:
            return cls.from_mapping(json.loads(p.read_text(encoding="utf-8")), source=str(p))
        except FileNotFoundError as exc:
            raise CalibrationMissingError(f"calibration profile missing: {p}") from exc
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"invalid calibration JSON: {p}") from exc

    from_json = load

    def matches(self, model_hashes: Mapping[str, str] | None = None,
                profile_id: str | None = None, source_sample_rate: int | None = None) -> bool:
        """Legacy compatibility check; this does not confer PASS authority."""
        if profile_id is not None and profile_id != self.profile_id:
            return False
        if source_sample_rate is not None and int(source_sample_rate) not in self.source_sample_rates:
            return False
        if model_hashes is not None and any(
            str(model_hashes.get(k, "")).lower() != v for k, v in self.model_hashes.items()
        ):
            return False
        return True


def _canonical_profile_mapping(profile: CalibrationProfile) -> dict[str, Any]:
    """Return every signed field in one normalized representation."""
    return {
        "approver": profile.approver,
        "calibration_version": profile.calibration_version,
        "config_hash": profile.config_hash,
        "dataset_manifest_hash": profile.dataset_manifest_hash,
        "metadata": _thaw(profile.metadata),
        "model_hashes": dict(profile.model_hashes),
        "profile_id": profile.profile_id,
        "provenance": _thaw(profile.provenance),
        "schema_version": profile.schema_version,
        "scorer_hash": profile.scorer_hash,
        "signer_key_id": profile.signer_key_id,
        "source_sample_rates": list(profile.source_sample_rates),
        "thresholds": dict(profile.thresholds),
    }


def canonical_calibration_bytes(profile: CalibrationProfile | Mapping[str, Any]) -> bytes:
    """Canonical signed bytes; only the signature itself is excluded."""
    parsed = profile if isinstance(profile, CalibrationProfile) else CalibrationProfile.from_mapping(profile)
    try:
        return json.dumps(
            _canonical_profile_mapping(parsed),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalibrationError("calibration profile is not canonical JSON data") from exc


_SHA256_LENGTH = 64
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == _SHA256_LENGTH and all(char in _HEX for char in text)


_REQUIRED_PROVENANCE_KEYS = frozenset({
    "annotation_schema_version",
    "created_at",
    "model_pack_id",
    "pipeline_version",
    "safety_constraints",
    "selection_objective",
})
_REQUIRED_PROVENANCE_TEXT_KEYS = _REQUIRED_PROVENANCE_KEYS - {"safety_constraints"}


def _has_complete_provenance(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    if not _REQUIRED_PROVENANCE_KEYS.issubset(value):
        return False
    if any(
        not isinstance(value[key], str) or not value[key].strip()
        for key in _REQUIRED_PROVENANCE_TEXT_KEYS
    ):
        return False
    safety_constraints = value["safety_constraints"]
    if (
        not isinstance(safety_constraints, (tuple, list))
        or not safety_constraints
        or any(
            not isinstance(constraint, str) or not constraint.strip()
            for constraint in safety_constraints
        )
    ):
        return False
    for key, item in value.items():
        if not str(key).strip() or item is None:
            return False
        if isinstance(item, str) and not item.strip():
            return False
        if isinstance(item, (Mapping, tuple, list, set, frozenset)) and not item:
            return False
    return True


def _validate_release_profile(profile: CalibrationProfile) -> None:
    if profile.schema_version != "1":
        raise CalibrationProvenanceError("unsupported calibration schema version")
    if not profile.thresholds:
        error = CalibrationProvenanceError("nonempty calibration thresholds are required")
        error.code = "Q_CALIBRATION_THRESHOLDS_EMPTY"
        raise error
    if any(
        not key.strip() or isinstance(value, bool) or not math.isfinite(value)
        for key, value in profile.thresholds.items()
    ):
        raise CalibrationError("threshold names and values must be finite")
    if not profile.model_hashes or any(
        not key.strip() or not _is_sha256(value) for key, value in profile.model_hashes.items()
    ):
        raise CalibrationProvenanceError("nonempty SHA-256 model hashes are required")
    if not all(_is_sha256(value) for value in (
        profile.dataset_manifest_hash, profile.scorer_hash, profile.config_hash
    )):
        raise CalibrationProvenanceError("dataset, scorer, and config SHA-256 hashes are required")
    if (
        not profile.calibration_version
        or not profile.approver
        or not profile.signer_key_id
        or not _has_complete_provenance(profile.provenance)
    ):
        raise CalibrationProvenanceError(
            "calibration version, approver, signer, and complete provenance are required"
        )
    if not profile.signature:
        raise CalibrationSignatureError("calibration signature is required")


def _read_diagnostic(diagnostics: Any, *names: str) -> Any:
    for name in names:
        if isinstance(diagnostics, Mapping) and name in diagnostics:
            return diagnostics[name]
        if hasattr(diagnostics, name):
            return getattr(diagnostics, name)
    return None


_VERIFIED_BINDING_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedCalibrationBinding:
    """Verifier-created, immutable authority for calibrated quality rules."""

    profile: CalibrationProfile
    model_hashes: Mapping[str, str]
    source_sample_rate: int
    config_hash: str
    profile_payload_sha256: str
    trust_level: str
    _verification_token: object = field(repr=False, compare=False)

    def __init__(self, profile: CalibrationProfile, model_hashes: Mapping[str, str],
                 source_sample_rate: int, config_hash: str, profile_payload_sha256: str,
                 trust_level: str,
                 *, _verification_token: object = None) -> None:
        if _verification_token is not _VERIFIED_BINDING_TOKEN:
            raise TypeError("VerifiedCalibrationBinding must be created by CalibrationProfileVerifier")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "model_hashes", MappingProxyType(dict(model_hashes)))
        object.__setattr__(self, "source_sample_rate", int(source_sample_rate))
        object.__setattr__(self, "config_hash", str(config_hash).lower())
        object.__setattr__(self, "profile_payload_sha256", str(profile_payload_sha256).lower())
        object.__setattr__(self, "trust_level", str(trust_level).upper())
        object.__setattr__(self, "_verification_token", _verification_token)

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    @property
    def thresholds(self) -> Mapping[str, float]:
        return self.profile.thresholds

    @property
    def valid(self) -> bool:
        return self._verification_token is _VERIFIED_BINDING_TOKEN

    @property
    def is_verified(self) -> bool:
        return self.valid

    @property
    def release_authorized(self) -> bool:
        return self.valid and self.trust_level == "RELEASE"

    def __bool__(self) -> bool:
        return self.valid

    def mismatch_reason_codes(self, diagnostics: Any) -> tuple[str, ...]:
        """Reject explicit runtime evidence that conflicts with this binding."""
        reasons: list[str] = []
        profile_id = _read_diagnostic(diagnostics, "calibration_profile_id", "profile_id")
        hashes = _read_diagnostic(diagnostics, "model_hashes")
        source_rate = _read_diagnostic(
            diagnostics, "source_sample_rate", "source_sample_rate_hz", "sample_rate_hz"
        )
        config_hash = _read_diagnostic(
            diagnostics, "config_hash", "quality_config_hash", "pipeline_config_hash"
        )
        if any(value is None for value in (profile_id, hashes, source_rate, config_hash)):
            reasons.append("Q_CALIBRATION_UNBOUND")
        if profile_id is not None and str(profile_id) != self.profile_id:
            reasons.append("Q_CALIBRATION_PROFILE_MISMATCH")
        if hashes is not None:
            normalized_hashes = (
                {key: value.lower() for key, value in hashes.items()}
                if isinstance(hashes, Mapping) and all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in hashes.items()
                )
                else None
            )
            if normalized_hashes != dict(self.model_hashes):
                reasons.append("Q_CALIBRATION_MODEL_HASH_MISMATCH")
        if source_rate is not None:
            matches_rate = (
                isinstance(source_rate, int)
                and not isinstance(source_rate, bool)
                and source_rate == self.source_sample_rate
            )
            if not matches_rate:
                reasons.append("Q_CALIBRATION_SOURCE_RATE_MISMATCH")
        if config_hash is not None and (
            not isinstance(config_hash, str) or config_hash.lower() != self.config_hash
        ):
            reasons.append("Q_CALIBRATION_CONFIG_HASH_MISMATCH")
        return tuple(reasons)


class CalibrationProfileVerifier:
    """Verify signature/provenance and bind a profile to one runtime contract."""

    def __init__(self, signature_verifier: CalibrationSignatureVerifier):
        if signature_verifier is None or not callable(getattr(signature_verifier, "verify", None)):
            raise CalibrationSignatureError("a calibration signature verifier is required")
        self.signature_verifier = signature_verifier
        self.trust_level = str(getattr(signature_verifier, "trust_level", "UNTRUSTED")).upper()

    def verify(
        self,
        profile: CalibrationProfile | Mapping[str, Any] | str | Path,
        *,
        model_hashes: Mapping[str, str],
        source_sample_rate: int,
        config_hash: str,
        profile_id: str | None = None,
    ) -> VerifiedCalibrationBinding:
        parsed = self._load(profile)
        _validate_release_profile(parsed)
        payload = canonical_calibration_bytes(parsed)
        try:
            signature = base64.b64decode(parsed.signature, validate=True)
        except (ValueError, TypeError) as exc:
            raise CalibrationSignatureError("invalid calibration signature encoding") from exc
        try:
            signature_valid = bool(self.signature_verifier.verify(
                payload, signature, parsed.signer_key_id
            ))
        except Exception as exc:
            raise CalibrationSignatureError("calibration signature verification failed") from exc
        if not signature_valid:
            raise CalibrationSignatureError("calibration signature verification failed")

        if not isinstance(model_hashes, Mapping) or not model_hashes:
            raise CalibrationUnboundError("runtime model hashes are required")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in model_hashes.items()
        ):
            raise CalibrationUnboundError("runtime model hash names and values must be strings")
        runtime_hashes = {key: value.lower() for key, value in model_hashes.items()}
        if any(not key.strip() or not _is_sha256(value) for key, value in runtime_hashes.items()):
            raise CalibrationUnboundError("runtime model hashes must be SHA-256 values")
        if runtime_hashes != dict(parsed.model_hashes):
            error = CalibrationMismatchError("calibration model hash mismatch")
            error.code = "Q_CALIBRATION_MODEL_HASH_MISMATCH"
            raise error
        if not isinstance(source_sample_rate, int) or isinstance(source_sample_rate, bool):
            raise CalibrationUnboundError("runtime source sample rate is required")
        rate = source_sample_rate
        if rate not in parsed.source_sample_rates:
            error = CalibrationMismatchError("calibration source sample rate mismatch")
            error.code = "Q_CALIBRATION_SOURCE_RATE_MISMATCH"
            raise error
        if not isinstance(config_hash, str):
            raise CalibrationUnboundError("runtime config hash must be a SHA-256 value")
        normalized_config_hash = config_hash.lower()
        if not _is_sha256(normalized_config_hash):
            raise CalibrationUnboundError("runtime config hash must be a SHA-256 value")
        if normalized_config_hash != parsed.config_hash:
            error = CalibrationMismatchError("calibration config hash mismatch")
            error.code = "Q_CALIBRATION_CONFIG_HASH_MISMATCH"
            raise error
        if profile_id is not None and (
            not isinstance(profile_id, str) or profile_id != parsed.profile_id
        ):
            error = CalibrationMismatchError("calibration profile id mismatch")
            error.code = "Q_CALIBRATION_PROFILE_MISMATCH"
            raise error
        return VerifiedCalibrationBinding(
            parsed,
            runtime_hashes,
            rate,
            normalized_config_hash,
            hashlib.sha256(payload).hexdigest(),
            self.trust_level,
            _verification_token=_VERIFIED_BINDING_TOKEN,
        )

    @staticmethod
    def _load(profile: CalibrationProfile | Mapping[str, Any] | str | Path) -> CalibrationProfile:
        if isinstance(profile, CalibrationProfile):
            return profile
        if isinstance(profile, (str, Path)):
            return CalibrationProfile.load(profile)
        if isinstance(profile, Mapping):
            return CalibrationProfile.from_mapping(profile)
        raise CalibrationError("calibration profile must be a profile, mapping, or local path")


CalibrationVerifier = CalibrationProfileVerifier


@dataclass(frozen=True, slots=True)
class CalibrationBinding:
    """Legacy inspection result; never authorizes a QualityGate PASS."""

    profile: CalibrationProfile | None
    valid: bool
    reason_codes: tuple[str, ...] = ()

    @property
    def profile_id(self) -> str | None:
        return self.profile.profile_id if self.valid and self.profile else None

    def __bool__(self) -> bool:
        return self.valid


def bind_calibration_profile(
    profile: CalibrationProfile | Mapping[str, Any] | str | Path | None,
    *,
    model_hashes: Mapping[str, str] | None = None,
    source_sample_rate: int | None = None,
    profile_id: str | None = None,
    config_hash: str | None = None,
    signature_verifier: CalibrationSignatureVerifier | None = None,
) -> VerifiedCalibrationBinding | CalibrationBinding:
    """Backwards-compatible fail-safe binder.

    Legacy calls retain their historical compatibility-match flag, but that
    legacy type is never PASS authority in QualityGate. Supplying every strict
    input delegates to the verifier and returns ``VerifiedCalibrationBinding``.
    """
    if profile is None:
        return CalibrationBinding(None, False, ("Q_CALIBRATION_MISSING",))
    try:
        parsed = CalibrationProfileVerifier._load(profile)
        # Preserve useful legacy mismatch diagnostics while keeping the result
        # non-authoritative until signature and runtime binding are complete.
        legacy_reasons: list[str] = []
        if profile_id is not None and str(profile_id) != parsed.profile_id:
            legacy_reasons.append("Q_CALIBRATION_PROFILE_MISMATCH")
        if source_sample_rate is not None:
            try:
                if isinstance(source_sample_rate, bool) or int(source_sample_rate) not in parsed.source_sample_rates:
                    legacy_reasons.append("Q_CALIBRATION_SOURCE_RATE_MISMATCH")
            except (TypeError, ValueError):
                legacy_reasons.append("Q_CALIBRATION_SOURCE_RATE_MISMATCH")
        if model_hashes is not None and (
            not isinstance(model_hashes, Mapping) or not parsed.matches(model_hashes)
        ):
            legacy_reasons.append("Q_CALIBRATION_MODEL_HASH_MISMATCH")
        if legacy_reasons:
            return CalibrationBinding(parsed, False, tuple(legacy_reasons))
        if signature_verifier is None:
            # Preserve the historical compatibility-match signal for callers
            # that use this loader outside QualityGate. QualityGate treats the
            # legacy type as unverified regardless of this flag.
            return CalibrationBinding(parsed, True, ())
        if model_hashes is None or source_sample_rate is None or config_hash is None:
            return CalibrationBinding(parsed, False, ("Q_CALIBRATION_UNBOUND",))
        return CalibrationProfileVerifier(signature_verifier).verify(
            parsed,
            model_hashes=model_hashes,
            source_sample_rate=source_sample_rate,
            config_hash=config_hash,
            profile_id=profile_id,
        )
    except CalibrationMissingError:
        return CalibrationBinding(None, False, ("Q_CALIBRATION_MISSING",))
    except (CalibrationError, TypeError, ValueError) as exc:
        return CalibrationBinding(None, False, (getattr(exc, "code", "Q_CALIBRATION_INVALID"),))


bind = bind_calibration_profile


def threshold_relation(value: float | None, threshold: float | None, *, higher_is_better: bool = True) -> str:
    """Return a serializable PASS/WARN/FAIL relation for one metric."""
    if value is None or threshold is None:
        return "NOT_EVALUATED"
    try:
        ok = float(value) >= float(threshold) if higher_is_better else float(value) <= float(threshold)
    except (TypeError, ValueError):
        return "FAIL"
    return "PASS" if ok else "FAIL"


def evaluate_thresholds(metrics: Mapping[str, float], thresholds: Mapping[str, float], *, higher_is_better: Mapping[str, bool] | None = None) -> dict[str, str]:
    directions = higher_is_better or {}
    return {
        name: threshold_relation(metrics.get(name), limit, higher_is_better=directions.get(name, True))
        for name, limit in thresholds.items()
    }
