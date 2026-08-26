"""Fail-safe, explainable file quality rules (no learned quality model)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .calibration import CalibrationBinding, VerifiedCalibrationBinding


@dataclass(frozen=True, slots=True)
class QualityConfig:
    unknown_ratio_warn: float = .20
    overlap_ratio_warn: float = .10
    unresolved_micro_ratio_warn: float = .10


def _get(x: Any, key: str, default: Any = None) -> Any:
    if isinstance(x, Mapping): return x.get(key, default)
    return getattr(x, key, default)


def _report(status: str, speaker_count_status: str, summary_mode: str,
            reasons: list[str], diagnostics: Any, calibration: Any) -> Any:
    metrics = _get(diagnostics, "metrics", {}) or {}
    relations = _get(diagnostics, "threshold_relations", {}) or {}
    fields = dict(status=status, speaker_count_status=speaker_count_status,
                  summary_mode=summary_mode, reason_codes=tuple(dict.fromkeys(reasons)),
                  metrics=dict(metrics), threshold_relations=dict(relations),
                  calibration_profile_id=_get(calibration, "profile_id", None))
    try:
        from .contracts import FileQualityReport
        return FileQualityReport(**fields)
    except ImportError:
        from dataclasses import make_dataclass
        return make_dataclass("FileQualityReport", [(k, type(v)) for k, v in fields.items()])(**fields)


def _calibration_reason_codes(calibration: Any, diagnostics: Any) -> tuple[str, ...]:
    """Return why calibration cannot authorize PASS, or an empty tuple.

    Exact-type checking is intentional: raw profiles, duck-typed objects,
    legacy bindings, and subclasses cannot manufacture verification authority.
    """
    if calibration is None:
        return ("Q_CALIBRATION_MISSING",)
    if type(calibration) is VerifiedCalibrationBinding:
        if not calibration.release_authorized:
            return ("Q_CALIBRATION_UNVERIFIED",)
        return calibration.mismatch_reason_codes(diagnostics)
    if type(calibration) is CalibrationBinding:
        return calibration.reason_codes or ("Q_CALIBRATION_UNVERIFIED",)
    return ("Q_CALIBRATION_UNVERIFIED",)


def evaluate_file_quality(diagnostics: Any, calibration: Any = None,
                          config: Any = None) -> Any:
    """Return policy status; only a release-verified exact binding can PASS."""
    cfg = config or QualityConfig()
    metrics = _get(diagnostics, "metrics", {}) or {}
    # Invalid measured values are a caller/configuration failure, not evidence
    # for a safe pass. Keep report serializable and conservative.
    bad_metric = any(isinstance(v, float) and not math.isfinite(v) for v in metrics.values())
    reasons: list[str] = []
    if bad_metric: reasons.append("Q_INVALID_METRIC")
    # Confirmed negative evidence is safe to report without calibration and
    # preserves the existing neutral-only behavior for hard out-of-profile
    # inputs. Calibration authority is required only for PASS states.
    if _get(diagnostics, "confirmed_hard_out_of_profile", False):
        reasons.extend(_get(diagnostics, "out_of_profile_reasons", ()) or ())
        if not reasons: reasons.append("Q_CONFIRMED_OUT_OF_PROFILE_SPEAKER_COUNT")
        verified = (
            calibration
            if (
                type(calibration) is VerifiedCalibrationBinding
                and calibration.release_authorized
                and not calibration.mismatch_reason_codes(diagnostics)
            )
            else None
        )
        return _report("UNSUPPORTED", "OUT_OF_PROFILE", "SPEAKER_NEUTRAL", reasons, diagnostics, verified)

    calibration_reasons = _calibration_reason_codes(calibration, diagnostics)
    if calibration_reasons:
        reasons.extend(calibration_reasons)
        reasons.extend(_get(diagnostics, "review_reasons", ()) or ())
        return _report("REVIEW_REQUIRED", "UNCERTAIN_1_OR_2", "MANUAL_REVIEW", reasons, diagnostics, None)

    count = _get(diagnostics, "speaker_count_status", "UNCERTAIN_1_OR_2")
    review_reasons = list(_get(diagnostics, "review_block_reasons", ()) or ())
    review_reasons.extend(_get(diagnostics, "review_reasons", ()) or ())
    if _get(diagnostics, "hypothesis_uncertain", False): review_reasons.append("Q_H1_H2_AMBIGUOUS")
    if review_reasons:
        return _report("REVIEW_REQUIRED", count, "MANUAL_REVIEW", review_reasons, diagnostics, calibration)

    degrade = list(_get(diagnostics, "unattributed_degrade_reasons", ()) or ())
    degrade.extend(_get(diagnostics, "unattributed_reasons", ()) or ())
    if _get(diagnostics, "word_boundary_crossing_present", False): degrade.append("Q_WORD_BOUNDARY_CROSSING_PRESENT")
    if _get(diagnostics, "overlap_unattributed_present", False): degrade.append("Q_OVERLAP_UNATTRIBUTED_PRESENT")
    unknown = _get(diagnostics, "unknown_ratio", metrics.get("unknown_ratio", 0.0))
    overlap = _get(diagnostics, "overlap_ratio", metrics.get("overlap_ratio", 0.0))
    if unknown > float(_get(cfg, "unknown_ratio_warn", .2)): degrade.append("Q_HIGH_UNKNOWN_RATIO")
    if overlap > float(_get(cfg, "overlap_ratio_warn", .1)): degrade.append("Q_HIGH_OVERLAP_RATIO")
    if degrade:
        return _report("PASS_WITH_UNATTRIBUTED", count, "SPEAKER_NEUTRAL", degrade, diagnostics, calibration)

    if bool(_get(diagnostics, "all_high_rules_pass", False)) and _get(diagnostics, "osd_coverage", "EVALUATED") != "NOT_EVALUATED":
        return _report("PASS_HIGH", count, "SPEAKER_AWARE", [], diagnostics, calibration)
    warnings = list(_get(diagnostics, "standard_warnings", ()) or ())
    return _report("PASS_STANDARD", count, "SPEAKER_AWARE", warnings, diagnostics, calibration)


class RuleBasedQualityGate:
    def __init__(self, config: Any = None): self.config = config or QualityConfig()
    def evaluate(self, diagnostics: Any, calibration: Any = None) -> Any:
        return evaluate_file_quality(diagnostics, calibration, self.config)
    __call__ = evaluate


evaluate = evaluate_file_quality
evaluate_quality = evaluate_file_quality
quality_gate = evaluate_file_quality
