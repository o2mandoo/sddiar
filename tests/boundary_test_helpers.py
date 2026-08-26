"""Test-only construction of sealed boundary values.

Production callers have no supported minting API until signed per-job boundary
evidence is implemented.  Unit tests use the private token only to exercise
the downstream interval mechanics.
"""

from sddiar.segmentation import (
    _ENFORCEMENT_TOKEN,
    _EnforceableOverlapEvent,
    _EnforceableSpeakerChangeEvent,
)


def sealed_scd(time_us, evidence, evidence_id, *, source_id="audio"):
    return _EnforceableSpeakerChangeEvent(
        _token=_ENFORCEMENT_TOKEN,
        time_us=time_us,
        evidence=evidence,
        evidence_id=evidence_id,
        source_id=source_id,
        calibration_profile_id="test-only",
        binding_sha256="f" * 64,
    )


def sealed_osd(start_us, end_us, evidence, evidence_ids, *, source_id="audio"):
    return _EnforceableOverlapEvent(
        _token=_ENFORCEMENT_TOKEN,
        start_us=start_us,
        end_us=end_us,
        overlap_evidence=evidence,
        evidence_ids=tuple(evidence_ids),
        source_id=source_id,
        calibration_profile_id="test-only",
        binding_sha256="f" * 64,
    )
