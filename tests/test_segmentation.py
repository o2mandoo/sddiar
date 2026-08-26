import base64
import hashlib
import unittest

from sddiar.calibration import CalibrationProfileVerifier, DigestCalibrationSignatureVerifier, canonical_calibration_bytes
from sddiar.segmentation import (
    OverlapEvent,
    ProbeEvidence,
    RuleEvidenceSegmentation,
    SegmentationConfig,
    SpeakerChangeEvent,
    authorize_overlap_event,
    authorize_scd_event,
)
from sddiar.vad import VadFrame
from boundary_test_helpers import sealed_osd, sealed_scd


class ReleaseVerifier(DigestCalibrationSignatureVerifier):
    trust_level = "RELEASE"


def release_binding():
    key = b"segmentation-test-key"
    profile = {
        "schema_version": "1", "profile_id": "p", "calibration_version": "v1",
        "model_hashes": {"m": "a" * 64}, "source_sample_rates": [8000],
        "thresholds": {"scd_evidence_min": 0.5, "osd_evidence_min": 0.5},
        "dataset_manifest_hash": "b" * 64, "scorer_hash": "c" * 64, "config_hash": "d" * 64,
        "approver": "a", "provenance": {
            "annotation_schema_version": "1", "created_at": "now", "model_pack_id": "m",
            "pipeline_version": "p", "safety_constraints": ["shadow"], "selection_objective": "safe",
        }, "signer_key_id": "k",
    }
    profile["signature"] = base64.b64encode(hashlib.sha256(key + canonical_calibration_bytes(profile)).digest()).decode()
    return CalibrationProfileVerifier(ReleaseVerifier(key)).verify(
        profile, model_hashes={"m": "a" * 64}, source_sample_rate=8000, config_hash="d" * 64, profile_id="p"
    )


class SegmentationTests(unittest.TestCase):
    def test_vad_frames_merge_into_source_time_regions(self) -> None:
        segmentation = RuleEvidenceSegmentation(SegmentationConfig(vad_merge_gap_us=50))
        evidence = segmentation.build(
            view_id="v",
            vad_frames=(VadFrame(0, 100, 0.9, True), VadFrame(120, 200, 0.8, True), VadFrame(400, 500, 0.8, True)),
        )
        self.assertEqual([(region.start_us, region.end_us) for region in evidence.speech_regions], [(0, 200), (400, 500)])

    def test_probe_discontinuity_is_diagnostic_not_unapproved_scd_cut(self) -> None:
        segmentation = RuleEvidenceSegmentation(SegmentationConfig(probe_discontinuity_min=0.2))
        evidence = segmentation.build(
            view_id="v",
            vad_frames=(VadFrame(0, 1_000, 1.0, True),),
            probes=(
                ProbeEvidence("a", 0, 400, 400, (1.0, 0.0), 1.0),
                ProbeEvidence("b", 400, 800, 400, (-1.0, 0.0), 1.0),
            ),
        )
        self.assertEqual(evidence.scd_events, ())
        self.assertEqual(len(evidence.probe_discontinuities), 1)

    def test_approved_scd_and_overlap_are_preserved(self) -> None:
        segmentation = RuleEvidenceSegmentation()
        approved = sealed_scd(500, 0.9, "scd", source_id="v")
        overlap = sealed_osd(600, 800, 0.9, ("osd",), source_id="v")
        evidence = segmentation.build(view_id="v", vad_frames=(), approved_scd_events=(approved,), approved_overlap_regions=(overlap,))
        self.assertEqual(evidence.scd_events, (approved,))
        self.assertEqual(evidence.overlap_regions, (overlap,))


if __name__ == "__main__":
    unittest.main()
