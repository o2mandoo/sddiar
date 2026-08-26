import base64
from dataclasses import fields
import hashlib
import unittest

from sddiar.boundary_evidence import (
    BoundaryGateConfig,
    BoundaryModelCandidate,
    BoundaryOverlapVeto,
    BoundaryProbe,
    DualEvidenceBoundaryGate,
)
from sddiar.calibration import (
    CalibrationProfileVerifier,
    DigestCalibrationSignatureVerifier,
    VerifiedCalibrationBinding,
    canonical_calibration_bytes,
)
from sddiar.contracts import ContractValidationError
from sddiar.diarization import DiarizationConfig, build_tracklets
from sddiar.segmentation import (
    OverlapEvent,
    SpeakerChangeEvent,
    authorize_overlap_event,
    authorize_scd_event,
    clip_and_coalesce_speech_mask,
)
from boundary_test_helpers import sealed_osd, sealed_scd


class ReleaseVerifier(DigestCalibrationSignatureVerifier):
    trust_level = "RELEASE"


def release_binding() -> VerifiedCalibrationBinding:
    key = b"boundary-test-key"
    model_hash = "a" * 64
    config_hash = "d" * 64
    profile = {
        "schema_version": "1",
        "profile_id": "boundary-profile",
        "calibration_version": "2026-08-26.1",
        "model_hashes": {"boundary-model": model_hash},
        "source_sample_rates": [8000],
        "thresholds": {"scd_evidence_min": 0.5, "osd_evidence_min": 0.5},
        "dataset_manifest_hash": "b" * 64,
        "scorer_hash": "c" * 64,
        "config_hash": config_hash,
        "approver": "boundary-approval",
        "provenance": {
            "annotation_schema_version": "1",
            "created_at": "2026-08-26T00:00:00Z",
            "model_pack_id": "boundary-pack",
            "pipeline_version": "boundary-pipeline",
            "safety_constraints": ["shadow-only"],
            "selection_objective": "minimize unsafe boundary enforcement",
        },
        "signer_key_id": "boundary-key",
    }
    profile["signature"] = base64.b64encode(
        hashlib.sha256(key + canonical_calibration_bytes(profile)).digest()
    ).decode("ascii")
    return CalibrationProfileVerifier(ReleaseVerifier(key)).verify(
        profile,
        model_hashes={"boundary-model": model_hash},
        source_sample_rate=8000,
        config_hash=config_hash,
        profile_id="boundary-profile",
    )


def probe(name, vector, block, *, start, end, source="audio", view="mix", config="cfg", quality=0.9, clean=300_000, embedding="embed-v1", overlap=False):
    return BoundaryProbe(
        name, vector, quality, clean, source, view, embedding, block,
        start, end, config, overlap,
    )


class BoundaryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.binding = release_binding()
        self.config = BoundaryGateConfig(enabled=True, embedding_dimension=2, config_identity="cfg")
        self.gate = DualEvidenceBoundaryGate(self.config)
        self.candidate = BoundaryModelCandidate(1_000_000, 0.9, "candidate-1", "audio", "mix", "model-v1", "cfg")
        self.left = (
            probe("l0", (1.0, 0.0), "left-0", start=0, end=300_000),
            probe("l1", (0.99, 0.1), "left-1", start=300_000, end=600_000),
        )
        self.right = (
            probe("r0", (0.0, 1.0), "right-0", start=1_000_000, end=1_300_000),
            probe("r1", (0.1, 0.99), "right-1", start=1_300_000, end=1_600_000),
        )

    def test_default_off_and_positive_is_shadow_only(self):
        self.assertEqual(DualEvidenceBoundaryGate().evaluate(self.candidate, self.left, self.right).decision, "REJECT")
        result = self.gate.evaluate(self.candidate, self.left, self.right)
        self.assertEqual(result.decision, "SHADOW_APPROVE_CANDIDATE")
        self.assertIsNone(result.approved_event)

    def test_model_only_embedding_only_quality_and_independence_reject(self):
        self.assertEqual(self.gate.evaluate(self.candidate).decision, "REJECT")
        self.assertEqual(self.gate.evaluate(None, self.left, self.right).decision, "REJECT")
        low = (probe("l0", (1.0, 0.0), "left-0", start=0, end=300_000, quality=0.1), self.left[1])
        self.assertEqual(self.gate.evaluate(self.candidate, low, self.right).decision, "REJECT")
        same_block = (
            probe("l0", (1.0, 0.0), "same", start=0, end=300_000),
            probe("l1", (0.99, 0.1), "same", start=300_000, end=600_000),
        )
        self.assertEqual(self.gate.evaluate(self.candidate, same_block, self.right).decision, "REJECT")

    def test_lineage_side_geometry_and_overlap_reject(self):
        wrong_source = (
            probe("l0", (1.0, 0.0), "left-0", start=0, end=300_000, source="other"),
            self.left[1],
        )
        self.assertEqual(self.gate.evaluate(self.candidate, wrong_source, self.right).reason, "SOURCE_VIEW_CONFIG_LINEAGE_MISMATCH")
        same_block = (
            probe("r0", (1.0, 0.0), "left-0", start=1_000_000, end=1_300_000),
            probe("r1", (0.99, 0.1), "right-1", start=1_300_000, end=1_600_000),
        )
        self.assertEqual(self.gate.evaluate(self.candidate, self.left, same_block).reason, "INDEPENDENT_BLOCKS_NOT_DISJOINT")
        close = (
            probe("r0", (1.0, 0.0), "right-0", start=1_000_000, end=1_300_000),
            probe("r1", (0.99, 0.1), "right-1", start=1_300_000, end=1_600_000),
        )
        self.assertEqual(self.gate.evaluate(self.candidate, self.left, close).decision, "REJECT")
        veto = BoundaryOverlapVeto("ov", "audio", "mix", "cfg", 900_000, 1_100_000)
        self.assertEqual(self.gate.evaluate(self.candidate, self.left, self.right, overlap_regions=(veto,)).reason, "OVERLAP_VETO")
        self.assertEqual(self.gate.evaluate(self.candidate, self.left, self.right, overlap_regions=({"start_us": 0},)).reason, "MALFORMED_OVERLAP_VETO")

    def test_separation_margin_and_global_dimension_reject(self):
        strict = DualEvidenceBoundaryGate(BoundaryGateConfig(enabled=True, embedding_dimension=2, config_identity="cfg", separation_margin_min=1.1))
        self.assertEqual(strict.evaluate(self.candidate, self.left, self.right).reason, "SEPARATION_MARGIN_NOT_MET")
        bad_dimension = BoundaryProbe(
            "r1", (0.0, 1.0, 0.0), 0.9, 300_000, "audio", "mix", "embed-v1", "right-1", 1_300_000, 1_600_000, "cfg"
        )
        self.assertEqual(self.gate.evaluate(self.candidate, self.left, (self.right[0], bad_dimension)).reason, "VECTOR_DIMENSION_OR_NORM_INVALID")

    def test_release_binding_alone_cannot_mint_per_job_boundary_evidence(self):
        with self.assertRaises(ContractValidationError):
            authorize_scd_event(self.binding, time_us=1_000_000, evidence=0.9, evidence_id="scd", source_id="audio", scd_evidence_min=0.5)
        with self.assertRaises(ContractValidationError):
            authorize_overlap_event(self.binding, start_us=500_000, end_us=700_000, overlap_evidence=0.9, evidence_ids=("osd",), source_id="audio", osd_evidence_min=0.5)
        with self.assertRaises(ContractValidationError):
            authorize_scd_event(object(), time_us=1, evidence=1.0, evidence_id="x", source_id="audio", scd_evidence_min=0.5)
        with self.assertRaises(ContractValidationError):
            authorize_scd_event(self.binding, time_us=1, evidence=0.4, evidence_id="x", source_id="audio", scd_evidence_min=0.5)
        with self.assertRaises(ContractValidationError):
            SpeakerChangeEvent(1, 1.0, "x", approved=True)
        with self.assertRaises(ContractValidationError):
            OverlapEvent(1, 2, 1.0, ("x",), is_high=True)
        self.assertIsInstance(SpeakerChangeEvent(1, 0.5, "diag"), SpeakerChangeEvent)

    def test_receipt_is_deterministic_and_true_scalar(self):
        one = self.gate.evaluate(self.candidate, self.left, self.right)
        two = self.gate.evaluate(self.candidate, self.left, self.right)
        self.assertEqual(one.receipt, two.receipt)
        scalar = (str, int, float, type(None), bool)
        for item in fields(one.receipt):
            self.assertIsInstance(getattr(one.receipt, item.name), scalar, item.name)
        self.assertEqual(len(one.receipt.source_key), 16)

    def test_clip_and_coalesce_source_bounds(self):
        self.assertEqual(
            clip_and_coalesce_speech_mask(1_000, [(-100, 100), (120, 200), (250, 500), (2_000, 2_100)], merge_gap_us=49),
            ((0, 200), (250, 500)),
        )

    def test_tracklet_builder_rejects_diagnostic_and_mapping_capabilities(self):
        with self.assertRaises(ContractValidationError):
            build_tracklets(({"start_us": 0, "end_us": 2_000_000},), scd_events=(SpeakerChangeEvent(1_000_000, 0.9, "diag"),))
        with self.assertRaises(ContractValidationError):
            build_tracklets(({"start_us": 0, "end_us": 2_000_000},), scd_events=({"time_us": 1_000_000, "evidence": 1.0, "approved": True},))
        with self.assertRaises(ContractValidationError):
            build_tracklets(({"start_us": 0, "end_us": 2_000_000},), overlap_regions=(OverlapEvent(500_000, 1_000_000, 1.0),))

    def test_tracklet_builder_accepts_only_release_authorized_values(self):
        scd = sealed_scd(1_000_000, 0.9, "approved")
        osd = sealed_osd(500_000, 700_000, 0.9, ("osd",))
        result = build_tracklets(
            ({"start_us": 0, "end_us": 2_000_000},), scd_events=(scd,), overlap_regions=(osd,),
            cfg=DiarizationConfig(min_split_side_us=100_000),
        )
        self.assertEqual([(item.start_us, item.end_us) for item in result.tracklets], [(0, 500_000), (700_000, 1_000_000), (1_000_000, 2_000_000)])


if __name__ == "__main__":
    unittest.main()
