import base64
import hashlib
import unittest
from types import SimpleNamespace as S

from sddiar.attribution import map_word, map_words
from sddiar.calibration import (
    CalibrationBinding,
    CalibrationProfile,
    CalibrationProfileVerifier,
    DigestCalibrationSignatureVerifier,
    bind_calibration_profile,
    canonical_calibration_bytes,
)
from sddiar.contracts import DiarizationSpan, Word, WordProvenance, WordTimeline
from sddiar.quality import evaluate_file_quality


def word(start: int = 0, end: int = 100, ident: str = "w") -> Word:
    return Word(ident, start, end, "hello")


class OfflineReleaseSignatureVerifier(DigestCalibrationSignatureVerifier):
    trust_level = "RELEASE"


class AttributionQualityTests(unittest.TestCase):
    calibration_key = b"quality-calibration-key"
    model_hash = "a" * 64
    config_hash = "d" * 64

    def verified_calibration(self, *, release: bool = True):
        profile = {
            "schema_version": "1",
            "profile_id": "cal-v1",
            "calibration_version": "1",
            "model_hashes": {"embed": self.model_hash},
            "source_sample_rates": [16000],
            "thresholds": {"unknown_ratio_warn": 0.2},
            "dataset_manifest_hash": "b" * 64,
            "scorer_hash": "c" * 64,
            "config_hash": self.config_hash,
            "approver": "approval-1",
            "provenance": {
                "annotation_schema_version": "1",
                "created_at": "2026-08-26T00:00:00Z",
                "model_pack_id": "pack-1",
                "pipeline_version": "pipeline-1",
                "safety_constraints": ["no false speaker PASS"],
                "selection_objective": "minimize unsafe attribution",
            },
            "signer_key_id": "release-key",
        }
        signature = hashlib.sha256(
            self.calibration_key + canonical_calibration_bytes(profile)
        ).digest()
        profile["signature"] = base64.b64encode(signature).decode("ascii")
        verifier_type = OfflineReleaseSignatureVerifier if release else DigestCalibrationSignatureVerifier
        return CalibrationProfileVerifier(verifier_type(self.calibration_key)).verify(
            profile,
            model_hashes={"embed": self.model_hash},
            source_sample_rate=16000,
            config_hash=self.config_hash,
        )

    def quality_diagnostics(self, **changes):
        values = {
            "metrics": {},
            "speaker_count_status": "CONFIDENT_2",
            "all_high_rules_pass": True,
            "osd_coverage": "EVALUATED",
            "calibration_profile_id": "cal-v1",
            "model_hashes": {"embed": self.model_hash},
            "source_sample_rate_hz": 16000,
            "config_hash": self.config_hash,
        }
        values.update(changes)
        return S(**values)

    def test_timewarp_boundary_is_unknown(self) -> None:
        out = map_word(
            word(),
            [DiarizationSpan("s", 0, 100, "SPEAKER_00", "ASSIGNED")],
            WordProvenance("w", crosses_timewarp_boundary=True),
        )
        self.assertEqual((out.speaker_id, out.attribution_status), ("UNKNOWN", "UNKNOWN_TIMEWARP_BOUNDARY"))

    def test_overlap_word_is_unattributed(self) -> None:
        out = map_word(word(), [DiarizationSpan("o", 20, 80, "OVERLAP", "OVERLAP")], WordProvenance("w"))
        self.assertEqual((out.speaker_id, out.attribution_status), ("OVERLAP", "OVERLAP_UNATTRIBUTED"))

    def test_word_crossing_speaker_boundary_is_unknown(self) -> None:
        spans = [
            DiarizationSpan("a", 0, 50, "SPEAKER_00", "ASSIGNED"),
            DiarizationSpan("b", 50, 100, "SPEAKER_01", "ASSIGNED"),
        ]
        out = map_word(word(), spans, WordProvenance("w"))
        self.assertEqual((out.speaker_id, out.attribution_status), ("UNKNOWN", "UNKNOWN_BOUNDARY"))

    def test_dominant_speaker_assignment_requires_whole_interval_evidence(self) -> None:
        spans = [
            DiarizationSpan("a", 0, 95, "SPEAKER_00", "ASSIGNED"),
            DiarizationSpan("u", 95, 100, "UNKNOWN", "UNKNOWN_INSUFFICIENT_EVIDENCE"),
        ]
        out = map_words(WordTimeline((word(),), {"w": WordProvenance("w")}), spans)[0]
        self.assertEqual((out.speaker_id, out.attribution_status, out.speaker_coverage_ratio), ("SPEAKER_00", "ASSIGNED", 0.95))

    def test_quality_without_calibration_is_review_and_never_pass(self) -> None:
        out = evaluate_file_quality(S(metrics={}, all_high_rules_pass=True))
        self.assertEqual(out.status, "REVIEW_REQUIRED")
        self.assertIn("Q_CALIBRATION_MISSING", out.reason_codes)

    def test_profile_id_raw_profile_and_legacy_binding_never_authorize_pass(self) -> None:
        raw = CalibrationProfile.from_mapping({
            "profile_id": "cal-v1",
            "model_hashes": {"embed": "abc"},
            "source_sample_rates": [16000],
        })
        legacy = CalibrationBinding(raw, True)
        for calibration in (S(profile_id="cal-v1"), raw, legacy, {"profile_id": "cal-v1"}):
            with self.subTest(calibration=type(calibration).__name__):
                out = evaluate_file_quality(self.quality_diagnostics(), calibration)
                self.assertEqual((out.status, out.summary_mode), ("REVIEW_REQUIRED", "MANUAL_REVIEW"))
                self.assertIn("Q_CALIBRATION_UNVERIFIED", out.reason_codes)
                self.assertIsNone(out.calibration_profile_id)

    def test_legacy_binder_compatibility_still_fails_closed_at_gate(self) -> None:
        legacy = bind_calibration_profile({
            "profile_id": "cal-v1",
            "model_hashes": {"embed": "abc"},
            "source_sample_rates": [16000],
        }, model_hashes={"embed": "abc"}, source_sample_rate=16000)
        self.assertTrue(legacy)
        out = evaluate_file_quality(self.quality_diagnostics(), legacy)
        self.assertEqual(out.status, "REVIEW_REQUIRED")
        self.assertIn("Q_CALIBRATION_UNVERIFIED", out.reason_codes)

    def test_only_release_verified_and_runtime_bound_calibration_can_pass(self) -> None:
        calibration = self.verified_calibration()
        out = evaluate_file_quality(self.quality_diagnostics(), calibration)
        self.assertEqual((out.status, out.summary_mode), ("PASS_HIGH", "SPEAKER_AWARE"))
        self.assertEqual(out.calibration_profile_id, "cal-v1")

        unbound = evaluate_file_quality(
            S(metrics={}, speaker_count_status="CONFIDENT_2", all_high_rules_pass=True),
            calibration,
        )
        self.assertEqual(unbound.status, "REVIEW_REQUIRED")
        self.assertIn("Q_CALIBRATION_UNBOUND", unbound.reason_codes)

    def test_verified_binding_runtime_mismatch_reason_is_preserved(self) -> None:
        out = evaluate_file_quality(
            self.quality_diagnostics(config_hash="e" * 64),
            self.verified_calibration(),
        )
        self.assertEqual(out.status, "REVIEW_REQUIRED")
        self.assertIn("Q_CALIBRATION_CONFIG_HASH_MISMATCH", out.reason_codes)
        self.assertNotIn("Q_CALIBRATION_MISSING", out.reason_codes)

    def test_development_verified_binding_never_authorizes_pass(self) -> None:
        out = evaluate_file_quality(
            self.quality_diagnostics(),
            self.verified_calibration(release=False),
        )
        self.assertEqual(out.status, "REVIEW_REQUIRED")
        self.assertIn("Q_CALIBRATION_UNVERIFIED", out.reason_codes)

    def test_quality_hard_out_of_profile_is_unsupported(self) -> None:
        calibration = S(profile_id="cal-v1")
        out = evaluate_file_quality(
            S(metrics={}, confirmed_hard_out_of_profile=True, out_of_profile_reasons=("Q_CONFIRMED_OUT_OF_PROFILE_SPEAKER_COUNT",)),
            calibration,
        )
        self.assertEqual((out.status, out.speaker_count_status, out.summary_mode), ("UNSUPPORTED", "OUT_OF_PROFILE", "SPEAKER_NEUTRAL"))


if __name__ == "__main__":
    unittest.main()
