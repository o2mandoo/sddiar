import unittest
from sddiar import *
from sddiar.errors import ContractValidationError, ProtectedOverlapError, TimebaseInvariantViolation

class ContractsTimebaseTests(unittest.TestCase):
    def test_timewarp_affine_and_range(self):
        w = TimeWarp((TimeWarpSegment("s", "v", 0, 16000, 0, 1_000_000),))
        self.assertEqual(w.map_range(8000, 12000), (500_000, 750_000))

    def test_timewarp_uses_integer_half_up_rounding(self):
        w = TimeWarp((TimeWarpSegment("s", "v", 0, 3, 0, 10),))
        self.assertEqual((w.map_sample(1), w.map_sample(2)), (3, 7))
    def test_timewarp_rejects_overlap(self):
        with self.assertRaises(TimebaseInvariantViolation):
            TimeWarp((TimeWarpSegment("a", "v", 0, 10, 0, 100), TimeWarpSegment("b", "v", 9, 20, 90, 200)))
    def test_words_need_source_int_us_and_provenance(self):
        word = Word("w", 0, 20, "안녕")
        timeline = WordTimeline((word,), {"w": WordProvenance("w")})
        self.assertEqual(timeline.words[0].text, "안녕")
        with self.assertRaises(ContractValidationError): Word("bad", 0.1, 2, "x")
    def test_overlap_is_protected(self):
        with self.assertRaises(ProtectedOverlapError):
            Tracklet("t", "r", "g", 0, 100, 50, "ANCHOR", protected_overlap=True)
        with self.assertRaises(ProtectedOverlapError):
            AttributedWord("w", 0, 10, "x", speaker_id="OVERLAP", attribution_status="ASSIGNED")
    def test_deterministic_id(self):
        h = "a" * 64
        self.assertEqual(deterministic_id(h, "word", 0, 10, 0), deterministic_id(h, "word", 0, 10, 0))

    def test_embedding_contract_requires_valid_l2_vector(self):
        embedding = EmbeddingResult("e", "t", True, (1.0, 0.0), dimension=2)
        self.assertTrue(embedding.is_valid)
        with self.assertRaises(ContractValidationError):
            EmbeddingResult("e", "t", True, (2.0, 0.0), dimension=2)
        with self.assertRaises(ContractValidationError):
            EmbeddingResult("e", "t", False, (1.0, 0.0), failure_reason="bad")

    def test_pass_quality_requires_calibration_profile(self):
        with self.assertRaises(ContractValidationError):
            FileQualityReport("PASS_HIGH", "CONFIDENT_1", "SPEAKER_AWARE", (), {})
        report = FileQualityReport("REVIEW_REQUIRED", "UNCERTAIN_1_OR_2", "MANUAL_REVIEW", ("Q_CALIBRATION_MISSING",), {})
        self.assertEqual(report.status, "REVIEW_REQUIRED")

if __name__ == "__main__": unittest.main()
