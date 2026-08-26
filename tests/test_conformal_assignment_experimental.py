import unittest
from fractions import Fraction

from sddiar.conformal_assignment_experimental import (
    ALGORITHM_VERSION,
    MAX_TOTAL_COORDINATE_WORK,
    AnchorBlock,
    CandidateBlock,
    ConformalAssignmentConfig,
    build_redacted_receipt,
    evaluate_bm_rcm,
    evaluate_bm_rcm_batch,
)


class BlockMondrianRelativeConformalTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "enabled": True,
            "epsilon": Fraction(1, 3),
            "min_quality": Fraction(1, 2),
            "min_clean_duration_us": 1,
        }
        values.update(changes)
        return ConformalAssignmentConfig(**values)

    @staticmethod
    def anchor(vector, block_id, speaker, *, duration=1_000, quality=Fraction(9, 10), **extra):
        return AnchorBlock(vector, block_id, speaker, True, quality, False, False, duration, ())

    @staticmethod
    def candidate(vector, block_id="candidate", *, duration=1_000, quality=Fraction(9, 10), **extra):
        return CandidateBlock(vector, block_id, True, quality, False, False, duration, ())

    def anchors(self):
        return [
            self.anchor((1.0, 0.0), "a-0", "A"),
            self.anchor((1.0, 0.04), "a-1", "A"),
            self.anchor((-1.0, 0.0), "b-0", "B"),
            self.anchor((-1.0, 0.40), "b-1", "B"),
        ]

    def test_cosine_oracle_singleton_midpoint_and_ambiguous_gamma(self):
        singleton = evaluate_bm_rcm(self.anchors(), self.candidate((1.0, 0.01)), self.config())
        self.assertEqual(singleton.decision, "SHADOW_SINGLETON")
        self.assertEqual(singleton.gamma, ("A",))

        # Symmetric midpoint has p=1/3 for both hypotheses.  v2 uses strict
        # p > epsilon, so equality is OOD rather than eligible.
        midpoint = evaluate_bm_rcm(self.anchors(), self.candidate((0.0, 1.0), "midpoint"), self.config())
        self.assertEqual(midpoint.decision, "OOD")
        self.assertEqual(midpoint.gamma, ())
        self.assertEqual(midpoint.p_values["A"], Fraction(1, 3))
        self.assertEqual(midpoint.p_values["B"], Fraction(1, 3))

        ambiguous_anchors = [
            self.anchor((1.0, 0.0), "a-0", "A"),
            self.anchor((1.0, 0.01), "a-1", "A"),
            self.anchor((1.0, 0.02), "a-2", "A"),
            self.anchor((1.0, 0.0), "b-0", "B"),
            self.anchor((1.0, 0.01), "b-1", "B"),
            self.anchor((1.0, 0.02), "b-2", "B"),
        ]
        ambiguous = evaluate_bm_rcm(
            ambiguous_anchors, self.candidate((0.0, 1.0), "ambiguous"), self.config(epsilon=Fraction(1, 4))
        )
        self.assertEqual(ambiguous.decision, "AMBIGUOUS")
        self.assertEqual(len(ambiguous.gamma), 2)

    def test_positive_scale_invariance_and_cross_backend_digest(self):
        first = evaluate_bm_rcm(self.anchors(), self.candidate((1.0, 0.01)), self.config())
        scale = 37.0 / 10.0
        scaled = [
            self.anchor(tuple(scale * value for value in item.vector), item.block_id, item.speaker_id)
            for item in self.anchors()
        ]
        second = evaluate_bm_rcm(scaled, self.candidate(tuple(scale * value for value in (1.0, 0.01))), self.config())
        self.assertEqual(first.decision, second.decision)
        self.assertEqual(first.input_digest, second.input_digest)
        self.assertEqual(first.receipt_hash, second.receipt_hash)

        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy is not installed")
        native = [
            AnchorBlock(np.asarray(item.vector, dtype=np.float64), item.block_id, item.speaker_id, True, item.quality, False, False, item.clean_duration_us, ())
            for item in self.anchors()
        ]
        np_result = evaluate_bm_rcm(native, CandidateBlock(np.asarray((1.0, 0.01)), "candidate", True, Fraction(9, 10), False, False, 1_000, ()), self.config())
        self.assertEqual(first.input_digest, np_result.input_digest)

    def test_fragment_split_invariance_and_tuple_ids(self):
        base = [
            self.anchor((1.0, 0.0), "block\x1fA", "A", duration=2_000),
            self.anchor((1.0, 0.04), "a-1", "A"),
            self.anchor((-1.0, 0.0), "block\x1fB", "B", duration=2_000),
            self.anchor((-1.0, 0.04), "b-1", "B"),
        ]
        split = [
            self.anchor((1.0, 0.0), "block\x1fA", "A", duration=1_000),
            self.anchor((1.0, 0.0), "block\x1fA", "A", duration=1_000),
            base[1], base[2], base[3],
        ]
        first = evaluate_bm_rcm(base, self.candidate((1.0, 0.01)), self.config())
        second = evaluate_bm_rcm(split, self.candidate((1.0, 0.01)), self.config())
        self.assertEqual(first.input_digest, second.input_digest)
        self.assertEqual(first.receipt_hash, second.receipt_hash)

        # IDs are tuple-keyed internally; a delimiter in an ID cannot collide
        # with a different speaker/block pair.
        collision_free = base + [self.anchor((1.0, 0.0), "block", "A")]
        self.assertNotEqual(evaluate_bm_rcm(collision_free, self.candidate((1.0, 0.01)), self.config()).decision, "FAIL_CLOSED")

    def test_invalid_anchor_and_candidate_contracts_fail_closed(self):
        bad_valid = self.anchors()
        bad_valid[0] = AnchorBlock(bad_valid[0].vector, "bad", "A", None, bad_valid[0].quality, False, False, 1_000, ())
        self.assertEqual(evaluate_bm_rcm(bad_valid, self.candidate((1.0, 0.0)), self.config()).decision, "FAIL_CLOSED")

        bad_quality = self.anchors()
        bad_quality[0] = self.anchor((1.0, 0.0), "bad-quality", "A", quality=Fraction(1, 10))
        self.assertEqual(evaluate_bm_rcm(bad_quality, self.candidate((1.0, 0.0)), self.config()).decision, "FAIL_CLOSED")

        bad_candidate = self.candidate((1.0, 0.0), quality=Fraction(1, 10))
        self.assertEqual(evaluate_bm_rcm(self.anchors(), bad_candidate, self.config()).decision, "FAIL_CLOSED")
        flags = self.candidate((1.0, 0.0), "too-many-flags")
        flags = CandidateBlock(flags.vector, flags.block_id, True, flags.quality, False, False, flags.clean_duration_us, tuple(range(33)))
        self.assertEqual(evaluate_bm_rcm(self.anchors(), flags, self.config()).decision, "FAIL_CLOSED")
        self.assertEqual(evaluate_bm_rcm(self.anchors(), self.candidate((0.0, 0.0), "zero"), self.config()).decision, "FAIL_CLOSED")

    def test_cancelling_centroid_and_arbitrary_weight_rejected(self):
        cancelling = [
            self.anchor((1.0, 0.0), "a-0", "A"),
            self.anchor((-1.0, 0.0), "a-1", "A"),
            self.anchor((-1.0, 0.0), "b-0", "B"),
            self.anchor((-1.0, 0.04), "b-1", "B"),
        ]
        self.assertEqual(evaluate_bm_rcm(cancelling, self.candidate((1.0, 0.0)), self.config()).decision, "FAIL_CLOSED")
        with self.assertRaises(TypeError):
            AnchorBlock((1.0, 0.0), "x", "A", True, Fraction(9, 10), False, False, 1_000, (), 7)  # type: ignore[arg-type]

    def test_large_prepared_calibration_and_batch_coordinate_budget(self):
        anchors = []
        dimension = 256
        for index in range(364):
            vector = [0.0] * dimension
            vector[0] = 1.0 if index < 182 else -1.0
            vector[1] = (index % 7) / 1000.0
            speaker = "A" if index < 182 else "B"
            anchors.append(self.anchor(tuple(vector), f"block-{index}", speaker))
        candidates = [self.candidate((1.0, (index % 5) / 1000.0) + (0.0,) * (dimension - 2), f"candidate-{index}") for index in range(8)]
        results = evaluate_bm_rcm_batch(anchors, candidates, self.config())
        self.assertEqual(len(results), len(candidates))
        self.assertTrue(all(item.decision != "FAIL_CLOSED" for item in results))
        self.assertLessEqual(
            364 * dimension + len(candidates) * 364 * dimension,
            MAX_TOTAL_COORDINATE_WORK,
        )
        oversized = evaluate_bm_rcm_batch(
            anchors,
            [self.candidate((1.0, 0.0) + (0.0,) * (dimension - 2), f"over-{index}") for index in range(16)],
            self.config(),
        )
        self.assertTrue(all(item.decision == "FAIL_CLOSED" for item in oversized))

    def test_disabled_parity_and_redacted_receipt(self):
        result = evaluate_bm_rcm((), {"vector": object(), "block_id": "raw"}, ConformalAssignmentConfig())
        self.assertEqual(result.decision, "DISABLED")
        receipt = build_redacted_receipt(result)
        self.assertEqual(receipt["schema"], "bm-rcm-v2")
        self.assertEqual(receipt["algorithm"], ALGORITHM_VERSION)
        self.assertNotIn("vector", str(receipt))


if __name__ == "__main__":
    unittest.main()
