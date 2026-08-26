from __future__ import annotations

import json
import time
import unittest
from unittest.mock import patch

from sddiar.proxy_stability import SCHEMA, ProxyStabilityError, audit


class ProxyStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonical = {
            "state": "H1_CONFIRMED",
            "spans": [
                {"span_id": "opaque-a", "start_us": 0, "end_us": 1_000_000, "speaker_id": "A", "central": True},
                {"span_id": "opaque-b", "start_us": 1_000_000, "end_us": 2_000_000, "speaker_id": "B", "central": True},
                {"span_id": "opaque-c", "start_us": 2_000_000, "end_us": 3_000_000, "speaker_id": "UNKNOWN", "central": True},
            ],
        }
        self.variant = {
            "state": "H2_CONFIRMED",
            "spans": [
                {"span_id": "opaque-a", "start_us": 0, "end_us": 1_000_000, "speaker_id": "X", "central": True, "text": "must not appear"},
                {"span_id": "opaque-b", "start_us": 1_000_000, "end_us": 2_000_000, "speaker_id": "Y", "central": True},
                {"span_id": "opaque-c", "start_us": 2_000_000, "end_us": 3_000_000, "speaker_id": "UNKNOWN", "central": True},
            ],
        }

    def test_permutation_invariant_metrics_and_fixed_authority(self) -> None:
        report = audit(self.canonical, [self.variant], clova={"score": 0.4})
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["release_authority"], "none")
        self.assertEqual(report["quality_status"], "REVIEW_REQUIRED")
        metrics = report["metrics"]
        self.assertEqual(metrics["co_assigned_agreement"]["mean"], 1.0)
        self.assertEqual(metrics["speaker_flip_rate"]["mean"], 0.0)
        self.assertEqual(metrics["canonical_attribution_retention"]["mean"], 1.0)
        self.assertEqual(metrics["chunk_central_agreement"]["mean"], 1.0)
        self.assertEqual(metrics["unknown_delta"]["mean"], 0.0)
        self.assertEqual(metrics["h1_h2_state_change"]["changed_count"], 1)
        self.assertFalse(report["clova"]["candidate_selection_allowed"])
        encoded = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("must not appear", encoded)
        self.assertNotIn("opaque-a", encoded)
        self.assertNotIn("opaque-b", encoded)

    def test_speech_iou_boundary_and_unknown_delta(self) -> None:
        variant = {"spans": [
            {"id": "0", "start_sec": 0, "end_sec": 0.8, "label": "X"},
            {"id": "1", "start_sec": 0.8, "end_sec": 2.2, "label": "UNKNOWN"},
        ]}
        report = audit(self.canonical, [variant])
        metrics = report["metrics"]
        self.assertGreater(metrics["speech_iou"]["mean"], 0.0)
        self.assertNotEqual(metrics["unknown_delta"]["mean"], 0.0)
        self.assertGreaterEqual(metrics["boundary_f1"]["mean"], 0.0)
        self.assertLessEqual(metrics["boundary_f1"]["mean"], 1.0)

    def test_metric_only_evidence_is_reviewable_and_cli_is_json_only(self) -> None:
        report = audit({"canonical": {"metrics": {"score": 1}}, "variants": [{"metrics": {"score": 2}}]})
        self.assertEqual(report["variant_count"], 1)
        self.assertIsNone(report["metrics"]["speech_iou"]["mean"])
        self.assertIsNone(report["metrics"]["boundary_f1"]["mean"])
        self.assertIsNone(report["metrics"]["h1_h2_state_change"]["rate"])
        self.assertEqual(report["permutation"][0]["evaluation_status"], "NO_CANONICAL_SPANS")
        self.assertNotIn("metrics", report["canonical"])

    def test_missing_chunk_central_evidence_is_not_reported_as_zero(self) -> None:
        canonical = {"spans": [{"start_us": 0, "end_us": 10, "speaker_id": "A"}]}
        variant = {"spans": [{"start_us": 0, "end_us": 10, "speaker_id": "X"}]}
        metric = audit(canonical, [variant])["metrics"]["chunk_central_agreement"]
        self.assertIsNone(metric["mean"])
        self.assertEqual(metric["evaluated_count"], 0)

    def test_missing_variant_time_counts_against_retention_and_flip(self) -> None:
        canonical = {"spans": [
            {"span_id": "a", "start_us": 0, "end_us": 10, "speaker_id": "A", "central": True},
            {"span_id": "b", "start_us": 10, "end_us": 20, "speaker_id": "B", "central": True},
        ]}
        variant = {"spans": [
            {"span_id": "x", "start_us": 0, "end_us": 10, "speaker_id": "X", "central": True},
        ]}
        metrics = audit(canonical, [variant])["metrics"]
        self.assertEqual(metrics["canonical_attribution_retention"]["mean"], 0.5)
        self.assertEqual(metrics["speaker_flip_rate"]["mean"], 0.5)
        self.assertEqual(metrics["chunk_central_agreement"]["mean"], 0.5)

    def test_state_aliases_are_normalized_and_missing_state_is_unevaluated(self) -> None:
        canonical = {"state": "H1", "spans": [{"start_us": 0, "end_us": 10, "speaker_id": "A"}]}
        same = {"state": "H1_CONFIRMED", "spans": [{"start_us": 0, "end_us": 10, "speaker_id": "X"}]}
        missing = {"spans": [{"start_us": 0, "end_us": 10, "speaker_id": "X"}]}
        state = audit(canonical, [same, missing])["metrics"]["h1_h2_state_change"]
        self.assertEqual(state["per_variant"], [False, None])
        self.assertEqual(state["evaluated_count"], 1)
        self.assertEqual(state["rate"], 0.0)

    def test_boundary_matching_is_maximum_cardinality(self) -> None:
        canonical = {"spans": [
            {"start_us": 0, "end_us": 10, "speaker_id": "A"},
            {"start_us": 10, "end_us": 20, "speaker_id": "B"},
            {"start_us": 20, "end_us": 30, "speaker_id": "A"},
        ]}
        variant = {"spans": [
            {"start_us": 0, "end_us": 16, "speaker_id": "X"},
            {"start_us": 16, "end_us": 21, "speaker_id": "Y"},
            {"start_us": 21, "end_us": 30, "speaker_id": "X"},
        ]}
        self.assertEqual(audit(canonical, [variant], boundary_tolerance_us=6)["metrics"]["boundary_f1"]["mean"], 1.0)

    def test_total_span_resource_bound_is_enforced(self) -> None:
        canonical = {"spans": [
            {"start_us": 0, "end_us": 1, "speaker_id": "A"},
            {"start_us": 1, "end_us": 2, "speaker_id": "B"},
        ]}
        variant = {"spans": [{"start_us": 0, "end_us": 1, "speaker_id": "X"}]}
        with patch("sddiar.proxy_stability.MAX_TOTAL_SPANS", 2):
            with self.assertRaises(ProxyStabilityError):
                audit(canonical, [variant])

    def test_no_assigned_evidence_is_unevaluated(self) -> None:
        canonical = {"spans": [{"start_us": 0, "end_us": 10, "speaker_id": "UNKNOWN"}]}
        variant = {"spans": [{"start_us": 0, "end_us": 10, "speaker_id": "UNKNOWN"}]}
        report = audit(canonical, [variant])
        for name in ("co_assigned_agreement", "speaker_flip_rate", "canonical_attribution_retention"):
            self.assertIsNone(report["metrics"][name]["mean"])
            self.assertEqual(report["metrics"][name]["evaluated_count"], 0)
        self.assertIsNone(report["metrics"]["speech_iou"]["mean"])
        self.assertEqual(report["permutation"][0]["evaluation_status"], "NO_ASSIGNED_EVIDENCE")

    def test_rejects_bad_interval(self) -> None:
        with self.assertRaises(ProxyStabilityError):
            audit({"spans": [{"start_us": 2, "end_us": 1, "speaker_id": "A"}]}, [])

    def test_rejects_duplicate_overlap_nonfinite_and_more_than_two_labels(self) -> None:
        duplicate = {"spans": [
            {"span_id": "same", "start_us": 0, "end_us": 1, "speaker_id": "A"},
            {"span_id": "same", "start_us": 1, "end_us": 2, "speaker_id": "A"},
        ]}
        overlap = {"spans": [
            {"span_id": "a", "start_us": 0, "end_us": 2, "speaker_id": "A"},
            {"span_id": "b", "start_us": 1, "end_us": 3, "speaker_id": "B"},
        ]}
        nonfinite = {"spans": [{"start_us": float("nan"), "end_us": 1, "speaker_id": "A"}]}
        boolean_time = {"spans": [{"start_sec": True, "end_sec": 1, "speaker_id": "A"}]}
        too_many = {"spans": [{"start_us": i, "end_us": i + 1, "speaker_id": label}
                               for i, label in enumerate(("A", "B", "C"))]}
        for payload in (duplicate, overlap, nonfinite, boolean_time, too_many):
            with self.subTest(payload=payload):
                with self.assertRaises(ProxyStabilityError):
                    audit(payload, [])

    def test_thousand_span_sweep_is_bounded_and_duration_based(self) -> None:
        canonical_rows = []
        variant_rows = []
        for index in range(1000):
            start, end = index * 10_000, (index + 1) * 10_000
            label = "A" if index < 900 else "B"
            canonical_rows.append({"start_us": start, "end_us": end, "speaker_id": label})
            # Deliberately use many variant rows; mapping must use duration, not
            # row count, and the labels are intentionally permuted.
            variant_rows.append({"start_us": start, "end_us": end, "speaker_id": "X" if label == "A" else "Y"})
        started = time.perf_counter()
        report = audit(canonical_rows, variant_rows)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.0)
        self.assertEqual(report["canonical"]["span_count"], 1000)
        self.assertEqual(report["summary"]["co_assigned_agreement"], 1.0)
        self.assertEqual(report["summary"]["canonical_attribution_retention"], 1.0)
        self.assertEqual(report["canonical"]["canonical_digest"], audit(canonical_rows, variant_rows)["canonical"]["canonical_digest"])


if __name__ == "__main__":
    unittest.main()
