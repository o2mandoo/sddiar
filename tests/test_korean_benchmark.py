from __future__ import annotations

import unittest
import hashlib
import json

from sddiar.evaluation import EvaluationRecording, RTTMRecord, ScoringConfig, UEMInterval, score_corpus
from sddiar.korean_benchmark import (
    BenchmarkEligibility,
    KoreanCorpusLock,
    KoreanBenchmarkError,
    KoreanReleasePolicy,
    KoreanReleaseGateReport,
    KoreanSuiteGateReport,
    ReferenceCapabilities,
    VerifiedKoreanCorpusLock,
    evaluate_benchmark_eligibility,
    evaluate_korean_release_gate,
    evaluate_korean_release_suite,
    parse_korean_corpus_lock,
    verify_korean_corpus_lock,
)


H = "a" * 64
KEY = b"test-release-key"


class ReleaseVerifier:
    trust_level = "RELEASE"

    def verify(self, payload, signature, signer_key_id):
        return (signer_key_id == "test-signer-001"
                and signature == hashlib.sha256(KEY + payload).hexdigest())


def verified(item):
    payload = json.dumps(
        item.as_dict(include_digest=False), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode()
    signature = hashlib.sha256(KEY + payload).hexdigest()
    return verify_korean_corpus_lock(
        item, signature=signature, signer_key_id="test-signer-001", verifier=ReleaseVerifier()
    )


def lock(*, role="GOLD", license_status="APPROVED_INTERNAL_EVALUATION",
         overlap=True, scd=True, audit="VERIFIED", timeline="VERIFIED"):
    return KoreanCorpusLock(
        corpus_id="nikl-regional-2021", corpus_version="1.0", authority_role=role,
        annotation_origin="PUBLISHER_HUMAN", license_status=license_status,
        continuous_timeline=timeline, audit_status=audit, speaker_independence="VERIFIED",
        reference_capabilities=ReferenceCapabilities(True, overlap, scd, False),
        source_archive_sha256=(H,), annotation_manifest_sha256="b" * 64,
        split_lock_sha256="c" * 64, license_text_sha256="d" * 64,
        audit_sha256="e" * 64 if audit == "VERIFIED" else None,
        release_holdout_locked=True,
    )


def perfect_report(count=2):
    recordings = []
    for index in range(count):
        file_id = f"opaque-{index:03d}"
        if index % 2:
            reference = (RTTMRecord(file_id, "REF_00", 0, 6_000_000),
                         RTTMRecord(file_id, "REF_01", 4_000_000, 8_000_000))
            hypothesis = (RTTMRecord(file_id, "SPEAKER_00", 0, 6_000_000),
                          RTTMRecord(file_id, "SPEAKER_01", 4_000_000, 8_000_000))
        else:
            reference = (RTTMRecord(file_id, "REF_00", 0, 8_000_000),)
            hypothesis = (RTTMRecord(file_id, "SPEAKER_00", 0, 8_000_000),)
        recordings.append(EvaluationRecording(
            file_id, reference, hypothesis, (UEMInterval(file_id, 0, 8_000_000),),
            subgroups=(("gender_pair", "UNKNOWN"),), overlap_reference_available=True,
        ))
    return score_corpus(tuple(recordings), config=ScoringConfig(bootstrap_iterations=8))


class KoreanBenchmarkTests(unittest.TestCase):
    def test_corpus_lock_is_canonical_and_round_trips(self):
        item = lock()
        value = item.as_dict(include_digest=False)
        self.assertEqual(parse_korean_corpus_lock(value), item)
        self.assertEqual(parse_korean_corpus_lock(item.as_dict()), item)
        self.assertEqual(len(item.lock_sha256), 64)
        reordered = dict(reversed(list(value.items())))
        self.assertEqual(parse_korean_corpus_lock(reordered).lock_sha256, item.lock_sha256)
        tampered = item.as_dict()
        tampered["corpus_version"] = "changed"
        with self.assertRaisesRegex(KoreanBenchmarkError, "digest mismatch"):
            parse_korean_corpus_lock(tampered)

    def test_verified_lock_constructor_and_nonrelease_verifier_fail_closed(self):
        with self.assertRaises(TypeError):
            VerifiedKoreanCorpusLock(lock(), "x", "a" * 64, "b" * 64)

        class DevVerifier(ReleaseVerifier):
            trust_level = "DEVELOPMENT"

        with self.assertRaisesRegex(KoreanBenchmarkError, "release-trusted"):
            verify_korean_corpus_lock(
                lock(), signature="x", signer_key_id="test-signer-001", verifier=DevVerifier()
            )

    def test_silver_challenge_and_research_data_never_open_release_scoring(self):
        silver = evaluate_benchmark_eligibility(lock(role="SILVER"), split="RELEASE_HOLDOUT")
        challenge = evaluate_benchmark_eligibility(lock(role="CHALLENGE"), split="RELEASE_HOLDOUT")
        research = evaluate_benchmark_eligibility(lock(license_status="RESEARCH_ONLY"), split="RELEASE_HOLDOUT")
        self.assertEqual((silver.status, challenge.status, research.status),
                         ("DEV_ONLY", "CHALLENGE_ONLY", "RESEARCH_ONLY"))
        self.assertFalse(silver.eligible_for_release_scoring)
        self.assertFalse(silver.eligible_for_metric_gating)
        unverified = evaluate_benchmark_eligibility(
            lock(license_status="UNVERIFIED"), split="RELEASE_HOLDOUT"
        )
        self.assertEqual(unverified.status, "DATA_INELIGIBLE")

    def test_gold_requires_verified_timeline_overlap_scd_audit_and_locked_holdout(self):
        result = evaluate_benchmark_eligibility(
            lock(overlap=False, scd=False, audit="PROVISIONAL", timeline="PROVISIONAL"),
            split="DEVELOPMENT_HOLDOUT",
        )
        self.assertEqual(result.status, "REVIEW_REQUIRED")
        self.assertIn("OVERLAP_REFERENCE_UNAVAILABLE", result.reason_codes)
        self.assertIn("NOT_RELEASE_HOLDOUT", result.reason_codes)

    def test_unavailable_capabilities_are_not_silently_scored_as_pass(self):
        report = perfect_report(2)
        item = lock(overlap=False, scd=False)
        result = evaluate_korean_release_gate(
            report, lock=item, split="RELEASE_HOLDOUT",
            policy=KoreanReleasePolicy(minimum_recordings=2, subgroup_minimum_recordings=2),
        )
        self.assertIsNone(result.gate_results["scd_f1"])
        self.assertIsNone(result.gate_results["osd_precision"])
        self.assertEqual(result.status, "REVIEW_REQUIRED")

    def test_single_speaker_only_gold_cannot_fake_scd_or_overlap_release_gates(self):
        recordings = []
        for index in range(20):
            file_id = f"only-h1-{index:03d}"
            records = (RTTMRecord(file_id, "REF_00", 0, 2_000_000),)
            predictions = (RTTMRecord(file_id, "SPEAKER_00", 0, 2_000_000),)
            recordings.append(EvaluationRecording(
                file_id, records, predictions, (UEMInterval(file_id, 0, 2_000_000),),
                overlap_reference_available=True,
            ))
        report = score_corpus(tuple(recordings), config=ScoringConfig(bootstrap_iterations=8))
        result = evaluate_korean_release_gate(
            report, lock=verified(lock()), split="RELEASE_HOLDOUT",
        )
        self.assertEqual(result.status, "METRIC_GATES_FAIL")
        self.assertFalse(result.gate_results["minimum_h2_files"])
        self.assertFalse(result.gate_results["minimum_scd_reference_events"])
        self.assertFalse(result.gate_results["minimum_overlap_reference_us"])

    def test_challenge_suite_can_veto_but_never_replace_gold(self):
        report = perfect_report(2)
        policy = KoreanReleasePolicy(
            minimum_recordings=2, subgroup_minimum_recordings=2,
            minimum_h1_files=1, minimum_h2_files=1,
            minimum_scd_reference_events=1, minimum_overlap_reference_us=1,
        )
        gold = evaluate_korean_release_gate(
            report, lock=verified(lock()), split="RELEASE_HOLDOUT", policy=policy,
        )
        failed = evaluate_korean_release_suite(
            gold, challenge_results={"synthetic-overlap": False},
            required_challenges=("synthetic-overlap",),
        )
        self.assertEqual(failed.status, "SUITE_METRICS_FAIL")
        missing = evaluate_korean_release_suite(
            gold, challenge_results={}, required_challenges=("synthetic-overlap",),
        )
        self.assertEqual(missing.status, "REVIEW_REQUIRED")

    def test_perfect_gold_can_only_become_metric_pass_requiring_review(self):
        report = perfect_report(2)
        policy = KoreanReleasePolicy(
            minimum_recordings=2, subgroup_minimum_recordings=2,
            minimum_h1_files=1, minimum_h2_files=1,
            minimum_scd_reference_events=1, minimum_overlap_reference_us=1,
        )
        raw = evaluate_korean_release_gate(report, lock=lock(), split="RELEASE_HOLDOUT", policy=policy)
        self.assertEqual(raw.status, "METRIC_GATES_PASS_REVIEW_REQUIRED")
        self.assertIn("CORPUS_LOCK_SIGNATURE_UNVERIFIED", raw.reason_codes)
        self.assertIn("EXTERNAL_RELEASE_AUTHORITY_REQUIRED", raw.reason_codes)
        result = evaluate_korean_release_gate(
            report, lock=verified(lock()), split="RELEASE_HOLDOUT", policy=policy
        )
        self.assertEqual(result.status, "METRIC_GATES_PASS_REVIEW_REQUIRED")
        self.assertEqual(result.release_authority, "none")
        self.assertEqual(result.metrics["der"], 0.0)
        self.assertTrue(all(value is True for value in result.gate_results.values()))

    def test_merge_and_missing_coverage_fail_release_candidate(self):
        file_id = "opaque-999"
        recording = EvaluationRecording(
            file_id,
            (RTTMRecord(file_id, "REF_00", 0, 4_000_000), RTTMRecord(file_id, "REF_01", 4_000_000, 8_000_000)),
            (RTTMRecord(file_id, "SPEAKER_00", 0, 8_000_000),),
            (UEMInterval(file_id, 0, 8_000_000),), overlap_reference_available=True,
        )
        report = score_corpus((recording,), config=ScoringConfig(bootstrap_iterations=4))
        result = evaluate_korean_release_gate(
            report, lock=verified(lock()), split="RELEASE_HOLDOUT",
            policy=KoreanReleasePolicy(
                minimum_recordings=1, subgroup_minimum_recordings=1,
                minimum_h1_files=1, minimum_h2_files=1,
                minimum_scd_reference_events=1, minimum_overlap_reference_us=1,
            ),
        )
        self.assertEqual(result.status, "METRIC_GATES_FAIL")
        self.assertFalse(result.gate_results["complete_merge_zero"])
        self.assertIn("GATE_FAILED_COMPLETE_MERGE_ZERO", result.reason_codes)

    def test_caller_controlled_release_verifier_cannot_mint_release_pass(self):
        class ForgedVerifier:
            trust_level = "RELEASE"

            @staticmethod
            def verify(_payload, _signature, _signer_key_id):
                return True

        forged = verify_korean_corpus_lock(
            lock(), signature="caller-controlled", signer_key_id="forged-signer-001",
            verifier=ForgedVerifier(),
        )
        policy = KoreanReleasePolicy(
            minimum_recordings=2, subgroup_minimum_recordings=2,
            minimum_h1_files=1, minimum_h2_files=1,
            minimum_scd_reference_events=1, minimum_overlap_reference_us=1,
        )
        result = evaluate_korean_release_gate(
            perfect_report(2), lock=forged, split="RELEASE_HOLDOUT", policy=policy,
        )
        self.assertEqual(result.status, "METRIC_GATES_PASS_REVIEW_REQUIRED")
        self.assertEqual(result.release_authority, "none")
        self.assertIn("EXTERNAL_RELEASE_AUTHORITY_REQUIRED", result.reason_codes)
        self.assertNotIn("RELEASE_CANDIDATE_PASS", json.dumps(result.as_dict()))

    def test_public_report_constructors_cannot_serialize_release_authority(self):
        with self.assertRaisesRegex(KoreanBenchmarkError, "cannot grant release authority"):
            BenchmarkEligibility(
                status="REVIEW_REQUIRED", split="RELEASE_HOLDOUT",
                eligible_for_metric_gating=True, eligible_for_release_scoring=True,
                reason_codes=("EXTERNAL_RELEASE_AUTHORITY_REQUIRED",),
                release_authority="attacker",
            )
        with self.assertRaisesRegex(KoreanBenchmarkError, "cannot grant release authority"):
            KoreanReleaseGateReport(
                status="RELEASE_CANDIDATE_PASS", release_authority="attacker",
                corpus_lock_sha256="a" * 64, policy_sha256="b" * 64,
                recording_count=1, metrics={}, gate_results={},
                reason_codes=("EXTERNAL_RELEASE_AUTHORITY_REQUIRED",),
            )
        with self.assertRaisesRegex(KoreanBenchmarkError, "cannot grant release authority"):
            KoreanSuiteGateReport(
                status="RELEASE_SUITE_CANDIDATE_PASS", release_authority="attacker",
                gold_status="METRIC_GATES_PASS_REVIEW_REQUIRED",
                required_challenges=("challenge-001",), challenge_results={"challenge-001": True},
                reason_codes=("EXTERNAL_RELEASE_AUTHORITY_REQUIRED",),
            )


if __name__ == "__main__":
    unittest.main()
