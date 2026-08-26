import json
import tempfile
import unittest
from pathlib import Path

from sddiar.service import (GenOSNotConfigured, GenOSServiceStub, SummaryPolicyAdapter,
                            SummaryPolicyError, atomic_publish, idempotency_key)
from sddiar.worker import JobStatus, LocalJobStore, LocalWorker, classify_retry


class WorkerServiceTests(unittest.TestCase):
    def test_idempotency_is_stable_and_order_independent(self):
        sha = "a" * 64
        self.assertEqual(idempotency_key(sha, "p", {"b": 2, "a": 1}), idempotency_key(sha, "p", {"a": 1, "b": 2}))
        self.assertNotEqual(idempotency_key(sha, "p", {"a": 1}), idempotency_key(sha, "p", {"a": 2}))
        self.assertNotEqual(
            idempotency_key(sha, "p", pipeline_version="v1", model_pack_id="m1", calibration_profile_id="c1", stt_backend_version="s1"),
            idempotency_key(sha, "p", pipeline_version="v1", model_pack_id="m2", calibration_profile_id="c1", stt_backend_version="s1"),
        )

    def test_local_worker_idempotent_success_and_atomic_result(self):
        with tempfile.TemporaryDirectory() as temp:
            worker = LocalWorker(LocalJobStore(temp))
            first = worker.submit(job_id="j1", source_ref="fixture.wav", source_sha256="a" * 64, profile_id="p")
            second = worker.submit(job_id="j2", source_ref="fixture.wav", source_sha256="a" * 64, profile_id="p")
            self.assertEqual(first.job_id, second.job_id)
            calls = []
            done = worker.run("j1", lambda job: calls.append(job.job_id) or {"ok": True})
            self.assertEqual(done.status, JobStatus.SUCCEEDED); self.assertEqual(calls, ["j1"])
            self.assertEqual(json.loads(Path(done.result_path).read_text()), {"ok": True})
            self.assertEqual(worker.run("j1", lambda _: calls.append("bad")), done)
            self.assertEqual(calls, ["j1"])

    def test_retry_is_bounded_and_classification_is_fail_closed(self):
        self.assertTrue(classify_retry(TimeoutError("temporary")))
        self.assertFalse(classify_retry(ValueError("bad input")))
        with tempfile.TemporaryDirectory() as temp:
            worker = LocalWorker(LocalJobStore(temp), max_attempts=2)
            worker.submit(job_id="j", source_ref="x", source_sha256="b" * 64, profile_id="p")
            self.assertEqual(worker.run("j", lambda _: (_ for _ in ()).throw(TimeoutError("x"))).status, JobStatus.FAILED_RETRYABLE)
            self.assertEqual(worker.run("j", lambda _: (_ for _ in ()).throw(TimeoutError("x"))).status, JobStatus.FAILED_TERMINAL)

    def test_summary_policy_blocks_unsafe_speaker_aware_output(self):
        policy = SummaryPolicyAdapter()
        for status in ("REVIEW_REQUIRED", "UNSUPPORTED", "PASS_WITH_UNATTRIBUTED"):
            with self.subTest(status=status):
                with self.assertRaises(SummaryPolicyError): policy.authorize({"status": status}, "SPEAKER_AWARE")
                self.assertIn(policy.authorize({"status": status}, "AUTO"), {"SPEAKER_NEUTRAL", "MANUAL_REVIEW"})
        self.assertEqual(policy.authorize({"status": "PASS_STANDARD"}, "AUTO"), "SPEAKER_AWARE")

    def test_genos_is_explicit_injection_stub(self):
        with self.assertRaises(GenOSNotConfigured): GenOSServiceStub().submit({})


if __name__ == "__main__": unittest.main()
