import base64
import hashlib
import tempfile
import time
import unittest
from pathlib import Path

from sddiar.calibration import (
    CalibrationError,
    CalibrationMismatchError,
    CalibrationProfile,
    CalibrationProfileVerifier,
    CalibrationProvenanceError,
    CalibrationSignatureError,
    DigestCalibrationSignatureVerifier,
    VerifiedCalibrationBinding,
    bind_calibration_profile,
    canonical_calibration_bytes,
    evaluate_thresholds,
    threshold_relation,
)
from sddiar.benchmark import StageMeasurement, account_long_file, run_repeated


class OfflineReleaseSignatureVerifier(DigestCalibrationSignatureVerifier):
    """Offline test double standing in for the audited release verifier."""

    trust_level = "RELEASE"


class CalibrationTests(unittest.TestCase):
    key = b"calibration-test-key"
    model_hash = "a" * 64
    config_hash = "d" * 64

    def signed_profile(self, **changes):
        profile = {
            "schema_version": "1",
            "profile_id": "p1",
            "calibration_version": "2026-08-26.1",
            "model_hashes": {"embed": self.model_hash},
            "source_sample_rates": [8000, 16000],
            "thresholds": {"unknown_ratio_warn": 0.2},
            "dataset_manifest_hash": "b" * 64,
            "scorer_hash": "c" * 64,
            "config_hash": self.config_hash,
            "approver": "quality-approval-1",
            "provenance": {
                "annotation_schema_version": "1",
                "created_at": "2026-08-26T00:00:00Z",
                "model_pack_id": "pack-1",
                "pipeline_version": "pipeline-1",
                "safety_constraints": ["no complete merge"],
                "selection_objective": "minimize unsafe attribution",
            },
            "signer_key_id": "calibration-release-key",
        }
        profile.update(changes)
        signature = hashlib.sha256(self.key + canonical_calibration_bytes(profile)).digest()
        profile["signature"] = base64.b64encode(signature).decode("ascii")
        return profile

    def release_verifier(self):
        return CalibrationProfileVerifier(OfflineReleaseSignatureVerifier(self.key))

    def test_immutable_and_compatible_binding(self):
        p = CalibrationProfile.from_mapping({"profile_id": "p1", "model_hashes": {"embed": "ABC"}, "source_sample_rates": [8000, 16000]})
        # Legacy compatibility matching remains available, but this type is not
        # PASS authority for QualityGate.
        self.assertTrue(bind_calibration_profile(p, model_hashes={"embed": "abc"}, source_sample_rate=8000))
        with self.assertRaises(TypeError): p.model_hashes["x"] = "y"

    def test_mismatch_and_missing_fail_safe(self):
        p = {"profile_id": "p1", "model_hashes": {"embed": "abc"}, "source_sample_rates": [16000]}
        self.assertFalse(bind_calibration_profile(None).valid)
        b = bind_calibration_profile(p, model_hashes={"embed": "wrong"}, source_sample_rate=8000)
        self.assertFalse(b.valid); self.assertIn("Q_CALIBRATION_MODEL_HASH_MISMATCH", b.reason_codes)

    def test_signed_release_profile_creates_immutable_exact_binding(self):
        binding = self.release_verifier().verify(
            self.signed_profile(),
            model_hashes={"embed": self.model_hash.upper()},
            source_sample_rate=8000,
            config_hash=self.config_hash.upper(),
            profile_id="p1",
        )
        self.assertIs(type(binding), VerifiedCalibrationBinding)
        self.assertTrue(binding.release_authorized)
        self.assertEqual(len(binding.profile_payload_sha256), 64)
        with self.assertRaises(TypeError):
            binding.thresholds["unknown_ratio_warn"] = 0.9
        with self.assertRaises(TypeError):
            binding.profile.provenance["created_at"] = "changed"

    def test_binding_constructor_is_sealed(self):
        profile = CalibrationProfile.from_mapping(self.signed_profile())
        with self.assertRaises(TypeError):
            VerifiedCalibrationBinding(
                profile, {"embed": self.model_hash}, 8000,
                self.config_hash, "e" * 64, "RELEASE",
            )

    def test_unsigned_empty_and_incomplete_profiles_cannot_bind(self):
        verifier = self.release_verifier()
        unsigned = self.signed_profile()
        unsigned["signature"] = ""
        with self.assertRaises(CalibrationSignatureError):
            verifier.verify(
                unsigned,
                model_hashes={"embed": self.model_hash},
                source_sample_rate=8000,
                config_hash=self.config_hash,
            )
        for changes in (
            {"thresholds": {}},
            {"dataset_manifest_hash": ""},
            {"scorer_hash": ""},
            {"config_hash": ""},
            {"approver": ""},
            {"provenance": {}},
            {"calibration_version": ""},
        ):
            profile = self.signed_profile(**changes)
            with self.subTest(changes=changes), self.assertRaises((CalibrationProvenanceError, CalibrationSignatureError)):
                verifier.verify(
                    profile,
                    model_hashes={"embed": self.model_hash},
                    source_sample_rate=8000,
                    config_hash=self.config_hash,
                )

    def test_signature_and_runtime_binding_fail_closed(self):
        profile = self.signed_profile()
        profile["signature"] = base64.b64encode(b"wrong").decode("ascii")
        with self.assertRaises(CalibrationSignatureError):
            self.release_verifier().verify(
                profile,
                model_hashes={"embed": self.model_hash},
                source_sample_rate=8000,
                config_hash=self.config_hash,
            )

        valid = self.signed_profile()
        cases = (
            ({"embed": "e" * 64}, 8000, self.config_hash),
            ({"embed": self.model_hash, "extra": "f" * 64}, 8000, self.config_hash),
            ({"embed": self.model_hash}, 44100, self.config_hash),
            ({"embed": self.model_hash}, 8000, "f" * 64),
        )
        for model_hashes, rate, config_hash in cases:
            with self.subTest(model_hashes=model_hashes, rate=rate, config_hash=config_hash), self.assertRaises(CalibrationMismatchError):
                self.release_verifier().verify(
                    valid,
                    model_hashes=model_hashes,
                    source_sample_rate=rate,
                    config_hash=config_hash,
                )

    def test_development_verifier_cannot_authorize_release(self):
        binding = CalibrationProfileVerifier(
            DigestCalibrationSignatureVerifier(self.key)
        ).verify(
            self.signed_profile(),
            model_hashes={"embed": self.model_hash},
            source_sample_rate=8000,
            config_hash=self.config_hash,
        )
        self.assertTrue(binding.is_verified)
        self.assertFalse(binding.release_authorized)

    def test_strict_profile_rejects_ambiguous_numeric_types(self):
        for changes in (
            {"source_sample_rates": [True]},
            {"source_sample_rates": [16000.5]},
            {"thresholds": {"unknown_ratio_warn": True}},
            {"thresholds": {"unknown_ratio_warn": float("nan")}},
            {"schema_version": "2"},
        ):
            with self.subTest(changes=changes), self.assertRaises(CalibrationError):
                profile = self.signed_profile(**changes)
                self.release_verifier().verify(
                    profile,
                    model_hashes={"embed": self.model_hash},
                    source_sample_rate=8000,
                    config_hash=self.config_hash,
                )

    def test_required_provenance_values_must_be_semantically_nonempty(self):
        base = {
            "annotation_schema_version": "1",
            "created_at": "2026-08-26T00:00:00Z",
            "model_pack_id": "pack-1",
            "pipeline_version": "pipeline-1",
            "safety_constraints": ["no complete merge"],
            "selection_objective": "minimize unsafe attribution",
        }
        invalid_values = {
            "annotation_schema_version": False,
            "created_at": 0,
            "model_pack_id": False,
            "pipeline_version": 0,
            "safety_constraints": [None],
            "selection_objective": True,
        }
        for key, value in invalid_values.items():
            provenance = dict(base)
            provenance[key] = value
            profile = self.signed_profile(provenance=provenance)
            with self.subTest(key=key), self.assertRaises(CalibrationProvenanceError):
                self.release_verifier().verify(
                    profile,
                    model_hashes={"embed": self.model_hash},
                    source_sample_rate=8000,
                    config_hash=self.config_hash,
                )

    def test_runtime_source_rate_requires_exact_integer(self):
        with self.assertRaises(CalibrationError):
            self.release_verifier().verify(
                self.signed_profile(),
                model_hashes={"embed": self.model_hash},
                source_sample_rate=8000.5,
                config_hash=self.config_hash,
            )

    def test_load_and_threshold_relations(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.json"; path.write_text('{"profile_id":"p","source_rates":[16000]}', encoding="utf-8")
            self.assertEqual(CalibrationProfile.load(path).source_sample_rates, (16000,))
        self.assertEqual(threshold_relation(0.9, 0.8), "PASS")
        self.assertEqual(threshold_relation(0.9, 0.8, higher_is_better=False), "FAIL")
        self.assertEqual(evaluate_thresholds({"x": 2}, {"x": 1}), {"x": "PASS"})


class BenchmarkTests(unittest.TestCase):
    def test_repeated_and_bounded_accounting(self):
        runs = run_repeated(lambda: time.sleep(0), repeats=2)
        self.assertEqual(len(runs), 2); self.assertTrue(all(x.elapsed_seconds >= 0 for x in runs))
        report = account_long_file((StageMeasurement("a", 2.0, 10.0, 100),), audio_seconds=100, max_audio_seconds=60)
        self.assertTrue(report["bounded"]); self.assertAlmostEqual(report["rtf"], .02)


if __name__ == "__main__": unittest.main()
