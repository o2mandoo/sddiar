from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import wave
import unittest
from unittest import mock

from sddiar.evaluation_io import EvaluationIOError, evaluate_korean_benchmark
from sddiar.korean_benchmark import KoreanCorpusLock, ReferenceCapabilities


class EvaluationIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ("audio", "rttm", "uem", "pred"):
            (self.root / folder).mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _make_audio(self, audio_id: str) -> None:
        with wave.open(str(self.root / "audio" / f"{audio_id}.wav"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(b"\0\0" * (16_000 * 2))

    def _reference_row(self, audio_id: str, split: str, *, two_speakers: bool = False,
                       overlap: bool = False) -> dict[str, object]:
        self._make_audio(audio_id)
        if two_speakers and overlap:
            rttm = (
                f"SPEAKER {audio_id} 1 0.100 0.900 <NA> <NA> REF_00 <NA> <NA>\n"
                f"SPEAKER {audio_id} 1 0.600 0.900 <NA> <NA> REF_01 <NA> <NA>\n"
            )
        elif two_speakers:
            rttm = (
                f"SPEAKER {audio_id} 1 0.100 0.800 <NA> <NA> REF_00 <NA> <NA>\n"
                f"SPEAKER {audio_id} 1 1.000 0.800 <NA> <NA> REF_01 <NA> <NA>\n"
            )
        else:
            rttm = f"SPEAKER {audio_id} 1 0.100 1.600 <NA> <NA> REF_00 <NA> <NA>\n"
        (self.root / "rttm" / f"{audio_id}.rttm").write_text(rttm, encoding="utf-8")
        (self.root / "uem" / f"{audio_id}.uem").write_text(
            f"{audio_id} 1 0.000 2.000\n", encoding="utf-8")
        audio_bytes = (self.root / "audio" / f"{audio_id}.wav").read_bytes()
        return {
            "audio_id": audio_id,
            "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
            "session_id": f"session-{audio_id[-3:]}",
            "split": split,
            "sample_rate_hz": 16_000,
            "speaker_count": 2 if two_speakers else 1,
            "gender_pair": "MF" if two_speakers else "M",
            "conditions": ["near", "overlap"] if overlap else ["near"],
            "speaker_group_ids": [f"person-hmac-{audio_id[-3:]}-1"] + (
                [f"person-hmac-{audio_id[-3:]}-2"] if two_speakers else []
            ),
            "reference_status": "GOLD_APPROVED",
            "uem_policy": "FULL_AUDIO",
            "conversion_evidence_sha256": "f" * 64,
            "audio": f"audio/{audio_id}.wav",
            "rttm": f"rttm/{audio_id}.rttm",
            "uem": f"uem/{audio_id}.uem",
            "rttm_sha256": hashlib.sha256((self.root / "rttm" / f"{audio_id}.rttm").read_bytes()).hexdigest(),
            "uem_sha256": hashlib.sha256((self.root / "uem" / f"{audio_id}.uem").read_bytes()).hexdigest(),
        }

    def _write_manifest(self, name: str, rows: list[dict[str, object]]) -> Path:
        path = self.root / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def _write_predictions(self, rows: list[tuple[str, str]]) -> Path:
        output: list[dict[str, str]] = []
        for audio_id, body in rows:
            path = self.root / "pred" / f"{audio_id}.rttm"
            path.write_text(body, encoding="utf-8")
            raw = path.read_bytes()
            output.append({
                "audio_id": audio_id,
                "rttm": f"pred/{audio_id}.rttm",
                "rttm_sha256": hashlib.sha256(raw).hexdigest(),
                "quality_status": "PASS_STANDARD",
            })
        return self._write_manifest("predictions.jsonl", output)

    def _corpus(self) -> tuple[Path, list[dict[str, object]]]:
        refs = [
            self._reference_row("opaque-001", "CALIBRATION"),
            self._reference_row("opaque-002", "DEVELOPMENT_HOLDOUT", two_speakers=True,
                                overlap=True),
            self._reference_row("opaque-003", "RELEASE_HOLDOUT"),
        ]
        manifest = self._write_manifest("manifest.jsonl", refs)
        predictions = self._write_predictions([
            ("opaque-001", "SPEAKER opaque-001 1 0.100 1.600 <NA> <NA> H1 <NA> <NA>\n"),
            ("opaque-002", "SPEAKER opaque-002 1 0.100 0.900 <NA> <NA> H1 <NA> <NA>\n"
             "SPEAKER opaque-002 1 0.600 0.900 <NA> <NA> H2 <NA> <NA>\n"),
            ("opaque-003", "SPEAKER opaque-003 1 0.100 1.600 <NA> <NA> H1 <NA> <NA>\n"),
        ])
        return manifest, [json.loads(line) for line in predictions.read_text().splitlines()]

    def _lock(self, manifest: Path, *, role: str = "GOLD") -> Path:
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        item = KoreanCorpusLock(
            corpus_id="synthetic-korean-001", corpus_version="test", authority_role=role,
            annotation_origin="PUBLISHER_HUMAN" if role == "GOLD" else "MACHINE",
            license_status="APPROVED_INTERNAL_EVALUATION", continuous_timeline="VERIFIED",
            audit_status="VERIFIED", speaker_independence="VERIFIED",
            reference_capabilities=ReferenceCapabilities(True, True, True, False),
            source_archive_sha256=("a" * 64,), annotation_manifest_sha256=digest,
            split_lock_sha256=digest, license_text_sha256="b" * 64,
            audit_sha256="c" * 64, release_holdout_locked=True,
        )
        path = self.root / "corpus.lock.json"
        path.write_text(json.dumps(item.as_dict(include_digest=False)), encoding="utf-8")
        return path

    def test_perfect_result_is_aggregate_only_and_overlap_is_detected(self) -> None:
        manifest, _ = self._corpus()
        prediction_manifest = self.root / "predictions.jsonl"
        result = evaluate_korean_benchmark(manifest, prediction_manifest,
                                           corpus_lock=self._lock(manifest),
                                           split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=8)
        self.assertEqual(result["overall"]["diarization_all"]["der"], 0.0)
        self.assertEqual(result["der_collar_us"], 250_000)
        self.assertEqual(result["strict_0ms"]["der_collar_us"], 0)
        self.assertEqual(result["metric_views"]["der_recording_macro"], 0.0)
        self.assertEqual(result["metric_views"]["nonoverlap_speech_coverage_duration_micro"], 1.0)
        labels = {item["subgroup"] for item in result["subgroups"]}
        self.assertNotIn("split=CALIBRATION", labels)
        self.assertIn("split=DEVELOPMENT_HOLDOUT", labels)
        self.assertIn("gender_pair=MF", labels)
        self.assertIn("sample_rate_hz=16000", labels)
        self.assertIn("condition=overlap", labels)
        self.assertTrue(result["overall"]["overlap"]["evaluated"])
        self.assertEqual(result["split"], "DEVELOPMENT_HOLDOUT")
        self.assertEqual(result["eligibility"]["status"], "REVIEW_REQUIRED")
        self.assertNotIn("recordings", result)
        redacted = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("opaque-001", redacted)
        self.assertNotIn("REF_00", redacted)
        self.assertNotIn(str(self.root), redacted)

    def test_prediction_mismatch_and_missing_ids_fail_closed(self) -> None:
        manifest, _ = self._corpus()
        bad = self.root / "bad.jsonl"
        bad.write_text(json.dumps({
            "audio_id": "opaque-001", "rttm": "pred/opaque-001.rttm",
            "rttm_sha256": hashlib.sha256(
                (self.root / "pred/opaque-001.rttm").read_bytes()).hexdigest(),
            "quality_status": "PASS_STANDARD",
        }) + "\n", encoding="utf-8")
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, bad, corpus_lock=self._lock(manifest),
                split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
            )
        self.assertEqual(context.exception.code, "PREDICTION_ID_SET_MISMATCH")

    def test_hash_and_path_errors_do_not_echo_sensitive_values(self) -> None:
        manifest, _ = self._corpus()
        row = {"audio_id": "opaque-001", "rttm": "../secret.rttm",
               "rttm_sha256": "0" * 64, "quality_status": "PASS_STANDARD"}
        prediction_manifest = self._write_manifest("bad-path.jsonl", [row])
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, prediction_manifest, corpus_lock=self._lock(manifest),
                split="DEVELOPMENT_HOLDOUT",
            )
        self.assertEqual(context.exception.code, "PATH_INVALID")
        self.assertNotIn("secret.rttm", str(context.exception))
        self.assertNotIn(str(self.root), str(context.exception))

    def test_prediction_hash_is_checked(self) -> None:
        manifest, _ = self._corpus()
        row = {"audio_id": "opaque-001", "rttm": "pred/opaque-001.rttm",
               "rttm_sha256": "0" * 64, "quality_status": "PASS_STANDARD"}
        prediction_manifest = self._write_manifest("bad-hash.jsonl", [row])
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, prediction_manifest, corpus_lock=self._lock(manifest),
                split="DEVELOPMENT_HOLDOUT",
            )
        self.assertEqual(context.exception.code, "PREDICTION_HASH_MISMATCH")

    def test_duplicate_json_keys_and_symlinked_lock_are_rejected(self) -> None:
        manifest, _ = self._corpus()
        prediction_manifest = self.root / "predictions.jsonl"
        lock_path = self.root / "duplicate.lock.json"
        lock_path.write_text('{"language":"ko","language":"ko"}', encoding="utf-8")
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, prediction_manifest, corpus_lock=lock_path,
                split="DEVELOPMENT_HOLDOUT",
            )
        self.assertEqual(context.exception.code, "CORPUS_LOCK_INVALID")
        linked = self.root / "linked.lock.json"
        try:
            linked.symlink_to(self._lock(manifest))
        except (OSError, NotImplementedError):
            return
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, prediction_manifest, corpus_lock=linked,
                split="DEVELOPMENT_HOLDOUT",
            )
        self.assertEqual(context.exception.code, "FILE_TYPE_INVALID")

    def test_bootstrap_work_is_bounded(self) -> None:
        manifest, _ = self._corpus()
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, self.root / "predictions.jsonl", corpus_lock=self._lock(manifest),
                split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=100_001,
            )
        self.assertEqual(context.exception.code, "SCORING_CONFIG_INVALID")

    def test_speaker_group_cardinality_is_bound_to_declared_speaker_count(self) -> None:
        manifest, _ = self._corpus()
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        rows[1]["speaker_group_ids"] = rows[1]["speaker_group_ids"][:1]
        manifest = self._write_manifest("manifest.jsonl", rows)
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, self.root / "predictions.jsonl", corpus_lock=self._lock(manifest),
                split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
            )
        self.assertEqual(context.exception.code, "REFERENCE_VALIDATION_FAILED")

    def test_second_reference_loader_revalidates_public_subgroup_allowlists(self) -> None:
        manifest, _ = self._corpus()
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        rows[1]["conditions"] = ["private-person-427"]
        rows[1]["gender_pair"] = "PRIVATE427"
        rows[1]["sample_rate_hz"] = 427
        manifest = self._write_manifest("manifest.jsonl", rows)
        with mock.patch("sddiar.evaluation_io.validate_annotation_dataset") as validator:
            validator.return_value.ok = True
            with self.assertRaises(EvaluationIOError) as context:
                evaluate_korean_benchmark(
                    manifest, self.root / "predictions.jsonl", corpus_lock=self._lock(manifest),
                    split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
                )
        self.assertEqual(context.exception.code, "REFERENCE_MANIFEST_SCHEMA")

    def test_second_reference_loader_rejects_split_leakage_independently(self) -> None:
        manifest, _ = self._corpus()
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        rows[1]["speaker_group_ids"][0] = rows[0]["speaker_group_ids"][0]
        manifest = self._write_manifest("manifest.jsonl", rows)
        with mock.patch("sddiar.evaluation_io.validate_annotation_dataset") as validator:
            validator.return_value.ok = True
            with self.assertRaises(EvaluationIOError) as context:
                evaluate_korean_benchmark(
                    manifest, self.root / "predictions.jsonl", corpus_lock=self._lock(manifest),
                    split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
                )
        self.assertEqual(context.exception.code, "REFERENCE_SPLIT_LEAKAGE")

    def test_manifest_change_between_validation_and_loading_is_rejected(self) -> None:
        manifest, _ = self._corpus()
        original = manifest.read_bytes()

        def swap_manifest(*_args, **_kwargs):
            manifest.write_bytes(original + b"\n")
            return mock.Mock(ok=True)

        with mock.patch(
            "sddiar.evaluation_io.validate_annotation_dataset", side_effect=swap_manifest,
        ):
            with self.assertRaises(EvaluationIOError) as context:
                evaluate_korean_benchmark(
                    manifest, self.root / "predictions.jsonl", corpus_lock=self._lock(manifest),
                    split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
                )
        self.assertEqual(context.exception.code, "REFERENCE_MANIFEST_CHANGED")

    def test_second_loader_binds_annotation_bounds_to_same_audio_snapshot(self) -> None:
        manifest, _ = self._corpus()
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        target = rows[1]
        uem_path = self.root / str(target["uem"])
        rttm_path = self.root / str(target["rttm"])
        uem_path.write_text("opaque-002 1 0.000 3.000\n", encoding="utf-8")
        rttm_path.write_text(
            "SPEAKER opaque-002 1 0.100 2.800 <NA> <NA> REF_00 <NA> <NA>\n"
            "SPEAKER opaque-002 1 0.600 0.900 <NA> <NA> REF_01 <NA> <NA>\n",
            encoding="utf-8",
        )
        target["uem_sha256"] = hashlib.sha256(uem_path.read_bytes()).hexdigest()
        target["rttm_sha256"] = hashlib.sha256(rttm_path.read_bytes()).hexdigest()
        manifest = self._write_manifest("manifest.jsonl", rows)
        with mock.patch("sddiar.evaluation_io.validate_annotation_dataset") as validator:
            validator.return_value.ok = True
            with self.assertRaises(EvaluationIOError) as context:
                evaluate_korean_benchmark(
                    manifest, self.root / "predictions.jsonl", corpus_lock=self._lock(manifest),
                    split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
                )
        self.assertEqual(context.exception.code, "ANNOTATION_OUTSIDE_AUDIO")

    def test_full_audio_uem_cannot_be_shrunk_across_validation_seam(self) -> None:
        manifest, _ = self._corpus()
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        target = rows[1]
        uem_path = self.root / str(target["uem"])
        uem_path.write_text("opaque-002 1 0.000 0.500\n", encoding="utf-8")
        target["uem_sha256"] = hashlib.sha256(uem_path.read_bytes()).hexdigest()
        manifest = self._write_manifest("manifest.jsonl", rows)
        with mock.patch("sddiar.evaluation_io.validate_annotation_dataset") as validator:
            validator.return_value.ok = True
            with self.assertRaises(EvaluationIOError) as context:
                evaluate_korean_benchmark(
                    manifest, self.root / "predictions.jsonl", corpus_lock=self._lock(manifest),
                    split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
                )
        self.assertEqual(context.exception.code, "REFERENCE_UEM_POLICY_MISMATCH")

    def test_deeply_nested_corpus_lock_becomes_redacted_contract_error(self) -> None:
        manifest, _ = self._corpus()
        lock_path = self.root / "deep.lock.json"
        lock_path.write_text("[" * 2_000 + "]" * 2_000, encoding="utf-8")
        with self.assertRaises(EvaluationIOError) as context:
            evaluate_korean_benchmark(
                manifest, self.root / "predictions.jsonl", corpus_lock=lock_path,
                split="DEVELOPMENT_HOLDOUT", bootstrap_iterations=0,
            )
        self.assertEqual(context.exception.code, "CORPUS_LOCK_INVALID")


if __name__ == "__main__":
    unittest.main()
