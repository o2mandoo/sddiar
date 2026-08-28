from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

from sddiar.evaluation_io import evaluate_korean_benchmark
from sddiar.korean_benchmark import KoreanCorpusLock, ReferenceCapabilities
from sddiar.nikl_adapter import (
    build_nikl_manifest_row,
    format_nikl_rttm,
    format_nikl_uem,
    parse_nikl_reference,
)


class PublicKoreanBenchmarkIntegrationTests(unittest.TestCase):
    def test_nikl_adapter_intake_and_corpus_scorer_connect_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ("audio", "rttm", "uem", "pred"):
                (root / folder).mkdir()
            rows = []
            predictions = []
            for index, split in enumerate((
                "CALIBRATION", "DEVELOPMENT_HOLDOUT", "RELEASE_HOLDOUT"
            ), 1):
                payload = {
                    "id": f"private-source-{index}",
                    "metadata": {"category": "spoken"},
                    "document": [{
                        "id": f"private-document-{index}",
                        "metadata": {"speaker": [{"id": "person-A"}, {"id": "person-B"}]},
                        "utterance": [
                            {"speaker_id": "person-A", "start": 0, "end": 1250,
                             "form": "sensitive transcript"},
                            {"speaker_id": "person-B", "start": 1000, "end": 2000,
                             "form": "another transcript"},
                        ],
                    }],
                }
                reference = parse_nikl_reference(payload, 2_500_000, time_unit="milliseconds")
                wav_path = root / "audio" / f"{reference.recording_id}.wav"
                with wave.open(str(wav_path), "wb") as stream:
                    stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                    stream.writeframes(b"\0\0" * 40_000)
                rttm = format_nikl_rttm(reference)
                uem = format_nikl_uem(reference)
                ref_path = root / "rttm" / f"{reference.recording_id}.rttm"
                uem_path = root / "uem" / f"{reference.recording_id}.uem"
                pred_path = root / "pred" / f"{reference.recording_id}.rttm"
                # Adapter hashes bind canonical UTF-8/LF bytes.  Avoid host
                # newline translation so the same fixture works on Windows.
                ref_path.write_bytes(rttm.encode("utf-8"))
                uem_path.write_bytes(uem.encode("utf-8"))
                pred_path.write_bytes(
                    rttm.replace("REF_00", "SPEAKER_00")
                    .replace("REF_01", "SPEAKER_01")
                    .encode("utf-8")
                )
                rows.append(build_nikl_manifest_row(
                    reference, audio_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                    sample_rate_hz=16000, split=split,
                    audio_path=f"audio/{wav_path.name}", rttm_path=f"rttm/{ref_path.name}",
                    uem_path=f"uem/{uem_path.name}", conditions=("regional-interview", "overlap"),
                    speaker_group_ids=(f"person-hmac-{index:03d}-1", f"person-hmac-{index:03d}-2"),
                ))
                prediction_bytes = pred_path.read_bytes()
                predictions.append({
                    "audio_id": reference.recording_id,
                    "rttm": f"pred/{pred_path.name}",
                    "rttm_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
                    "quality_status": "REVIEW_REQUIRED",
                })
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            prediction_manifest = root / "predictions.jsonl"
            prediction_manifest.write_text(
                "\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8"
            )
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            corpus_lock = KoreanCorpusLock(
                corpus_id="nikl-regional-fixture-001", corpus_version="synthetic-v1",
                authority_role="GOLD", annotation_origin="PUBLISHER_HUMAN",
                license_status="APPROVED_INTERNAL_EVALUATION", continuous_timeline="VERIFIED",
                audit_status="VERIFIED", speaker_independence="VERIFIED",
                reference_capabilities=ReferenceCapabilities(True, True, True, False),
                source_archive_sha256=("a" * 64,), annotation_manifest_sha256=manifest_sha,
                split_lock_sha256=manifest_sha, license_text_sha256="b" * 64,
                audit_sha256="c" * 64, release_holdout_locked=True,
            )
            lock_path = root / "corpus.lock.json"
            lock_path.write_text(json.dumps(corpus_lock.as_dict(include_digest=False)), encoding="utf-8")
            result = evaluate_korean_benchmark(
                manifest, prediction_manifest, corpus_lock=lock_path,
                split="DEVELOPMENT_HOLDOUT", dataset_root=root,
                prediction_root=root, bootstrap_iterations=8,
            )
            self.assertEqual(result["metric_views"]["der_duration_micro"], 0.0)
            self.assertEqual(result["metric_views"]["nonoverlap_speech_coverage_duration_micro"], 1.0)
            self.assertTrue(result["overall"]["overlap"]["evaluated"])
            self.assertEqual(result["eligibility"]["status"], "REVIEW_REQUIRED")
            self.assertIn("REFERENCE_ROWS_NOT_GOLD_APPROVED", result["release_gate"]["reason_codes"])
            public = json.dumps(result, ensure_ascii=False)
            self.assertNotIn("sensitive transcript", public)
            self.assertNotIn("person-A", public)
            self.assertNotIn(str(root), public)


if __name__ == "__main__":
    unittest.main()
