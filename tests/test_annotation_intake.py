from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import struct
import wave
import unittest
from unittest import mock

from sddiar.annotation_intake import load_words_artifact, validate_annotation_dataset


class AnnotationIntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for folder in ("audio", "rttm", "uem"):
            (self.root / folder).mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _audio(self, audio_id: str, rate: int = 16000) -> tuple[str, int]:
        path = self.root / "audio" / f"{audio_id}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(b"\0\0" * (rate * 2))
        return f"audio/{audio_id}.wav", path.stat().st_size

    def _row(self, audio_id: str, split: str, *, rate: int = 16000,
             speaker_count: int = 1, gender_pair: str = "M",
             conditions: list[str] | None = None) -> dict:
        audio, _ = self._audio(audio_id, rate)
        speakers = ["REF_00"] if speaker_count == 1 else ["REF_00", "REF_01"]
        (self.root / "rttm" / f"{audio_id}.rttm").write_text(
            (f"SPEAKER {audio_id} 1 0.100 0.800 <NA> <NA> {speakers[0]} <NA> <NA>\n" if speaker_count == 1 else
             f"SPEAKER {audio_id} 1 0.100 0.800 <NA> <NA> {speakers[0]} <NA> <NA>\n"
             f"SPEAKER {audio_id} 1 0.500 0.800 <NA> <NA> {speakers[1]} <NA> <NA>\n"), encoding="utf-8")
        (self.root / "uem" / f"{audio_id}.uem").write_text(
            f"{audio_id} 1 0.000 2.000\n", encoding="utf-8")
        digest = hashlib.sha256((self.root / audio).read_bytes()).hexdigest()
        return {
            "audio_id": audio_id,
            "audio_sha256": digest,
            "session_id": f"session-{audio_id[-3:]}",
            "split": split,
            "sample_rate_hz": rate,
            "speaker_count": speaker_count,
            "gender_pair": gender_pair,
            "conditions": conditions or ["near"],
            "audio": audio,
            "rttm": f"rttm/{audio_id}.rttm",
            "uem": f"uem/{audio_id}.uem",
        }

    def _write(self, rows: list[dict]) -> Path:
        path = self.root / "manifest.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_valid_manifest_is_offline_and_redacted(self):
        rows = [
            self._row("opaque-001", "CALIBRATION", rate=16000, speaker_count=1,
                      gender_pair="M", conditions=["near", "quiet-secondary", "short-turn"]),
            self._row("opaque-002", "DEVELOPMENT_HOLDOUT", rate=8000, speaker_count=2,
                      gender_pair="MM", conditions=["far", "noisy"]),
            self._row("opaque-003", "RELEASE_HOLDOUT", rate=16000, speaker_count=2,
                      gender_pair="MF", conditions=["near", "overlap"]),
        ]
        report = validate_annotation_dataset(self._write(rows))
        self.assertTrue(report.ok, report.as_dict())
        payload = report.as_dict()
        self.assertEqual(payload["records_valid"], 3)
        self.assertEqual(payload["readiness"]["overlap_duration_seconds"], 0.8)
        self.assertNotIn(str(self.root), json.dumps(payload))
        self.assertNotIn("REF_00", json.dumps(payload))

    def test_split_leakage_is_rejected(self):
        first = self._row("opaque-101", "CALIBRATION")
        second = self._row("opaque-102", "RELEASE_HOLDOUT", speaker_count=2, gender_pair="MF")
        first["source_recording_id"] = "source-777"
        second["source_recording_id"] = "source-777"
        report = validate_annotation_dataset(self._write([first, second, self._row("opaque-103", "DEVELOPMENT_HOLDOUT")]))
        self.assertIn("SPLIT_LEAKAGE", {item.code for item in report.errors})

    def test_hash_path_and_privacy_failures_are_redacted(self):
        row = self._row("opaque-201", "CALIBRATION")
        row["audio_sha256"] = "0" * 64
        row["rttm"] = "../secret.rttm"
        row["session_id"] = "John Doe"
        report = validate_annotation_dataset(self._write([
            row,
            self._row("opaque-202", "DEVELOPMENT_HOLDOUT"),
            self._row("opaque-203", "RELEASE_HOLDOUT"),
        ]))
        codes = {item.code for item in report.errors}
        self.assertTrue({"AUDIO_HASH_MISMATCH", "PATH_INVALID", "PRIVACY_ID"} <= codes)
        self.assertNotIn("secret.rttm", json.dumps(report.as_dict()))
        self.assertNotIn("John Doe", json.dumps(report.as_dict()))

    def test_timing_and_file_id_failures(self):
        row = self._row("opaque-301", "CALIBRATION")
        (self.root / row["uem"]).write_text("wrong-id 1 0.0 3.0\n", encoding="utf-8")
        (self.root / row["rttm"]).write_text(
            "SPEAKER opaque-301 1 1.900 1.000 <NA> <NA> REF_00 <NA> <NA>\n", encoding="utf-8")
        report = validate_annotation_dataset(self._write([
            row,
            self._row("opaque-302", "DEVELOPMENT_HOLDOUT"),
            self._row("opaque-303", "RELEASE_HOLDOUT"),
        ]))
        codes = {item.code for item in report.errors}
        self.assertTrue({"FILE_ID_MISMATCH", "UEM_OUTSIDE_AUDIO", "RTTM_OUTSIDE_AUDIO"} <= codes)

    def test_missing_required_split_and_symlink_are_rejected(self):
        row = self._row("opaque-401", "CALIBRATION")
        link = self.root / "audio" / "linked-401.wav"
        try:
            link.symlink_to(self.root / row["audio"])
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this host")
        row["audio"] = "audio/linked-401.wav"
        report = validate_annotation_dataset(self._write([row]))
        codes = {item.code for item in report.errors}
        self.assertIn("PATH_INVALID", codes)
        self.assertIn("REQUIRED_SPLIT_MISSING", codes)

    def test_optional_words_artifact_is_hash_bound_and_redacted(self):
        row = self._row("opaque-501", "CALIBRATION")
        words_path = self.root / "words"
        words_path.mkdir()
        words = (
            '{"recording_id":"opaque-501","word_id":"word-001",'
            '"start_us":200000,"end_us":600000,"text":"민감한 문장",'
            '"ref_speaker_id":"REF_00","attributable":true,'
            '"overlap_flag":false,"boundary_crossing_flag":false,"micro_flag":true}\n'
        ).encode("utf-8")
        artifact = words_path / "opaque-501.jsonl"
        artifact.write_bytes(words)
        row.update({
            "words": "words/opaque-501.jsonl",
            "words_sha256": hashlib.sha256(words).hexdigest(),
            "words_timebase": "microseconds",
        })
        rows = [row, self._row("opaque-502", "DEVELOPMENT_HOLDOUT"),
                self._row("opaque-503", "RELEASE_HOLDOUT")]
        report = validate_annotation_dataset(self._write(rows))
        self.assertTrue(report.ok, report.as_dict())
        self.assertEqual(report.readiness["words"]["records_with_artifact"], 1)
        self.assertEqual(report.readiness["words"]["rows"], 1)
        self.assertEqual(report.readiness["words"]["micro_flagged_rows"], 1)
        redacted = json.dumps(report.as_dict(), ensure_ascii=False)
        self.assertNotIn("민감한 문장", redacted)
        loaded = load_words_artifact(
            artifact, expected_recording_id="opaque-501",
            expected_sha256=hashlib.sha256(words).hexdigest(),
            duration_us=2_000_000, uem=((0, 2_000_000),), reference_speakers={"REF_00"},
        )
        self.assertEqual(loaded.words[0].word_id, "word-001")
        self.assertTrue(loaded.micro_flags)

    def test_words_schema_hash_timebase_and_record_id_fail_closed(self):
        row = self._row("opaque-601", "CALIBRATION")
        words_path = self.root / "words"
        words_path.mkdir()
        words = (
            '{"recording_id":"wrong-999","word_id":"word-001",'
            '"start_us":100000,"end_us":300000,"text":"secret",'
            '"ref_speaker_id":"REF_99","attributable":true,'
            '"overlap_flag":false,"boundary_crossing_flag":false,"micro_flag":"yes"}\n'
        ).encode("utf-8")
        (words_path / "opaque-601.jsonl").write_bytes(words)
        row.update({
            "words": "words/opaque-601.jsonl",
            "words_sha256": "0" * 64,
            "words_timebase": "seconds",
        })
        rows = [row, self._row("opaque-602", "DEVELOPMENT_HOLDOUT"),
                self._row("opaque-603", "RELEASE_HOLDOUT")]
        report = validate_annotation_dataset(self._write(rows))
        codes = {item.code for item in report.errors}
        self.assertTrue({"WORDS_HASH_MISMATCH", "WORDS_TIMEBASE_SCHEMA", "WORDS_RECORD_ID",
                         "WORDS_REF_SPEAKER_UNKNOWN", "WORDS_FLAG_SCHEMA"} <= codes)
        self.assertNotIn("secret", json.dumps(report.as_dict()))

    def test_annotation_hashes_and_speaker_group_split_are_bound(self):
        first = self._row("opaque-701", "CALIBRATION")
        second = self._row("opaque-702", "RELEASE_HOLDOUT")
        third = self._row("opaque-703", "DEVELOPMENT_HOLDOUT")
        for row in (first, second, third):
            row["rttm_sha256"] = hashlib.sha256((self.root / row["rttm"]).read_bytes()).hexdigest()
            row["uem_sha256"] = hashlib.sha256((self.root / row["uem"]).read_bytes()).hexdigest()
        first["speaker_group_ids"] = ["person-hmac-001"]
        second["speaker_group_ids"] = ["person-hmac-001"]
        third["speaker_group_ids"] = ["person-hmac-003"]
        first["rttm_sha256"] = "0" * 64
        report = validate_annotation_dataset(self._write([first, second, third]))
        codes = {item.code for item in report.errors}
        self.assertIn("ANNOTATION_HASH_MISMATCH", codes)
        self.assertIn("SPLIT_LEAKAGE", codes)

    def test_declared_speaker_groups_must_match_speaker_count(self):
        row = self._row("opaque-711", "CALIBRATION", speaker_count=1)
        row["speaker_group_ids"] = ["person-hmac-001", "person-hmac-002"]
        report = validate_annotation_dataset(self._write([
            row,
            self._row("opaque-712", "DEVELOPMENT_HOLDOUT"),
            self._row("opaque-713", "RELEASE_HOLDOUT"),
        ]))
        self.assertIn("SPEAKER_GROUP_COUNT_MISMATCH", {item.code for item in report.errors})

    def test_wave_format_extensible_pcm_is_accepted(self):
        row = self._row("opaque-501", "CALIBRATION")
        path = self.root / row["audio"]
        samples = b"\0\0" * 32_000
        guid = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt = struct.pack("<HHIIHHH", 0xFFFE, 1, 16000, 32000, 2, 16, 22)
        fmt += struct.pack("<HI", 16, 0) + guid
        chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
        path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)
        row["audio_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = validate_annotation_dataset(self._write([
            row,
            self._row("opaque-502", "DEVELOPMENT_HOLDOUT"),
            self._row("opaque-503", "RELEASE_HOLDOUT"),
        ]))
        self.assertTrue(report.ok, report.as_dict())

    def test_manifest_and_annotation_reads_are_bounded(self):
        row = self._row("opaque-801", "CALIBRATION")
        manifest = self._write([row])
        with mock.patch("sddiar.annotation_intake.MAX_MANIFEST_BYTES", 1):
            report = validate_annotation_dataset(manifest)
        self.assertIn("MANIFEST_READ", {item.code for item in report.errors})
        rows = [row, self._row("opaque-802", "DEVELOPMENT_HOLDOUT"),
                self._row("opaque-803", "RELEASE_HOLDOUT")]
        manifest = self._write(rows)
        with mock.patch("sddiar.annotation_intake.MAX_ANNOTATION_BYTES", 1):
            report = validate_annotation_dataset(manifest)
        self.assertIn("ANNOTATION_READ", {item.code for item in report.errors})

    def test_condition_labels_are_fixed_public_buckets(self):
        row = self._row("opaque-901", "CALIBRATION", conditions=["customer-acme"])
        report = validate_annotation_dataset(self._write([
            row, self._row("opaque-902", "DEVELOPMENT_HOLDOUT"),
            self._row("opaque-903", "RELEASE_HOLDOUT"),
        ]))
        self.assertIn("CONDITIONS_SCHEMA", {item.code for item in report.errors})
        self.assertNotIn("customer-acme", json.dumps(report.as_dict()))

    def test_deep_json_huge_time_and_enum_type_confusion_fail_as_redacted_issues(self):
        deep_manifest = self.root / "deep.jsonl"
        deep_manifest.write_text("[" * 2_000 + "]" * 2_000 + "\n", encoding="utf-8")
        report = validate_annotation_dataset(deep_manifest)
        self.assertTrue(
            {"MANIFEST_JSON", "MANIFEST_ROW_SCHEMA"}
            & {item.code for item in report.errors}
        )

        row = self._row("opaque-911", "CALIBRATION")
        row.update({
            "reference_status": [],
            "uem_policy": "FULL_AUDIO",
            "conversion_evidence_sha256": "a" * 64,
        })
        (self.root / row["rttm"]).write_text(
            "SPEAKER opaque-911 1 1e999999999 0.5 <NA> <NA> REF_00 <NA> <NA>\n",
            encoding="utf-8",
        )
        report = validate_annotation_dataset(self._write([
            row,
            self._row("opaque-912", "DEVELOPMENT_HOLDOUT"),
            self._row("opaque-913", "RELEASE_HOLDOUT"),
        ]))
        codes = {item.code for item in report.errors}
        self.assertIn("REFERENCE_STATUS_SCHEMA", codes)
        self.assertIn("RTTM_TIMING", codes)

    def test_words_deep_json_and_nonregular_manifest_fail_closed(self):
        words = self.root / "deep-words.jsonl"
        raw = ("[" * 2_000 + "]" * 2_000 + "\n").encode("utf-8")
        words.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "invalid words artifact"):
            load_words_artifact(
                words,
                expected_recording_id="opaque-921",
                expected_sha256=hashlib.sha256(raw).hexdigest(),
            )
        report = validate_annotation_dataset(self.root)
        self.assertIn("MANIFEST_READ", {item.code for item in report.errors})


if __name__ == "__main__":
    unittest.main()
