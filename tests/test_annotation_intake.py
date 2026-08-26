from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import struct
import wave
import unittest

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


if __name__ == "__main__":
    unittest.main()
