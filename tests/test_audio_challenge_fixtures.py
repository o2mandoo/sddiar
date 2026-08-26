from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.build_audio_challenge_fixtures import (
    FixtureError,
    build_fixtures,
    build_plan,
    evaluate_manifest,
)


def _source_wav(path: Path, frame_count: int = 16_003) -> None:
    values = tuple(((index * 7919) % 24_000) - 12_000 for index in range(frame_count))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(struct.pack("<%dh" % len(values), *values))


def _reference(path: Path, *, first_label: str = "REF_00", second_label: str = "REF_01") -> None:
    path.write_text(
        json.dumps(
            {
                "turns": [
                    {"start_us": 0, "end_us": 250_000, "speaker": first_label},
                    {"start_us": 500_000, "end_us": 750_000, "speaker": second_label},
                ]
            }
        ),
        encoding="utf-8",
    )


class AudioChallengeFixtureTests(unittest.TestCase):
    def test_plan_only_is_offline_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, output = root / "private-original-name.wav", root / "timing.json", root / "planned"
            _source_wav(source)
            _reference(reference)
            plan = build_plan(source, reference, chunk_frames=31)
            self.assertFalse(output.exists())
            self.assertTrue(plan.relationship_id.startswith("rel-"))
            self.assertEqual(len(plan.artifacts), 4)
            self.assertTrue(all(item.filename.startswith("artifact-") for item in plan.artifacts))

    def test_artifacts_are_deterministic_across_chunking_and_hashes_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, first, second = root / "input.wav", root / "timing.json", root / "one", root / "two"
            _source_wav(source)
            _reference(reference)
            report_one = build_fixtures(source, reference, first, chunk_frames=17)
            report_two = build_fixtures(source, reference, second, chunk_frames=4096)
            self.assertEqual(
                [entry["sha256"] for entry in report_one["artifacts"]],
                [entry["sha256"] for entry in report_two["artifacts"]],
            )
            self.assertTrue(evaluate_manifest(first / "manifest.json", source_path=source, timing_reference_path=reference)["ok"])
            self.assertTrue(evaluate_manifest(second / "manifest.json")["ok"])
            for entry in report_one["artifacts"]:
                path = first / entry["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])

    def test_wav_contract_duration_mapping_and_no_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, output = root / "source.wav", root / "timing.json", root / "fixtures"
            _source_wav(source, frame_count=16_003)
            _reference(reference)
            report = build_fixtures(source, reference, output)
            for entry in report["artifacts"]:
                with wave.open(str(output / entry["path"]), "rb") as handle:
                    self.assertEqual((handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()), (16_000, 1, 2, "NONE"))
                    self.assertEqual(handle.getnframes(), 16_003)
                    samples = struct.unpack("<%dh" % handle.getnframes(), handle.readframes(handle.getnframes()))
                self.assertLessEqual(max(samples), 32_767)
                self.assertGreaterEqual(min(samples), -32_768)
                self.assertEqual(entry["duration_us"], 1_000_187)
                self.assertEqual(entry["source_time_start_us"], 0)
                self.assertEqual(entry["source_time_end_us"], entry["duration_us"])
                self.assertEqual(entry["clipped_sample_count"], 0)

    def test_speaker_labels_and_raw_names_do_not_affect_audio_or_relationship(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "customer-real-name.wav"
            reference_one, reference_two = root / "one.json", root / "two.json"
            first, second = root / "first", root / "second"
            _source_wav(source)
            _reference(reference_one, first_label="Alice", second_label="Bob")
            _reference(reference_two, first_label="totally-private-name", second_label="another-private-name")
            report_one = build_fixtures(source, reference_one, first)
            report_two = build_fixtures(source, reference_two, second)
            self.assertEqual(report_one["challenge_relationship_id"], report_two["challenge_relationship_id"])
            self.assertEqual([a["sha256"] for a in report_one["artifacts"]], [a["sha256"] for a in report_two["artifacts"]])
            manifest_text = (first / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn(source.name, manifest_text)
            self.assertNotIn("Alice", manifest_text)
            self.assertNotIn("Bob", manifest_text)
            self.assertNotIn("private", manifest_text.lower())
            self.assertFalse(report_one["timing_reference"]["speaker_labels_used"])

    def test_existing_files_fail_without_overwrite_and_tampering_fails_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reference, output = root / "source.wav", root / "timing.json", root / "fixtures"
            _source_wav(source)
            _reference(reference)
            build_fixtures(source, reference, output)
            with self.assertRaises(FixtureError):
                build_fixtures(source, reference, output)
            artifact = next(output.glob("artifact-*.wav"))
            artifact.write_bytes(artifact.read_bytes() + b"x")
            result = evaluate_manifest(output / "manifest.json")
            self.assertFalse(result["ok"])
            self.assertTrue(any("hash mismatch" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
