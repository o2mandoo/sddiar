from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import wave

from sddiar.blind_annotation import (
    BlindAnnotationError,
    build_blind_annotation_pack,
    public_evidence,
    verify_pack,
)


RATE = 8_000
SECONDS = 600


def _wav(path: Path, *, seconds: int = SECONDS) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        block = tuple(((index * 97) % 20_000) - 10_000 for index in range(4096))
        remaining = RATE * seconds
        while remaining:
            count = min(remaining, len(block))
            handle.writeframes(struct.pack("<%dh" % count, *block[:count]))
            remaining -= count


def _reference(path: Path) -> None:
    path.write_text(json.dumps({
        "turns": [
            {"start_us": index * 30_000_000, "end_us": index * 30_000_000 + 1_200_000,
             "speaker": "PRIVATE PERSON NAME", "text": "민감한 전사"}
            for index in range(1, 19)
        ]
    }), encoding="utf-8")


def _system(path: Path, *, marked: bool = True) -> None:
    rows = []
    for index in range(1, 16):
        row = {
            "start_us": index * 35_000_000 + 500_000,
            "end_us": index * 35_000_000 + 2_000_000,
            "speaker_id": "UNKNOWN" if marked else "HUMAN_PRIVATE_NAME",
            "attribution_status": "UNKNOWN_LOW_MARGIN" if marked and index % 3 == 0 else "ATTRIBUTED",
            "reason": "CONFLICT" if marked and index % 5 == 0 else "NONE",
        }
        rows.append(row)
    path.write_text(json.dumps({"spans": rows}), encoding="utf-8")


class BlindAnnotationTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path]:
        source, reference, system = root / "private-customer.wav", root / "reference.json", root / "system.json"
        _wav(source)
        _reference(reference)
        _system(system)
        return source, reference, system

    def _target(self, root: Path, name: str) -> Path:
        return root / ".private" / "blind-annotation" / name

    def test_sealed_and_blind_bundles_are_deterministic_and_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference, system = self._inputs(root)
            first = build_blind_annotation_pack(source, self._target(root, "pack-one"), reference_path=reference, system_path=system, repo_root=root, seed=7, presentation_nonce="secret-nonce")
            second = build_blind_annotation_pack(source, self._target(root, "pack-two"), reference_path=reference, system_path=system, repo_root=root, seed=7, presentation_nonce="secret-nonce")
            self.assertEqual(first.evidence, second.evidence)
            self.assertEqual(first.manifest_path.read_bytes(), second.manifest_path.read_bytes())
            self.assertEqual(first.label_template_path.read_bytes(), second.label_template_path.read_bytes())
            verified = verify_pack(first.manifest_path.parent, first.evidence["manifest_sha256"])
            self.assertEqual(verified, first.evidence)
            with first.manifest_path.open(encoding="utf-8") as handle:
                evaluator = json.load(handle)
            annotator = json.loads((first.manifest_path.parent / "annotator" / "manifest.json").read_text(encoding="utf-8"))
            template = first.label_template_path.read_text(encoding="utf-8")
            self.assertEqual(len(evaluator["clips"]), 48)
            self.assertEqual(evaluator["selection"]["metric_overlap_count"], 0)
            self.assertEqual(evaluator["selection"]["second_annotator_slot_count"], 12)
            clip_hashes = [row["audio_sha256"] for row in evaluator["clips"]]
            self.assertEqual(evaluator["selection"]["qc_duplicate_audio_count"], len(clip_hashes) - len(set(clip_hashes)))
            reference_points = [point for index in range(1, 19) for point in (index * 30_000_000, index * 30_000_000 + 1_200_000)]
            marked_intervals = [(index * 35_000_000 + 500_000, index * 35_000_000 + 2_000_000) for index in range(1, 16)]
            for row in evaluator["clips"]:
                start_frame = row["source_time_start_us"] * RATE // 1_000_000
                end_frame = row["source_time_end_us"] * RATE // 1_000_000
                if row["category"] == "REFERENCE_BOUNDARY_TIME_ONLY":
                    self.assertTrue(any(start_frame <= point * RATE // 1_000_000 < end_frame for point in reference_points))
                elif row["category"] == "SYSTEM_DISAGREEMENT_UNKNOWN_STRESS":
                    self.assertTrue(any(start_frame < end * RATE // 1_000_000 and end_frame > start * RATE // 1_000_000 for start, end in marked_intervals))
            self.assertEqual({row["category"] for row in evaluator["clips"]}, {
                "UNIFORM_RANDOM_NONOVERLAP_REMAINDER", "REFERENCE_BOUNDARY_TIME_ONLY", "SYSTEM_DISAGREEMENT_UNKNOWN_STRESS",
            })
            self.assertNotIn("category", json.dumps(annotator))
            self.assertNotIn("source_time", json.dumps(annotator))
            self.assertNotIn("audit_slot", json.dumps(annotator))
            self.assertTrue(all(len(row["audio_sha256"]) == 64 for row in annotator["clips"]))
            self.assertNotIn("PRIVATE PERSON NAME", json.dumps(evaluator) + json.dumps(annotator) + template)
            self.assertNotIn("HUMAN_PRIVATE_NAME", json.dumps(evaluator) + json.dumps(annotator) + template)
            self.assertNotIn("민감한 전사", json.dumps(evaluator, ensure_ascii=False) + json.dumps(annotator, ensure_ascii=False) + template)
            self.assertIn("HUMAN_SPK_0", template)
            self.assertIn("change boundary", template)
            self.assertEqual(len({path.name for path in first.clip_paths}), 48)

    def test_public_evidence_fixed_keys_and_clip_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference, system = self._inputs(root)
            result = build_blind_annotation_pack(source, self._target(root, "tamper"), reference_path=reference, system_path=system, repo_root=root)
            expected = result.evidence["manifest_sha256"]
            evidence = public_evidence(result.manifest_path, expected)
            self.assertEqual(set(evidence), {
                "schema_version", "manifest_sha256", "annotator_manifest_sha256", "label_template_sha256",
                "clip_count", "category_counts", "metric_union_duration_us", "metric_overlap_count",
                "metric_excluded_overlap_duration_us", "qc_duplicate_audio_count", "second_annotator_slot_count",
            })
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("template verifier must stream")):
                self.assertEqual(verify_pack(result.manifest_path.parent, expected), evidence)
            clip = result.clip_paths[0]
            clip.write_bytes(clip.read_bytes() + b"tamper")
            with self.assertRaises(BlindAnnotationError):
                verify_pack(result.manifest_path.parent, expected)

    def test_explicit_schema_unknown_and_rttm_resource_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference, system = self._inputs(root)
            reference.write_text(json.dumps({"nested": {"turns": json.loads(reference.read_text())["turns"]}}), encoding="utf-8")
            with self.assertRaises(BlindAnnotationError):
                build_blind_annotation_pack(source, self._target(root, "deep"), reference_path=reference, system_path=system, repo_root=root)
            _reference(reference)
            system.write_text("SPEAKER " + ("x" * 9000) + " 1 0.0 1.0\n", encoding="utf-8")
            with self.assertRaises(BlindAnnotationError):
                build_blind_annotation_pack(source, self._target(root, "long-rttm"), reference_path=reference, system_path=system, repo_root=root)

    def test_no_marked_system_events_fail_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference, system = self._inputs(root)
            _system(system, marked=False)
            with self.assertRaisesRegex(BlindAnnotationError, "normalized"):
                build_blind_annotation_pack(source, self._target(root, "unmarked"), reference_path=reference, system_path=system, repo_root=root)

    def test_path_symlink_permissions_gitignore_and_late_failure_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference, system = self._inputs(root)
            with self.assertRaises(BlindAnnotationError):
                build_blind_annotation_pack(source, root / ".private" / "blind-annotation" / ".." / "escape", reference_path=reference, system_path=system, repo_root=root)
            linked = root / "linked"
            try:
                linked.symlink_to(root / "real", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaises(BlindAnnotationError):
                build_blind_annotation_pack(source, linked / ".private" / "blind-annotation" / "bad", reference_path=reference, system_path=system, repo_root=root)
            target = self._target(root, "late-failure")
            original = __import__("sddiar.blind_annotation", fromlist=["_write_file"])._write_file

            def fail_manifest(path: Path, payload: bytes) -> str:
                if path.name == "manifest.json":
                    raise BlindAnnotationError("injected late failure")
                return original(path, payload)

            with mock.patch("sddiar.blind_annotation._write_file", side_effect=fail_manifest):
                with self.assertRaises(BlindAnnotationError):
                    build_blind_annotation_pack(source, target, reference_path=reference, system_path=system, repo_root=root)
            self.assertFalse(target.exists())
            self.assertFalse(any(item.name.startswith(".late-failure.staging-") for item in target.parent.iterdir()))
            self.assertFalse(any(item.name.startswith(".sddiar-input-") for item in target.parent.iterdir()))

    def test_permissions_are_owner_only_and_unknown_schema_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference, system = self._inputs(root)
            result = build_blind_annotation_pack(source, self._target(root, "modes"), reference_path=reference, system_path=system, repo_root=root)
            for path in [result.manifest_path, result.label_template_path, *result.clip_paths]:
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            for directory in [result.manifest_path.parent, result.manifest_path.parent / "annotator", result.manifest_path.parent / "annotator" / "clips"]:
                self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual(hashlib.sha256(result.manifest_path.read_bytes()).hexdigest(), result.evidence["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
