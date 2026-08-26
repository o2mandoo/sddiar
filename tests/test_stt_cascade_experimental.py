from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from sddiar.stt_cascade_experimental import (
    SttCascadeArtifactError,
    SttCascadeContractError,
    VerifiedLocalSttPack,
    analyze_stt_cascade_oracle,
    verify_local_stt_pack,
)


def metric_payload(segments, *, reference_unit_count=100):
    return {
        "metric_kind": "character_edit_count",
        "reference_unit_count": reference_unit_count,
        "segments": segments,
    }


class SttCascadeExperimentalTests(unittest.TestCase):
    def test_scalar_oracle_curve_is_redacted_and_default_off(self):
        rows = metric_payload([
            {"start_us": 0, "end_us": 1_000_000, "draft_error_count": 10, "refiner_error_count": 8},
            {"start_us": 1_000_000, "end_us": 3_000_000, "draft_error_count": 2, "refiner_error_count": 2},
            {"start_us": 3_000_000, "end_us": 4_000_000, "draft_error_count": 8, "refiner_error_count": 1},
            {"start_us": 4_000_000, "end_us": 8_000_000, "draft_error_count": 1, "refiner_error_count": 4},
        ])
        report = analyze_stt_cascade_oracle(rows)
        self.assertEqual(report["schema"], "stt_cascade_oracle_v2")
        self.assertFalse(report["default_enabled"])
        self.assertEqual(report["release_authority"], "none")
        self.assertEqual(report["quality_status"], "REVIEW_REQUIRED")
        self.assertEqual([row["budget_percent"] for row in report["oracle_curve"]], [10, 20, 30, 40])
        self.assertEqual(report["budget_basis"], "duration_us")
        self.assertEqual(report["oracle_curve"][0]["selected_segment_count"], 0)
        self.assertEqual(report["oracle_curve"][1]["oracle_error_count"], 14)
        self.assertEqual(report["oracle_curve"][1]["router_hard_segment_recall"], 0.5)
        self.assertEqual(report["segment_count_curve"][0]["selected_segment_count"], 1)
        encoded = json.dumps(report, ensure_ascii=True)
        self.assertNotIn("text", encoded.lower())  # only the privacy contract key is allowed below
        self.assertNotIn("transcript", encoded.lower())
        self.assertNotIn("segment_id", encoded.lower())
        self.assertNotIn("/", encoded)

    def test_redacted_character_counts_become_scalar_errors_and_risk(self):
        report = analyze_stt_cascade_oracle(metric_payload([
            {"start_us": 0, "end_us": 100, "draft_error_count": 0, "refiner_error_count": 2,
             "draft_redacted_chars": 10, "refiner_redacted_chars": 8, "reference_redacted_chars": 10,
             "stitch_duplicate_risk": 0.1},
            {"start_us": 100, "end_us": 200, "draft_error_count": 0, "refiner_error_count": 10,
             "draft_redacted_chars": 10, "refiner_redacted_chars": 0, "reference_redacted_chars": 10},
        ], reference_unit_count=20))
        self.assertEqual(report["draft_error_count"], 0)
        self.assertEqual(report["full_refiner_error_count"], 12)
        self.assertGreater(report["full_refiner"]["stitch_duplicate_risk"], 0)
        self.assertEqual(report["full_refiner"]["stitch_deletion_risk"], 0.6)

    def test_character_lengths_cannot_masquerade_as_asr_error(self):
        with self.assertRaises(SttCascadeContractError):
            analyze_stt_cascade_oracle(metric_payload([{
                "start_us": 0, "end_us": 10,
                "draft_redacted_chars": 100, "refiner_redacted_chars": 1,
            }]))

    def test_oracle_rejects_transcript_text_at_top_level_and_segment(self):
        payload = metric_payload([{
            "start_us": 0, "end_us": 10,
            "draft_error_count": 1, "refiner_error_count": 0,
            "text": "secret transcript",
        }])
        with self.assertRaises(SttCascadeContractError):
            analyze_stt_cascade_oracle(payload)
        payload = metric_payload([{
            "start_us": 0, "end_us": 10,
            "draft_error_count": 1, "refiner_error_count": 0,
        }])
        payload["transcript"] = "secret transcript"
        with self.assertRaises(SttCascadeContractError):
            analyze_stt_cascade_oracle(payload)

    def test_invalid_segment_contract_fails_closed(self):
        with self.assertRaises(SttCascadeContractError):
            analyze_stt_cascade_oracle(metric_payload([{"start_us": 2, "end_us": 1, "draft_error_count": 1, "refiner_error_count": 1}]))

    def test_duration_budget_never_overshoots_and_nonpositive_gain_routes_zero(self):
        report = analyze_stt_cascade_oracle(metric_payload([
            {"start_us": 0, "end_us": 90, "draft_error_count": 1, "refiner_error_count": 2},
            {"start_us": 90, "end_us": 100, "draft_error_count": 1, "refiner_error_count": 1},
        ]), include_segment_count_diagnostic=False)
        self.assertTrue(all(row["selected_segment_count"] == 0 for row in report["oracle_curve"]))
        with self.assertRaises(SttCascadeContractError):
            analyze_stt_cascade_oracle(metric_payload([{"start_us": 0, "end_us": 1, "draft_error_count": 1, "refiner_error_count": 0}]), max_segments=0)
        with self.assertRaises(SttCascadeContractError):
            analyze_stt_cascade_oracle(metric_payload([{"start_us": 0, "end_us": 1, "draft_error_count": 1, "refiner_error_count": 0}]), max_total_duration_us=0)
        with self.assertRaises(SttCascadeContractError):
            analyze_stt_cascade_oracle(metric_payload([
                {"start_us": 0, "end_us": 1, "draft_error_count": 1, "refiner_error_count": 0},
                {"start_us": 1, "end_us": 2, "draft_error_count": 1, "refiner_error_count": 0},
            ]), max_segments=1)

    def test_pack_requires_three_roles_and_is_deeply_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptors = []
            for artifact_id, group in (("engine", "engine"), ("model", "model"), ("tokenizer", "tokenizer")):
                path = root / f"{artifact_id}.bin"
                path.write_bytes(artifact_id.encode())
                descriptors.append({"artifact_id": artifact_id, "group": group, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            pack = verify_local_stt_pack({"pack_id": "fixture", "strategy": "ctranslate2", "runtime_abi": "cp311", "platform": "darwin-arm64", "artifact_root": str(root), "artifacts": descriptors})
            self.assertIsInstance(pack, VerifiedLocalSttPack)
            self.assertEqual(set(pack.artifacts), {"engine", "model", "tokenizer"})
            with self.assertRaises(TypeError):
                pack.artifacts["engine"] = pack.artifacts["model"]
            with self.assertRaises(TypeError):
                pack.public_identity()["artifacts"] = {}
            self.assertNotIn("path", repr(pack.public_identity()).lower())
            self.assertNotIn("path", json.dumps(pack.to_dict()).lower())
            (root / "tokenizer.bin").write_bytes(b"changed")
            with self.assertRaises(SttCascadeArtifactError):
                pack.assert_artifacts_unchanged()

    def test_pack_rejects_missing_role_and_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vad.bin"
            path.write_bytes(b"vad")
            descriptor = {"artifact_id": "vad", "group": "vad", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaises(SttCascadeArtifactError):
                verify_local_stt_pack({"pack_id": "fixture", "strategy": "ctranslate2", "runtime_abi": "cp311", "platform": "darwin-arm64", "artifact_root": str(path.parent), "artifacts": [descriptor]})
            with self.assertRaises(SttCascadeArtifactError):
                verify_local_stt_pack({"pack_id": "fixture", "strategy": "ctranslate2", "runtime_abi": "cp311", "platform": "darwin-arm64", "artifact_root": str(path.parent), "artifacts": [{"artifact_id": "vad", "group": "vad", "path": "https://example.invalid/vad", "sha256": "0" * 64}]})

    def test_pack_rejects_text_fields_and_duplicate_canonical_ct2_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared.bin"
            shared.write_bytes(b"shared")
            digest = hashlib.sha256(shared.read_bytes()).hexdigest()
            rows = [
                {"artifact_id": group, "group": group, "path": shared.name, "sha256": digest, "bytes": shared.stat().st_size}
                for group in ("engine", "model", "tokenizer")
            ]
            with self.assertRaises(SttCascadeArtifactError):
                verify_local_stt_pack({
                    "pack_id": "fixture", "strategy": "ctranslate2", "runtime_abi": "cp311",
                    "platform": "darwin-arm64", "artifact_root": str(root), "artifacts": rows,
                })
            rows[0]["text"] = "secret"
            with self.assertRaises(SttCascadeArtifactError):
                verify_local_stt_pack({
                    "pack_id": "fixture", "strategy": "ctranslate2", "runtime_abi": "cp311",
                    "platform": "darwin-arm64", "artifact_root": str(root), "artifacts": rows,
                })

    @unittest.skipIf(os.name == "nt", "symlink fixture requires POSIX permissions")
    def test_pack_rechecks_artifact_root_parent_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            real_parent = base / "real-parent"
            real_parent.mkdir()
            root = real_parent / "pack"
            root.mkdir()
            rows = []
            for artifact_id, group in (("engine", "engine"), ("model", "model"), ("tokenizer", "tokenizer")):
                path = root / f"{artifact_id}.bin"
                path.write_bytes(artifact_id.encode())
                rows.append({"artifact_id": artifact_id, "group": group, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
            link = base / "pack-link"
            link.symlink_to(real_parent, target_is_directory=True)
            pack = verify_local_stt_pack({
                "pack_id": "fixture", "strategy": "ctranslate2", "runtime_abi": "cp311",
                "platform": "darwin-arm64", "artifact_root": str(link / "pack"), "artifacts": rows,
            })
            other_parent = base / "other-parent"
            other_parent.mkdir()
            shutil.copytree(root, other_parent / "pack")
            link.unlink()
            link.symlink_to(other_parent, target_is_directory=True)
            with self.assertRaises(SttCascadeArtifactError):
                pack.assert_artifacts_unchanged()

    def test_pack_rejects_artifacts_outside_explicit_root(self):
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir) / "engine.bin"
            outside.write_bytes(b"engine")
            descriptor = {
                "artifact_id": "engine", "group": "engine", "path": str(outside),
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
            with self.assertRaises(SttCascadeArtifactError):
                verify_local_stt_pack({
                    "pack_id": "fixture", "strategy": "ctranslate2",
                    "runtime_abi": "cp311", "platform": "linux-x86_64",
                    "artifact_root": root_dir, "artifacts": [descriptor],
                })

    def test_pack_rejects_oversized_file_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "large.bin"
            artifact.write_bytes(b"four")
            descriptor = {
                "artifact_id": "engine", "group": "engine", "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            with patch("sddiar.stt_cascade_experimental.MAX_STT_PACK_BYTES", 3):
                with self.assertRaises(SttCascadeArtifactError):
                    verify_local_stt_pack({
                        "pack_id": "fixture", "strategy": "ctranslate2",
                        "runtime_abi": "cp311", "platform": "linux-x86_64",
                        "artifact_root": str(root), "artifacts": [descriptor],
                    })

    def test_directory_artifact_and_whisper_strategy_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "openvino-model"
            model.mkdir()
            (model / "model.xml").write_bytes(b"xml")
            (model / "model.bin").write_bytes(b"bin")
            descriptors = []
            for artifact_id, group, path in (("engine", "engine", root / "engine"), ("model", "model", model), ("vad", "vad", root / "vad"), ("encoder", "encoder", root / "encoder"), ("runtime", "runtime", root / "runtime")):
                if path != model:
                    path.write_bytes(artifact_id.encode())
                digest = hashlib.sha256()
                if path.is_file():
                    digest.update(path.read_bytes())
                    size, count = path.stat().st_size, 1
                else:
                    size, count = 0, 0
                    for child in sorted(path.rglob("*")):
                        if child.is_file():
                            payload = child.read_bytes(); child_digest = hashlib.sha256(payload).hexdigest()
                            digest.update(child.relative_to(path).as_posix().encode() + b"\0" + child_digest.encode() + b"\0" + str(len(payload)).encode() + b"\n")
                            size += len(payload); count += 1
                descriptors.append({"artifact_id": artifact_id, "group": group, "path": str(path), "sha256": digest.hexdigest(), "bytes": size})
            pack = verify_local_stt_pack({"pack_id": "ov", "strategy": "whispercpp_openvino", "runtime_abi": "cp311", "platform": "linux-x86_64", "artifact_root": str(root), "artifacts": descriptors})
            self.assertEqual(pack.artifacts["model"].file_count, 2)

    def test_duration_oracle_300_seconds_25_segments_is_bounded(self):
        rows = metric_payload([
            {"start_us": index * 12_000_000, "end_us": (index + 1) * 12_000_000,
             "draft_error_count": (index % 7) + 1, "refiner_error_count": index % 3}
            for index in range(25)
        ], reference_unit_count=500)
        started = time.perf_counter()
        report = analyze_stt_cascade_oracle(rows)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 5.0)
        self.assertEqual(report["total_duration_us"], 300_000_000)
        self.assertLessEqual(max(item["selected_duration_us"] for item in report["oracle_curve"]), 120_000_000)
        self.assertIn("oracle_curve_is_an_upper_bound_not_release_evidence", report["limitations"])

    def test_larger_oracle_uses_state_bounded_solver(self):
        rows = metric_payload([
            {"start_us": index * 1_000_000, "end_us": (index + 1) * 1_000_000,
             "draft_error_count": (index % 5) + 1, "refiner_error_count": 0}
            for index in range(30)
        ], reference_unit_count=600)
        report = analyze_stt_cascade_oracle(rows, max_solver_states=1_000)
        self.assertTrue(all(row["selected_duration_us"] <= row["target_duration_us"]
                            for row in report["oracle_curve"]))

    def test_cli_emits_only_report(self):
        payload = json.dumps(metric_payload([
            {"start_us": 0, "end_us": 10, "draft_error_count": 2, "refiner_error_count": 1}
        ]))
        script = Path(__file__).parents[1] / "scripts" / "analyze_stt_cascade_oracle.py"
        completed = subprocess.run([sys.executable, str(script), "-"], input=payload, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["quality_status"], "REVIEW_REQUIRED")
        self.assertNotIn("/", completed.stdout)

    def test_cli_error_does_not_echo_input_path(self):
        script = Path(__file__).parents[1] / "scripts" / "analyze_stt_cascade_oracle.py"
        missing = "/private/sensitive/customer/input.json"
        completed = subprocess.run([sys.executable, str(script), missing], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertNotIn(missing, completed.stderr)
        self.assertNotIn("customer", completed.stderr)


if __name__ == "__main__":
    unittest.main()
