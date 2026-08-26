import hashlib
import json
import os
import platform
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import sddiar.rnnoise_experimental as rnnoise_module
from sddiar.media import WavPcmDecoder
from sddiar.rnnoise_experimental import (
    ExperimentalRNNoisePreprocessor,
    NativeInvocation,
    NativeRunOutcome,
    RNNoiseArtifactError,
    RNNoiseBusyError,
    RNNoiseConfigurationError,
    RNNoiseEnhancementPolicy,
    RNNoiseExecutionError,
    RNNoiseReceipt,
    RNNoiseTimebaseError,
    SubprocessArgvRunner,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _descriptor(name: str, digest: str, size: int = 1):
    return {"logical_name": name, "sha256": digest, "bytes": size}


def _duration_us(frame_count: int) -> int:
    return (frame_count * 1_000_000 + 8_000) // 16_000


def _write_wav(path: Path, frame_count: int = 321) -> None:
    values = b"".join(((index % 200) - 100).to_bytes(2, "little", signed=True) for index in range(frame_count))
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(values)


class FakeNativeRunner:
    def __init__(self, *, wrong_stage: str | None = None, extra_file: bool = False, mutate: Path | None = None):
        self.invocations = []
        self.wrong_stage = wrong_stage
        self.extra_file = extra_file
        self.mutate = mutate
        self.active = 0
        self.maximum_active = 0
        self.guard = threading.Lock()

    @staticmethod
    def _value(argv, name):
        return argv[argv.index(name) + 1]

    def run(self, invocation):
        self.invocations.append(invocation)
        with self.guard:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.003)
            if invocation.stage == "UPSAMPLE_48K":
                samples = int(self._value(invocation.argv, "--exact-output-samples"))
                source = Path(self._value(invocation.argv, "--input")).read_bytes()
                words = [source[index:index + 2] for index in range(0, len(source), 2)]
                payload = b"".join(word * 3 for word in words)
                if len(payload) != samples * 2:
                    raise AssertionError("fake upsampler contract mismatch")
                if invocation.stage == self.wrong_stage:
                    payload += b"\x00\x00"
                invocation.expected_output.write_bytes(payload)
            elif invocation.stage == "DOWNSAMPLE_16K":
                samples = int(self._value(invocation.argv, "--exact-output-samples"))
                source = Path(self._value(invocation.argv, "--input")).read_bytes()
                payload = b"".join(source[index:index + 2] for index in range(0, len(source), 6))
                payload = payload[:samples * 2]
                if invocation.stage == self.wrong_stage:
                    payload += b"\x00\x00"
                invocation.expected_output.write_bytes(payload)
            elif invocation.stage == "RNNOISE_48K":
                source = Path(invocation.argv[1]).read_bytes()
                # rnnoise_process_frame emits the previous delayed frame.  The
                # demo drops the first zero output, so an appended flush input
                # makes its output equal every prior signal/pad input frame.
                payload = source[:-480 * 2]
                if invocation.stage == self.wrong_stage:
                    payload = payload[:-2]
                invocation.expected_output.write_bytes(payload)
            else:
                raise AssertionError(invocation.stage)
            if self.extra_file:
                (invocation.cwd / "unexpected-secret.log").write_bytes(b"not allowed")
                self.extra_file = False
            if self.mutate is not None:
                self.mutate.write_bytes(self.mutate.read_bytes() + b"tampered")
                self.mutate = None
            return NativeRunOutcome(invocation.stage, 0)
        finally:
            with self.guard:
                self.active -= 1


class RNNoiseExperimentalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_root = self.root / "sensitive input"
        self.artifact_root = self.root / "native artifacts"
        self.work_root = self.root / "private work"
        for directory in (self.input_root, self.artifact_root, self.work_root):
            directory.mkdir()
        self.audio = self.input_root / "private-client-name; touch injected.wav"
        _write_wav(self.audio)
        self.rnnoise = self.artifact_root / "rnnoise_demo"
        self.resampler = self.artifact_root / "raw_resampler"
        self.rnnoise.write_bytes(b"rnnoise-native-fixture")
        self.resampler.write_bytes(b"resampler-native-fixture")
        if os.name != "nt":
            self.rnnoise.chmod(0o700)
            self.resampler.chmod(0o700)
        self.build_attestation = self.artifact_root / "rnnoise-build-attestation.json"
        runtime_target = {
            ("Windows", "amd64"): "windows-x86_64",
            ("Windows", "x86_64"): "windows-x86_64",
            ("Linux", "x86_64"): "linux-x86_64",
            ("Linux", "aarch64"): "linux-aarch64",
            ("Linux", "arm64"): "linux-aarch64",
            ("Darwin", "x86_64"): "macos-x86_64",
            ("Darwin", "arm64"): "macos-arm64",
            ("Darwin", "aarch64"): "macos-arm64",
        }[(platform.system(), platform.machine().lower())]
        roles = {
            "windows-x86_64": ["c_compiler", "mingw_w64", "posix_shell", "autoconf", "automake", "libtool", "make"],
            "linux-x86_64": ["c_compiler", "sysroot", "autoconf", "automake", "libtool", "make"],
            "linux-aarch64": ["c_compiler", "sysroot", "autoconf", "automake", "libtool", "make"],
            "macos-x86_64": ["c_compiler", "macos_sdk", "autoconf", "automake", "libtool", "make"],
            "macos-arm64": ["c_compiler", "macos_sdk", "autoconf", "automake", "libtool", "make"],
        }[runtime_target]
        configure = ["sh", "./configure", "--host=x86_64-w64-mingw32"] if runtime_target == "windows-x86_64" else ["./configure"]
        configure.extend(
            [
                "--disable-doc",
                "--enable-examples",
                "--disable-x86-rtcd",
                "--disable-shared",
                "--enable-static",
            ]
        )
        native_name = "rnnoise_demo.exe" if runtime_target == "windows-x86_64" else "rnnoise_demo"
        model_sha = "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"
        empty_submodules = []
        attestation = {
            "schema_version": "1.0",
            "kind": "rnnoise-offline-build-attestation",
            "experimental": True,
            "default_enabled": False,
            "production_approved": False,
            "source": {
                "repository": "https://gitlab.xiph.org/xiph/rnnoise.git",
                "expected_commit": "70f1d256acd4b34a572f999a05c87bf00b67730d",
                "checkout_commit": "70f1d256acd4b34a572f999a05c87bf00b67730d",
                "source_archive": _descriptor("rnnoise-source.tar", "1" * 64),
                "tracked_tree_sha256": "2" * 64,
                "license_spdx": "BSD-3-Clause",
            },
            "submodules": {
                "records": empty_submodules,
                "count": 0,
                "canonical_sha256": _canonical_sha256(empty_submodules),
            },
            "model": {
                "archive": _descriptor(f"rnnoise_data-{model_sha}.tar.gz", model_sha),
                "expected_sha256": model_sha,
                "model_version": model_sha,
                "imported_offline": True,
                "staged_files": [
                    {"relative_path": "src/rnnoise_data.c", "sha256": "3" * 64, "bytes": 1},
                    {"relative_path": "src/rnnoise_data.h", "sha256": "4" * 64, "bytes": 1},
                ],
            },
            "target": {"id": runtime_target, "endianness": "little"},
            "host": {
                "system": platform.system(),
                "machine": platform.machine().lower(),
                "matches_target": True,
            },
            "toolchain": {
                "manifest": _descriptor("toolchain.json", "5" * 64),
                "manifest_payload_sha256": "6" * 64,
                "required_roles": roles,
            },
            "configuration": {
                "commands": [["autoreconf", "-isf"], configure, ["make", "-j1", f"examples/{native_name}"]],
                "autogen_executed": False,
                "download_model_executed": False,
                "build_network_required_state": "disabled",
                "x86_rtcd": False,
                "compile_time_vectorization": "target_compiler_default_not_scalar_claim",
                "jobs": 1,
            },
            "native_binary": _descriptor(native_name, _sha256(self.rnnoise), self.rnnoise.stat().st_size),
            "build_log": _descriptor("build.log", "7" * 64),
            "dependency_report": _descriptor("dependencies.json", "8" * 64),
            "validation": {
                "single_target_native_build_recorded": True,
                "binary_functional_smoke": "not_run",
                "four_platform_native_validation": "not_run",
                "xeon_validation": "not_run",
                "independent_review": "not_run",
                "release_authority": "none",
            },
        }
        integrity_inputs = {
            key: attestation[key]
            for key in ("source", "submodules", "model", "target", "host", "toolchain", "configuration")
        }
        integrity_outputs = {
            key: attestation[key] for key in ("native_binary", "build_log", "dependency_report")
        }
        inputs_sha = _canonical_sha256(integrity_inputs)
        outputs_sha = _canonical_sha256(integrity_outputs)
        attestation["integrity"] = {
            "build_inputs_sha256": inputs_sha,
            "build_outputs_sha256": outputs_sha,
            "statement_sha256": _canonical_sha256(
                {
                    "schema_version": attestation["schema_version"],
                    "kind": attestation["kind"],
                    "inputs_sha256": inputs_sha,
                    "outputs_sha256": outputs_sha,
                    "validation": attestation["validation"],
                }
            ),
            "signature_status": "unsigned",
            "verification_scope": "hash_bound_structure_only",
            "cryptographic_authenticity": "not_verified",
        }
        self.build_attestation.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
        self.timebase_proof = self.artifact_root / "rnnoise-timebase-proof.json"
        self.timebase_proof.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "kind": "rnnoise-resampler-timebase-proof",
                    "experimental": True,
                    "production_approved": False,
                    "artifacts": {
                        "rnnoise_binary_sha256": _sha256(self.rnnoise),
                        "resampler_binary_sha256": _sha256(self.resampler),
                    },
                    "contract": {
                        "argv_contract_version": "raw-s16le-exact-v1",
                        "source_sample_rate_hz": 16_000,
                        "rnnoise_sample_rate_hz": 48_000,
                        "demo_warmup_output_samples_48k": 480,
                        "demo_flush_input_samples_48k": 480,
                        "roundtrip_impulse_peak_shift_samples_16k": 0,
                        "first_marker_shift_samples_16k": 0,
                        "last_marker_shift_samples_16k": 0,
                        "exact_duration_frame_cases": [1, 159, 160, 161, 479, 480, 481, 16_003],
                        "all_frame_cases_exact": True,
                    },
                    "review": {
                        "authority": "CALLER_HASH_BOUND_EXPERIMENTAL",
                        "release_authority": "none",
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def preprocessor(self, runner=None, **policy_values):
        return ExperimentalRNNoisePreprocessor(
            policy=RNNoiseEnhancementPolicy(enabled=True, **policy_values),
            rnnoise_binary=self.rnnoise,
            rnnoise_binary_sha256=_sha256(self.rnnoise),
            resampler_binary=self.resampler,
            resampler_binary_sha256=_sha256(self.resampler),
            rnnoise_build_attestation=self.build_attestation,
            rnnoise_build_attestation_sha256=_sha256(self.build_attestation),
            timebase_proof=self.timebase_proof,
            timebase_proof_sha256=_sha256(self.timebase_proof),
            artifact_root=self.artifact_root,
            input_root=self.input_root,
            work_root=self.work_root,
            runner=runner or FakeNativeRunner(),
        )

    def prepare_kwargs(self):
        return {
            "expected_source_sha256": _sha256(self.audio),
            "expected_duration_us": _duration_us(321),
        }

    def test_default_off_is_a_true_noop_without_binary_or_root_io(self):
        preprocessor = ExperimentalRNNoisePreprocessor()
        missing = self.root / "does-not-exist.wav"
        with preprocessor.prepare(missing) as prepared:
            self.assertFalse(prepared.enhanced)
            self.assertTrue(prepared.source_time_authorized)
            self.assertEqual(prepared.local_path, missing)
            self.assertEqual(prepared.receipt.status, "DISABLED")
            self.assertFalse(missing.exists())
        self.assertEqual(list(self.work_root.iterdir()), [])
        with self.assertRaises(RNNoiseConfigurationError):
            RNNoiseEnhancementPolicy(enabled="true")

    def test_happy_path_is_exact_length_dual_consumer_and_redacted(self):
        runner = FakeNativeRunner()
        preprocessor = self.preprocessor(runner)
        with preprocessor.prepare(self.audio, **self.prepare_kwargs()) as prepared:
            private_path = prepared.local_path
            self.assertTrue(private_path.is_file())
            self.assertTrue(prepared.enhanced)
            self.assertEqual(prepared.frame_count, 321)
            self.assertEqual(prepared.duration_us, _duration_us(321))
            chunks = tuple(prepared.iter_chunks(frames_per_chunk=100))
            self.assertEqual(sum(len(chunk.samples) for chunk in chunks), 321)
            region = prepared.read_mono_samples(0, prepared.duration_us)
            self.assertEqual(len(region), 321)
            chunk_samples = tuple(frame[0] for chunk in chunks for frame in chunk.samples)
            self.assertEqual(chunk_samples, region)
            original = tuple(
                frame[0]
                for chunk in tuple(WavPcmDecoder().iter_decode_chunks(self.audio))
                for frame in chunk.samples
            )
            self.assertEqual(region, original)
            self.assertEqual(prepared.source_sha256, _sha256(self.audio))
            self.assertIsNotNone(prepared.output_sha256)
            receipt = prepared.receipt.to_json()
            self.assertNotIn(str(self.root), receipt)
            self.assertNotIn(self.audio.name, receipt)
            self.assertNotIn("--input", receipt)
            self.assertEqual(json.loads(receipt)["production_approved"], False)
            self.assertTrue(json.loads(receipt)["timebase"]["sample_count_preserved"])
            self.assertEqual(
                json.loads(receipt)["timebase"]["evidence_status"],
                "CALLER_HASH_BOUND_STRUCTURAL_RECORD",
            )
            self.assertEqual(
                json.loads(receipt)["artifacts"]["lineage_status"],
                "CALLER_HASH_BOUND_STRUCTURAL_ATTESTATION",
            )
            self.assertEqual(
                json.loads(receipt)["execution_policy"]["native_child_egress"],
                "NOT_VERIFIED_REQUIRES_EXTERNAL_SANDBOX",
            )
            self.assertEqual(
                json.loads(receipt)["execution_policy"]["runner_execution_internals"],
                "INJECTED_NOT_VERIFIED",
            )
            self.assertFalse(prepared.source_time_authorized)
            for invocation in runner.invocations:
                self.assertIsInstance(invocation.argv, tuple)
                self.assertTrue(Path(invocation.argv[0]).is_absolute())
                self.assertNotIn(str(self.audio), invocation.argv)
        self.assertFalse(private_path.exists())
        self.assertEqual(list(self.work_root.iterdir()), [])

    def test_disabled_receipt_does_not_claim_rnnoise_latency_compensation(self):
        with ExperimentalRNNoisePreprocessor().prepare(self.audio) as prepared:
            timebase = prepared.receipt.to_dict()["timebase"]
        self.assertEqual(timebase["compensation"], "NOT_APPLIED")
        self.assertEqual(timebase["discarded_demo_warmup_output_samples_48k"], 0)
        self.assertEqual(timebase["flush_input_samples_48k"], 0)
        self.assertEqual(timebase["unrecoverable_initial_us"], 0)

    def test_unvalidated_receipt_object_cannot_claim_lineage_or_identity(self):
        receipt = RNNoiseReceipt(
            status="EXPERIMENTAL_APPLIED",
            source_sha256_prefix="a" * 16,
            output_sha256_prefix="b" * 16,
            source_frame_count=1,
            output_frame_count=1,
            duration_us=63,
            rnnoise_binary_sha256="c" * 64,
            resampler_binary_sha256="d" * 64,
            build_attestation_sha256="e" * 64,
            timebase_proof_sha256="f" * 64,
            runner_kind="INJECTED_NOT_VERIFIED",
            policy_sha256="0" * 64,
        ).to_dict()
        self.assertEqual(receipt["timebase"]["public_mapping"], "NOT_APPLIED")
        self.assertEqual(receipt["artifacts"]["lineage_status"], "NOT_APPLIED")

    def test_mismatched_build_attestation_or_timebase_proof_fails_before_runner(self):
        runner = FakeNativeRunner()
        original_proof = self.timebase_proof.read_text(encoding="utf-8")
        proof = json.loads(original_proof)
        proof["contract"]["roundtrip_impulse_peak_shift_samples_16k"] = 1
        self.timebase_proof.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")
        with self.assertRaises(RNNoiseTimebaseError):
            with self.preprocessor(runner).prepare(self.audio, **self.prepare_kwargs()):
                pass
        self.assertEqual(runner.invocations, [])

        # Restore a valid proof, then break the attested native lineage while
        # supplying the new outer file hash.  Structural mismatch still fails.
        self.timebase_proof.write_text(original_proof, encoding="utf-8")
        attestation = json.loads(self.build_attestation.read_text(encoding="utf-8"))
        attestation["native_binary"]["sha256"] = "0" * 64
        self.build_attestation.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
        with self.assertRaises(RNNoiseArtifactError):
            with self.preprocessor(runner).prepare(self.audio, **self.prepare_kwargs()):
                pass
        self.assertEqual(runner.invocations, [])

    def test_odd_tail_and_demo_flush_preserve_sample_count_without_shift(self):
        for frames in (1, 159, 160, 161, 479, 480, 481, 16_003):
            audio = self.input_root / f"tail-{frames}.wav"
            _write_wav(audio, frames)
            with self.preprocessor().prepare(
                audio,
                expected_source_sha256=_sha256(audio),
                expected_duration_us=_duration_us(frames),
            ) as prepared:
                with wave.open(str(prepared.local_path), "rb") as enhanced:
                    self.assertEqual(enhanced.getnframes(), frames)
                    self.assertEqual(enhanced.getframerate(), 16_000)

    def test_hash_path_and_timebase_fail_closed_before_publish(self):
        runner = FakeNativeRunner()
        with self.assertRaises(RNNoiseArtifactError):
            with self.preprocessor(runner).prepare(
                self.audio,
                expected_source_sha256="0" * 64,
                expected_duration_us=_duration_us(321),
            ):
                pass
        self.assertEqual(runner.invocations, [])

        with self.assertRaises(RNNoiseTimebaseError):
            with self.preprocessor(runner).prepare(
                self.audio,
                expected_source_sha256=_sha256(self.audio),
                expected_duration_us=_duration_us(321) + 1,
            ):
                pass
        self.assertEqual(runner.invocations, [])

        outside = self.root / "outside.wav"
        _write_wav(outside)
        with self.assertRaises(RNNoiseArtifactError):
            with self.preprocessor(runner).prepare(
                outside,
                expected_source_sha256=_sha256(outside),
                expected_duration_us=_duration_us(321),
            ):
                pass

        if hasattr(os, "symlink"):
            link = self.input_root / "linked.wav"
            try:
                link.symlink_to(self.audio)
            except OSError:
                pass
            else:
                with self.assertRaises(RNNoiseArtifactError):
                    with self.preprocessor(runner).prepare(
                        link,
                        expected_source_sha256=_sha256(self.audio),
                        expected_duration_us=_duration_us(321),
                    ):
                        pass

            real = self.input_root / "real-subdir"
            real.mkdir()
            nested = real / "nested.wav"
            _write_wav(nested)
            parent_link = self.input_root / "linked-subdir"
            try:
                parent_link.symlink_to(real, target_is_directory=True)
            except OSError:
                pass
            else:
                linked_nested = parent_link / "nested.wav"
                with self.assertRaises(RNNoiseArtifactError):
                    with self.preprocessor(runner).prepare(
                        linked_nested,
                        expected_source_sha256=_sha256(nested),
                        expected_duration_us=_duration_us(321),
                    ):
                        pass

    def test_binary_hash_is_rechecked_after_each_native_stage(self):
        original_hash = _sha256(self.resampler)
        runner = FakeNativeRunner(mutate=self.resampler)
        preprocessor = ExperimentalRNNoisePreprocessor(
            policy=RNNoiseEnhancementPolicy(enabled=True),
            rnnoise_binary=self.rnnoise,
            rnnoise_binary_sha256=_sha256(self.rnnoise),
            resampler_binary=self.resampler,
            resampler_binary_sha256=original_hash,
            rnnoise_build_attestation=self.build_attestation,
            rnnoise_build_attestation_sha256=_sha256(self.build_attestation),
            timebase_proof=self.timebase_proof,
            timebase_proof_sha256=_sha256(self.timebase_proof),
            artifact_root=self.artifact_root,
            input_root=self.input_root,
            work_root=self.work_root,
            runner=runner,
        )
        with self.assertRaises(RNNoiseArtifactError):
            with preprocessor.prepare(self.audio, **self.prepare_kwargs()):
                pass
        self.assertEqual(len(runner.invocations), 1)
        self.assertEqual(list(self.work_root.iterdir()), [])

    def test_source_replacement_between_initial_check_and_snapshot_fails_closed(self):
        alternate = self.input_root / "alternate.wav"
        _write_wav(alternate)
        raw = bytearray(alternate.read_bytes())
        raw[-1] ^= 0x7F
        alternate.write_bytes(bytes(raw))
        runner = FakeNativeRunner()
        expected = self.prepare_kwargs()
        original_verify = rnnoise_module._verified_executable
        replaced = False

        def verify_then_replace(*args, **kwargs):
            nonlocal replaced
            result = original_verify(*args, **kwargs)
            if not replaced and kwargs.get("role") == "RNNoise executable":
                os.replace(alternate, self.audio)
                replaced = True
            return result

        with mock.patch.object(rnnoise_module, "_verified_executable", side_effect=verify_then_replace):
            with self.assertRaises(RNNoiseArtifactError):
                with self.preprocessor(runner).prepare(self.audio, **expected):
                    pass
        self.assertEqual(runner.invocations, [])
        self.assertEqual(list(self.work_root.iterdir()), [])

    def test_wrong_size_and_unexpected_workspace_entry_fail_closed_and_clean(self):
        for runner, error in (
            (FakeNativeRunner(wrong_stage="RNNOISE_48K"), RNNoiseTimebaseError),
            (FakeNativeRunner(extra_file=True), RNNoiseExecutionError),
        ):
            with self.assertRaises(error):
                with self.preprocessor(runner).prepare(self.audio, **self.prepare_kwargs()):
                    pass
            self.assertEqual(list(self.work_root.iterdir()), [])

    def test_limits_are_enforced_before_native_invocation(self):
        runner = FakeNativeRunner()
        with self.assertRaises(RNNoiseConfigurationError):
            with self.preprocessor(runner, max_workspace_bytes=100).prepare(
                self.audio, **self.prepare_kwargs()
            ):
                pass
        self.assertEqual(runner.invocations, [])
        with self.assertRaises(RNNoiseConfigurationError):
            with self.preprocessor(runner, max_output_bytes=100).prepare(
                self.audio, **self.prepare_kwargs()
            ):
                pass
        self.assertEqual(runner.invocations, [])

    def test_process_wide_gate_serializes_two_instances_and_releases_after_error(self):
        runner = FakeNativeRunner()
        processors = (self.preprocessor(runner), self.preprocessor(runner))
        failures = []

        def work(processor):
            try:
                with processor.prepare(self.audio, **self.prepare_kwargs()):
                    time.sleep(0.01)
            except Exception as exc:  # pragma: no cover - diagnostic collection
                failures.append(exc)

        threads = [threading.Thread(target=work, args=(processor,)) for processor in processors]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(runner.maximum_active, 1)

        with self.assertRaises(RNNoiseTimebaseError):
            with self.preprocessor(FakeNativeRunner(wrong_stage="DOWNSAMPLE_16K")).prepare(
                self.audio, **self.prepare_kwargs()
            ):
                pass
        with self.preprocessor().prepare(self.audio, **self.prepare_kwargs()) as prepared:
            self.assertTrue(prepared.enhanced)

    def test_process_wide_gate_has_a_bounded_queue_timeout(self):
        entered = threading.Event()
        release = threading.Event()
        failures = []

        def holder():
            try:
                with self.preprocessor().prepare(self.audio, **self.prepare_kwargs()):
                    entered.set()
                    release.wait(2.0)
            except Exception as exc:  # pragma: no cover - diagnostic collection
                failures.append(exc)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(entered.wait(2.0))
        try:
            with self.assertRaises(RNNoiseBusyError):
                with self.preprocessor(queue_timeout_seconds=0.01).prepare(
                    self.audio, **self.prepare_kwargs()
                ):
                    pass
        finally:
            release.set()
            thread.join()
        self.assertEqual(failures, [])

    def test_subprocess_runner_uses_argv_no_shell_and_discards_logs(self):
        output = self.work_root / "out.raw"
        invocation = NativeInvocation(
            "RNNOISE_48K",
            (str(self.rnnoise.resolve()), "in.raw", "out.raw"),
            self.work_root.resolve(),
            output,
            0,
            1.0,
        )
        with mock.patch("sddiar.rnnoise_experimental.subprocess.run") as run:
            run.return_value.returncode = 0
            outcome = SubprocessArgvRunner().run(invocation)
        self.assertEqual(outcome.returncode, 0)
        args, kwargs = run.call_args
        self.assertIsInstance(args[0], list)
        self.assertIs(kwargs["shell"], False)
        self.assertIs(kwargs["stdin"], __import__("subprocess").DEVNULL)
        self.assertIs(kwargs["stdout"], __import__("subprocess").DEVNULL)
        self.assertIs(kwargs["stderr"], __import__("subprocess").DEVNULL)
        self.assertNotIn("PATH", kwargs["env"])


if __name__ == "__main__":
    unittest.main()
