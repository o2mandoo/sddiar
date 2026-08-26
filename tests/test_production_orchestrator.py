from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sddiar.production_orchestrator as orchestrator_module

from sddiar.calibration import (
    CalibrationProfileVerifier,
    DigestCalibrationSignatureVerifier,
    canonical_calibration_bytes,
)
from sddiar.contracts import (
    DiarizationSpan,
    ParticipantBinding,
    Word,
    WordProvenance,
    WordTimeline,
)
from sddiar.errors import ContractValidationError, ModelHashMismatch, OfflinePolicyViolation
from sddiar.model_pack import (
    DigestSignatureVerifier,
    ModelPackVerifier,
    canonical_manifest_bytes,
)
from sddiar.media import WavPcmAccessor as RealWavPcmAccessor
from sddiar.production_orchestrator import (
    DiarizationEnvelope,
    HashVerifiedLocalTranscriptBackend,
    Pcm16CanonicalAdapter,
    ProductionOrchestrationError,
    ProductionOrchestrator,
    ProductionOrchestratorConfig,
    ProductionQualityEvidence,
    canonical_production_config_hash,
    verify_local_stt_identity,
)


class _ReleasePackVerifier(DigestSignatureVerifier):
    trust_level = "RELEASE"


class _ReleaseCalibrationVerifier(DigestCalibrationSignatureVerifier):
    trust_level = "RELEASE"


def _write_wav(path: Path, *, rate: int, samples: tuple[int, ...]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _timeline(*words: Word) -> WordTimeline:
    return WordTimeline(tuple(words), {
        word.word_id: WordProvenance(word.word_id) for word in words
    })


class _FakeDiarizer:
    def __init__(self, *, split: bool = True, duration_delta: int = 0):
        self.split = split
        self.duration_delta = duration_delta
        self.calls = 0
        self.seen = []

    def process(self, path):
        self.calls += 1
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            prefix = handle.readframes(min(frames, 8))
        duration = (frames * 1_000_000 + rate // 2) // rate
        self.seen.append({
            "path": str(path),
            "exists": Path(path).exists(),
            "rate": rate,
            "frames": frames,
            "channels": channels,
            "width": width,
            "prefix": prefix,
        })
        if self.split and duration >= 2:
            middle = duration // 2
            spans = (
                DiarizationSpan("span-00", 0, middle, "SPEAKER_00", "ASSIGNED"),
                DiarizationSpan("span-01", middle, duration, "SPEAKER_01", "ASSIGNED"),
            )
            decision = "H2_CONFIRMED"
        else:
            spans = (
                DiarizationSpan("span-00", 0, duration, "SPEAKER_00", "ASSIGNED"),
            )
            decision = "H1_CONFIRMED"
        return SimpleNamespace(
            spans=spans,
            duration_us=duration + self.duration_delta,
            sample_rate_hz=rate,
            decision=decision,
            decision_reasons=(),
            metrics={"rtf": 0.01, "peak_rss_mb": 12.5},
        )


class _FakeLocalSttEngine:
    def __init__(self, timeline: WordTimeline):
        self.timeline = timeline
        self.calls = 0
        self.seen = []

    def transcribe(self, canonical_audio_path, source):
        self.calls += 1
        with wave.open(str(canonical_audio_path), "rb") as handle:
            self.seen.append((
                handle.getframerate(), handle.getnframes(), source.duration_us,
                "offline-stt-test",
            ))
        return self.timeline


class _FailingDiarizer:
    def process(self, path):
        raise RuntimeError("SECRET_DIARIZATION_EXCEPTION /private/input.wav")


class _ConflictingStatusDiarizer:
    def process(self, path):
        with wave.open(str(path), "rb") as handle:
            duration = (handle.getnframes() * 1_000_000) // handle.getframerate()
        return SimpleNamespace(
            spans=(DiarizationSpan(
                "span-conflict", 0, duration, "SPEAKER_00", "UNKNOWN_SHORT"
            ),),
            duration_us=duration,
            sample_rate_hz=16_000,
            decision="H1_CONFIRMED",
            decision_reasons=(),
            metrics={},
        )


class _InterveningSpeakerDiarizer:
    def process(self, path):
        with wave.open(str(path), "rb") as handle:
            duration = (handle.getnframes() * 1_000_000) // handle.getframerate()
        return SimpleNamespace(
            spans=(
                DiarizationSpan("span-a", 0, 400_000, "SPEAKER_00", "ASSIGNED"),
                DiarizationSpan("span-b", 400_000, 800_000, "SPEAKER_01", "ASSIGNED"),
                DiarizationSpan("span-c", 800_000, duration, "SPEAKER_00", "ASSIGNED"),
            ),
            duration_us=duration,
            sample_rate_hz=16_000,
            decision="H2_CONFIRMED",
            decision_reasons=(),
            metrics={},
        )


class _FailingLocalSttEngine:
    def transcribe(self, canonical_audio_path, source):
        raise RuntimeError("SECRET_STT_EXCEPTION /private/transcript.txt")


class _MutatingLocalSttEngine:
    def __init__(self, timeline):
        self.timeline = timeline

    def transcribe(self, canonical_audio_path, source):
        path = Path(canonical_audio_path)
        with wave.open(str(path), "rb") as handle:
            rate, frames = handle.getframerate(), handle.getnframes()
        path.chmod(0o600)
        _write_wav(path, rate=rate, samples=(0,) * frames)
        return self.timeline


class _MutatingSttArtifactEngine:
    def __init__(self, timeline):
        self.timeline = timeline
        self.model_path = None

    def transcribe(self, canonical_audio_path, source):
        self.model_path.write_bytes(b"mutated during inference")
        return self.timeline


class _ConcurrentLocalSttEngine:
    def __init__(self, timeline):
        self.timeline = timeline
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def transcribe(self, canonical_audio_path, source):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        return self.timeline


class ProductionOrchestratorTests(unittest.TestCase):
    pack_key = b"release-pack-key"
    calibration_key = b"release-calibration-key"
    config_hash = canonical_production_config_hash(
        ProductionOrchestratorConfig(), backend_kind="LOCAL_ONNX_DEFAULT"
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def audio(self, *, rate: int = 16_000, frames: int | None = None, name: str = "private customer audio.wav") -> Path:
        count = frames or rate * 2
        path = self.root / name
        samples = tuple(((index * 7919) % 20_000) - 10_000 for index in range(count))
        _write_wav(path, rate=rate, samples=samples)
        return path

    def release_pack(self):
        pack_root = self.root / "pack"
        models = pack_root / "models"
        models.mkdir(parents=True)
        vad = models / "vad.onnx"
        embed = models / "embed.onnx"
        vad.write_bytes(b"release-vad")
        embed.write_bytes(b"release-embedding")
        files = []
        for file_id, role, path in (
            ("vad-model", "vad", vad),
            ("embed-model", "speaker_embedding", embed),
        ):
            raw = path.read_bytes()
            files.append({
                "file_id": file_id,
                "role": role,
                "relative_path": str(path.relative_to(pack_root)),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        manifest = {
            "schema_version": "1.0",
            "pack_id": "release-pack-1",
            "pack_version": "1",
            "production_approved": True,
            "integrity": {"signer_key_id": "release-pack-key"},
            "runtime_compatibility": {
                "onnxruntime": {"exact_build_id": "ort-release", "exact_version": "1.29.0"},
                "allowed_execution_providers": ["CPUExecutionProvider"],
                "target_matrix": [{"os": "linux", "arch": "x86_64", "python_abi": "cp311"}],
            },
            "files": files,
        }
        signature = hashlib.sha256(
            self.pack_key + canonical_manifest_bytes(manifest)
        ).digest()
        manifest["integrity"]["signature"] = base64.b64encode(signature).decode("ascii")
        runtime = {
            "exact_build_id": "ort-release",
            "exact_version": "1.29.0",
            "execution_provider": "CPUExecutionProvider",
            "os": "linux",
            "arch": "x86_64",
            "python_abi": "cp311",
        }
        pack = ModelPackVerifier(
            pack_root,
            runtime=runtime,
            signature_verifier=_ReleasePackVerifier(self.pack_key),
            require_release_trust=True,
        ).verify(manifest)
        return pack, vad, embed

    def release_calibration(self, pack, *, source_rate: int = 16_000):
        model_hashes = {
            artifact.file_id: artifact.sha256 for artifact in pack.artifacts
            if artifact.path.suffix == ".onnx"
        }
        profile = {
            "schema_version": "1",
            "profile_id": f"cal-{source_rate}",
            "calibration_version": "1",
            "model_hashes": model_hashes,
            "source_sample_rates": [source_rate],
            "thresholds": {"unknown_ratio_warn": 0.2},
            "dataset_manifest_hash": "b" * 64,
            "scorer_hash": "c" * 64,
            "config_hash": self.config_hash,
            "approver": "quality-approval-1",
            "provenance": {
                "annotation_schema_version": "1",
                "created_at": "2026-08-26T00:00:00Z",
                "model_pack_id": pack.pack_id,
                "pipeline_version": "production-orchestrator-v1",
                "safety_constraints": ["no unsafe speaker assignment"],
                "selection_objective": "maximize safe coverage",
            },
            "signer_key_id": "release-calibration-key",
        }
        signature = hashlib.sha256(
            self.calibration_key + canonical_calibration_bytes(profile)
        ).digest()
        profile["signature"] = base64.b64encode(signature).decode("ascii")
        return CalibrationProfileVerifier(
            _ReleaseCalibrationVerifier(self.calibration_key)
        ).verify(
            profile,
            model_hashes=model_hashes,
            source_sample_rate=source_rate,
            config_hash=self.config_hash,
        )

    def local_stt_backend(self, timeline: WordTimeline, implementation=None):
        engine_path = self.root / "offline-stt-engine.whl"
        model_path = self.root / "offline-stt-model.bin"
        engine_path.write_bytes(b"offline-stt-engine")
        model_path.write_bytes(b"offline-stt-model")
        identity = verify_local_stt_identity(
            backend_id="offline-stt-test",
            backend_version="1.0.0",
            engine_path=engine_path,
            engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(),
            model_path=model_path,
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        )
        implementation = implementation or _FakeLocalSttEngine(timeline)
        if hasattr(implementation, "model_path"):
            implementation.model_path = model_path
        return HashVerifiedLocalTranscriptBackend(identity, implementation), implementation, model_path

    @staticmethod
    def complete_quality():
        return ProductionQualityEvidence(
            metrics={"assigned_time_accuracy": 0.99},
            threshold_relations={"unknown_ratio_warn": "PASS"},
            all_required_metrics_evaluated=True,
            osd_coverage="EVALUATED",
        )

    def test_production_orchestrator_rejects_more_than_one_thread(self):
        with self.assertRaisesRegex(ContractValidationError, "exactly one thread"):
            ProductionOrchestratorConfig(threads=2)

    def test_production_contracts_reject_ambiguous_scalar_and_collection_types(self):
        for kwargs in (
            {"config_hash": "A" * 64},
            {"pipeline_version": "pipeline\x00secret"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractValidationError):
                ProductionOrchestratorConfig(**kwargs)
        with self.assertRaisesRegex(ContractValidationError, "flags must be boolean"):
            ProductionQualityEvidence(all_required_metrics_evaluated=1)
        with self.assertRaisesRegex(ContractValidationError, "collections must be tuples"):
            ProductionQualityEvidence(review_reasons="Q_BAD")

        audio = self.audio()
        malformed = WordTimeline(
            (Word("word-1", 100_000, 200_000, "text"),),
            {"word-1": WordProvenance("word-1", source_chunk_ids="chunk-1")},
        )
        with self.assertRaisesRegex(ContractValidationError, "chunk IDs must be tuples"):
            ProductionOrchestrator(diarizer=_FakeDiarizer()).process(
                audio, supplied_word_timeline=malformed
            )

    def test_one_orchestrator_serializes_local_stt_jobs(self):
        audio = self.audio()
        timeline = _timeline(Word("word-1", 100_000, 200_000, "text"))
        engine = _ConcurrentLocalSttEngine(timeline)
        backend, _, _ = self.local_stt_backend(timeline, engine)
        orchestrator = ProductionOrchestrator(
            diarizer=_FakeDiarizer(), transcript_backend=backend
        )
        errors = []

        def run():
            try:
                orchestrator.process(audio)
            except Exception as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)

        workers = [threading.Thread(target=run) for _ in range(3)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(errors, [])
        self.assertEqual(engine.maximum, 1)

    def test_injected_16k_maps_whole_words_and_missing_authority_is_neutral_review(self):
        audio = self.audio()
        words = _timeline(
            Word("word-1", 100_000, 500_000, "SECRET_TRANSCRIPT"),
            Word("word-2", 900_000, 1_100_000, "boundary"),
        )
        fake = _FakeDiarizer()
        orchestrator = ProductionOrchestrator(diarizer=fake)
        result = orchestrator.process(
            audio,
            supplied_word_timeline=words,
            quality_evidence=self.complete_quality(),
        )
        self.assertEqual(result.source.native_sample_rate_hz, 16_000)
        self.assertNotEqual(fake.seen[0]["path"], str(audio))
        self.assertFalse(Path(fake.seen[0]["path"]).exists())
        self.assertEqual(
            [(word.speaker_id, word.attribution_status) for word in result.attributed_words],
            [("SPEAKER_00", "ASSIGNED"), ("UNKNOWN", "UNKNOWN_BOUNDARY")],
        )
        self.assertEqual((result.quality.status, result.quality.summary_mode),
                         ("REVIEW_REQUIRED", "SPEAKER_NEUTRAL"))
        self.assertIn("Q_RELEASE_MODEL_AUTHORITY_MISSING", result.quality.reason_codes)
        self.assertIn("Q_CALIBRATION_MISSING", result.quality.reason_codes)
        self.assertEqual(result.participant_bindings, ())
        self.assertEqual(result.speaker_aware_transcript, ())
        self.assertTrue(all(word.speaker_id in {"UNKNOWN", "OVERLAP"}
                            for word in result.speaker_neutral_transcript))

        payload = json.loads(orchestrator.process_json(
            audio,
            supplied_word_timeline=words,
            quality_evidence=self.complete_quality(),
        ))
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(str(audio), serialized)
        self.assertIn("SECRET_TRANSCRIPT", serialized)
        for key in ("run", "quality", "extensions"):
            self.assertNotIn("SECRET_TRANSCRIPT", json.dumps(payload[key], ensure_ascii=False))
        self.assertFalse(payload["extensions"]["orchestrator"]["raw_transcript_in_diagnostics"])

    def test_8k_is_bounded_canonical_16k_with_exact_source_time(self):
        audio = self.audio(rate=8_000, frames=8_003, name="private-8k.wav")
        source_duration = (8_003 * 1_000_000 + 4_000) // 8_000
        fake = _FakeDiarizer(split=False)
        word = Word("word-8k", 0, source_duration, "eight k")
        backend, implementation, _ = self.local_stt_backend(_timeline(word))
        result = ProductionOrchestrator(
            diarizer=fake, transcript_backend=backend
        ).process(audio)
        self.assertEqual(fake.seen[0]["rate"], 16_000)
        self.assertEqual(fake.seen[0]["frames"], 16_006)
        self.assertEqual((fake.seen[0]["channels"], fake.seen[0]["width"]), (1, 2))
        self.assertTrue(fake.seen[0]["exists"])
        self.assertFalse(Path(fake.seen[0]["path"]).exists())
        self.assertEqual(result.source.native_sample_rate_hz, 8_000)
        self.assertEqual(result.source.duration_us, source_duration)
        self.assertEqual(result.source.timebase.duration_us, source_duration)
        self.assertEqual(
            implementation.seen[0],
            (16_000, 16_006, source_duration, "offline-stt-test"),
        )
        self.assertEqual((result.attributed_words[0].start_us, result.attributed_words[0].end_us),
                         (0, source_duration))
        self.assertTrue(result.extensions["orchestrator"]["source_resampled_to_16k"])

        with Pcm16CanonicalAdapter(chunk_frames=17).prepare(audio) as canonical:
            segment = canonical.time_warp[0]
            self.assertEqual(segment.source_us(canonical.canonical_frame_count), source_duration)
            self.assertEqual(segment.source_us(0), 0)
            digest_small = hashlib.sha256(canonical.canonical_path.read_bytes()).hexdigest()
        with Pcm16CanonicalAdapter(chunk_frames=4_096).prepare(audio) as canonical:
            digest_large = hashlib.sha256(canonical.canonical_path.read_bytes()).hexdigest()
        self.assertEqual(digest_small, digest_large)

    def test_participant_mapping_is_explicit_only_and_collisions_fail_closed(self):
        audio = self.audio()
        words = _timeline(Word("word-role", 100_000, 200_000, "저는 상담원입니다"))
        fake = _FakeDiarizer()
        orchestrator = ProductionOrchestrator(diarizer=fake)
        without = orchestrator.process(audio, supplied_word_timeline=words)
        self.assertEqual(without.participant_bindings, ())
        self.assertTrue(all(turn.speaker_id.startswith("SPEAKER_") for turn in without.speaker_turns))

        explicit = (
            ParticipantBinding("SPEAKER_00", "participant-00", "상담원", "EXTERNAL_AUTHORITATIVE_METADATA", 1.0, ("evidence-00",)),
            ParticipantBinding("SPEAKER_01", "participant-01", "고객", "EXTERNAL_AUTHORITATIVE_METADATA", 1.0, ("evidence-01",)),
        )
        mapped = orchestrator.process(audio, supplied_word_timeline=words,
                                      participant_bindings=explicit)
        self.assertEqual(mapped.participant_bindings, explicit)
        self.assertTrue(all(turn.speaker_id in {"SPEAKER_00", "UNKNOWN"}
                            for turn in mapped.speaker_turns))

        invalid_unknown = ParticipantBinding(
            "UNKNOWN", "participant-02", None, "EXTERNAL_AUTHORITATIVE_METADATA", 1.0,
            ("evidence-02",),
        )
        with self.assertRaisesRegex(ContractValidationError, "UNKNOWN"):
            orchestrator.process(audio, supplied_word_timeline=words,
                                 participant_bindings=(invalid_unknown,))
        collision = (
            explicit[0],
            ParticipantBinding("SPEAKER_01", "PARTICIPANT-00", "다른 역할", "EXTERNAL_AUTHORITATIVE_METADATA", 1.0, ("evidence-03",)),
        )
        with self.assertRaisesRegex(ContractValidationError, "collision"):
            orchestrator.process(audio, supplied_word_timeline=words,
                                 participant_bindings=collision)

    def test_release_pack_and_calibration_connect_but_raw_evidence_cannot_pass(self):
        audio = self.audio()
        pack, _, _ = self.release_pack()
        calibration = self.release_calibration(pack)
        fake = _FakeDiarizer()
        with mock.patch(
            "sddiar.production_orchestrator.LocalOnnxDiarizer", return_value=fake
        ) as constructor:
            orchestrator = ProductionOrchestrator(
                model_pack=pack,
                calibration=calibration,
                config=ProductionOrchestratorConfig(config_hash=self.config_hash),
            )
        result = orchestrator.process(
            audio,
            supplied_word_timeline=_timeline(Word("word-1", 100_000, 500_000, "hello")),
            quality_evidence=self.complete_quality(),
        )
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual((result.quality.status, result.quality.summary_mode),
                         ("REVIEW_REQUIRED", "SPEAKER_NEUTRAL"))
        self.assertIn("Q_RELEASE_EVIDENCE_UNVERIFIED", result.quality.reason_codes)
        self.assertEqual(result.quality.calibration_profile_id, "cal-16000")
        self.assertEqual(result.speaker_aware_transcript, ())
        self.assertEqual(result.run.model_pack_id, pack.pack_id)
        self.assertEqual(result.run.model_hashes, calibration.model_hashes)
        self.assertTrue(result.extensions["orchestrator"]["release_model_authority"])
        with self.assertRaises(AttributeError):
            orchestrator.diarizer = _FakeDiarizer()
        with self.assertRaises(AttributeError):
            orchestrator.quality_gate = object()

    def test_source_rate_mismatch_does_not_claim_calibration_authority(self):
        audio = self.audio(rate=8_000, frames=8_000)
        pack, _, _ = self.release_pack()
        calibration = self.release_calibration(pack, source_rate=16_000)
        with mock.patch(
            "sddiar.production_orchestrator.LocalOnnxDiarizer",
            return_value=_FakeDiarizer(split=False),
        ):
            orchestrator = ProductionOrchestrator(
                model_pack=pack,
                calibration=calibration,
                config=ProductionOrchestratorConfig(config_hash=self.config_hash),
            )
        result = orchestrator.process(
            audio,
            supplied_word_timeline=_timeline(
                Word("word-1", 100_000, 200_000, "text")
            ),
            quality_evidence=self.complete_quality(),
        )
        self.assertIsNone(result.run.calibration_profile_id)
        self.assertFalse(
            result.extensions["orchestrator"]["release_calibration_authority"]
        )
        self.assertIn(
            "Q_CALIBRATION_SOURCE_RATE_MISMATCH", result.quality.reason_codes
        )

    def test_result_identity_binds_transcript_and_explicit_participant_inputs(self):
        audio = self.audio()
        orchestrator = ProductionOrchestrator(diarizer=_FakeDiarizer())
        first_words = _timeline(
            Word("word-1", 100_000, 200_000, "first transcript")
        )
        second_words = _timeline(
            Word("word-1", 100_000, 200_000, "changed transcript")
        )
        first = orchestrator.process(audio, supplied_word_timeline=first_words)
        repeat = orchestrator.process(audio, supplied_word_timeline=first_words)
        second = orchestrator.process(audio, supplied_word_timeline=second_words)
        self.assertEqual((first.result_id, first.run.run_id),
                         (repeat.result_id, repeat.run.run_id))
        self.assertNotEqual(first.result_id, second.result_id)
        self.assertNotEqual(first.run.run_id, second.run.run_id)

        binding = ParticipantBinding(
            "SPEAKER_00", "participant-00", "상담원",
            "EXTERNAL_AUTHORITATIVE_METADATA", 1.0, ("evidence-00",),
        )
        bound = orchestrator.process(
            audio, supplied_word_timeline=first_words,
            participant_bindings=(binding,),
        )
        self.assertNotEqual(first.result_id, bound.result_id)

    def test_release_artifact_mutation_and_injected_timebase_mismatch_fail(self):
        audio = self.audio()
        pack, vad, _ = self.release_pack()
        fake = _FakeDiarizer()
        with mock.patch(
            "sddiar.production_orchestrator.LocalOnnxDiarizer", return_value=fake
        ):
            orchestrator = ProductionOrchestrator(model_pack=pack)
        vad.write_bytes(b"tampered after verification")
        with self.assertRaises(ModelHashMismatch):
            orchestrator.process(audio)

        mismatched = ProductionOrchestrator(
            diarizer=_FakeDiarizer(duration_delta=1)
        )
        with self.assertRaisesRegex(ContractValidationError, "timebase"):
            mismatched.process(audio)

        with self.assertRaisesRegex(ContractValidationError, "status conflict"):
            ProductionOrchestrator(
                diarizer=_ConflictingStatusDiarizer()
            ).process(
                audio,
                supplied_word_timeline=_timeline(
                    Word("word-1", 100_000, 200_000, "text")
                ),
            )

    def test_turns_do_not_merge_across_intervening_speaker_span(self):
        audio = self.audio()
        result = ProductionOrchestrator(
            diarizer=_InterveningSpeakerDiarizer()
        ).process(
            audio,
            supplied_word_timeline=_timeline(
                Word("word-1", 100_000, 200_000, "first"),
                Word("word-2", 900_000, 1_000_000, "second"),
            ),
        )
        self.assertEqual(
            [(turn.start_us, turn.end_us, turn.speaker_id) for turn in result.speaker_turns],
            [(100_000, 200_000, "SPEAKER_00"),
             (900_000, 1_000_000, "SPEAKER_00")],
        )

    def test_enhancement_hook_is_default_off_and_cannot_self_authorize(self):
        audio = self.audio()
        fake = _FakeDiarizer()

        class Hook:
            release_authorized = True

            def __init__(self):
                self.calls = 0

            def enhance(self, canonical, result):
                self.calls += 1
                return DiarizationEnvelope(
                    result.spans, result.duration_us, result.sample_rate_hz,
                    result.decision, result.decision_reasons, result.rtf,
                    result.peak_rss_mb,
                )

        hook = Hook()
        disabled = ProductionOrchestrator(diarizer=fake, enhancement_hook=hook)
        disabled.process(audio)
        self.assertEqual(hook.calls, 0)

        enabled = ProductionOrchestrator(
            diarizer=fake,
            enhancement_hook=hook,
            config=ProductionOrchestratorConfig(enable_enhancement=True),
        )
        result = enabled.process(audio)
        self.assertEqual(hook.calls, 1)
        self.assertEqual(result.quality.status, "REVIEW_REQUIRED")
        self.assertIn("Q_ENHANCEMENT_UNVERIFIED", result.quality.reason_codes)

    def test_hash_verified_local_stt_drives_end_to_end_transcript_and_attribution(self):
        audio = self.audio()
        words = _timeline(
            Word("stt-word-1", 100_000, 400_000, "LOCAL_STT_SECRET_TEXT"),
            Word("stt-word-2", 1_200_000, 1_500_000, "second"),
        )
        backend, implementation, _ = self.local_stt_backend(words)
        orchestrator = ProductionOrchestrator(
            diarizer=_FakeDiarizer(), transcript_backend=backend
        )
        result = orchestrator.process(audio)
        self.assertEqual(implementation.calls, 1)
        self.assertEqual(implementation.seen[0][:2], (16_000, 32_000))
        self.assertEqual(
            [word.speaker_id for word in result.attributed_words],
            ["SPEAKER_00", "SPEAKER_01"],
        )
        transcript_receipt = result.extensions["orchestrator"]["stt_backend"]
        self.assertEqual(transcript_receipt["kind"], "HASH_BOUND_INJECTED_LOCAL_STT")
        self.assertEqual(
            transcript_receipt["implementation_binding"],
            "INJECTED_CONTRACT_REQUIRES_RELEASE_AUDIT",
        )
        self.assertEqual(transcript_receipt["backend_id"], "offline-stt-test")
        self.assertEqual(len(transcript_receipt["engine_sha256"]), 64)
        self.assertEqual(len(transcript_receipt["model_sha256"]), 64)

        payload = json.loads(orchestrator.process_json(audio))
        self.assertIn("LOCAL_STT_SECRET_TEXT", json.dumps(payload, ensure_ascii=False))
        for key in ("run", "quality", "extensions"):
            self.assertNotIn(
                "LOCAL_STT_SECRET_TEXT", json.dumps(payload[key], ensure_ascii=False)
            )
        self.assertNotIn(str(audio), json.dumps(payload, ensure_ascii=False))

    def test_local_stt_model_hash_is_rechecked_before_inference(self):
        audio = self.audio()
        backend, implementation, model_path = self.local_stt_backend(
            _timeline(Word("stt-word-1", 100_000, 200_000, "text"))
        )
        model_path.write_bytes(b"tampered-local-stt-model")
        orchestrator = ProductionOrchestrator(
            diarizer=_FakeDiarizer(), transcript_backend=backend
        )
        with self.assertRaises(ModelHashMismatch):
            orchestrator.process(audio)
        self.assertEqual(implementation.calls, 0)

        timeline = _timeline(Word("stt-word-2", 100_000, 200_000, "text"))
        runtime_backend, _, _ = self.local_stt_backend(
            timeline, _MutatingSttArtifactEngine(timeline)
        )
        with self.assertRaises(ModelHashMismatch):
            ProductionOrchestrator(
                diarizer=_FakeDiarizer(), transcript_backend=runtime_backend
            ).process(audio)
        with self.assertRaises(AttributeError):
            runtime_backend.implementation = _FakeLocalSttEngine(timeline)

    def test_stt_cannot_mutate_canonical_audio_seen_by_diarizer_or_source(self):
        audio = self.audio()
        original_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        mutation_timeline = _timeline(
            Word("stt-word-1", 100_000, 200_000, "text")
        )
        backend, _, _ = self.local_stt_backend(
            mutation_timeline, _MutatingLocalSttEngine(mutation_timeline)
        )
        diarizer = _FakeDiarizer()
        with self.assertRaisesRegex(ContractValidationError, "canonical audio changed"):
            ProductionOrchestrator(
                diarizer=diarizer, transcript_backend=backend
            ).process(audio)
        self.assertEqual(diarizer.calls, 0)
        self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), original_hash)

    def test_source_path_replacement_cannot_split_hash_from_timebase_snapshot(self):
        audio = self.audio(rate=16_000, frames=100, name="source-race.wav")
        replacement = self.audio(
            rate=8_000, frames=100, name="replacement-race.wav"
        )
        original_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        replacement_hash = hashlib.sha256(replacement.read_bytes()).hexdigest()
        replaced = False

        def accessor(path):
            nonlocal replaced
            if not replaced:
                replaced = True
                replacement.replace(audio)
            return RealWavPcmAccessor(path)

        diarizer = _FakeDiarizer(split=False)
        with mock.patch(
            "sddiar.production_orchestrator.WavPcmAccessor",
            side_effect=accessor,
        ):
            result = ProductionOrchestrator(diarizer=diarizer).process(audio)
        self.assertTrue(replaced)
        self.assertEqual(result.source.audio_sha256, original_hash)
        self.assertEqual(result.source.native_sample_rate_hz, 16_000)
        self.assertEqual(result.source.duration_us, 6_250)
        self.assertEqual(diarizer.seen[0]["rate"], 16_000)
        self.assertEqual(hashlib.sha256(audio.read_bytes()).hexdigest(), replacement_hash)

    def test_source_snapshot_rejects_symlink_and_requests_windows_binary_mode(self):
        audio = self.audio(name="source-real.wav")
        link = self.root / "source-link.wav"
        link.symlink_to(audio.name)
        with self.assertRaisesRegex(ContractValidationError, "non-symlink"):
            ProductionOrchestrator(diarizer=_FakeDiarizer()).process(link)

        sentinel = 1 << 29
        captured = []
        real_open = os.open

        def binary_open(path, flags, *args, **kwargs):
            captured.append(flags)
            return real_open(path, flags & ~sentinel, *args, **kwargs)

        with mock.patch.object(
            orchestrator_module.os, "O_BINARY", sentinel, create=True
        ), mock.patch.object(
            orchestrator_module.os, "open", side_effect=binary_open
        ):
            with Pcm16CanonicalAdapter().prepare(audio):
                pass
        self.assertTrue(any(flags & sentinel for flags in captured))

    def test_local_stt_source_time_and_offline_inputs_fail_closed(self):
        audio = self.audio()
        duration = 2_000_000
        backend, _, _ = self.local_stt_backend(
            _timeline(Word("late-word", duration, duration + 1, "late"))
        )
        with self.assertRaisesRegex(ContractValidationError, "source duration"):
            ProductionOrchestrator(
                diarizer=_FakeDiarizer(), transcript_backend=backend
            ).process(audio)

        with self.assertRaises(OfflinePolicyViolation):
            ProductionOrchestrator(diarizer=_FakeDiarizer()).process(
                "https://example.invalid/audio.wav"
            )
        with self.assertRaises(OfflinePolicyViolation):
            verify_local_stt_identity(
                backend_id="offline-stt-test",
                backend_version="1.0.0",
                engine_path="https://example.invalid/engine.whl",
                engine_sha256="0" * 64,
                model_path=self.root / "missing-model.bin",
                model_sha256="0" * 64,
            )

        safe_backend, _, _ = self.local_stt_backend(
            _timeline(Word("stt-word-1", 0, 1, "x")),
            _FailingLocalSttEngine(),
        )
        with self.assertRaises(ProductionOrchestrationError) as caught:
            ProductionOrchestrator(
                diarizer=_FakeDiarizer(), transcript_backend=safe_backend
            ).process(audio)
        self.assertEqual(str(caught.exception), "local STT backend failed")
        self.assertNotIn("SECRET_STT_EXCEPTION", str(caught.exception))

        secret_path = self.audio(name="SECRET_PATIENT_PATH.wav")
        with mock.patch(
            "sddiar.production_orchestrator.WavPcmAccessor",
            side_effect=PermissionError(str(secret_path)),
        ), self.assertRaises(ProductionOrchestrationError) as audio_error:
            ProductionOrchestrator(diarizer=_FakeDiarizer()).process(secret_path)
        self.assertEqual(str(audio_error.exception), "local audio intake failed")
        self.assertNotIn("SECRET_PATIENT_PATH", str(audio_error.exception))

    def test_diarization_runtime_failure_preserves_neutral_local_stt(self):
        audio = self.audio()
        words = _timeline(
            Word("stt-word-1", 100_000, 400_000, "PRESERVED_STT_TEXT")
        )
        backend, implementation, _ = self.local_stt_backend(words)
        orchestrator = ProductionOrchestrator(
            diarizer=_FailingDiarizer(), transcript_backend=backend
        )
        # Even explicit participant metadata is discarded when no speaker
        # topology survived; it must not prevent neutral transcript delivery.
        binding = ParticipantBinding(
            "SPEAKER_00", "participant-00", "상담원",
            "EXTERNAL_AUTHORITATIVE_METADATA", 1.0, ("evidence-00",),
        )
        result = orchestrator.process(audio, participant_bindings=(binding,))
        self.assertEqual(implementation.calls, 1)
        self.assertEqual(result.diarization_spans, ())
        self.assertEqual(result.participant_bindings, ())
        self.assertEqual(result.attributed_words[0].speaker_id, "UNKNOWN")
        self.assertEqual(result.speaker_neutral_transcript[0].text, "PRESERVED_STT_TEXT")
        self.assertEqual(result.speaker_aware_transcript, ())
        self.assertEqual((result.quality.status, result.quality.summary_mode),
                         ("REVIEW_REQUIRED", "SPEAKER_NEUTRAL"))
        self.assertIn("Q_DIARIZATION_FAILED", result.quality.reason_codes)
        self.assertEqual(
            result.extensions["orchestrator"]["diarization_status"],
            "FAILED_NEUTRAL_FALLBACK",
        )
        serialized = orchestrator.serializer.serialize(result).decode("utf-8")
        self.assertIn("PRESERVED_STT_TEXT", serialized)
        self.assertNotIn("SECRET_DIARIZATION_EXCEPTION", serialized)
        self.assertNotIn("/private/input.wav", serialized)

    def test_release_diarizer_construction_failure_also_preserves_local_stt(self):
        audio = self.audio()
        pack, _, _ = self.release_pack()
        backend, implementation, _ = self.local_stt_backend(
            _timeline(Word("stt-word-1", 100_000, 400_000, "STT_SURVIVES_ORT_FAILURE"))
        )
        with mock.patch(
            "sddiar.production_orchestrator.LocalOnnxDiarizer",
            side_effect=RuntimeError("ORT failed with /private/model/path"),
        ):
            orchestrator = ProductionOrchestrator(
                model_pack=pack, transcript_backend=backend
            )
        result = orchestrator.process(audio)
        self.assertEqual(implementation.calls, 1)
        self.assertEqual(result.quality.status, "REVIEW_REQUIRED")
        self.assertEqual(result.speaker_neutral_transcript[0].text,
                         "STT_SURVIVES_ORT_FAILURE")
        self.assertEqual(result.diarization_spans, ())
        serialized = orchestrator.serializer.serialize(result).decode("utf-8")
        self.assertNotIn("/private/model/path", serialized)


if __name__ == "__main__":
    unittest.main()
