import hashlib
import tempfile
import unittest
from pathlib import Path

from sddiar.contracts import EmbeddingRegion
from sddiar.errors import ContractValidationError, ModelRuntimeIncompatible
from sddiar.model_pack import VerifiedArtifact
from sddiar.wespeaker_runtime import (
    DEVELOPMENT_APPROXIMATION_TAG,
    STRICT_NATIVE_FBANK_TAG,
    WeSpeakerCpuEmbeddingBackend,
    WeSpeakerFeatureConfig,
    kaldi_native_fbank_features,
    wespeaker_logmel_features_approx,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime
    np = None


class _Meta:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _Session:
    def __init__(self, providers=("CPUExecutionProvider",)):
        self.providers = providers
        self.calls = []

    def get_providers(self):
        return list(self.providers)

    def get_inputs(self):
        return [_Meta("feats", ["batch", "frames", 80])]

    def get_outputs(self):
        return [_Meta("embs", ["batch", 3])]

    def run(self, outputs, feeds):
        x = feeds["feats"]
        self.calls.append(x)
        return [np.tile(np.array([[3.0, 0.0, 4.0]], dtype=np.float32), (x.shape[0], 1))]


class _WindowSession(_Session):
    def run(self, outputs, feeds):
        x = feeds["feats"]
        self.calls.append(x)
        vectors = np.zeros((x.shape[0], 3), dtype=np.float32)
        for index in range(x.shape[0]):
            vectors[index, index % 2] = 1.0
        return [vectors]


class _ContentSession(_Session):
    """Return one deterministic row-local embedding for batch parity tests."""

    def __init__(self, providers=("CPUExecutionProvider",), tracker=None):
        super().__init__(providers)
        self.tracker = tracker

    def run(self, outputs, feeds):
        x = feeds["feats"]
        self.calls.append(x.copy())
        if self.tracker is not None:
            self.tracker.max_features_before_run = max(
                self.tracker.max_features_before_run,
                self.tracker.features_since_run,
            )
            self.tracker.features_since_run = 0
        vectors = np.empty((x.shape[0], 3), dtype=np.float32)
        vectors[:, 0] = x.mean(axis=(1, 2))
        vectors[:, 1] = x[:, 0, :].mean(axis=1) + 1.0
        vectors[:, 2] = x[:, -1, :].mean(axis=1) + 2.0
        return [vectors]


class _FeatureBufferTracker:
    def __init__(self):
        self.features_since_run = 0
        self.max_features_before_run = 0


class _NativeFrameOptions:
    pass


class _NativeMelOptions:
    pass


class _NativeFbankOptions:
    last = None

    def __init__(self):
        self.frame_opts = _NativeFrameOptions()
        self.mel_opts = _NativeMelOptions()
        self.energy_floor = None
        self.use_energy = None
        self.use_log_fbank = None
        self.use_power = None
        _NativeFbankOptions.last = self


class _NativeOnlineFbank:
    def __init__(self, opts):
        self.opts = opts
        self._frames = []

    def accept_waveform(self, sample_rate, samples):
        assert sample_rate == 16000
        frame_len = round(sample_rate * self.opts.frame_opts.frame_length_ms / 1000.0)
        frame_shift = round(sample_rate * self.opts.frame_opts.frame_shift_ms / 1000.0)
        count = 1 + (len(samples) - frame_len) // frame_shift if len(samples) >= frame_len else 0
        bins = self.opts.mel_opts.num_bins
        self._frames = [np.arange(bins, dtype=np.float32) + frame for frame in range(count)]

    @property
    def num_frames_ready(self):
        return len(self._frames)

    def get_frame(self, index):
        return self._frames[index]


class _NativeModule:
    FbankOptions = _NativeFbankOptions
    OnlineFbank = _NativeOnlineFbank


@unittest.skipIf(np is None, "numpy is optional")
class WeSpeakerRuntimeTests(unittest.TestCase):
    def artifact(self, path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return VerifiedArtifact("wespeaker", "speaker_embedding", path, digest, path.stat().st_size)

    def test_feature_contract_and_determinism(self):
        samples = np.sin(np.arange(1600, dtype=np.float32) / 17.0)
        first = wespeaker_logmel_features_approx(samples)
        second = wespeaker_logmel_features_approx(samples)
        self.assertEqual(first.shape[1], 80)
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(DEVELOPMENT_APPROXIMATION_TAG, "wespeaker-logmel-fbank-development-approximation")
        self.assertEqual(STRICT_NATIVE_FBANK_TAG, "wespeaker-kaldi-native-fbank")

    def test_strict_native_fbank_2s_frame_count_and_settings(self):
        samples = np.linspace(-0.5, 0.5, 32_000, dtype=np.float32)
        features = kaldi_native_fbank_features(samples, module=_NativeModule)
        self.assertEqual(features.shape, (198, 80))
        np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=1e-6)
        opts = _NativeFbankOptions.last
        self.assertEqual(opts.frame_opts.samp_freq, 16_000)
        self.assertEqual(opts.frame_opts.dither, 0.0)
        self.assertEqual(opts.frame_opts.window_type, "hamming")
        self.assertEqual(opts.frame_opts.frame_shift_ms, 10.0)
        self.assertEqual(opts.frame_opts.frame_length_ms, 25.0)
        self.assertTrue(opts.frame_opts.snip_edges)
        self.assertTrue(opts.frame_opts.remove_dc_offset)
        self.assertEqual(opts.frame_opts.preemph_coeff, 0.97)
        self.assertEqual(opts.mel_opts.num_bins, 80)
        self.assertEqual(opts.mel_opts.low_freq, 20.0)
        self.assertEqual(opts.mel_opts.high_freq, 0.0)
        self.assertEqual(opts.use_energy, False)
        self.assertEqual(opts.use_log_fbank, True)
        self.assertEqual(opts.use_power, True)

    def test_installed_native_fbank_golden_shape_and_cmn(self):
        try:
            import kaldi_native_fbank  # type: ignore
        except ImportError:
            self.skipTest("kaldi_native_fbank is an optional platform wheel")
        samples = np.sin(2.0 * np.pi * 440.0 * np.arange(32_000, dtype=np.float32) / 16_000.0)
        features = kaldi_native_fbank_features(samples, module=kaldi_native_fbank)
        self.assertEqual(features.shape, (198, 80))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=2e-5)
        self.assertTrue(np.isfinite(features).all())

    def test_strict_native_fbank_rejects_short_input(self):
        with self.assertRaises(ContractValidationError):
            kaldi_native_fbank_features(np.zeros(399, dtype=np.float32), module=_NativeModule)

    def test_strict_native_fbank_missing_package_fails_closed(self):
        import unittest.mock
        with unittest.mock.patch(
            "sddiar.wespeaker_runtime._load_kaldi_native_fbank",
            side_effect=ModelRuntimeIncompatible("missing native frontend"),
        ):
            with self.assertRaises(ModelRuntimeIncompatible):
                kaldi_native_fbank_features(np.zeros(400, dtype=np.float32))

    def test_cpu_session_batches_regions_and_normalizes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            session = _Session()
            backend = WeSpeakerCpuEmbeddingBackend(
                self.artifact(path), session=session, max_batch_regions=1,
                feature_mode="development_approximation",
            )
            regions = [EmbeddingRegion("r1", "t1", 0, 1_000_000, 800_000, .8), EmbeddingRegion("r2", "t2", 1_000_000, 2_000_000, 800_000, .8)]
            audio = {r.embedding_region_id: np.zeros(1600, dtype=np.float32) for r in regions}
            results = backend.embed(regions, audio)
            self.assertEqual(len(results), 2)
            self.assertEqual(len(session.calls), 2)
            self.assertAlmostEqual(np.linalg.norm(results[0].vector), 1.0, places=6)
            self.assertEqual(results[0].dimension, 3)

    def test_per_call_audio_provider_overrides_instance_provider(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            region = EmbeddingRegion("r1", "t1", 0, 1_000_000, 800_000, .8)
            seen = []

            def extract(audio, *, config):
                seen.append(np.asarray(audio).copy())
                return np.zeros((5, 80), dtype=np.float32)

            backend = WeSpeakerCpuEmbeddingBackend(
                self.artifact(path),
                session=_Session(),
                feature_extractor=extract,
                audio_provider=lambda _region: np.full(1600, 1.0, dtype=np.float32),
            )
            backend.embed([region])
            backend.embed(
                [region],
                audio_provider=lambda _region: np.full(1600, 2.0, dtype=np.float32),
            )
            self.assertEqual(len(seen), 2)
            np.testing.assert_array_equal(seen[0], 1.0)
            np.testing.assert_array_equal(seen[1], 2.0)

    def test_exact_length_batches_match_batch_one_and_restore_order(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            frame_counts = (5, 7, 5, 9, 7, 5, 9, 7, 5)
            regions = tuple(
                EmbeddingRegion(
                    f"r{index}", f"t{index}", index * 1_000_000,
                    (index + 1) * 1_000_000, 800_000, .8,
                )
                for index in range(len(frame_counts))
            )
            features = {
                region.embedding_region_id: np.full(
                    (frames, 80), index + 0.25, dtype=np.float32
                )
                for index, (region, frames) in enumerate(zip(regions, frame_counts))
            }
            extractor = lambda audio, *, config: audio
            baseline_session = _ContentSession()
            baseline = WeSpeakerCpuEmbeddingBackend(
                self.artifact(path), session=baseline_session, max_batch_regions=1,
                feature_extractor=extractor,
            ).embed(regions, features)
            self.assertEqual(
                [result.embedding_region_id for result in baseline],
                [region.embedding_region_id for region in regions],
            )

            for batch_size in (2, 4, 8):
                with self.subTest(batch_size=batch_size):
                    session = _ContentSession()
                    actual = WeSpeakerCpuEmbeddingBackend(
                        self.artifact(path), session=session,
                        max_batch_regions=batch_size,
                        feature_extractor=extractor,
                        exact_length_batching=True,
                        batch_buffer_regions=5,
                    ).embed(regions, features)
                    self.assertEqual(
                        [result.embedding_region_id for result in actual],
                        [region.embedding_region_id for region in regions],
                    )
                    for expected, observed in zip(baseline, actual):
                        np.testing.assert_array_equal(expected.vector, observed.vector)
                    self.assertTrue(all(call.shape[0] <= batch_size for call in session.calls))
                    self.assertTrue(all(
                        call.shape[1] in set(frame_counts) for call in session.calls
                    ))
                    # Equal-length grouping must create real multi-row batches.
                    self.assertTrue(any(call.shape[0] > 1 for call in session.calls))

    def test_exact_length_feature_lookahead_and_session_batches_are_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            tracker = _FeatureBufferTracker()
            session = _ContentSession(tracker=tracker)

            def extract(audio, *, config):
                tracker.features_since_run += 1
                return audio

            regions = tuple(
                EmbeddingRegion(
                    f"r{index}", f"t{index}", index * 1_000_000,
                    (index + 1) * 1_000_000, 800_000, .8,
                )
                for index in range(11)
            )
            features = {
                region.embedding_region_id: np.full(
                    ((5, 7, 9)[index % 3], 80), index + 1, dtype=np.float32
                )
                for index, region in enumerate(regions)
            }
            results = WeSpeakerCpuEmbeddingBackend(
                self.artifact(path), session=session, max_batch_regions=2,
                feature_extractor=extract,
                exact_length_batching=True,
                batch_buffer_regions=3,
            ).embed(regions, features)
            self.assertEqual(len(results), len(regions))
            self.assertEqual(tracker.max_features_before_run, 3)
            self.assertTrue(all(call.shape[0] <= 2 for call in session.calls))
            self.assertEqual(tracker.features_since_run, 0)

    def test_exact_length_batching_rejects_invalid_bounds_and_subsegments(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            artifact = self.artifact(path)
            for options in (
                {"exact_length_batching": "yes"},
                {"exact_length_batching": True, "batch_buffer_regions": 0},
                {"exact_length_batching": True, "subsegment_windows": True},
                {"exact_length_batching": True, "subsegment_region_ids": ("r1",)},
            ):
                with self.subTest(options=options), self.assertRaises(ContractValidationError):
                    WeSpeakerCpuEmbeddingBackend(
                        artifact, session=_Session(), **options,
                    )

    def test_real_ort_exact_length_batches_match_batch_one(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "artifacts" / "dev" / "models" / "voxceleb_resnet34.onnx"
        )
        if not model_path.is_file():
            self.skipTest("real WeSpeaker ONNX development artifact is unavailable")
        try:
            import onnxruntime  # noqa: F401  # type: ignore
        except ImportError:
            self.skipTest("onnxruntime is an optional platform wheel")
        from sddiar.ort_cpu import create_ort_session
        session = create_ort_session(model_path, threads=1)

        rng = np.random.default_rng(20260826)
        frame_counts = (148, 198, 148, 298, 198, 298, 148, 198)
        regions = tuple(
            EmbeddingRegion(
                f"real-r{index}", f"real-t{index}", index * 3_000_000,
                (index + 1) * 3_000_000, 2_500_000, .9,
            )
            for index in range(len(frame_counts))
        )
        features = {
            region.embedding_region_id: rng.standard_normal(
                (frames, 80), dtype=np.float32
            )
            for region, frames in zip(regions, frame_counts)
        }
        extractor = lambda audio, *, config: audio
        artifact = self.artifact(model_path)
        baseline = WeSpeakerCpuEmbeddingBackend(
            artifact, session=session, max_batch_regions=1,
            feature_extractor=extractor,
        ).embed(regions, features)
        for batch_size in (2, 4, 8):
            with self.subTest(batch_size=batch_size):
                actual = WeSpeakerCpuEmbeddingBackend(
                    artifact, session=session, max_batch_regions=batch_size,
                    feature_extractor=extractor,
                    exact_length_batching=True,
                    batch_buffer_regions=8,
                ).embed(regions, features)
                self.assertEqual(
                    [result.embedding_region_id for result in actual],
                    [region.embedding_region_id for region in regions],
                )
                for expected, observed in zip(baseline, actual):
                    left = np.asarray(expected.vector, dtype=np.float64)
                    right = np.asarray(observed.vector, dtype=np.float64)
                    cosine = float(np.dot(left, right))
                    max_abs = float(np.max(np.abs(left - right)))
                    self.assertGreaterEqual(cosine, 1.0 - 2e-6)
                    self.assertLessEqual(max_abs, 2e-5)

    def test_optional_upstream_subsegment_windows_aggregate_normalized_embeddings(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            session = _WindowSession()
            feature = np.arange(5 * 80, dtype=np.float32).reshape(5, 80)
            backend = WeSpeakerCpuEmbeddingBackend(
                self.artifact(path), session=session, max_batch_regions=1,
                feature_extractor=lambda audio, *, config: feature,
                subsegment_windows=True, window_frames=3, period_frames=2,
            )
            region = EmbeddingRegion("r1", "t1", 0, 50_000, 40_000, .8)
            result = backend.embed([region], {"r1": np.zeros(400, dtype=np.float32)})[0]
            self.assertTrue(result.is_valid)
            self.assertEqual(result.valid_window_count, 2)
            self.assertAlmostEqual(result.intra_window_consistency, 2 ** -0.5, places=5)
            self.assertAlmostEqual(np.linalg.norm(result.vector), 1.0, places=6)
            np.testing.assert_allclose(session.calls[0].mean(axis=1), 0.0, atol=1e-6)
            self.assertEqual(session.calls[0].shape, (2, 3, 80))

    def test_selective_subsegments_replace_only_selected_region_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            session = _WindowSession()
            feature = np.arange(5 * 80, dtype=np.float32).reshape(5, 80)
            backend = WeSpeakerCpuEmbeddingBackend(
                self.artifact(path), session=session, max_batch_regions=1,
                feature_extractor=lambda audio, *, config: feature,
                window_frames=3, period_frames=2,
                subsegment_region_ids=("r2",),
            )
            regions = (
                EmbeddingRegion("r1", "t1", 0, 50_000, 40_000, .8),
                EmbeddingRegion("r2", "t2", 50_000, 100_000, 40_000, .8),
            )
            audio = {region.embedding_region_id: np.zeros(400, dtype=np.float32) for region in regions}
            selected = backend.embed(regions, audio)
            baseline = backend.last_selector_baseline_results
            self.assertIsNotNone(baseline)
            self.assertEqual(selected[0], baseline[0])
            self.assertEqual(selected[0].vector, baseline[0].vector)
            self.assertNotEqual(selected[1].vector, baseline[1].vector)
            self.assertEqual(selected[1].valid_window_count, 2)
            self.assertEqual([call.shape for call in session.calls], [(1, 5, 80), (1, 5, 80), (2, 3, 80)])

    def test_global_and_selective_subsegments_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            with self.assertRaises(ContractValidationError):
                WeSpeakerCpuEmbeddingBackend(
                    self.artifact(path), session=_Session(),
                    subsegment_windows=True,
                    subsegment_region_ids=("r1",),
                )

    def test_rejects_non_cpu_or_wrong_inspected_names(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            with self.assertRaises(ModelRuntimeIncompatible):
                WeSpeakerCpuEmbeddingBackend(self.artifact(path), session=_Session(("CUDAExecutionProvider",)))

    def test_backend_defaults_to_native_and_does_not_fallback(self):
        import unittest.mock
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.onnx"
            path.write_bytes(b"local verified fixture")
            backend = WeSpeakerCpuEmbeddingBackend(self.artifact(path), session=_Session())
            region = EmbeddingRegion("r1", "t1", 0, 1_000_000, 800_000, .8)
            with unittest.mock.patch(
                "sddiar.wespeaker_runtime._load_kaldi_native_fbank",
                side_effect=ModelRuntimeIncompatible("missing native frontend"),
            ):
                with self.assertRaises(ModelRuntimeIncompatible):
                    backend.embed([region], {"r1": np.zeros(1600, dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
