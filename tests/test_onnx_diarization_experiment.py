from __future__ import annotations

import importlib.util
import math
import sys
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from sddiar.contracts import EmbeddingResult
from sddiar.vad import VadFrame


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "run_onnx_diarization_experiment.py"
    spec = importlib.util.spec_from_file_location("onnx_experiment_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeSilero:
    def infer_chunk_stream(self, chunks):
        tuple(chunks)  # prove the bounded decoder path is consumable
        return tuple(VadFrame(i * 2_500_000, i * 2_500_000 + 2_000_000, 1.0, True) for i in range(4))


class CapturingSilero(FakeSilero):
    def __init__(self):
        self.first_sample = None

    def infer_chunk_stream(self, chunks):
        materialized = tuple(chunks)
        first = materialized[0].samples[0]
        self.first_sample = float(first[0] if isinstance(first, tuple) else first)
        return tuple(VadFrame(i * 2_500_000, i * 2_500_000 + 2_000_000, 1.0, True) for i in range(4))


class FakeEmbedding:
    def embed(self, regions):
        return tuple(
            EmbeddingResult(
                region.embedding_region_id, region.tracklet_id, True,
                (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0),
                dimension=2, valid_window_count=1, clean_window_coverage=1.0,
                intra_window_consistency=1.0, quality=1.0,
            )
            for index, region in enumerate(regions)
        )


class FakeSelectiveEmbedding(FakeEmbedding):
    subsegment_region_ids = frozenset()
    last_selector_baseline_results = None

    def embed(self, regions):
        results = super().embed(regions)
        self.last_selector_baseline_results = results
        return results


class OnnxExperimentTests(unittest.TestCase):
    def test_fake_runtime_runs_and_output_is_redacted(self):
        module = load_module()
        with TemporaryDirectory() as directory:
            wav_path = Path(directory) / "fixture.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000)
                handle.writeframes(b"\x00\x10" * (16_000 * 10))
            reference = Path(directory) / "reference.json"
            reference.write_text('{"turns":[{"speaker_id":"REF_00","start_sec":0,"end_sec":2},{"speaker_id":"REF_01","start_sec":2.5,"end_sec":4.5},{"speaker_id":"REF_00","start_sec":5,"end_sec":7},{"speaker_id":"REF_01","start_sec":7.5,"end_sec":9.5}]}')
            result = module.run_experiment(
                wav_path, "local-silero.onnx", "local-wespeaker.onnx", reference,
                silero_runtime=FakeSilero(), embedding_backend=FakeEmbedding(),
            )
        self.assertEqual(result["schema"], "onnx_diarization_experiment_v2")
        self.assertEqual(result["valid_embeddings"], 4)
        self.assertIn("metrics", result)
        self.assertEqual(len(result["span_timeline_sha256"]), 64)
        self.assertNotIn("decoder_calibration", result)
        self.assertNotIn("transcript", str(result).lower())
        self.assertNotIn("REF_00", str(result))

        expected_stages = {
            "vad_decode_wall_sec",
            "segmentation_tracklets_wall_sec",
            "embeddings_wall_sec",
            "hypothesis_state_wall_sec",
            "finalization_proxy_calibration_wall_sec",
            "evaluation_scoring_wall_sec",
        }
        self.assertEqual(set(result["stage_timings"]), expected_stages)
        for value in result["stage_timings"].values():
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        self.assertTrue(math.isfinite(result["process_cpu_sec"]))
        self.assertGreaterEqual(result["process_cpu_sec"], 0.0)
        self.assertTrue(math.isfinite(result["cpu_seconds_per_wall_second"]))
        self.assertGreaterEqual(result["cpu_seconds_per_wall_second"], 0.0)

    def test_auto_gain_runner_flag_is_opt_in_and_hashed(self):
        module = load_module()
        with TemporaryDirectory() as directory:
            wav_path = Path(directory) / "quiet.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000)
                handle.writeframes((100).to_bytes(2, "little", signed=True) * (16_000 * 10))
            reference = Path(directory) / "reference.json"
            reference.write_text('{"turns":[{"speaker_id":"REF_00","start_sec":0,"end_sec":2},{"speaker_id":"REF_01","start_sec":2.5,"end_sec":4.5},{"speaker_id":"REF_00","start_sec":5,"end_sec":7},{"speaker_id":"REF_01","start_sec":7.5,"end_sec":9.5}]}')
            implicit = module.run_experiment(
                wav_path, "local-silero.onnx", "local-wespeaker.onnx", reference,
                silero_runtime=FakeSilero(), embedding_backend=FakeEmbedding(),
            )
            explicit = module.run_experiment(
                wav_path, "local-silero.onnx", "local-wespeaker.onnx", reference,
                auto_gain_normalization=False,
                silero_runtime=FakeSilero(), embedding_backend=FakeEmbedding(),
            )
            capture = CapturingSilero()
            challenger = module.run_experiment(
                wav_path, "local-silero.onnx", "local-wespeaker.onnx", reference,
                auto_gain_normalization=True,
                silero_runtime=capture, embedding_backend=FakeEmbedding(),
            )

        self.assertEqual(implicit["span_timeline_sha256"], explicit["span_timeline_sha256"])
        self.assertEqual(implicit["decision"], explicit["decision"])
        self.assertFalse(implicit["runtime_config"]["auto_gain_normalization"])
        self.assertFalse(implicit["audio_gain_normalization"]["enabled"])
        self.assertTrue(challenger["runtime_config"]["auto_gain_normalization"])
        self.assertEqual(challenger["audio_gain_normalization"]["applied_gain"], 4.0)
        self.assertAlmostEqual(capture.first_sample, 100.0 / 32768.0 * 4.0, places=8)
        self.assertIn("audio_gain_profile_wall_sec", challenger["stage_timings"])
        self.assertEqual(
            challenger["runtime_config"]["audio_gain_policy_sha256"],
            challenger["audio_gain_normalization"]["policy_sha256"],
        )

    def test_opt_in_decoder_calibration_reports_one_selected_holdout_and_aggregate_trace(self):
        module = load_module()
        with TemporaryDirectory() as directory:
            wav_path = Path(directory) / "fixture.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000)
                handle.writeframes(b"\x00\x10" * (16_000 * 10))
            reference = Path(directory) / "reference.json"
            reference.write_text('{"turns":[{"speaker_id":"REF_00","start_sec":0,"end_sec":2},{"speaker_id":"REF_01","start_sec":2.5,"end_sec":4.5},{"speaker_id":"REF_00","start_sec":5,"end_sec":7},{"speaker_id":"REF_01","start_sec":7.5,"end_sec":9.5}]}')
            result = module.run_experiment(
                wav_path, "local-silero.onnx", "local-wespeaker.onnx", reference,
                calibrate_decoder=True,
                silero_runtime=FakeSilero(), embedding_backend=FakeEmbedding(),
            )
        report = result["decoder_calibration"]
        self.assertEqual(report["kind"], "FROZEN_HYPOTHESIS_DECODER_CALIBRATION_V1")
        self.assertEqual(report["candidate_count"], 21)
        self.assertEqual(len(report["candidates"]), 21)
        self.assertTrue(all("holdout_metrics" not in candidate for candidate in report["candidates"]))
        self.assertIn("selected_holdout_metrics", report)
        self.assertIn("selected_full_metrics", report)
        self.assertIn("local_groups", report["aggregate_diagnostics"])
        self.assertIn("baseline_viterbi_overrides", report["aggregate_diagnostics"])
        self.assertIn("selected_viterbi_overrides", report["aggregate_diagnostics"])
        self.assertEqual(result["span_timeline_sha256"], report["emitted_span_timeline_sha256"])
        self.assertNotIn("REF_00", str(report))

    def test_micro_selector_requires_decoder_and_reports_exact_non_micro_parity(self):
        module = load_module()
        with TemporaryDirectory() as directory:
            wav_path = Path(directory) / "fixture.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000)
                handle.writeframes(b"\x00\x10" * (16_000 * 10))
            reference = Path(directory) / "reference.json"
            reference.write_text('{"turns":[{"speaker_id":"REF_00","start_sec":0,"end_sec":2},{"speaker_id":"REF_01","start_sec":2.5,"end_sec":4.5},{"speaker_id":"REF_00","start_sec":5,"end_sec":7},{"speaker_id":"REF_01","start_sec":7.5,"end_sec":9.5}]}')
            with self.assertRaises(ValueError):
                module.run_experiment(
                    wav_path, "local-silero.onnx", "local-wespeaker.onnx", reference,
                    micro_subsegment_windows=True,
                    fast_fp32_baseline_rtf=1.0,
                    silero_runtime=FakeSilero(), embedding_backend=FakeSelectiveEmbedding(),
                )
            result = module.run_experiment(
                wav_path, "local-silero.onnx", "local-wespeaker.onnx", reference,
                micro_subsegment_windows=True,
                calibrate_decoder=True,
                fast_fp32_baseline_rtf=1.0,
                silero_runtime=FakeSilero(), embedding_backend=FakeSelectiveEmbedding(),
            )
        report = result["micro_subsegment_experiment"]
        self.assertTrue(report["embedding_parity"]["anchor_support_exact"])
        self.assertTrue(report["embedding_parity"]["decision_exact"])
        self.assertTrue(report["embedding_parity"]["h2_diagnostics_exact"])
        self.assertIn("rtf_lte_1_5x_fast_fp32", report["gates"])
        self.assertEqual(result["span_timeline_sha256"], report["emitted_span_timeline_sha256"])


if __name__ == "__main__":
    unittest.main()
