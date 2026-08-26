import hashlib
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/quantize_wespeaker_static.py"
SPEC = importlib.util.spec_from_file_location("quantize_wespeaker_static", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
quant = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quant
SPEC.loader.exec_module(quant)


def write_pcm16(path: Path, values: list[int], sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = b"".join(int(value).to_bytes(2, "little", signed=True) for value in values)
        handle.writeframes(payload)


class QuantizationSelectionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is optional")
    def test_duration_energy_selection_is_deterministic_and_split_bounded(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.wav"
            values = []
            for sample in range(16_000 * 12):
                second = sample // 16_000
                amplitude = 200 + second * 500
                values.append(round(amplitude * math.sin(2.0 * math.pi * 220.0 * sample / 16_000)))
            write_pcm16(path, values)
            accessor = quant.WavPcmAccessor(path)
            split = round(accessor.layout.frame_count * 0.60)
            candidates = quant.enumerate_energy_candidates(
                accessor,
                segment_start_sample=0,
                segment_end_sample=split,
                durations_sec=(0.5, 1.0),
                stride_sec=1.0,
                np=np,
            )
            first = quant.select_duration_energy_diverse(candidates, max_windows=8)
            second = quant.select_duration_energy_diverse(candidates, max_windows=8)
            self.assertEqual(first, second)
            self.assertEqual(quant.selection_digest(first), quant.selection_digest(second))
            self.assertEqual(len(first), 8)
            self.assertEqual({window.duration_samples for window in first}, {8_000, 16_000})
            self.assertTrue(all(window.end_sample <= split for window in first))
            self.assertGreater(len({window.mean_square_pcm for window in first}), 2)

            features_a, digest_a = quant.build_features(accessor, first, np=np)
            features_b, digest_b = quant.build_features(accessor, second, np=np)
            self.assertEqual(digest_a, digest_b)
            self.assertEqual(len(features_a), 8)
            self.assertEqual([feature.shape for feature in features_a], [feature.shape for feature in features_b])
            self.assertTrue(all(feature.dtype == np.float32 for feature in features_a))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is optional")
    def test_spearman_uses_average_tie_ranks(self):
        import numpy as np

        self.assertAlmostEqual(quant.spearman([1, 1, 2, 3], [10, 10, 20, 30], np=np), 1.0)
        self.assertAlmostEqual(quant.spearman([1, 2, 3], [3, 2, 1], np=np), -1.0)

    def test_builder_has_no_network_imports(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertTrue(imported.isdisjoint({"socket", "urllib", "http", "requests"}))


@unittest.skipUnless(
    importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
    "ONNX quantization tooling is optional",
)
class QuantizationRecipeTests(unittest.TestCase):
    @staticmethod
    def tiny_model(path: Path) -> None:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        graph_input = helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, 8, 80])
        graph_output = helper.make_tensor_value_info("embs", TensorProto.FLOAT, [1, 4])
        axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axes")
        conv_weight = numpy_helper.from_array(
            np.asarray([[[[0.5]]], [[[1.5]]]], dtype=np.float32), "conv.weight"
        )
        conv_bias = numpy_helper.from_array(np.asarray([0.1, -0.2], dtype=np.float32), "conv.bias")
        conv2_weight = numpy_helper.from_array(
            np.asarray(
                [
                    [[[0.8]], [[0.2]]],
                    [[[0.1]], [[0.9]]],
                ],
                dtype=np.float32,
            ),
            "conv2.weight",
        )
        conv2_bias = numpy_helper.from_array(np.asarray([0.0, 0.0], dtype=np.float32), "conv2.bias")
        gemm_weight = numpy_helper.from_array(
            np.linspace(-0.1, 0.1, 1280 * 4, dtype=np.float32).reshape(1280, 4),
            "gemm.weight",
        )
        gemm_bias = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "gemm.bias")
        nodes = [
            helper.make_node("Unsqueeze", ["feats", "axes"], ["x4"], name="Unsqueeze"),
            helper.make_node(
                "Conv", ["x4", "conv.weight", "conv.bias"], ["conv"], name="Conv", kernel_shape=[1, 1]
            ),
            helper.make_node("Relu", ["conv"], ["relu"], name="Relu"),
            helper.make_node(
                "Conv", ["relu", "conv2.weight", "conv2.bias"], ["conv2"], name="Conv2", kernel_shape=[1, 1]
            ),
            helper.make_node("Flatten", ["conv2"], ["flat"], name="Flatten", axis=1),
            helper.make_node("Gemm", ["flat", "gemm.weight", "gemm.bias"], ["embs"], name="Gemm"),
        ]
        graph = helper.make_graph(
            nodes,
            "tiny-wespeaker",
            [graph_input],
            [graph_output],
            [axes, conv_weight, conv_bias, conv2_weight, conv2_bias, gemm_weight, gemm_bias],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 10
        onnx.checker.check_model(model)
        onnx.save(model, path)

    def test_recipe_emits_static_qdq_s8s8_and_preserves_source(self):
        import numpy as np
        import onnx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            preprocessed = root / "preprocessed.onnx"
            candidate = root / "candidate.int8.onnx"
            self.tiny_model(source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            rng = np.random.default_rng(7)
            calibration = tuple(
                rng.normal(size=(1, 8, 80)).astype(np.float32) for _ in range(8)
            )
            summary = quant.build_static_qdq_candidate(
                source, preprocessed, candidate, calibration
            )
            after = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(preprocessed.is_file())
            self.assertTrue(candidate.is_file())
            self.assertTrue(summary["input_output_fp32"])
            self.assertGreater(summary["qdq_nodes"], 0)
            self.assertGreater(summary["int8_initializer_count"], 0)
            self.assertGreater(summary["per_channel_scale_count"], 0)
            self.assertEqual(summary["integer_operator_nodes"], [])
            onnx.checker.check_model(onnx.load(candidate))

    def test_q1b_keeps_only_first_conv_and_final_gemm_weights_fp32(self):
        import numpy as np
        import onnx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            preprocessed = root / "q1b.preprocessed.onnx"
            candidate = root / "q1b.int8.onnx"
            self.tiny_model(source)
            rng = np.random.default_rng(11)
            calibration = tuple(
                rng.normal(size=(1, 8, 80)).astype(np.float32) for _ in range(8)
            )
            summary = quant.build_static_qdq_candidate(
                source,
                preprocessed,
                candidate,
                calibration,
                keep_first_conv_last_gemm_fp32=True,
            )
            self.assertEqual(summary["excluded_fp32_nodes"], ["Conv", "Gemm"])
            model = onnx.load(candidate)
            initializers = {initializer.name: initializer for initializer in model.graph.initializer}
            nodes = {node.name: node for node in model.graph.node}
            self.assertEqual(
                initializers[nodes["Conv"].input[1]].data_type,
                onnx.TensorProto.FLOAT,
            )
            self.assertEqual(
                initializers[nodes["Gemm"].input[1]].data_type,
                onnx.TensorProto.FLOAT,
            )

    def test_q1_reuse_requires_exact_selection_and_preserved_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            dev = Path(directory) / "dev"
            models = dev / "models"
            models.mkdir(parents=True)
            artifact = models / "q1.int8.onnx"
            artifact.write_bytes(b"q1")
            artifact_hash = hashlib.sha256(b"q1").hexdigest()
            selection = {
                "algorithm": "equal-duration-budget-even-energy-ranks-v1",
                "durations_sec": [0.5, 0.9],
                "candidate_stride_sec": 4.0,
                "calibration_candidate_count": 10,
                "calibration_window_count": 4,
                "calibration_selection_sha256": "a" * 64,
                "calibration_feature_sha256": "b" * 64,
                "holdout_candidate_count": 8,
                "holdout_window_count": 4,
                "holdout_selection_sha256": "c" * 64,
                "holdout_feature_sha256": "d" * 64,
            }
            metadata = models / "q1.metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "candidate_id": "wespeaker-voxceleb-resnet34-qdq-s8s8-pc-minmax-dev-20260826",
                        "production_approved": False,
                        "source": {"approved_audio_sha256": "e" * 64},
                        "split": {"split_sample": 123},
                        "selection": selection,
                        "artifacts": {
                            "int8_model": {
                                "relative_path": "models/q1.int8.onnx",
                                "sha256": artifact_hash,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = quant.verify_q1_selection_reuse(
                metadata,
                audio_sha256="e" * 64,
                split_sample=123,
                selection=selection,
            )
            self.assertTrue(result["exact_reuse_verified"])
            changed = dict(selection)
            changed["holdout_feature_sha256"] = "f" * 64
            with self.assertRaisesRegex(quant.QuantizationCandidateError, "differs from Q1"):
                quant.verify_q1_selection_reuse(
                    metadata,
                    audio_sha256="e" * 64,
                    split_sample=123,
                    selection=changed,
                )


if __name__ == "__main__":
    unittest.main()
