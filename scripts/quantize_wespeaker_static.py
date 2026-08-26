#!/usr/bin/env python3
"""Build and evaluate a development-only static INT8 WeSpeaker candidate.

The builder is intentionally separate from the serving package.  It uses an
approved local PCM16/16 kHz/mono WAV, never reads speaker labels, calibrates on
the first 60 percent of source samples, and evaluates embedding parity on the
last 40 percent.  The source FP32 model is hash checked before and after the
build and is never modified.

The emitted model uses ONNX Runtime's static QDQ S8S8 recipe with per-channel
weights, MinMax calibration, ``reduce_range=False``, and only Conv/Gemm as
quantized operator types.  Inputs and outputs remain float32.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sddiar.media import WavPcmAccessor  # noqa: E402
from sddiar.wespeaker_runtime import kaldi_native_fbank_features  # noqa: E402


FP32_MODEL_SHA256 = "9fea6516d7ad6bf0a76c7689f5a49b65d330fad6dde96c91bb4435ffbfe056a1"
DEFAULT_DURATIONS_SEC = (0.5, 0.9, 1.5, 2.2, 3.0)
DEFAULT_FP32_MODEL = ROOT / "artifacts/dev/models/voxceleb_resnet34.onnx"
DEFAULT_PREPROCESSED_MODEL = (
    ROOT / "artifacts/dev/models/voxceleb_resnet34.qdq-s8s8-pc-minmax.preprocessed.onnx"
)
DEFAULT_INT8_MODEL = (
    ROOT / "artifacts/dev/models/voxceleb_resnet34.qdq-s8s8-pc-minmax.int8.onnx"
)
DEFAULT_METADATA = (
    ROOT / "artifacts/dev/models/voxceleb_resnet34.qdq-s8s8-pc-minmax.int8.metadata.json"
)
DEFAULT_Q1B_PREPROCESSED_MODEL = ROOT / (
    "artifacts/dev/models/"
    "voxceleb_resnet34.qdq-s8s8-pc-minmax.q1b-firstconv-lastgemm-fp32.preprocessed.onnx"
)
DEFAULT_Q1B_INT8_MODEL = ROOT / (
    "artifacts/dev/models/"
    "voxceleb_resnet34.qdq-s8s8-pc-minmax.q1b-firstconv-lastgemm-fp32.int8.onnx"
)
DEFAULT_Q1B_METADATA = ROOT / (
    "artifacts/dev/models/"
    "voxceleb_resnet34.qdq-s8s8-pc-minmax.q1b-firstconv-lastgemm-fp32.int8.metadata.json"
)
DEFAULT_Q1_METADATA = DEFAULT_METADATA
DEFAULT_TOOLING_MANIFEST = ROOT / "artifacts/dev/tooling/quantization-tooling.json"


class QuantizationCandidateError(RuntimeError):
    """Raised when a build, provenance, or parity invariant fails."""


@dataclass(frozen=True, slots=True)
class SampleWindow:
    start_sample: int
    end_sample: int
    duration_samples: int
    mean_square_pcm: int

    @property
    def duration_sec(self) -> float:
        return self.duration_samples / 16_000.0

    @property
    def rms(self) -> float:
        return math.sqrt(self.mean_square_pcm) / 32_768.0


@dataclass(frozen=True, slots=True)
class ParityGates:
    embedding_cosine_median_min: float = 0.995
    embedding_cosine_p01_min: float = 0.980
    pairwise_spearman_min: float = 0.995
    pairwise_abs_error_p95_max: float = 0.020
    model_bytes_max: int = 9 * 1024 * 1024


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> str:
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise QuantizationCandidateError(f"{label} expected SHA-256 is invalid")
    if not path.is_file() or path.is_symlink():
        raise QuantizationCandidateError(f"{label} must be a regular non-symlink file")
    actual = sha256_path(path)
    if actual != expected:
        raise QuantizationCandidateError(f"{label} SHA-256 mismatch")
    return actual


def _read_window(accessor: WavPcmAccessor, window: SampleWindow, np: Any) -> Any:
    layout = accessor.layout
    if layout.sample_width_bytes != 2 or layout.channel_count != 1:
        raise QuantizationCandidateError("calibration requires PCM16 mono WAV")
    if window.start_sample < 0 or window.end_sample > layout.frame_count:
        raise QuantizationCandidateError("sample window exceeds WAV bounds")
    count = window.end_sample - window.start_sample
    if count != window.duration_samples or count <= 0:
        raise QuantizationCandidateError("sample window length invariant failed")
    with accessor.path.open("rb") as handle:
        handle.seek(layout.data_offset + window.start_sample * 2)
        payload = handle.read(count * 2)
    if len(payload) != count * 2:
        raise QuantizationCandidateError("WAV ended before a selected window")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768.0)


def _mean_square_pcm(samples: Any, np: Any) -> int:
    pcm = np.rint(samples * np.float32(32768.0)).astype(np.int64)
    if pcm.size == 0:
        raise QuantizationCandidateError("cannot score an empty calibration window")
    square_sum = int(np.sum(pcm * pcm, dtype=np.int64))
    return square_sum // int(pcm.size)


def _window_starts(start: int, end: int, duration: int, stride: int) -> tuple[int, ...]:
    last = end - duration
    if last < start:
        return ()
    values = list(range(start, last + 1, stride))
    if not values or values[-1] != last:
        values.append(last)
    return tuple(values)


def enumerate_energy_candidates(
    accessor: WavPcmAccessor,
    *,
    segment_start_sample: int,
    segment_end_sample: int,
    durations_sec: Sequence[float],
    stride_sec: float,
    np: Any,
) -> tuple[SampleWindow, ...]:
    """Enumerate bounded deterministic windows and integer PCM energy."""
    if accessor.layout.sample_rate_hz != 16_000:
        raise QuantizationCandidateError("calibration requires native 16 kHz WAV")
    if not 0 <= segment_start_sample < segment_end_sample <= accessor.layout.frame_count:
        raise QuantizationCandidateError("invalid calibration segment bounds")
    stride = round(stride_sec * 16_000)
    if stride <= 0:
        raise QuantizationCandidateError("candidate stride must be positive")
    candidates: list[SampleWindow] = []
    for duration_sec in durations_sec:
        if not math.isfinite(duration_sec) or duration_sec < 0.025 or duration_sec > 3.0:
            raise QuantizationCandidateError("durations must be finite and within [0.025, 3.0] seconds")
        duration = round(duration_sec * 16_000)
        for start in _window_starts(segment_start_sample, segment_end_sample, duration, stride):
            provisional = SampleWindow(start, start + duration, duration, 0)
            samples = _read_window(accessor, provisional, np)
            candidates.append(
                SampleWindow(start, start + duration, duration, _mean_square_pcm(samples, np))
            )
    if not candidates:
        raise QuantizationCandidateError("no calibration candidates fit inside the selected split")
    return tuple(candidates)


def _even_energy_indices(count: int, take: int) -> tuple[int, ...]:
    if take <= 0 or count <= 0:
        return ()
    if take >= count:
        return tuple(range(count))
    if take == 1:
        return (count // 2,)
    return tuple(round(index * (count - 1) / (take - 1)) for index in range(take))


def select_duration_energy_diverse(
    candidates: Sequence[SampleWindow], *, max_windows: int
) -> tuple[SampleWindow, ...]:
    """Select equal duration budgets and evenly spaced energy ranks."""
    if max_windows <= 0:
        raise QuantizationCandidateError("max_windows must be positive")
    grouped: dict[int, list[SampleWindow]] = {}
    for window in candidates:
        grouped.setdefault(window.duration_samples, []).append(window)
    durations = sorted(grouped)
    base, remainder = divmod(max_windows, len(durations))
    selected: list[SampleWindow] = []
    selected_keys: set[tuple[int, int]] = set()
    for ordinal, duration in enumerate(durations):
        budget = base + int(ordinal < remainder)
        ordered = sorted(
            grouped[duration],
            key=lambda window: (window.mean_square_pcm, window.start_sample),
        )
        for index in _even_energy_indices(len(ordered), budget):
            window = ordered[index]
            key = (window.start_sample, window.end_sample)
            if key not in selected_keys:
                selected.append(window)
                selected_keys.add(key)
    if len(selected) < max_windows:
        remaining = sorted(
            (
                window
                for window in candidates
                if (window.start_sample, window.end_sample) not in selected_keys
            ),
            key=lambda window: (
                window.duration_samples,
                window.mean_square_pcm,
                window.start_sample,
            ),
        )
        selected.extend(remaining[: max_windows - len(selected)])
    selected = selected[:max_windows]
    if not selected:
        raise QuantizationCandidateError("duration/energy selection produced no windows")
    return tuple(
        sorted(
            selected,
            key=lambda window: (
                window.duration_samples,
                window.mean_square_pcm,
                window.start_sample,
            ),
        )
    )


def selection_digest(windows: Sequence[SampleWindow]) -> str:
    payload = json.dumps(
        [asdict(window) for window in windows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_features(
    accessor: WavPcmAccessor, windows: Sequence[SampleWindow], *, np: Any
) -> tuple[tuple[Any, ...], str]:
    features: list[Any] = []
    digest = hashlib.sha256()
    for window in windows:
        samples = _read_window(accessor, window, np)
        feature = np.asarray(kaldi_native_fbank_features(samples), dtype=np.float32)
        if feature.ndim != 2 or feature.shape[1] != 80 or not np.isfinite(feature).all():
            raise QuantizationCandidateError("strict FBank produced an invalid calibration tensor")
        contiguous = np.ascontiguousarray(feature[np.newaxis, :, :], dtype=np.float32)
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
        features.append(contiguous)
    if not features:
        raise QuantizationCandidateError("no feature tensors were built")
    return tuple(features), digest.hexdigest()


class ArrayCalibrationReader:
    """Minimal deterministic CalibrationDataReader for one-input WeSpeaker."""

    def __init__(self, features: Sequence[Any]):
        self._features = tuple(features)
        self._index = 0

    def get_next(self) -> Mapping[str, Any] | None:
        if self._index >= len(self._features):
            return None
        feature = self._features[self._index]
        self._index += 1
        return {"feats": feature}

    def rewind(self) -> None:
        self._index = 0


def _load_tooling_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise QuantizationCandidateError("tooling manifest is missing or is a symlink")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuantizationCandidateError("tooling manifest is invalid") from exc
    if manifest.get("production_approved") is not False or manifest.get("runtime_dependency") is not False:
        raise QuantizationCandidateError("tooling manifest must be development-only and build-time-only")
    artifact_root = path.parent.parent
    installed: dict[str, str] = {}
    for package in manifest.get("packages", []):
        name, expected_version = package.get("name"), package.get("version")
        relative, expected_hash = package.get("path"), package.get("sha256")
        if not all(isinstance(value, str) and value for value in (name, expected_version, relative, expected_hash)):
            raise QuantizationCandidateError("tooling manifest package entry is incomplete")
        package_path = artifact_root / relative
        require_sha256(package_path, expected_hash, f"tooling wheel {name}")
        try:
            actual_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise QuantizationCandidateError(f"tooling package {name} is not installed") from exc
        if actual_version != expected_version:
            raise QuantizationCandidateError(
                f"tooling package {name} version mismatch: {actual_version} != {expected_version}"
            )
        installed[name] = actual_version
    return manifest, sha256_path(path)


def _check_output_targets(source: Path, outputs: Sequence[Path], *, overwrite: bool) -> None:
    source_resolved = source.resolve()
    seen: set[Path] = set()
    for output in outputs:
        resolved = output.resolve()
        if resolved == source_resolved:
            raise QuantizationCandidateError("derived output must not replace the FP32 source model")
        if resolved in seen:
            raise QuantizationCandidateError("derived output paths must be distinct")
        seen.add(resolved)
        if output.exists() and not overwrite:
            raise QuantizationCandidateError(f"derived output already exists: {output.name}")
        output.parent.mkdir(parents=True, exist_ok=True)


def _temp_path(target: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    path = Path(handle.name)
    handle.close()
    path.unlink(missing_ok=True)
    return path


def first_conv_last_gemm_node_names(model: Any) -> tuple[str, str]:
    convs = [node for node in model.graph.node if node.op_type == "Conv"]
    gemms = [node for node in model.graph.node if node.op_type == "Gemm"]
    if not convs or not gemms:
        raise QuantizationCandidateError("mixed Q1b requires at least one Conv and one Gemm")
    first_conv, last_gemm = convs[0], gemms[-1]
    if not first_conv.name or not last_gemm.name:
        raise QuantizationCandidateError("Q1b excluded nodes must have stable non-empty names")
    return first_conv.name, last_gemm.name


def _verify_excluded_weights_fp32(model: Any, node_names: Sequence[str]) -> None:
    import onnx

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    nodes = {node.name: node for node in model.graph.node}
    for name in node_names:
        node = nodes.get(name)
        if node is None or node.op_type not in {"Conv", "Gemm"} or len(node.input) < 2:
            raise QuantizationCandidateError(f"excluded Q1b node is missing or invalid: {name}")
        weight = initializers.get(node.input[1])
        if weight is None or weight.data_type != onnx.TensorProto.FLOAT:
            raise QuantizationCandidateError(f"excluded Q1b node weight is not FP32: {name}")


def build_static_qdq_candidate(
    fp32_model: Path,
    preprocessed_output: Path,
    int8_output: Path,
    calibration_features: Sequence[Any],
    *,
    overwrite: bool = False,
    keep_first_conv_last_gemm_fp32: bool = False,
) -> dict[str, Any]:
    """Preprocess separately, then emit static QDQ S8S8 Conv+Gemm."""
    import onnx
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    _check_output_targets(fp32_model, (preprocessed_output, int8_output), overwrite=overwrite)
    pre_tmp, int8_tmp = _temp_path(preprocessed_output), _temp_path(int8_output)
    try:
        quant_pre_process(
            input_model=fp32_model,
            output_model_path=pre_tmp,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            save_as_external_data=False,
        )
        preprocessed_model = onnx.load(pre_tmp)
        onnx.checker.check_model(preprocessed_model)
        nodes_to_exclude: list[str] = []
        if keep_first_conv_last_gemm_fp32:
            nodes_to_exclude = list(first_conv_last_gemm_node_names(preprocessed_model))
        quantize_static(
            model_input=pre_tmp,
            model_output=int8_tmp,
            calibration_data_reader=ArrayCalibrationReader(calibration_features),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv", "Gemm"],
            nodes_to_exclude=nodes_to_exclude,
            per_channel=True,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"],
        )
        model = onnx.load(int8_tmp)
        onnx.checker.check_model(model)
        summary = inspect_quantized_graph(model)
        if not summary["input_output_fp32"]:
            raise QuantizationCandidateError("quantized graph changed public input/output types")
        if summary["qdq_nodes"] <= 0 or summary["int8_initializer_count"] <= 0:
            raise QuantizationCandidateError("quantized graph contains no QDQ INT8 weights")
        if summary["integer_operator_nodes"]:
            raise QuantizationCandidateError("candidate is not a pure QDQ representation")
        if nodes_to_exclude:
            _verify_excluded_weights_fp32(model, nodes_to_exclude)
        summary["excluded_fp32_nodes"] = nodes_to_exclude
        os.replace(pre_tmp, preprocessed_output)
        os.replace(int8_tmp, int8_output)
        return summary
    finally:
        pre_tmp.unlink(missing_ok=True)
        int8_tmp.unlink(missing_ok=True)


def inspect_quantized_graph(model: Any) -> dict[str, Any]:
    import onnx

    node_counts = Counter(node.op_type for node in model.graph.node)
    initializer_counts = Counter(initializer.data_type for initializer in model.graph.initializer)
    io_values = list(model.graph.input) + list(model.graph.output)
    io_fp32 = bool(io_values) and all(
        value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for value in io_values
    )
    integer_ops = sorted(
        op
        for op in ("ConvInteger", "MatMulInteger", "QLinearConv", "QGemm", "QLinearMatMul")
        if node_counts[op]
    )
    per_channel_scales = 0
    for initializer in model.graph.initializer:
        if initializer.name.endswith("_scale") and len(initializer.dims) == 1 and initializer.dims[0] > 1:
            per_channel_scales += 1
    return {
        "node_counts": dict(sorted(node_counts.items())),
        "qdq_nodes": node_counts["QuantizeLinear"] + node_counts["DequantizeLinear"],
        "conv_nodes": node_counts["Conv"],
        "gemm_nodes": node_counts["Gemm"],
        "int8_initializer_count": initializer_counts[onnx.TensorProto.INT8],
        "uint8_initializer_count": initializer_counts[onnx.TensorProto.UINT8],
        "per_channel_scale_count": per_channel_scales,
        "integer_operator_nodes": integer_ops,
        "input_output_fp32": io_fp32,
    }


def _session_options(ort: Any, *, threads: int) -> Any:
    if threads != 1:
        raise QuantizationCandidateError("Q1 evaluation requires exactly one ORT thread")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def _create_cpu_session(path: Path, *, threads: int, ort: Any) -> Any:
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise QuantizationCandidateError("CPUExecutionProvider is unavailable")
    session = ort.InferenceSession(
        str(path),
        sess_options=_session_options(ort, threads=threads),
        providers=["CPUExecutionProvider"],
    )
    if tuple(session.get_providers()) != ("CPUExecutionProvider",):
        raise QuantizationCandidateError("runtime session is not CPU-only")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "feats" or inputs[0].type != "tensor(float)":
        raise QuantizationCandidateError("runtime input contract is not float32 feats")
    if len(outputs) != 1 or outputs[0].name != "embs" or outputs[0].type != "tensor(float)":
        raise QuantizationCandidateError("runtime output contract is not float32 embs")
    return session


def _l2_normalize(matrix: Any, np: Any) -> Any:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise QuantizationCandidateError("embedding matrix is invalid")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise QuantizationCandidateError("embedding matrix contains zero vectors")
    return values / norms


def _average_ranks(values: Any, np: Any) -> Any:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        ranks[order[position:end]] = rank
        position = end
    return ranks


def spearman(values_a: Any, values_b: Any, *, np: Any) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise QuantizationCandidateError("Spearman correlation requires equal non-trivial arrays")
    ranks_a, ranks_b = _average_ranks(values_a, np), _average_ranks(values_b, np)
    centered_a, centered_b = ranks_a - ranks_a.mean(), ranks_b - ranks_b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator <= 0.0:
        raise QuantizationCandidateError("Spearman correlation is undefined for constant ranks")
    return float(np.dot(centered_a, centered_b) / denominator)


def _run_embeddings(session: Any, features: Sequence[Any], *, np: Any) -> tuple[Any, list[float]]:
    # Warm the graph/session without including first-run packing in parity timing.
    session.run(["embs"], {"feats": features[0]})
    vectors: list[Any] = []
    timings_ms: list[float] = []
    for feature in features:
        started = time.perf_counter()
        raw = session.run(["embs"], {"feats": feature})[0]
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        array = np.asarray(raw)
        if array.ndim != 2 or array.shape[0] != 1:
            raise QuantizationCandidateError("runtime returned an invalid embedding shape")
        vectors.append(array[0])
    return _l2_normalize(np.stack(vectors), np), timings_ms


def percentile(values: Sequence[float] | Any, quantile: float, *, np: Any) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def evaluate_embedding_parity(
    fp32_model: Path,
    int8_model: Path,
    holdout_features: Sequence[Any],
    *,
    threads: int,
    gates: ParityGates,
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    fp32_session = _create_cpu_session(fp32_model, threads=threads, ort=ort)
    int8_session = _create_cpu_session(int8_model, threads=threads, ort=ort)
    fp32, fp32_times = _run_embeddings(fp32_session, holdout_features, np=np)
    int8, int8_times = _run_embeddings(int8_session, holdout_features, np=np)
    if fp32.shape != int8.shape:
        raise QuantizationCandidateError("FP32 and INT8 embedding shapes differ")
    embedding_cosines = np.sum(fp32 * int8, axis=1)
    upper = np.triu_indices(fp32.shape[0], k=1)
    fp32_scores = (fp32 @ fp32.T)[upper]
    int8_scores = (int8 @ int8.T)[upper]
    absolute_error = np.abs(fp32_scores - int8_scores)
    metrics = {
        "holdout_region_count": int(fp32.shape[0]),
        "embedding_dimension": int(fp32.shape[1]),
        "embedding_cosine_median": percentile(embedding_cosines, 0.50, np=np),
        "embedding_cosine_p01": percentile(embedding_cosines, 0.01, np=np),
        "embedding_cosine_min": float(np.min(embedding_cosines)),
        "pairwise_score_count": int(fp32_scores.size),
        "pairwise_score_spearman": spearman(fp32_scores, int8_scores, np=np),
        "pairwise_abs_error_p95": percentile(absolute_error, 0.95, np=np),
        "pairwise_abs_error_max": float(np.max(absolute_error)),
        "fp32_inference_ms_p50": percentile(fp32_times, 0.50, np=np),
        "fp32_inference_ms_p95": percentile(fp32_times, 0.95, np=np),
        "int8_inference_ms_p50": percentile(int8_times, 0.50, np=np),
        "int8_inference_ms_p95": percentile(int8_times, 0.95, np=np),
    }
    metrics["int8_over_fp32_inference_p50"] = (
        metrics["int8_inference_ms_p50"] / metrics["fp32_inference_ms_p50"]
    )
    checks = {
        "embedding_cosine_median": (
            metrics["embedding_cosine_median"] >= gates.embedding_cosine_median_min
        ),
        "embedding_cosine_p01": metrics["embedding_cosine_p01"] >= gates.embedding_cosine_p01_min,
        "pairwise_score_spearman": (
            metrics["pairwise_score_spearman"] >= gates.pairwise_spearman_min
        ),
        "pairwise_abs_error_p95": (
            metrics["pairwise_abs_error_p95"] <= gates.pairwise_abs_error_p95_max
        ),
        "model_bytes": int8_model.stat().st_size <= gates.model_bytes_max,
    }
    return {
        "metrics": metrics,
        "gates": asdict(gates),
        "checks": checks,
        "passed": all(checks.values()),
        "providers": ["CPUExecutionProvider"],
        "runtime_network_imports": "none; verified by project static zero-network scan",
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verify_q1_selection_reuse(
    q1_metadata_path: Path,
    *,
    audio_sha256: str,
    split_sample: int,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless Q1b reuses Q1's exact unlabeled sample contract."""
    if not q1_metadata_path.is_file() or q1_metadata_path.is_symlink():
        raise QuantizationCandidateError("Q1 metadata is missing or is a symlink")
    try:
        q1 = json.loads(q1_metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuantizationCandidateError("Q1 metadata is invalid") from exc
    if q1.get("candidate_id") != "wespeaker-voxceleb-resnet34-qdq-s8s8-pc-minmax-dev-20260826":
        raise QuantizationCandidateError("Q1 metadata candidate identity mismatch")
    if q1.get("production_approved") is not False:
        raise QuantizationCandidateError("Q1 metadata must remain development-only")
    if q1.get("source", {}).get("approved_audio_sha256") != audio_sha256:
        raise QuantizationCandidateError("Q1b audio does not match Q1")
    if q1.get("split", {}).get("split_sample") != split_sample:
        raise QuantizationCandidateError("Q1b 60/40 split does not match Q1")
    keys = (
        "algorithm",
        "durations_sec",
        "candidate_stride_sec",
        "calibration_candidate_count",
        "calibration_window_count",
        "calibration_selection_sha256",
        "calibration_feature_sha256",
        "holdout_candidate_count",
        "holdout_window_count",
        "holdout_selection_sha256",
        "holdout_feature_sha256",
    )
    q1_selection = q1.get("selection", {})
    mismatches = [key for key in keys if q1_selection.get(key) != selection.get(key)]
    if mismatches:
        raise QuantizationCandidateError(
            "Q1b selection/feature contract differs from Q1: " + ",".join(mismatches)
        )
    q1_artifact = q1.get("artifacts", {}).get("int8_model", {})
    q1_relative = q1_artifact.get("relative_path")
    q1_hash = q1_artifact.get("sha256")
    if not isinstance(q1_relative, str) or not isinstance(q1_hash, str):
        raise QuantizationCandidateError("Q1 artifact provenance is incomplete")
    q1_model_path = q1_metadata_path.parents[1] / q1_relative
    require_sha256(q1_model_path, q1_hash, "preserved Q1 artifact")
    return {
        "q1_metadata_sha256": sha256_path(q1_metadata_path),
        "q1_candidate_sha256": q1_hash,
        "calibration_selection_sha256": selection["calibration_selection_sha256"],
        "calibration_feature_sha256": selection["calibration_feature_sha256"],
        "holdout_selection_sha256": selection["holdout_selection_sha256"],
        "holdout_feature_sha256": selection["holdout_feature_sha256"],
        "exact_reuse_verified": True,
    }


def _parse_durations(value: str) -> tuple[float, ...]:
    try:
        durations = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("durations must be comma-separated floats") from exc
    if not durations:
        raise argparse.ArgumentTypeError("at least one duration is required")
    return durations


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnx
    import onnxruntime as ort

    fp32_before = require_sha256(args.fp32_model, args.fp32_sha256, "FP32 model")
    audio_hash = require_sha256(args.audio, args.audio_sha256, "approved calibration audio")
    if args.keep_first_conv_last_gemm_fp32 and (
        args.int8_output.resolve() == DEFAULT_INT8_MODEL.resolve()
        or args.metadata_output.resolve() == DEFAULT_METADATA.resolve()
    ):
        raise QuantizationCandidateError("Q1b must use distinct artifact and metadata paths")
    if args.metadata_output.exists() and not args.overwrite:
        raise QuantizationCandidateError(
            f"derived output already exists: {args.metadata_output.name}"
        )
    tooling, tooling_manifest_hash = _load_tooling_manifest(args.tooling_manifest)
    accessor = WavPcmAccessor(args.audio)
    layout = accessor.layout
    if (layout.sample_width_bytes, layout.channel_count, layout.sample_rate_hz) != (2, 1, 16_000):
        raise QuantizationCandidateError("approved calibration audio must be PCM16 mono 16 kHz")
    split_sample = round(layout.frame_count * 0.60)
    if split_sample <= 0 or split_sample >= layout.frame_count:
        raise QuantizationCandidateError("60/40 split is empty")
    calibration_candidates = enumerate_energy_candidates(
        accessor,
        segment_start_sample=0,
        segment_end_sample=split_sample,
        durations_sec=args.durations_sec,
        stride_sec=args.candidate_stride_sec,
        np=np,
    )
    holdout_candidates = enumerate_energy_candidates(
        accessor,
        segment_start_sample=split_sample,
        segment_end_sample=layout.frame_count,
        durations_sec=args.durations_sec,
        stride_sec=args.candidate_stride_sec,
        np=np,
    )
    calibration_windows = select_duration_energy_diverse(
        calibration_candidates, max_windows=args.calibration_count
    )
    holdout_windows = select_duration_energy_diverse(
        holdout_candidates, max_windows=args.holdout_count
    )
    if any(window.end_sample > split_sample for window in calibration_windows):
        raise QuantizationCandidateError("calibration window leaked into holdout")
    if any(window.start_sample < split_sample for window in holdout_windows):
        raise QuantizationCandidateError("holdout window leaked into calibration")
    calibration_features, calibration_feature_digest = build_features(
        accessor, calibration_windows, np=np
    )
    holdout_features, holdout_feature_digest = build_features(
        accessor, holdout_windows, np=np
    )
    selection_metadata = {
        "algorithm": "equal-duration-budget-even-energy-ranks-v1",
        "durations_sec": list(args.durations_sec),
        "candidate_stride_sec": args.candidate_stride_sec,
        "calibration_candidate_count": len(calibration_candidates),
        "calibration_window_count": len(calibration_windows),
        "calibration_selection_sha256": selection_digest(calibration_windows),
        "calibration_feature_sha256": calibration_feature_digest,
        "holdout_candidate_count": len(holdout_candidates),
        "holdout_window_count": len(holdout_windows),
        "holdout_selection_sha256": selection_digest(holdout_windows),
        "holdout_feature_sha256": holdout_feature_digest,
        "raw_audio_retained": False,
        "features_retained": False,
        "embeddings_retained": False,
    }
    q1_reuse = None
    if args.keep_first_conv_last_gemm_fp32:
        q1_reuse = verify_q1_selection_reuse(
            args.q1_metadata,
            audio_sha256=audio_hash,
            split_sample=split_sample,
            selection=selection_metadata,
        )
    graph_summary = build_static_qdq_candidate(
        args.fp32_model,
        args.preprocessed_output,
        args.int8_output,
        calibration_features,
        overwrite=args.overwrite,
        keep_first_conv_last_gemm_fp32=args.keep_first_conv_last_gemm_fp32,
    )
    fp32_after = require_sha256(args.fp32_model, args.fp32_sha256, "FP32 model after build")
    if fp32_before != fp32_after:
        raise QuantizationCandidateError("FP32 source model changed during quantization")
    parity = evaluate_embedding_parity(
        args.fp32_model,
        args.int8_output,
        holdout_features,
        threads=args.threads,
        gates=ParityGates(),
    )
    int8_hash = sha256_path(args.int8_output)
    preprocessed_hash = sha256_path(args.preprocessed_output)
    duration_sec = layout.frame_count / layout.sample_rate_hz
    installed_versions = {
        name: importlib.metadata.version(name)
        for name in ("onnx", "onnxruntime", "numpy", "protobuf", "kaldi-native-fbank")
    }
    is_q1b = args.keep_first_conv_last_gemm_fp32
    metadata: dict[str, Any] = {
        "schema": "sddiar_wespeaker_static_int8_candidate_v1",
        "candidate_id": (
            "wespeaker-voxceleb-resnet34-qdq-s8s8-pc-minmax-q1b-firstconv-lastgemm-fp32-dev-20260826"
            if is_q1b
            else "wespeaker-voxceleb-resnet34-qdq-s8s8-pc-minmax-dev-20260826"
        ),
        "experiment": "Q1b" if is_q1b else "Q1",
        "status": "PARITY_PASS" if parity["passed"] else "PARITY_REJECTED",
        "production_approved": False,
        "development_only": True,
        "source": {
            "fp32_model_sha256": fp32_before,
            "fp32_model_bytes": args.fp32_model.stat().st_size,
            "approved_audio_sha256": audio_hash,
            "audio_duration_sec": round(duration_sec, 6),
            "sample_rate_hz": layout.sample_rate_hz,
            "channel_count": layout.channel_count,
            "sample_width_bytes": layout.sample_width_bytes,
            "source_path": "omitted",
            "speaker_labels_used": False,
        },
        "split": {
            "calibration_first_ratio": 0.60,
            "holdout_last_ratio": 0.40,
            "split_sample": split_sample,
            "calibration_end_sec": round(split_sample / layout.sample_rate_hz, 6),
            "last40_independence_status": (
                "NOT_INDEPENDENT_AFTER_Q1_REVIEW"
                if is_q1b
                else "DEVELOPMENT_SINGLE_FILE_HOLDOUT"
            ),
            "release_claim_allowed": False,
            "fresh_file_level_holdout_required": True,
        },
        "selection": selection_metadata,
        "q1_exact_reuse": q1_reuse,
        "quantization": {
            "preprocessing": "onnxruntime.quantization.quant_pre_process separate pass",
            "format": "QDQ",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "reduce_range": False,
            "calibration_method": "MinMax",
            "op_types": ["Conv", "Gemm"],
            "kept_fp32_nodes": graph_summary["excluded_fp32_nodes"],
            "mixed_precision_contingency": is_q1b,
            "input_output_type": "float32",
            "calibration_provider": "CPUExecutionProvider",
        },
        "artifacts": {
            "preprocessed_model": {
                "relative_path": str(args.preprocessed_output.relative_to(ROOT / "artifacts/dev")),
                "bytes": args.preprocessed_output.stat().st_size,
                "sha256": preprocessed_hash,
            },
            "int8_model": {
                "relative_path": str(args.int8_output.relative_to(ROOT / "artifacts/dev")),
                "bytes": args.int8_output.stat().st_size,
                "sha256": int8_hash,
                "size_ratio_vs_fp32": (
                    args.int8_output.stat().st_size / args.fp32_model.stat().st_size
                ),
            },
        },
        "graph": graph_summary,
        "parity": parity,
        "runtime_validation": {
            "execution_provider": "CPUExecutionProvider",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "spinning": False,
            "network_imports": "none; verified by project static zero-network scan",
            "docker_network_none_required_for_release_gate": True,
            "onnx_checker_passed": True,
        },
        "tooling": {
            "manifest_sha256": tooling_manifest_hash,
            "production_approved": tooling["production_approved"],
            "runtime_dependency": tooling["runtime_dependency"],
            "installed_versions": installed_versions,
            "packages": tooling["packages"],
        },
        "limits": [
            "One approved recording was split 60/40; this is a development parity check, not release calibration.",
            "No speaker labels, transcript text, or diarization reference were used for quantization or embedding parity.",
            "Intel Xeon Gold 6230R CPU performance remains a separate gate.",
            (
                "The last40 region set was already reviewed in Q1 and is not an independent holdout for Q1b; "
                "a fresh file-level holdout is mandatory for any release claim."
                if is_q1b
                else "A fresh independent file-level holdout is mandatory for any release claim."
            ),
        ],
    }
    # Validate the final model once more after it has reached its durable path.
    onnx.checker.check_model(onnx.load(args.int8_output))
    if tuple(
        ort.InferenceSession(
            str(args.int8_output),
            sess_options=_session_options(ort, threads=1),
            providers=["CPUExecutionProvider"],
        ).get_providers()
    ) != ("CPUExecutionProvider",):
        raise QuantizationCandidateError("durable candidate does not load CPU-only")
    _atomic_json(args.metadata_output, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build static QDQ S8S8 WeSpeaker ResNet34 and evaluate unlabeled 60/40 parity"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_FP32_MODEL)
    parser.add_argument("--fp32-sha256", default=FP32_MODEL_SHA256)
    parser.add_argument("--preprocessed-output", type=Path, default=DEFAULT_PREPROCESSED_MODEL)
    parser.add_argument("--int8-output", type=Path, default=DEFAULT_INT8_MODEL)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--tooling-manifest", type=Path, default=DEFAULT_TOOLING_MANIFEST)
    parser.add_argument("--q1-metadata", type=Path, default=DEFAULT_Q1_METADATA)
    parser.add_argument("--durations-sec", type=_parse_durations, default=DEFAULT_DURATIONS_SEC)
    parser.add_argument("--candidate-stride-sec", type=float, default=4.0)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--holdout-count", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--keep-first-conv-last-gemm-fp32",
        action="store_true",
        help="Q1b only: exclude the graph's first Conv and final Gemm from QDQ quantization",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.keep_first_conv_last_gemm_fp32:
        if args.preprocessed_output == DEFAULT_PREPROCESSED_MODEL:
            args.preprocessed_output = DEFAULT_Q1B_PREPROCESSED_MODEL
        if args.int8_output == DEFAULT_INT8_MODEL:
            args.int8_output = DEFAULT_Q1B_INT8_MODEL
        if args.metadata_output == DEFAULT_METADATA:
            args.metadata_output = DEFAULT_Q1B_METADATA
    metadata = run(args)
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "candidate_sha256": metadata["artifacts"]["int8_model"]["sha256"],
                "candidate_bytes": metadata["artifacts"]["int8_model"]["bytes"],
                "parity": metadata["parity"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0 if metadata["parity"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
