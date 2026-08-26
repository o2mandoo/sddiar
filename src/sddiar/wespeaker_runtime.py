"""CPU WeSpeaker inference seam.

The default feature path delegates to the explicitly provisioned
``kaldi_native_fbank`` package and matches the public WeSpeaker ``infer_onnx``
contract.  The NumPy implementation remains available only as a named
development approximation.  Models are never downloaded here: callers must
provide a hash-verified local artifact.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import EmbeddingRegion, EmbeddingResult
from .errors import ContractValidationError, ModelHashMismatch, ModelNotFound, ModelRuntimeIncompatible
from .model_pack import VerifiedArtifact
from .ort_cpu import create_ort_session

DEVELOPMENT_APPROXIMATION_TAG = "wespeaker-logmel-fbank-development-approximation"
STRICT_NATIVE_FBANK_TAG = "wespeaker-kaldi-native-fbank"


@dataclass(frozen=True, slots=True)
class WeSpeakerFeatureConfig:
    sample_rate_hz: int = 16_000
    feature_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    dither: float = 0.0
    preemphasis: float = 0.97
    low_hz: float = 20.0
    high_hz: float | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000 or self.feature_bins != 80:
            raise ContractValidationError("WeSpeaker contract requires 16 kHz and 80 bins")
        if self.frame_length_ms <= 0 or self.frame_shift_ms <= 0 or self.dither != 0.0:
            raise ContractValidationError("invalid WeSpeaker feature timing/dither contract")


def _hz_to_mel(hz: float) -> float:
    return 1127.0 * math.log1p(hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * math.expm1(mel / 1127.0)


def _load_kaldi_native_fbank() -> Any:
    """Import the approved native frontend through one testable seam."""
    try:
        import kaldi_native_fbank as knf  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised through the seam
        raise ModelRuntimeIncompatible(
            "kaldi_native_fbank is required for strict WeSpeaker FBank inference; "
            "no approximation fallback is permitted"
        ) from exc
    return knf


def kaldi_native_fbank_features(
    samples: Sequence[float],
    *,
    config: WeSpeakerFeatureConfig | None = None,
    module: Any | None = None,
) -> Any:
    """Return strict WeSpeaker/Kaldi-compatible ``float32[frames, 80]``.

    ``samples`` are canonical normalized mono samples in ``[-1, 1]``.  The
    official WeSpeaker ONNX frontend scales them to PCM-like amplitude before
    calling Kaldi FBank.  ``kaldi_native_fbank`` is deliberately an explicit
    runtime dependency; missing or incompatible native bindings are surfaced
    as a typed runtime error instead of silently using the approximation.
    """
    cfg = config or WeSpeakerFeatureConfig()
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when optional dep absent
        raise ModelRuntimeIncompatible("numpy is required for native WeSpeaker FBank inference") from exc
    # ``samples`` may be the bounded PCM decoder's ndarray.  Let NumPy perform
    # the conversion in one operation instead of iterating every sample in
    # Python; tuples and other existing sequence callers remain supported.
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ContractValidationError("audio region must be a finite, non-empty mono sequence")
    frame_len = round(cfg.sample_rate_hz * cfg.frame_length_ms / 1000.0)
    if x.size < frame_len:
        raise ContractValidationError(
            f"audio region is shorter than one strict FBank frame ({frame_len} samples)"
        )
    knf = module or _load_kaldi_native_fbank()
    try:
        opts = knf.FbankOptions()
        frame_opts = opts.frame_opts
        mel_opts = opts.mel_opts
        frame_opts.samp_freq = cfg.sample_rate_hz
        frame_opts.dither = cfg.dither
        frame_opts.window_type = "hamming"
        frame_opts.frame_shift_ms = float(cfg.frame_shift_ms)
        frame_opts.frame_length_ms = float(cfg.frame_length_ms)
        frame_opts.snip_edges = True
        frame_opts.preemph_coeff = float(cfg.preemphasis)
        frame_opts.remove_dc_offset = True
        mel_opts.num_bins = cfg.feature_bins
        mel_opts.low_freq = float(cfg.low_hz)
        mel_opts.high_freq = 0.0 if cfg.high_hz is None else float(cfg.high_hz)
        mel_opts.debug_mel = False
        opts.energy_floor = 0.0
        opts.use_energy = False
        opts.use_log_fbank = True
        opts.use_power = True
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(cfg.sample_rate_hz, (x * (1 << 15)).tolist())
        frame_count = int(fbank.num_frames_ready)
        if frame_count <= 0:
            raise ContractValidationError("strict FBank produced no complete frames")
        features = np.asarray(
            [fbank.get_frame(index) for index in range(frame_count)], dtype=np.float32
        )
    except ContractValidationError:
        raise
    except Exception as exc:
        raise ModelRuntimeIncompatible("kaldi_native_fbank strict FBank extraction failed") from exc
    if features.ndim != 2 or features.shape[1] != cfg.feature_bins:
        raise ModelRuntimeIncompatible("kaldi_native_fbank returned an invalid FBank shape")
    # Official WeSpeaker infer_onnx applies utterance CMN without variance normalization.
    features -= features.mean(axis=0, keepdims=True)
    return features


def wespeaker_logmel_features_approx(
    samples: Sequence[float], *, config: WeSpeakerFeatureConfig | None = None
) -> Any:
    """Return ``float32[frames, 80]`` features for a 16 kHz mono region.

    This dependency is optional and imported lazily.  The implementation uses
    NumPy FFT/matrix operations, no random dither, and reflect padding for a
    short final region so every non-empty region has at least one frame.
    """
    cfg = config or WeSpeakerFeatureConfig()
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when optional dep absent
        raise ModelRuntimeIncompatible("numpy is required for WeSpeaker CPU inference") from exc
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ContractValidationError("audio region must be a finite, non-empty mono sequence")
    frame_len = round(cfg.sample_rate_hz * cfg.frame_length_ms / 1000.0)
    frame_shift = round(cfg.sample_rate_hz * cfg.frame_shift_ms / 1000.0)
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size), mode="reflect" if x.size > 1 else "edge")
    count = 1 + max(0, (x.size - frame_len) // frame_shift)
    padded = (count - 1) * frame_shift + frame_len
    if x.size < padded:
        x = np.pad(x, (0, padded - x.size), mode="reflect")
    frames = np.stack([x[i * frame_shift:i * frame_shift + frame_len] for i in range(count)])
    frames = frames - cfg.preemphasis * np.concatenate((frames[:, :1], frames[:, :-1]), axis=1)
    frames *= np.hamming(frame_len).astype(np.float32)
    nfft = 1 << (frame_len - 1).bit_length()
    power = (np.abs(np.fft.rfft(frames, n=nfft, axis=1)) ** 2) / nfft
    high = cfg.high_hz or cfg.sample_rate_hz / 2.0
    edges = np.linspace(_hz_to_mel(cfg.low_hz), _hz_to_mel(high), cfg.feature_bins + 2)
    bins = np.floor((nfft + 1) * np.array([_mel_to_hz(v) for v in edges]) / cfg.sample_rate_hz).astype(int)
    bank = np.zeros((cfg.feature_bins, nfft // 2 + 1), dtype=np.float32)
    for i in range(cfg.feature_bins):
        left, center, right = bins[i:i + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(max(0, left), min(center, bank.shape[1])):
            bank[i, k] = (k - left) / (center - left)
        for k in range(max(0, center), min(right, bank.shape[1])):
            bank[i, k] = (right - k) / (right - center)
    features = np.log(np.maximum(power @ bank.T, 1e-10)).astype(np.float32)
    # Upstream infer_onnx applies utterance CMN without variance normalization.
    features -= features.mean(axis=0, keepdims=True)
    return features


class WeSpeakerCpuEmbeddingBackend:
    """Bounded CPU ONNX adapter for the inspected ``feats``/``embs`` graph."""

    development_approximation_tag = DEVELOPMENT_APPROXIMATION_TAG

    def __init__(self, artifact: VerifiedArtifact, *, model_id: str = "", runtime_contract: Mapping[str, Any] | None = None,
                 max_batch_regions: int = 8, session: Any = None,
                 feature_config: WeSpeakerFeatureConfig | None = None,
                 audio_provider: Callable[[EmbeddingRegion], Sequence[float]] | None = None,
                 feature_mode: str = "kaldi_native", feature_extractor: Callable[..., Any] | None = None,
                 native_fbank_module: Any | None = None,
                 subsegment_windows: bool = False, window_frames: int = 150,
                 period_frames: int = 75,
                 subsegment_region_ids: Sequence[str] = (),
                 exact_length_batching: bool = False,
                 batch_buffer_regions: int = 32) -> None:
        if not isinstance(artifact, VerifiedArtifact) or artifact.role not in ("embedding", "speaker_embedding"):
            raise ModelRuntimeIncompatible("WeSpeaker backend requires a verified speaker-embedding artifact")
        path = Path(artifact.path)
        if not path.is_file():
            raise ModelNotFound("verified ONNX artifact is missing")
        if path.suffix.lower() != ".onnx":
            raise ModelRuntimeIncompatible("speaker embedding artifact must be ONNX")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise ModelHashMismatch("verified ONNX artifact hash changed")
        if type(max_batch_regions) is not int or max_batch_regions <= 0:
            raise ContractValidationError("max_batch_regions must be positive")
        if type(window_frames) is not int or window_frames <= 0:
            raise ContractValidationError("window_frames must be a positive integer")
        if type(period_frames) is not int or period_frames <= 0:
            raise ContractValidationError("period_frames must be a positive integer")
        if type(exact_length_batching) is not bool:
            raise ContractValidationError("exact_length_batching must be boolean")
        if type(batch_buffer_regions) is not int or batch_buffer_regions <= 0:
            raise ContractValidationError("batch_buffer_regions must be a positive integer")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise ModelRuntimeIncompatible("numpy is required for WeSpeaker CPU inference") from exc
        if session is None:
            try:
                session = create_ort_session(path)
            except Exception as exc:
                raise ModelRuntimeIncompatible("ONNX session creation failed") from exc
        providers = tuple(session.get_providers())
        if providers != ("CPUExecutionProvider",):
            raise ModelRuntimeIncompatible("ONNX session is not CPU-only")
        self._np, self._session, self._max_batch = np, session, max_batch_regions
        self.model_id, self.model_hash = model_id or artifact.file_id, artifact.sha256
        self.feature_config = feature_config or WeSpeakerFeatureConfig()
        self.audio_provider = audio_provider
        if type(subsegment_windows) is not bool:
            raise ContractValidationError("subsegment_windows must be boolean")
        if isinstance(subsegment_region_ids, (str, bytes)) or not isinstance(subsegment_region_ids, Sequence):
            raise ContractValidationError("subsegment_region_ids must be a sequence of region IDs")
        if any(not isinstance(region_id, str) or not region_id for region_id in subsegment_region_ids):
            raise ContractValidationError("subsegment_region_ids must contain non-empty strings")
        if subsegment_windows and subsegment_region_ids:
            raise ContractValidationError(
                "global subsegment_windows and selective subsegment_region_ids are mutually exclusive"
            )
        if exact_length_batching and (subsegment_windows or subsegment_region_ids):
            raise ContractValidationError(
                "exact_length_batching is incompatible with global or selective subsegment windows"
            )
        self.subsegment_windows = subsegment_windows
        self.subsegment_region_ids = frozenset(subsegment_region_ids)
        self._last_selector_baseline_results: tuple[EmbeddingResult, ...] | None = None
        self.exact_length_batching = exact_length_batching
        self.batch_buffer_regions = batch_buffer_regions
        self.window_frames = window_frames
        self.period_frames = period_frames
        if feature_mode not in {"kaldi_native", "development_approximation"}:
            raise ContractValidationError(
                "feature_mode must be kaldi_native or development_approximation"
            )
        if feature_extractor is not None:
            self.feature_mode = "injected"
            self._feature_extractor = feature_extractor
        elif feature_mode == "development_approximation":
            self.feature_mode = feature_mode
            self._feature_extractor = wespeaker_logmel_features_approx
        else:
            self.feature_mode = feature_mode
            self._feature_extractor = lambda audio, *, config: kaldi_native_fbank_features(
                audio, config=config, module=native_fbank_module
            )
        self._validate_io(runtime_contract or {})

    def _region_feature_windows(
        self, region: EmbeddingRegion, feature: Any, *, use_subsegments: bool | None = None
    ) -> list[Any]:
        """Create the optional fixed windows used by upstream WeSpeaker diarization.

        This follows https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/diar/extract_emb.py:
        a short/tail slice is resized deterministically to ``window_frames`` and
        CMN is applied per window before ONNX inference.  The default path does
        not enter this branch and retains one embedding per region.
        """
        array = self._np.asarray(feature, dtype=self._np.float32)
        if array.ndim != 2 or array.shape[1] != 80 or array.shape[0] <= 0:
            raise ModelRuntimeIncompatible("WeSpeaker feature extractor returned an invalid region shape")
        enabled = self.subsegment_windows if use_subsegments is None else use_subsegments
        if not enabled:
            return [array]
        frame_shift_us = max(1, round(self.feature_config.frame_shift_ms * 1000.0))
        segment_length = max(1, (region.end_us - region.start_us) // frame_shift_us)
        if segment_length <= self.window_frames:
            ranges = [(0, segment_length)]
        else:
            max_subsegment_begin = segment_length - self.window_frames + self.period_frames
            ranges = [
                (start, min(start + self.window_frames, segment_length))
                for start in range(0, max_subsegment_begin, self.period_frames)
            ]
        windows = []
        for start, end in ranges:
            # np.resize matches upstream's deterministic repeat/truncate behavior.
            window = self._np.resize(array[start:end], (self.window_frames, 80)).astype(
                self._np.float32, copy=False
            )
            window -= window.mean(axis=0, keepdims=True)
            windows.append(window)
        return windows

    def _validate_io(self, runtime_contract: Mapping[str, Any]) -> None:
        ins, outs = self._session.get_inputs(), self._session.get_outputs()
        if len(ins) != 1 or len(outs) != 1:
            raise ModelRuntimeIncompatible("WeSpeaker graph must have one input and one output")
        inp, out = ins[0], outs[0]
        if inp.name != "feats" or out.name != "embs" or len(inp.shape) != 3 or len(out.shape) != 2:
            raise ModelRuntimeIncompatible("inspected WeSpeaker I/O must be feats[batch,frames,80] -> embs[batch,dimension]")
        if inp.shape[-1] not in (80, "80"):
            raise ModelRuntimeIncompatible("WeSpeaker feature metadata is not 80 bins")
        for key, actual in (("input_name", inp.name), ("output_name", out.name), ("feature_bins", 80)):
            if key in runtime_contract and runtime_contract[key] != actual:
                raise ModelRuntimeIncompatible(f"runtime contract {key} mismatch")
        self.input_name, self.output_name = inp.name, out.name
        shape = out.shape[-1]
        if not isinstance(shape, int) or shape <= 0:
            raise ModelRuntimeIncompatible("embedding dimension metadata is dynamic or invalid")
        self.embedding_dimension = shape

    def _embed_once(
        self,
        regions: Sequence[EmbeddingRegion],
        samples_by_region: Mapping[str, Sequence[float]],
        *,
        subsegment_region_ids: frozenset[str] | None,
        audio_provider: Callable[[EmbeddingRegion], Sequence[float]] | None = None,
    ) -> tuple[EmbeddingResult, ...]:
        provider = self.audio_provider if audio_provider is None else audio_provider
        results: list[EmbeddingResult] = []
        for offset in range(0, len(regions), self._max_batch):
            batch = regions[offset:offset + self._max_batch]
            feats = []
            valid_batch: list[EmbeddingRegion] = []
            window_counts: list[int] = []
            batch_results: list[EmbeddingResult | None] = [None] * len(batch)
            for index, region in enumerate(batch):
                if not isinstance(region, EmbeddingRegion):
                    raise ContractValidationError("regions must contain EmbeddingRegion")
                audio = samples_by_region.get(region.embedding_region_id)
                if audio is None and provider is not None:
                    audio = provider(region)
                if audio is None:
                    raise ContractValidationError("audio samples are required for each embedding region")
                try:
                    feature = self._feature_extractor(audio, config=self.feature_config)
                    windows = self._region_feature_windows(
                        region,
                        feature,
                        use_subsegments=(
                            region.embedding_region_id in subsegment_region_ids
                            if subsegment_region_ids is not None
                            else None
                        ),
                    )
                except ContractValidationError as exc:
                    batch_results[index] = EmbeddingResult(
                        region.embedding_region_id,
                        region.tracklet_id,
                        False,
                        failure_reason=f"feature_window_rejected:{exc}",
                        model_pack_id=self.model_id,
                        model_hash=self.model_hash,
                    )
                    continue
                except ModelRuntimeIncompatible:
                    raise
                feats.extend(windows)
                valid_batch.append(region)
                window_counts.append(len(windows))
            if not valid_batch:
                results.extend(result for result in batch_results if result is not None)
                continue
            max_frames = max(x.shape[0] for x in feats)
            tensor = self._np.zeros((len(feats), max_frames, 80), dtype=self._np.float32)
            for i, feat in enumerate(feats):
                tensor[i, :feat.shape[0], :] = feat
            try:
                raw = self._session.run([self.output_name], {self.input_name: tensor})[0]
            except Exception as exc:
                raise ModelRuntimeIncompatible("WeSpeaker ONNX inference failed") from exc
            out = self._np.asarray(raw)
            if out.ndim != 2 or out.shape[0] != len(feats) or out.shape[1] != self.embedding_dimension:
                raise ModelRuntimeIncompatible("WeSpeaker output shape does not match inspected metadata")
            valid_index = 0
            valid_batch_index = 0
            for index, region in enumerate(batch):
                if batch_results[index] is not None:
                    continue
                units = []
                for vector in out[valid_index:valid_index + window_counts[valid_batch_index]]:
                    valid_index += 1
                    norm = float(self._np.linalg.norm(vector))
                    if math.isfinite(norm) and norm > 0:
                        units.append(vector / norm)
                valid_batch_index += 1
                if not units:
                    batch_results[index] = EmbeddingResult(
                        region.embedding_region_id,
                        region.tracklet_id,
                        False,
                        failure_reason="zero_or_nonfinite_embedding",
                        model_pack_id=self.model_id,
                        model_hash=self.model_hash,
                    )
                    continue
                aggregate = self._np.mean(self._np.stack(units), axis=0)
                aggregate_norm = float(self._np.linalg.norm(aggregate))
                if not math.isfinite(aggregate_norm) or aggregate_norm <= 0:
                    batch_results[index] = EmbeddingResult(
                        region.embedding_region_id,
                        region.tracklet_id,
                        False,
                        failure_reason="inconsistent_window_embeddings",
                        model_pack_id=self.model_id,
                        model_hash=self.model_hash,
                    )
                    continue
                unit_array = aggregate / aggregate_norm
                consistency = float(self._np.mean([
                    self._np.dot(vector, unit_array) for vector in units
                ]))
                unit = tuple(float(v) for v in unit_array)
                batch_results[index] = EmbeddingResult(
                    region.embedding_region_id,
                    region.tracklet_id,
                    True,
                    unit,
                    dimension=len(unit),
                    valid_window_count=len(units),
                    clean_window_coverage=region.speech_coverage_ratio,
                    intra_window_consistency=max(-1.0, min(1.0, consistency)),
                    quality=region.speech_coverage_ratio,
                    model_pack_id=self.model_id,
                    model_hash=self.model_hash,
                )
            results.extend(result for result in batch_results if result is not None)
        return tuple(results)

    def _embed_exact_length_bounded(
        self,
        regions: Sequence[EmbeddingRegion],
        samples_by_region: Mapping[str, Sequence[float]],
        *,
        audio_provider: Callable[[EmbeddingRegion], Sequence[float]] | None = None,
    ) -> tuple[EmbeddingResult, ...]:
        """Batch only equal-length region features within a bounded look-ahead.

        Unlike the legacy region batching path, this mode never pads one region
        to another region's frame count.  At most ``batch_buffer_regions``
        region features are materialized before inference starts, and every ORT
        input has at most ``max_batch_regions`` rows.  Results are written back
        by their original region index because grouping changes inference order.

        Subsegment modes are rejected by ``__init__``: one region can expand to
        multiple windows there, so a region-count bound would not also bound the
        number of resident feature tensors.
        """

        provider = self.audio_provider if audio_provider is None else audio_provider
        ordered: list[EmbeddingResult | None] = [None] * len(regions)
        for buffer_start in range(0, len(regions), self.batch_buffer_regions):
            buffer_end = min(len(regions), buffer_start + self.batch_buffer_regions)
            by_frame_count: dict[int, list[tuple[int, EmbeddingRegion, Any]]] = {}
            for result_index in range(buffer_start, buffer_end):
                region = regions[result_index]
                if not isinstance(region, EmbeddingRegion):
                    raise ContractValidationError("regions must contain EmbeddingRegion")
                audio = samples_by_region.get(region.embedding_region_id)
                if audio is None and provider is not None:
                    audio = provider(region)
                if audio is None:
                    raise ContractValidationError("audio samples are required for each embedding region")
                try:
                    feature = self._feature_extractor(audio, config=self.feature_config)
                    windows = self._region_feature_windows(region, feature, use_subsegments=False)
                except ContractValidationError as exc:
                    ordered[result_index] = EmbeddingResult(
                        region.embedding_region_id,
                        region.tracklet_id,
                        False,
                        failure_reason=f"feature_window_rejected:{exc}",
                        model_pack_id=self.model_id,
                        model_hash=self.model_hash,
                    )
                    continue
                except ModelRuntimeIncompatible:
                    raise
                if len(windows) != 1:
                    raise ModelRuntimeIncompatible(
                        "exact-length region batching requires exactly one feature window per region"
                    )
                item = self._np.asarray(windows[0], dtype=self._np.float32)
                by_frame_count.setdefault(int(item.shape[0]), []).append(
                    (result_index, region, item)
                )

            for frame_count, group in by_frame_count.items():
                for group_start in range(0, len(group), self._max_batch):
                    inference_group = group[group_start:group_start + self._max_batch]
                    tensor = self._np.stack(
                        [item[2] for item in inference_group], axis=0
                    ).astype(self._np.float32, copy=False)
                    if tensor.shape != (len(inference_group), frame_count, 80):
                        raise ModelRuntimeIncompatible(
                            "exact-length batch tensor shape does not match grouped feature shapes"
                        )
                    try:
                        raw = self._session.run(
                            [self.output_name], {self.input_name: tensor}
                        )[0]
                    except Exception as exc:
                        raise ModelRuntimeIncompatible("WeSpeaker ONNX inference failed") from exc
                    out = self._np.asarray(raw)
                    if (
                        out.ndim != 2
                        or out.shape[0] != len(inference_group)
                        or out.shape[1] != self.embedding_dimension
                    ):
                        raise ModelRuntimeIncompatible(
                            "WeSpeaker output shape does not match inspected metadata"
                        )
                    for (result_index, region, _), vector in zip(inference_group, out):
                        norm = float(self._np.linalg.norm(vector))
                        units = []
                        if math.isfinite(norm) and norm > 0:
                            units.append(vector / norm)
                        if not units:
                            ordered[result_index] = EmbeddingResult(
                                region.embedding_region_id,
                                region.tracklet_id,
                                False,
                                failure_reason="zero_or_nonfinite_embedding",
                                model_pack_id=self.model_id,
                                model_hash=self.model_hash,
                            )
                            continue
                        # Keep the same two-step normalize/aggregate contract as
                        # the default path, even though this mode has one window.
                        aggregate = self._np.mean(self._np.stack(units), axis=0)
                        aggregate_norm = float(self._np.linalg.norm(aggregate))
                        if not math.isfinite(aggregate_norm) or aggregate_norm <= 0:
                            ordered[result_index] = EmbeddingResult(
                                region.embedding_region_id,
                                region.tracklet_id,
                                False,
                                failure_reason="inconsistent_window_embeddings",
                                model_pack_id=self.model_id,
                                model_hash=self.model_hash,
                            )
                            continue
                        unit_array = aggregate / aggregate_norm
                        consistency = float(self._np.mean([
                            self._np.dot(item, unit_array) for item in units
                        ]))
                        unit = tuple(float(value) for value in unit_array)
                        ordered[result_index] = EmbeddingResult(
                            region.embedding_region_id,
                            region.tracklet_id,
                            True,
                            unit,
                            dimension=len(unit),
                            valid_window_count=1,
                            clean_window_coverage=region.speech_coverage_ratio,
                            intra_window_consistency=max(-1.0, min(1.0, consistency)),
                            quality=region.speech_coverage_ratio,
                            model_pack_id=self.model_id,
                            model_hash=self.model_hash,
                        )

        if any(result is None for result in ordered):
            raise ModelRuntimeIncompatible("exact-length batching did not produce every region result")
        return tuple(result for result in ordered if result is not None)

    @property
    def last_selector_baseline_results(self) -> tuple[EmbeddingResult, ...] | None:
        """Return the exact default-path results retained by the last selector run."""

        return self._last_selector_baseline_results

    def embed(
        self,
        regions: Sequence[EmbeddingRegion],
        samples_by_region: Mapping[str, Sequence[float]] | None = None,
        audio_provider: Callable[[EmbeddingRegion], Sequence[float]] | None = None,
    ) -> tuple[EmbeddingResult, ...]:
        if not isinstance(regions, Sequence) or not regions:
            raise ContractValidationError("regions must be a non-empty sequence")
        samples = samples_by_region or {}
        self._last_selector_baseline_results = None
        if self.exact_length_batching:
            return self._embed_exact_length_bounded(regions, samples, audio_provider=audio_provider)
        if not self.subsegment_region_ids:
            return self._embed_once(
                regions, samples, subsegment_region_ids=None, audio_provider=audio_provider
            )

        # Run the unmodified path once for every region.  Non-selected results
        # are returned directly from this tuple, so their floats and failure
        # contracts are exactly identical to a default backend call.  Only the
        # selected regions are recomputed with upstream-style fixed windows and
        # replaced in their original positions.
        baseline = self._embed_once(
            regions, samples, subsegment_region_ids=frozenset(), audio_provider=audio_provider
        )
        self._last_selector_baseline_results = baseline
        selected_regions = tuple(
            region for region in regions if region.embedding_region_id in self.subsegment_region_ids
        )
        if not selected_regions:
            return baseline
        selected = self._embed_once(
            selected_regions,
            samples,
            subsegment_region_ids=self.subsegment_region_ids,
            audio_provider=audio_provider,
        )
        replacements = {result.embedding_region_id: result for result in selected}
        return tuple(replacements.get(result.embedding_region_id, result) for result in baseline)
