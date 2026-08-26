"""Small, strict CPU runtime adapter for an already approved Silero ONNX model.

The adapter intentionally knows nothing about downloading or model discovery.  Silero
releases have used slightly different tensor names, so the input and output roles are
inferred from the inspected ONNX metadata (and checked again at run time).
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Iterable

from .ort_cpu import create_ort_session
from .vad import VadError, VadFrame, VadUnavailableError


def _rank(shape: Any) -> int | None:
    try:
        return len(shape)
    except TypeError:
        return None


def _is_state_name(name: str) -> bool:
    n = name.lower()
    return "state" in n or n in {"h", "hn", "hidden"}


class SileroOnnxRuntime:
    """Run Silero VAD with ONNX Runtime's CPU execution provider only.

    ``session`` is primarily a dependency-free testing seam.  In production it is
    created from ``model_path`` and is never downloaded or implicitly installed.
    ``infer_samples`` accepts normalized mono float samples at 16 kHz, optionally in
    bounded chunks; state is reset at the beginning of every call (every file).
    """

    def __init__(self, model_path: str | Path | Any, *, session: Any = None,
                 threshold: float = 0.5, window_samples: int = 512,
                 context_samples: int = 64) -> None:
        if not 0 < threshold < 1 or window_samples <= 0 or context_samples < 0:
            raise ValueError("invalid Silero runtime configuration")
        path = getattr(model_path, "path", model_path)
        self.model_path = Path(path) if path is not None else None
        if session is None:
            if self.model_path is None or not self.model_path.is_file():
                raise VadUnavailableError("verified local Silero model is missing")
            try:
                session = create_ort_session(self.model_path)
            except VadUnavailableError:
                raise
            except Exception as exc:
                raise VadUnavailableError(f"cannot open local Silero model: {exc}") from exc
        self._session = session
        providers = getattr(session, "get_providers", lambda: ("CPUExecutionProvider",))()
        if tuple(providers) != ("CPUExecutionProvider",):
            raise VadUnavailableError("Silero session is not CPU-only")
        self.threshold = float(threshold)
        self.window_samples = int(window_samples)
        self.context_samples = int(context_samples)
        self._np = self._load_numpy()
        self._inspect_metadata()

    @staticmethod
    def _load_numpy():
        try:
            return importlib.import_module("numpy")
        except ImportError as exc:
            raise VadUnavailableError("numpy is required for Silero ONNX inference") from exc

    def _inspect_metadata(self) -> None:
        inputs = tuple(self._session.get_inputs())
        outputs = tuple(self._session.get_outputs())
        if not inputs or not outputs:
            raise VadError("Silero model has no input/output metadata")
        signal = [x for x in inputs if _rank(getattr(x, "shape", None)) == 2 and not _is_state_name(x.name)]
        states = [x for x in inputs if _rank(getattr(x, "shape", None)) == 3 or _is_state_name(x.name)]
        rates = [x for x in inputs if x not in signal and x not in states]
        if len(signal) != 1 or len(states) != 1 or len(rates) != 1:
            raise VadError("unsupported Silero input contract; expected audio, state and sample-rate inputs")
        state_out = [x for x in outputs if _rank(getattr(x, "shape", None)) == 3 or _is_state_name(x.name)]
        prob_out = [x for x in outputs if x not in state_out]
        if len(state_out) != 1 or len(prob_out) != 1:
            raise VadError("unsupported Silero output contract; expected probability and state output")
        self._audio_name, self._state_name, self._sr_name = signal[0].name, states[0].name, rates[0].name
        self._prob_name, self._state_out_name = prob_out[0].name, state_out[0].name
        shape = getattr(states[0], "shape", None)
        if shape is None or len(shape) != 3:
            raise VadError("Silero state input must have rank 3")
        # Upstream graphs use a dynamic batch axis: [2, batch, 128].
        resolved = tuple(1 if value is None or isinstance(value, str) else int(value) for value in shape)
        if any(value <= 0 for value in resolved):
            raise VadError("Silero state shape contains an invalid dimension")
        self._state_shape = resolved

    def _new_state(self):
        return self._np.zeros(self._state_shape, dtype=self._np.float32)

    def _call(self, samples, state):
        feeds = {
            self._audio_name: self._np.asarray(samples, dtype=self._np.float32).reshape(1, -1),
            self._state_name: state,
            self._sr_name: self._np.asarray(16000, dtype=self._np.int64),
        }
        try:
            values = self._session.run([self._prob_name, self._state_out_name], feeds)
            probability = float(self._np.asarray(values[0]).reshape(-1)[0])
            next_state = self._np.asarray(values[1], dtype=self._np.float32)
        except Exception as exc:
            raise VadError(f"Silero inference failed: {exc}") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise VadError("Silero probability is outside [0,1]")
        if tuple(next_state.shape) != self._state_shape:
            raise VadError("Silero state output shape mismatch")
        return probability, next_state

    def infer_samples(self, samples: Iterable[float], *, sample_rate_hz: int = 16000,
                      start_us: int = 0) -> tuple[VadFrame, ...]:
        if sample_rate_hz != 16000:
            raise VadError("Silero runtime requires normalized 16 kHz audio")
        values = tuple(float(x) for x in samples)
        if not all(math.isfinite(x) and -1.0 <= x <= 1.0 for x in values):
            raise VadError("samples must be finite normalized mono floats")
        state = self._new_state()  # reset for each file, never leak speaker context
        context = self._np.zeros((1, self.context_samples), dtype=self._np.float32)
        result: list[VadFrame] = []
        for start in range(0, len(values), self.window_samples):
            end = min(len(values), start + self.window_samples)
            block = self._np.asarray(values[start:end], dtype=self._np.float32)
            if block.size < self.window_samples:
                block = self._np.pad(block, (0, self.window_samples - block.size))
            model_input = self._np.concatenate((context, block.reshape(1, -1)), axis=1)
            probability, state = self._call(model_input, state)
            if self.context_samples:
                context = model_input[:, -self.context_samples:]
            end_us = start_us + round(end * 1_000_000 / 16000)
            begin_us = start_us + round(start * 1_000_000 / 16000)
            if end_us > begin_us:
                result.append(VadFrame(begin_us, end_us, probability, probability >= self.threshold))
        return tuple(result)

    def infer_chunk_stream(self, chunks: Iterable[Any]) -> tuple[VadFrame, ...]:
        """Run one file across bounded decoded chunks while preserving state/context."""
        state = self._new_state()
        context = self._np.zeros((1, self.context_samples), dtype=self._np.float32)
        pending = self._np.empty((0,), dtype=self._np.float32)
        processed = 0
        expected_source_start = 0
        result: list[VadFrame] = []
        for chunk in chunks:
            if int(chunk.sample_rate_hz) != 16000 or int(chunk.channel_count) != 1:
                raise VadError("Silero stream requires 16 kHz mono decoded chunks")
            if int(chunk.source_start_sample) != expected_source_start:
                raise VadError("Silero stream chunks are not contiguous")
            # The stdlib decoder exposes historical frame-major tuples, while
            # the bounded NumPy PCM fast path exposes a 1-D mono array.  Keep
            # both contracts at this seam without a Python per-frame loop.
            try:
                values = self._np.asarray(chunk.samples, dtype=self._np.float32)
            except (TypeError, ValueError) as exc:
                raise VadError("Silero stream chunks are not valid mono samples") from exc
            if values.ndim == 2:
                if values.shape[1] != 1:
                    raise VadError("Silero stream requires mono decoded chunks")
                values = values[:, 0]
            elif values.ndim != 1:
                raise VadError("Silero stream chunks are not valid mono samples")
            if not self._np.isfinite(values).all() or (self._np.abs(values) > 1.0).any():
                raise VadError("samples must be finite normalized mono floats")
            pending = self._np.concatenate((pending, values))
            expected_source_start = int(chunk.source_end_sample)
            while pending.size >= self.window_samples:
                block, pending = pending[:self.window_samples], pending[self.window_samples:]
                model_input = self._np.concatenate((context, block.reshape(1, -1)), axis=1)
                probability, state = self._call(model_input, state)
                if self.context_samples:
                    context = model_input[:, -self.context_samples:]
                begin_us = round(processed * 1_000_000 / 16000)
                processed += self.window_samples
                end_us = round(processed * 1_000_000 / 16000)
                result.append(VadFrame(begin_us, end_us, probability, probability >= self.threshold))
        if pending.size:
            valid = int(pending.size)
            block = self._np.pad(pending, (0, self.window_samples - valid))
            model_input = self._np.concatenate((context, block.reshape(1, -1)), axis=1)
            probability, state = self._call(model_input, state)
            begin_us = round(processed * 1_000_000 / 16000)
            end_us = round((processed + valid) * 1_000_000 / 16000)
            if end_us > begin_us:
                result.append(VadFrame(begin_us, end_us, probability, probability >= self.threshold))
            processed += valid
        if processed != expected_source_start:
            raise VadError("Silero stream sample-count invariant failed")
        return tuple(result)

    def infer(self, audio) -> tuple[VadFrame, ...]:
        rate = audio.view.sample_rate_hz
        if rate != 16000:
            raise VadError("Silero runtime requires normalized 16 kHz audio")
        return self.infer_samples(audio.samples, start_us=audio.view.time_warp[0].source_start_us)


SileroOnnxVadRuntime = SileroOnnxRuntime
