"""One bounded, CPU-only ONNX Runtime session factory.

The container can expose the host's logical CPU count even when its cgroup has
only one CPU-equivalent of quota.  All production ONNX sessions therefore go
through this module.  ONNX Runtime itself is imported lazily so the rest of the
offline package remains usable without the optional runtime dependency.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_env import RuntimeCpuSnapshot, read_cpu_snapshot


class OrtCpuError(RuntimeError):
    """Base error for a CPU session that cannot be safely provisioned."""


class OrtCpuUnavailableError(OrtCpuError):
    """ONNX Runtime or its CPU execution provider is unavailable."""


class OrtCpuBudgetExceededError(OrtCpuError):
    """The requested worker count exceeds the detected cgroup budget."""


@dataclass(frozen=True, slots=True)
class OrtCpuConfig:
    """Immutable session budget.

    ``threads`` is deliberately an integer worker count.  The default is one
    even when ``os.cpu_count()`` reports a much larger host CPU count.
    """

    threads: int = 1

    def __post_init__(self) -> None:
        if type(self.threads) is not int or self.threads <= 0:
            raise ValueError("threads must be a positive integer")


# A descriptive alias makes the budget intent obvious to callers while keeping
# one canonical immutable type for type checking and tests.
OrtCpuBudget = OrtCpuConfig


def _load_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise OrtCpuUnavailableError("onnxruntime is not installed") from exc


def _check_budget(config: OrtCpuConfig, snapshot: RuntimeCpuSnapshot) -> None:
    effective = snapshot.effective_cpu_equivalent
    # Unknown cgroup state is allowed for portability (macOS and unrestricted
    # hosts).  A detected limit, however, is authoritative and fail-closed.
    if effective is not None and effective < config.threads:
        raise OrtCpuBudgetExceededError(
            "requested ONNX Runtime threads exceed the detected CPU quota "
            f"(requested={config.threads}, effective={effective:g})"
        )


def create_ort_session(
    model_path: str | Path,
    *,
    threads: int | None = None,
    config: OrtCpuConfig | None = None,
    runtime_snapshot: RuntimeCpuSnapshot | None = None,
) -> Any:
    """Create one strictly CPU ONNX Runtime session.

    ``runtime_snapshot`` is an explicit read-only test seam; normal callers
    capture the current cgroup limits through :mod:`sddiar.runtime_env`.
    ``threads=None`` selects the safe one-thread default.  No provider other
    than ``CPUExecutionProvider`` is ever requested.
    """

    if config is not None and threads is not None:
        raise ValueError("provide either config or threads, not both")
    selected = config or OrtCpuConfig(threads=1 if threads is None else threads)
    if not isinstance(selected, OrtCpuConfig):
        raise TypeError("config must be an OrtCpuConfig")
    snapshot = runtime_snapshot or read_cpu_snapshot()
    _check_budget(selected, snapshot)

    ort = _load_onnxruntime()
    try:
        available = tuple(ort.get_available_providers())
    except Exception as exc:
        raise OrtCpuUnavailableError("cannot inspect ONNX Runtime providers") from exc
    if "CPUExecutionProvider" not in available:
        raise OrtCpuUnavailableError("CPUExecutionProvider is unavailable")

    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = selected.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        providers = tuple(getattr(session, "get_providers", lambda: ("CPUExecutionProvider",))())
        if providers != ("CPUExecutionProvider",):
            raise OrtCpuUnavailableError("ONNX session is not CPU-only")
        return session
    except OrtCpuError:
        raise
    except Exception as exc:
        raise OrtCpuError(f"ONNX CPU session creation failed: {exc}") from exc


# Short name for small adapters and compatibility with common ORT terminology.
open_ort_cpu_session = create_ort_session


__all__ = [
    "OrtCpuError",
    "OrtCpuUnavailableError",
    "OrtCpuBudgetExceededError",
    "OrtCpuConfig",
    "OrtCpuBudget",
    "create_ort_session",
    "open_ort_cpu_session",
]
