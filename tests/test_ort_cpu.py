import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sddiar.ort_cpu import OrtCpuBudgetExceededError, OrtCpuConfig, create_ort_session
from sddiar.runtime_env import RuntimeCpuSnapshot


class _Options:
    def __init__(self):
        self.entries = []

    def add_session_config_entry(self, key, value):
        self.entries.append((key, value))


class _Ort:
    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    def __init__(self):
        self.options = None
        self.calls = []

    def get_available_providers(self):
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]

    def SessionOptions(self):
        self.options = _Options()
        return self.options

    def InferenceSession(self, path, *, sess_options, providers):
        self.calls.append((path, sess_options, providers))
        return SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])


def _one_cpu_snapshot():
    return RuntimeCpuSnapshot(
        platform="linux",
        cgroup_version="v2",
        quota_us=100_000,
        period_us=100_000,
        cpuset_cpus=tuple(range(104)),
    )


class OrtCpuTests(unittest.TestCase):
    def test_visible_host_cpus_do_not_override_one_cpu_quota(self):
        ort = _Ort()
        with patch("sddiar.ort_cpu._load_onnxruntime", return_value=ort):
            create_ort_session("model.onnx", runtime_snapshot=_one_cpu_snapshot())
        self.assertEqual(ort.options.intra_op_num_threads, 1)
        self.assertEqual(ort.options.inter_op_num_threads, 1)

    def test_explicit_two_threads_fail_closed_at_one_cpu_quota(self):
        with self.assertRaises(OrtCpuBudgetExceededError):
            create_ort_session("model.onnx", threads=2, runtime_snapshot=_one_cpu_snapshot())

    def test_session_is_sequential_optimized_and_non_spinning_cpu_only(self):
        ort = _Ort()
        with patch("sddiar.ort_cpu._load_onnxruntime", return_value=ort):
            create_ort_session("model.onnx", config=OrtCpuConfig(threads=1), runtime_snapshot=_one_cpu_snapshot())
        options = ort.options
        self.assertEqual(options.execution_mode, "sequential")
        self.assertEqual(options.graph_optimization_level, "all")
        self.assertEqual(options.entries, [
            ("session.intra_op.allow_spinning", "0"),
            ("session.inter_op.allow_spinning", "0"),
        ])
        self.assertEqual(ort.calls[0][2], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
