import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_xeon_onecpu_target.py"
SPEC = importlib.util.spec_from_file_location("verify_xeon_onecpu_target_test", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class FakeOrt:
    __version__ = "1.29.0"

    @staticmethod
    def get_available_providers():
        # A GPU provider being installed is diagnostic only.  It must not fail
        # the target until a caller actually selects it.
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _target_env():
    result = {name: "1" for name in validator.THREAD_ONE_ENV}
    result.update(validator.PASSIVE_ENV)
    result.update(validator.DYNAMIC_FALSE_ENV)
    result.update(validator.OTHER_THREAD_ENV)
    return result


def _cpuinfo():
    return (
        "processor : 0\n"
        "model name : Intel Xeon Gold 6230R @ 2.10GHz\n"
        "flags : fpu avx2 AVX-512 VNNI sse4_2\n"
    )


def _fixture(tmp: Path):
    root = tmp / "cgroup"
    cpu = root / "cpu" / "job"
    cpuset = root / "cpuset" / "job"
    cpu.mkdir(parents=True)
    cpuset.mkdir(parents=True)
    (cpu / "cpu.cfs_quota_us").write_text("100000\n", encoding="utf-8")
    (cpu / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    (cpu / "cpu.shares").write_text("1024\n", encoding="utf-8")
    (cpuset / "cpuset.cpus").write_text("0-103\n", encoding="utf-8")
    proc = tmp / "proc-self-cgroup"
    proc.write_text("2:cpu,cpuacct:/job\n3:cpuset:/job\n", encoding="utf-8")
    return root, proc


class XeonOneCpuPreflightTests(unittest.TestCase):
    def test_injected_linux_target_passes_and_redacts_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            root, proc = _fixture(tmp)
            artifact = tmp / "model.onnx"
            artifact.write_bytes(b"fixture-model")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            report = validator.run_preflight(
                cgroup_root=root,
                proc_cgroup_path=proc,
                cpuinfo_text=_cpuinfo(),
                platform_name="linux",
                machine_name="x86_64",
                python_implementation="cpython",
                python_version="3.11.9",
                environment=_target_env(),
                ort_module=FakeOrt(),
                artifacts=[("model", artifact, digest)],
            )
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["checks"]["cgroup"]["cpuset_count"], 104)
        self.assertEqual(report["checks"]["cgroup"]["cpuset_allowed"], "0-103")
        self.assertNotIn(str(tmp), repr(report))
        self.assertEqual(report["checks"]["onnxruntime"]["providers"]["available_other_providers"], ["CUDAExecutionProvider"])

    def test_mac_proxy_fixture_fails_target_platform_without_host_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            root, proc = _fixture(tmp)
            report = validator.run_preflight(
                cgroup_root=root,
                proc_cgroup_path=proc,
                cpuinfo_text=_cpuinfo(),
                platform_name="darwin",
                machine_name="arm64",
                python_implementation="cpython",
                python_version="3.11.9",
                environment=_target_env(),
                ort_module=FakeOrt(),
            )
        self.assertFalse(report["accepted"])
        self.assertIn("platform_not_linux", report["reasons"])
        self.assertIn("architecture_not_x86_64", report["reasons"])

    def test_hash_mismatch_and_gpu_selection_fail_but_gpu_availability_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            root, proc = _fixture(tmp)
            report = validator.run_preflight(
                cgroup_root=root,
                proc_cgroup_path=proc,
                cpuinfo_text=_cpuinfo(),
                platform_name="linux",
                machine_name="x86_64",
                python_implementation="cpython",
                python_version="3.11",
                environment=_target_env(),
                ort_module=FakeOrt(),
                selected_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                artifacts=[("wheel", tmp / "missing.whl", "0" * 64)],
            )
        self.assertFalse(report["accepted"])
        self.assertIn("gpu_execution_provider_selected", report["reasons"])
        self.assertIn("artifact_hash:wheel:missing.whl", report["reasons"])

    def test_proxy_never_reports_target_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            root, proc = _fixture(tmp)
            report = validator.run_proxy(cgroup_root=root, proc_cgroup_path=proc, platform_name="darwin")
        self.assertEqual(report["mode"], "proxy")
        self.assertEqual(report["target_evaluation"], "not_run")
        self.assertFalse(report["accepted"])
        self.assertEqual(report["cgroup"]["version"], None)


if __name__ == "__main__":
    unittest.main()

