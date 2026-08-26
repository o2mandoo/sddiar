import tempfile
import unittest
from pathlib import Path

from sddiar.runtime_env import (
    CgroupCpuStat,
    delta_cpu_snapshots,
    read_cpu_snapshot,
)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_cgroup_v2_quota_cpuset_and_throttling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cgroup"
            group = root / "job"
            group.mkdir(parents=True)
            (root / "cgroup.controllers").write_text("cpu cpuset memory\n", encoding="utf-8")
            (group / "cpu.max").write_text("150000 100000\n", encoding="utf-8")
            (group / "cpuset.cpus.effective").write_text("0-3,8\n", encoding="utf-8")
            (group / "cpu.stat").write_text(
                "usage_usec 1200\nuser_usec 900\nsystem_usec 300\n"
                "nr_periods 12\nnr_throttled 2\nthrottled_usec 450\n",
                encoding="utf-8",
            )
            proc = Path(tmp) / "self-cgroup"
            proc.write_text("0::/job\n", encoding="utf-8")

            snapshot = read_cpu_snapshot(cgroup_root=root, proc_cgroup_path=proc, platform_name="linux")

        self.assertEqual(snapshot.cgroup_version, "v2")
        self.assertEqual(snapshot.quota_us, 150000)
        self.assertEqual(snapshot.period_us, 100000)
        self.assertEqual(snapshot.cpuset_cpus, (0, 1, 2, 3, 8))
        self.assertEqual(snapshot.cpuset_cpu_count, 5)
        self.assertEqual(snapshot.effective_cpu_equivalent, 1.5)
        self.assertIsNotNone(snapshot.cpu_stat)
        assert snapshot.cpu_stat is not None
        self.assertEqual(snapshot.cpu_stat.nr_throttled, 2)
        self.assertEqual(snapshot.cpu_stat.throttled_usec, 450)

    def test_cgroup_v1_unlimited_quota_uses_cpuset_and_normalizes_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cgroup"
            cpu = root / "cpu" / "batch"
            cpuset = root / "cpuset" / "batch"
            cpu.mkdir(parents=True)
            cpuset.mkdir(parents=True)
            (cpu / "cpu.cfs_quota_us").write_text("-1\n", encoding="utf-8")
            (cpu / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
            (cpu / "cpu.stat").write_text("nr_periods 10\nnr_throttled 3\nthrottled_time 2500000\n", encoding="utf-8")
            (cpuset / "cpuset.cpus").write_text("2-3\n", encoding="utf-8")
            proc = Path(tmp) / "self-cgroup"
            proc.write_text("2:cpu,cpuacct:/batch\n3:cpuset:/batch\n", encoding="utf-8")

            snapshot = read_cpu_snapshot(cgroup_root=root, proc_cgroup_path=proc, platform_name="linux")

        self.assertEqual(snapshot.cgroup_version, "v1")
        self.assertEqual(snapshot.quota_us, -1)
        self.assertEqual(snapshot.effective_cpu_equivalent, 2.0)
        assert snapshot.cpu_stat is not None
        self.assertEqual(snapshot.cpu_stat.throttled_time_ns, 2500000)
        self.assertEqual(snapshot.cpu_stat.throttled_usec, 2500)

    def test_mac_and_missing_cgroup_are_safe_and_do_not_use_host_cpu_count(self):
        mac = read_cpu_snapshot(platform_name="darwin", cgroup_root="/does/not/exist")
        self.assertIsNone(mac.cgroup_version)
        self.assertIsNone(mac.effective_cpu_equivalent)
        self.assertEqual(mac.unavailable_reason, "non_linux")

        with tempfile.TemporaryDirectory() as tmp:
            missing = read_cpu_snapshot(
                cgroup_root=Path(tmp) / "missing",
                proc_cgroup_path=Path(tmp) / "missing-proc",
                platform_name="linux",
            )
        self.assertIsNone(missing.cgroup_version)
        self.assertIsNone(missing.effective_cpu_equivalent)
        self.assertEqual(missing.unavailable_reason, "cgroup_unavailable")

    def test_delta_reports_throttling_and_counter_reset(self):
        before = CgroupCpuStat.from_text("usage_usec 100\nnr_throttled 1\nthrottled_usec 20\n")
        after = CgroupCpuStat.from_text("usage_usec 180\nnr_throttled 4\nthrottled_usec 70\n")
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        delta = delta_cpu_snapshots(
            read_cpu_snapshot(platform_name="darwin"),
            read_cpu_snapshot(platform_name="darwin"),
        )
        self.assertIsNone(delta.cpu_stat)

        from sddiar.runtime_env import RuntimeCpuSnapshot

        result = delta_cpu_snapshots(
            RuntimeCpuSnapshot("linux", cpu_stat=before),
            RuntimeCpuSnapshot("linux", cpu_stat=after),
        )
        assert result.cpu_stat is not None
        self.assertEqual(result.cpu_stat.usage_usec, 80)
        self.assertEqual(result.cpu_stat.nr_throttled, 3)
        self.assertEqual(result.cpu_stat.throttled_usec, 50)

        reset = delta_cpu_snapshots(
            RuntimeCpuSnapshot("linux", cpu_stat=after),
            RuntimeCpuSnapshot("linux", cpu_stat=before),
        )
        assert reset.cpu_stat is not None
        self.assertTrue(reset.cpu_stat.reset_detected)
        self.assertIsNone(reset.cpu_stat.usage_usec)


if __name__ == "__main__":
    unittest.main()
