from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).parents[1] / "bench" / "one_cpu" / "run_repeated_worker.py"
SPEC = importlib.util.spec_from_file_location("run_repeated_worker", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class _Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.01
        return self.value


class _FakeDiarizer:
    session_backend_reused = True
    session_reuse_count = 1
    backend_reuse_count = 1

    def __init__(self, drift=False, fallback=False):
        self.calls = 0
        self.drift = drift
        self.fallback_used = fallback

    def process(self, path):
        self.calls += 1
        end = 10 if not self.drift or self.calls == 1 else 11
        return SimpleNamespace(
            duration_us=1_000_000,
            decision="H2_CONFIRMED",
            quality_status="REVIEW_REQUIRED",
            metrics={"tracklet_count": 1, "anchor_count": 1, "valid_embedding_count": 1},
            spans=(SimpleNamespace(start_us=0, end_us=end, speaker_id="SPEAKER_00",
                                   attribution_status="ASSIGNED", evidence_ids=("e",),
                                   reason_codes=()),),
        )


def _cpu_reader_factory():
    calls = 0

    def reader():
        nonlocal calls
        calls += 1
        stat = SimpleNamespace(usage_usec=calls * 10, user_usec=calls * 8,
                               system_usec=calls * 2, nr_periods=calls,
                               nr_throttled=0, throttled_usec=0,
                               reset_detected=False)
        return SimpleNamespace(effective_cpu_equivalent=1.0, cgroup_version="v2",
                               cpuset_cpu_count=1, cpu_stat=stat)

    return reader


def _delta(before, after):
    return SimpleNamespace(
        cpu_stat=SimpleNamespace(usage_usec=10, user_usec=8, system_usec=2,
                                 nr_periods=1, nr_throttled=0, throttled_usec=0,
                                 reset_detected=False),
        cgroup_version="v2", cpuset_cpu_count=1,
    )


def _resources():
    return {"VmRSS_bytes": 100, "VmHWM_bytes": 120, "Threads": 1}


def _smaps():
    return {"Rss_bytes": 100, "Pss_bytes": 80}


def _memory():
    return {"current_bytes": 200, "peak_bytes": 220}


def _tree():
    return {"process_count": 1, "thread_count": 1, "process_tree_rss_bytes": 100,
            "process_tree_pss_bytes": 80, "read_bytes": 1000, "write_bytes": 10}


def _rusage():
    return {"ru_maxrss_bytes": 100, "user_cpu_sec": 1.0, "system_cpu_sec": 0.1}


class RepeatedWorkerTests(unittest.TestCase):
    def run_worker(self, factory, *, repetitions=3, rss_growth_limit_mb=1.0):
        clock = _Clock()
        return worker.run_repeated_worker(
            ["/private/input/audio.wav"], factory, repetitions=repetitions,
            config={"threads": 1, "audio_path": "/private/input/audio.wav"},
            duration_reader=lambda _: 1_000_000,
            cpu_snapshot_reader=_cpu_reader_factory(), cpu_delta_reader=_delta,
            status_reader=_resources, smaps_reader=_smaps, memory_reader=_memory,
            process_tree_reader=_tree, cpuacct_usage_reader=lambda: None,
            rusage_reader=_rusage, clock=clock, process_clock=clock,
            gc_collect=lambda: 0, gc_counts=lambda: (0, 0, 0),
            rss_growth_limit_mb=rss_growth_limit_mb,
        )

    def test_one_instance_is_reused_and_output_is_redacted(self):
        made = []

        def factory():
            item = _FakeDiarizer()
            made.append(item)
            return item

        payload = self.run_worker(factory, repetitions=2)
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0].calls, 2)
        self.assertTrue(payload["worker"]["reused_diarizer"])
        self.assertTrue(payload["worker"]["session_backend_reuse"]["session_backend_reused"])
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["runs"][0]["timeline_digest"], payload["runs"][1]["timeline_digest"])
        self.assertEqual(payload["runs"][0]["cpu"]["usage_usec"], 10)
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("/private/input/audio.wav", serialized)
        self.assertNotIn('"spans"', serialized)
        self.assertNotIn('"source_names"', serialized)

    def test_digest_drift_fails(self):
        with self.assertRaises(worker.DigestDriftError):
            self.run_worker(lambda: _FakeDiarizer(drift=True), repetitions=2)

    def test_quota_and_fallback_fail_closed(self):
        def quota_reader():
            return SimpleNamespace(effective_cpu_equivalent=0.5)

        with self.assertRaises(worker.QuotaMismatchError):
            worker.run_repeated_worker(
                ["audio.wav"], lambda: _FakeDiarizer(), repetitions=1,
                duration_reader=lambda _: 1_000_000, cpu_snapshot_reader=quota_reader,
                status_reader=_resources, smaps_reader=_smaps, memory_reader=_memory,
                process_tree_reader=_tree, cpuacct_usage_reader=lambda: None,
                rusage_reader=_rusage, gc_collect=lambda: 0, gc_counts=lambda: (0, 0, 0),
            )
        with self.assertRaises(worker.FallbackError):
            self.run_worker(lambda: _FakeDiarizer(fallback=True), repetitions=1)

    def test_memory_parser_converts_proc_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "status"
            status.write_text("VmRSS: 12 kB\nVmHWM: 14 kB\nThreads: 3\n", encoding="utf-8")
            smaps = Path(tmp) / "smaps_rollup"
            smaps.write_text("Rss: 20 kB\nPss: 10 kB\n", encoding="utf-8")
            self.assertEqual(worker.read_proc_status(status)["VmRSS_bytes"], 12 * 1024)
            self.assertEqual(worker.read_proc_status(status)["Threads"], 3)
            self.assertEqual(worker.read_smaps_rollup(smaps)["Pss_bytes"], 10 * 1024)

    def test_wave_format_extensible_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extended.wav"
            samples = b"\0\0" * 16_000
            pcm_guid = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
            fmt = struct.pack("<HHIIHHH", 0xFFFE, 1, 16000, 32000, 2, 16, 22)
            fmt += struct.pack("<HI", 16, 0) + pcm_guid
            chunks = (b"fmt " + struct.pack("<I", len(fmt)) + fmt
                      + b"data" + struct.pack("<I", len(samples)) + samples)
            path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)
            self.assertEqual(worker.read_wav_duration_us(path), 1_000_000)

    def test_target_mode_requires_full_evidence_and_duration(self):
        with self.assertRaises(ValueError):
            worker.run_repeated_worker(
                ["audio.wav"], lambda: _FakeDiarizer(), repetitions=1,
                evidence_mode="target", min_total_audio_minutes=0,
                duration_reader=lambda _: 1_000_000,
            )

    def test_process_tree_and_cgroup_v1_cpuacct_are_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            for pid, ppid, rss_kb, pss_kb, read_bytes in (
                (100, 1, 10, 8, 1000), (200, 100, 20, 15, 2000), (300, 1, 99, 90, 9000),
            ):
                root = proc / str(pid)
                root.mkdir(parents=True)
                (root / "status").write_text(
                    f"PPid:\t{ppid}\nVmRSS:\t{rss_kb} kB\nThreads:\t2\n", encoding="utf-8")
                (root / "smaps_rollup").write_text(
                    f"Rss: {rss_kb + 1} kB\nPss: {pss_kb} kB\n", encoding="utf-8")
                (root / "io").write_text(
                    f"read_bytes: {read_bytes}\nwrite_bytes: 4\n", encoding="utf-8")
            tree = worker.read_process_tree_resources(proc, root_pid=100, platform_name="linux")
            self.assertEqual(tree["process_count"], 2)
            self.assertEqual(tree["process_tree_rss_bytes"], 32 * 1024)
            self.assertEqual(tree["process_tree_pss_bytes"], 23 * 1024)
            self.assertEqual(tree["read_bytes"], 3000)

            cgroup = Path(tmp) / "cgroup"
            leaf = cgroup / "cpu,cpuacct" / "kubepods" / "unit"
            leaf.mkdir(parents=True)
            (leaf / "cpuacct.usage").write_text("1234567000\n", encoding="utf-8")
            membership = Path(tmp) / "cgroup-membership"
            membership.write_text("5:cpu,cpuacct:/kubepods/unit\n", encoding="utf-8")
            self.assertEqual(worker.read_cgroup_cpuacct_usage_usec(
                cgroup, membership, platform_name="linux"), 1_234_567)

    def test_runtime_limits_fail_closed(self):
        clock = _Clock()
        with self.assertRaises(worker.RuntimeLimitError):
            worker.run_repeated_worker(
                ["audio.wav"], lambda: _FakeDiarizer(), repetitions=1,
                duration_reader=lambda _: 1_000_000,
                cpu_snapshot_reader=_cpu_reader_factory(), cpu_delta_reader=_delta,
                status_reader=_resources, smaps_reader=_smaps, process_tree_reader=_tree,
                memory_reader=_memory, cpuacct_usage_reader=lambda: None, rusage_reader=_rusage,
                clock=clock, process_clock=clock, gc_collect=lambda: 0,
                gc_counts=lambda: (0, 0, 0), max_rtf=0.001,
            )


if __name__ == "__main__":
    unittest.main()
