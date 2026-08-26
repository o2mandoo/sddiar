#!/usr/bin/env python3
"""Read-only preflight for the Linux Xeon Gold 6230R / one-CPU target.

The target preflight is intentionally stricter than the portable runtime.  It
is a release-gate diagnostic: it never changes the process environment,
starts a process, installs a package, or contacts a network.  ``--proxy`` is
provided for local development and reports only the cgroup snapshot; it never
claims that the target gate passed.

The public ``run_preflight`` function has filesystem, platform, Python, ORT,
and environment injection points.  They keep the checks deterministic on
macOS and in unit tests while the command-line defaults read the current
process using :mod:`sddiar.runtime_env`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform as platform_module
import re
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sddiar.runtime_env import RuntimeCpuSnapshot, read_cpu_snapshot


SCHEMA = "xeon_onecpu_preflight_v1"
EXPECTED_MODEL = "Intel Xeon Gold 6230R"
EXPECTED_QUOTA_US = 100_000
EXPECTED_PERIOD_US = 100_000
EXPECTED_CPU_SHARES = 1024
EXPECTED_ORT = "1.29.0"

# These are the variables used by the one-CPU image.  Keeping the policy here
# rather than setting it is important: a preflight must not mutate its caller.
THREAD_ONE_ENV = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
PASSIVE_ENV = {"OMP_WAIT_POLICY": "PASSIVE"}
DYNAMIC_FALSE_ENV = {"OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE"}
OTHER_THREAD_ENV = {"KMP_BLOCKTIME": "0"}


def _check(ok: bool, observed: Any = None, required: Any = None, reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": bool(ok)}
    if observed is not None:
        result["observed"] = observed
    if required is not None:
        result["required"] = required
    if not ok and reason:
        result["reason"] = reason
    return result


def _redacted_name(path: Path) -> str:
    """Return a non-sensitive artifact label instead of an absolute path."""
    return path.name or "<unnamed>"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.strip())
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, PermissionError):
        return None


def _normalise_flags(cpuinfo_text: str) -> set[str]:
    """Extract CPU flags, tolerating Linux spelling/case variants.

    ``avx512_vnni``, ``avx512vnni``, ``AVX-512 VNNI`` and
    ``avx512-vnni`` all normalize to ``avx512vnni``.
    """
    flag_texts: list[str] = []
    for line in cpuinfo_text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() in {"flags", "features"}:
            flag_texts.append(value)
    normalized: set[str] = set()
    for text in flag_texts:
        compact_line = re.sub(r"[^a-z0-9]", "", text.casefold())
        # Some fixtures and procfs-compatible tools render this as the two
        # words ``AVX-512 VNNI`` rather than Linux's usual underscore token.
        if "avx512vnni" in compact_line:
            normalized.add("avx512vnni")
        for token in re.split(r"\s+", text.strip()):
            compact = re.sub(r"[^a-z0-9]", "", token.casefold())
            if compact:
                normalized.add(compact)
    return normalized


def _cpuinfo_model(cpuinfo_text: str) -> str | None:
    for line in cpuinfo_text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() in {"model name", "hardware", "model"}:
            return value.strip()
    return None


def _cpu_range(cpus: Sequence[int] | None) -> str | None:
    if cpus is None:
        return None
    values = sorted(set(cpus))
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _selected_providers(ort_module: Any, selected: Sequence[str] | None) -> tuple[str, ...] | None:
    if selected is not None:
        return tuple(str(item) for item in selected)
    # A fake/injected runtime may expose the providers selected by its caller.
    # Real ORT exposes no process-global selected-provider list, so absence is
    # deliberately represented as unknown rather than inferred from availability.
    for name in ("selected_providers", "providers"):
        value = getattr(ort_module, name, None)
        if isinstance(value, (tuple, list)):
            return tuple(str(item) for item in value)
    getter = getattr(ort_module, "get_selected_providers", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            return None
        if isinstance(value, (tuple, list)):
            return tuple(str(item) for item in value)
    return None


def _ort_check(ort_module: Any | None, selected_providers: Sequence[str] | None) -> dict[str, Any]:
    if ort_module is None:
        try:
            ort_module = importlib.import_module("onnxruntime")
        except Exception:
            failed = _check(False, reason="onnxruntime_unavailable")
            return {"version": failed, "providers": failed, "selection": failed}
    version = str(getattr(ort_module, "__version__", ""))
    version_check = _check(version == EXPECTED_ORT, observed=version or None, required=EXPECTED_ORT,
                           reason="onnxruntime_version_mismatch")
    try:
        available = tuple(str(item) for item in ort_module.get_available_providers())
    except Exception:
        available = ()
    ep_check = _check("CPUExecutionProvider" in available,
                      observed=list(available), required="CPUExecutionProvider",
                      reason="cpu_execution_provider_unavailable")
    selected = _selected_providers(ort_module, selected_providers)
    gpu_selected = bool(selected and any("ExecutionProvider" in item and item != "CPUExecutionProvider" for item in selected))
    selected_check = _check(not gpu_selected,
                            observed=list(selected) if selected is not None else None,
                            required="CPUExecutionProvider only when selected providers are known",
                            reason="gpu_execution_provider_selected")
    # Availability is diagnostic only; selection policy is independent.
    ep_check["available_other_providers"] = [item for item in available if item != "CPUExecutionProvider"]
    return {"version": version_check, "providers": ep_check, "selection": selected_check}


def _hash_checks(artifacts: Sequence[tuple[str, Path, str]]) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    for role, path, expected in artifacts:
        label = f"{role}:{_redacted_name(path)}"
        valid_expected = bool(re.fullmatch(r"[0-9a-fA-F]{64}", expected or ""))
        actual = _sha256(path) if valid_expected else None
        ok = valid_expected and actual is not None and actual.casefold() == expected.casefold()
        checks[label] = _check(ok, observed=(actual[:12] if actual else None), required=(expected[:12] if expected else None),
                               reason="artifact_hash_mismatch" if actual is not None else "artifact_unreadable")
        if not valid_expected:
            checks[label]["reason"] = "invalid_expected_sha256"
        if not ok:
            reasons.append(f"artifact_hash:{role}:{_redacted_name(path)}")
    return checks, reasons


def _env_checks(environment: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    expected: dict[str, str] = {name: "1" for name in THREAD_ONE_ENV}
    expected.update(PASSIVE_ENV)
    expected.update(DYNAMIC_FALSE_ENV)
    expected.update(OTHER_THREAD_ENV)
    for name, wanted in expected.items():
        observed = environment.get(name)
        # Values such as false/FALSE are equivalent policy spellings.
        ok = observed == wanted or (wanted in {"FALSE", "PASSIVE"} and str(observed).casefold() == wanted.casefold())
        checks[name] = _check(ok, observed=observed, required=wanted, reason="thread_environment_mismatch")
        if not ok:
            reasons.append(f"thread_env:{name}")
    return checks, reasons


def _cgroup_report(snapshot: RuntimeCpuSnapshot, shares: int | None, shares_readable: bool) -> dict[str, Any]:
    return {
        "version": snapshot.cgroup_version,
        "quota_us": snapshot.quota_us,
        "period_us": snapshot.period_us,
        "effective_cpu_equivalent": snapshot.effective_cpu_equivalent,
        "cpuset_count": snapshot.cpuset_cpu_count,
        "cpuset_allowed": _cpu_range(snapshot.cpuset_cpus),
        "cpu_shares": shares if shares_readable else None,
        "cpu_shares_readable": shares_readable,
    }


def run_preflight(
    *,
    cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
    proc_cgroup_path: str | os.PathLike[str] = "/proc/self/cgroup",
    cpu_shares_path: str | os.PathLike[str] | None = None,
    cpuinfo_path: str | os.PathLike[str] = "/proc/cpuinfo",
    cpuinfo_text: str | None = None,
    platform_name: str | None = None,
    machine_name: str | None = None,
    python_implementation: str | None = None,
    python_version: str | None = None,
    environment: Mapping[str, str] | None = None,
    ort_module: Any | None = None,
    selected_providers: Sequence[str] | None = None,
    artifacts: Sequence[tuple[str, str | os.PathLike[str], str]] = (),
) -> dict[str, Any]:
    """Return a redacted, JSON-serializable target acceptance report."""
    platform_value = platform_name or sys.platform
    machine_value = machine_name or platform_module.machine()
    implementation = python_implementation or sys.implementation.name
    version_value = python_version or platform_module.python_version()
    env = dict(os.environ if environment is None else environment)
    if cpuinfo_text is None:
        cpuinfo_text = _read(Path(cpuinfo_path)) or ""

    snapshot = read_cpu_snapshot(cgroup_root=cgroup_root, proc_cgroup_path=proc_cgroup_path,
                                 platform_name=platform_value)
    shares_file = Path(cpu_shares_path) if cpu_shares_path else (snapshot.cgroup_path / "cpu.shares" if snapshot.cgroup_path else None)
    shares_text = _read(shares_file) if shares_file else None
    shares = _parse_int(shares_text)
    shares_readable = shares_text is not None

    model = _cpuinfo_model(cpuinfo_text)
    flags = _normalise_flags(cpuinfo_text)
    checks: dict[str, Any] = {
        "platform": _check(platform_value.casefold().startswith("linux"), observed=platform_value, required="linux",
                            reason="platform_not_linux"),
        "architecture": _check(machine_value.casefold() in {"x86_64", "amd64"}, observed=machine_value, required="x86_64",
                                reason="architecture_not_x86_64"),
        "cpu_model": _check(model is not None and EXPECTED_MODEL.casefold() in model.casefold(),
                             observed=model, required=EXPECTED_MODEL, reason="cpu_model_mismatch"),
        "cpu_flags": _check("avx2" in flags and "avx512vnni" in flags,
                            observed=sorted(flag for flag in flags if flag in {"avx2", "avx512vnni"}),
                            required=["avx2", "avx512_vnni"], reason="required_cpu_flags_missing"),
        "cgroup_version": _check(snapshot.cgroup_version == "v1", observed=snapshot.cgroup_version, required="v1",
                                  reason="cgroup_version_mismatch"),
        "cpu_quota": _check(snapshot.quota_us == EXPECTED_QUOTA_US and snapshot.period_us == EXPECTED_PERIOD_US,
                             observed={"quota_us": snapshot.quota_us, "period_us": snapshot.period_us},
                             required={"quota_us": EXPECTED_QUOTA_US, "period_us": EXPECTED_PERIOD_US}, reason="cpu_quota_mismatch"),
        "effective_cpu": _check(snapshot.effective_cpu_equivalent == 1.0,
                                 observed=snapshot.effective_cpu_equivalent, required=1.0, reason="effective_cpu_mismatch"),
        "cpu_shares": _check((not shares_readable) or shares == EXPECTED_CPU_SHARES,
                              observed=shares if shares_readable else "unreadable", required=EXPECTED_CPU_SHARES,
                              reason="cpu_shares_mismatch"),
        "python": _check(implementation.casefold() == "cpython" and version_value.split(".")[:2] == ["3", "11"],
                          observed={"implementation": implementation, "version": version_value}, required="CPython 3.11",
                          reason="python_runtime_mismatch"),
    }
    checks["cgroup"] = _cgroup_report(snapshot, shares, shares_readable)
    env_checks, env_reasons = _env_checks(env)
    checks["thread_environment"] = env_checks
    ort_checks = _ort_check(ort_module, selected_providers)
    checks["onnxruntime"] = ort_checks
    hash_checks, hash_reasons = _hash_checks([(role, Path(path), expected) for role, path, expected in artifacts])
    checks["artifacts"] = hash_checks

    reasons: list[str] = []
    for name, value in checks.items():
        if name in {"cgroup", "artifacts", "thread_environment", "onnxruntime"}:
            continue
        if isinstance(value, Mapping) and not value.get("ok", True):
            reasons.append(str(value.get("reason", f"{name}_failed")))
    reasons.extend(env_reasons)
    for section in ort_checks.values():
        if isinstance(section, Mapping) and not section.get("ok", True):
            reasons.append(str(section.get("reason", "onnxruntime_check_failed")))
    reasons.extend(hash_reasons)
    # Keep reasons stable for CI and avoid duplicate output when a fixture has
    # several independent symptoms of the same failure.
    reasons = list(dict.fromkeys(reasons))
    accepted = not reasons
    return {
        "schema": SCHEMA,
        "mode": "target",
        "accepted": accepted,
        "ok": accepted,
        "checks": checks,
        "reasons": reasons,
    }


def run_proxy(*, cgroup_root: str | os.PathLike[str] = "/sys/fs/cgroup",
              proc_cgroup_path: str | os.PathLike[str] = "/proc/self/cgroup",
              platform_name: str | None = None) -> dict[str, Any]:
    """Report local cgroup state without evaluating or passing the target."""
    snapshot = read_cpu_snapshot(cgroup_root=cgroup_root, proc_cgroup_path=proc_cgroup_path,
                                 platform_name=platform_name or sys.platform)
    shares_file = snapshot.cgroup_path / "cpu.shares" if snapshot.cgroup_path else None
    shares_text = _read(shares_file) if shares_file else None
    return {
        "schema": SCHEMA,
        "mode": "proxy",
        "ok": True,
        "accepted": False,
        "target_evaluation": "not_run",
        "cgroup": _cgroup_report(snapshot, _parse_int(shares_text), shares_text is not None),
        "reasons": [],
    }


def _pairs(paths: Sequence[str], hashes: Sequence[str], role: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    issues: list[str] = []
    if len(paths) != len(hashes):
        issues.append(f"{role}_hash_count_mismatch")
    pairs = [(role, path, digest) for path, digest in zip(paths, hashes)]
    return pairs, issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", action="store_true", help="report local cgroup only; never evaluate target acceptance")
    parser.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    parser.add_argument("--proc-cgroup-path", default="/proc/self/cgroup")
    parser.add_argument("--cpu-shares-path")
    parser.add_argument("--cpuinfo-path", default="/proc/cpuinfo")
    parser.add_argument("--model", action="append", default=[], help="local model path (repeatable)")
    parser.add_argument("--model-sha256", "--model-hash", action="append", default=[], dest="model_hashes")
    parser.add_argument("--wheel", action="append", default=[], help="local wheel path (repeatable)")
    parser.add_argument("--wheel-sha256", "--wheel-hash", action="append", default=[], dest="wheel_hashes")
    parser.add_argument("--lock", action="append", default=[], help="local lock path (repeatable)")
    parser.add_argument("--lock-sha256", "--lock-hash", action="append", default=[], dest="lock_hashes")
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE:PATH=SHA256",
                        help="generic local artifact hash (role is model, wheel, or lock)")
    parser.add_argument("--selected-provider", action="append", default=[], dest="selected_providers")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.proxy:
        print(json.dumps(run_proxy(cgroup_root=args.cgroup_root, proc_cgroup_path=args.proc_cgroup_path),
                         ensure_ascii=False, sort_keys=True))
        return 0

    artifacts: list[tuple[str, str, str]] = []
    pairing_issues: list[str] = []
    for paths, hashes, role in ((args.model, args.model_hashes, "model"),
                                (args.wheel, args.wheel_hashes, "wheel"),
                                (args.lock, args.lock_hashes, "lock")):
        pairs, issues = _pairs(paths, hashes, role)
        artifacts.extend(pairs)
        pairing_issues.extend(issues)
    for value in args.artifact:
        match = re.fullmatch(r"(model|wheel|lock):(.+)=([0-9a-fA-F]{64})", value)
        if match:
            artifacts.append((match.group(1), match.group(2), match.group(3)))
        else:
            pairing_issues.append("invalid_artifact_argument")
    report = run_preflight(cgroup_root=args.cgroup_root, proc_cgroup_path=args.proc_cgroup_path,
                           cpu_shares_path=args.cpu_shares_path, cpuinfo_path=args.cpuinfo_path,
                           selected_providers=args.selected_providers or None, artifacts=artifacts)
    if pairing_issues:
        report["accepted"] = report["ok"] = False
        report["reasons"] = list(dict.fromkeys([*report["reasons"], *pairing_issues]))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
