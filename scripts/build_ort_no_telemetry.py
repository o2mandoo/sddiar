#!/usr/bin/env python3
"""Plan and attest a pinned, target-native ONNX Runtime 1.29.0 build.

This helper never downloads source and never claims that a produced binary is
telemetry-free.  It emits the exact source/build configuration that a target
host must run, and records ``telemetry_status=not_verified`` in attestations
until an independent binary/source audit is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ORT_VERSION = "1.29.0"
SOURCE_URL = "https://github.com/microsoft/onnxruntime.git"
SOURCE_REF = "v1.29.0"
# Resolved with: git ls-remote <SOURCE_URL> refs/tags/v1.29.0
SOURCE_COMMIT = "2e2543fbe9fae542f921d47a72d21d5a4ef0b710"
PYTHON_ABI = "cp311"

# These are the reproducibility pins used by the v1.29.0 upstream CI where
# they are published.  The host must record the actual compiler/Xcode and
# container digest in the attestation; a mutable image tag is not accepted as
# proof of a target-native build.
TOOLCHAIN_PINS: dict[str, Any] = {
    "cmake": {
        "version": "3.31.8",
        "sha512": "99cc9c63ae49f21253efb5921de2ba84ce136018abf08632c92c060ba91d552e0f6acc214e9ba8123dee0cf6d1cf089ca389e321879fd9d719a60d975bcffcc8",
    },
    "vcpkg": {
        "version": "2025.08.27",
        "sha512": "9a4b32849792e13bee1d24726f073b3881acae4165206ddf1a6378e44a4ddd05b3ee93f55ff46d8e8873b3cbcd06606212989e248f0bd615a5bf365070074079",
    },
    "windows": {"generator": "Visual Studio 17 2022", "architecture": "x64", "record_vswhere_version": True},
    "linux": {"compiler_family": "GCC 14", "record_container_digest": True},
    "macos": {"record_xcodebuild_version": True, "record_sdk_and_architecture": True},
}

TARGETS: dict[str, dict[str, Any]] = {
    "windows-x86_64": {
        "os": "windows",
        "arch": "x86_64",
        "generator": "Visual Studio 17 2022",
        "target_arch": "x64",
        "shell": "powershell",
        "cmake_arch": None,
    },
    "linux-x86_64": {
        "os": "linux",
        "arch": "x86_64",
        "generator": "Ninja",
        "target_arch": None,
        "shell": "posix",
        "cmake_arch": None,
    },
    "linux-aarch64": {
        "os": "linux",
        "arch": "aarch64",
        "generator": "Ninja",
        "target_arch": None,
        "shell": "posix",
        "cmake_arch": None,
    },
    "macos-x86_64": {
        "os": "darwin",
        "arch": "x86_64",
        "generator": "Xcode",
        "target_arch": None,
        "shell": "posix",
        "cmake_arch": "x86_64",
    },
    "macos-arm64": {
        "os": "darwin",
        "arch": "arm64",
        "generator": "Xcode",
        "target_arch": None,
        "shell": "posix",
        "cmake_arch": "arm64",
    },
}


def _build_command(target: str) -> list[str]:
    spec = TARGETS[target]
    command = [
        "python3.11" if spec["os"] != "windows" else "py",
    ]
    if spec["os"] == "windows":
        command.append("-3.11")
    command.extend([
        "tools/ci_build/build.py",
        "--build_dir",
        f"build/ort-no-telemetry/{target}",
        "--config",
        "Release",
        "--build_wheel",
        "--skip_tests",
        "--skip_submodule_sync",
        "--parallel",
        "1",
        "--enable_pybind",
        "--skip_pip_install",
        "--use_vcpkg",
        "--no_telemetry",
        "--cmake_generator",
        spec["generator"],
    ])
    # The upstream v1.29 build.py switch is the authoritative intent.  Keep
    # the CMake definition too, so a wrapper/default cannot silently turn
    # telemetry on and the generated command remains independently auditable.
    command.extend(["--cmake_extra_defines", "onnxruntime_USE_TELEMETRY=OFF"])
    if spec["cmake_arch"]:
        command.append(f"CMAKE_OSX_ARCHITECTURES={spec['cmake_arch']}")
    return command


def build_plan(target: str | None = None) -> dict[str, Any]:
    selected = [target] if target else list(TARGETS)
    return {
        "schema_version": "1.0",
        "kind": "onnxruntime-no-telemetry-build-plan",
        "production_approved": False,
        "source": {
            "url": SOURCE_URL,
            "ref": SOURCE_REF,
            "commit": SOURCE_COMMIT,
            "clone": "git clone --branch v1.29.0 --recurse-submodules --shallow-submodules",
            "submodules": "record git submodule status --recursive after checkout; do not resolve floating refs",
        },
        "runtime": {
            "onnxruntime_version": ORT_VERSION,
            "python": "CPython 3.11.x",
            "python_abi": PYTHON_ABI,
            "execution_provider": "CPUExecutionProvider",
            "upstream_telemetry_switch": "--no_telemetry",
            "telemetry_cmake_definition": "onnxruntime_USE_TELEMETRY=OFF",
        },
        "toolchain_policy": {
            **TOOLCHAIN_PINS,
            "ninja": "if selected, target image must pin and record an exact Ninja 1.x version",
            "target_native": "compiler, SDK, and container/image digest are mandatory attestation fields",
            "network": "source and submodules are fetched before the closed-network build; build/install has no network access",
        },
        "targets": {
            name: {
                **TARGETS[name],
                "command": _build_command(name),
                "command_shell": shlex.join(_build_command(name)),
            }
            for name in selected
        },
        "attestation_schema": {
            "required": [
                "schema_version",
                "source",
                "target",
                "artifact",
                "toolchain",
                "configuration",
                "telemetry",
            ],
            "telemetry_status_allowed": ["not_verified"],
            "production_approval": "must remain false until independent audit",
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _git_commit(source_root: Path) -> str | None:
    return _run_text(["git", "rev-parse", "HEAD"], source_root)


def _git_checkout_clean(source_root: Path) -> bool | None:
    """Return whether tracked and untracked checkout state is clean."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return not result.stdout.strip()


def _submodules(source_root: Path) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["git", "submodule", "status", "--recursive"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    output = result.stdout
    records: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        # git submodule status uses the first column as a state marker:
        # space=clean, -=uninitialised, +=different checkout, U=conflict.
        state = line[0] if line[0] in " -+U" else " "
        fields = line[1:].strip().split(maxsplit=2)
        if len(fields) >= 2:
            records.append({"state": state, "commit": fields[0], "path": fields[1]})
    return records


def _tool_version(executable: str, *args: str) -> str | None:
    return _run_text([executable, *args])


def _compiler_version(target: str) -> str | None:
    target_os = TARGETS[target]["os"]
    if target_os == "windows":
        return _tool_version("cl")
    if target_os == "linux":
        return _tool_version("gcc", "--version") or _tool_version("clang", "--version")
    return _tool_version("clang", "--version")


def _target_native_evidence(target: str) -> dict[str, str | None]:
    target_os = TARGETS[target]["os"]
    return {
        "vswhere_version": _tool_version("vswhere", "-version") if target_os == "windows" else None,
        # The digest is supplied by the pinned native build environment; a
        # mutable image tag is deliberately not accepted as evidence.
        "container_digest": os.environ.get("ORT_BUILD_CONTAINER_DIGEST") if target_os == "linux" else None,
        "xcodebuild_version": _tool_version("xcodebuild", "-version") if target_os == "darwin" else None,
        "sdk": _tool_version("xcrun", "--sdk", "macosx", "--show-sdk-path") if target_os == "darwin" else None,
    }


def make_attestation(target: str, artifact: Path, source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    artifact = artifact.resolve()
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return {
        "schema_version": "1.0",
        "kind": "onnxruntime-no-telemetry-build-attestation",
        "production_approved": False,
        "source": {
            "url": SOURCE_URL,
            "ref": SOURCE_REF,
            "commit": _git_commit(source_root),
            "expected_commit": SOURCE_COMMIT,
            "checkout_clean": _git_checkout_clean(source_root),
            "root": str(source_root),
            "submodules": _submodules(source_root),
        },
        "target": {"id": target, **TARGETS[target], "python_abi": PYTHON_ABI},
        "artifact": {"path": str(artifact), "sha256": _sha256(artifact)},
        "toolchain": {
            "python": platform.python_version(),
            "cmake": _tool_version("cmake", "--version"),
            "ninja": _tool_version("ninja", "--version"),
            "compiler": _compiler_version(target),
            "required_pins": TOOLCHAIN_PINS,
            "target_native_evidence": _target_native_evidence(target),
        },
        "configuration": {
            "onnxruntime_version": ORT_VERSION,
            "command": _build_command(target),
            "cmake_definitions": {"onnxruntime_USE_TELEMETRY": "OFF"},
        },
        # This is deliberately not a telemetry-free claim.  A separate audit
        # must inspect the binary, source, and dependency closure.
        "telemetry": {
            "build_flag": "onnxruntime_USE_TELEMETRY=OFF",
            "status": "not_verified",
            "evidence": [],
        },
    }


def validate_attestation(payload: dict[str, Any], *, check_files: bool = True) -> list[str]:
    issues: list[str] = []
    for key in ("schema_version", "source", "target", "artifact", "toolchain", "configuration", "telemetry"):
        if key not in payload:
            issues.append(f"missing:{key}")
    if payload.get("schema_version") != "1.0":
        issues.append("schema_version_not_1.0")
    if payload.get("production_approved") is not False:
        issues.append("production_approved_must_be_false")
    source = payload.get("source", {})
    if source.get("url") != SOURCE_URL:
        issues.append("source_url_mismatch")
    if source.get("ref") != SOURCE_REF:
        issues.append("source_ref_mismatch")
    if source.get("expected_commit") != SOURCE_COMMIT:
        issues.append("source_expected_commit_mismatch")
    if source.get("commit") != SOURCE_COMMIT:
        issues.append("source_checkout_not_at_pinned_commit")
    if source.get("checkout_clean") is not True:
        issues.append("source_checkout_not_clean")
    submodules = source.get("submodules")
    if not isinstance(submodules, list) or not submodules:
        issues.append("source_submodules_missing")
    else:
        for record in submodules:
            if not isinstance(record, dict) or record.get("state") != " ":
                issues.append("source_submodules_not_clean")
                break
            commit = record.get("commit")
            if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                issues.append("source_submodule_commit_not_pinned")
                break
    target = payload.get("target", {})
    target_id = target.get("id")
    if target_id not in TARGETS:
        issues.append("unknown_target")
    configuration = payload.get("configuration", {})
    if configuration.get("onnxruntime_version") != ORT_VERSION:
        issues.append("onnxruntime_version_mismatch")
    if configuration.get("cmake_definitions", {}).get("onnxruntime_USE_TELEMETRY") != "OFF":
        issues.append("telemetry_cmake_definition_not_off")
    command = configuration.get("command")
    if target_id in TARGETS and command != _build_command(target_id):
        issues.append("build_command_not_pinned")
    if target_id in TARGETS:
        required_command_tokens = (
            "--build_dir",
            "--no_telemetry",
            "--cmake_extra_defines",
            "onnxruntime_USE_TELEMETRY=OFF",
        )
        if not isinstance(command, list) or any(token not in command for token in required_command_tokens):
            issues.append("build_command_missing_required_switch")
    toolchain = payload.get("toolchain", {})
    if toolchain.get("required_pins") != TOOLCHAIN_PINS:
        issues.append("toolchain_pins_missing_or_mismatched")
    if not isinstance(toolchain.get("cmake"), str) or not toolchain["cmake"].strip():
        issues.append("cmake_version_missing")
    if not isinstance(toolchain.get("compiler"), str) or not toolchain["compiler"].strip():
        issues.append("compiler_version_missing")
    evidence = toolchain.get("target_native_evidence")
    if not isinstance(evidence, dict):
        issues.append("target_native_evidence_missing")
    elif target_id in TARGETS:
        target_os = TARGETS[target_id]["os"]
        required_evidence = {
            "windows": ("vswhere_version",),
            "linux": ("container_digest",),
            "darwin": ("xcodebuild_version", "sdk"),
        }[target_os]
        for key in required_evidence:
            value = evidence.get(key)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"target_native_{key}_missing")
    telemetry = payload.get("telemetry", {})
    if telemetry.get("status") != "not_verified":
        issues.append("telemetry_status_must_remain_not_verified")
    if telemetry.get("build_flag") != "onnxruntime_USE_TELEMETRY=OFF":
        issues.append("telemetry_build_flag_missing")
    artifact = payload.get("artifact", {})
    artifact_path = Path(artifact.get("path", ""))
    if check_files:
        if not artifact_path.is_file():
            issues.append("artifact_missing")
        elif artifact.get("sha256") != _sha256(artifact_path):
            issues.append("artifact_hash_mismatch")
    if not isinstance(telemetry.get("evidence"), list):
        issues.append("telemetry_evidence_must_be_list")
    return issues


def _write_or_print(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="emit pinned target-native build commands")
    plan_parser.add_argument("--target", choices=tuple(TARGETS))
    plan_parser.add_argument("--output", type=Path)

    attest_parser = subparsers.add_parser("attest", help="record an unverified build attestation")
    attest_parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    attest_parser.add_argument("--artifact", type=Path, required=True)
    attest_parser.add_argument("--source-root", type=Path, required=True)
    attest_parser.add_argument("--output", type=Path)

    verify_parser = subparsers.add_parser("verify-attestation", help="validate an attestation without network")
    verify_parser.add_argument("path", type=Path)
    verify_parser.add_argument("--no-file-check", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "plan":
        _write_or_print(build_plan(args.target), args.output)
        return 0
    if args.command == "attest":
        try:
            payload = make_attestation(args.target, args.artifact, args.source_root)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        _write_or_print(payload, args.output)
        return 0
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    issues = validate_attestation(payload, check_files=not args.no_file_check)
    report = {"ok": not issues, "path": str(args.path), "issues": issues}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
