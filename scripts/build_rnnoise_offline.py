#!/usr/bin/env python3
"""Plan and attest an imported, network-disabled RNNoise demo build.

The helper never downloads source, model data, submodules, or toolchains.  In
particular, it never executes upstream ``autogen.sh`` because that script calls
``download_model.sh``.  The model archive must be imported separately, checked
against the pinned ``model_version`` SHA-256, and staged with this helper before
the ordinary ``autoreconf``/``configure``/``make`` argv sequences are run.

An emitted attestation remains experimental and non-production.  It records a
single target-native build but cannot claim four-platform or Xeon validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SOURCE_URL = "https://gitlab.xiph.org/xiph/rnnoise.git"
SOURCE_COMMIT = "70f1d256acd4b34a572f999a05c87bf00b67730d"
MODEL_TAR_SHA256 = "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"
MODEL_TAR_NAME = f"rnnoise_data-{MODEL_TAR_SHA256}.tar.gz"
LICENSE_SPDX = "BSD-3-Clause"
MAX_MODEL_EXPANDED_BYTES = 512 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_BUILD_TOKENS = (
    "autogen.sh",
    "download_model.sh",
    "wget ",
    "curl ",
    "invoke-webrequest",
    "start-bitstransfer",
)


TARGETS: dict[str, dict[str, Any]] = {
    "windows-x86_64": {
        "system": "Windows",
        "machines": ("amd64", "x86_64"),
        "arch": "x86_64",
        "binary_name": "rnnoise_demo.exe",
        "configure_prefix": ["sh", "./configure", "--host=x86_64-w64-mingw32"],
        "toolchain_roles": (
            "c_compiler",
            "mingw_w64",
            "posix_shell",
            "autoconf",
            "automake",
            "libtool",
            "make",
        ),
    },
    "linux-x86_64": {
        "system": "Linux",
        "machines": ("x86_64", "amd64"),
        "arch": "x86_64",
        "binary_name": "rnnoise_demo",
        "configure_prefix": ["./configure"],
        "toolchain_roles": ("c_compiler", "sysroot", "autoconf", "automake", "libtool", "make"),
    },
    "linux-aarch64": {
        "system": "Linux",
        "machines": ("aarch64", "arm64"),
        "arch": "aarch64",
        "binary_name": "rnnoise_demo",
        "configure_prefix": ["./configure"],
        "toolchain_roles": ("c_compiler", "sysroot", "autoconf", "automake", "libtool", "make"),
    },
    "macos-x86_64": {
        "system": "Darwin",
        "machines": ("x86_64", "amd64"),
        "arch": "x86_64",
        "binary_name": "rnnoise_demo",
        "configure_prefix": ["./configure"],
        "toolchain_roles": ("c_compiler", "macos_sdk", "autoconf", "automake", "libtool", "make"),
    },
    "macos-arm64": {
        "system": "Darwin",
        "machines": ("arm64", "aarch64"),
        "arch": "arm64",
        "binary_name": "rnnoise_demo",
        "configure_prefix": ["./configure"],
        "toolchain_roles": ("c_compiler", "macos_sdk", "autoconf", "automake", "libtool", "make"),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    return {"logical_name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _build_commands(target: str) -> list[list[str]]:
    if target not in TARGETS:
        raise ValueError(f"unsupported target: {target}")
    spec = TARGETS[target]
    configure = [
        *spec["configure_prefix"],
        "--disable-doc",
        "--enable-examples",
        "--disable-x86-rtcd",
        "--disable-shared",
        "--enable-static",
    ]
    binary_target = f"examples/{spec['binary_name']}"
    return [["autoreconf", "-isf"], configure, ["make", "-j1", binary_target]]


def build_plan(target: str | None = None) -> dict[str, Any]:
    selected = [target] if target else list(TARGETS)
    if any(item not in TARGETS for item in selected):
        raise ValueError("unknown RNNoise target")
    payload = {
        "schema_version": "1.0",
        "kind": "rnnoise-offline-source-build-plan",
        "experimental": True,
        "default_enabled": False,
        "production_approved": False,
        "source": {
            "repository": SOURCE_URL,
            "commit": SOURCE_COMMIT,
            "license_spdx": LICENSE_SPDX,
            "required_imports": {
                "source_archive": "caller-supplied exact SHA-256 is mandatory",
                "source_checkout": "tracked-tree SHA-256 and exact git commit are mandatory",
                "submodules": "recursive status and canonical SHA-256 are mandatory; explicit zero is allowed",
            },
        },
        "model": {
            "archive_name": MODEL_TAR_NAME,
            "archive_sha256": MODEL_TAR_SHA256,
            "model_version": MODEL_TAR_SHA256,
            "ingress": "imported and hash-verified before network-disabled build",
            "extraction": "safe regular-file-only staging by this helper",
        },
        "download_prevention": {
            "autogen_sh": "PROHIBITED",
            "download_model_sh": "PROHIBITED",
            "replacement": ["stage-model", "autoreconf -isf"],
            "build_network": "disabled by external OS/container policy",
            "runtime_network": "none",
        },
        "reproduction_profile": {
            "rnnoise_mode": "x86-rtcd-disabled-target-compiler-vectorization",
            "x86_runtime_dispatch": False,
            "scalar_equivalence": "not_claimed; SSE2/AVX or NEON may be selected at compile time",
            "input_output": "raw signed 16-bit little-endian mono PCM at 48000 Hz",
            "demo_discarded_warmup_output_samples": 480,
        },
        "targets": {
            name: {
                "system": TARGETS[name]["system"],
                "arch": TARGETS[name]["arch"],
                "target_native_required": True,
                "cross_build_is_not_target_validation": True,
                "required_toolchain_roles": list(TARGETS[name]["toolchain_roles"]),
                "commands": _build_commands(name),
                "native_binary": f"examples/{TARGETS[name]['binary_name']}",
                "attestation_inputs": [
                    "source archive + ingress SHA-256",
                    "checked-out tracked-tree SHA-256",
                    "recursive submodule status SHA-256",
                    "model tar exact SHA-256",
                    "toolchain manifest + exact SHA-256",
                    "build log + exact SHA-256",
                    "linked-dependency report + exact SHA-256",
                    "native demo binary SHA-256",
                ],
            }
            for name in selected
        },
        "approval_boundary": {
            "four_platform_native_validation": "not_run",
            "xeon_validation": "not_run",
            "resampler": "separate hash-verified native artifact and attestation required",
            "model_redistribution": "separate review required; source-code license is not inferred for model data",
            "release_authority": "none",
        },
    }
    return payload


def _safe_model_members(model_tar: Path) -> list[tarfile.TarInfo]:
    if model_tar.name != MODEL_TAR_NAME or _sha256(model_tar) != MODEL_TAR_SHA256:
        raise ValueError("model archive name or SHA-256 does not match pinned model_version")
    members: list[tarfile.TarInfo] = []
    expanded = 0
    with tarfile.open(model_tar, mode="r:gz") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or not pure.parts
                or any(part in ("", ".", "..") for part in pure.parts)
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("model archive contains an unsafe member")
            if member.isfile():
                expanded += member.size
                if expanded > MAX_MODEL_EXPANDED_BYTES:
                    raise ValueError("model archive exceeds the expanded-byte limit")
            members.append(member)
    file_names = {PurePosixPath(member.name).as_posix() for member in members if member.isfile()}
    if not {"src/rnnoise_data.c", "src/rnnoise_data.h"}.issubset(file_names):
        raise ValueError("model archive is missing required RNNoise C data files")
    return members


def stage_model(model_tar: Path, source_root: Path) -> dict[str, Any]:
    """Safely stage only pinned model members into an imported checkout."""

    if source_root.is_symlink():
        raise ValueError("source root must be a real directory")
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("source root must be a real directory")
    version_path = source_root / "model_version"
    if not version_path.is_file() or version_path.read_text(encoding="utf-8").strip() != MODEL_TAR_SHA256:
        raise ValueError("source checkout model_version does not match the pinned model archive")
    members = _safe_model_members(model_tar.resolve(strict=True))
    staged: list[dict[str, Any]] = []
    with tarfile.open(model_tar, mode="r:gz") as archive:
        for member in members:
            destination = source_root.joinpath(*PurePosixPath(member.name).parts)
            resolved_parent = destination.parent.resolve(strict=True)
            if resolved_parent != source_root and source_root not in resolved_parent.parents:
                raise ValueError("model archive member escapes the source checkout")
            if member.isdir():
                continue
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("model staging refuses to overwrite an existing checkout file")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("model archive regular file could not be read")
            with extracted, destination.open("xb") as output:
                for block in iter(lambda: extracted.read(1 << 20), b""):
                    output.write(block)
            if os.name != "nt":
                destination.chmod(0o600)
            staged.append(
                {
                    "relative_path": PurePosixPath(member.name).as_posix(),
                    "sha256": _sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
    payload = {
        "ok": True,
        "model_archive_sha256": MODEL_TAR_SHA256,
        "staged_files": sorted(staged, key=lambda item: item["relative_path"]),
    }
    return payload


def _run(command: Sequence[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot inspect imported RNNoise checkout") from exc
    return completed.stdout.strip()


def _git_commit(source_root: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], source_root)


def _submodules(source_root: Path) -> list[dict[str, str]]:
    output = _run(["git", "submodule", "status", "--recursive"], source_root)
    records: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        state = line[0] if line[0] in " -+U" else "?"
        fields = line[1:].strip().split(maxsplit=2)
        if len(fields) < 2:
            raise ValueError("malformed recursive submodule status")
        records.append({"state": state, "commit": fields[0], "path": fields[1]})
    return records


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tracked_tree_sha256(source_root: Path) -> str:
    output = subprocess.run(
        ["git", "ls-files", "-z", "--recurse-submodules"],
        cwd=source_root,
        check=True,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout
    digest = hashlib.sha256()
    for raw_name in output.split(b"\0"):
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        path = (source_root / name).resolve(strict=True)
        if path != source_root and source_root not in path.parents:
            raise ValueError("tracked source file escapes checkout")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise ValueError("tracked source tree contains a non-regular entry")
        digest.update(raw_name)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _tracked_checkout_clean(source_root: Path) -> bool:
    for command in (
        ["git", "diff", "--quiet", "HEAD", "--"],
        ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
    ):
        completed = subprocess.run(
            command,
            cwd=source_root,
            check=False,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            return False
    return True


def _verify_staged_model(model_tar: Path, source_root: Path) -> list[dict[str, Any]]:
    members = _safe_model_members(model_tar)
    records: list[dict[str, Any]] = []
    with tarfile.open(model_tar, mode="r:gz") as archive:
        for member in members:
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name)
            target = source_root.joinpath(*relative.parts)
            if not target.is_file() or target.is_symlink() or target.stat().st_size != member.size:
                raise ValueError("staged model file is missing or changed")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("model archive regular file could not be read")
            digest = hashlib.sha256()
            with extracted:
                for block in iter(lambda: extracted.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != _sha256(target):
                raise ValueError("staged model file does not match the pinned model tar")
            records.append(
                {
                    "relative_path": relative.as_posix(),
                    "sha256": digest.hexdigest(),
                    "bytes": member.size,
                }
            )
    return sorted(records, key=lambda item: item["relative_path"])


def _validate_toolchain_manifest(payload: Any, target: str) -> list[str]:
    issues: list[str] = []
    if not isinstance(payload, Mapping):
        return ["toolchain_manifest_not_object"]
    if payload.get("schema_version") != "1.0":
        issues.append("toolchain_schema_version_mismatch")
    if payload.get("target") != target:
        issues.append("toolchain_target_mismatch")
    if payload.get("production_approved") is not False:
        issues.append("toolchain_production_approved_must_be_false")
    components = payload.get("components")
    if not isinstance(components, list):
        return issues + ["toolchain_components_missing"]
    roles: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping):
            issues.append("toolchain_component_not_object")
            continue
        role = component.get("role")
        version = component.get("version")
        digest = component.get("sha256")
        if not isinstance(role, str) or not role:
            issues.append("toolchain_component_role_missing")
        else:
            roles.add(role)
        if not isinstance(version, str) or not version.strip():
            issues.append("toolchain_component_version_missing")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            issues.append("toolchain_component_sha256_invalid")
    missing = set(TARGETS[target]["toolchain_roles"]) - roles
    if missing:
        issues.append("toolchain_required_roles_missing:" + ",".join(sorted(missing)))
    return issues


def _check_native_host(target: str) -> dict[str, Any]:
    spec = TARGETS[target]
    system = platform.system()
    machine = platform.machine().lower()
    matches = system == spec["system"] and machine in spec["machines"]
    if not matches:
        raise ValueError("attestation must be generated on the exact target-native host")
    return {"system": system, "machine": machine, "matches_target": True}


def _verified_input(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} SHA-256 mismatch")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise ValueError(f"{label} must be a non-empty regular file")
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
            after = os.fstat(stream.fileno())
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely") from exc
    if (
        digest.hexdigest() != expected_sha256
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"{label} SHA-256 mismatch")
    return {"logical_name": path.name, "sha256": expected_sha256, "bytes": before.st_size}


def _verified_small_input(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    record = _verified_input(path, expected_sha256, label=label)
    if record["bytes"] > max_bytes:
        raise ValueError(f"{label} exceeds its byte bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.resolve(strict=True), flags)
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read(max_bytes + 1)
            info = os.fstat(stream.fileno())
    except OSError as exc:
        raise ValueError(f"{label} could not be read safely") from exc
    if (
        len(raw) != record["bytes"]
        or info.st_size != record["bytes"]
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError(f"{label} changed after verification")
    return record, raw


def make_attestation(
    target: str,
    *,
    source_root: Path,
    source_archive: Path,
    source_archive_sha256: str,
    model_tar: Path,
    toolchain_manifest: Path,
    toolchain_manifest_sha256: str,
    build_log: Path,
    build_log_sha256: str,
    dependency_report: Path,
    dependency_report_sha256: str,
    native_binary: Path,
) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError("unknown RNNoise target")
    host = _check_native_host(target)
    if source_root.is_symlink():
        raise ValueError("RNNoise source root may not be a symlink")
    source_root = source_root.resolve(strict=True)
    if _git_commit(source_root) != SOURCE_COMMIT:
        raise ValueError("RNNoise checkout is not at the pinned commit")
    if not _tracked_checkout_clean(source_root):
        raise ValueError("RNNoise checkout has modified tracked files")
    version = (source_root / "model_version").read_text(encoding="utf-8").strip()
    if version != MODEL_TAR_SHA256:
        raise ValueError("source model_version does not match the pinned model tar")
    source_record = _verified_input(source_archive, source_archive_sha256, label="source archive")
    model_record = _verified_input(model_tar, MODEL_TAR_SHA256, label="model archive")
    staged_model_files = _verify_staged_model(model_tar, source_root)
    toolchain_record, toolchain_raw = _verified_small_input(
        toolchain_manifest,
        toolchain_manifest_sha256,
        label="toolchain manifest",
        max_bytes=1024 * 1024,
    )
    toolchain_payload = json.loads(toolchain_raw.decode("utf-8"))
    toolchain_issues = _validate_toolchain_manifest(toolchain_payload, target)
    if toolchain_issues:
        raise ValueError("invalid toolchain manifest: " + ";".join(toolchain_issues))
    build_log_record, build_log_raw = _verified_small_input(
        build_log,
        build_log_sha256,
        label="build log",
        max_bytes=16 * 1024 * 1024,
    )
    build_log_text = build_log_raw.decode("utf-8", errors="replace").lower()
    if any(token in build_log_text for token in _FORBIDDEN_BUILD_TOKENS):
        raise ValueError("build log contains a prohibited download/autogen command")
    dependency_record = _verified_input(
        dependency_report, dependency_report_sha256, label="dependency report"
    )
    if native_binary.is_symlink():
        raise ValueError("native binary may not be a symlink")
    binary = native_binary.resolve(strict=True)
    expected_binary = (source_root / "examples" / TARGETS[target]["binary_name"]).resolve(strict=True)
    if binary != expected_binary or binary.name != TARGETS[target]["binary_name"] or not binary.is_file():
        raise ValueError("native binary name/type does not match the target")
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise ValueError("native binary is not executable")
    binary_record = _verified_input(
        binary,
        _sha256(binary),
        label="native RNNoise binary",
    )
    submodules = _submodules(source_root)
    for record in submodules:
        if record["state"] != " " or _COMMIT.fullmatch(record["commit"]) is None:
            raise ValueError("submodule checkout is not clean and pinned")
    commands = _build_commands(target)
    payload = {
        "schema_version": "1.0",
        "kind": "rnnoise-offline-build-attestation",
        "experimental": True,
        "default_enabled": False,
        "production_approved": False,
        "source": {
            "repository": SOURCE_URL,
            "expected_commit": SOURCE_COMMIT,
            "checkout_commit": SOURCE_COMMIT,
            "source_archive": source_record,
            "tracked_tree_sha256": _tracked_tree_sha256(source_root),
            "license_spdx": LICENSE_SPDX,
        },
        "submodules": {
            "records": submodules,
            "count": len(submodules),
            "canonical_sha256": _canonical_sha256(submodules),
            "explicit_zero_allowed": True,
        },
        "model": {
            "archive": model_record,
            "expected_sha256": MODEL_TAR_SHA256,
            "model_version": version,
            "imported_offline": True,
            "staged_files": staged_model_files,
        },
        "target": {"id": target, "arch": TARGETS[target]["arch"], "endianness": "little"},
        "host": host,
        "toolchain": {
            "manifest": toolchain_record,
            "manifest_payload_sha256": _canonical_sha256(toolchain_payload),
            "required_roles": list(TARGETS[target]["toolchain_roles"]),
        },
        "configuration": {
            "commands": commands,
            "autogen_executed": False,
            "download_model_executed": False,
            "x86_rtcd": False,
            "compile_time_vectorization": "target_compiler_default_not_scalar_claim",
            "jobs": 1,
            "build_network_required_state": "disabled",
        },
        "native_binary": binary_record,
        "build_log": build_log_record,
        "dependency_report": dependency_record,
        "validation": {
            "single_target_native_build_recorded": True,
            "binary_functional_smoke": "not_run",
            "four_platform_native_validation": "not_run",
            "xeon_validation": "not_run",
            "independent_review": "not_run",
            "release_authority": "none",
        },
        "redaction": {"local_paths": "omitted", "environment": "omitted"},
    }
    _attach_integrity(payload)
    return payload


def _integrity_values(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    inputs = {
        key: payload.get(key)
        for key in ("source", "submodules", "model", "target", "host", "toolchain", "configuration")
    }
    outputs = {key: payload.get(key) for key in ("native_binary", "build_log", "dependency_report")}
    inputs_sha256 = _canonical_sha256(inputs)
    outputs_sha256 = _canonical_sha256(outputs)
    statement_sha256 = _canonical_sha256(
        {
            "schema_version": payload.get("schema_version"),
            "kind": payload.get("kind"),
            "inputs_sha256": inputs_sha256,
            "outputs_sha256": outputs_sha256,
            "validation": payload.get("validation"),
        }
    )
    return inputs_sha256, outputs_sha256, statement_sha256


def _attach_integrity(payload: dict[str, Any]) -> None:
    inputs_sha256, outputs_sha256, statement_sha256 = _integrity_values(payload)
    payload["integrity"] = {
        "build_inputs_sha256": inputs_sha256,
        "build_outputs_sha256": outputs_sha256,
        "statement_sha256": statement_sha256,
        "signature_status": "unsigned",
        "verification_scope": "hash_bound_structure_only",
        "cryptographic_authenticity": "not_verified",
    }


def _descriptor_issues(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    logical_name = value.get("logical_name")
    return (
        not isinstance(logical_name, str)
        or not logical_name
        or Path(logical_name).name != logical_name
        or "\\" in logical_name
        or _SHA256.fullmatch(str(value.get("sha256", ""))) is None
        or type(value.get("bytes")) is not int
        or value.get("bytes") <= 0
    )


def validate_attestation(payload: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(payload, Mapping):
        return ["attestation_not_object"]
    if payload.get("schema_version") != "1.0":
        issues.append("schema_version_mismatch")
    if payload.get("kind") != "rnnoise-offline-build-attestation":
        issues.append("kind_mismatch")
    if payload.get("experimental") is not True:
        issues.append("experimental_flag_required")
    if payload.get("default_enabled") is not False:
        issues.append("default_enabled_must_be_false")
    if payload.get("production_approved") is not False:
        issues.append("production_approved_must_be_false")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        issues.append("source_missing")
        source = {}
    if source.get("repository") != SOURCE_URL:
        issues.append("source_repository_mismatch")
    if source.get("expected_commit") != SOURCE_COMMIT or source.get("checkout_commit") != SOURCE_COMMIT:
        issues.append("source_commit_mismatch")
    if source.get("license_spdx") != LICENSE_SPDX:
        issues.append("source_license_mismatch")
    if _SHA256.fullmatch(str(source.get("tracked_tree_sha256", ""))) is None:
        issues.append("source_tree_sha256_invalid")
    source_archive = source.get("source_archive")
    if _descriptor_issues(source_archive):
        issues.append("source_archive_sha256_invalid")
    submodules = payload.get("submodules")
    if not isinstance(submodules, Mapping):
        issues.append("submodules_missing")
    else:
        records = submodules.get("records")
        if not isinstance(records, list) or submodules.get("count") != len(records):
            issues.append("submodule_count_mismatch")
        else:
            for record in records:
                if (
                    not isinstance(record, Mapping)
                    or record.get("state") != " "
                    or _COMMIT.fullmatch(str(record.get("commit", ""))) is None
                    or not isinstance(record.get("path"), str)
                    or "\\" in str(record.get("path", ""))
                    or PurePosixPath(str(record.get("path"))).is_absolute()
                    or ".." in PurePosixPath(str(record.get("path"))).parts
                ):
                    issues.append("submodule_record_invalid")
                    break
            if submodules.get("canonical_sha256") != _canonical_sha256(records):
                issues.append("submodule_canonical_sha256_mismatch")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        issues.append("model_missing")
    else:
        archive = model.get("archive")
        if (
            model.get("expected_sha256") != MODEL_TAR_SHA256
            or model.get("model_version") != MODEL_TAR_SHA256
            or model.get("imported_offline") is not True
            or not isinstance(archive, Mapping)
            or archive.get("sha256") != MODEL_TAR_SHA256
            or archive.get("logical_name") != MODEL_TAR_NAME
            or _descriptor_issues(archive)
        ):
            issues.append("model_pin_mismatch")
        staged_files = model.get("staged_files")
        if not isinstance(staged_files, list) or not staged_files:
            issues.append("staged_model_files_missing")
        else:
            staged_names = {
                record.get("relative_path")
                for record in staged_files
                if isinstance(record, Mapping) and not _descriptor_issues(
                    {
                        "logical_name": Path(str(record.get("relative_path", ""))).name,
                        "sha256": record.get("sha256"),
                        "bytes": record.get("bytes"),
                    }
                )
            }
            if not {"src/rnnoise_data.c", "src/rnnoise_data.h"}.issubset(staged_names):
                issues.append("staged_model_files_invalid")
    target = payload.get("target")
    target_id = target.get("id") if isinstance(target, Mapping) else None
    if target_id not in TARGETS:
        issues.append("unknown_target")
    elif target.get("arch") != TARGETS[target_id]["arch"] or target.get("endianness") != "little":
        issues.append("target_contract_mismatch")
    host = payload.get("host")
    if not isinstance(host, Mapping) or host.get("matches_target") is not True:
        issues.append("target_native_host_missing")
    elif target_id in TARGETS:
        if (
            host.get("system") != TARGETS[target_id]["system"]
            or str(host.get("machine", "")).lower() not in TARGETS[target_id]["machines"]
        ):
            issues.append("target_native_host_mismatch")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        issues.append("configuration_missing")
    elif target_id in TARGETS:
        if configuration.get("commands") != _build_commands(target_id):
            issues.append("build_commands_mismatch")
        if configuration.get("autogen_executed") is not False:
            issues.append("autogen_must_not_execute")
        if configuration.get("download_model_executed") is not False:
            issues.append("download_model_must_not_execute")
        if configuration.get("x86_rtcd") is not False or configuration.get("jobs") != 1:
            issues.append("x86_rtcd_disabled_single_job_configuration_required")
        if configuration.get("compile_time_vectorization") != "target_compiler_default_not_scalar_claim":
            issues.append("compile_time_vectorization_claim_invalid")
        if configuration.get("build_network_required_state") != "disabled":
            issues.append("build_network_not_disabled")
        flattened = " ".join(token for command in configuration.get("commands", []) for token in command).lower()
        if any(token.strip() in flattened for token in _FORBIDDEN_BUILD_TOKENS):
            issues.append("forbidden_download_or_autogen_command")
    toolchain = payload.get("toolchain")
    if not isinstance(toolchain, Mapping):
        issues.append("toolchain_missing")
    elif target_id in TARGETS:
        manifest = toolchain.get("manifest")
        if _descriptor_issues(manifest):
            issues.append("toolchain_manifest_sha256_invalid")
        if _SHA256.fullmatch(str(toolchain.get("manifest_payload_sha256", ""))) is None:
            issues.append("toolchain_manifest_payload_sha256_invalid")
        if toolchain.get("required_roles") != list(TARGETS[target_id]["toolchain_roles"]):
            issues.append("toolchain_roles_mismatch")
    for key in ("native_binary", "build_log", "dependency_report"):
        descriptor = payload.get(key)
        if _descriptor_issues(descriptor):
            issues.append(f"{key}_descriptor_invalid")
        elif (
            not isinstance(descriptor.get("logical_name"), str)
            or Path(descriptor["logical_name"]).name != descriptor["logical_name"]
            or "\\" in descriptor["logical_name"]
        ):
            issues.append(f"{key}_logical_name_invalid")
    native = payload.get("native_binary")
    if (
        target_id in TARGETS
        and isinstance(native, Mapping)
        and native.get("logical_name") != TARGETS[target_id]["binary_name"]
    ):
        issues.append("native_binary_name_mismatch")
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        issues.append("validation_missing")
    else:
        if validation.get("single_target_native_build_recorded") is not True:
            issues.append("single_target_native_build_record_must_be_true")
        if validation.get("binary_functional_smoke") != "not_run":
            issues.append("binary_functional_smoke_must_remain_not_run")
        if validation.get("four_platform_native_validation") != "not_run":
            issues.append("four_platform_validation_must_remain_not_run")
        if validation.get("xeon_validation") != "not_run":
            issues.append("xeon_validation_must_remain_not_run")
        if validation.get("independent_review") != "not_run" or validation.get("release_authority") != "none":
            issues.append("release_authority_claim_prohibited")
    integrity = payload.get("integrity")
    if not isinstance(integrity, Mapping):
        issues.append("integrity_missing")
    else:
        expected_inputs, expected_outputs, expected_statement = _integrity_values(payload)
        if integrity.get("build_inputs_sha256") != expected_inputs:
            issues.append("build_inputs_sha256_mismatch")
        if integrity.get("build_outputs_sha256") != expected_outputs:
            issues.append("build_outputs_sha256_mismatch")
        if integrity.get("statement_sha256") != expected_statement:
            issues.append("statement_sha256_mismatch")
        if (
            integrity.get("signature_status") != "unsigned"
            or integrity.get("verification_scope") != "hash_bound_structure_only"
            or integrity.get("cryptographic_authenticity") != "not_verified"
        ):
            issues.append("integrity_scope_overclaim")
    return issues


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan", help="emit the pinned offline build plan")
    plan_parser.add_argument("--target", choices=tuple(TARGETS))
    plan_parser.add_argument("--output", type=Path)

    stage_parser = commands.add_parser("stage-model", help="safely stage the imported pinned model tar")
    stage_parser.add_argument("--model-tar", type=Path, required=True)
    stage_parser.add_argument("--source-root", type=Path, required=True)
    stage_parser.add_argument("--output", type=Path)

    attest_parser = commands.add_parser("attest", help="record one non-production target-native build")
    attest_parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    attest_parser.add_argument("--source-root", type=Path, required=True)
    attest_parser.add_argument("--source-archive", type=Path, required=True)
    attest_parser.add_argument("--source-archive-sha256", required=True)
    attest_parser.add_argument("--model-tar", type=Path, required=True)
    attest_parser.add_argument("--toolchain-manifest", type=Path, required=True)
    attest_parser.add_argument("--toolchain-manifest-sha256", required=True)
    attest_parser.add_argument("--build-log", type=Path, required=True)
    attest_parser.add_argument("--build-log-sha256", required=True)
    attest_parser.add_argument("--dependency-report", type=Path, required=True)
    attest_parser.add_argument("--dependency-report-sha256", required=True)
    attest_parser.add_argument("--native-binary", type=Path, required=True)
    attest_parser.add_argument("--output", type=Path)

    verify_parser = commands.add_parser(
        "verify-attestation", help="hash-bind and lint an unsigned redacted attestation offline"
    )
    verify_parser.add_argument("path", type=Path)
    verify_parser.add_argument("--expected-sha256", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            payload = build_plan(args.target)
        elif args.command == "stage-model":
            payload = stage_model(args.model_tar, args.source_root)
        elif args.command == "attest":
            payload = make_attestation(
                args.target,
                source_root=args.source_root,
                source_archive=args.source_archive,
                source_archive_sha256=args.source_archive_sha256,
                model_tar=args.model_tar,
                toolchain_manifest=args.toolchain_manifest,
                toolchain_manifest_sha256=args.toolchain_manifest_sha256,
                build_log=args.build_log,
                build_log_sha256=args.build_log_sha256,
                dependency_report=args.dependency_report,
                dependency_report_sha256=args.dependency_report_sha256,
                native_binary=args.native_binary,
            )
        else:
            raw = args.path.read_bytes()
            outer_hash_ok = _SHA256.fullmatch(args.expected_sha256) is not None and hashlib.sha256(raw).hexdigest() == args.expected_sha256
            payload = json.loads(raw.decode("utf-8"))
            issues = validate_attestation(payload)
            if not outer_hash_ok:
                issues.append("attestation_file_sha256_mismatch")
            print(
                json.dumps(
                    {
                        "ok": not issues,
                        "issues": issues,
                        "verification_scope": "caller_hash_bound_structure_only",
                        "cryptographic_authenticity": "not_verified",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if not issues else 1
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        parser.error(str(exc))
    _write_or_print(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
