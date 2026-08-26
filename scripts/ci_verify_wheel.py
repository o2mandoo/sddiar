#!/usr/bin/env python3
"""Verify a freshly built wheel in an isolated, network-disabled install.

The script is intentionally cross-platform and uses no shell.  It validates
wheel/source module parity, metadata version, runner architecture, isolated
``pip --no-index --no-deps`` installation, package import, and CLI startup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile


def _python_in(venv_root: Path) -> Path:
    candidate = venv_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not candidate.is_file():
        raise RuntimeError("isolated Python executable is missing")
    return candidate


def _script_in(venv_root: Path) -> Path:
    candidate = venv_root / ("Scripts/sddiar.exe" if os.name == "nt" else "bin/sddiar")
    if not candidate.is_file():
        raise RuntimeError("isolated sddiar console script is missing")
    return candidate


def _architecture_matches(expected: str) -> bool:
    aliases = {
        "x86_64": {"x86_64", "amd64"},
        "arm64": {"arm64", "aarch64"},
    }
    return platform.machine().lower() in aliases.get(expected, {expected})


def verify(wheel_dir: Path, source_root: Path, expected_version: str, expected_arch: str) -> dict:
    wheels = sorted(wheel_dir.glob(f"sddiar-{expected_version}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("exactly one expected sddiar wheel is required")
    if not _architecture_matches(expected_arch):
        raise RuntimeError("runner architecture does not match the declared CI lane")
    wheel = wheels[0]
    with ZipFile(wheel) as archive:
        wheel_modules = sorted(
            name for name in archive.namelist()
            if name.startswith("sddiar/") and name.endswith(".py")
        )
        source_modules = sorted(
            f"sddiar/{path.name}" for path in (source_root / "sddiar").glob("*.py")
        )
        if wheel_modules != source_modules:
            raise RuntimeError("wheel/source module inventory mismatch")
        for name in wheel_modules:
            if archive.read(name) != (source_root / name).read_bytes():
                raise RuntimeError("wheel/source module content mismatch")
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError("wheel metadata is missing or ambiguous")
        metadata = archive.read(metadata_names[0]).decode("utf-8", errors="strict")
        if f"Version: {expected_version}\n" not in metadata.replace("\r\n", "\n"):
            raise RuntimeError("wheel metadata version mismatch")

    temp_parent = os.environ.get("RUNNER_TEMP")
    if temp_parent is None and sys.platform == "darwin" and Path("/private/tmp").is_dir():
        # Some relocatable local CPython builds cannot create a working venv
        # below macOS's /var/folders symlink tree. GitHub's setup-python does
        # not have that defect, but the local verification path stays real.
        temp_parent = "/private/tmp"
    with tempfile.TemporaryDirectory(prefix="sddiar-ci-install-", dir=temp_parent) as directory:
        isolated_root = Path(directory) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True, symlinks=os.name != "nt").create(isolated_root)
        isolated_python = _python_in(isolated_root)
        subprocess.run(
            [
                str(isolated_python), "-m", "pip", "install",
                "--disable-pip-version-check", "--no-index", "--no-deps", str(wheel),
            ],
            check=True,
        )
        probe = (
            "import importlib.metadata,json,platform,sddiar;"
            f"assert importlib.metadata.version('sddiar') == {expected_version!r};"
            "from sddiar.production_orchestrator import ProductionOrchestrator;"
            "from sddiar.whispercpp_backend import WHISPER_CPP_COMMIT;"
            "print(json.dumps({'architecture':platform.machine().lower(),"
            "'version':importlib.metadata.version('sddiar'),"
            "'orchestrator':ProductionOrchestrator.__name__,"
            "'whisper_commit':WHISPER_CPP_COMMIT},sort_keys=True))"
        )
        completed = subprocess.run(
            [str(isolated_python), "-I", "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [str(_script_in(isolated_root)), "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return {
        "schema": "sddiar-ci-wheel-verification-v1",
        "version": expected_version,
        "architecture": platform.machine().lower(),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "module_count": len(wheel_modules),
        "isolated_probe": json.loads(completed.stdout),
        "network_install": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-arch", choices=("x86_64", "arm64"), required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(
            args.wheel_dir.resolve(), args.source_root.resolve(),
            args.expected_version, args.expected_arch,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
