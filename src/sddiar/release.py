"""Offline release layout and integrity validation (stdlib only).

This module validates an already assembled release; it deliberately does not
download, resolve, install, or execute any release artifact.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

TARGETS = ("windows-x64-cp311", "linux-x64-cp311", "macos-arm64-cp311", "macos-x64-cp311")
REQUIRED_DIRS = ("wheelhouse", "model", "native")
# SDD names these wheels/models as ``wheels``/``models`` and SBOM/notices as
# directories.  The aliases keep intake compatible with both spellings while
# retaining one strict semantic requirement per artifact class.
DIR_ALIASES = {"wheelhouse": ("wheelhouse", "wheels"), "model": ("model", "models"), "native": ("native",)}
FILE_OR_DIR_ALIASES = {
    "sbom": ("SBOM.spdx.json", "sbom.cdx.json", "sbom.json", "sbom"),
    "notice": ("NOTICE", "notices"),
}
LOCKFILE_NAMES = ("requirements.lock", "requirements.txt")
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.I)
_PRODUCTION_MODES = ("development", "production")
_TRUSTED_STATUSES = frozenset(("verified", "independently_verified", "independent_verified", "pass", "passed", "trusted"))
_ORT_VERIFIED_STATUSES = frozenset(("verified", "independently_verified", "independent_verified", "pass", "passed"))


@dataclass(frozen=True)
class ReleaseIssue:
    code: str
    message: str
    path: str | None = None


@dataclass
class ReleaseReport:
    root: Path
    targets: tuple[str, ...]
    mode: str = "development"
    issues: list[ReleaseIssue] = field(default_factory=list)
    scanned_files: int = 0
    verified_files: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> "ReleaseReport":
        if self.issues:
            raise ReleaseValidationError(self.issues)
        return self


class ReleaseValidationError(ValueError):
    def __init__(self, issues: Sequence[ReleaseIssue]):
        self.issues = tuple(issues)
        super().__init__("offline release validation failed: " + "; ".join(i.code for i in self.issues))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(root: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    for name in ("release-catalog.json", "release-manifest.json", "manifest.json"):
        path = root / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, Mapping) else None, name
            except (OSError, UnicodeError, json.JSONDecodeError):
                return None, name
    return None, None


def _manifest_entries(manifest: Mapping[str, Any], target: str) -> list[Mapping[str, Any]]:
    """Accept both {targets:{name:{files:[]}}} and flat files manifests."""
    targets = manifest.get("targets")
    section: Any = targets.get(target) if isinstance(targets, Mapping) else None
    if section is None and isinstance(manifest.get(target), Mapping):
        section = manifest[target]
    if section is None:
        return []
    files = section.get("files", section) if isinstance(section, Mapping) else section
    return [x for x in files if isinstance(x, Mapping)] if isinstance(files, list) else []


def _target_root(root: Path, target: str) -> Path:
    candidate = root / "targets" / target
    return candidate if candidate.exists() else root / target


def _first_existing(parent: Path, names: Sequence[str]) -> Path | None:
    return next((parent / name for name in names if (parent / name).exists()), None)


def _target_section(manifest: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    """Return the target metadata section without interpreting arbitrary values."""
    targets = manifest.get("targets")
    if isinstance(targets, Mapping) and isinstance(targets.get(target), Mapping):
        return targets[target]  # type: ignore[return-value]
    direct = manifest.get(target)
    if isinstance(direct, Mapping):
        return direct  # type: ignore[return-value]
    # A target-local manifest commonly carries its target in the root object.
    if _target_matches(manifest.get("target"), target):
        return manifest
    return {}


def _canonical_target(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.lower().replace("_", "-").replace(" ", "")
    text = text.replace("windows-x86-64", "windows-x64")
    text = text.replace("windows-amd64", "windows-x64")
    text = text.replace("linux-x86-64", "linux-x64")
    text = text.replace("linux-amd64", "linux-x64")
    text = text.replace("linux-aarch64", "linux-arm64")
    text = text.replace("macos-x86-64", "macos-x64")
    text = text.replace("darwin-x86-64", "macos-x64")
    text = text.replace("darwin-arm64", "macos-arm64")
    return text


def _target_matches(value: Any, target: str) -> bool:
    """Accept spelling aliases while still requiring an explicit target identity."""
    return _canonical_target(value) == _canonical_target(target)


def _safe_target_path(target_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    # Manifests are exchanged between Windows and POSIX hosts.  Treat either
    # separator as a path separator before applying the traversal check.
    normalized = value.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", normalized):
        return None
    relative = Path(normalized)
    if relative.is_absolute():
        return None
    path = target_root / relative
    try:
        path.resolve().relative_to(target_root.resolve())
    except ValueError:
        return None
    return path


def _descriptor_path(descriptor: Any) -> Any:
    if isinstance(descriptor, Mapping):
        return descriptor.get("path", descriptor.get("manifest", descriptor.get("file")))
    return descriptor


def _descriptor_hash(descriptor: Any) -> str | None:
    if not isinstance(descriptor, Mapping):
        return None
    sources: list[Mapping[str, Any]] = [descriptor]
    integrity = descriptor.get("integrity")
    if isinstance(integrity, Mapping):
        sources.append(integrity)
    for source in sources:
        for key in ("sha256", "hash", "digest"):
            value = source.get(key)
            if isinstance(value, str):
                value = value.lower().removeprefix("sha256:")
                if _SHA256.fullmatch(value):
                    return value
    return None


def _load_json_artifact(
    target_root: Path,
    descriptor: Any,
    *,
    code: str,
    require_hash: bool,
    default_names: Sequence[str] = (),
) -> tuple[Mapping[str, Any] | None, Path | None, list[ReleaseIssue]]:
    """Load a target-local JSON evidence file and bind it to an optional hash."""
    issues: list[ReleaseIssue] = []
    path_value = _descriptor_path(descriptor)
    path = _safe_target_path(target_root, path_value) if path_value is not None else None
    if path is None and default_names:
        path = _first_existing(target_root, default_names)
    if path is not None and path.is_dir() and default_names:
        candidates = sorted(path.rglob("*.json"))
        path = candidates[0] if candidates else None
    if path is None and default_names:
        for directory in default_names:
            candidate_dir = target_root / directory
            if candidate_dir.is_dir():
                candidates = sorted(candidate_dir.rglob("*.json"))
                if candidates:
                    path = candidates[0]
                    break
    if path is None:
        issues.append(ReleaseIssue(f"{code}_MISSING", "evidence path is missing", str(target_root)))
        return None, None, issues
    if path.is_symlink():
        issues.append(ReleaseIssue("RELEASE_SYMLINK_PROHIBITED", f"symlink evidence is prohibited for {code}", str(path)))
        return None, path, issues
    if not path.is_file():
        issues.append(ReleaseIssue(f"{code}_MISSING", "evidence file is missing", str(path)))
        return None, path, issues
    expected = _descriptor_hash(descriptor)
    if require_hash and expected is None:
        issues.append(ReleaseIssue(f"{code}_HASH_MISSING", "evidence SHA-256 is required", str(path)))
    if expected is not None and sha256_file(path) != expected:
        issues.append(ReleaseIssue("HASH_MISMATCH", f"{code} evidence hash mismatch", str(path)))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(ReleaseIssue(f"{code}_INVALID", str(exc), str(path)))
        return None, path, issues
    if not isinstance(value, Mapping):
        issues.append(ReleaseIssue(f"{code}_INVALID", "JSON evidence root must be an object", str(path)))
        return None, path, issues
    return value, path, issues


def _verification_record(container: Any) -> Mapping[str, Any] | None:
    if not isinstance(container, Mapping):
        return None
    for key in ("verification", "verification_evidence", "trusted_verification", "independent_verification"):
        value = container.get(key)
        if isinstance(value, Mapping):
            if key == "independent_verification" and value.get("independent") is not True:
                return {**value, "independent": True}
            return value
    integrity = container.get("integrity")
    if isinstance(integrity, Mapping):
        for key in ("verification", "verification_evidence", "trusted_verification", "independent_verification"):
            value = integrity.get(key)
            if isinstance(value, Mapping):
                if key == "independent_verification" and value.get("independent") is not True:
                    return {**value, "independent": True}
                return value
    return None


def _verification_flags(record: Mapping[str, Any] | None) -> tuple[bool, bool, bool, bool]:
    if record is None:
        return False, False, False, False
    status = str(record.get("status", "")).lower()
    verified = record.get("verified") is True or status in _TRUSTED_STATUSES
    trusted = record.get("trusted") is True or str(record.get("trust", "")).lower() == "trusted"
    independent = record.get("independent") is True or record.get("independently_verified") is True
    evidence = any(
        bool(record.get(key))
        for key in ("evidence_id", "record_id", "verification_id", "report", "source")
    )
    return verified, trusted, independent, evidence


def _signature_value(container: Any) -> str | None:
    if not isinstance(container, Mapping):
        return None
    candidates: list[Any] = [container.get("signature"), container.get("signature_b64")]
    integrity = container.get("integrity")
    if isinstance(integrity, Mapping):
        candidates.extend((integrity.get("signature"), integrity.get("signature_b64")))
    for value in candidates:
        if isinstance(value, Mapping):
            value = value.get("value", value.get("encoded"))
        if isinstance(value, str) and value.strip():
            return value
    return None


def _signature_gate(container: Any, prefix: str, *, require_independent: bool = False) -> list[ReleaseIssue]:
    issues: list[ReleaseIssue] = []
    if _signature_value(container) is None:
        issues.append(ReleaseIssue(f"{prefix}_SIGNATURE_MISSING", "signed evidence is required"))
    integrity = container.get("integrity") if isinstance(container, Mapping) else None
    key_id = None
    for source in (container, integrity):
        if isinstance(source, Mapping):
            key_id = source.get("signer_key_id", source.get("key_id")) or key_id
    if not isinstance(key_id, str) or not key_id.strip():
        issues.append(ReleaseIssue(f"{prefix}_SIGNER_MISSING", "trusted signer key id is required"))
    verified, trusted, independent, evidence = _verification_flags(_verification_record(container))
    if not verified:
        issues.append(ReleaseIssue(f"{prefix}_VERIFICATION_MISSING", "verification evidence is not verified"))
    if not trusted:
        issues.append(ReleaseIssue(f"{prefix}_TRUST_MISSING", "verification evidence is not trusted"))
    if not evidence:
        issues.append(ReleaseIssue(f"{prefix}_EVIDENCE_MISSING", "verification evidence id is required"))
    if require_independent and not independent:
        issues.append(ReleaseIssue(f"{prefix}_INDEPENDENT_VERIFICATION_MISSING", "independent verification is required"))
    return issues


def _model_pack_descriptor(section: Mapping[str, Any]) -> Any:
    for key in ("model_pack", "signed_model_pack", "model_manifest", "model_pack_manifest", "signed_model_pack_manifest"):
        if key in section:
            return section[key]
    return None


def _report_target_issue(report: Mapping[str, Any], target: str, code: str) -> ReleaseIssue | None:
    value = report.get("target", report.get("target_id", report.get("platform")))
    if value is None:
        return ReleaseIssue(f"{code}_TARGET_MISSING", "target identity is required")
    if not _target_matches(value, target):
        return ReleaseIssue(f"{code}_TARGET_MISMATCH", f"report target {value!r} does not match {target}")
    return None


def _report_is_verified(report: Mapping[str, Any]) -> bool:
    status = str(report.get("status", report.get("verification_status", report.get("report_status", "")))).lower()
    if report.get("verified") is True or report.get("passed") is True or status in _TRUSTED_STATUSES:
        return True
    nested = _verification_record(report)
    verified, _, _, _ = _verification_flags(nested)
    return verified


def _digest_values(report: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    raw = report.get(
        "digests",
        report.get("golden_digests", report.get("golden_digest", report.get("files"))),
    )
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return values
    for item in raw:
        value = item.get("sha256", item.get("digest")) if isinstance(item, Mapping) else item
        if isinstance(value, str):
            value = value.lower().removeprefix("sha256:")
            if _SHA256.fullmatch(value):
                values.append(value)
    return values


def _cyclonedx_hashes(sbom: Mapping[str, Any]) -> set[str]:
    hashes: set[str] = set()
    components = sbom.get("components")
    if not isinstance(components, list):
        return hashes
    for component in components:
        if not isinstance(component, Mapping):
            continue
        raw_hashes = component.get("hashes")
        if not isinstance(raw_hashes, list):
            continue
        for item in raw_hashes:
            if not isinstance(item, Mapping):
                continue
            algorithm = str(item.get("alg", "")).replace("-", "").lower()
            content = str(item.get("content", "")).lower().removeprefix("sha256:")
            if algorithm == "sha256" and _SHA256.fullmatch(content):
                hashes.add(content)
    return hashes


def _production_catalog_issues(data: Mapping[str, Any]) -> list[ReleaseIssue]:
    issues: list[ReleaseIssue] = []
    if data.get("production_approved") is not True:
        issues.append(ReleaseIssue("PRODUCTION_APPROVAL_REQUIRED", "catalog production_approved must be true"))
    issues.extend(_signature_gate(data, "CATALOG"))
    return issues


def _production_target_issues(
    data: Mapping[str, Any],
    section: Mapping[str, Any],
    target_root: Path,
    target: str,
    entries: Sequence[Mapping[str, Any]],
) -> list[ReleaseIssue]:
    issues: list[ReleaseIssue] = []
    if not section:
        return [ReleaseIssue("PRODUCTION_TARGET_METADATA_MISSING", f"production target metadata is missing for {target}")]
    if section.get("production_approved") is not True:
        issues.append(ReleaseIssue("PRODUCTION_TARGET_APPROVAL_REQUIRED", f"target production_approved must be true for {target}"))

    sbom_descriptor = section.get("sbom", section.get("sbom_artifact"))
    sbom, _, sbom_issues = _load_json_artifact(
        target_root,
        sbom_descriptor,
        code="SBOM",
        require_hash=isinstance(sbom_descriptor, Mapping),
        default_names=FILE_OR_DIR_ALIASES["sbom"],
    )
    issues.extend(sbom_issues)
    if sbom is not None:
        if str(sbom.get("bomFormat", "")).lower() != "cyclonedx":
            issues.append(ReleaseIssue("SBOM_NOT_CYCLONEDX", "production SBOM must use CycloneDX"))
        components = sbom.get("components")
        if not isinstance(components, list) or not components:
            issues.append(ReleaseIssue("SBOM_COMPONENTS_EMPTY", "production CycloneDX SBOM must contain components"))
        elif any(not isinstance(component, Mapping) or not component.get("name") for component in components):
            issues.append(ReleaseIssue("SBOM_COMPONENT_INVALID", "CycloneDX components require names"))
        else:
            covered_hashes = _cyclonedx_hashes(sbom)
            for item in entries:
                rel = item.get("path", item.get("relative_path"))
                if not isinstance(rel, str):
                    continue
                normalized = rel.replace("\\", "/")
                if not normalized.startswith(("wheels/", "wheelhouse/", "models/", "model/", "native/")):
                    continue
                expected = str(item.get("sha256", "")).lower()
                if _SHA256.fullmatch(expected) and expected not in covered_hashes:
                    issues.append(ReleaseIssue("SBOM_COMPONENT_COVERAGE_MISSING", f"CycloneDX has no hash component for {rel}"))

    descriptor = _model_pack_descriptor(section)
    model_data, _, model_issues = _load_json_artifact(
        target_root,
        descriptor,
        code="MODEL_PACK",
        require_hash=True,
        default_names=("model-pack.json", "models/manifest.json", "model/manifest.json"),
    )
    issues.extend(model_issues)
    model_gate_containers: list[Any] = []
    if model_data is not None:
        model_gate_containers.append(model_data)
        if model_data.get("production_approved") is not True:
            issues.append(ReleaseIssue("MODEL_PACK_APPROVAL_REQUIRED", "model pack production_approved must be true"))
    if isinstance(descriptor, Mapping):
        model_gate_containers.append(descriptor)
    model_gate_results = [_signature_gate(container, "MODEL_PACK") for container in model_gate_containers]
    if model_gate_results and not any(not result for result in model_gate_results):
        # Keep the detailed reason from the most authoritative file first.
        issues.extend(model_gate_results[0])
    elif not model_gate_results:
        issues.append(ReleaseIssue("MODEL_PACK_SIGNATURE_MISSING", "signed model-pack manifest is required"))

    lock = section.get("offline_lock", section.get("hash_lock", section.get("lock")))
    if not isinstance(lock, Mapping):
        issues.append(ReleaseIssue("TARGET_LOCK_MISSING", f"target-specific exact hash lock is required for {target}"))
    else:
        lock_target = lock.get("target", lock.get("platform"))
        if not _target_matches(lock_target, target):
            issues.append(ReleaseIssue("TARGET_LOCK_TARGET_MISMATCH", f"lock target {lock_target!r} does not match {target}"))
        lock_path = _safe_target_path(target_root, lock.get("path"))
        lock_hash = _descriptor_hash(lock)
        if lock_path is None:
            issues.append(ReleaseIssue("TARGET_LOCK_PATH_MISSING", "target lock path is invalid"))
        elif lock_hash is None:
            issues.append(ReleaseIssue("TARGET_LOCK_HASH_MISSING", "target lock SHA-256 is required", str(lock_path)))
        elif not lock_path.is_file():
            issues.append(ReleaseIssue("TARGET_LOCK_MISSING", "target lock file is missing", str(lock_path)))
        elif sha256_file(lock_path) != lock_hash:
            issues.append(ReleaseIssue("TARGET_LOCK_HASH_MISMATCH", "target lock SHA-256 does not match", str(lock_path)))
        if lock_path is not None and lock_path.is_file():
            issues.extend(_lockfile_issues(lock_path, target))
            declared_paths = {
                str(item.get("path", item.get("relative_path"))).replace("\\", "/")
                for item in entries
                if isinstance(item, Mapping)
            }
            if lock_path.relative_to(target_root).as_posix() not in declared_paths:
                issues.append(ReleaseIssue("TARGET_LOCK_UNMANIFESTED", "target lock must be listed in the catalog", str(lock_path)))

    ort_descriptor = section.get(
        "ort_telemetry_attestation",
        section.get("ort_attestation", section.get("telemetry_attestation")),
    )
    ort_data, _, ort_issues = _load_json_artifact(target_root, ort_descriptor, code="ORT_ATTESTATION", require_hash=True)
    issues.extend(ort_issues)
    if ort_data is not None:
        target_issue = _report_target_issue(ort_data, target, "ORT_ATTESTATION")
        if target_issue:
            issues.append(target_issue)
        telemetry = ort_data.get("telemetry") if isinstance(ort_data.get("telemetry"), Mapping) else ort_data
        status = str(telemetry.get("status", "")).lower()
        if status not in _ORT_VERIFIED_STATUSES:
            issues.append(ReleaseIssue("ORT_ATTESTATION_NOT_INDEPENDENTLY_VERIFIED", "ORT telemetry attestation is not independently verified"))
        if telemetry.get("build_flag") != "onnxruntime_USE_TELEMETRY=OFF":
            issues.append(ReleaseIssue("ORT_ATTESTATION_TELEMETRY_NOT_OFF", "ORT telemetry build flag is not OFF"))
        verification = _verification_record(ort_data) or _verification_record(telemetry)
        verified, trusted, independent, evidence = _verification_flags(verification)
        if not (verified and trusted and independent and evidence):
            issues.append(ReleaseIssue("ORT_ATTESTATION_TRUST_MISSING", "independent trusted ORT verification evidence is required"))
        if ort_data.get("production_approved") is not True:
            issues.append(ReleaseIssue("ORT_ATTESTATION_APPROVAL_REQUIRED", "ORT attestation production_approved must be true"))

    golden_descriptor = section.get(
        "golden_digest_report",
        section.get("golden_report", section.get("golden_digest")),
    )
    golden, _, golden_issues = _load_json_artifact(target_root, golden_descriptor, code="GOLDEN_DIGEST_REPORT", require_hash=True)
    issues.extend(golden_issues)
    if golden is not None:
        target_issue = _report_target_issue(golden, target, "GOLDEN_DIGEST_REPORT")
        if target_issue:
            issues.append(target_issue)
        if not _report_is_verified(golden):
            issues.append(ReleaseIssue("GOLDEN_DIGEST_REPORT_UNVERIFIED", "golden digest report is not verified"))
        if not _digest_values(golden):
            issues.append(ReleaseIssue("GOLDEN_DIGESTS_EMPTY", "golden digest report has no SHA-256 digests"))

    abi_descriptor = section.get(
        "target_native_abi_report",
        section.get("abi_report", section.get("target_native_abi")),
    )
    abi, _, abi_issues = _load_json_artifact(target_root, abi_descriptor, code="ABI_REPORT", require_hash=True)
    issues.extend(abi_issues)
    if abi is not None:
        target_issue = _report_target_issue(abi, target, "ABI_REPORT")
        if target_issue:
            issues.append(target_issue)
        if not _report_is_verified(abi):
            issues.append(ReleaseIssue("ABI_REPORT_UNVERIFIED", "target-native ABI report is not verified"))
        if not any(abi.get(key) for key in ("abi", "python_abi", "platform", "architecture")):
            issues.append(ReleaseIssue("ABI_REPORT_IDENTITY_MISSING", "ABI report must declare runtime identity"))
        libraries = abi.get("libraries", abi.get("linked_libraries", abi.get("dependencies")))
        if not isinstance(libraries, (list, Mapping)) or not libraries:
            issues.append(ReleaseIssue("ABI_REPORT_LIBRARIES_EMPTY", "ABI report must declare linked libraries"))
    return issues


def _lockfile_issues(path: Path, target: str) -> list[ReleaseIssue]:
    """Validate an offline pip lock without resolving or installing it."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [ReleaseIssue("LOCKFILE_INVALID", str(exc), str(path))]
    logical: list[str] = []
    pending = ""
    for source in raw.splitlines():
        line = source.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].strip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)
    issues: list[ReleaseIssue] = []
    if not logical:
        return [ReleaseIssue("LOCKFILE_EMPTY", f"offline lock is empty for {target}", str(path))]
    for requirement in logical:
        lowered = requirement.lower()
        if "://" in requirement or any(flag in lowered for flag in ("--index-url", "--extra-index-url", "--find-links", "-r ", "--requirement")):
            issues.append(ReleaseIssue("LOCKFILE_NETWORK_OR_INCLUDE", requirement, str(path)))
            continue
        hashes = re.findall(r"--hash=sha256:([^\s]+)", lowered)
        if "==" not in requirement or not hashes:
            issues.append(ReleaseIssue("LOCKFILE_UNPINNED", requirement, str(path)))
        elif any(not _SHA256.fullmatch(value) for value in hashes):
            issues.append(ReleaseIssue("LOCKFILE_HASH_INVALID", requirement, str(path)))
    return issues


def _inventory_issues(target_root: Path, entries: Sequence[Mapping[str, Any]], target: str) -> list[ReleaseIssue]:
    declared = {
        str(item.get("path", item.get("relative_path"))).replace("\\", "/")
        for item in entries
        if isinstance(item.get("path", item.get("relative_path")), str)
    }
    issues: list[ReleaseIssue] = []
    for path in target_root.rglob("*"):
        if path.is_symlink():
            issues.append(ReleaseIssue("RELEASE_SYMLINK_PROHIBITED", f"symlink in {target}", str(path)))
            continue
        if not path.is_file() or path.name == "release-manifest.json":
            continue
        rel = path.relative_to(target_root).as_posix()
        if rel not in declared:
            issues.append(ReleaseIssue("UNMANIFESTED_ARTIFACT", rel, str(path)))
    return issues


def validate_release_layout(
    release_root: str | Path,
    *,
    targets: Sequence[str] = TARGETS,
    manifest: str | Path | None = None,
    require_manifest: bool = True,
    mode: str = "development",
    production: bool | None = None,
    production_mode: bool | None = None,
) -> ReleaseReport:
    """Validate an offline release layout.

    Development mode checks the local candidate layout and hash syntax.
    Production mode is explicit and fail-closed: it additionally requires
    approval, signed/trusted catalog and model-pack evidence, independently
    verified ORT telemetry evidence, golden digests, target-native ABI data,
    and a target-bound exact lock.
    """
    if production is not None and production_mode is not None and production != production_mode:
        raise ValueError("production and production_mode disagree")
    if production_mode is not None:
        production = production_mode
    if production is not None:
        mode = "production" if production else "development"
    if mode not in _PRODUCTION_MODES:
        raise ValueError(f"unsupported release validation mode: {mode!r}")
    root = Path(release_root).resolve()
    report = ReleaseReport(root, tuple(targets), mode=mode)
    if not root.is_dir():
        report.issues.append(ReleaseIssue("RELEASE_ROOT_MISSING", "release root is not a directory", str(root)))
        return report
    data: Mapping[str, Any] | None = None
    manifest_name: str | None = None
    if manifest is not None:
        mp = Path(manifest)
        if not mp.is_absolute(): mp = root / mp
        manifest_name = str(mp.relative_to(root)) if mp.is_relative_to(root) else str(mp)
        try: data = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            report.issues.append(ReleaseIssue("MANIFEST_INVALID", "manifest is missing or invalid", str(mp)))
    else:
        data, manifest_name = _load_manifest(root)
        local_manifests = all((_target_root(root, t) / "release-manifest.json").is_file() for t in targets)
        if data is None and require_manifest and not local_manifests:
            report.issues.append(ReleaseIssue("MANIFEST_MISSING", "release catalog/manifest or target manifests are required", str(root)))
    if data is not None:
        declared = data.get("targets")
        if not isinstance(declared, Mapping):
            report.issues.append(ReleaseIssue("MANIFEST_TARGETS_MISSING", "manifest must declare targets"))
        if mode == "production":
            report.issues.extend(_production_catalog_issues(data))
        elif data.get("production_approved") is True:
            report.issues.append(ReleaseIssue("DEVELOPMENT_APPROVAL_MISMATCH", "development validation cannot approve a production catalog"))
    elif mode == "production":
        report.issues.append(ReleaseIssue("PRODUCTION_MANIFEST_REQUIRED", "production validation requires a catalog manifest"))
    for target in targets:
        tr = _target_root(root, target)
        if not tr.is_dir():
            report.issues.append(ReleaseIssue("TARGET_MISSING", f"target directory missing: {target}", str(tr)))
            continue
        for dirname in REQUIRED_DIRS:
            p = _first_existing(tr, DIR_ALIASES[dirname]) or (tr / dirname)
            if not p.is_dir(): report.issues.append(ReleaseIssue("ARTIFACT_DIR_MISSING", f"{dirname}/ missing for {target}", str(p)))
            elif not any(x.is_file() for x in p.rglob("*")):
                report.issues.append(ReleaseIssue("ARTIFACT_EMPTY", f"{dirname}/ is empty for {target}", str(p)))
        for kind, names in FILE_OR_DIR_ALIASES.items():
            p = _first_existing(tr, names)
            if p is None or (p.is_dir() and not any(x.is_file() for x in p.rglob("*"))):
                report.issues.append(ReleaseIssue("RELEASE_FILE_MISSING", f"{kind} missing for {target}", str(tr)))
        lock = _first_existing(tr, LOCKFILE_NAMES)
        if lock is None or not lock.is_file():
            report.issues.append(ReleaseIssue("LOCKFILE_MISSING", f"hash-locked requirements missing for {target}", str(tr)))
        else:
            report.issues.extend(_lockfile_issues(lock, target))
        if data is not None:
            entries = _manifest_entries(data, target)
            section = _target_section(data, target)
            # A target-local manifest is the authoritative linkage when the
            # catalog only describes target metadata.
            if not entries:
                local = tr / "release-manifest.json"
                if local.is_file():
                    try:
                        local_data = json.loads(local.read_text(encoding="utf-8"))
                        entries = local_data.get("files", []) if isinstance(local_data, Mapping) else []
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        report.issues.append(ReleaseIssue("MANIFEST_INVALID", "target release-manifest.json is invalid", str(local)))
            if not entries: report.issues.append(ReleaseIssue("MANIFEST_TARGET_MISSING", f"manifest has no file entries for {target}"))
            for item in entries:
                rel = item.get("path", item.get("relative_path"))
                expected = str(item.get("sha256", "")).lower()
                if not isinstance(rel, str) or not _SHA256.fullmatch(expected):
                    report.issues.append(ReleaseIssue("MANIFEST_ENTRY_INVALID", f"invalid path/hash entry for {target}")); continue
                path = tr / rel
                try: path.resolve().relative_to(tr.resolve())
                except ValueError: report.issues.append(ReleaseIssue("MANIFEST_PATH_TRAVERSAL", rel, str(path))); continue
                if path.is_symlink(): report.issues.append(ReleaseIssue("RELEASE_SYMLINK_PROHIBITED", rel, str(path))); continue
                if not path.is_file(): report.issues.append(ReleaseIssue("MANIFEST_ARTIFACT_MISSING", rel, str(path))); continue
                report.verified_files += 1
                if sha256_file(path) != expected: report.issues.append(ReleaseIssue("HASH_MISMATCH", rel, str(path)))
            report.issues.extend(_inventory_issues(tr, entries, target))
            if mode == "production":
                report.issues.extend(_production_target_issues(data, section, tr, target, entries))
        elif require_manifest:
            local = tr / "release-manifest.json"
            try:
                local_data = json.loads(local.read_text(encoding="utf-8"))
                entries = local_data.get("files", []) if isinstance(local_data, Mapping) else []
                if not entries: report.issues.append(ReleaseIssue("MANIFEST_TARGET_MISSING", f"target manifest has no file entries for {target}"))
                for item in entries:
                    rel, expected = item.get("path", item.get("relative_path")), str(item.get("sha256", "")).lower()
                    path = tr / rel if isinstance(rel, str) else tr / "<invalid>"
                    inside = False
                    if isinstance(rel, str):
                        try: path.resolve().relative_to(tr.resolve()); inside = True
                        except ValueError: pass
                    if not isinstance(rel, str) or not inside or not _SHA256.fullmatch(expected) or path.is_symlink() or not path.is_file():
                        report.issues.append(ReleaseIssue("MANIFEST_ENTRY_INVALID", f"invalid or missing entry for {target}", str(path))); continue
                    report.verified_files += 1
                    if sha256_file(path) != expected: report.issues.append(ReleaseIssue("HASH_MISMATCH", rel, str(path)))
                report.issues.extend(_inventory_issues(tr, entries, target))
            except (OSError, UnicodeError, json.JSONDecodeError):
                report.issues.append(ReleaseIssue("MANIFEST_INVALID", "target manifest is missing or invalid", str(local)))
    return report


_NETWORK_MODULES = (
    "urllib.request", "http.client", "httpx", "requests", "socket", "ftplib", "webbrowser",
)
_NETWORK_CALLS = {"urlopen", "urlretrieve", "get", "post", "request", "download", "install", "connect"}


def _is_network_module(module: str) -> bool:
    return any(module == candidate or module.startswith(candidate + ".") for candidate in _NETWORK_MODULES)


def _root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


def scan_zero_network_policy(paths: Sequence[str | Path]) -> list[ReleaseIssue]:
    """Static scan Python sources for network/download/install primitives."""
    issues: list[ReleaseIssue] = []
    files: list[Path] = []
    for supplied in paths:
        p = Path(supplied)
        files.extend([p] if p.is_file() else [x for x in p.rglob("*.py") if x.is_file()])
    for path in files:
        try: tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            issues.append(ReleaseIssue("STATIC_SCAN_ERROR", str(exc), str(path))); continue
        network_aliases: set[str] = set()
        direct_network_calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_network_module(alias.name):
                        network_aliases.add(alias.asname or alias.name.split(".")[0])
                        issues.append(ReleaseIssue("NETWORK_IMPORT", alias.name, str(path)))
            elif isinstance(node, ast.ImportFrom) and node.module and _is_network_module(node.module):
                issues.append(ReleaseIssue("NETWORK_IMPORT", node.module, str(path)))
                for alias in node.names:
                    direct_network_calls.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in _NETWORK_CALLS and _root_name(node.func.value) in network_aliases:
                    issues.append(ReleaseIssue("NETWORK_CALL", node.func.attr, str(path)))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in direct_network_calls or node.func.id in {"urlopen", "urlretrieve"}:
                    issues.append(ReleaseIssue("NETWORK_CALL", node.func.id, str(path)))
    return issues


def clean_install_plan(target: str, release_root: str | Path = ".", *, python: str | None = None) -> tuple[str, ...]:
    """Return OS-appropriate commands for an offline, hash-locked install.

    Windows deliberately uses the Python launcher form ``py -3.11 -m pip``;
    POSIX targets use ``python3.11``.  A caller may override the executable
    for an isolated toolchain, but the default remains target-specific.
    """
    if target not in TARGETS: raise ValueError(f"unsupported target: {target}")
    tr = _target_root(Path(release_root), target)
    wheel_dir = _first_existing(tr, DIR_ALIASES["wheelhouse"]) or (tr / "wheelhouse")
    lock = _first_existing(tr, LOCKFILE_NAMES) or (tr / "requirements.lock")
    windows = target.startswith("windows-")
    python_command = python or ("py -3.11" if windows else "python3.11")
    if windows:
        venv = f".venv-{target}"
        wheel_text = str(wheel_dir).replace("/", "\\")
        lock_text = str(lock).replace("/", "\\")
        root_text = str(Path(release_root)).replace("/", "\\")
        return (
            f"{python_command} -m venv {venv}",
            f"{venv}\\Scripts\\activate",
            f"{python_command} -m pip install --no-index --find-links {wheel_text} --require-hashes -r {lock_text}",
            f"{python_command} scripts\\verify_offline_release.py {root_text} --target {target}",
        )
    return (
        f"{python_command} -m venv .venv-{target}",
        f". .venv-{target}/bin/activate",
        f"{python_command} -m pip install --no-index --find-links {wheel_dir} --require-hashes -r {lock}",
        f"{python_command} scripts/verify_offline_release.py {Path(release_root)} --target {target}",
    )
