"""Signed, immutable model-pack intake for offline execution."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .offline import OfflinePolicyViolation, reject_url

try:  # The main contracts/errors lane may provide the canonical classes.
    from .errors import (  # type: ignore
        ModelHashMismatch,
        ModelNotFound,
        ModelRuntimeIncompatible,
        ManifestSignatureInvalid,
    )
except ImportError:  # pragma: no cover - development bootstrap
    class _PackError(RuntimeError):
        code = "MODEL_PACK_ERROR"
    class ModelNotFound(_PackError): code = "MODEL_NOT_FOUND"
    class ModelHashMismatch(_PackError): code = "MODEL_HASH_MISMATCH"
    class ModelRuntimeIncompatible(_PackError): code = "MODEL_RUNTIME_INCOMPATIBLE"
    class ManifestSignatureInvalid(_PackError): code = "MANIFEST_SIGNATURE_INVALID"


class SignatureVerifier(Protocol):
    trust_level: str
    def verify(self, payload: bytes, signature: bytes, key_id: str) -> bool: ...


class DigestSignatureVerifier:
    """Deterministic development verifier; production supplies Ed25519 verifier.

    A digest verifier is intentionally not presented as cryptographic signing:
    it is useful for local tests and the pluggable boundary, while release
    profiles must inject an audited verifier.
    """
    trust_level = "DEVELOPMENT"

    def __init__(self, key: bytes): self.key = bytes(key)
    def verify(self, payload: bytes, signature: bytes, key_id: str) -> bool:
        expected = hashlib.sha256(self.key + payload).digest()
        return signature in (expected, expected.hex().encode("ascii"), base64.b64encode(expected))


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    file_id: str
    role: str
    path: Path
    sha256: str
    bytes: int


_PACK_SEAL = object()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, init=False)
class VerifiedModelPack:
    pack_id: str
    pack_version: str
    manifest: Mapping[str, Any]
    artifacts: tuple[VerifiedArtifact, ...]
    runtime_compatibility: Mapping[str, Any]
    manifest_sha256: str
    trust_level: str
    signature_verified: bool
    production_approved: bool

    def __init__(self, pack_id: str, pack_version: str, manifest: Mapping[str, Any],
                 artifacts: tuple[VerifiedArtifact, ...], runtime_compatibility: Mapping[str, Any],
                 manifest_sha256: str, trust_level: str, signature_verified: bool,
                 production_approved: bool, *, _seal: object | None = None):
        if _seal is not _PACK_SEAL:
            raise TypeError("VerifiedModelPack is created only by ModelPackVerifier")
        object.__setattr__(self, "pack_id", pack_id)
        object.__setattr__(self, "pack_version", pack_version)
        object.__setattr__(self, "manifest", _deep_freeze(manifest))
        object.__setattr__(self, "artifacts", tuple(artifacts))
        object.__setattr__(self, "runtime_compatibility", _deep_freeze(runtime_compatibility))
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "trust_level", trust_level)
        object.__setattr__(self, "signature_verified", bool(signature_verified))
        object.__setattr__(self, "production_approved", bool(production_approved))

    @property
    def release_trusted(self) -> bool:
        return (self.trust_level == "RELEASE" and self.signature_verified
                and self.production_approved)

    def assert_artifacts_unchanged(self) -> None:
        """Re-hash verified files before a release binding consumes them."""

        for artifact in self.artifacts:
            path = artifact.path
            if path.is_symlink() or not path.is_file() or path.stat().st_size != artifact.bytes:
                raise _fail(ModelHashMismatch, f"artifact {artifact.file_id} changed after verification")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != artifact.sha256:
                raise _fail(ModelHashMismatch, f"artifact {artifact.file_id} changed after verification")


def _fail(cls: type[Exception], message: str) -> Exception:
    try: return cls(message)  # type: ignore[call-arg]
    except TypeError: return cls()  # type: ignore[call-arg]


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Canonical bytes signed by release tooling (signature fields excluded)."""
    data = json.loads(json.dumps(manifest))
    integrity = data.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("signature", None)
        integrity.pop("canonical_manifest_sha256", None)
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


class ModelPackVerifier:
    def __init__(self, pack_root: str | os.PathLike[str], *, runtime: Mapping[str, Any] | None = None,
                 signature_verifier: SignatureVerifier | None = None, development_mode: bool = False,
                 require_release_trust: bool = False):
        self.root = Path(pack_root)
        self.runtime = dict(runtime or current_runtime())
        self.signature_verifier = signature_verifier
        self.development_mode = development_mode
        self.require_release_trust = require_release_trust
        if development_mode and require_release_trust:
            raise ValueError("release trust cannot use development_mode")

    def verify(self, manifest: Mapping[str, Any] | str | os.PathLike[str] = "manifest.json") -> VerifiedModelPack:
        if isinstance(manifest, Mapping): data = json.loads(json.dumps(manifest))
        else:
            reject_url(manifest)
            try: data = json.loads(Path(manifest if Path(manifest).is_absolute() else self.root / manifest).read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc: raise _fail(ModelNotFound, "manifest.json is missing or invalid") from exc
        if not self.root.is_dir(): raise _fail(ModelNotFound, "model pack root is missing")
        self._validate_manifest(data)
        signature_verified = self._verify_signature(data)
        compatibility = data["runtime_compatibility"]
        self._verify_runtime(compatibility)
        artifacts = tuple(self._verify_file(item) for item in data["files"])
        release_signature = (signature_verified and
                             getattr(self.signature_verifier, "trust_level", None) == "RELEASE")
        production_approved = data["production_approved"]
        trust_level = "RELEASE" if release_signature and production_approved else "DEVELOPMENT"
        if self.require_release_trust and trust_level != "RELEASE":
            raise _fail(ManifestSignatureInvalid, "release-trusted signature and production approval are required")
        manifest_digest = hashlib.sha256(canonical_manifest_bytes(data)).hexdigest()
        return VerifiedModelPack(
            data["pack_id"], data["pack_version"], data, artifacts, compatibility,
            manifest_digest, trust_level, signature_verified, production_approved, _seal=_PACK_SEAL,
        )

    verify_pack = verify

    def _validate_manifest(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise _fail(ModelHashMismatch, "manifest must be an object")
        for key in ("schema_version", "pack_id", "pack_version"):
            if not isinstance(data.get(key), str) or not str(data[key]).strip():
                raise _fail(ModelHashMismatch, f"manifest {key} is required")
        if not isinstance(data.get("production_approved"), bool):
            raise _fail(ModelHashMismatch, "manifest production_approved boolean is required")
        compatibility = data.get("runtime_compatibility")
        if not isinstance(compatibility, Mapping) or not compatibility:
            raise _fail(ModelRuntimeIncompatible, "runtime compatibility contract is required")
        if not any(key in compatibility for key in
                   ("onnxruntime", "allowed_execution_providers", "target_matrix")):
            raise _fail(ModelRuntimeIncompatible, "runtime compatibility contract is incomplete")
        files = data.get("files")
        if not isinstance(files, list) or not files:
            raise _fail(ModelNotFound, "model pack has no artifacts")
        file_ids: set[str] = set()
        paths: set[str] = set()
        calibration_count = 0
        for item in files:
            if not isinstance(item, Mapping):
                raise _fail(ModelHashMismatch, "artifact entry must be an object")
            file_id, role, relative = item.get("file_id"), item.get("role"), item.get("relative_path")
            if not isinstance(file_id, str) or not file_id.strip() or file_id in file_ids:
                raise _fail(ModelHashMismatch, "artifact file_id must be nonempty and unique")
            if not isinstance(role, str) or not role.strip():
                raise _fail(ModelHashMismatch, "artifact role is required")
            if not isinstance(relative, str) or not relative or relative in paths:
                raise _fail(ModelHashMismatch, "artifact relative_path must be unique")
            file_ids.add(file_id); paths.add(relative)
            if role.lower() in {"calibration", "calibration_profile", "quality_calibration"}:
                calibration_count += 1
        if calibration_count > 1:
            raise _fail(ModelHashMismatch, "duplicate calibration artifacts are prohibited")

    def _verify_signature(self, data: Mapping[str, Any]) -> bool:
        integ = data.get("integrity", {})
        sig = integ.get("signature") if isinstance(integ, Mapping) else None
        if not sig and self.development_mode: return False
        if not sig or self.signature_verifier is None: raise _fail(ManifestSignatureInvalid, "manifest signature is required")
        key_id = integ.get("signer_key_id") if isinstance(integ, Mapping) else None
        if not isinstance(key_id, str) or not key_id.strip():
            raise _fail(ManifestSignatureInvalid, "manifest signer_key_id is required")
        try: raw = base64.b64decode(str(sig), validate=True)
        except Exception as exc: raise _fail(ManifestSignatureInvalid, "invalid manifest signature encoding") from exc
        payload = canonical_manifest_bytes(data)
        declared_digest = integ.get("canonical_manifest_sha256") if isinstance(integ, Mapping) else None
        if declared_digest is not None and declared_digest != hashlib.sha256(payload).hexdigest():
            raise _fail(ManifestSignatureInvalid, "canonical manifest digest mismatch")
        if not self.signature_verifier.verify(payload, raw, key_id):
            raise _fail(ManifestSignatureInvalid, "manifest signature verification failed")
        return True

    def _verify_runtime(self, compat: Mapping[str, Any]) -> None:
        ort = compat.get("onnxruntime", {})
        for key in ("exact_build_id", "exact_version"):
            if key in ort and self.runtime.get(key) != ort[key]: raise _fail(ModelRuntimeIncompatible, f"runtime {key} mismatch")
        allowed = compat.get("allowed_execution_providers")
        ep = self.runtime.get("execution_provider") or self.runtime.get("execution_providers")
        eps = [ep] if isinstance(ep, str) else list(ep or [])
        if allowed is not None and (not eps or any(x not in allowed for x in eps)): raise _fail(ModelRuntimeIncompatible, "execution provider mismatch")
        for target in compat.get("target_matrix", []):
            if all(self.runtime.get(k) == target.get(k) for k in ("os", "arch", "python_abi") if k in target):
                return
        if compat.get("target_matrix"): raise _fail(ModelRuntimeIncompatible, "platform target mismatch")

    def _verify_file(self, item: Mapping[str, Any]) -> VerifiedArtifact:
        rel = item.get("relative_path")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or "\x00" in rel: raise _fail(ModelHashMismatch, "invalid artifact path")
        reject_url(rel)
        raw_path = self.root / Path(rel)
        # Reject every symlink, including one that resolves inside the pack.
        # Calling stat() first would follow the link and miss this condition.
        if raw_path.is_symlink():
            raise _fail(ModelHashMismatch, "symlink artifact is prohibited")
        path = raw_path.resolve()
        root = self.root.resolve()
        if path != root and root not in path.parents: raise _fail(ModelHashMismatch, "artifact path traversal")
        try: st = path.stat()
        except FileNotFoundError as exc: raise _fail(ModelNotFound, f"missing artifact {item.get('file_id', '')}") from exc
        if not stat.S_ISREG(st.st_mode):
            raise _fail(ModelHashMismatch, "artifact must be a regular file")
        expected_size, expected_hash = item.get("bytes"), str(item.get("sha256", "")).lower()
        if not isinstance(expected_size, int) or not expected_hash: raise _fail(ModelHashMismatch, "artifact hash/size is required")
        digest = hashlib.sha256(); size = 0
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""): size += len(chunk); digest.update(chunk)
        if size != expected_size or digest.hexdigest() != expected_hash: raise _fail(ModelHashMismatch, f"artifact {item.get('file_id', '')} mismatch")
        return VerifiedArtifact(str(item.get("file_id", "")), str(item.get("role", "")), path, digest.hexdigest(), size)


def current_runtime() -> dict[str, str]:
    return {"os": sys.platform, "arch": platform.machine(), "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}"}
