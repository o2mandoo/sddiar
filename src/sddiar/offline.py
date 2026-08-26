"""Small, deliberately boring helpers for the closed-network boundary.

This module never downloads, resolves a URL, or falls back to a cache.  It is
kept separate from model-pack parsing so every local artifact entry point can
use the same policy.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse

from .errors import OfflinePolicyViolation

_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def reject_url(value: object) -> None:
    """Raise for URL-like values (including file:// and Windows drive URLs)."""
    if not isinstance(value, (str, os.PathLike)):
        return
    text = os.fspath(value)
    # A Windows absolute local path is not a URL even though urlparse sees a
    # one-letter scheme. The release package supports Windows x64.
    if _WINDOWS_DRIVE_PATH.match(text):
        return
    parsed = urlparse(text)
    if parsed.scheme or text.startswith(("//", "\\\\")):
        raise OfflinePolicyViolation("network or URL artifact sources are prohibited")


def local_path(value: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None) -> Path:
    """Return an existing local path, constrained to *root* when supplied."""
    reject_url(value)
    path = Path(value)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(path)) from exc
    if root is not None:
        base = Path(root).resolve(strict=True)
        if resolved != base and base not in resolved.parents:
            raise OfflinePolicyViolation("artifact escapes the offline pack root")
    return resolved


def ensure_no_fallback(*, artifact_error: BaseException) -> None:
    """Document/enforce the fail-closed boundary for callers."""
    raise artifact_error
