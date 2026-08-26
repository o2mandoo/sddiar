#!/usr/bin/env python3
"""Fail if Git tracks audio, model/runtime binaries, or private input trees."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import PurePosixPath


FORBIDDEN_SUFFIXES = frozenset({
    ".wav", ".wave", ".bw64", ".rf64", ".w64", ".m4a", ".m4b", ".m4r",
    ".mp3", ".flac", ".aac", ".ogg", ".oga", ".opus", ".aif", ".aiff",
    ".aifc", ".caf", ".au", ".snd", ".amr", ".awb", ".wma", ".ac3",
    ".eac3", ".dts", ".ape", ".wv", ".tta", ".mka", ".pcm", ".raw",
    ".sw", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg",
    ".m4v", ".3gp", ".3gpp", ".3g2", ".npy", ".npz", ".bin", ".pt",
    ".pth", ".safetensors", ".gguf", ".onnx", ".ort", ".tflite",
    ".mlmodel", ".ckpt", ".whl", ".dll", ".dylib", ".so", ".exe",
})
FORBIDDEN_PARTS = frozenset({
    "audio", "recording", "recordings", "transcript", "transcripts",
    "media_inputs", "enhanced_audio", "denoised_audio", "normalized_audio",
})


def forbidden_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    parts = tuple(part.lower() for part in pure.parts)
    if any(part in FORBIDDEN_PARTS for part in parts[:-1]):
        return True
    if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    return len(parts) == 1 and pure.suffix.lower() == ".txt"


def media_magic(payload: bytes) -> bool:
    head = payload[:16]
    if len(head) >= 12 and head[:4] in {b"RIFF", b"RF64", b"BW64"} and head[8:12] == b"WAVE":
        return True
    if len(head) >= 12 and head[:4] == b"FORM" and head[8:12] in {b"AIFF", b"AIFC"}:
        return True
    if head.startswith((b"fLaC", b"OggS", b"ID3", b"caff", b".snd")):
        return True
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return True
    if head.startswith(b"\x1aE\xdf\xa3"):
        return True
    return len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def audit() -> dict:
    current = [item.decode("utf-8", errors="strict") for item in _git("ls-files", "-z").split(b"\0") if item]
    historical: list[tuple[str, str]] = []
    for line in _git("rev-list", "--objects", "--all").decode("utf-8", errors="strict").splitlines():
        object_id, separator, path = line.partition(" ")
        if separator and path:
            historical.append((object_id, path))

    forbidden_current = sorted(path for path in current if forbidden_path(path))
    forbidden_history = sorted({path for _, path in historical if forbidden_path(path)})
    magic_violations: list[str] = []
    seen: set[str] = set()
    for object_id, path in historical:
        if object_id in seen:
            continue
        seen.add(object_id)
        object_type = _git("cat-file", "-t", object_id).strip()
        if object_type != b"blob":
            continue
        payload = _git("cat-file", "blob", object_id)
        if media_magic(payload):
            magic_violations.append(path)

    return {
        "schema": "sddiar-git-media-policy-v1",
        "tracked_path_count": len(current),
        "historical_object_path_count": len(historical),
        "forbidden_current_paths": forbidden_current,
        "forbidden_history_paths": forbidden_history,
        "media_magic_history_paths": sorted(set(magic_violations)),
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    try:
        result = audit()
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 2
    violations = sum(len(result[key]) for key in (
        "forbidden_current_paths", "forbidden_history_paths", "media_magic_history_paths"
    ))
    print(json.dumps({"ok": violations == 0, **result}, sort_keys=True))
    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
