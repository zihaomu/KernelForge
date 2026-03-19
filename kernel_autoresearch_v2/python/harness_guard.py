from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .utils import dump_json, load_json


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(repo_root: Path, harness_files: list[Path], manifest_path: Path) -> dict[str, Any]:
    entries = []
    for f in sorted(harness_files, key=lambda p: str(p)):
        rel = str(f.resolve().relative_to(repo_root.resolve()))
        entries.append({"path": rel, "sha256": _sha256_file(f)})
    manifest = {"version": 1, "entries": entries}
    dump_json(manifest_path, manifest)
    return manifest


def verify_manifest(repo_root: Path, manifest_path: Path) -> tuple[bool, list[str]]:
    if not manifest_path.exists():
        return False, [f"missing_manifest:{manifest_path}"]
    manifest = load_json(manifest_path)
    mismatches: list[str] = []
    for entry in manifest.get("entries", []):
        rel = str(entry["path"])
        expected = str(entry["sha256"])
        abs_path = (repo_root / rel).resolve()
        if not abs_path.exists():
            mismatches.append(f"missing_file:{rel}")
            continue
        actual = _sha256_file(abs_path)
        if actual != expected:
            mismatches.append(f"hash_mismatch:{rel}")
    return len(mismatches) == 0, mismatches

