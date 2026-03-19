from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import ensure_dir


def build_cpp_runner(
    *,
    repo_root: Path,
    source_dir: Path,
    build_dir: Path,
    build_type: str,
    binary_name: str,
) -> Path:
    ensure_dir(build_dir)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ],
        cwd=str(repo_root),
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "-j"], cwd=str(repo_root), check=True)
    binary = build_dir / binary_name
    if not binary.exists():
        raise FileNotFoundError(f"runner binary not found: {binary}")
    return binary

