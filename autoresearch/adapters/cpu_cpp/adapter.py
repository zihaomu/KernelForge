from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from autoresearch.core.utils import ensure_dir


def build_runner(
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


def run_gemm_candidate(
    *,
    runner_binary: Path,
    shape: dict[str, int],
    candidate: dict[str, Any],
    warmup: int,
    iters: int,
    verify: bool,
    timeout_sec: int,
    input_mode: str,
) -> dict[str, Any]:
    args = [
        str(runner_binary),
        "--m",
        str(shape["m"]),
        "--n",
        str(shape["n"]),
        "--k",
        str(shape["k"]),
        "--kernel_variant",
        str(candidate["kernel_variant"]),
        "--bm",
        str(candidate["block_m"]),
        "--bn",
        str(candidate["block_n"]),
        "--bk",
        str(candidate["block_k"]),
        "--pack_a",
        "1" if candidate["pack_a"] else "0",
        "--pack_b",
        "1" if candidate["pack_b"] else "0",
        "--simd",
        "1" if candidate["simd"] else "0",
        "--threads",
        str(candidate["threads"]),
        "--unroll_k",
        str(candidate["unroll_k"]),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--verify",
        "1" if verify else "0",
        "--json",
        "1",
        "--input_mode",
        input_mode,
    ]

    try:
        cp = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"valid": False, "error": f"timeout_after_{timeout_sec}s:{exc}", "shape_name": shape["name"]}

    if cp.returncode != 0:
        return {"valid": False, "error": f"runner_rc={cp.returncode}: {cp.stderr.strip()[:200]}", "shape_name": shape["name"]}

    row: dict[str, Any] = {}
    for line in reversed(cp.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not row:
        row = {"valid": False, "error": f"invalid_json:{cp.stdout[:200]}"}
    row["shape_name"] = shape["name"]
    return row

