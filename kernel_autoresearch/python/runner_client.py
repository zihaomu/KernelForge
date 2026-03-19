from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def build_cpp_runner(
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
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j"],
        cwd=str(repo_root),
        check=True,
    )
    binary = build_dir / binary_name
    if not binary.exists():
        raise FileNotFoundError(f"runner binary not found: {binary}")
    return binary


def _parse_runner_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"failed to parse runner json from stdout: {stdout[:500]}")


def run_trial(
    runner_binary: Path,
    shape: dict[str, int],
    candidate: dict[str, Any],
    warmup_iters: int,
    measure_iters: int,
    verify: bool,
    timeout_sec: int,
    risk_level: str,
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
        str(candidate["bm"]),
        "--bn",
        str(candidate["bn"]),
        "--bk",
        str(candidate["bk"]),
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
        "--input_dtype",
        str(candidate.get("input_dtype", "f32")),
        "--output_dtype",
        str(candidate.get("output_dtype", "f32")),
        "--warmup",
        str(warmup_iters),
        "--iters",
        str(measure_iters),
        "--verify",
        "1" if verify else "0",
        "--json",
        "1",
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
        if cp.returncode != 0:
            return {
                "shape_name": shape["name"],
                **candidate,
                "valid": False,
                "latency_ms_p50": 0.0,
                "latency_ms_p95": 0.0,
                "gflops": 0.0,
                "max_abs_err": 0.0,
                "max_rel_err": 0.0,
                "risk_level": risk_level,
                "error": f"runner rc={cp.returncode}: {cp.stderr.strip()[:400]}",
            }
        row = _parse_runner_json(cp.stdout)
        row.update(candidate)
        row["shape_name"] = shape["name"]
        row["risk_level"] = risk_level
        return row
    except Exception as exc:  # noqa: BLE001
        return {
            "shape_name": shape["name"],
            **candidate,
            "valid": False,
            "latency_ms_p50": 0.0,
            "latency_ms_p95": 0.0,
            "gflops": 0.0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "risk_level": risk_level,
            "error": str(exc),
        }
