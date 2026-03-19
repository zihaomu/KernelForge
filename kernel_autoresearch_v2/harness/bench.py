from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from kernel_autoresearch_v2.harness.reference import deterministic_reference_checksum
from kernel_autoresearch_v2.python.candidate_generator import candidate_signature


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    if "bm" in out and "block_m" not in out:
        out["block_m"] = int(out["bm"])
    if "bn" in out and "block_n" not in out:
        out["block_n"] = int(out["bn"])
    if "bk" in out and "block_k" not in out:
        out["block_k"] = int(out["bk"])

    out["kernel_variant"] = str(out.get("kernel_variant", "naive"))
    out["block_m"] = int(out.get("block_m", 0))
    out["block_n"] = int(out.get("block_n", 0))
    out["block_k"] = int(out.get("block_k", 0))
    out["pack_a"] = bool(out.get("pack_a", False))
    out["pack_b"] = bool(out.get("pack_b", False))
    out["simd"] = bool(out.get("simd", False))
    out["threads"] = max(1, int(out.get("threads", 1)))
    out["unroll_k"] = max(1, int(out.get("unroll_k", 1)))
    return out


def _run_runner(
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
    c = _normalize_candidate(candidate)
    args = [
        str(runner_binary),
        "--m",
        str(shape["m"]),
        "--n",
        str(shape["n"]),
        "--k",
        str(shape["k"]),
        "--kernel_variant",
        str(c["kernel_variant"]),
        "--bm",
        str(c["block_m"]),
        "--bn",
        str(c["block_n"]),
        "--bk",
        str(c["block_k"]),
        "--pack_a",
        "1" if c["pack_a"] else "0",
        "--pack_b",
        "1" if c["pack_b"] else "0",
        "--simd",
        "1" if c["simd"] else "0",
        "--threads",
        str(c["threads"]),
        "--unroll_k",
        str(c["unroll_k"]),
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
        return {
            "valid": False,
            "error": f"timeout_after_{timeout_sec}s:{exc}",
            "shape_name": shape["name"],
        }

    if cp.returncode != 0:
        return {
            "valid": False,
            "error": f"runner_rc={cp.returncode}: {cp.stderr.strip()[:200]}",
            "shape_name": shape["name"],
        }

    row = {}
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


def _validate_candidate(candidate: dict[str, Any]) -> tuple[bool, str]:
    c = _normalize_candidate(candidate)
    variant = c["kernel_variant"]
    if variant not in ("naive", "blocked", "blocked_pack", "blocked_pack_simd"):
        return False, "invalid_kernel_variant"
    if c["threads"] <= 0:
        return False, "threads_must_be_positive"
    if c["unroll_k"] <= 0:
        return False, "unroll_k_must_be_positive"
    if variant == "naive":
        return True, "ok"
    for k in ("block_m", "block_n", "block_k"):
        if c[k] <= 0:
            return False, f"{k}_must_be_positive_for_blocked_variants"
    return True, "ok"


def evaluate_candidate(
    *,
    runner_binary: Path,
    candidate: dict[str, Any],
    shapes: list[dict[str, int]],
    warmup_iters: int,
    measure_iters: int,
    trial_timeout_sec: int,
    tiny_shape: dict[str, int],
    checksum_atol: float,
    checksum_rtol: float,
    stability_repeat: int,
) -> dict[str, Any]:
    c = _normalize_candidate(candidate)
    ok, reason = _validate_candidate(c)
    if not ok:
        return {
            "correctness_pass": False,
            "failure_stage": "gate1_schema",
            "reason": reason,
            "rows": [],
            "avg_latency_us": 0.0,
            "avg_gflops": 0.0,
            "candidate_signature": candidate_signature(c),
        }

    tiny = {"name": "tiny_gate", **tiny_shape}
    tiny_row = _run_runner(
        runner_binary=runner_binary,
        shape=tiny,
        candidate=c,
        warmup=0,
        iters=1,
        verify=False,
        timeout_sec=trial_timeout_sec,
        input_mode="deterministic",
    )
    if not tiny_row.get("valid", False):
        return {
            "correctness_pass": False,
            "failure_stage": "gate2_deterministic_run",
            "reason": tiny_row.get("error", "tiny_gate_run_invalid"),
            "rows": [tiny_row],
            "avg_latency_us": 0.0,
            "avg_gflops": 0.0,
            "candidate_signature": candidate_signature(c),
        }

    ref = deterministic_reference_checksum(int(tiny_shape["m"]), int(tiny_shape["n"]), int(tiny_shape["k"]))
    out_sum = float(tiny_row.get("output_sum", 0.0))
    out_l2 = float(tiny_row.get("output_l2", 0.0))
    sum_ok = math.isclose(out_sum, float(ref["output_sum"]), rel_tol=checksum_rtol, abs_tol=checksum_atol)
    l2_ok = math.isclose(out_l2, float(ref["output_l2"]), rel_tol=checksum_rtol, abs_tol=checksum_atol)
    if not (sum_ok and l2_ok):
        return {
            "correctness_pass": False,
            "failure_stage": "gate2_checksum",
            "reason": "checksum_mismatch",
            "rows": [tiny_row],
            "avg_latency_us": 0.0,
            "avg_gflops": 0.0,
            "candidate_signature": candidate_signature(c),
        }

    rows: list[dict[str, Any]] = []
    for shape in shapes:
        row = _run_runner(
            runner_binary=runner_binary,
            shape=shape,
            candidate=c,
            warmup=warmup_iters,
            iters=measure_iters,
            verify=True,
            timeout_sec=trial_timeout_sec,
            input_mode="random",
        )
        rows.append(row)
        if not row.get("valid", False):
            return {
                "correctness_pass": False,
                "failure_stage": "gate3_shape_correctness",
                "reason": row.get("error", "shape_invalid"),
                "rows": rows,
                "avg_latency_us": 0.0,
                "avg_gflops": 0.0,
                "candidate_signature": candidate_signature(c),
            }

    if shapes and stability_repeat > 0:
        for _ in range(stability_repeat):
            row = _run_runner(
                runner_binary=runner_binary,
                shape=shapes[0],
                candidate=c,
                warmup=1,
                iters=1,
                verify=True,
                timeout_sec=trial_timeout_sec,
                input_mode="random",
            )
            if not row.get("valid", False):
                return {
                    "correctness_pass": False,
                    "failure_stage": "gate4_stability",
                    "reason": row.get("error", "stability_invalid"),
                    "rows": rows + [row],
                    "avg_latency_us": 0.0,
                    "avg_gflops": 0.0,
                    "candidate_signature": candidate_signature(c),
                }

    avg_latency_ms = sum(float(r.get("latency_ms_p50", 0.0)) for r in rows) / max(1, len(rows))
    avg_latency_us = avg_latency_ms * 1000.0
    avg_gflops = sum(float(r.get("gflops", 0.0)) for r in rows) / max(1, len(rows))
    return {
        "correctness_pass": True,
        "failure_stage": "",
        "reason": "ok",
        "rows": rows,
        "avg_latency_us": avg_latency_us,
        "avg_gflops": avg_gflops,
        "candidate_signature": candidate_signature(c),
    }

