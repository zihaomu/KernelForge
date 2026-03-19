from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from kernel_autoresearch.harness.reference import deterministic_reference_checksum


def candidate_signature(candidate: dict[str, Any]) -> str:
    keys = (
        "kernel_variant",
        "bm",
        "bn",
        "bk",
        "pack_a",
        "pack_b",
        "simd",
        "threads",
        "unroll_k",
        "input_dtype",
        "output_dtype",
    )
    return "|".join(str(candidate.get(k)) for k in keys)


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
        "--input_dtype",
        str(candidate.get("input_dtype", "f32")),
        "--output_dtype",
        str(candidate.get("output_dtype", "f32")),
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
            "error": f"runner rc={cp.returncode}: {cp.stderr.strip()[:200]}",
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
    required_keys = {
        "kernel_variant",
        "bm",
        "bn",
        "bk",
        "pack_a",
        "pack_b",
        "simd",
        "threads",
        "unroll_k",
    }
    missing = [k for k in required_keys if k not in candidate]
    if missing:
        return False, f"missing_keys:{','.join(sorted(missing))}"
    if str(candidate["kernel_variant"]) not in ("naive", "blocked", "blocked_pack"):
        return False, "invalid_kernel_variant"
    input_dtype = str(candidate.get("input_dtype", "f32"))
    output_dtype = str(candidate.get("output_dtype", "f32"))
    if input_dtype not in ("f32", "f16", "i8"):
        return False, "invalid_input_dtype"
    if input_dtype == "i8" and output_dtype != "i32":
        return False, "invalid_dtype_pair_i8_requires_i32"
    if input_dtype == "f16" and output_dtype != "f16":
        return False, "invalid_dtype_pair_f16_requires_f16"
    if input_dtype == "f32" and output_dtype != "f32":
        return False, "invalid_dtype_pair_f32_requires_f32"
    for key in ("bm", "bn", "bk", "threads", "unroll_k"):
        if int(candidate[key]) < 0:
            return False, f"negative_param:{key}"
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
    ok, reason = _validate_candidate(candidate)
    if not ok:
        return {
            "correctness_pass": False,
            "failure_stage": "gate1_schema",
            "reason": reason,
            "rows": [],
            "avg_latency_ms": 0.0,
            "avg_gflops": 0.0,
            "candidate_signature": candidate_signature(candidate),
        }

    tiny = {"name": "tiny_gate", **tiny_shape}
    tiny_row = _run_runner(
        runner_binary=runner_binary,
        shape=tiny,
        candidate=candidate,
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
            "reason": tiny_row.get("error", "tiny_run_invalid"),
            "rows": [tiny_row],
            "avg_latency_ms": 0.0,
            "avg_gflops": 0.0,
            "candidate_signature": candidate_signature(candidate),
        }
    ref = deterministic_reference_checksum(
        int(tiny_shape["m"]),
        int(tiny_shape["n"]),
        int(tiny_shape["k"]),
        input_dtype=str(candidate.get("input_dtype", "f32")),
    )
    out_sum = float(tiny_row.get("output_sum", 0.0))
    out_l2 = float(tiny_row.get("output_l2", 0.0))
    ref_sum = float(ref["output_sum"])
    ref_l2 = float(ref["output_l2"])
    gate_rtol = checksum_rtol
    gate_atol = checksum_atol
    if str(candidate.get("input_dtype", "f32")) == "f16":
        gate_rtol *= 20.0
        gate_atol *= 20.0
    sum_ok = math.isclose(out_sum, ref_sum, rel_tol=gate_rtol, abs_tol=gate_atol)
    l2_ok = math.isclose(out_l2, ref_l2, rel_tol=gate_rtol, abs_tol=gate_atol)
    if not (sum_ok and l2_ok):
        return {
            "correctness_pass": False,
            "failure_stage": "gate2_checksum",
            "reason": f"checksum_mismatch sum={out_sum} ref={ref_sum} l2={out_l2} ref_l2={ref_l2}",
            "rows": [tiny_row],
            "avg_latency_ms": 0.0,
            "avg_gflops": 0.0,
            "candidate_signature": candidate_signature(candidate),
        }

    rows = []
    for shape in shapes:
        row = _run_runner(
            runner_binary=runner_binary,
            shape=shape,
            candidate=candidate,
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
                "avg_latency_ms": 0.0,
                "avg_gflops": 0.0,
                "candidate_signature": candidate_signature(candidate),
            }

    if shapes and stability_repeat > 0:
        shape = shapes[0]
        for _ in range(stability_repeat):
            stable_row = _run_runner(
                runner_binary=runner_binary,
                shape=shape,
                candidate=candidate,
                warmup=1,
                iters=1,
                verify=True,
                timeout_sec=trial_timeout_sec,
                input_mode="random",
            )
            if not stable_row.get("valid", False):
                return {
                    "correctness_pass": False,
                    "failure_stage": "gate4_stability",
                    "reason": stable_row.get("error", "stability_invalid"),
                    "rows": rows + [stable_row],
                    "avg_latency_ms": 0.0,
                    "avg_gflops": 0.0,
                    "candidate_signature": candidate_signature(candidate),
                }

    avg_latency_ms = sum(float(r.get("latency_ms_p50", 0.0)) for r in rows) / max(len(rows), 1)
    avg_gflops = sum(float(r.get("gflops", 0.0)) for r in rows) / max(len(rows), 1)
    return {
        "correctness_pass": True,
        "failure_stage": "",
        "reason": "ok",
        "rows": rows,
        "avg_latency_ms": avg_latency_ms,
        "avg_gflops": avg_gflops,
        "candidate_signature": candidate_signature(candidate),
    }
