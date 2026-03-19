from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

from autoresearch.adapters.cpu_cpp.adapter import build_runner, run_gemm_candidate
from autoresearch.core.decision_policy import make_decision
from autoresearch.core.utils import dump_json, ensure_dir, load_yaml
from kernel_autoresearch_v2.harness.reference import deterministic_reference_checksum


def _baseline_candidate() -> dict[str, Any]:
    return {
        "kernel_variant": "naive",
        "block_m": 0,
        "block_n": 0,
        "block_k": 0,
        "pack_a": False,
        "pack_b": False,
        "simd": False,
        "threads": 1,
        "unroll_k": 1,
    }


def _candidate_signature(c: dict[str, Any]) -> str:
    keys = ("kernel_variant", "block_m", "block_n", "block_k", "pack_a", "pack_b", "simd", "threads", "unroll_k")
    return "|".join(str(c[k]) for k in keys)


def _generate_candidates(space: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    max_candidates = int(space.get("max_candidates", 96))
    variants = [str(x) for x in space.get("kernel_variants", ["naive", "blocked", "blocked_pack"])]
    bm = [int(x) for x in space.get("block_m", [32, 64])]
    bn = [int(x) for x in space.get("block_n", [32, 64])]
    bk = [int(x) for x in space.get("block_k", [32, 64])]
    uk = [int(x) for x in space.get("unroll_k", [1, 2])]
    th = [int(x) for x in space.get("thread_choices", [1, 2, 4])]

    out = [_baseline_candidate()]
    for v, m, n, k, t, u in itertools.product(variants, bm, bn, bk, th, uk):
        if v == "naive":
            continue
        uses_pack = v in ("blocked_pack", "blocked_pack_simd")
        simd = v == "blocked_pack_simd"
        out.append(
            {
                "kernel_variant": v,
                "block_m": m,
                "block_n": n,
                "block_k": k,
                "pack_a": uses_pack,
                "pack_b": uses_pack,
                "simd": simd,
                "threads": t,
                "unroll_k": u,
            }
        )
        if v in ("blocked", "blocked_pack"):
            out.append(
                {
                    "kernel_variant": v,
                    "block_m": m,
                    "block_n": n,
                    "block_k": k,
                    "pack_a": uses_pack,
                    "pack_b": uses_pack,
                    "simd": True,
                    "threads": t,
                    "unroll_k": u,
                }
            )

    dedup: dict[str, dict[str, Any]] = {}
    for c in out:
        dedup[_candidate_signature(c)] = c
    naive = [c for c in dedup.values() if c["kernel_variant"] == "naive"]
    others = [c for c in dedup.values() if c["kernel_variant"] != "naive"]
    rng = random.Random(seed)
    rng.shuffle(others)
    return naive + others[: max(0, max_candidates - len(naive))]


def _score(
    *,
    avg_latency_us: float,
    avg_gflops: float,
    baseline_latency_us: float,
    baseline_gflops: float,
    alpha_throughput: float,
) -> float:
    if avg_latency_us <= 0.0 or avg_gflops <= 0.0:
        return -1e9
    tp = avg_gflops / max(baseline_gflops, 1e-9)
    lat = baseline_latency_us / max(avg_latency_us, 1e-9)
    return alpha_throughput * tp + (1.0 - alpha_throughput) * lat


def _evaluate_candidate(
    *,
    runner_binary: Path,
    candidate: dict[str, Any],
    shapes: list[dict[str, int]],
    warmup_iters: int,
    measure_iters: int,
    timeout_sec: int,
    tiny_shape: dict[str, int],
    checksum_atol: float,
    checksum_rtol: float,
    stability_repeat: int,
) -> dict[str, Any]:
    tiny = {"name": "tiny_gate", **tiny_shape}
    tiny_row = run_gemm_candidate(
        runner_binary=runner_binary,
        shape=tiny,
        candidate=candidate,
        warmup=0,
        iters=1,
        verify=False,
        timeout_sec=timeout_sec,
        input_mode="deterministic",
    )
    if not tiny_row.get("valid", False):
        return {
            "correctness_pass": False,
            "reason": tiny_row.get("error", "tiny_gate_run_invalid"),
            "rows": [tiny_row],
            "avg_latency_us": 0.0,
            "avg_gflops": 0.0,
        }

    ref = deterministic_reference_checksum(int(tiny_shape["m"]), int(tiny_shape["n"]), int(tiny_shape["k"]))
    out_sum = float(tiny_row.get("output_sum", 0.0))
    out_l2 = float(tiny_row.get("output_l2", 0.0))
    sum_ok = math.isclose(out_sum, float(ref["output_sum"]), rel_tol=checksum_rtol, abs_tol=checksum_atol)
    l2_ok = math.isclose(out_l2, float(ref["output_l2"]), rel_tol=checksum_rtol, abs_tol=checksum_atol)
    if not (sum_ok and l2_ok):
        return {
            "correctness_pass": False,
            "reason": "checksum_mismatch",
            "rows": [tiny_row],
            "avg_latency_us": 0.0,
            "avg_gflops": 0.0,
        }

    rows: list[dict[str, Any]] = []
    for shape in shapes:
        row = run_gemm_candidate(
            runner_binary=runner_binary,
            shape=shape,
            candidate=candidate,
            warmup=warmup_iters,
            iters=measure_iters,
            verify=True,
            timeout_sec=timeout_sec,
            input_mode="random",
        )
        rows.append(row)
        if not row.get("valid", False):
            return {
                "correctness_pass": False,
                "reason": row.get("error", "shape_invalid"),
                "rows": rows,
                "avg_latency_us": 0.0,
                "avg_gflops": 0.0,
            }

    if shapes and stability_repeat > 0:
        for _ in range(stability_repeat):
            row = run_gemm_candidate(
                runner_binary=runner_binary,
                shape=shapes[0],
                candidate=candidate,
                warmup=1,
                iters=1,
                verify=True,
                timeout_sec=timeout_sec,
                input_mode="random",
            )
            if not row.get("valid", False):
                return {
                    "correctness_pass": False,
                    "reason": row.get("error", "stability_invalid"),
                    "rows": rows + [row],
                    "avg_latency_us": 0.0,
                    "avg_gflops": 0.0,
                }

    avg_latency_ms = sum(float(r.get("latency_ms_p50", 0.0)) for r in rows) / max(1, len(rows))
    avg_gflops = sum(float(r.get("gflops", 0.0)) for r in rows) / max(1, len(rows))
    return {
        "correctness_pass": True,
        "reason": "ok",
        "rows": rows,
        "avg_latency_us": avg_latency_ms * 1000.0,
        "avg_gflops": avg_gflops,
    }


def run_task(
    *,
    repo_root: Path,
    task: dict[str, Any],
    registry_entry: dict[str, Any],
    global_cfg: dict[str, Any],
    pack_cfg_path: Path,
    platform_cfg_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    del task, registry_entry
    pack_cfg = load_yaml(pack_cfg_path)
    platform_cfg = load_yaml(platform_cfg_path)
    exec_cfg = global_cfg["execution"]

    limits = pack_cfg.get("task_limits", {})
    max_iterations = int(limits.get("max_iterations", exec_cfg["max_iterations_per_task"]))
    patience = int(limits.get("patience_no_improve", exec_cfg["patience_no_improve"]))
    min_improve_ratio = float(exec_cfg["min_improve_ratio"])
    alpha = float(pack_cfg["metric"]["alpha_throughput"])

    bench_cfg = platform_cfg["benchmark"]
    warmup_iters = int(bench_cfg["warmup_iters"])
    measure_iters = int(bench_cfg["measure_iters"])
    timeout_sec = int(bench_cfg["timeout_sec"])
    gate_cfg = pack_cfg["gates"]

    runner_cfg = platform_cfg["runner"]
    runner = build_runner(
        repo_root=repo_root,
        source_dir=repo_root / runner_cfg["source_dir"],
        build_dir=repo_root / runner_cfg["build_dir"],
        build_type=runner_cfg["cmake_build_type"],
        binary_name=runner_cfg["binary_name"],
    )

    shapes = list(pack_cfg.get("shapes", []))
    candidates = _generate_candidates(pack_cfg["candidate_space"], int(exec_cfg.get("seed", 7)))
    ensure_dir(run_dir)
    results_path = run_dir / "results.tsv"
    log_path = run_dir / "run.log"
    detail_dir = run_dir / "iter_details"
    ensure_dir(detail_dir)

    header = (
        "iteration\tcandidate_signature\tcorrectness_pass\tavg_latency_us\tavg_gflops\t"
        "score\tbest_score_before\tbest_score_after\tdecision\treason\n"
    )
    results_path.write_text(header, encoding="utf-8")
    log_path.write_text("", encoding="utf-8")

    baseline = candidates[0]
    baseline_eval = _evaluate_candidate(
        runner_binary=runner,
        candidate=baseline,
        shapes=shapes,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        timeout_sec=timeout_sec,
        tiny_shape=pack_cfg["gates"]["tiny_shape"],
        checksum_atol=float(gate_cfg["checksum_atol"]),
        checksum_rtol=float(gate_cfg["checksum_rtol"]),
        stability_repeat=int(gate_cfg["stability_repeat"]),
    )
    if not baseline_eval["correctness_pass"]:
        summary = {
            "status": "error",
            "reason": f"baseline_failed:{baseline_eval['reason']}",
            "iterations": 0,
            "best_score": 0.0,
            "best_candidate_signature": "",
        }
        dump_json(run_dir / "summary.json", summary)
        return summary

    baseline_latency = float(baseline_eval["avg_latency_us"])
    baseline_gflops = float(baseline_eval["avg_gflops"])
    best_score = 1.0
    best_candidate = baseline
    best_sig = _candidate_signature(best_candidate)

    no_improve = 0
    iterations = 0
    all_rows: list[dict[str, Any]] = []
    seen: set[str] = {best_sig}

    for idx in range(1, len(candidates)):
        if iterations >= max_iterations:
            break
        if no_improve >= patience:
            break
        cand = candidates[idx]
        sig = _candidate_signature(cand)
        if sig in seen:
            continue
        seen.add(sig)

        eval_row = _evaluate_candidate(
            runner_binary=runner,
            candidate=cand,
            shapes=shapes,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            timeout_sec=timeout_sec,
            tiny_shape=pack_cfg["gates"]["tiny_shape"],
            checksum_atol=float(gate_cfg["checksum_atol"]),
            checksum_rtol=float(gate_cfg["checksum_rtol"]),
            stability_repeat=int(gate_cfg["stability_repeat"]),
        )
        score = _score(
            avg_latency_us=float(eval_row["avg_latency_us"]),
            avg_gflops=float(eval_row["avg_gflops"]),
            baseline_latency_us=baseline_latency,
            baseline_gflops=baseline_gflops,
            alpha_throughput=alpha,
        )
        decision = make_decision(
            best_score=best_score,
            current_score=score,
            correctness_pass=bool(eval_row["correctness_pass"]),
            min_improve_ratio=min_improve_ratio,
        )
        best_before = best_score
        if decision["decision"] == "keep":
            best_score = float(decision["new_best_score"])
            best_candidate = cand
            best_sig = sig
            no_improve = 0
        else:
            no_improve += 1

        iterations += 1
        log_line = (
            f"iter={iterations} cand={sig} decision={decision['decision']} "
            f"score={score:.6f} reason={decision['reason']}\n"
        )
        log_path.write_text(log_path.read_text(encoding="utf-8") + log_line, encoding="utf-8")
        row = (
            f"{iterations}\t{sig}\t{int(bool(eval_row['correctness_pass']))}\t"
            f"{float(eval_row['avg_latency_us']):.3f}\t{float(eval_row['avg_gflops']):.6f}\t"
            f"{score:.6f}\t{best_before:.6f}\t{best_score:.6f}\t{decision['decision']}\t{decision['reason']}\n"
        )
        results_path.write_text(results_path.read_text(encoding="utf-8") + row, encoding="utf-8")

        detail = {
            "iteration": iterations,
            "candidate": cand,
            "candidate_signature": sig,
            "evaluation": eval_row,
            "score": score,
            "decision": decision,
        }
        all_rows.append(detail)
        dump_json(detail_dir / f"iter_{iterations:04d}.json", detail)

    dump_json(run_dir / "best_candidate.json", best_candidate)
    dump_json(run_dir / "all_iterations.json", all_rows)
    summary = {
        "status": "completed",
        "iterations": iterations,
        "best_score": best_score,
        "best_candidate_signature": best_sig,
        "best_candidate": best_candidate,
        "baseline_latency_us": baseline_latency,
        "baseline_gflops": baseline_gflops,
    }
    dump_json(run_dir / "summary.json", summary)
    return summary

