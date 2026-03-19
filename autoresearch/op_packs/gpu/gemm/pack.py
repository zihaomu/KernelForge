from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any

from autoresearch.adapters.gpu_torch.adapter import run_torch_gemm_candidate, torch_runtime_available
from autoresearch.core.decision_policy import make_decision
from autoresearch.core.utils import dump_json, ensure_dir, load_yaml


def _candidate_signature(c: dict[str, Any]) -> str:
    return f"dtype={c['dtype']}|allow_tf32={int(bool(c['allow_tf32']))}"


def _generate_candidates(space: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    dtypes = [str(x) for x in space.get("dtypes", ["fp32", "fp16"])]
    tf32 = [bool(x) for x in space.get("allow_tf32", [False, True])]
    out = [{"dtype": d, "allow_tf32": t} for d, t in itertools.product(dtypes, tf32)]
    # Keep deterministic baseline first: fp32 + no tf32.
    baseline = [{"dtype": "fp32", "allow_tf32": False}]
    rest = [c for c in out if not (c["dtype"] == "fp32" and c["allow_tf32"] is False)]
    rng = random.Random(seed + 17)
    rng.shuffle(rest)
    return baseline + rest


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
    candidate: dict[str, Any],
    shapes: list[dict[str, int]],
    warmup_iters: int,
    measure_iters: int,
    correctness_cfg: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dtype = str(candidate["dtype"])
    tol = correctness_cfg.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
    atol = float(tol.get("atol", 1e-2))
    rtol = float(tol.get("rtol", 1e-2))

    for shape in shapes:
        row = run_torch_gemm_candidate(
            shape=shape,
            candidate=candidate,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            atol=atol,
            rtol=rtol,
        )
        rows.append(row)
        if not row.get("valid", False):
            return {
                "correctness_pass": False,
                "reason": f"shape_invalid:{shape['name']}",
                "rows": rows,
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
    del repo_root, task, registry_entry
    ok, reason = torch_runtime_available()
    ensure_dir(run_dir)
    if not ok:
        summary = {
            "status": "skipped",
            "reason": reason,
            "iterations": 0,
            "best_score": 0.0,
            "best_candidate_signature": "",
        }
        dump_json(run_dir / "summary.json", summary)
        return summary

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
    correctness_cfg = platform_cfg["correctness"]

    shapes = list(pack_cfg.get("shapes", []))
    candidates = _generate_candidates(pack_cfg["candidate_space"], int(exec_cfg.get("seed", 7)))
    results_path = run_dir / "results.tsv"
    log_path = run_dir / "run.log"
    detail_dir = run_dir / "iter_details"
    ensure_dir(detail_dir)
    results_path.write_text(
        "iteration\tcandidate_signature\tcorrectness_pass\tavg_latency_us\tavg_gflops\t"
        "score\tbest_score_before\tbest_score_after\tdecision\treason\n",
        encoding="utf-8",
    )
    log_path.write_text("", encoding="utf-8")

    baseline = candidates[0]
    baseline_eval = _evaluate_candidate(
        candidate=baseline,
        shapes=shapes,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        correctness_cfg=correctness_cfg,
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

    for cand in candidates[1:]:
        if iterations >= max_iterations:
            break
        if no_improve >= patience:
            break
        sig = _candidate_signature(cand)
        eval_row = _evaluate_candidate(
            candidate=cand,
            shapes=shapes,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            correctness_cfg=correctness_cfg,
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
        before = best_score
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
            f"{score:.6f}\t{before:.6f}\t{best_score:.6f}\t{decision['decision']}\t{decision['reason']}\n"
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

