from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_autoresearch_v2.harness.bench import evaluate_candidate

from .candidate_generator import (
    baseline_candidate,
    candidate_signature,
    generate_pool_for_bucket,
    mutate_candidate,
    split_shapes_by_bucket,
)
from .decision_policy import make_decision
from .harness_guard import verify_manifest
from .logbook import (
    append_results_tsv,
    append_run_log,
    init_logbook,
    load_state_or_default,
    save_state,
)
from .report import render_final_report
from .runner_client import build_cpp_runner
from .utils import dump_json, ensure_dir, load_yaml, write_jsonl


def _score_against_baseline(
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


def _default_state(
    *,
    bucket_order: list[str],
    max_iterations: int,
    patience_no_improve: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "running",
        "iteration": 0,
        "max_iterations": max_iterations,
        "bucket_order": bucket_order,
        "active_bucket_idx": 0,
        "no_improve_count": 0,
        "patience_no_improve": patience_no_improve,
        "candidate_cursor_by_bucket": {b: 0 for b in bucket_order},
        "seen_signatures_by_bucket": {b: [] for b in bucket_order},
        "baseline_by_bucket": {},
        "best_by_bucket": {},
        "history_tail": [],
    }


def _build_hypothesis(candidate: dict[str, Any], bucket: str) -> str:
    return (
        f"bucket={bucket} try {candidate['kernel_variant']} "
        f"(bm={candidate['block_m']},bn={candidate['block_n']},bk={candidate['block_k']},"
        f"th={candidate['threads']},uk={candidate['unroll_k']},simd={int(candidate['simd'])})"
    )


def _ensure_artifact(artifact_path: Path) -> dict[str, Any]:
    ensure_dir(artifact_path.parent)
    if artifact_path.exists():
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    c = baseline_candidate()
    artifact_path.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    return c


def _prepare_context(config_path: Path, repo_root: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    paths_cfg = cfg["paths"]
    exp_cfg = cfg["experiment"]
    auto_cfg = cfg["autoresearch"]
    cand_cfg = cfg["candidate_space"]

    shapes_cfg = load_yaml(repo_root / paths_cfg["shapes_config"])
    shapes = list(shapes_cfg.get("shapes", []))
    buckets_cfg = exp_cfg["buckets"]
    by_bucket = split_shapes_by_bucket(shapes=shapes, buckets_cfg=buckets_cfg)
    bucket_order = [b for b in buckets_cfg if by_bucket.get(b)]

    pools = {
        b: generate_pool_for_bucket(
            bucket=b,
            shapes=by_bucket[b],
            candidate_cfg=cand_cfg,
            max_candidates=int(exp_cfg.get("max_candidates_per_bucket", 96)),
            seed=int(exp_cfg.get("seed", 7)),
        )
        for b in bucket_order
    }
    return {
        "cfg": cfg,
        "paths_cfg": paths_cfg,
        "exp_cfg": exp_cfg,
        "auto_cfg": auto_cfg,
        "cand_cfg": cand_cfg,
        "shapes": shapes,
        "by_bucket": by_bucket,
        "bucket_order": bucket_order,
        "pools": pools,
        "buckets_cfg": buckets_cfg,
    }


def run_once(config_path: Path, repo_root: Path) -> dict[str, Any]:
    ctx = _prepare_context(config_path, repo_root)
    auto_cfg = ctx["auto_cfg"]
    exp_cfg = ctx["exp_cfg"]

    cpp_cfg = ctx["paths_cfg"]["cpp_runner"]
    runner = build_cpp_runner(
        repo_root=repo_root,
        source_dir=repo_root / cpp_cfg["source_dir"],
        build_dir=repo_root / cpp_cfg["build_dir"],
        build_type=cpp_cfg["cmake_build_type"],
        binary_name=cpp_cfg["binary_name"],
    )

    harness_manifest = repo_root / auto_cfg["harness_manifest"]
    ok_manifest, mismatches = verify_manifest(repo_root=repo_root, manifest_path=harness_manifest)
    if not ok_manifest:
        raise RuntimeError(f"harness manifest mismatch: {mismatches}")

    workspace_dir = repo_root / auto_cfg["workspace_dir"]
    artifact = _ensure_artifact(workspace_dir / auto_cfg["artifact_file"])
    gate_cfg = auto_cfg["gates"]
    result = evaluate_candidate(
        runner_binary=runner,
        candidate=artifact,
        shapes=ctx["shapes"],
        warmup_iters=int(exp_cfg["warmup_iters"]),
        measure_iters=int(exp_cfg["measure_iters"]),
        trial_timeout_sec=int(exp_cfg["trial_timeout_sec"]),
        tiny_shape=gate_cfg["tiny_shape"],
        checksum_atol=float(gate_cfg["checksum_atol"]),
        checksum_rtol=float(gate_cfg["checksum_rtol"]),
        stability_repeat=int(gate_cfg["stability_repeat"]),
    )
    # Structured metric lines for autoresearch parsers.
    print(f"METRIC avg_latency_us={float(result['avg_latency_us']):.3f}")
    print(f"METRIC avg_gflops={float(result['avg_gflops']):.6f}")
    print(f"METRIC correctness_pass={int(bool(result['correctness_pass']))}")
    return result


def run_orchestration(
    config_path: Path,
    repo_root: Path,
    *,
    agent_mode_override: str | None = None,
) -> dict[str, Any]:
    ctx = _prepare_context(config_path, repo_root)
    cfg = ctx["cfg"]
    auto_cfg = ctx["auto_cfg"]
    exp_cfg = ctx["exp_cfg"]
    cand_cfg = ctx["cand_cfg"]
    buckets_cfg = ctx["buckets_cfg"]
    bucket_order = ctx["bucket_order"]
    pools = ctx["pools"]
    by_bucket = ctx["by_bucket"]

    workspace_dir = repo_root / auto_cfg["workspace_dir"]
    artifact_path = workspace_dir / auto_cfg["artifact_file"]
    harness_manifest = repo_root / auto_cfg["harness_manifest"]
    results_tsv_rel = auto_cfg["results_tsv"]
    run_log_rel = auto_cfg["run_log"]
    state_rel = auto_cfg["state_file"]
    final_report_rel = auto_cfg["final_report"]

    max_iterations = int(auto_cfg["max_iterations"])
    patience_no_improve = int(auto_cfg["patience_no_improve"])
    min_improve_ratio = float(auto_cfg["min_improve_ratio"])
    gate_cfg = auto_cfg["gates"]
    agent_cfg = dict(auto_cfg.get("agent", {}))
    if agent_mode_override:
        agent_cfg["mode"] = agent_mode_override
    agent_mode = str(agent_cfg.get("mode", "rules_only"))
    mutation_rate = float(agent_cfg.get("heuristic_mutation_rate", 0.35))

    cpp_cfg = ctx["paths_cfg"]["cpp_runner"]
    runner = build_cpp_runner(
        repo_root=repo_root,
        source_dir=repo_root / cpp_cfg["source_dir"],
        build_dir=repo_root / cpp_cfg["build_dir"],
        build_type=cpp_cfg["cmake_build_type"],
        binary_name=cpp_cfg["binary_name"],
    )

    ok_manifest, mismatches = verify_manifest(repo_root=repo_root, manifest_path=harness_manifest)
    if not ok_manifest:
        raise RuntimeError(f"harness manifest mismatch: {mismatches}")

    logs = init_logbook(
        workspace_dir=workspace_dir,
        results_tsv_rel=results_tsv_rel,
        run_log_rel=run_log_rel,
        state_rel=state_rel,
    )
    state = load_state_or_default(
        logs["state"],
        _default_state(
            bucket_order=bucket_order,
            max_iterations=max_iterations,
            patience_no_improve=patience_no_improve,
        ),
    )
    _ensure_artifact(artifact_path)

    if not state.get("baseline_by_bucket"):
        for bucket in bucket_order:
            baseline = pools[bucket][0]
            gate = evaluate_candidate(
                runner_binary=runner,
                candidate=baseline,
                shapes=by_bucket[bucket],
                warmup_iters=int(exp_cfg["warmup_iters"]),
                measure_iters=int(exp_cfg["measure_iters"]),
                trial_timeout_sec=int(exp_cfg["trial_timeout_sec"]),
                tiny_shape=gate_cfg["tiny_shape"],
                checksum_atol=float(gate_cfg["checksum_atol"]),
                checksum_rtol=float(gate_cfg["checksum_rtol"]),
                stability_repeat=int(gate_cfg["stability_repeat"]),
            )
            if not gate["correctness_pass"]:
                raise RuntimeError(f"baseline failed bucket={bucket}: {gate['reason']}")
            state["baseline_by_bucket"][bucket] = {
                "avg_latency_us": float(gate["avg_latency_us"]),
                "avg_gflops": float(gate["avg_gflops"]),
            }
            state["best_by_bucket"][bucket] = {
                "score": 1.0,
                "candidate": baseline,
                "candidate_signature": candidate_signature(baseline),
            }
            state["seen_signatures_by_bucket"][bucket] = [candidate_signature(baseline)]
        save_state(logs["state"], state)

    detail_dir = workspace_dir / "results"
    ensure_dir(detail_dir)
    all_rows: list[dict[str, Any]] = []

    while (
        state["status"] == "running"
        and int(state["iteration"]) < int(state["max_iterations"])
        and int(state["active_bucket_idx"]) < len(bucket_order)
    ):
        bucket = bucket_order[int(state["active_bucket_idx"])]
        candidates = pools[bucket]
        cursor = int(state["candidate_cursor_by_bucket"].get(bucket, 0))
        if cursor >= len(candidates):
            append_run_log(logs["run_log"], f"move_on bucket={bucket} reason=exhausted_candidates")
            state["active_bucket_idx"] = int(state["active_bucket_idx"]) + 1
            state["no_improve_count"] = 0
            save_state(logs["state"], state)
            continue

        candidate = dict(candidates[cursor])
        state["candidate_cursor_by_bucket"][bucket] = cursor + 1

        # Hybrid / agent_only uses heuristic mutation now; future can plug LLM proposer here.
        if agent_mode in ("hybrid", "agent_only"):
            best_cand = state["best_by_bucket"][bucket]["candidate"]
            iter_seed = int(cfg["experiment"].get("seed", 7)) + int(state["iteration"]) * 97
            if ((iter_seed % 100) / 100.0) < mutation_rate:
                candidate = mutate_candidate(base=best_cand, candidate_cfg=cand_cfg, iteration_seed=iter_seed)

        cand_sig = candidate_signature(candidate)
        seen = set(state["seen_signatures_by_bucket"].get(bucket, []))
        if cand_sig in seen:
            append_run_log(
                logs["run_log"],
                f"iter={state['iteration']} bucket={bucket} skip=seen_candidate candidate={cand_sig}",
            )
            continue
        state["seen_signatures_by_bucket"].setdefault(bucket, []).append(cand_sig)
        hypothesis = _build_hypothesis(candidate, bucket)

        prev_artifact = artifact_path.read_text(encoding="utf-8")
        artifact_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")

        ok_iter_manifest, iter_mismatches = verify_manifest(repo_root=repo_root, manifest_path=harness_manifest)
        if not ok_iter_manifest:
            raise RuntimeError(f"harness manifest mismatch during loop: {iter_mismatches}")

        gate = evaluate_candidate(
            runner_binary=runner,
            candidate=candidate,
            shapes=by_bucket[bucket],
            warmup_iters=int(exp_cfg["warmup_iters"]),
            measure_iters=int(exp_cfg["measure_iters"]),
            trial_timeout_sec=int(exp_cfg["trial_timeout_sec"]),
            tiny_shape=gate_cfg["tiny_shape"],
            checksum_atol=float(gate_cfg["checksum_atol"]),
            checksum_rtol=float(gate_cfg["checksum_rtol"]),
            stability_repeat=int(gate_cfg["stability_repeat"]),
        )
        base = state["baseline_by_bucket"][bucket]
        alpha = float(buckets_cfg[bucket]["alpha_throughput"])
        score = _score_against_baseline(
            avg_latency_us=float(gate["avg_latency_us"]),
            avg_gflops=float(gate["avg_gflops"]),
            baseline_latency_us=float(base["avg_latency_us"]),
            baseline_gflops=float(base["avg_gflops"]),
            alpha_throughput=alpha,
        )

        best_before = float(state["best_by_bucket"][bucket]["score"])
        decision = make_decision(
            best_score=best_before,
            current_score=score,
            correctness_pass=bool(gate["correctness_pass"]),
            min_improve_ratio=min_improve_ratio,
        )

        keep = decision["decision"] == "keep"
        if keep:
            state["best_by_bucket"][bucket] = {
                "score": float(decision["new_best_score"]),
                "candidate": candidate,
                "candidate_signature": cand_sig,
            }
            state["no_improve_count"] = 0
        else:
            artifact_path.write_text(prev_artifact, encoding="utf-8")
            state["no_improve_count"] = int(state["no_improve_count"]) + 1

        state["iteration"] = int(state["iteration"]) + 1
        state["history_tail"] = (
            state.get("history_tail", [])
            + [
                {
                    "iteration": state["iteration"],
                    "bucket": bucket,
                    "candidate_signature": cand_sig,
                    "decision": decision["decision"],
                    "reason": decision["reason"],
                    "score": score,
                    "correctness_pass": gate["correctness_pass"],
                }
            ]
        )[-40:]

        append_results_tsv(
            logs["results_tsv"],
            iteration=int(state["iteration"]),
            bucket=bucket,
            candidate_signature=cand_sig,
            correctness_pass=bool(gate["correctness_pass"]),
            avg_latency_us=float(gate["avg_latency_us"]),
            avg_gflops=float(gate["avg_gflops"]),
            score=score,
            best_score_before=best_before,
            best_score_after=float(state["best_by_bucket"][bucket]["score"]),
            decision=str(decision["decision"]),
            reason=str(decision["reason"]),
            hypothesis=hypothesis,
        )
        append_run_log(
            logs["run_log"],
            f"iter={state['iteration']} bucket={bucket} candidate={cand_sig} "
            f"decision={decision['decision']} score={score:.6f} reason={decision['reason']}",
        )

        iter_detail = {
            "iteration": state["iteration"],
            "bucket": bucket,
            "hypothesis": hypothesis,
            "candidate": candidate,
            "gate": gate,
            "score": score,
            "decision": decision,
        }
        dump_json(detail_dir / f"iter_{int(state['iteration']):04d}.json", iter_detail)
        all_rows.extend(gate.get("rows", []))

        if int(state["no_improve_count"]) >= int(state["patience_no_improve"]):
            append_run_log(logs["run_log"], f"move_on bucket={bucket} reason=patience_exceeded")
            state["active_bucket_idx"] = int(state["active_bucket_idx"]) + 1
            state["no_improve_count"] = 0

        save_state(logs["state"], state)

    state["status"] = "completed"
    save_state(logs["state"], state)

    write_jsonl(detail_dir / "all_trials.jsonl", all_rows)
    best_config = {
        "best_by_bucket": state.get("best_by_bucket", {}),
        "baseline_by_bucket": state.get("baseline_by_bucket", {}),
    }
    dump_json(workspace_dir / "best_config.json", best_config)
    render_final_report(
        best_by_bucket=state.get("best_by_bucket", {}),
        baseline_by_bucket=state.get("baseline_by_bucket", {}),
        out_path=workspace_dir / final_report_rel,
    )
    return {
        "workspace_dir": str(workspace_dir),
        "artifact": str(artifact_path),
        "state": str(logs["state"]),
        "results_tsv": str(logs["results_tsv"]),
        "run_log": str(logs["run_log"]),
        "best_config": str(workspace_dir / "best_config.json"),
        "final_report": str(workspace_dir / final_report_rel),
    }

