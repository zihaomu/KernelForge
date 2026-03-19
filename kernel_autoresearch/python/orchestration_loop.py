from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kernel_autoresearch.harness.bench import candidate_signature, evaluate_candidate

from .agent_proposer import choose_candidate
from .candidate_generator import generate_candidates
from .cloud_patterns import extract_cloud_patterns
from .decision_policy import make_decision
from .harness_guard import verify_manifest
from .local_patterns import extract_local_patterns
from .logbook import (
    append_results_tsv,
    append_run_log,
    init_logbook,
    load_state_or_default,
    save_state,
)
from .pattern_merge import merge_patterns
from .runner_client import build_cpp_runner
from .scheduler import bucket_for_shape
from .utils import dump_json, ensure_dir, load_yaml, write_jsonl


def _score_against_baseline(
    *,
    avg_latency_ms: float,
    avg_gflops: float,
    baseline_latency_ms: float,
    baseline_gflops: float,
    alpha_throughput: float,
) -> float:
    if avg_latency_ms <= 0 or avg_gflops <= 0:
        return -1e9
    tp = avg_gflops / max(baseline_gflops, 1e-9)
    lat = baseline_latency_ms / max(avg_latency_ms, 1e-9)
    return alpha_throughput * tp + (1.0 - alpha_throughput) * lat


def _bucket_shapes(shapes: list[dict[str, int]], buckets_cfg: dict[str, Any]) -> dict[str, list[dict[str, int]]]:
    out: dict[str, list[dict[str, int]]] = {k: [] for k in buckets_cfg}
    for shape in shapes:
        b = bucket_for_shape(shape, buckets_cfg)
        out.setdefault(b, []).append(shape)
    return out


def _candidate_pools_by_bucket(
    *,
    shapes: list[dict[str, int]],
    buckets_cfg: dict[str, Any],
    pattern_scores: dict[str, float],
    candidate_cfg: dict[str, Any],
    max_trials_per_bucket: int,
    seed: int,
    input_dtype: str,
) -> dict[str, list[dict[str, Any]]]:
    by_shape = generate_candidates(
        shapes=shapes,
        buckets_cfg=buckets_cfg,
        pattern_scores=pattern_scores,
        candidate_cfg=candidate_cfg,
        max_trials_per_bucket=max_trials_per_bucket,
        seed=seed,
        input_dtype=input_dtype,
    )
    pool: dict[str, dict[str, dict[str, Any]]] = {k: {} for k in buckets_cfg}
    for shape in shapes:
        bucket = bucket_for_shape(shape, buckets_cfg)
        for cand in by_shape[shape["name"]]:
            sig = candidate_signature(cand)
            pool[bucket][sig] = cand

    fallback_baseline = {
        "kernel_variant": "naive",
        "bm": 0,
        "bn": 0,
        "bk": 0,
        "pack_a": False,
        "pack_b": False,
        "simd": False,
        "threads": 1,
        "unroll_k": 1,
        "input_dtype": input_dtype,
        "output_dtype": "i32" if input_dtype == "i8" else ("f16" if input_dtype == "f16" else "f32"),
    }

    def pick_baseline(values: list[dict[str, Any]]) -> dict[str, Any]:
        for c in values:
            if (
                c.get("kernel_variant") == "blocked"
                and c.get("simd", False)
                and int(c.get("threads", 1)) <= 8
            ):
                return c
        for c in values:
            if c.get("kernel_variant") == "blocked_pack" and c.get("simd", False):
                return c
        for c in values:
            if c.get("kernel_variant") == "naive":
                return c
        return fallback_baseline

    final: dict[str, list[dict[str, Any]]] = {}
    for bucket, cand_map in pool.items():
        values = list(cand_map.values())
        values.sort(key=candidate_signature)
        baseline = pick_baseline(values)
        # Ensure baseline is the first candidate.
        values = [c for c in values if candidate_signature(c) != candidate_signature(baseline)]
        final[bucket] = [baseline] + values
    return final


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
        "best_score_by_bucket": {},
        "best_candidate_by_bucket": {},
        "baseline_by_bucket": {},
        "seen_signatures_by_bucket": {b: [] for b in bucket_order},
        "history_tail": [],
    }


def run_orchestration(
    config_path: Path,
    repo_root: Path,
    *,
    agent_mode_override: str | None = None,
    agent_model_override: str | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    paths_cfg = cfg["paths"]
    search_cfg = cfg["search"]
    exp_cfg = cfg["experiment"]
    cand_cfg = cfg["candidate_space"]
    auto_cfg = cfg["autoresearch"]
    input_dtype = str(exp_cfg.get("input_dtype", "f32")).strip().lower()
    if input_dtype not in ("f32", "f16", "i8"):
        raise ValueError(f"unsupported experiment.input_dtype: {input_dtype}")
    output_dtype = "i32" if input_dtype == "i8" else ("f16" if input_dtype == "f16" else "f32")

    code_base_agent_gen_dir = repo_root / paths_cfg["code_base_agent_gen_dir"]
    pattern_db_dir = repo_root / paths_cfg["pattern_db_dir"]
    shapes_cfg_path = repo_root / paths_cfg["shapes_config"]
    cloud_sources_path = repo_root / paths_cfg["cloud_sources_config"]
    cpp_cfg = paths_cfg["cpp_runner"]

    workspace_dir = repo_root / auto_cfg["workspace_dir"]
    artifact_path = workspace_dir / auto_cfg["artifact_file"]
    harness_manifest = repo_root / auto_cfg["harness_manifest"]
    results_tsv_rel = auto_cfg["results_tsv"]
    run_log_rel = auto_cfg["run_log"]
    state_rel = auto_cfg["state_file"]
    min_improve_ratio = float(auto_cfg["min_improve_ratio"])
    max_iterations = int(auto_cfg["max_iterations"])
    patience_no_improve = int(auto_cfg["patience_no_improve"])
    gate_cfg = auto_cfg["gates"]
    agent_cfg = dict(auto_cfg.get("agent", {}))
    if agent_mode_override:
        agent_cfg["mode"] = agent_mode_override
    if agent_model_override:
        agent_cfg["model"] = agent_model_override
    agent_mode = str(agent_cfg.get("mode", "rules_only"))

    ensure_dir(pattern_db_dir)
    local_path = pattern_db_dir / "local_patterns.json"
    cloud_path = pattern_db_dir / "cloud_patterns.json"
    merged_path = pattern_db_dir / "merged_patterns.json"

    local = extract_local_patterns(repo_root, code_base_agent_gen_dir, local_path)
    _ = extract_cloud_patterns(
        cloud_sources_config=cloud_sources_path,
        output_path=cloud_path,
        enabled=bool(search_cfg.get("enabled", True)),
        timeout_sec=int(search_cfg.get("timeout_sec", 10)),
        max_results_per_query=int(search_cfg.get("max_results_per_query", 5)),
        user_agent=str(search_cfg.get("user_agent", "kernel-autoresearch/0.1")),
    )
    merged = merge_patterns(local_path, cloud_path, merged_path)
    pattern_scores = merged.get("pattern_scores", {})

    shapes_cfg = load_yaml(shapes_cfg_path)
    shapes = shapes_cfg.get("shapes", [])
    buckets_cfg = exp_cfg["buckets"]
    shape_groups = _bucket_shapes(shapes, buckets_cfg)
    bucket_order = [b for b in buckets_cfg if shape_groups.get(b)]

    pools = _candidate_pools_by_bucket(
        shapes=shapes,
        buckets_cfg=buckets_cfg,
        pattern_scores=pattern_scores,
        candidate_cfg=cand_cfg,
        max_trials_per_bucket=int(exp_cfg["max_trials_per_bucket"]),
        seed=int(exp_cfg.get("seed", 7)),
        input_dtype=input_dtype,
    )

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
    state.setdefault("seen_signatures_by_bucket", {})
    for b in bucket_order:
        state["seen_signatures_by_bucket"].setdefault(b, [])

    ensure_dir(artifact_path.parent)
    if not artifact_path.exists():
        baseline_candidate = pools[bucket_order[0]][0] if bucket_order else {
            "kernel_variant": "naive",
            "bm": 0,
            "bn": 0,
            "bk": 0,
            "pack_a": False,
            "pack_b": False,
            "simd": False,
            "threads": 1,
            "unroll_k": 1,
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
        }
        artifact_path.write_text(json.dumps(baseline_candidate, ensure_ascii=False, indent=2), encoding="utf-8")

    # Initialize bucket baselines only once.
    if not state["baseline_by_bucket"]:
        for bucket in bucket_order:
            baseline_candidate = pools[bucket][0]
            gate = evaluate_candidate(
                runner_binary=runner,
                candidate=baseline_candidate,
                shapes=shape_groups[bucket],
                warmup_iters=int(exp_cfg["warmup_iters"]),
                measure_iters=int(exp_cfg["measure_iters"]),
                trial_timeout_sec=int(exp_cfg["trial_timeout_sec"]),
                tiny_shape=gate_cfg["tiny_shape"],
                checksum_atol=float(gate_cfg["checksum_atol"]),
                checksum_rtol=float(gate_cfg["checksum_rtol"]),
                stability_repeat=int(gate_cfg["stability_repeat"]),
            )
            if not gate["correctness_pass"]:
                raise RuntimeError(f"baseline candidate failed for bucket={bucket}: {gate['reason']}")
            state["baseline_by_bucket"][bucket] = {
                "avg_latency_ms": gate["avg_latency_ms"],
                "avg_gflops": gate["avg_gflops"],
            }
            state["best_score_by_bucket"][bucket] = 1.0
            state["best_candidate_by_bucket"][bucket] = baseline_candidate
            state.setdefault("seen_signatures_by_bucket", {}).setdefault(bucket, []).append(
                candidate_signature(baseline_candidate)
            )
        save_state(logs["state"], state)

    detail_dir = workspace_dir / "results"
    ensure_dir(detail_dir)
    all_rows: list[dict[str, Any]] = []

    while (
        state["status"] == "running"
        and state["iteration"] < state["max_iterations"]
        and state["active_bucket_idx"] < len(bucket_order)
    ):
        bucket = bucket_order[state["active_bucket_idx"]]
        cursor = int(state["candidate_cursor_by_bucket"].get(bucket, 0))
        candidates = pools[bucket]
        if cursor >= len(candidates):
            append_run_log(logs["run_log"], f"move_on bucket={bucket} reason=exhausted_candidates")
            state["active_bucket_idx"] += 1
            state["no_improve_count"] = 0
            save_state(logs["state"], state)
            continue

        cursor_candidate = candidates[cursor]
        state["candidate_cursor_by_bucket"][bucket] = cursor + 1
        seen_set = set(state.get("seen_signatures_by_bucket", {}).get(bucket, []))
        proposal = choose_candidate(
            mode=agent_mode,
            agent_cfg=agent_cfg,
            bucket=bucket,
            cursor_candidate=cursor_candidate,
            pool=candidates,
            seen_signatures=seen_set,
            history_tail=state.get("history_tail", []),
            best_candidate=state.get("best_candidate_by_bucket", {}).get(bucket),
            baseline=state.get("baseline_by_bucket", {}).get(bucket),
            iteration_seed=int(state["iteration"]) + 17,
        )
        candidate = proposal["candidate"]
        proposal_source = str(proposal.get("proposal_source", "rules"))
        proposal_note = str(proposal.get("proposal_note", ""))
        cand_sig = candidate_signature(candidate)
        state.setdefault("seen_signatures_by_bucket", {}).setdefault(bucket, [])
        if cand_sig not in state["seen_signatures_by_bucket"][bucket]:
            state["seen_signatures_by_bucket"][bucket].append(cand_sig)

        prev_artifact = artifact_path.read_text(encoding="utf-8")
        artifact_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")

        ok_manifest_iter, mismatches_iter = verify_manifest(repo_root=repo_root, manifest_path=harness_manifest)
        if not ok_manifest_iter:
            raise RuntimeError(f"harness manifest mismatch during iteration: {mismatches_iter}")

        gate = evaluate_candidate(
            runner_binary=runner,
            candidate=candidate,
            shapes=shape_groups[bucket],
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
            avg_latency_ms=float(gate["avg_latency_ms"]),
            avg_gflops=float(gate["avg_gflops"]),
            baseline_latency_ms=float(base["avg_latency_ms"]),
            baseline_gflops=float(base["avg_gflops"]),
            alpha_throughput=alpha,
        )
        best_before = state["best_score_by_bucket"].get(bucket)
        decision = make_decision(
            best_score=best_before,
            current_score=score,
            correctness_pass=bool(gate["correctness_pass"]),
            min_improve_ratio=min_improve_ratio,
        )

        keep = decision["decision"] == "keep"
        if keep:
            state["best_score_by_bucket"][bucket] = decision["new_best_score"]
            state["best_candidate_by_bucket"][bucket] = candidate
            state["no_improve_count"] = 0
        else:
            artifact_path.write_text(prev_artifact, encoding="utf-8")
            state["no_improve_count"] = int(state["no_improve_count"]) + 1

        state["iteration"] = int(state["iteration"]) + 1
        state["history_tail"] = (state.get("history_tail", []) + [
            {
                "iteration": state["iteration"],
                "bucket": bucket,
                "candidate_signature": cand_sig,
                "proposal_source": proposal_source,
                "decision": decision["decision"],
                "reason": decision["reason"],
                "score": score,
                "correctness_pass": gate["correctness_pass"],
            }
        ])[-40:]

        append_results_tsv(
            logs["results_tsv"],
            iteration=state["iteration"],
            bucket=bucket,
            candidate_signature=cand_sig,
            correctness_pass=bool(gate["correctness_pass"]),
            avg_latency_ms=float(gate["avg_latency_ms"]),
            avg_gflops=float(gate["avg_gflops"]),
            score=score,
            best_score_before=best_before,
            best_score_after=state["best_score_by_bucket"].get(bucket),
            decision=decision["decision"],
            reason=decision["reason"],
            proposal_source=proposal_source,
            proposal_note=proposal_note,
        )
        append_run_log(
            logs["run_log"],
            f"iter={state['iteration']} bucket={bucket} proposal={proposal_source} candidate={cand_sig} "
            f"decision={decision['decision']} score={score:.6f} reason={decision['reason']}"
            + (f" note={proposal_note}" if proposal_note else ""),
        )

        iter_detail = {
            "iteration": state["iteration"],
            "bucket": bucket,
            "proposal_source": proposal_source,
            "proposal_note": proposal_note,
            "candidate": candidate,
            "gate": gate,
            "score": score,
            "decision": decision,
        }
        dump_json(detail_dir / f"iter_{state['iteration']:04d}.json", iter_detail)
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
        "best_candidate_by_bucket": state.get("best_candidate_by_bucket", {}),
        "best_score_by_bucket": state.get("best_score_by_bucket", {}),
    }
    dump_json(workspace_dir / "best_config.json", best_config)
    return {
        "workspace_dir": str(workspace_dir),
        "artifact": str(artifact_path),
        "state": str(logs["state"]),
        "results_tsv": str(logs["results_tsv"]),
        "run_log": str(logs["run_log"]),
        "best_config": str(workspace_dir / "best_config.json"),
        "local_patterns": str(local_path),
        "merged_patterns": str(merged_path),
        "local_record_count": local.get("stats", {}).get("record_count", 0),
    }
