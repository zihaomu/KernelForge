from __future__ import annotations

from pathlib import Path
from typing import Any

from .candidate_generator import generate_candidates
from .cloud_patterns import extract_cloud_patterns
from .local_patterns import extract_local_patterns
from .pattern_merge import merge_patterns
from .report import generate_report
from .runner_client import build_cpp_runner, run_trial
from .scheduler import build_selection
from .utils import dump_json, ensure_dir, load_yaml, now_ts, write_jsonl


def _candidate_risk(candidate: dict[str, Any], pattern_scores: dict[str, float]) -> str:
    if candidate.get("kernel_variant") == "naive":
        return "low"
    if candidate.get("kernel_variant") == "blocked_pack":
        if pattern_scores.get("pack", 0.0) < 0.5:
            return "high"
        return "medium"
    if candidate.get("threads", 1) >= 16 and not candidate.get("simd", False):
        return "medium"
    return "low"


def run_autoresearch(config_path: Path, repo_root: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    paths_cfg = cfg["paths"]
    search_cfg = cfg["search"]
    exp_cfg = cfg["experiment"]
    scoring_cfg = cfg["scoring"]
    cand_cfg = cfg["candidate_space"]
    input_dtype = str(exp_cfg.get("input_dtype", "f32")).strip().lower()
    if input_dtype not in ("f32", "f16", "i8"):
        raise ValueError(f"unsupported experiment.input_dtype: {input_dtype}")

    code_base_agent_gen_dir = repo_root / paths_cfg["code_base_agent_gen_dir"]
    pattern_db_dir = repo_root / paths_cfg["pattern_db_dir"]
    runs_dir = repo_root / paths_cfg["runs_dir"]
    shapes_cfg_path = repo_root / paths_cfg["shapes_config"]
    cloud_sources_path = repo_root / paths_cfg["cloud_sources_config"]
    cpp_cfg = paths_cfg["cpp_runner"]
    cpp_source = repo_root / cpp_cfg["source_dir"]
    cpp_build = repo_root / cpp_cfg["build_dir"]
    cpp_binary_name = cpp_cfg["binary_name"]
    cpp_build_type = cpp_cfg["cmake_build_type"]

    ensure_dir(pattern_db_dir)
    ensure_dir(runs_dir)
    local_path = pattern_db_dir / "local_patterns.json"
    cloud_path = pattern_db_dir / "cloud_patterns.json"
    merged_path = pattern_db_dir / "merged_patterns.json"

    local = extract_local_patterns(repo_root, code_base_agent_gen_dir, local_path)
    cloud = extract_cloud_patterns(
        cloud_sources_config=cloud_sources_path,
        output_path=cloud_path,
        enabled=bool(search_cfg.get("enabled", True)),
        timeout_sec=int(search_cfg.get("timeout_sec", 10)),
        max_results_per_query=int(search_cfg.get("max_results_per_query", 5)),
        user_agent=str(search_cfg.get("user_agent", "kernel-autoresearch/0.1")),
    )
    merged = merge_patterns(local_path, cloud_path, merged_path)
    _ = cloud  # keep around for future extensions

    shapes_cfg = load_yaml(shapes_cfg_path)
    shapes = shapes_cfg.get("shapes", [])
    buckets_cfg = exp_cfg["buckets"]
    pattern_scores = merged.get("pattern_scores", {})
    candidates_by_shape = generate_candidates(
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
        source_dir=cpp_source,
        build_dir=cpp_build,
        build_type=cpp_build_type,
        binary_name=cpp_binary_name,
    )

    trials_by_shape: dict[str, list[dict[str, Any]]] = {}
    for shape in shapes:
        shape_name = shape["name"]
        trials: list[dict[str, Any]] = []
        for cand in candidates_by_shape[shape_name]:
            risk_level = _candidate_risk(cand, pattern_scores)
            row = run_trial(
                runner_binary=runner,
                shape=shape,
                candidate=cand,
                warmup_iters=int(exp_cfg["warmup_iters"]),
                measure_iters=int(exp_cfg["measure_iters"]),
                verify=bool(exp_cfg["verify"]),
                timeout_sec=int(exp_cfg["trial_timeout_sec"]),
                risk_level=risk_level,
            )
            trials.append(row)
        trials_by_shape[shape_name] = trials

    selection = build_selection(
        trials_by_shape=trials_by_shape,
        shapes=shapes,
        buckets_cfg=buckets_cfg,
        risk_penalties=scoring_cfg["risk_penalties"],
        invalid_penalty=float(scoring_cfg["invalid_penalty"]),
    )

    run_dir = runs_dir / now_ts()
    ensure_dir(run_dir)
    all_trials = [t for vals in trials_by_shape.values() for t in vals]
    write_jsonl(run_dir / "trials.jsonl", all_trials)
    best_config = {
        "best_by_shape": selection.get("best_by_shape", {}),
        "best_by_bucket": selection.get("best_by_bucket", {}),
    }
    dump_json(run_dir / "best_config.json", best_config)
    report_path = generate_report(
        run_dir=run_dir,
        pattern_scores=pattern_scores,
        selection=selection,
        trials_by_shape=trials_by_shape,
    )
    summary = {
        "run_dir": str(run_dir),
        "report": str(report_path),
        "best_config": str(run_dir / "best_config.json"),
        "trials": str(run_dir / "trials.jsonl"),
        "local_patterns": str(local_path),
        "cloud_patterns": str(cloud_path),
        "merged_patterns": str(merged_path),
        "local_record_count": local.get("stats", {}).get("record_count", 0),
        "merged_record_count": merged.get("stats", {}).get("record_count", 0),
    }
    return summary
