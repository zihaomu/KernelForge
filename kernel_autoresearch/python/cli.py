from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cloud_patterns import extract_cloud_patterns
from .harness_guard import build_manifest
from .local_patterns import extract_local_patterns
from .orchestration_loop import run_orchestration
from .orchestrator import run_autoresearch
from .pattern_merge import merge_patterns
from .progress_report import generate_progress_report
from .utils import load_yaml, repo_root_from_file


def _default_config(repo_root: Path) -> Path:
    return repo_root / "kernel_autoresearch/configs/default.yaml"


def cmd_extract_local(config_path: Path, repo_root: Path) -> None:
    cfg = load_yaml(config_path)
    paths_cfg = cfg["paths"]
    out = repo_root / paths_cfg["pattern_db_dir"] / "local_patterns.json"
    result = extract_local_patterns(
        repo_root=repo_root,
        code_base_agent_gen_dir=repo_root / paths_cfg["code_base_agent_gen_dir"],
        output_path=out,
    )
    print(json.dumps({"output": str(out), "records": result["stats"]["record_count"]}, ensure_ascii=False))


def cmd_extract_cloud(config_path: Path, repo_root: Path) -> None:
    cfg = load_yaml(config_path)
    paths_cfg = cfg["paths"]
    search_cfg = cfg["search"]
    out = repo_root / paths_cfg["pattern_db_dir"] / "cloud_patterns.json"
    result = extract_cloud_patterns(
        cloud_sources_config=repo_root / paths_cfg["cloud_sources_config"],
        output_path=out,
        enabled=bool(search_cfg.get("enabled", True)),
        timeout_sec=int(search_cfg.get("timeout_sec", 10)),
        max_results_per_query=int(search_cfg.get("max_results_per_query", 5)),
        user_agent=str(search_cfg.get("user_agent", "kernel-autoresearch/0.1")),
    )
    print(
        json.dumps(
            {"output": str(out), "records": result["stats"]["record_count"], "errors": result["stats"]["error_count"]},
            ensure_ascii=False,
        )
    )


def cmd_merge(config_path: Path, repo_root: Path) -> None:
    cfg = load_yaml(config_path)
    pattern_dir = repo_root / cfg["paths"]["pattern_db_dir"]
    result = merge_patterns(
        local_path=pattern_dir / "local_patterns.json",
        cloud_path=pattern_dir / "cloud_patterns.json",
        output_path=pattern_dir / "merged_patterns.json",
    )
    print(
        json.dumps(
            {"output": str(pattern_dir / "merged_patterns.json"), "records": result["stats"]["record_count"]},
            ensure_ascii=False,
        )
    )


def cmd_run(config_path: Path, repo_root: Path) -> None:
    result = run_autoresearch(config_path=config_path, repo_root=repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_orchestrate(
    config_path: Path,
    repo_root: Path,
    *,
    agent_mode: str | None = None,
    agent_model: str | None = None,
) -> None:
    result = run_orchestration(
        config_path=config_path,
        repo_root=repo_root,
        agent_mode_override=agent_mode,
        agent_model_override=agent_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_refresh_harness_manifest(config_path: Path, repo_root: Path) -> None:
    cfg = load_yaml(config_path)
    auto_cfg = cfg["autoresearch"]
    manifest_path = repo_root / auto_cfg["harness_manifest"]
    harness_files = [repo_root / p for p in auto_cfg.get("harness_files", [])]
    manifest = build_manifest(repo_root=repo_root, harness_files=harness_files, manifest_path=manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "entry_count": len(manifest.get("entries", [])),
            },
            ensure_ascii=False,
        )
    )


def cmd_progress_report(config_path: Path, repo_root: Path) -> None:
    cfg = load_yaml(config_path)
    ws_dir = repo_root / cfg["autoresearch"]["workspace_dir"]
    results_tsv = ws_dir / cfg["autoresearch"]["results_tsv"]
    run_log = ws_dir / cfg["autoresearch"]["run_log"]
    out_dir = ws_dir / "progress"
    result = generate_progress_report(results_tsv=results_tsv, run_log=run_log, out_dir=out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    repo_root = repo_root_from_file(__file__)
    parser = argparse.ArgumentParser(description="CPU GEMM autoresearch MVP")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config(repo_root),
        help="path to default.yaml",
    )
    parser.add_argument(
        "--agent-mode",
        type=str,
        default=None,
        help="override autoresearch.agent.mode (rules_only|hybrid|agent_only)",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help="override autoresearch.agent.model",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract-local")
    sub.add_parser("extract-cloud")
    sub.add_parser("merge")
    sub.add_parser("run")
    sub.add_parser("orchestrate")
    sub.add_parser("refresh-harness-manifest")
    sub.add_parser("progress-report")

    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else (repo_root / args.config)

    if args.command == "extract-local":
        cmd_extract_local(config_path, repo_root)
    elif args.command == "extract-cloud":
        cmd_extract_cloud(config_path, repo_root)
    elif args.command == "merge":
        cmd_merge(config_path, repo_root)
    elif args.command == "run":
        cmd_run(config_path, repo_root)
    elif args.command == "orchestrate":
        cmd_orchestrate(
            config_path,
            repo_root,
            agent_mode=args.agent_mode,
            agent_model=args.agent_model,
        )
    elif args.command == "refresh-harness-manifest":
        cmd_refresh_harness_manifest(config_path, repo_root)
    elif args.command == "progress-report":
        cmd_progress_report(config_path, repo_root)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
