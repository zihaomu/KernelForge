from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness_guard import build_manifest
from .orchestration_loop import run_once, run_orchestration
from .progress_report import generate_progress_report
from .utils import load_yaml, repo_root_from_file


def _default_config(repo_root: Path) -> Path:
    return repo_root / "kernel_autoresearch_v2/configs/default.yaml"


def cmd_refresh_harness_manifest(config_path: Path, repo_root: Path) -> None:
    cfg = load_yaml(config_path)
    auto_cfg = cfg["autoresearch"]
    manifest_path = repo_root / auto_cfg["harness_manifest"]
    harness_files = [repo_root / p for p in auto_cfg.get("harness_files", [])]
    manifest = build_manifest(repo_root=repo_root, harness_files=harness_files, manifest_path=manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "entry_count": len(manifest.get("entries", []))}, ensure_ascii=False))


def cmd_orchestrate(config_path: Path, repo_root: Path, *, agent_mode: str | None = None) -> None:
    result = run_orchestration(config_path=config_path, repo_root=repo_root, agent_mode_override=agent_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_run_once(config_path: Path, repo_root: Path) -> None:
    result = run_once(config_path=config_path, repo_root=repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


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
    parser = argparse.ArgumentParser(description="CPU GEMM autoresearch v2")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config(repo_root),
        help="path to kernel_autoresearch_v2 default.yaml",
    )
    parser.add_argument(
        "--agent-mode",
        type=str,
        default=None,
        help="override autoresearch.agent.mode (rules_only|hybrid|agent_only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("refresh-harness-manifest")
    sub.add_parser("run-once")
    sub.add_parser("orchestrate")
    sub.add_parser("progress-report")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    if args.command == "refresh-harness-manifest":
        cmd_refresh_harness_manifest(config_path, repo_root)
    elif args.command == "run-once":
        cmd_run_once(config_path, repo_root)
    elif args.command == "orchestrate":
        cmd_orchestrate(config_path, repo_root, agent_mode=args.agent_mode)
    elif args.command == "progress-report":
        cmd_progress_report(config_path, repo_root)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

