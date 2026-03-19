from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import load_run_status, run_portfolio
from .utils import repo_root_from_file


def _default_global_config(repo_root: Path) -> Path:
    return repo_root / "autoresearch/configs/global.yaml"


def main() -> None:
    repo_root = repo_root_from_file(__file__)
    parser = argparse.ArgumentParser(description="Unified all-ops autoresearch CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=_default_global_config(repo_root),
        help="global config path",
    )
    run_parser.add_argument(
        "--portfolio",
        type=Path,
        default=None,
        help="portfolio config path",
    )

    status_parser = sub.add_parser("status")
    status_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run directory (autoresearch/workspace/runs/<run_id>)",
    )

    args = parser.parse_args()
    if args.command == "run":
        cfg = args.config if args.config.is_absolute() else (repo_root / args.config)
        portfolio = None
        if args.portfolio is not None:
            portfolio = args.portfolio if args.portfolio.is_absolute() else (repo_root / args.portfolio)
        result = run_portfolio(repo_root=repo_root, global_config_path=cfg, portfolio_path=portfolio)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "status":
        run_dir = args.run_dir if args.run_dir.is_absolute() else (repo_root / args.run_dir)
        result = load_run_status(run_dir=run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()

