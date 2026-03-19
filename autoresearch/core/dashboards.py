from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir


def build_portfolio_dashboard(*, run_dir: Path, task_summaries: list[dict[str, Any]]) -> dict[str, str]:
    ensure_dir(run_dir)
    tsv_path = run_dir / "portfolio.tsv"
    md_path = run_dir / "index.md"

    header = "op_id\tstatus\titerations\tbest_score\tbest_candidate\trun_dir\n"
    rows = [header]
    for s in task_summaries:
        rows.append(
            f"{s.get('op_id','')}\t{s.get('status','')}\t{s.get('iterations',0)}\t"
            f"{float(s.get('best_score',0.0)):.6f}\t{s.get('best_candidate_signature','')}\t"
            f"{s.get('task_run_dir','')}\n"
        )
    tsv_path.write_text("".join(rows), encoding="utf-8")

    md_lines = []
    md_lines.append("# Portfolio Dashboard")
    md_lines.append("")
    md_lines.append("| op_id | status | iterations | best_score | best_candidate |")
    md_lines.append("|---|---|---:|---:|---|")
    for s in task_summaries:
        md_lines.append(
            f"| {s.get('op_id','')} | {s.get('status','')} | {s.get('iterations',0)} | "
            f"{float(s.get('best_score',0.0)):.6f} | `{s.get('best_candidate_signature','')}` |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"portfolio_tsv": str(tsv_path), "index_md": str(md_path)}

