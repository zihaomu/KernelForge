from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .utils import dump_json, ensure_dir, load_json


RESULTS_HEADER = (
    "ts\titeration\tbucket\tcandidate_signature\tcorrectness_pass\tavg_latency_ms\tavg_gflops\t"
    "score\tbest_score_before\tbest_score_after\tdecision\treason\tproposal_source\tproposal_note\n"
)


def init_logbook(workspace_dir: Path, results_tsv_rel: str, run_log_rel: str, state_rel: str) -> dict[str, Path]:
    ensure_dir(workspace_dir)
    results_tsv = workspace_dir / results_tsv_rel
    run_log = workspace_dir / run_log_rel
    state_path = workspace_dir / state_rel
    ensure_dir(results_tsv.parent)
    ensure_dir(run_log.parent)
    ensure_dir(state_path.parent)
    if not results_tsv.exists():
        results_tsv.write_text(RESULTS_HEADER, encoding="utf-8")
    else:
        raw = results_tsv.read_text(encoding="utf-8")
        first_line = raw.splitlines()[0] + "\n" if raw.splitlines() else ""
        if first_line != RESULTS_HEADER:
            lines = raw.splitlines()
            migrated = [RESULTS_HEADER.rstrip("\n")]
            for line in lines[1:]:
                cols = line.split("\t")
                while len(cols) < 14:
                    cols.append("")
                migrated.append("\t".join(cols[:14]))
            results_tsv.write_text("\n".join(migrated) + "\n", encoding="utf-8")
    if not run_log.exists():
        run_log.write_text("", encoding="utf-8")
    return {"results_tsv": results_tsv, "run_log": run_log, "state": state_path}


def append_run_log(run_log: Path, message: str) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    run_log.write_text(run_log.read_text(encoding="utf-8") + f"[{ts}] {message}\n", encoding="utf-8")


def append_results_tsv(
    results_tsv: Path,
    *,
    iteration: int,
    bucket: str,
    candidate_signature: str,
    correctness_pass: bool,
    avg_latency_ms: float,
    avg_gflops: float,
    score: float,
    best_score_before: float | None,
    best_score_after: float | None,
    decision: str,
    reason: str,
    proposal_source: str = "",
    proposal_note: str = "",
) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    row = (
        f"{ts}\t{iteration}\t{bucket}\t{candidate_signature}\t{int(correctness_pass)}\t"
        f"{avg_latency_ms:.6f}\t{avg_gflops:.6f}\t{score:.6f}\t"
        f"{'' if best_score_before is None else f'{best_score_before:.6f}'}\t"
        f"{'' if best_score_after is None else f'{best_score_after:.6f}'}\t"
        f"{decision}\t{reason}\t{proposal_source}\t{proposal_note}\n"
    )
    results_tsv.write_text(results_tsv.read_text(encoding="utf-8") + row, encoding="utf-8")


def load_state_or_default(state_path: Path, default_state: dict[str, Any]) -> dict[str, Any]:
    if not state_path.exists():
        return default_state
    return load_json(state_path)


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    dump_json(state_path, state)
