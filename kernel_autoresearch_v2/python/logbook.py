from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import dump_json, ensure_dir, load_json, now_iso


RESULTS_HEADER = (
    "ts\titeration\tbucket\tcandidate_signature\tcorrectness_pass\tavg_latency_us\tavg_gflops\t"
    "score\tbest_score_before\tbest_score_after\tdecision\treason\thypothesis\n"
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
    if not run_log.exists():
        run_log.write_text("", encoding="utf-8")
    return {"results_tsv": results_tsv, "run_log": run_log, "state": state_path}


def append_run_log(run_log: Path, message: str) -> None:
    line = f"[{now_iso()}] {message}\n"
    run_log.write_text(run_log.read_text(encoding="utf-8") + line, encoding="utf-8")


def append_results_tsv(
    results_tsv: Path,
    *,
    iteration: int,
    bucket: str,
    candidate_signature: str,
    correctness_pass: bool,
    avg_latency_us: float,
    avg_gflops: float,
    score: float,
    best_score_before: float | None,
    best_score_after: float | None,
    decision: str,
    reason: str,
    hypothesis: str,
) -> None:
    row = (
        f"{now_iso()}\t{iteration}\t{bucket}\t{candidate_signature}\t{int(correctness_pass)}\t"
        f"{avg_latency_us:.3f}\t{avg_gflops:.6f}\t{score:.6f}\t"
        f"{'' if best_score_before is None else f'{best_score_before:.6f}'}\t"
        f"{'' if best_score_after is None else f'{best_score_after:.6f}'}\t"
        f"{decision}\t{reason}\t{hypothesis}\n"
    )
    results_tsv.write_text(results_tsv.read_text(encoding="utf-8") + row, encoding="utf-8")


def load_state_or_default(state_path: Path, default_state: dict[str, Any]) -> dict[str, Any]:
    if not state_path.exists():
        return default_state
    return load_json(state_path)


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    dump_json(state_path, state)

