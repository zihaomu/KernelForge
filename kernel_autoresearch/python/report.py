from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_report(
    run_dir: Path,
    pattern_scores: dict[str, float],
    selection: dict[str, Any],
    trials_by_shape: dict[str, list[dict[str, Any]]],
) -> Path:
    report_path = run_dir / "report.md"
    lines: list[str] = []
    lines.append("# GEMM Autoresearch Report")
    lines.append("")
    lines.append("## Top Pattern Signals")
    if pattern_scores:
        for k, v in list(pattern_scores.items())[:10]:
            lines.append(f"- {k}: {v:.4f}")
    else:
        lines.append("- (empty)")

    lines.append("")
    lines.append("## Best Per Shape")
    best_by_shape = selection.get("best_by_shape", {})
    if not best_by_shape:
        lines.append("- no valid best config found")
    for shape_name, best in best_by_shape.items():
        lines.append(
            "- "
            f"{shape_name}: variant={best.get('kernel_variant')} "
            f"bm/bn/bk={best.get('bm')}/{best.get('bn')}/{best.get('bk')} "
            f"threads={best.get('threads')} simd={best.get('simd')} "
            f"gflops={float(best.get('gflops', 0.0)):.3f} "
            f"lat_p50={float(best.get('latency_ms_p50', 0.0)):.3f}ms "
            f"score={float(best.get('score_balance', 0.0)):.4f}"
        )

    lines.append("")
    lines.append("## Bucket Policy")
    best_by_bucket = selection.get("best_by_bucket", {})
    if not best_by_bucket:
        lines.append("- no bucket policy")
    for bucket, row in best_by_bucket.items():
        lines.append(f"- {bucket}: {row.get('candidate_signature')} avg_score={row.get('avg_score'):.4f}")

    lines.append("")
    lines.append("## Trial Stats")
    total_trials = sum(len(v) for v in trials_by_shape.values())
    failed_trials = sum(1 for vals in trials_by_shape.values() for t in vals if not t.get("valid", False))
    lines.append(f"- total_trials: {total_trials}")
    lines.append(f"- failed_trials: {failed_trials}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path

