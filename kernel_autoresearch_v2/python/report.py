from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import now_iso


def render_final_report(
    *,
    best_by_bucket: dict[str, dict[str, Any]],
    baseline_by_bucket: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    lines = []
    lines.append("# CPU GEMM Autoresearch Final Report")
    lines.append("")
    lines.append(f"- generated_at: {now_iso()}")
    lines.append("")
    lines.append("## Bucket Summary")
    lines.append("")
    lines.append("| bucket | baseline_latency_us | baseline_gflops | best_score | best_candidate |")
    lines.append("|---|---:|---:|---:|---|")
    for bucket in sorted(baseline_by_bucket.keys()):
        base = baseline_by_bucket.get(bucket, {})
        best = best_by_bucket.get(bucket, {})
        cand = best.get("candidate", {})
        cand_sig = (
            f"{cand.get('kernel_variant','naive')}"
            f"/bm{cand.get('block_m',0)}"
            f"/bn{cand.get('block_n',0)}"
            f"/bk{cand.get('block_k',0)}"
            f"/th{cand.get('threads',1)}"
            f"/uk{cand.get('unroll_k',1)}"
            f"/simd{int(bool(cand.get('simd', False)))}"
        )
        lines.append(
            f"| {bucket} | {float(base.get('avg_latency_us', 0.0)):.3f} | {float(base.get('avg_gflops', 0.0)):.6f} | "
            f"{float(best.get('score', 0.0)):.6f} | `{cand_sig}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- correctness gate is always prior to performance gate")
    lines.append("- keep/revert follows min_improve_ratio policy")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

