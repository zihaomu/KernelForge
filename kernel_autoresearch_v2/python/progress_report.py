from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "ts": row.get("ts", ""),
                    "iteration": int(row.get("iteration", "0") or 0),
                    "bucket": row.get("bucket", ""),
                    "decision": row.get("decision", ""),
                    "score": float(row.get("score", "0") or 0.0),
                    "latency_us": float(row.get("avg_latency_us", "0") or 0.0),
                    "gflops": float(row.get("avg_gflops", "0") or 0.0),
                    "reason": row.get("reason", ""),
                    "hypothesis": row.get("hypothesis", ""),
                }
            )
    return rows


def _read_run_log(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or not raw.startswith("[") or "]" not in raw:
            continue
        ts, msg = raw[1:].split("]", 1)
        rows.append({"ts": ts.strip(), "message": msg.strip()})
    return rows


def _group_logs_by_iteration(log_rows: list[dict[str, str]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for row in log_rows:
        m = re.search(r"\biter=(\d+)\b", row["message"])
        if not m:
            continue
        it = int(m.group(1))
        out.setdefault(it, []).append(row["message"])
    return out


def _attach_logs(
    results_rows: list[dict[str, Any]],
    log_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_iter = _group_logs_by_iteration(log_rows)
    out: list[dict[str, Any]] = []
    for row in results_rows:
        enriched = dict(row)
        enriched["log_messages"] = by_iter.get(int(row["iteration"]), [])
        out.append(enriched)
    return out


def _build_score_svg(rows: list[dict[str, Any]], svg_path: Path) -> None:
    ensure_dir(svg_path.parent)
    if not rows:
        svg_path.write_text(
            """<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260">
<text x="20" y="40" font-size="20">No data</text>
</svg>
""",
            encoding="utf-8",
        )
        return

    width, height = 980, 320
    pad_l, pad_r, pad_t, pad_b = 70, 30, 30, 50
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    xs = [r["iteration"] for r in rows]
    ys = [r["score"] for r in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1.0

    def x_map(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def y_map(y: float) -> float:
        return pad_t + (y_max - y) / (y_max - y_min) * plot_h

    poly = " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(xs, ys))
    circles = []
    for r in rows:
        cx = x_map(r["iteration"])
        cy = y_map(r["score"])
        color = "#2ca02c" if r.get("decision") == "keep" else "#d62728"
        circles.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{color}"/>')

    y_ticks = []
    for i in range(6):
        t = y_min + (y_max - y_min) * (i / 5.0)
        y = y_map(t)
        y_ticks.append(f'<line x1="{pad_l}" y1="{y:.2f}" x2="{width-pad_r}" y2="{y:.2f}" stroke="#eee"/>')
        y_ticks.append(
            f'<text x="{pad_l-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#666">{t:.2f}</text>'
        )

    x_ticks = []
    tick_count = min(8, len(rows))
    for i in range(tick_count):
        idx = int(round(i * (len(rows) - 1) / max(tick_count - 1, 1)))
        xv = rows[idx]["iteration"]
        x = x_map(xv)
        x_ticks.append(f'<line x1="{x:.2f}" y1="{pad_t}" x2="{x:.2f}" y2="{height-pad_b}" stroke="#f5f5f5"/>')
        x_ticks.append(
            f'<text x="{x:.2f}" y="{height-pad_b+20}" text-anchor="middle" font-size="12" fill="#666">{xv}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
<text x="{pad_l}" y="20" font-size="16" fill="#222">Autoresearch Score vs Iteration</text>
<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#444"/>
<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#444"/>
{''.join(y_ticks)}
{''.join(x_ticks)}
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{poly}"/>
{''.join(circles)}
<text x="{width-240}" y="24" font-size="12" fill="#2ca02c">green: keep</text>
<text x="{width-130}" y="24" font-size="12" fill="#d62728">red: revert</text>
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def _write_timeline_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    header = "ts\titeration\tbucket\tdecision\tscore\tlatency_us\tgflops\treason\thypothesis\tlog_messages\n"
    body = [header]
    for r in rows:
        logs = " | ".join(r.get("log_messages", []))
        body.append(
            f"{r['ts']}\t{r['iteration']}\t{r['bucket']}\t{r['decision']}\t"
            f"{r['score']:.6f}\t{r['latency_us']:.3f}\t{r['gflops']:.6f}\t"
            f"{r['reason']}\t{r['hypothesis']}\t{logs}\n"
        )
    out_path.write_text("".join(body), encoding="utf-8")


def _build_html(rows: list[dict[str, Any]], score_svg_rel: str) -> str:
    iters = [r["iteration"] for r in rows]
    scores = [r["score"] for r in rows]
    lats = [r["latency_us"] for r in rows]
    gflops = [r["gflops"] for r in rows]
    keeps = sum(1 for r in rows if r["decision"] == "keep")
    reverts = sum(1 for r in rows if r["decision"] == "revert")

    table_rows = []
    for r in rows:
        logs = "<br/>".join(html.escape(x) for x in r.get("log_messages", []))
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(r['ts'])}</td>"
            f"<td>{r['iteration']}</td>"
            f"<td>{html.escape(r['bucket'])}</td>"
            f"<td>{html.escape(r['decision'])}</td>"
            f"<td>{r['score']:.4f}</td>"
            f"<td>{r['latency_us']:.3f}</td>"
            f"<td>{r['gflops']:.6f}</td>"
            f"<td>{html.escape(r['reason'])}</td>"
            f"<td>{html.escape(r['hypothesis'])}</td>"
            f"<td>{logs}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Autoresearch v2 Progress</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ margin-bottom: 14px; color: #555; }}
    .cards {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; min-width: 140px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f7f7f7; }}
  </style>
</head>
<body>
  <h1>Autoresearch v2 Dashboard</h1>
  <div class="meta">Per-step decision, metrics, and log mapping for CPU GEMM optimization.</div>
  <div class="cards">
    <div class="card"><div>Total Iters</div><b>{len(rows)}</b></div>
    <div class="card"><div>Keep</div><b>{keeps}</b></div>
    <div class="card"><div>Revert</div><b>{reverts}</b></div>
    <div class="card"><div>Best Score</div><b>{max(scores) if scores else 0:.4f}</b></div>
  </div>
  <div class="grid">
    <div class="panel">
      <h3>Static Curve</h3>
      <img src="{html.escape(score_svg_rel)}" style="max-width:100%; border:1px solid #eee"/>
    </div>
    <div class="panel">
      <h3>Interactive Curves</h3>
      <canvas id="chart_score" height="120"></canvas>
      <canvas id="chart_perf" height="120"></canvas>
    </div>
    <div class="panel">
      <h3>Step Timeline</h3>
      <table>
        <thead>
          <tr>
            <th>ts</th><th>iter</th><th>bucket</th><th>decision</th><th>score</th>
            <th>latency_us</th><th>gflops</th><th>reason</th><th>hypothesis</th><th>logs</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>
  </div>
  <script>
    const iters = {iters};
    const scores = {scores};
    const lats = {lats};
    const gflops = {gflops};
    new Chart(document.getElementById('chart_score'), {{
      type: 'line',
      data: {{
        labels: iters,
        datasets: [
          {{label: 'Score', data: scores, borderColor: '#1f77b4'}}
        ]
      }},
      options: {{responsive: true, interaction: {{mode: 'index', intersect: false}}}}
    }});
    new Chart(document.getElementById('chart_perf'), {{
      type: 'line',
      data: {{
        labels: iters,
        datasets: [
          {{label: 'Latency (us)', data: lats, borderColor: '#d62728'}},
          {{label: 'GFLOPS', data: gflops, borderColor: '#2ca02c'}}
        ]
      }},
      options: {{responsive: true, interaction: {{mode: 'index', intersect: false}}}}
    }});
  </script>
</body>
</html>
"""


def generate_progress_report(*, results_tsv: Path, run_log: Path, out_dir: Path) -> dict[str, str]:
    ensure_dir(out_dir)
    rows = _read_results(results_tsv)
    run_log_rows = _read_run_log(run_log)
    rows = _attach_logs(rows, run_log_rows)
    keeps = [r for r in rows if r["decision"] == "keep"]
    best = max(rows, key=lambda x: x["score"]) if rows else None

    summary = {
        "total_iterations": len(rows),
        "keep_count": len(keeps),
        "revert_count": sum(1 for r in rows if r["decision"] == "revert"),
        "best": best or {},
        "run_log": str(run_log),
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    score_svg = out_dir / "score_curve.svg"
    _build_score_svg(rows, score_svg)

    timeline_tsv = out_dir / "timeline.tsv"
    _write_timeline_tsv(rows, timeline_tsv)

    html_path = out_dir / "index.html"
    html_path.write_text(_build_html(rows, "score_curve.svg"), encoding="utf-8")

    md_lines = []
    md_lines.append("# Autoresearch Progress Summary")
    md_lines.append("")
    md_lines.append(f"- total_iterations: {summary['total_iterations']}")
    md_lines.append(f"- keep_count: {summary['keep_count']}")
    md_lines.append(f"- revert_count: {summary['revert_count']}")
    if best:
        md_lines.append(f"- best_iteration: {best['iteration']}")
        md_lines.append(f"- best_score: {best['score']:.6f}")
        md_lines.append(f"- best_bucket: {best['bucket']}")
    md_lines.append("")
    md_lines.append("## Last 10 Iterations")
    md_lines.append("")
    md_lines.append("| iter | bucket | decision | score | latency_us | gflops | reason |")
    md_lines.append("|---:|---|---|---:|---:|---:|---|")
    for r in rows[-10:]:
        md_lines.append(
            f"| {r['iteration']} | {r['bucket']} | {r['decision']} | {r['score']:.6f} | "
            f"{r['latency_us']:.3f} | {r['gflops']:.6f} | {r['reason']} |"
        )
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "index_html": str(html_path),
        "score_svg": str(score_svg),
        "timeline_tsv": str(timeline_tsv),
    }
