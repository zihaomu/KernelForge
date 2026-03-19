from __future__ import annotations

import csv
import datetime as dt
import html
import re
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def _parse_iso_ts(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.strip())


def read_results_tsv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            rows.append(
                {
                    "ts": row.get("ts", ""),
                    "iteration": int(row.get("iteration", "0") or 0),
                    "bucket": row.get("bucket", ""),
                    "candidate_signature": row.get("candidate_signature", ""),
                    "correctness_pass": row.get("correctness_pass", "0") == "1",
                    "avg_latency_ms": float(row.get("avg_latency_ms", "0") or 0.0),
                    "avg_gflops": float(row.get("avg_gflops", "0") or 0.0),
                    "score": float(row.get("score", "0") or 0.0),
                    "best_score_before": float(row.get("best_score_before", "0") or 0.0),
                    "best_score_after": float(row.get("best_score_after", "0") or 0.0),
                    "decision": row.get("decision", ""),
                    "reason": row.get("reason", ""),
                    "proposal_source": row.get("proposal_source", ""),
                    "proposal_note": row.get("proposal_note", ""),
                }
            )
    return rows


def read_run_log(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or not raw.startswith("[") or "]" not in raw:
            continue
        ts, msg = raw[1:].split("]", 1)
        out.append({"ts": ts.strip(), "message": msg.strip()})
    return out


def _group_logs_by_ts(log_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for r in log_rows:
        grouped.setdefault(r["ts"], []).append(r["message"])
    return grouped


def _group_logs_by_iteration(log_rows: list[dict[str, str]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for r in log_rows:
        m = re.search(r"\biter=(\d+)\b", r["message"])
        if not m:
            continue
        it = int(m.group(1))
        out.setdefault(it, []).append(r["message"])
    return out


def _attach_logs(results_rows: list[dict[str, Any]], log_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = _group_logs_by_ts(log_rows)
    by_iter = _group_logs_by_iteration(log_rows)
    out = []
    for r in results_rows:
        row = dict(r)
        iter_logs = by_iter.get(int(r["iteration"]), [])
        if iter_logs:
            row["log_messages"] = iter_logs
        else:
            row["log_messages"] = grouped.get(r["ts"], [])
        out.append(row)
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

    w, h = 980, 320
    pad_l, pad_r, pad_t, pad_b = 70, 30, 30, 50
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
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

    pts = " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(xs, ys))
    circles = []
    for r in rows:
        cx, cy = x_map(r["iteration"]), y_map(r["score"])
        color = "#2ca02c" if r.get("decision") == "keep" else "#d62728"
        circles.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{color}"/>')

    y_ticks = []
    for i in range(6):
        t = y_min + (y_max - y_min) * (i / 5.0)
        y = y_map(t)
        y_ticks.append(f'<line x1="{pad_l}" y1="{y:.2f}" x2="{w-pad_r}" y2="{y:.2f}" stroke="#eee"/>')
        y_ticks.append(
            f'<text x="{pad_l-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#666">{t:.2f}</text>'
        )
    x_ticks = []
    tick_count = min(8, len(rows))
    for i in range(tick_count):
        idx = int(round(i * (len(rows) - 1) / max(tick_count - 1, 1)))
        xv = rows[idx]["iteration"]
        x = x_map(xv)
        x_ticks.append(f'<line x1="{x:.2f}" y1="{pad_t}" x2="{x:.2f}" y2="{h-pad_b}" stroke="#f5f5f5"/>')
        x_ticks.append(
            f'<text x="{x:.2f}" y="{h-pad_b+20}" text-anchor="middle" font-size="12" fill="#666">{xv}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>
<text x="{pad_l}" y="20" font-size="16" fill="#222">Autoresearch Score vs Iteration</text>
<line x1="{pad_l}" y1="{h-pad_b}" x2="{w-pad_r}" y2="{h-pad_b}" stroke="#444"/>
<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="#444"/>
{''.join(y_ticks)}
{''.join(x_ticks)}
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{pts}"/>
{''.join(circles)}
<text x="{w-240}" y="24" font-size="12" fill="#2ca02c">green: keep</text>
<text x="{w-130}" y="24" font-size="12" fill="#d62728">red: revert</text>
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def _build_html(rows: list[dict[str, Any]], score_svg_rel: str) -> str:
    iters = [r["iteration"] for r in rows]
    scores = [r["score"] for r in rows]
    lat = [r["avg_latency_ms"] for r in rows]
    gflops = [r["avg_gflops"] for r in rows]
    decisions = [r["decision"] for r in rows]
    keeps = sum(1 for d in decisions if d == "keep")
    reverts = sum(1 for d in decisions if d == "revert")

    table_rows = []
    for r in rows:
        logs = "<br/>".join(html.escape(x) for x in r.get("log_messages", []))
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(r['ts'])}</td>"
            f"<td>{r['iteration']}</td>"
            f"<td>{html.escape(r['bucket'])}</td>"
            f"<td>{html.escape(r.get('proposal_source',''))}</td>"
            f"<td>{html.escape(r['decision'])}</td>"
            f"<td>{r['score']:.4f}</td>"
            f"<td>{r['avg_latency_ms']:.4f}</td>"
            f"<td>{r['avg_gflops']:.4f}</td>"
            f"<td>{html.escape(r['reason'])}</td>"
            f"<td>{logs}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Autoresearch Progress</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ margin-bottom: 14px; color: #555; }}
    .cards {{ display: flex; gap: 12px; margin-bottom: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px; min-width: 140px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f7f7f7; }}
  </style>
</head>
<body>
  <h1>Autoresearch Progress Dashboard</h1>
  <div class="meta">Generated from <code>results.tsv</code> and <code>run.log</code> with timestamp mapping.</div>
  <div class="cards">
    <div class="card"><div>Total Iters</div><b>{len(rows)}</b></div>
    <div class="card"><div>Keep</div><b>{keeps}</b></div>
    <div class="card"><div>Revert</div><b>{reverts}</b></div>
    <div class="card"><div>Best Score</div><b>{max(scores) if scores else 0:.4f}</b></div>
  </div>
  <div class="grid">
    <div class="panel">
      <h3>Static Image (Score Curve)</h3>
      <img src="{html.escape(score_svg_rel)}" style="max-width:100%; border:1px solid #eee"/>
    </div>
    <div class="panel">
      <h3>Interactive Curves</h3>
      <canvas id="chart1" height="120"></canvas>
      <canvas id="chart2" height="120"></canvas>
    </div>
    <div class="panel">
      <h3>Timestamp-Log Mapping</h3>
      <table>
        <thead>
          <tr>
            <th>ts</th><th>iter</th><th>bucket</th><th>proposal</th><th>decision</th><th>score</th>
            <th>lat(ms)</th><th>gflops</th><th>reason</th><th>log messages</th>
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
    const lat = {lat};
    const gflops = {gflops};
    new Chart(document.getElementById('chart1'), {{
      type: 'line',
      data: {{
        labels: iters,
        datasets: [
          {{label: 'Score', data: scores, borderColor: '#1f77b4', yAxisID: 'y'}},
        ]
      }},
      options: {{responsive: true, interaction: {{mode: 'index', intersect: false}}}}
    }});
    new Chart(document.getElementById('chart2'), {{
      type: 'line',
      data: {{
        labels: iters,
        datasets: [
          {{label: 'Latency (ms)', data: lat, borderColor: '#d62728', yAxisID: 'y'}},
          {{label: 'GFLOPS', data: gflops, borderColor: '#2ca02c', yAxisID: 'y1'}}
        ]
      }},
      options: {{
        responsive: true,
        scales: {{
          y: {{type: 'linear', position: 'left'}},
          y1: {{type: 'linear', position: 'right', grid: {{drawOnChartArea: false}}}}
        }}
      }}
    }});
  </script>
</body>
</html>
"""


def _build_timeline_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "ts",
                "iteration",
                "bucket",
                "proposal_source",
                "decision",
                "score",
                "avg_latency_ms",
                "avg_gflops",
                "reason",
                "log_messages",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["ts"],
                    r["iteration"],
                    r["bucket"],
                    r.get("proposal_source", ""),
                    r["decision"],
                    f"{r['score']:.6f}",
                    f"{r['avg_latency_ms']:.6f}",
                    f"{r['avg_gflops']:.6f}",
                    r["reason"],
                    " | ".join(r.get("log_messages", [])),
                ]
            )


def generate_progress_report(
    *,
    results_tsv: Path,
    run_log: Path,
    out_dir: Path,
) -> dict[str, str]:
    ensure_dir(out_dir)
    rows = read_results_tsv(results_tsv)
    log_rows = read_run_log(run_log)
    combined = _attach_logs(rows, log_rows)
    combined.sort(key=lambda x: (x["iteration"], _parse_iso_ts(x["ts"])))

    score_svg = out_dir / "score_curve.svg"
    html_path = out_dir / "index.html"
    timeline_tsv = out_dir / "timeline.tsv"
    _build_score_svg(combined, score_svg)
    html_path.write_text(_build_html(combined, "score_curve.svg"), encoding="utf-8")
    _build_timeline_tsv(combined, timeline_tsv)
    return {
        "html": str(html_path),
        "svg": str(score_svg),
        "timeline_tsv": str(timeline_tsv),
        "rows": str(len(combined)),
    }
