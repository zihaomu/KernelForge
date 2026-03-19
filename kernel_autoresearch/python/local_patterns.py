from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .patterns import infer_risk_level, normalize_patterns
from .utils import dump_json, load_json, load_yaml, parse_jsonl, resolve_path


def _is_cpu_repo(manifest: dict[str, Any]) -> bool:
    backend = manifest.get("backend") or {}
    tags = [str(x).lower() for x in backend.get("tags", [])]
    if "cpu" in tags:
        return True
    path_backend_map = backend.get("path_backend_map") or {}
    for _, values in path_backend_map.items():
        lv = [str(x).lower() for x in values]
        if "cpu" in lv:
            return True
    return False


def _has_gemm(manifest: dict[str, Any]) -> bool:
    op_map = manifest.get("op_map") or {}
    return "gemm" in op_map


def _read_research_pack_hints(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    hints: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "可迁移建议" not in line:
            continue
        _, right = line.split(":", 1) if ":" in line else ("", line)
        patterns = normalize_patterns([right])
        if not patterns:
            continue
        hints.append(
            {
                "source_type": "research_pack",
                "patterns": patterns,
                "text": right.strip(),
                "risk_level": "medium",
                "confidence": 0.6,
            }
        )
    return hints


def _triage_risk(path: Path) -> str:
    if not path.exists():
        return "low"
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    if re.search(r"高风险|high risk|平台绑定|数值风险", text):
        return "high"
    if re.search(r"待复核|uncertain|不确定", text):
        return "medium"
    return "low"


def _priority_to_score(priority: Any) -> float:
    if priority is None:
        return 0.5
    if isinstance(priority, (int, float)):
        return float(priority)
    sp = str(priority).strip().lower()
    if sp in ("p0", "critical", "high"):
        return 1.0
    if sp in ("p1", "medium"):
        return 0.7
    return 0.4


def extract_local_patterns(
    repo_root: Path,
    code_base_agent_gen_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    summary_path = code_base_agent_gen_dir / "run_summary_kernel_triage.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = load_json(summary_path)
    entries = summary.get("entries", [])

    records: list[dict[str, Any]] = []
    skipped = 0
    for entry in entries:
        manifest_path = resolve_path(repo_root, entry.get("manifest"))
        snippets_path = resolve_path(repo_root, entry.get("snippets_path"))
        research_pack_path = resolve_path(repo_root, entry.get("research_pack"))
        triage_path = resolve_path(repo_root, entry.get("triage"))
        if not manifest_path.exists() or not snippets_path.exists():
            skipped += 1
            continue

        manifest = load_yaml(manifest_path)
        if not _is_cpu_repo(manifest) or not _has_gemm(manifest):
            continue

        triage_risk = _triage_risk(triage_path)
        repo_name = str(entry.get("repo", manifest.get("repo", "unknown")))
        for row in parse_jsonl(snippets_path):
            if str(row.get("op", "")).lower() != "gemm":
                continue
            backends = [str(x).lower() for x in row.get("backend", [])]
            if "cpu" not in backends:
                continue
            patterns = normalize_patterns(row.get("optimization_pattern", []))
            if not patterns:
                continue
            snippet_risk = infer_risk_level(row.get("risk_note"))
            risk = triage_risk if triage_risk == "high" else snippet_risk
            record = {
                "repo": repo_name,
                "source_type": "snippet",
                "source_path": row.get("source_path"),
                "symbol": row.get("symbol"),
                "patterns": patterns,
                "backend": backends,
                "dtype": row.get("dtype", []),
                "risk_level": risk,
                "confidence": _priority_to_score(row.get("priority")),
            }
            records.append(record)

        records.extend(
            {
                "repo": repo_name,
                "source_type": "research_pack",
                "source_path": str(research_pack_path.relative_to(repo_root))
                if research_pack_path.exists()
                else "",
                "symbol": "",
                "patterns": hint["patterns"],
                "backend": ["cpu"],
                "dtype": [],
                "risk_level": hint["risk_level"],
                "confidence": hint["confidence"],
                "hint_text": hint["text"],
            }
            for hint in _read_research_pack_hints(research_pack_path)
        )

    pattern_hist: dict[str, int] = {}
    for record in records:
        for p in record["patterns"]:
            pattern_hist[p] = pattern_hist.get(p, 0) + 1

    result = {
        "source": "code_base_agent_gen",
        "records": records,
        "stats": {
            "record_count": len(records),
            "skipped_entries": skipped,
            "pattern_histogram": dict(sorted(pattern_hist.items(), key=lambda x: -x[1])),
        },
    }
    dump_json(output_path, result)
    return result

