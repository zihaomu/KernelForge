from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import dump_json, load_json


SOURCE_WEIGHT = {
    "snippet": 1.0,
    "research_pack": 0.7,
    "cloud_page": 0.6,
}

RISK_WEIGHT = {
    "low": 1.0,
    "medium": 0.85,
    "high": 0.65,
}


def merge_patterns(local_path: Path, cloud_path: Path, output_path: Path) -> dict[str, Any]:
    local = load_json(local_path) if local_path.exists() else {"records": []}
    cloud = load_json(cloud_path) if cloud_path.exists() else {"records": []}

    records = []
    records.extend(local.get("records", []))
    records.extend(cloud.get("records", []))

    pattern_scores: dict[str, float] = {}
    by_source: dict[str, int] = {}
    for r in records:
        source_type = str(r.get("source_type", "snippet"))
        risk = str(r.get("risk_level", "low"))
        conf = float(r.get("confidence", 0.5))
        score = SOURCE_WEIGHT.get(source_type, 0.5) * RISK_WEIGHT.get(risk, 0.8) * conf
        for p in r.get("patterns", []):
            pattern_scores[p] = pattern_scores.get(p, 0.0) + score
        by_source[source_type] = by_source.get(source_type, 0) + 1

    ranked_patterns = sorted(pattern_scores.items(), key=lambda x: -x[1])
    result = {
        "records": records,
        "pattern_scores": {k: round(v, 6) for k, v in ranked_patterns},
        "stats": {
            "record_count": len(records),
            "by_source": by_source,
        },
    }
    dump_json(output_path, result)
    return result

