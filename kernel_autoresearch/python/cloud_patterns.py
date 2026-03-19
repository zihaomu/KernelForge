from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .patterns import normalize_patterns
from .utils import dump_json, load_yaml


def _http_get(url: str, timeout_sec: int, user_agent: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        content = resp.read()
    return content.decode("utf-8", errors="ignore")


def _duckduckgo_search(query: str, timeout_sec: int, max_results: int, user_agent: str) -> list[str]:
    q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={q}"
    raw = _http_get(url, timeout_sec=timeout_sec, user_agent=user_agent)
    candidates = re.findall(r'href="([^"]+)"', raw)
    out: list[str] = []
    seen = set()
    for c in candidates:
        if "uddg=" in c:
            parsed = urllib.parse.urlparse(c)
            params = urllib.parse.parse_qs(parsed.query)
            url_values = params.get("uddg")
            if not url_values:
                continue
            target = urllib.parse.unquote(url_values[0])
        else:
            target = c
        if not target.startswith("http"):
            continue
        low = target.lower()
        if "duckduckgo.com" in low:
            continue
        if low.startswith("javascript:"):
            continue
        if target in seen:
            continue
        seen.add(target)
        out.append(target)
        if len(out) >= max_results:
            break
    return out


def _extract_text_from_html(raw_html: str) -> str:
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_title(raw_html: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html)
    if not m:
        return ""
    return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip())


def extract_cloud_patterns(
    cloud_sources_config: Path,
    output_path: Path,
    enabled: bool,
    timeout_sec: int,
    max_results_per_query: int,
    user_agent: str,
) -> dict[str, Any]:
    if not enabled:
        result = {"enabled": False, "records": [], "errors": []}
        dump_json(output_path, result)
        return result

    cfg = load_yaml(cloud_sources_config)
    queries = [str(x) for x in cfg.get("queries", [])]
    seed_urls = [str(x) for x in cfg.get("seed_urls", [])]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    urls: list[tuple[str, str]] = []

    for q in queries:
        try:
            found = _duckduckgo_search(q, timeout_sec, max_results_per_query, user_agent)
            urls.extend((q, u) for u in found)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"search failed for query={q!r}: {exc}")
    urls.extend(("seed", u) for u in seed_urls)

    dedup = set()
    for query, url in urls:
        if url in dedup:
            continue
        dedup.add(url)
        try:
            raw = _http_get(url, timeout_sec=timeout_sec, user_agent=user_agent)
            text = _extract_text_from_html(raw)
            title = _extract_title(raw)
            patterns = normalize_patterns([text[:3000], title, query])
            if not patterns:
                continue
            confidence = min(0.9, 0.3 + 0.1 * len(patterns))
            records.append(
                {
                    "source_type": "cloud_page",
                    "query": query,
                    "url": url,
                    "title": title,
                    "patterns": patterns,
                    "risk_level": "medium",
                    "confidence": confidence,
                    "snippet": text[:400],
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"fetch failed for url={url!r}: {exc}")

    result = {
        "enabled": True,
        "records": records,
        "errors": errors,
        "stats": {
            "query_count": len(queries),
            "seed_url_count": len(seed_urls),
            "record_count": len(records),
            "error_count": len(errors),
        },
    }
    dump_json(output_path, result)
    return result
