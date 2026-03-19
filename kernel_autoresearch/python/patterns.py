from __future__ import annotations

from typing import Iterable


CANONICAL_PATTERN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tiling": ("tile", "tiling", "blocked", "block"),
    "pack": ("pack", "packing"),
    "vectorize": ("vector", "vectorize", "simd", "neon", "avx", "sve"),
    "threading": ("thread", "openmp", "parallel"),
    "cache_blocking": ("cache", "reuse", "blocking"),
    "jit": ("jit", "codegen"),
    "reorder": ("reorder", "layout", "transpose"),
    "microkernel": ("microkernel", "ukernel"),
    "fuse": ("fuse", "fusion", "fused"),
    "unroll": ("unroll",),
}


def normalize_patterns(tokens: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen = set()
    for token in tokens:
        lt = token.lower()
        hit = None
        for canonical, keywords in CANONICAL_PATTERN_KEYWORDS.items():
            if any(k in lt for k in keywords):
                hit = canonical
                break
        if hit and hit not in seen:
            seen.add(hit)
            result.append(hit)
    return result


def infer_risk_level(risk_note: str | None) -> str:
    text = (risk_note or "").lower()
    if any(k in text for k in ("高风险", "high risk", "平台绑定", "numerical risk", "精度")):
        return "high"
    if any(k in text for k in ("待复核", "uncertain", "medium", "中风险")):
        return "medium"
    return "low"

