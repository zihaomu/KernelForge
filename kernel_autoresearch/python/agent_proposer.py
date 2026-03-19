from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from typing import Any

from kernel_autoresearch.harness.bench import candidate_signature


REQUIRED_KEYS = (
    "kernel_variant",
    "bm",
    "bn",
    "bk",
    "pack_a",
    "pack_b",
    "simd",
    "threads",
    "unroll_k",
    "input_dtype",
    "output_dtype",
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_candidate(candidate: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    out = dict(fallback)
    out.update(candidate)
    out["kernel_variant"] = str(out.get("kernel_variant", "blocked"))
    if out["kernel_variant"] not in ("naive", "blocked", "blocked_pack"):
        out["kernel_variant"] = "blocked"
    for k in ("bm", "bn", "bk", "threads", "unroll_k"):
        try:
            out[k] = int(out.get(k, fallback.get(k, 1)))
        except Exception:  # noqa: BLE001
            out[k] = int(fallback.get(k, 1))
    out["bm"] = max(0, out["bm"])
    out["bn"] = max(0, out["bn"])
    out["bk"] = max(0, out["bk"])
    out["threads"] = max(1, out["threads"])
    out["unroll_k"] = max(1, out["unroll_k"])
    out["pack_a"] = bool(out.get("pack_a", fallback.get("pack_a", False)))
    out["pack_b"] = bool(out.get("pack_b", fallback.get("pack_b", False)))
    out["simd"] = bool(out.get("simd", fallback.get("simd", False)))
    input_dtype = str(out.get("input_dtype", fallback.get("input_dtype", "f32"))).lower()
    if input_dtype not in ("f32", "f16", "i8"):
        input_dtype = str(fallback.get("input_dtype", "f32")).lower()
    out["input_dtype"] = input_dtype
    default_output = "i32" if input_dtype == "i8" else ("f16" if input_dtype == "f16" else "f32")
    output_dtype = str(out.get("output_dtype", fallback.get("output_dtype", default_output))).lower()
    if input_dtype == "i8":
        output_dtype = "i32"
    elif input_dtype == "f16":
        output_dtype = "f16"
    elif output_dtype != "f32":
        output_dtype = "f32"
    out["output_dtype"] = output_dtype
    if out["kernel_variant"] == "naive":
        out["bm"] = out["bn"] = out["bk"] = 0
        out["pack_a"] = False
        out["pack_b"] = False
    return out


def _mutate_from_base(base: dict[str, Any], pool: list[dict[str, Any]], rate: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    # First try picking a close candidate from pool to stay valid and diverse.
    if pool and rng.random() < 0.6:
        return dict(rng.choice(pool))

    c = dict(base)
    if rng.random() < rate:
        c["simd"] = not bool(c.get("simd", False))
    if rng.random() < rate:
        c["pack_a"] = not bool(c.get("pack_a", False))
        c["pack_b"] = c["pack_a"]
    if rng.random() < rate:
        c["threads"] = max(1, int(c.get("threads", 1)) * (2 if rng.random() < 0.5 else 1))
    if rng.random() < rate:
        c["unroll_k"] = 1 if int(c.get("unroll_k", 1)) > 1 else 2
    if rng.random() < rate:
        c["kernel_variant"] = rng.choice(["blocked", "blocked_pack"])
    for key in ("bm", "bn", "bk"):
        if rng.random() < rate:
            cur = int(c.get(key, 64))
            delta = rng.choice([-32, -16, 16, 32])
            c[key] = max(16, cur + delta)
    return c


def _openai_propose(
    *,
    agent_cfg: dict[str, Any],
    bucket: str,
    history_tail: list[dict[str, Any]],
    best_candidate: dict[str, Any] | None,
    baseline: dict[str, float] | None,
    pool: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    endpoint = str(agent_cfg.get("endpoint", ""))
    model = str(agent_cfg.get("model", "gpt-5.4"))
    api_env = str(agent_cfg.get("api_key_env", "OPENAI_API_KEY"))
    temperature = float(agent_cfg.get("temperature", 0.2))
    max_tokens = int(agent_cfg.get("max_tokens", 300))
    api_key = os.getenv(api_env, "").strip()
    if not api_key:
        return None, f"missing_api_key_env:{api_env}"
    if not endpoint:
        return None, "missing_endpoint"

    recent = history_tail[-8:] if history_tail else []
    prompt = {
        "bucket": bucket,
        "recent_trials": recent,
        "best_candidate": best_candidate,
        "baseline": baseline,
        "candidate_schema": {
            "kernel_variant": "naive|blocked|blocked_pack",
            "bm": "int",
            "bn": "int",
            "bk": "int",
            "pack_a": "bool",
            "pack_b": "bool",
            "simd": "bool",
            "threads": "int",
            "unroll_k": "int",
            "input_dtype": "f32|f16|i8",
            "output_dtype": "f32|f16|i32",
        },
        "rules": [
            "return only one JSON object",
            "candidate must prioritize improving score on current bucket",
            "if uncertain, propose blocked or blocked_pack",
        ],
    }
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "You are a kernel autotuning agent. Output JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        return None, f"openai_request_failed:{exc}"

    try:
        obj = json.loads(raw)
        content = obj["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content if isinstance(content, str) else json.dumps(content))
        if not parsed:
            return None, "openai_content_no_json"
        return parsed, "openai"
    except Exception as exc:  # noqa: BLE001
        parsed = _extract_json_object(raw)
        if parsed:
            return parsed, "openai_raw_json"
        return None, f"openai_parse_failed:{exc}"


def choose_candidate(
    *,
    mode: str,
    agent_cfg: dict[str, Any],
    bucket: str,
    cursor_candidate: dict[str, Any],
    pool: list[dict[str, Any]],
    seen_signatures: set[str],
    history_tail: list[dict[str, Any]],
    best_candidate: dict[str, Any] | None,
    baseline: dict[str, float] | None,
    iteration_seed: int,
) -> dict[str, Any]:
    """
    Returns:
      {
        "candidate": dict,
        "proposal_source": "rules|agent_openai|agent_heuristic",
        "proposal_note": str,
      }
    """
    mode = mode.strip().lower()
    if mode not in ("rules_only", "hybrid", "agent_only"):
        mode = "rules_only"

    fallback = dict(cursor_candidate)
    if mode == "rules_only":
        return {"candidate": fallback, "proposal_source": "rules", "proposal_note": "rules_only"}

    openai_candidate = None
    openai_note = ""
    if str(agent_cfg.get("provider", "openai")).lower() == "openai":
        openai_candidate, openai_note = _openai_propose(
            agent_cfg=agent_cfg,
            bucket=bucket,
            history_tail=history_tail,
            best_candidate=best_candidate,
            baseline=baseline,
            pool=pool,
        )

    if openai_candidate is not None:
        norm = normalize_candidate(openai_candidate, fallback)
        sig = candidate_signature(norm)
        if sig not in seen_signatures:
            return {"candidate": norm, "proposal_source": "agent_openai", "proposal_note": openai_note or "openai"}

    # Heuristic fallback for hybrid/agent_only.
    rate = float(agent_cfg.get("heuristic_mutation_rate", 0.35))
    heuristic = _mutate_from_base(best_candidate or fallback, pool, rate, seed=iteration_seed)
    norm_h = normalize_candidate(heuristic, fallback)
    sig_h = candidate_signature(norm_h)
    if sig_h not in seen_signatures:
        note = "heuristic_after_openai_fail" if openai_note else "heuristic"
        return {"candidate": norm_h, "proposal_source": "agent_heuristic", "proposal_note": note}

    if mode == "agent_only":
        # Still return fallback to avoid deadlock, but label it.
        return {
            "candidate": fallback,
            "proposal_source": "agent_fallback_rules",
            "proposal_note": openai_note or "agent_only_no_new_candidate",
        }
    return {
        "candidate": fallback,
        "proposal_source": "rules_fallback",
        "proposal_note": openai_note or "hybrid_no_new_candidate",
    }
