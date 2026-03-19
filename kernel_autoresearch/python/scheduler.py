from __future__ import annotations

from typing import Any

from .scorer import balance_score


def _shape_volume(shape: dict[str, int]) -> int:
    return int(shape["m"]) * int(shape["n"]) * int(shape["k"])


def bucket_for_shape(shape: dict[str, int], buckets_cfg: dict[str, Any]) -> str:
    vol = _shape_volume(shape)
    for name, cfg in buckets_cfg.items():
        max_volume = cfg.get("max_volume")
        if max_volume is None:
            return name
        if vol <= int(max_volume):
            return name
    return "large"


def _candidate_signature(trial: dict[str, Any]) -> str:
    keys = (
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
    return "|".join(str(trial.get(k)) for k in keys)


def rank_trials(
    trials: list[dict[str, Any]],
    shape: dict[str, int],
    buckets_cfg: dict[str, Any],
    risk_penalties: dict[str, float],
    invalid_penalty: float,
) -> list[dict[str, Any]]:
    bucket = bucket_for_shape(shape, buckets_cfg)
    alpha = float(buckets_cfg[bucket]["alpha_throughput"])

    valid_trials = [t for t in trials if t.get("valid", False)]
    if valid_trials:
        best_latency = min(float(t["latency_ms_p50"]) for t in valid_trials)
        best_gflops = max(float(t["gflops"]) for t in valid_trials)
    else:
        best_latency = 1e9
        best_gflops = 1e-9

    ranked = []
    for trial in trials:
        risk_level = str(trial.get("risk_level", "low"))
        risk_penalty = float(risk_penalties.get(risk_level, 0.0))
        score = balance_score(
            latency_ms_p50=float(trial.get("latency_ms_p50", 0.0)),
            gflops=float(trial.get("gflops", 0.0)),
            best_latency_ms=best_latency,
            best_gflops=best_gflops,
            alpha_throughput=alpha,
            risk_penalty=risk_penalty,
            invalid_penalty=invalid_penalty,
            valid=bool(trial.get("valid", False)),
        )
        tr = dict(trial)
        tr["shape_bucket"] = bucket
        tr["alpha_throughput"] = alpha
        tr["score_balance"] = score
        tr["candidate_signature"] = _candidate_signature(trial)
        ranked.append(tr)
    ranked.sort(key=lambda x: x["score_balance"], reverse=True)
    return ranked


def build_selection(
    trials_by_shape: dict[str, list[dict[str, Any]]],
    shapes: list[dict[str, int]],
    buckets_cfg: dict[str, Any],
    risk_penalties: dict[str, float],
    invalid_penalty: float,
) -> dict[str, Any]:
    shape_map = {s["name"]: s for s in shapes}
    best_by_shape: dict[str, dict[str, Any]] = {}
    ranked_by_shape: dict[str, list[dict[str, Any]]] = {}
    bucket_candidates: dict[str, dict[str, list[float]]] = {}

    for shape_name, trials in trials_by_shape.items():
        shape = shape_map[shape_name]
        ranked = rank_trials(trials, shape, buckets_cfg, risk_penalties, invalid_penalty)
        ranked_by_shape[shape_name] = ranked
        if ranked:
            best = ranked[0]
            best_by_shape[shape_name] = best
            bucket = best["shape_bucket"]
            sig = best["candidate_signature"]
            bucket_candidates.setdefault(bucket, {}).setdefault(sig, []).append(best["score_balance"])

    best_by_bucket = {}
    for bucket, sig_scores in bucket_candidates.items():
        best_sig = max(sig_scores.items(), key=lambda x: sum(x[1]) / max(len(x[1]), 1))[0]
        best_score = sum(sig_scores[best_sig]) / max(len(sig_scores[best_sig]), 1)
        best_by_bucket[bucket] = {"candidate_signature": best_sig, "avg_score": best_score}

    return {
        "best_by_shape": best_by_shape,
        "best_by_bucket": best_by_bucket,
        "ranked_by_shape": ranked_by_shape,
    }
