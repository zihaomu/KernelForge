from __future__ import annotations


def balance_score(
    *,
    latency_ms_p50: float,
    gflops: float,
    best_latency_ms: float,
    best_gflops: float,
    alpha_throughput: float,
    risk_penalty: float,
    invalid_penalty: float,
    valid: bool,
) -> float:
    if not valid:
        return invalid_penalty
    if latency_ms_p50 <= 0 or gflops <= 0:
        return invalid_penalty
    norm_tp = gflops / max(best_gflops, 1e-9)
    norm_lat = best_latency_ms / max(latency_ms_p50, 1e-9)
    score = alpha_throughput * norm_tp + (1.0 - alpha_throughput) * norm_lat
    return score - risk_penalty

