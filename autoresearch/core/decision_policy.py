from __future__ import annotations

from typing import Any


def make_decision(
    *,
    best_score: float | None,
    current_score: float,
    correctness_pass: bool,
    min_improve_ratio: float,
) -> dict[str, Any]:
    if not correctness_pass:
        return {
            "decision": "revert",
            "reason": "correctness_failed",
            "new_best_score": best_score,
            "improved": False,
        }

    if best_score is None:
        return {
            "decision": "keep",
            "reason": "first_valid_candidate",
            "new_best_score": current_score,
            "improved": True,
        }

    threshold = abs(best_score) * float(min_improve_ratio)
    delta = current_score - best_score
    if delta > threshold:
        return {
            "decision": "keep",
            "reason": "score_improved",
            "new_best_score": current_score,
            "improved": True,
            "threshold": threshold,
            "delta": delta,
        }
    return {
        "decision": "revert",
        "reason": "improvement_below_threshold",
        "new_best_score": best_score,
        "improved": False,
        "threshold": threshold,
        "delta": delta,
    }

