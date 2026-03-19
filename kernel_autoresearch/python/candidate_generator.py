from __future__ import annotations

import itertools
import os
import random
from typing import Any

from .scheduler import bucket_for_shape


def _pick_block_sizes(dim: int, choices: list[int], k: int = 3) -> list[int]:
    ranked = sorted(choices, key=lambda x: (abs(dim - x), -x))
    out = ranked[:k]
    out = sorted(set(out))
    return out if out else [choices[0]]


def _thread_candidates(shape_bucket: str, cfg_choices: list[int]) -> list[int]:
    cpu_cap = max(1, os.cpu_count() or 1)
    choices = sorted(set(c for c in cfg_choices if c <= cpu_cap))
    if not choices:
        choices = [1]
    if shape_bucket == "small":
        return [1, min(choices[-1], 4)]
    if shape_bucket == "medium":
        return [1, choices[len(choices) // 2], choices[-1]]
    return [max(1, choices[len(choices) // 2]), choices[-1]]


def generate_candidates_for_shape(
    shape: dict[str, int],
    shape_bucket: str,
    pattern_scores: dict[str, float],
    candidate_cfg: dict[str, Any],
    max_trials: int,
    seed: int,
    input_dtype: str = "f32",
) -> list[dict[str, Any]]:
    block_m = [int(x) for x in candidate_cfg.get("block_m", [32, 64])]
    block_n = [int(x) for x in candidate_cfg.get("block_n", [32, 64])]
    block_k = [int(x) for x in candidate_cfg.get("block_k", [32, 64])]
    unroll_k = [int(x) for x in candidate_cfg.get("unroll_k", [1, 2])]
    thread_choices = [int(x) for x in candidate_cfg.get("thread_choices", [1, 2, 4])]

    m, n, k = int(shape["m"]), int(shape["n"]), int(shape["k"])
    bms = _pick_block_sizes(m, block_m)
    bns = _pick_block_sizes(n, block_n)
    bks = _pick_block_sizes(k, block_k)
    ths = sorted(set(_thread_candidates(shape_bucket, thread_choices)))

    prefer_pack = pattern_scores.get("pack", 0.0) >= 1.0 and candidate_cfg.get("allow_pack", True)
    prefer_simd = pattern_scores.get("vectorize", 0.0) + pattern_scores.get("tiling", 0.0) >= 1.0
    simd_space = [False, True] if candidate_cfg.get("allow_simd", True) else [False]
    if prefer_simd:
        simd_space = [True, False]

    candidates: list[dict[str, Any]] = []
    if input_dtype == "i8":
        output_dtype = "i32"
    elif input_dtype == "f16":
        output_dtype = "f16"
    else:
        output_dtype = "f32"
    candidates.append(
        {
            "kernel_variant": "naive",
            "bm": 0,
            "bn": 0,
            "bk": 0,
            "pack_a": False,
            "pack_b": False,
            "simd": False,
            "threads": 1,
            "unroll_k": 1,
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
        }
    )

    blocked_variants = ["blocked", "blocked_pack"] if candidate_cfg.get("allow_pack", True) else ["blocked"]
    if not prefer_pack and "blocked_pack" in blocked_variants:
        blocked_variants = ["blocked", "blocked_pack"]
    if prefer_pack and "blocked_pack" in blocked_variants:
        blocked_variants = ["blocked_pack", "blocked"]

    for variant, bm, bn, bk, th, uk, simd in itertools.product(
        blocked_variants, bms, bns, bks, ths, unroll_k, simd_space
    ):
        candidates.append(
            {
                "kernel_variant": variant,
                "bm": bm,
                "bn": bn,
                "bk": bk,
                "pack_a": variant == "blocked_pack",
                "pack_b": variant == "blocked_pack",
                "simd": simd,
                "threads": th,
                "unroll_k": uk,
                "input_dtype": input_dtype,
                "output_dtype": output_dtype,
            }
        )

    # Deduplicate.
    dedup = {}
    for c in candidates:
        key = (
            c["kernel_variant"],
            c["bm"],
            c["bn"],
            c["bk"],
            c["pack_a"],
            c["pack_b"],
            c["simd"],
            c["threads"],
            c["unroll_k"],
            c.get("input_dtype", "f32"),
            c.get("output_dtype", "f32"),
        )
        dedup[key] = c

    final = list(dedup.values())
    rng = random.Random(seed + m * 13 + n * 7 + k * 5)
    rng.shuffle(final)
    # Always keep naive as the first baseline.
    naive = [c for c in final if c["kernel_variant"] == "naive"]
    non_naive = [c for c in final if c["kernel_variant"] != "naive"]
    non_naive = non_naive[: max(0, max_trials - len(naive))]
    return naive + non_naive


def generate_candidates(
    shapes: list[dict[str, int]],
    buckets_cfg: dict[str, Any],
    pattern_scores: dict[str, float],
    candidate_cfg: dict[str, Any],
    max_trials_per_bucket: int,
    seed: int,
    input_dtype: str = "f32",
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for shape in shapes:
        bucket = bucket_for_shape(shape, buckets_cfg)
        out[shape["name"]] = generate_candidates_for_shape(
            shape=shape,
            shape_bucket=bucket,
            pattern_scores=pattern_scores,
            candidate_cfg=candidate_cfg,
            max_trials=max_trials_per_bucket,
            seed=seed,
            input_dtype=input_dtype,
        )
    return out
