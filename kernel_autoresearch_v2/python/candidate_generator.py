from __future__ import annotations

import itertools
import random
from typing import Any

from .scheduler import bucket_for_shape


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = dict(candidate)
    if "bm" in out and "block_m" not in out:
        out["block_m"] = int(out["bm"])
    if "bn" in out and "block_n" not in out:
        out["block_n"] = int(out["bn"])
    if "bk" in out and "block_k" not in out:
        out["block_k"] = int(out["bk"])
    out["kernel_variant"] = str(out.get("kernel_variant", "naive"))
    out["block_m"] = int(out.get("block_m", 0))
    out["block_n"] = int(out.get("block_n", 0))
    out["block_k"] = int(out.get("block_k", 0))
    out["pack_a"] = bool(out.get("pack_a", False))
    out["pack_b"] = bool(out.get("pack_b", False))
    out["simd"] = bool(out.get("simd", False))
    out["threads"] = int(out.get("threads", 1))
    out["unroll_k"] = int(out.get("unroll_k", 1))
    return out


def candidate_signature(candidate: dict[str, Any]) -> str:
    c = _normalize_candidate(candidate)
    keys = ("kernel_variant", "block_m", "block_n", "block_k", "pack_a", "pack_b", "simd", "threads", "unroll_k")
    return "|".join(str(c[k]) for k in keys)


def baseline_candidate() -> dict[str, Any]:
    return {
        "kernel_variant": "naive",
        "block_m": 0,
        "block_n": 0,
        "block_k": 0,
        "pack_a": False,
        "pack_b": False,
        "simd": False,
        "threads": 1,
        "unroll_k": 1,
    }


def _shape_centered_choices(dim: int, choices: list[int], k: int = 3) -> list[int]:
    ranked = sorted(choices, key=lambda x: (abs(dim - x), -x))
    picked = sorted(set(ranked[:k]))
    return picked if picked else [choices[0]]


def _pick_threads(bucket: str, choices: list[int]) -> list[int]:
    uniq = sorted(set(max(1, int(x)) for x in choices))
    if bucket == "small":
        return sorted(set([1, min(4, uniq[-1])]))
    if bucket == "medium":
        return sorted(set([1, uniq[len(uniq) // 2], uniq[-1]]))
    return sorted(set([uniq[len(uniq) // 2], uniq[-1]]))


def generate_pool_for_bucket(
    *,
    bucket: str,
    shapes: list[dict[str, int]],
    candidate_cfg: dict[str, Any],
    max_candidates: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not shapes:
        return [baseline_candidate()]
    avg_m = int(sum(int(s["m"]) for s in shapes) / len(shapes))
    avg_n = int(sum(int(s["n"]) for s in shapes) / len(shapes))
    avg_k = int(sum(int(s["k"]) for s in shapes) / len(shapes))

    bms = _shape_centered_choices(avg_m, [int(x) for x in candidate_cfg.get("block_m", [32, 64])])
    bns = _shape_centered_choices(avg_n, [int(x) for x in candidate_cfg.get("block_n", [32, 64])])
    bks = _shape_centered_choices(avg_k, [int(x) for x in candidate_cfg.get("block_k", [32, 64])])
    threads = _pick_threads(bucket, [int(x) for x in candidate_cfg.get("thread_choices", [1, 2, 4])])
    unrolls = [int(x) for x in candidate_cfg.get("unroll_k", [1, 2])]
    variants = [str(x) for x in candidate_cfg.get("kernel_variants", ["naive", "blocked", "blocked_pack"])]

    allow_pack = bool(candidate_cfg.get("allow_pack", True))
    allow_simd = bool(candidate_cfg.get("allow_simd", True))

    out: list[dict[str, Any]] = [baseline_candidate()]
    for variant, bm, bn, bk, th, uk in itertools.product(variants, bms, bns, bks, threads, unrolls):
        if variant == "naive":
            continue
        uses_pack = variant in ("blocked_pack", "blocked_pack_simd")
        if uses_pack and not allow_pack:
            continue
        simd_flag = variant == "blocked_pack_simd"
        if simd_flag and not allow_simd:
            continue
        out.append(
            {
                "kernel_variant": variant,
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "pack_a": uses_pack,
                "pack_b": uses_pack,
                "simd": simd_flag,
                "threads": th,
                "unroll_k": uk,
            }
        )
        if allow_simd and variant in ("blocked", "blocked_pack"):
            out.append(
                {
                    "kernel_variant": variant,
                    "block_m": bm,
                    "block_n": bn,
                    "block_k": bk,
                    "pack_a": uses_pack,
                    "pack_b": uses_pack,
                    "simd": True,
                    "threads": th,
                    "unroll_k": uk,
                }
            )

    dedup: dict[str, dict[str, Any]] = {}
    for c in out:
        c_norm = _normalize_candidate(c)
        dedup[candidate_signature(c_norm)] = c_norm
    items = list(dedup.values())
    naive = [c for c in items if c["kernel_variant"] == "naive"]
    others = [c for c in items if c["kernel_variant"] != "naive"]
    rng = random.Random(seed + len(shapes) * 31 + avg_m * 3 + avg_n * 5 + avg_k * 7)
    rng.shuffle(others)
    others = others[: max(0, max_candidates - len(naive))]
    return naive + others


def mutate_candidate(
    *,
    base: dict[str, Any],
    candidate_cfg: dict[str, Any],
    iteration_seed: int,
) -> dict[str, Any]:
    c = _normalize_candidate(base)
    rng = random.Random(iteration_seed)
    knob = rng.choice(["block_m", "block_n", "block_k", "threads", "unroll_k", "simd"])
    if knob in ("block_m", "block_n", "block_k"):
        choices = [int(x) for x in candidate_cfg.get(knob, [32, 64])]
        c[knob] = rng.choice(choices)
    elif knob == "threads":
        choices = [max(1, int(x)) for x in candidate_cfg.get("thread_choices", [1, 2, 4])]
        c["threads"] = rng.choice(choices)
    elif knob == "unroll_k":
        choices = [max(1, int(x)) for x in candidate_cfg.get("unroll_k", [1, 2])]
        c["unroll_k"] = rng.choice(choices)
    elif knob == "simd":
        c["simd"] = not bool(c["simd"])

    if c["kernel_variant"] == "naive":
        c["block_m"] = 0
        c["block_n"] = 0
        c["block_k"] = 0
        c["pack_a"] = False
        c["pack_b"] = False
        c["simd"] = False
    return c


def split_shapes_by_bucket(
    *,
    shapes: list[dict[str, int]],
    buckets_cfg: dict[str, Any],
) -> dict[str, list[dict[str, int]]]:
    out: dict[str, list[dict[str, int]]] = {k: [] for k in buckets_cfg}
    for shape in shapes:
        bucket = bucket_for_shape(shape, buckets_cfg)
        out.setdefault(bucket, []).append(shape)
    return out

