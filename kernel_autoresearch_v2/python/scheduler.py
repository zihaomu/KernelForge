from __future__ import annotations

from typing import Any


def shape_volume(shape: dict[str, int]) -> int:
    return int(shape["m"]) * int(shape["n"]) * int(shape["k"])


def bucket_for_shape(shape: dict[str, int], buckets_cfg: dict[str, Any]) -> str:
    vol = shape_volume(shape)
    for name, cfg in buckets_cfg.items():
        max_volume = cfg.get("max_volume")
        if max_volume is None:
            return name
        if vol <= int(max_volume):
            return name
    return "large"

