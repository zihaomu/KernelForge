#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.topk.python.kc_topk import topk


def bench_one(name, fn, iters=500):
    for _ in range(50):
        v, i = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        v, i = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1e3 / iters
    print(f"{name:18s} {ms:8.4f} ms  v_mean={v.float().mean().item():.6f}")
    return ms


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    shape = (4096, 4096)
    k = 8
    x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    print(f"shape={shape} dtype={x.dtype} k={k}")

    bench_one("torch.topk", lambda: torch.topk(x, k=k, dim=-1, largest=True))
    bench_one("kc.topk", lambda: topk(x, k=k, dim=-1, largest=True))


if __name__ == "__main__":
    main()

