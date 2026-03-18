#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.reduce.python.kc_reduce import reduce_max, reduce_sum


def bench_one(name, fn, x, iters=2000):
    for _ in range(50):
        y = fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1e3 / iters
    print(f"{name:22s} {ms:8.4f} ms  out_mean={y.float().mean().item():.6f}")
    return ms


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    shape = (4096, 1024)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    print(f"shape={shape} dtype={x.dtype}")

    bench_one("torch.sum(dim=-1)", lambda t: torch.sum(t, dim=-1), x)
    bench_one("kc.reduce_sum", lambda t: reduce_sum(t, dim=-1), x)
    bench_one("torch.max(dim=-1)", lambda t: torch.max(t, dim=-1).values, x)
    bench_one("kc.reduce_max", lambda t: reduce_max(t, dim=-1), x)


if __name__ == "__main__":
    main()

