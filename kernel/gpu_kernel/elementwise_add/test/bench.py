#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.elementwise_add.python.kc_elementwise_add import add


def bench_one(name, fn, a, b, iters=2000):
    for _ in range(50):
        y = fn(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1e3 / iters
    print(f"{name:18s} {ms:8.4f} ms  out_mean={y.float().mean().item():.6f}")
    return ms


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    shape = (1 << 24,)  # ~16M
    a = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    print(f"shape={shape} dtype={a.dtype}")

    ms_ref = bench_one("torch.add", lambda x, y: x + y, a, b)
    ms_kc = bench_one("kc.add", add, a, b)
    print(f"speedup: {ms_ref / ms_kc:.2f}x (torch / kc)")


if __name__ == "__main__":
    main()

