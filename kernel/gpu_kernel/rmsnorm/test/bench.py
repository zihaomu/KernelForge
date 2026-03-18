#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.rmsnorm.python.kc_rmsnorm import rmsnorm


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    mean = (x.float() * x.float()).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(mean + eps)
    return (x.float() * inv * w.float()).to(dtype=x.dtype)


def bench_one(name, fn, iters=2000):
    for _ in range(50):
        y = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1e3 / iters
    print(f"{name:20s} {ms:8.4f} ms  out_mean={y.float().mean().item():.6f}")
    return ms


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    shape = (4096, 1024)
    eps = 1e-5
    x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    w = (torch.randn(shape[-1], device="cuda", dtype=torch.float16) * 0.1 + 1.0).contiguous()
    print(f"shape={shape} dtype={x.dtype}")

    bench_one("torch(ref)", lambda: rmsnorm_ref(x, w, eps))
    bench_one("kc.rmsnorm", lambda: rmsnorm(x, w, eps))


if __name__ == "__main__":
    main()

