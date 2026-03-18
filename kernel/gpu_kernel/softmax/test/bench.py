#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.softmax.python.kc_softmax import softmax


def bench_one(name, fn, x, iters=200):
    # Warmup
    for _ in range(20):
        y = fn(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1e3 / iters
    print(f"{name:18s} {ms:8.4f} ms  out_mean={y.float().mean().item():.6f}")
    return ms


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    # Typical transformer-ish logits: [B*T, vocab] would be huge; keep it moderate.
    shape = (4096, 1024)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()

    print(f"shape={shape} dtype={x.dtype}")
    ms_ref = bench_one("torch.softmax", lambda t: torch.softmax(t, dim=-1), x)
    ms_kc = bench_one("kc.softmax", lambda t: softmax(t, dim=-1), x)
    print(f"speedup: {ms_ref / ms_kc:.2f}x (torch / kc)")


if __name__ == "__main__":
    main()
