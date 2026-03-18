#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.reduce.python.kc_reduce import reduce_max, reduce_sum


def _check(dtype: torch.dtype, shape: tuple[int, ...], atol: float, rtol: float):
    x = (torch.randn(*shape, device="cuda", dtype=dtype) * 3.0).contiguous()
    y_sum_ref = torch.sum(x, dim=-1)
    y_max_ref = torch.max(x, dim=-1).values
    y_sum = reduce_sum(x, dim=-1)
    y_max = reduce_max(x, dim=-1)
    if not torch.allclose(y_sum, y_sum_ref, atol=atol, rtol=rtol):
        max_abs = (y_sum - y_sum_ref).abs().max().item()
        raise AssertionError(f"reduce_sum allclose failed dtype={dtype} shape={shape} max_abs={max_abs}")
    if not torch.allclose(y_max, y_max_ref, atol=atol, rtol=rtol):
        max_abs = (y_max - y_max_ref).abs().max().item()
        raise AssertionError(f"reduce_max allclose failed dtype={dtype} shape={shape} max_abs={max_abs}")


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    torch.manual_seed(0)
    # Different reduction orders can introduce tiny FP32 diffs vs torch.
    _check(torch.float32, (128, 257), atol=2e-5, rtol=1e-5)
    _check(torch.float16, (128, 257), atol=5e-3, rtol=5e-3)
    _check(torch.float16, (16, 32, 257), atol=5e-3, rtol=5e-3)
    print("kc_reduce accuracy: OK")


if __name__ == "__main__":
    main()
