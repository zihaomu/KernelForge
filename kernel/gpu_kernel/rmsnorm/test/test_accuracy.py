#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.rmsnorm.python.kc_rmsnorm import rmsnorm


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # x: [..., cols], w: [cols]
    mean = (x.float() * x.float()).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(mean + eps)
    y = x.float() * inv * w.float()
    return y.to(dtype=x.dtype)


def _check(dtype: torch.dtype, shape: tuple[int, ...], eps: float, atol: float, rtol: float):
    cols = shape[-1]
    x = (torch.randn(*shape, device="cuda", dtype=dtype) * 3.0).contiguous()
    w = (torch.randn(cols, device="cuda", dtype=dtype) * 0.1 + 1.0).contiguous()
    y_ref = rmsnorm_ref(x, w, eps)
    y = rmsnorm(x, w, eps=eps)
    if not torch.allclose(y, y_ref, atol=atol, rtol=rtol):
        max_abs = (y - y_ref).abs().max().item()
        max_rel = ((y - y_ref).abs() / (y_ref.abs() + 1e-9)).max().item()
        raise AssertionError(
            f"allclose failed dtype={dtype} shape={shape} eps={eps} max_abs={max_abs} max_rel={max_rel}"
        )


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    torch.manual_seed(0)
    _check(torch.float32, (128, 1024), eps=1e-5, atol=1e-5, rtol=1e-5)
    _check(torch.float16, (128, 1024), eps=1e-5, atol=5e-3, rtol=5e-3)
    _check(torch.float16, (16, 32, 1024), eps=1e-5, atol=5e-3, rtol=5e-3)
    print("kc_rmsnorm accuracy: OK")


if __name__ == "__main__":
    main()

