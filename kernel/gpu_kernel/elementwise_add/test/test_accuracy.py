#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.elementwise_add.python.kc_elementwise_add import add


def _check(dtype: torch.dtype, shape: tuple[int, ...], atol: float, rtol: float):
    a = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    b = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    y_ref = a + b
    y = add(a, b)
    if not torch.allclose(y, y_ref, atol=atol, rtol=rtol):
        max_abs = (y - y_ref).abs().max().item()
        raise AssertionError(f"allclose failed dtype={dtype} shape={shape} max_abs={max_abs}")


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    torch.manual_seed(0)
    _check(torch.float32, (1024, 1024), atol=0.0, rtol=0.0)
    _check(torch.float16, (1024, 1024), atol=0.0, rtol=0.0)
    print("kc_elementwise_add accuracy: OK")


if __name__ == "__main__":
    main()

