#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.topk.python.kc_topk import topk


def _check(dtype: torch.dtype, shape: tuple[int, ...], k: int, atol: float, rtol: float):
    x = (torch.randn(*shape, device="cuda", dtype=dtype) * 3.0).contiguous()
    v_ref, i_ref = torch.topk(x, k=k, dim=-1, largest=True)
    v, i = topk(x, k=k, dim=-1, largest=True)
    if not torch.allclose(v, v_ref, atol=atol, rtol=rtol):
        max_abs = (v - v_ref).abs().max().item()
        raise AssertionError(f"values allclose failed dtype={dtype} shape={shape} k={k} max_abs={max_abs}")
    # Indices can differ when there are ties (especially in fp16). Verify indices are consistent with values.
    gathered = x.gather(dim=-1, index=i)
    if not torch.allclose(gathered, v, atol=atol, rtol=rtol):
        max_abs = (gathered - v).abs().max().item()
        raise AssertionError(f"gather(values) mismatch dtype={dtype} shape={shape} k={k} max_abs={max_abs}")


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    torch.manual_seed(0)
    _check(torch.float32, (64, 257), k=8, atol=1e-6, rtol=1e-5)
    _check(torch.float16, (64, 257), k=8, atol=5e-3, rtol=5e-3)
    _check(torch.float16, (16, 32, 257), k=8, atol=5e-3, rtol=5e-3)
    print("kc_topk accuracy: OK")


if __name__ == "__main__":
    main()
