from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXT: Optional[object] = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    this_dir = Path(__file__).resolve().parent.parent  # rmsnorm/
    cuda_dir = this_dir / "cuda_kernel"
    sources = [str(cuda_dir / "binding.cpp"), str(cuda_dir / "kernel.cu")]

    _EXT = load(
        name="kc_rmsnorm_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
        is_python_module=True,
    )
    return _EXT


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("kc_rmsnorm: expected CUDA tensors")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"kc_rmsnorm: unsupported dtype {x.dtype}")
    if weight.dtype != x.dtype:
        raise ValueError("kc_rmsnorm: weight dtype must match x dtype (v1)")
    if weight.dim() != 1:
        raise ValueError("kc_rmsnorm: weight must be 1D")
    if x.numel() == 0:
        return x.clone()
    cols = x.size(-1)
    if weight.numel() != cols:
        raise ValueError(f"kc_rmsnorm: weight length {weight.numel()} != x.size(-1) {cols}")
    ext = _load_ext()
    return ext.rmsnorm_forward(x.contiguous(), weight.contiguous(), float(eps))

