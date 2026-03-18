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

    this_dir = Path(__file__).resolve().parent.parent  # reduce/
    cuda_dir = this_dir / "cuda_kernel"
    sources = [str(cuda_dir / "binding.cpp"), str(cuda_dir / "kernel.cu")]

    _EXT = load(
        name="kc_reduce_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
        is_python_module=True,
    )
    return _EXT


def reduce_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("kc_reduce_sum: expected CUDA tensor")
    if dim != -1 and dim != x.dim() - 1:
        raise ValueError("kc_reduce_sum(v1) only supports dim=-1")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"kc_reduce_sum: unsupported dtype {x.dtype}")
    ext = _load_ext()
    return ext.reduce_sum_lastdim(x.contiguous())


def reduce_max(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("kc_reduce_max: expected CUDA tensor")
    if dim != -1 and dim != x.dim() - 1:
        raise ValueError("kc_reduce_max(v1) only supports dim=-1")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"kc_reduce_max: unsupported dtype {x.dtype}")
    ext = _load_ext()
    return ext.reduce_max_lastdim(x.contiguous())

