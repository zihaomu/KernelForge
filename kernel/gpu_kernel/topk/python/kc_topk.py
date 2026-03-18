from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load


_EXT: Optional[object] = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    this_dir = Path(__file__).resolve().parent.parent  # topk/
    cuda_dir = this_dir / "cuda_kernel"
    sources = [str(cuda_dir / "binding.cpp"), str(cuda_dir / "kernel.cu")]

    _EXT = load(
        name="kc_topk_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
        is_python_module=True,
    )
    return _EXT


def topk(x: torch.Tensor, k: int, dim: int = -1, largest: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise ValueError("kc_topk: expected CUDA tensor")
    if dim != -1 and dim != x.dim() - 1:
        raise ValueError("kc_topk(v1) only supports dim=-1")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"kc_topk: unsupported dtype {x.dtype}")
    if k <= 0:
        raise ValueError("kc_topk: k must be > 0")
    if k > 32:
        raise ValueError("kc_topk(v1): k must be <= 32")
    ext = _load_ext()
    return ext.topk_lastdim(x.contiguous(), int(k), bool(largest))

