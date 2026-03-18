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

    this_dir = Path(__file__).resolve().parent.parent  # softmax/
    cuda_dir = this_dir / "cuda_kernel"

    sources = [
        str(cuda_dir / "binding.cpp"),
        str(cuda_dir / "kernel.cu"),
    ]

    # No ninja in this environment; force legacy build.
    _EXT = load(
        name="kc_softmax_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
        is_python_module=True,
    )
    return _EXT


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    v1: CUDA softmax on last dimension only.
    """
    if not x.is_cuda:
        raise ValueError("kc_softmax: expected a CUDA tensor")
    if dim != -1 and dim != x.dim() - 1:
        raise ValueError("kc_softmax(v1) only supports dim=-1 (last dimension)")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"kc_softmax: unsupported dtype {x.dtype} (expected fp16/fp32)")

    ext = _load_ext()
    return ext.softmax_forward(x.contiguous())
