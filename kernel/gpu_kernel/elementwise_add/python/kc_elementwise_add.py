from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXT: Optional[object] = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    this_dir = Path(__file__).resolve().parent.parent  # elementwise_add/
    cuda_dir = this_dir / "cuda_kernel"

    sources = [
        str(cuda_dir / "binding.cpp"),
        str(cuda_dir / "kernel.cu"),
    ]

    _EXT = load(
        name="kc_elementwise_add_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
        is_python_module=True,
    )
    return _EXT


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("kc_elementwise_add: expected CUDA tensors")
    if a.shape != b.shape:
        raise ValueError(f"kc_elementwise_add: shape mismatch {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"kc_elementwise_add: dtype mismatch {a.dtype} vs {b.dtype}")
    if a.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"kc_elementwise_add: unsupported dtype {a.dtype}")

    ext = _load_ext()
    return ext.add_forward(a.contiguous(), b.contiguous())

