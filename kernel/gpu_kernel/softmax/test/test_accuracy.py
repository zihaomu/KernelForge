#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

# When executed as a file (python path/to/test.py), Python's sys.path does not
# include the repo root. Add it so `import kernel...` works.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.softmax.python.kc_softmax import softmax


def _check(dtype: torch.dtype, shape: tuple[int, ...], atol: float, rtol: float):
    x = (torch.randn(*shape, device="cuda", dtype=dtype) * 3.0).contiguous()
    y_ref = torch.softmax(x, dim=-1)
    y = softmax(x, dim=-1)
    ok = torch.allclose(y, y_ref, atol=atol, rtol=rtol)
    if not ok:
        max_abs = (y - y_ref).abs().max().item()
        max_rel = ((y - y_ref).abs() / (y_ref.abs() + 1e-9)).max().item()
        raise AssertionError(
            f"allclose failed dtype={dtype} shape={shape} max_abs={max_abs} max_rel={max_rel}"
        )


def main():
    if not torch.cuda.is_available():
        print("cuda not available: skip")
        return

    torch.manual_seed(0)
    # Small and medium shapes; dim=-1 only (v1).
    _check(torch.float32, (4, 128), atol=1e-6, rtol=1e-5)
    _check(torch.float32, (16, 1024), atol=1e-6, rtol=1e-5)
    _check(torch.float16, (4, 128), atol=5e-3, rtol=5e-3)
    _check(torch.float16, (16, 1024), atol=5e-3, rtol=5e-3)
    print("kc_softmax accuracy: OK")


if __name__ == "__main__":
    main()
