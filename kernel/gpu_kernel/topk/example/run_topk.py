#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.topk.python.kc_topk import topk


def main():
    if not torch.cuda.is_available():
        print("cuda not available")
        return

    x = torch.tensor([[1.0, 9.0, 2.0, -1.0, 3.0, 7.0, 0.0, 4.0]], device="cuda", dtype=torch.float32)
    v, i = topk(x, k=3, dim=-1, largest=True)
    print("x:", x)
    print("values:", v)
    print("indices:", i)


if __name__ == "__main__":
    main()

