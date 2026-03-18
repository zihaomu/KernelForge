#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.rmsnorm.python.kc_rmsnorm import rmsnorm


def main():
    if not torch.cuda.is_available():
        print("cuda not available")
        return

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=torch.float32)
    w = torch.ones(4, device="cuda", dtype=torch.float32)
    y = rmsnorm(x, w, eps=1e-5)
    print("x:", x)
    print("y:", y)


if __name__ == "__main__":
    main()

