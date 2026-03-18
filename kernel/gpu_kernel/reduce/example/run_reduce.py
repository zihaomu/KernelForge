#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.reduce.python.kc_reduce import reduce_max, reduce_sum


def main():
    if not torch.cuda.is_available():
        print("cuda not available")
        return

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]], device="cuda", dtype=torch.float32)
    print("x:", x)
    print("sum:", reduce_sum(x, dim=-1))
    print("max:", reduce_max(x, dim=-1))


if __name__ == "__main__":
    main()

