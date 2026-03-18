#!/usr/bin/env python3

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from kernel.gpu_kernel.elementwise_add.python.kc_elementwise_add import add


def main():
    if not torch.cuda.is_available():
        print("cuda not available")
        return

    a = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    b = torch.tensor([10.0, 20.0, 30.0], device="cuda", dtype=torch.float32)
    y = add(a, b)
    print("a:", a)
    print("b:", b)
    print("y:", y)


if __name__ == "__main__":
    main()

