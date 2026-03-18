#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n = 12345
    a = (rng.standard_normal((n,), dtype=np.float32) * 3.0).astype(np.float32)
    b = (rng.standard_normal((n,), dtype=np.float32) * 3.0).astype(np.float32)
    y = a + b
    params = np.array([n], dtype=np.int32)

    np.save(out_dir / "a.npy", a)
    np.save(out_dir / "b.npy", b)
    np.save(out_dir / "out.npy", y)
    np.save(out_dir / "params.npy", params)
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

