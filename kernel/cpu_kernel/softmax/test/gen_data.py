#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np


def softmax_ref(x: np.ndarray) -> np.ndarray:
    # x: [rows, cols]
    x = x.astype(np.float32)
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = 17
    cols = 257
    x = (rng.standard_normal((rows, cols), dtype=np.float32) * 3.0).astype(np.float32)
    y = softmax_ref(x)

    params = np.array([rows, cols], dtype=np.int32)
    np.save(out_dir / "input.npy", x)
    np.save(out_dir / "output.npy", y)
    np.save(out_dir / "params.npy", params)
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

