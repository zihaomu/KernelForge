#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np


def rmsnorm_ref(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    # x: [rows, cols], w: [cols]
    x = x.astype(np.float32)
    w = w.astype(np.float32)
    mean = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean + eps)
    return (x * inv) * w[None, :]


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = 17
    cols = 1024
    eps = 1e-5
    x = (rng.standard_normal((rows, cols), dtype=np.float32) * 3.0).astype(np.float32)
    w = (rng.standard_normal((cols,), dtype=np.float32) * 0.1 + 1.0).astype(np.float32)
    y = rmsnorm_ref(x, w, eps).astype(np.float32)
    params = np.array([rows, cols], dtype=np.int32)
    eps_arr = np.array([eps], dtype=np.float32)

    np.save(out_dir / "input.npy", x)
    np.save(out_dir / "weight.npy", w)
    np.save(out_dir / "output.npy", y)
    np.save(out_dir / "params.npy", params)
    np.save(out_dir / "eps.npy", eps_arr)
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

