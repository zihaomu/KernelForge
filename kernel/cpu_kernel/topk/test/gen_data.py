#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np


def topk_ref(x: np.ndarray, k: int, descend: int):
    # x: [rows, cols]
    if descend:
        idx = np.argsort(-x, axis=-1)[:, :k]
    else:
        idx = np.argsort(x, axis=-1)[:, :k]
    rows = x.shape[0]
    vals = np.take_along_axis(x, idx, axis=-1).astype(np.float32)
    idx = idx.astype(np.int32)
    return vals, idx


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = 19
    cols = 257
    k = 8
    descend = 1
    x = (rng.standard_normal((rows, cols), dtype=np.float32) * 3.0).astype(np.float32)
    vals, idx = topk_ref(x, k, descend)
    params = np.array([rows, cols, k, descend], dtype=np.int32)

    np.save(out_dir / "input.npy", x)
    np.save(out_dir / "values.npy", vals)
    np.save(out_dir / "indices.npy", idx)
    np.save(out_dir / "params.npy", params)
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

