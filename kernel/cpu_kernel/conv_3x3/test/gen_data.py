#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np


def conv2d_nchw_ref(x, w, b, stride_h, stride_w, pad_h, pad_w):
    # x: [N, Cin, H, W]
    # w: [Cout, Cin, 3, 3]
    n, cin, h, w_in = x.shape
    cout = w.shape[0]
    assert w.shape[1] == cin and w.shape[2] == 3 and w.shape[3] == 3
    hout = (h + 2 * pad_h - 3) // stride_h + 1
    wout = (w_in + 2 * pad_w - 3) // stride_w + 1
    y = np.zeros((n, cout, hout, wout), dtype=np.float32)

    xpad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    for ni in range(n):
        for co in range(cout):
            bias = float(b[co]) if b is not None else 0.0
            for yo in range(hout):
                yi0 = yo * stride_h
                for xo in range(wout):
                    xi0 = xo * stride_w
                    # [Cin,3,3] dot
                    acc = bias
                    patch = xpad[ni, :, yi0 : yi0 + 3, xi0 : xi0 + 3]
                    acc += float(np.sum(patch * w[co, :, :, :]))
                    y[ni, co, yo, xo] = acc
    return y


def main():
    here = Path(__file__).resolve().parent
    out_dir = here / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # Keep it small-ish for a unit test, but non-trivial.
    n = 1
    cin = 8
    cout = 16
    h = 13
    w = 17
    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1

    x = (rng.standard_normal((n, cin, h, w), dtype=np.float32) * 0.1).astype(np.float32)
    wt = (rng.standard_normal((cout, cin, 3, 3), dtype=np.float32) * 0.1).astype(np.float32)
    b = (rng.standard_normal((cout,), dtype=np.float32) * 0.1).astype(np.float32)

    y = conv2d_nchw_ref(x, wt, b, stride_h, stride_w, pad_h, pad_w)

    params = np.array([n, cin, h, w, cout, stride_h, stride_w, pad_h, pad_w], dtype=np.int32)

    np.save(out_dir / "input.npy", x)
    np.save(out_dir / "weight.npy", wt)
    np.save(out_dir / "bias.npy", b)
    np.save(out_dir / "output.npy", y)
    np.save(out_dir / "params.npy", params)

    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    # Make it robust when launched from other working dirs.
    os.chdir(Path(__file__).resolve().parent)
    main()

