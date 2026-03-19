from __future__ import annotations

from typing import Any


def deterministic_value(idx: int, salt: int) -> float:
    x = (idx * 1315423911 + salt * 2654435761) & 0xFFFFFFFF
    v = int(x % 1024) - 512
    return float(v) / 128.0


def deterministic_matrix(rows: int, cols: int, salt: int) -> list[float]:
    return [deterministic_value(i, salt) for i in range(rows * cols)]


def gemm_reference(a: list[float], b: list[float], m: int, n: int, k: int) -> list[float]:
    c = [0.0] * (m * n)
    for i in range(m):
        for p in range(k):
            av = a[i * k + p]
            for j in range(n):
                c[i * n + j] += av * b[p * n + j]
    return c


def checksum(vec: list[float]) -> dict[str, float]:
    out_sum = 0.0
    out_l2 = 0.0
    for x in vec:
        out_sum += x
        out_l2 += x * x
    return {"output_sum": out_sum, "output_l2": out_l2}


def deterministic_reference_checksum(m: int, n: int, k: int) -> dict[str, Any]:
    a = deterministic_matrix(m, k, 123)
    b = deterministic_matrix(k, n, 321)
    c = gemm_reference(a, b, m, n, k)
    sums = checksum(c)
    return {"m": m, "n": n, "k": k, **sums}

