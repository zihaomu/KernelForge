from __future__ import annotations

import struct
from typing import Any


def deterministic_value(idx: int, salt: int) -> float:
    x = (idx * 1315423911 + salt * 2654435761) & 0xFFFFFFFF
    v = int(x % 1024) - 512
    return float(v) / 128.0


def deterministic_matrix(rows: int, cols: int, salt: int) -> list[float]:
    return [deterministic_value(i, salt) for i in range(rows * cols)]


def fp16_round(x: float) -> float:
    return struct.unpack("<e", struct.pack("<e", float(x)))[0]


def deterministic_matrix_f16(rows: int, cols: int, salt: int) -> list[float]:
    return [fp16_round(deterministic_value(i, salt)) for i in range(rows * cols)]


def deterministic_value_i8(idx: int, salt: int) -> int:
    x = (idx * 1315423911 + salt * 2654435761) & 0xFFFFFFFF
    return int(x % 255) - 127


def deterministic_matrix_i8(rows: int, cols: int, salt: int) -> list[int]:
    return [deterministic_value_i8(i, salt) for i in range(rows * cols)]


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


def gemm_reference_i8_i32(a: list[int], b: list[int], m: int, n: int, k: int) -> list[int]:
    c = [0] * (m * n)
    for i in range(m):
        for p in range(k):
            av = int(a[i * k + p])
            for j in range(n):
                c[i * n + j] += av * int(b[p * n + j])
    return c


def gemm_reference_f16_f16(a: list[float], b: list[float], m: int, n: int, k: int) -> list[float]:
    c = [0.0] * (m * n)
    for i in range(m):
        for p in range(k):
            av = float(a[i * k + p])
            for j in range(n):
                c[i * n + j] += av * float(b[p * n + j])
    return [fp16_round(x) for x in c]


def deterministic_reference_checksum(m: int, n: int, k: int, input_dtype: str = "f32") -> dict[str, Any]:
    if input_dtype == "i8":
        a = deterministic_matrix_i8(m, k, 123)
        b = deterministic_matrix_i8(k, n, 321)
        c = gemm_reference_i8_i32(a, b, m, n, k)
    elif input_dtype == "f16":
        a = deterministic_matrix_f16(m, k, 123)
        b = deterministic_matrix_f16(k, n, 321)
        c = gemm_reference_f16_f16(a, b, m, n, k)
    else:
        a = deterministic_matrix(m, k, 123)
        b = deterministic_matrix(k, n, 321)
        c = gemm_reference(a, b, m, n, k)
    sums = checksum([float(x) for x in c])
    return {"m": m, "n": n, "k": k, "input_dtype": input_dtype, **sums}
