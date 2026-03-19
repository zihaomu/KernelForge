from __future__ import annotations

import math
import time
from typing import Any


def torch_runtime_available() -> tuple[bool, str]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return False, f"torch_import_failed:{exc}"
    if not torch.cuda.is_available():
        return False, "cuda_not_available"
    return True, "ok"


def _dtype_from_name(name: str) -> Any:
    import torch  # type: ignore

    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported_dtype:{name}")
    return mapping[name]


def run_torch_gemm_candidate(
    *,
    shape: dict[str, int],
    candidate: dict[str, Any],
    warmup_iters: int,
    measure_iters: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    import torch  # type: ignore

    device = "cuda"
    dtype_name = str(candidate["dtype"])
    allow_tf32 = bool(candidate["allow_tf32"])
    dtype = _dtype_from_name(dtype_name)

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    m = int(shape["m"])
    n = int(shape["n"])
    k = int(shape["k"])

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    ref = torch.matmul(a.float(), b.float())

    for _ in range(max(0, warmup_iters)):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    lat_ms: list[float] = []
    for _ in range(max(1, measure_iters)):
        t0 = time.perf_counter()
        out = torch.matmul(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)

    lat_ms_sorted = sorted(lat_ms)
    p50 = lat_ms_sorted[len(lat_ms_sorted) // 2]
    flops = 2.0 * float(m) * float(n) * float(k)
    gflops = flops / (max(p50, 1e-9) * 1e6)

    out_f = out.float()
    diff = (out_f - ref).abs()
    max_abs_err = float(diff.max().item())
    denom = torch.maximum(ref.abs(), torch.tensor(1e-9, device=ref.device))
    max_rel_err = float((diff / denom).max().item())
    valid = bool(math.isclose(max_abs_err, 0.0, rel_tol=rtol, abs_tol=atol) or max_rel_err <= rtol)

    return {
        "valid": valid,
        "shape_name": shape["name"],
        "dtype": dtype_name,
        "allow_tf32": allow_tf32,
        "latency_ms_p50": p50,
        "gflops": gflops,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
    }

