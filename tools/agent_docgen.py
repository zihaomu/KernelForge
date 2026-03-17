#!/usr/bin/env python3
"""
Generate per-repo README.agent.md + patterns/ skeletons for kernel metadata search.

Scope constraints (per user request):
- GPU: only NVIDIA backend (CUDA)
- CPU: only ARM architecture (NEON/SVE). Other CPU/GPU backends are treated as out of scope.

This is intentionally heuristic: it does lightweight static scanning (ripgrep counts + file presence)
to build "metadata" that is more useful for an agent than raw source text.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _have_rg() -> bool:
    return shutil.which("rg") is not None


def _rg_count(cwd: Path, pattern: str, globs: List[str]) -> int:
    """
    Return total match count across files using `rg -c` and summing.
    """
    if not _have_rg():
        return 0
    cmd = ["rg", "-c", "-S", "--no-heading", "--color", "never"]
    for g in globs:
        cmd += ["-g", g]
    cmd.append(pattern)
    cp = _run(cmd, cwd=cwd)
    if cp.returncode not in (0, 1):  # 0: found, 1: not found
        return 0
    total = 0
    for line in cp.stdout.splitlines():
        # format: path:count
        try:
            _, cnt = line.rsplit(":", 1)
            total += int(cnt.strip())
        except Exception:
            continue
    return total


def _count_files(cwd: Path, globs: List[str]) -> int:
    if not _have_rg():
        return 0
    cmd = ["rg", "--files"]
    for g in globs:
        cmd += ["-g", g]
    cp = _run(cmd, cwd=cwd)
    if cp.returncode != 0:
        return 0
    return len([ln for ln in cp.stdout.splitlines() if ln.strip()])


def _exists_any(cwd: Path, rel_paths: Iterable[str]) -> bool:
    for p in rel_paths:
        if (cwd / p).exists():
            return True
    return False


def _read_first_paragraph(readme_path: Path, max_lines: int = 14, max_chars: int = 320) -> str:
    """
    Extract a short human-readable description from README.
    Skips badge lines / headings and returns up to a few lines.
    """
    try:
        raw = readme_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""

    lines: List[str] = []
    for ln in raw:
        s = ln.strip()
        if not s:
            if lines:
                break
            continue
        # Skip common badge-only lines / html-only lines
        if s.startswith("[![") or s.startswith("![") or s.startswith("<"):
            continue
        # Skip horizontal rules / separators
        if s and set(s) <= {"-", "=", "_"}:
            continue
        # Skip top-level heading
        if s.startswith("#"):
            continue
        lines.append(s)
        if len(lines) >= max_lines:
            break
    out = " ".join(lines).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out


def _pick_readme(repo: Path) -> Optional[Path]:
    for name in ("README.md", "README.MD", "readme.md", "Readme.md"):
        p = repo / name
        if p.exists():
            return p
    # fall back to any README*
    cands = []
    for p in sorted(repo.glob("README*")):
        bn = p.name
        if bn.startswith("README.agent.md"):
            continue
        if bn.endswith(".bak") or ".bak." in bn:
            continue
        cands.append(p)
    return cands[0] if cands else None


def _safe_write(path: Path, content: str, *, backup_if_exists: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8", errors="ignore")
            if existing == content:
                return
        except Exception:
            # If we can't read it, proceed with best-effort backup/write.
            pass
    if path.exists() and backup_if_exists:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        except Exception:
            # Best effort: if we can't read, still write new file.
            pass
    path.write_text(content, encoding="utf-8")


def _ensure_pattern_file(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bool_to_yesno(v: bool) -> str:
    return "yes" if v else "no"


@dataclass
class RepoSignals:
    cu_files: int
    asm_files: int
    tests_present: bool
    bench_present: bool
    ci_present: bool
    gpu_relevant: bool
    cpu_arm_relevant: bool


@dataclass(frozen=True)
class OpSpec:
    name: str
    # Map target -> rg regex
    detect: Dict[str, str]
    min_files: int = 1


@dataclass
class OpScan:
    name: str
    target: str  # "gpu" or "cpu"
    hits: int
    files: List[str]  # repo-relative paths
    opt_tag_counts: Dict[str, int]


@dataclass
class RepoScan:
    repo: Path
    name: str
    desc: str
    sig: RepoSignals
    style: str
    dtypes: List[str]
    layouts: List[str]
    production_grade: str
    ops: List[OpScan]


def _detect_repo_signals(repo: Path) -> RepoSignals:
    cu_files = _count_files(repo, ["*.cu", "*.cuh"])
    asm_files = _count_files(repo, ["*.S", "*.s", "*.asm"])
    tests_present = _exists_any(repo, ["test", "tests"]) or _count_files(repo, ["*test*", "*Test*"]) > 0
    bench_present = _exists_any(repo, ["bench", "benchmark", "benchmarks"]) or _count_files(
        repo, ["*bench*", "*benchmark*"]
    ) > 0
    ci_present = (repo / ".github" / "workflows").exists()

    # GPU relevance: presence of CUDA sources or strong CUDA markers
    cuda_markers = _rg_count(repo, r"(__global__|cuda(Stream|Error|Memcpy)|__device__|#include\\s*<cuda)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp"])
    gpu_relevant = cu_files > 0 or cuda_markers > 0

    # ARM CPU relevance: NEON/SVE markers or lots of asm.
    arm_markers = _rg_count(
        repo,
        r"(arm_neon\.h|__ARM_NEON|__aarch64__|\bsve\b|\bneon\b)",
        ["*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.S", "*.s", "*.asm"],
    )
    # Avoid false positives from a single guard/comment: require meaningful signal
    # unless there is explicit ASM present.
    cpu_arm_relevant = asm_files > 0 or arm_markers >= 20

    return RepoSignals(
        cu_files=cu_files,
        asm_files=asm_files,
        tests_present=tests_present,
        bench_present=bench_present,
        ci_present=ci_present,
        gpu_relevant=gpu_relevant,
        cpu_arm_relevant=cpu_arm_relevant,
    )


def _infer_production_grade(sig: RepoSignals, repo_name: str) -> str:
    # Repo name hints
    if re.search(r"tutorial|example", repo_name, re.IGNORECASE):
        return "low"
    if sig.tests_present and sig.bench_present and sig.ci_present:
        return "high"
    if sig.tests_present and (sig.bench_present or sig.ci_present):
        return "medium"
    if sig.tests_present or sig.bench_present:
        return "medium"
    return "low"


def _infer_style(sig: RepoSignals, repo_name: str, repo: Path) -> str:
    # Very rough: "xnnpack-style" if microkernel/asm-heavy; "oneflow-style" if CUDA template heavy; else "mixed".
    if repo_name.lower() == "xnnpack":
        return "xnnpack_style"
    if repo_name.lower() == "oneflow":
        return "oneflow_style"
    if re.search(r"tutorial", repo_name, re.IGNORECASE):
        return "teaching_code"
    microkernel_hits = _rg_count(repo, r"(microkernel|ukernel|__asm__|\.macro)", ["*.c", "*.cc", "*.S", "*.s", "*.asm"])
    cutlass_hits = _rg_count(repo, r"(cutlass|cute::|cublasLt)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp"])
    if sig.asm_files > 50 or microkernel_hits > 50:
        return "xnnpack_style"
    if sig.cu_files > 20 and cutlass_hits > 0:
        return "oneflow_style"
    if sig.gpu_relevant and sig.cpu_arm_relevant:
        return "mixed_gpu_cpu"
    if sig.gpu_relevant:
        return "gpu_cuda_focus"
    if sig.cpu_arm_relevant:
        return "cpu_arm_focus"
    return "unknown"


def _infer_dtypes(repo: Path) -> List[str]:
    # Keep this conservative: only add dtype if we see strong tokens.
    dtypes: List[Tuple[str, int]] = []
    dtypes.append(("fp16", _rg_count(repo, r"(__half|half2|__half2|\bfp16\b)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp", "*.c"])))
    dtypes.append(("bf16", _rg_count(repo, r"(nv_bfloat16|\bbf16\b|\bbfloat16\b)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp", "*.c"])))
    dtypes.append(("fp32", _rg_count(repo, r"(\bfloat\b|\bfp32\b)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp", "*.c"])))
    dtypes.append(("int8", _rg_count(repo, r"(\bint8\b|int8_t)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp", "*.c"])))

    # Presence threshold: >0 matches, but avoid listing fp32 if it is only a handful of generic "float" mentions.
    out: List[str] = []
    for name, cnt in dtypes:
        if name == "fp32":
            if cnt >= 200:
                out.append(name)
        else:
            if cnt > 0:
                out.append(name)
    # stable ordering
    order = {"fp16": 0, "bf16": 1, "fp32": 2, "int8": 3}
    out.sort(key=lambda x: order.get(x, 99))
    return out


def _infer_layouts(repo: Path) -> List[str]:
    # For GEMM: row_major/col_major are common; for conv: NCHW/NHWC.
    layouts: List[str] = []
    if _rg_count(repo, r"(row[_ ]major|RowMajor)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp"]) > 0:
        layouts.append("row_major")
    if _rg_count(repo, r"(col[_ ]major|ColMajor)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp"]) > 0:
        layouts.append("col_major")
    if _rg_count(repo, r"\bNHWC\b", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp", "*.c"]) > 0:
        layouts.append("NHWC")
    if _rg_count(repo, r"\bNCHW\b", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp", "*.c"]) > 0:
        layouts.append("NCHW")
    # If nothing detected, leave empty (agent should not assume).
    return sorted(set(layouts))


def _infer_gpu_opt_tags(repo: Path) -> Tuple[List[str], List[str]]:
    tags: List[str] = []
    constraints: List[str] = []

    tensor = _rg_count(repo, r"(wmma|mma\.sync|ldmatrix|nvcuda::wmma)", ["*.cu", "*.cuh"])
    if tensor > 0:
        tags.append("tensor_core")
        # Conservative: tensor cores exist from sm70, but many modern code assumes sm75/80.
        constraints.append("sm70+ (inferred: tensor core tokens present)")

    cp_async = _rg_count(repo, r"(cp\.async|cuda::memcpy_async|pipeline)", ["*.cu", "*.cuh"])
    if cp_async > 0:
        tags.append("async_copy_pipeline")
        constraints.append("sm80+ (inferred: cp.async/pipeline tokens present)")

    shared = _rg_count(repo, r"(__shared__|shared memory)", ["*.cu", "*.cuh"])
    if shared > 0:
        tags.append("shared_memory_tiling")

    vec = _rg_count(repo, r"(half2|__half2|float4|int4|reinterpret_cast<[^>]*4>)", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp"])
    if vec > 0:
        tags.append("vectorized_load_store")
        constraints.append("alignment_16 (inferred: vectorized load/store tokens present)")

    warp = _rg_count(repo, r"(__shfl_|warp(Reduce|_reduce)|cooperative_groups::)", ["*.cu", "*.cuh"])
    if warp > 0:
        tags.append("warp_reduce")

    dbl = _rg_count(repo, r"(double buffer|double_buffer|ping_pong|stage\[\s*2\s*\])", ["*.cu", "*.cuh", "*.cc", "*.cpp", "*.h", "*.hpp"])
    if dbl > 0:
        tags.append("double_buffer")

    # Dedup while preserving first-seen ordering
    def _dedup(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return _dedup(tags), _dedup(constraints)


def _infer_cpu_opt_tags(repo: Path) -> List[str]:
    tags: List[str] = []
    if _rg_count(repo, r"(arm_neon\.h|__ARM_NEON|\bneon\b)", ["*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.S", "*.s", "*.asm"]) > 0:
        tags.append("neon_vectorization")
    if _rg_count(repo, r"(\bsve\b)", ["*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.S", "*.s", "*.asm"]) > 0:
        tags.append("sve_vectorization")
    if _rg_count(repo, r"(prefetch|__builtin_prefetch)", ["*.c", "*.cc", "*.cpp", "*.h", "*.hpp"]) > 0:
        tags.append("prefetch")
    if _rg_count(repo, r"(cache|L1|L2|blocking|block)", ["*.c", "*.cc", "*.cpp", "*.h", "*.hpp"]) > 0:
        tags.append("cache_blocking")
    if _rg_count(repo, r"(asm|__asm__)", ["*.c", "*.cc", "*.S", "*.s", "*.asm"]) > 0:
        tags.append("handwritten_asm")
    # keep it coarse; patterns folder carries the detailed breakdown
    return list(dict.fromkeys(tags))

def _rg_files(cwd: Path, pattern: str, globs: List[str], limit: int = 16) -> List[str]:
    if not _have_rg():
        return []
    cmd = ["rg", "-l", "-S", "--no-heading", "--color", "never"]
    for g in globs:
        cmd += ["-g", g]
    cmd.append(pattern)
    cp = _run(cmd, cwd=cwd)
    if cp.returncode not in (0, 1):
        return []
    files = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
    files.sort()
    return files[:limit]


def _rg_count_in_files(cwd: Path, pattern: str, files: List[str]) -> int:
    if not _have_rg() or not files:
        return 0
    # Keep the file list small (we cap it upstream) to avoid CLI arg explosion.
    cmd = ["rg", "-c", "-S", "--no-heading", "--color", "never", pattern, "--"]
    cmd += files
    cp = _run(cmd, cwd=cwd)
    if cp.returncode not in (0, 1):
        return 0
    total = 0
    for line in cp.stdout.splitlines():
        # format: path:count
        try:
            _, cnt = line.rsplit(":", 1)
            total += int(cnt.strip())
        except Exception:
            continue
    return total


_OPS: List[OpSpec] = [
    OpSpec(
        name="gemm",
        detect={
            "gpu": r"(\bgemm\b|\bmatmul\b|\bbmm\b|sgemm|dgemm|bgemm|cublasLt|cutlass)",
            "cpu": r"(\bgemm\b|\bmatmul\b|\bsgemm\b|\bdgemm\b|microkernel|ukernel)",
        },
        min_files=1,
    ),
    OpSpec(
        name="conv",
        detect={
            "gpu": r"(conv2d|convolution|depthwise|winograd|im2col|deconvolution|conv1x1|conv3x3)",
            "cpu": r"(conv2d|convolution|depthwise|winograd|im2col|deconvolution|conv1x1|conv3x3)",
        },
        min_files=2,
    ),
    OpSpec(
        name="softmax",
        detect={
            "gpu": r"(softmax|log_softmax|logSoftMax)",
            "cpu": r"(softmax|log_softmax|logSoftMax)",
        },
        min_files=1,
    ),
    OpSpec(
        name="elementwise",
        detect={
            "gpu": r"(eltwise|elementwise|binaryop|unaryop|BinaryOp|UnaryOp|broadcast)",
            "cpu": r"(eltwise|elementwise|binaryop|unaryop|BinaryOp|UnaryOp|broadcast)",
        },
        min_files=3,
    ),
    OpSpec(
        name="activation",
        detect={
            "gpu": r"(relu|gelu|silu|swish|tanh|sigmoid|leakyrelu|prelu)",
            "cpu": r"(relu|gelu|silu|swish|tanh|sigmoid|leakyrelu|prelu)",
        },
        min_files=2,
    ),
    OpSpec(
        name="norm",
        detect={
            "gpu": r"(layernorm|LayerNorm|rmsnorm|RMSNorm|groupnorm|GroupNorm|batchnorm|BatchNorm|instancenorm|InstanceNorm)",
            "cpu": r"(layernorm|LayerNorm|rmsnorm|RMSNorm|groupnorm|GroupNorm|batchnorm|BatchNorm|instancenorm|InstanceNorm)",
        },
        min_files=1,
    ),
    OpSpec(
        name="embedding",
        detect={
            "gpu": r"(embedding|Embedding|gather|Gather|lookup|index_select|take\b)",
            "cpu": r"(embedding|Embedding|gather|Gather|lookup|index_select|take\b)",
        },
        min_files=1,
    ),
    OpSpec(
        name="ffn",
        detect={
            "gpu": r"(ffn|feedforward|mlp|MLP|fused_mlp|fused_ffn|swiglu|SwiGLU|geglu|GEGLU)",
            "cpu": r"(ffn|feedforward|mlp|MLP|swiglu|SwiGLU|geglu|GEGLU)",
        },
        min_files=1,
    ),
    OpSpec(
        name="attention",
        detect={
            "gpu": r"(attention|Attention|flash|Flash|paged[_ ]kv|kv[_ ]cache|kv_cache)",
            "cpu": r"(attention|Attention)",
        },
        min_files=1,
    ),
    OpSpec(
        name="rope",
        detect={
            "gpu": r"(rotary|Rotary|rope|RoPE)",
            "cpu": r"(rotary|Rotary|rope|RoPE)",
        },
        min_files=1,
    ),
    OpSpec(
        name="transpose",
        detect={
            "gpu": r"(transpose|permute|layout[_ ]transform|reorder)",
            "cpu": r"(transpose|permute|layout[_ ]transform|reorder)",
        },
        min_files=2,
    ),
    OpSpec(
        name="quantization",
        detect={
            "gpu": r"(quant|dequant|Quantize|Dequant|int8|int4|fp8)",
            "cpu": r"(quant|dequant|Quantize|Dequant|int8|int4)",
        },
        min_files=2,
    ),
    OpSpec(
        name="reduce",
        detect={
            "gpu": r"(\breduce\b|reduction|cub::BlockReduce|cub::DeviceReduce|BlockReduce|DeviceReduce|thrust::reduce)",
            "cpu": r"(\breduce\b|reduction|reduce_sum|reduce_max|reduce_min|ReduceSum|ReduceMax|ReduceMin)",
        },
        min_files=1,
    ),
    OpSpec(
        name="argmax",
        detect={
            "gpu": r"(ArgMax|ArgMin|argmax|argmin)",
            "cpu": r"(ArgMax|ArgMin|argmax|argmin)",
        },
        min_files=1,
    ),
    OpSpec(
        name="topk",
        detect={
            "gpu": r"(topk|TopK|TopKV2|top_k)",
            "cpu": r"(topk|TopK|TopKV2|top_k|nth_element|partial_sort)",
        },
        min_files=1,
    ),
    OpSpec(
        name="sampling",
        detect={
            # Intentionally do NOT match on `curand`/`philox` alone to avoid
            # classifying generic RNG kernels as "token sampling".
            "gpu": r"(sampling|Sampler|top_p|topp|top-p|nucleus|multinomial)",
            "cpu": r"(sampling|Sampler|top_p|topp|top-p|nucleus|multinomial)",
        },
        min_files=1,
    ),
    OpSpec(
        name="kv_cache",
        detect={
            "gpu": r"(kv[_ ]cache|kv_cache|save_kv_cache|paged[_ ]kv|kv_cache_loc|slot_mapping)",
        },
        min_files=1,
    ),
    OpSpec(
        name="paged_attention",
        detect={
            "gpu": r"(paged_attention|PagedAttention|page_table|block_table|paged[_ ]kv)",
        },
        min_files=1,
    ),
]


def _scan_ops(repo: Path, *, sig: RepoSignals) -> List[OpScan]:
    ops: List[OpScan] = []
    gpu_cuda_globs = ["*.cu", "*.cuh"]
    gpu_glue_globs = ["*.cc", "*.cpp", "*.h", "*.hpp"]
    cpu_globs = ["*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.S", "*.s", "*.asm"]

    # Target-specific optimization tags, counted within op-matching files.
    gpu_tag_patterns = {
        "tensor_core": r"(wmma|mma\.sync|ldmatrix|nvcuda::wmma)",
        "async_copy_pipeline": r"(cp\.async|cuda::memcpy_async|pipeline)",
        "shared_memory_tiling": r"(__shared__|shared memory)",
        "vectorized_load_store": r"(half2|__half2|float4|int4|reinterpret_cast<[^>]*4>)",
        "warp_reduce": r"(__shfl_|cooperative_groups::)",
        "double_buffer": r"(double buffer|double_buffer|ping_pong|stage\[\s*2\s*\])",
    }
    cpu_tag_patterns = {
        "neon_vectorization": r"(arm_neon\.h|__ARM_NEON|\bneon\b)",
        "sve_vectorization": r"(\bsve\b)",
        "prefetch": r"(prefetch|__builtin_prefetch)",
        "cache_blocking": r"(cache|L1|L2|blocking|block)",
        "handwritten_asm": r"(asm|__asm__|\.macro)",
        "winograd": r"(winograd)",
        "im2col": r"(im2col)",
    }

    for spec in _OPS:
        if sig.gpu_relevant and "gpu" in spec.detect:
            pat = spec.detect["gpu"]
            cuda_files = _rg_files(repo, pat, gpu_cuda_globs, limit=24)
            # Only treat as "GPU op" if we have at least one CUDA source that matches.
            if cuda_files:
                glue_files = _rg_files(repo, pat, gpu_glue_globs, limit=24)
                files = cuda_files + [f for f in glue_files if f not in cuda_files]
                hits = _rg_count(repo, pat, gpu_cuda_globs) + _rg_count(repo, pat, gpu_glue_globs)
                tag_counts = {k: _rg_count_in_files(repo, v, cuda_files) for k, v in gpu_tag_patterns.items()}
                ops.append(OpScan(name=spec.name, target="gpu", hits=hits, files=files[:24], opt_tag_counts=tag_counts))
        if sig.cpu_arm_relevant and "cpu" in spec.detect:
            pat = spec.detect["cpu"]
            files = _rg_files(repo, pat, cpu_globs, limit=24)
            if len(files) >= spec.min_files:
                hits = _rg_count(repo, pat, cpu_globs)
                tag_counts = {k: _rg_count_in_files(repo, v, files) for k, v in cpu_tag_patterns.items()}
                ops.append(OpScan(name=spec.name, target="cpu", hits=hits, files=files, opt_tag_counts=tag_counts))

    # Dedup (name, target) keeping higher hit variant (shouldn't happen often)
    by_key: Dict[Tuple[str, str], OpScan] = {}
    for o in ops:
        k = (o.name, o.target)
        if k not in by_key or o.hits > by_key[k].hits:
            by_key[k] = o
    out = list(by_key.values())
    out.sort(key=lambda x: (x.target, x.name))
    return out


def _applicable_shapes(op: str, platform: str, opt_tags: List[str]) -> List[str]:
    shapes: List[str] = []
    if op == "gemm" and platform == "nvidia":
        if "tensor_core" in opt_tags:
            shapes += ["large_square", "tall_skinny", "multiples_of_8_16"]
        else:
            shapes += ["large_square", "tall_skinny"]
        if "shared_memory_tiling" in opt_tags:
            shapes.append("large_K")
        if "vectorized_load_store" in opt_tags:
            shapes.append("aligned_inputs")
    if op == "conv" and platform == "arm":
        shapes += ["small_batch", "mobile_shapes"]
    if op in ("attention", "softmax", "norm") and platform == "nvidia":
        shapes += ["large_batch_seq", "long_context"]
    # Dedup
    return list(dict.fromkeys(shapes))


def _render_kernel_block(
    *,
    op_type: str,
    backend: str,
    source_repo: str,
    platform: str,
    dtype: List[str],
    layout: List[str],
    optimization_tags: List[str],
    applicable_shapes: List[str],
    constraints: List[str],
    correctness_test: bool,
    benchmark_present: bool,
    production_grade: str,
    notes: List[str],
) -> str:
    # YAML-like, but kept tolerant and human-readable.
    def _list(xs: List[str]) -> str:
        return "[" + ", ".join(xs) + "]"

    lines: List[str] = []
    lines.append(f"op_type: {op_type}")
    lines.append(f"backend: {backend}")
    lines.append(f"source_repo: {source_repo}")
    lines.append(f"platform: {platform}")
    lines.append(f"dtype: {_list(dtype) if dtype else '[]'}")
    lines.append(f"layout: {_list(layout) if layout else '[]'}")
    lines.append("optimization_tags:")
    for t in optimization_tags:
        lines.append(f"  - {t}")
    if not optimization_tags:
        lines.append("  - (unknown)")
    lines.append("applicable_shapes:")
    for s in applicable_shapes:
        lines.append(f"  - {s}")
    if not applicable_shapes:
        lines.append("  - (unknown)")
    lines.append("constraints:")
    for c in constraints:
        lines.append(f"  - {c}")
    if not constraints:
        lines.append("  - (unknown)")
    lines.append("quality_signals:")
    lines.append(f"  correctness_test: {_bool_to_yesno(correctness_test)}")
    lines.append(f"  benchmark_present: {_bool_to_yesno(benchmark_present)}")
    lines.append(f"production_grade: {production_grade}")
    lines.append("notes:")
    for n in notes:
        lines.append(f"  - {n}")
    if not notes:
        lines.append("  - (none)")
    return "\n".join(lines)


def _patterns_templates() -> List[Tuple[str, str]]:
    # (relative_path, content)
    return [
        (
            "patterns/gpu/gemm/block_tiling.md",
            """# Block Tiling (GPU GEMM)

**Intent**: Improve data reuse by loading A/B tiles into shared memory and computing a C tile per CTA.

**When It Works**
- Medium/large GEMM sizes with sufficient arithmetic intensity.
- Input alignment allows vectorized loads.

**Recognition Signals (Code)**
- `__shared__` buffers for A/B tiles.
- Outer loops over K with `tile_k` and `__syncthreads()`.
- Per-thread fragments accumulate into registers.

**Tradeoffs / Failure Modes**
- Small-K or tiny matrices: shared memory overhead dominates.
- Bank conflicts and poor memory coalescing can erase gains.

**Agent Notes**
- Pair with `vectorized_load_store` and `double_buffer` when K is large.
""",
        ),
        (
            "patterns/gpu/gemm/warp_tiling.md",
            """# Warp Tiling (GPU GEMM)

**Intent**: Map a tile of C to each warp to improve locality, reduce synchronization, and control register usage.

**When It Works**
- Large GEMM where a warp-level compute tile balances occupancy and reuse.
- Useful when block tiling alone hits register or shared memory limits.

**Recognition Signals (Code)**
- Warp-level indexing: `warp_id`, `lane_id`, `threadIdx.x / 32`.
- Each warp loads / computes its own sub-tiles.
- Warp shuffles (`__shfl_*`) for reductions or fragment exchange.

**Tradeoffs / Failure Modes**
- Too-large warp tiles inflate registers and reduce occupancy.
- Too-small warp tiles underutilize memory bandwidth.
""",
        ),
        (
            "patterns/gpu/gemm/tensor_core.md",
            """# Tensor Core (GPU GEMM)

**Intent**: Use Tensor Cores (WMMA / MMA) to increase throughput for FP16/BF16/INT8-like GEMMs.

**When It Works**
- Shapes compatible with MMA tile sizes (often multiples of 8/16).
- Data layout and alignment can be arranged to feed fragments efficiently.

**Recognition Signals (Code)**
- Tokens like `wmma`, `mma.sync`, `ldmatrix`.
- Fragment-based compute and epilogue scaling.
- Possible use of CUTLASS/CUTE abstractions.

**Constraints**
- Architecture-dependent (often `sm70+`; advanced pipelines `sm80+`).
- Layout constraints are common; expect padding / swizzle.

**Tradeoffs**
- Complex code; fragile performance if alignment/layout not met.
""",
        ),
        (
            "patterns/gpu/gemm/double_buffer.md",
            """# Double Buffer (GPU GEMM)

**Intent**: Overlap global memory loads with compute by ping-ponging shared memory stages.

**When It Works**
- Large-K loops where load latency is significant.
- Often paired with `cp.async` (Ampere+) or manual prefetch.

**Recognition Signals (Code)**
- Two shared buffers or stage index `% 2` (or more stages).
- Load stage `n+1` while computing stage `n`.
- `cp.async` / pipeline primitives on sm80+.

**Tradeoffs / Failure Modes**
- Increases shared memory footprint.
- Harder correctness and synchronization; can regress for small workloads.
""",
        ),
        (
            "patterns/gpu/softmax/online_softmax.md",
            """# Online Softmax (GPU)

**Intent**: Compute `softmax(x)` in a numerically stable way while streaming through data once (or in few passes),
typically using warp/block reductions for `max` and `sum(exp(x - max))`.

**When It Works**
- Long vectors (e.g., attention scores) where memory bandwidth dominates.
- When fusing with surrounding ops (masking, scaling) reduces memory traffic.

**Recognition Signals (Code)**
- Two-stage reduction: compute `max`, then accumulate `exp(x - max)`.
- Warp/block reductions via shuffles or cooperative groups.
- Vectorized loads (`half2`, `float4`) and contiguous memory access.

**Tradeoffs**
- Must handle tails and masking carefully to avoid numerical issues.
- Register pressure can limit occupancy for very wide tiles.
""",
        ),
        (
            "patterns/gpu/norm/welford_layernorm.md",
            """# Welford LayerNorm (GPU)

**Intent**: Use Welford's algorithm to compute mean/variance stably, then normalize and apply affine transform.

**When It Works**
- Large hidden sizes where reduction dominates.
- When fusing bias/residual/dropout reduces memory traffic.

**Recognition Signals (Code)**
- Welford combine steps across threads/warps.
- Separate passes for stats and normalization, or fused when possible.
- Warp reductions (`__shfl_*`) and vectorized loads.

**Tradeoffs**
- Fused variants are complex; numerical parity needs explicit tests.
""",
        ),
        (
            "patterns/gpu/attention/flash_attention.md",
            """# Flash Attention Style (GPU)

**Intent**: Tile attention computation to avoid materializing the full attention matrix, improving memory efficiency.

**When It Works**
- Long context, where `QK^T` is too large to store.
- FP16/BF16 with Tensor Cores and careful accumulation.

**Recognition Signals (Code)**
- Blocking over sequence dimension; compute tiles of `QK` and `PV`.
- Online softmax within tiles (keep running max/sum).
- Heavy use of shared memory and/or `cp.async` pipelines.

**Tradeoffs**
- Complex masking (causal/padding) handling.
- Sensitive to tile sizes and head dimension.
""",
        ),
        (
            "patterns/gpu/elementwise/vectorized_elementwise.md",
            """# Vectorized Elementwise (GPU)

**Intent**: Maximize memory throughput for elementwise ops via vectorized loads/stores and fusion.

**When It Works**
- Large contiguous tensors with simple pointwise math.
- When multiple elementwise ops can be fused into one kernel.

**Recognition Signals (Code)**
- `half2` / `float4` loads, aligned pointers, stride-1 access.
- Minimal branching; predication for tails.

**Tradeoffs**
- Misalignment and non-contiguous layouts degrade quickly.
""",
        ),
        (
            "patterns/gpu/ffn/fused_mlp.md",
            """# Fused MLP/FFN (GPU)

**Intent**: Fuse GEMM + activation (GELU/SwiGLU) + GEMM (and possibly bias) to reduce memory traffic.

**When It Works**
- Transformer FFN blocks with large batch*seq and hidden sizes.
- When epilogue fusion can be expressed efficiently (CUTLASS-like).

**Recognition Signals (Code)**
- Two GEMMs with an activation in between, possibly fused epilogues.
- Tensor Core usage and tiled pipelines.

**Tradeoffs**
- Harder scheduling and more intermediate precision choices.
""",
        ),
        (
            "patterns/gpu/embedding/gather.md",
            """# Gather/Embedding (GPU)

**Intent**: Optimize embedding lookup / gather by improving memory coalescing and caching behavior.

**When It Works**
- Large embedding tables with many lookups.
- When indices have locality (reuse within a block).

**Recognition Signals (Code)**
- Loads indexed rows; may use shared memory for indices or staging.
- Use of read-only cache (`__ldg`-like patterns) and vectorized loads.

**Tradeoffs**
- Random indices are bandwidth-bound; caching may not help.
""",
        ),
        (
            "patterns/gpu/transpose/transpose_tiling.md",
            """# Tiled Transpose (GPU)

**Intent**: Use shared memory tiling to transpose matrices/tensors while preserving coalesced accesses.

**When It Works**
- 2D/3D transposes where naive global loads/stores are uncoalesced.

**Recognition Signals (Code)**
- Shared-memory tile with padding to avoid bank conflicts.
- Two-phase load/store with swapped indices.

**Tradeoffs**
- Overhead can dominate for small tensors.
""",
        ),
        (
            "patterns/gpu/quantization/int8_dequant_epilogue.md",
            """# INT8 Dequant / Epilogue Fusion (GPU)

**Intent**: Fuse dequantization (scale/zero-point) with GEMM epilogue or elementwise to reduce memory traffic.

**When It Works**
- INT8 GEMM pipelines where output is immediately consumed by another op.

**Recognition Signals (Code)**
- Per-channel/per-tensor scales applied in epilogue.
- Vectorized int8 loads, accumulation to int32/float.

**Tradeoffs**
- Scale precision and rounding behavior must match reference.
""",
        ),
        (
            "patterns/gpu/reduce/block_reduce.md",
            """# Block Reduce (GPU)

**Intent**: Efficiently reduce values within a block (sum/max/min/argmax-like) using warp-level primitives and shared memory.

**When It Works**
- Reductions over moderately large contiguous dimensions.
- As a building block for softmax, layernorm, topk, and attention.

**Recognition Signals (Code)**
- CUB: `cub::BlockReduce`, `cub::DeviceReduce`
- Warp shuffles + shared memory staging
- Two-level reduction: per-warp then across warps

**Tradeoffs**
- Register pressure and shared memory usage can cap occupancy.
- For tiny reductions, overhead dominates.
""",
        ),
        (
            "patterns/gpu/topk/radix_select.md",
            """# TopK / Selection (GPU)

**Intent**: Find top-k elements without fully sorting, using selection networks, partial sorts, or radix-based selection.

**When It Works**
- Small `k` (e.g., 1..128) relative to vocabulary/length.
- Logits/topk in decoding loops where latency matters.

**Recognition Signals (Code)**
- Bitonic / sorting networks, `nth_element`-like logic
- Shared memory heaps or per-thread candidate lists
- Use of warp-level primitives to merge candidates

**Tradeoffs**
- Large vocab topk becomes memory-bound; consider two-stage (block candidates + final merge).
- Numerical ties and stable ordering requirements complicate correctness.
""",
        ),
        (
            "patterns/gpu/sampling/topk_topp.md",
            """# Sampling (Top-k / Top-p) (GPU)

**Intent**: Sample next tokens from logits efficiently, often combining top-k/top-p filtering with RNG.

**When It Works**
- Decoding loops with many steps; kernel launch overhead and memory traffic matter.

**Recognition Signals (Code)**
- RNG: `curand`, Philox counters, uniform samples
- Prefix-sum / cumulative probability for top-p
- Integration with topk selection

**Tradeoffs**
- Correct RNG stream management is critical (reproducibility vs speed).
- Numerical stability: softmax temperature and exp overflow.
""",
        ),
        (
            "patterns/gpu/kv_cache/layout.md",
            """# KV Cache Layout (GPU)

**Intent**: Store and access K/V cache with a layout that enables coalesced loads in attention decode/prefill.

**When It Works**
- Decode: repeated small queries over large cached keys/values.
- Prefill: bulk writes; alignment and strides are important.

**Recognition Signals (Code)**
- Tokens: `kv_cache`, `paged_kv`, `slot_mapping`, `kv_cache_loc`
- Indirection via `page_table`/`block_table`
- Stride math for `[seq, head, dim]` or blocked layouts

**Tradeoffs**
- Indirection improves memory usage but adds pointer chasing.
- Layout choices affect which dimension is contiguous (head vs dim vs seq).
""",
        ),
        (
            "patterns/gpu/paged_attention/paged_attention.md",
            """# Paged Attention / Blocked KV (GPU)

**Intent**: Use paged/block tables to index KV cache blocks, reducing fragmentation and enabling efficient decode.

**When It Works**
- Variable-length sequences and long contexts.
- Systems with dynamic allocation and reuse of KV blocks.

**Recognition Signals (Code)**
- Tokens: `paged_attention`, `page_table`, `block_table`, `paged_kv`
- Gather/scatter from KV blocks, often with head-dim vectorization

**Tradeoffs**
- Indirection can dominate if not cached or coalesced.
- Requires careful bounds/mask handling for partially filled blocks.
""",
        ),
        (
            "patterns/cpu/conv/direct_conv_3x3.md",
            """# Direct Conv 3x3 (CPU ARM)

**Intent**: Compute 3x3 convolution directly (no transform) with tight inner loops and NEON/SVE vectorization.

**When It Works**
- Small kernels (3x3) and typical mobile batch sizes.
- Depthwise or standard conv depending on layout.

**Recognition Signals (Code)**
- Explicit 3x3 unrolled multiply-adds.
- Intrinsics: `<arm_neon.h>` / NEON types.
- Careful input pointer arithmetic, possible prefetch.

**Tradeoffs**
- Less flexible than im2col; harder to generalize.
- Sensitive to layout (NHWC vs NCHW) and channel blocking.
""",
        ),
        (
            "patterns/cpu/conv/winograd.md",
            """# Winograd (CPU ARM)

**Intent**: Use Winograd transforms to reduce multiply count for small conv kernels (commonly 3x3).

**When It Works**
- 3x3 conv with sufficient spatial size.
- FP16/FP32 depending on numerical tolerance.

**Recognition Signals (Code)**
- Keywords: `winograd`, transform matrices, `Bt * d * B`, `G * g * Gt`.
- Separate transform + GEMM-like multiply + inverse transform.

**Tradeoffs**
- Extra transforms can dominate for small shapes.
- Numerical stability can be worse than direct conv.
""",
        ),
        (
            "patterns/cpu/conv/cache_blocking.md",
            """# Cache Blocking (CPU ARM)

**Intent**: Improve cache locality by blocking loops over channels/spatial tiles and reusing weights/inputs from cache.

**When It Works**
- Larger convolutions / GEMM-like paths (im2col+GEMM).
- When memory bandwidth is the limiter.

**Recognition Signals (Code)**
- Outer loops with block sizes (e.g., `ic_block`, `oc_block`, `tile_h`, `tile_w`).
- Explicit packing of weights/activations.
- Comments or variables referencing L1/L2.

**Tradeoffs**
- Wrong block sizes can regress due to TLB/cache thrash.
""",
        ),
        (
            "patterns/cpu/conv/xsimd_vectorization.md",
            """# SIMD Vectorization (CPU ARM)

**Intent**: Use SIMD (NEON/SVE or xsimd-like abstraction) to widen inner loops and increase throughput.

**When It Works**
- Regular inner loops with contiguous memory.
- Data can be packed/blocked to match vector width.

**Recognition Signals (Code)**
- NEON intrinsics, or vector abstraction layers.
- Accumulation into vector registers; horizontal reductions.
- Handling tails (remainder) carefully.

**Tradeoffs**
- Tail handling and alignment complicate code.
- Over-vectorization can reduce frequency/occupancy on some cores.
""",
        ),
        (
            "patterns/cpu/gemm/microkernel.md",
            """# Microkernel GEMM (CPU ARM)

**Intent**: Hand-tuned inner kernels (often in asm) that compute a small MRxNR tile efficiently.

**When It Works**
- Large GEMMs where packing + microkernel amortize overhead.
- Consistent shapes or batched workloads.

**Recognition Signals (Code)**
- Files named `ukernel`/`microkernel`, lots of NEON/SVE/ASM.
- Packing routines for A/B panels.
- Kernel loops unrolled around vector FMA instructions.

**Tradeoffs**
- Many specialized kernels for different shapes/dtypes.
- Hard to maintain; correctness requires extensive tests.
""",
        ),
        (
            "patterns/cpu/softmax/stable_softmax.md",
            """# Stable Softmax (CPU ARM)

**Intent**: Compute softmax stably using `max` subtraction and SIMD reductions.

**When It Works**
- Medium/long vectors (e.g., logits) where vectorization helps.
- When multi-threading across batches is available.

**Recognition Signals (Code)**
- Pass 1: reduce max; pass 2: exp + sum; pass 3: normalize.
- NEON/SVE vector exp approximations or lookup tables.

**Tradeoffs**
- Accurate exp approximations can be expensive.
""",
        ),
        (
            "patterns/cpu/elementwise/vectorized_elementwise.md",
            """# Vectorized Elementwise (CPU ARM)

**Intent**: Use NEON/SVE (or xsimd abstraction) to process multiple elements per iteration.

**When It Works**
- Contiguous tensors; simple pointwise ops.
- Fusion at a higher level (operator fusion) improves further.

**Recognition Signals (Code)**
- xsimd/NEON intrinsics, loop unrolling, aligned loads.
- Tail handling logic.
""",
        ),
        (
            "patterns/cpu/norm/layernorm.md",
            """# LayerNorm (CPU ARM)

**Intent**: Efficient mean/variance computation and normalization with SIMD reductions.

**When It Works**
- Large hidden dimensions; batch*seq parallelism.

**Recognition Signals (Code)**
- Reduce mean/var; vectorized affine transform.
- Welford or two-pass methods.
""",
        ),
        (
            "patterns/cpu/quantization/int8_quant.md",
            """# INT8 Quant/Dequant (CPU ARM)

**Intent**: Efficient quantization/dequantization and int8 dot-products with NEON/SVE.

**When It Works**
- Edge/mobile inference pipelines with int8 weights/activations.

**Recognition Signals (Code)**
- `int8_t` paths, per-channel scales, saturation/rounding.
- Dot-product instructions or widened accumulators.
""",
        ),
        (
            "patterns/cpu/reduce/simd_reduce.md",
            """# SIMD Reduce (CPU ARM)

**Intent**: Use NEON/SVE to accelerate reductions (sum/max/min/argmax-like) over contiguous arrays.

**When It Works**
- Large contiguous reductions where memory access is predictable.

**Recognition Signals (Code)**
- Vector loads and horizontal reductions
- Two-stage reduce: SIMD lanes then scalar tail

**Tradeoffs**
- Argmax requires tracking indices; increases register pressure.
""",
        ),
        (
            "patterns/cpu/topk/partial_sort.md",
            """# TopK / Partial Sort (CPU ARM)

**Intent**: Compute top-k using partial sorting (`nth_element`/heap) without full sort.

**When It Works**
- Small `k` relative to length/vocab.

**Recognition Signals (Code)**
- Heap maintenance or `nth_element` calls
- Two-stage: block candidates then merge

**Tradeoffs**
- Branching-heavy; SIMD helps less than cache locality.
""",
        ),
        (
            "patterns/cpu/sampling/topk_topp.md",
            """# Sampling (Top-k / Top-p) (CPU ARM)

**Intent**: Efficient sampling from logits on CPU, often combining top-k/top-p filtering with RNG.

**When It Works**
- CPU-only inference or small batch decoding.

**Recognition Signals (Code)**
- RNG and cumulative probability logic
- Partial sort/select for top-k

**Tradeoffs**
- Typically memory/branch bound; batching improves throughput.
""",
        ),
    ]


def _render_readme_agent(
    scan: RepoScan,
    *,
    op_index: Dict[Tuple[str, str], List[str]],
) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d")

    repo_name = scan.name
    repo = scan.repo
    sig = scan.sig
    style = scan.style
    dtypes = scan.dtypes
    layouts = scan.layouts
    production_grade = scan.production_grade

    gpu_opt_tags, gpu_constraints = _infer_gpu_opt_tags(repo)
    cpu_opt_tags = _infer_cpu_opt_tags(repo)

    ops = scan.ops
    gpu_ops = [o for o in ops if o.target == "gpu"]
    cpu_ops = [o for o in ops if o.target == "cpu"]

    def _ref_id(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()

    ref_defs: Dict[str, str] = {}

    lines: List[str] = []
    lines.append(f"# {repo_name} - README.agent.md")
    lines.append("")
    lines.append(
        "**AUTO-GENERATED** (heuristic static scan). If you edit this file manually, re-run the generator with `--overwrite-readme` to refresh it (a timestamped backup will be created)."
    )
    lines.append("")
    lines.append("## Project Summary")
    lines.append(f"- source_repo: `{repo_name}`")
    lines.append(f"- scanned_at: `{now}`")
    lines.append("- scope_for_agent:")
    lines.append("  - gpu: `nvidia` + `cuda` only")
    lines.append("  - cpu: `arm` only (NEON/SVE/ASM)")
    if scan.desc:
        lines.append(f"- upstream_description: {scan.desc}")
    else:
        lines.append("- upstream_description: (not found)")
    lines.append("")
    lines.append("## Quick Signals")
    lines.append(f"- cu_files: `{sig.cu_files}`")
    lines.append(f"- asm_files: `{sig.asm_files}`")
    lines.append(f"- tests_present: `{_bool_to_yesno(sig.tests_present)}`")
    lines.append(f"- benchmark_present: `{_bool_to_yesno(sig.bench_present)}`")
    lines.append(f"- ci_present: `{_bool_to_yesno(sig.ci_present)}`")
    lines.append(f"- style_inference: `{style}`")
    lines.append("")
    lines.append("## Kernel Inventory (Agent-Facing)")
    if not gpu_ops and not cpu_ops:
        lines.append("- (No obvious kernel keywords found. Repo may still contain kernels; consider adding manual metadata.)")
    else:
        if gpu_ops:
            lines.append("### GPU (NVIDIA CUDA)")
            for o in gpu_ops:
                local_ref = f"op_{_ref_id(repo_name)}_{o.target}_{o.name}"
                ref_defs[local_ref] = f"ops/{o.target}/{o.name}.md"
                line = f"- [{o.name}][{local_ref}]"
                others = [r for r in op_index.get((o.target, o.name), []) if r != repo_name]
                if others:
                    see_parts: List[str] = []
                    for other in others:
                        see_ref = f"see_{_ref_id(other)}_{o.target}_{o.name}"
                        ref_defs[see_ref] = f"../{other}/ops/{o.target}/{o.name}.md"
                        see_parts.append(f"[{other}][{see_ref}]")
                    line += " (see also: " + ", ".join(see_parts) + ")"
                lines.append(line)
        if cpu_ops:
            lines.append("### CPU (ARM)")
            for o in cpu_ops:
                local_ref = f"op_{_ref_id(repo_name)}_{o.target}_{o.name}"
                ref_defs[local_ref] = f"ops/{o.target}/{o.name}.md"
                line = f"- [{o.name}][{local_ref}]"
                others = [r for r in op_index.get((o.target, o.name), []) if r != repo_name]
                if others:
                    see_parts = []
                    for other in others:
                        see_ref = f"see_{_ref_id(other)}_{o.target}_{o.name}"
                        ref_defs[see_ref] = f"../{other}/ops/{o.target}/{o.name}.md"
                        see_parts.append(f"[{other}][{see_ref}]")
                    line += " (see also: " + ", ".join(see_parts) + ")"
                lines.append(line)
    lines.append("")
    lines.append("## Kernel Details (Heuristic Metadata)")
    lines.append("Notes:")
    lines.append("- `dtype/layout/constraints` are inferred from token presence and may be incomplete.")
    lines.append("- Treat missing fields as \"unknown\" instead of assuming defaults.")
    lines.append("")

    # Keep a small YAML summary per op/target. This is intentionally redundant with ops/*.md pages.
    for o in ops:
        if o.target == "gpu":
            backend = "cuda"
            platform = "nvidia"
            opt_tags = gpu_opt_tags[:]  # copy
            constraints = gpu_constraints[:]
            notes = [
                "GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.",
            ]
            if style in ("teaching_code",) or re.search(r"tutorial", repo_name, re.IGNORECASE):
                notes.append("Repo looks tutorial-like; treat performance claims as educational unless benchmarks validate.")
            block = _render_kernel_block(
                op_type=o.name,
                backend=backend,
                source_repo=repo_name,
                platform=platform,
                dtype=dtypes,
                layout=layouts,
                optimization_tags=opt_tags,
                applicable_shapes=_applicable_shapes(o.name, platform, opt_tags),
                constraints=constraints,
                correctness_test=sig.tests_present,
                benchmark_present=sig.bench_present,
                production_grade=production_grade,
                notes=notes,
            )
            lines.append(f"### {o.name} (gpu/cuda)")
            lines.append("")
            lines.append("```yaml")
            lines.append(block)
            lines.append("```")
            lines.append("")
        else:
            backend = "arm"
            platform = "arm"
            notes = [
                "CPU metadata is scoped to ARM only; x86 paths (SSE/AVX) are ignored.",
            ]
            if sig.asm_files > 0:
                notes.append("Handwritten ASM present; likely microkernel-style tuning.")
            block = _render_kernel_block(
                op_type=o.name,
                backend=backend,
                source_repo=repo_name,
                platform=platform,
                dtype=dtypes,
                layout=layouts,
                optimization_tags=cpu_opt_tags,
                applicable_shapes=_applicable_shapes(o.name, platform, cpu_opt_tags),
                constraints=["aarch64 (inferred)" if sig.cpu_arm_relevant else "(unknown)"],
                correctness_test=sig.tests_present,
                benchmark_present=sig.bench_present,
                production_grade=production_grade,
                notes=notes,
            )
            lines.append(f"### {o.name} (cpu/arm)")
            lines.append("")
            lines.append("```yaml")
            lines.append(block)
            lines.append("```")
            lines.append("")

    lines.append("## Patterns")
    lines.append("- This repo contains a `patterns/` folder generated by the doc tool.")
    lines.append("- These files are intended as a shared vocabulary for optimization-strategy search (not code search).")
    lines.append("")
    lines.append("## Out Of Scope")
    lines.append("- Any GPU backend other than NVIDIA CUDA.")
    lines.append("- Any CPU architecture other than ARM (including x86).")
    lines.append("")

    if ref_defs:
        lines.append("## Link References")
        lines.append("")
        for k in sorted(ref_defs.keys()):
            lines.append(f"[{k}]: {ref_defs[k]}")
        lines.append("")
    return "\n".join(lines)


def _find_repos(root: Path) -> List[Path]:
    """
    Find repo-like directories under code_base/.
    Not all imported code has a .git dir (e.g. snapshots), so we include all top-level dirs.
    """
    base = root / "code_base"
    repos: List[Path] = []
    if not base.exists():
        return []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        # ignore helper folders/files
        if p.name.startswith("."):
            continue
        repos.append(p)
    return sorted(repos, key=lambda p: p.name.lower())


def scan_repo(repo: Path) -> RepoScan:
    repo_name = repo.name
    sig = _detect_repo_signals(repo)
    style = _infer_style(sig, repo_name, repo)
    dtypes = _infer_dtypes(repo)
    layouts = _infer_layouts(repo)

    readme = _pick_readme(repo)
    desc = _read_first_paragraph(readme) if readme else ""

    production_grade = _infer_production_grade(sig, repo_name)
    ops = _scan_ops(repo, sig=sig)

    return RepoScan(
        repo=repo,
        name=repo_name,
        desc=desc,
        sig=sig,
        style=style,
        dtypes=dtypes,
        layouts=layouts,
        production_grade=production_grade,
        ops=ops,
    )


def _render_op_page(
    scan: RepoScan,
    op: OpScan,
    *,
    op_index: Dict[Tuple[str, str], List[str]],
) -> str:
    repo_name = scan.name
    sig = scan.sig

    backend = "cuda" if op.target == "gpu" else "arm"
    platform = "nvidia" if op.target == "gpu" else "arm"

    # Relative paths from ops/{target}/{op}.md
    patterns_base = "../../patterns"
    other_repo_base = "../../../"  # to code_base/

    pattern_links: List[str] = []
    key = (op.target, op.name)
    if key == ("gpu", "gemm"):
        pattern_links = [
            f"{patterns_base}/gpu/gemm/block_tiling.md",
            f"{patterns_base}/gpu/gemm/warp_tiling.md",
            f"{patterns_base}/gpu/gemm/tensor_core.md",
            f"{patterns_base}/gpu/gemm/double_buffer.md",
        ]
    elif key == ("gpu", "softmax"):
        pattern_links = [f"{patterns_base}/gpu/softmax/online_softmax.md"]
    elif key == ("gpu", "norm"):
        pattern_links = [f"{patterns_base}/gpu/norm/welford_layernorm.md"]
    elif key == ("gpu", "attention"):
        pattern_links = [f"{patterns_base}/gpu/attention/flash_attention.md"]
    elif key == ("gpu", "elementwise"):
        pattern_links = [f"{patterns_base}/gpu/elementwise/vectorized_elementwise.md"]
    elif key == ("gpu", "ffn"):
        pattern_links = [f"{patterns_base}/gpu/ffn/fused_mlp.md"]
    elif key == ("gpu", "embedding"):
        pattern_links = [f"{patterns_base}/gpu/embedding/gather.md"]
    elif key == ("gpu", "transpose"):
        pattern_links = [f"{patterns_base}/gpu/transpose/transpose_tiling.md"]
    elif key == ("gpu", "quantization"):
        pattern_links = [f"{patterns_base}/gpu/quantization/int8_dequant_epilogue.md"]
    elif key == ("gpu", "reduce"):
        pattern_links = [f"{patterns_base}/gpu/reduce/block_reduce.md"]
    elif key == ("gpu", "argmax"):
        pattern_links = [f"{patterns_base}/gpu/reduce/block_reduce.md"]
    elif key == ("gpu", "topk"):
        pattern_links = [f"{patterns_base}/gpu/topk/radix_select.md"]
    elif key == ("gpu", "sampling"):
        pattern_links = [f"{patterns_base}/gpu/sampling/topk_topp.md"]
    elif key == ("gpu", "kv_cache"):
        pattern_links = [f"{patterns_base}/gpu/kv_cache/layout.md"]
    elif key == ("gpu", "paged_attention"):
        pattern_links = [
            f"{patterns_base}/gpu/kv_cache/layout.md",
            f"{patterns_base}/gpu/paged_attention/paged_attention.md",
        ]
    elif key == ("cpu", "conv"):
        pattern_links = [
            f"{patterns_base}/cpu/conv/direct_conv_3x3.md",
            f"{patterns_base}/cpu/conv/winograd.md",
            f"{patterns_base}/cpu/conv/cache_blocking.md",
            f"{patterns_base}/cpu/conv/xsimd_vectorization.md",
        ]
    elif key == ("cpu", "gemm"):
        pattern_links = [f"{patterns_base}/cpu/gemm/microkernel.md"]
    elif key == ("cpu", "softmax"):
        pattern_links = [f"{patterns_base}/cpu/softmax/stable_softmax.md"]
    elif key == ("cpu", "elementwise"):
        pattern_links = [f"{patterns_base}/cpu/elementwise/vectorized_elementwise.md"]
    elif key == ("cpu", "norm"):
        pattern_links = [f"{patterns_base}/cpu/norm/layernorm.md"]
    elif key == ("cpu", "quantization"):
        pattern_links = [f"{patterns_base}/cpu/quantization/int8_quant.md"]
    elif key == ("cpu", "reduce"):
        pattern_links = [f"{patterns_base}/cpu/reduce/simd_reduce.md"]
    elif key == ("cpu", "argmax"):
        pattern_links = [f"{patterns_base}/cpu/reduce/simd_reduce.md"]
    elif key == ("cpu", "topk"):
        pattern_links = [f"{patterns_base}/cpu/topk/partial_sort.md"]
    elif key == ("cpu", "sampling"):
        pattern_links = [f"{patterns_base}/cpu/sampling/topk_topp.md"]

    guide: Dict[Tuple[str, str], List[str]] = {
        ("gpu", "gemm"): [
            "Prefer block tiling + shared memory to increase A/B reuse; tune CTA tile sizes against shared/reg limits.",
            "If Tensor Core tokens are present, check alignment/layout constraints and whether epilogue fusion exists.",
            "Pipeline global->shared copies (e.g., `cp.async`) and consider double-buffering for large-K.",
        ],
        ("gpu", "softmax"): [
            "Use stable softmax: subtract max, then exp+sum, then normalize; avoid extra global passes when possible.",
            "Warp/block reductions are the core; vectorized loads help when inputs are aligned and contiguous.",
            "For attention scores, prefer online softmax within tiles (FlashAttention-style).",
        ],
        ("gpu", "norm"): [
            "Stats reduction (mean/var) is the bottleneck; use Welford or two-pass depending on numerical needs.",
            "Fuse affine transform (gamma/beta) and nearby elementwise ops to reduce memory traffic.",
        ],
        ("gpu", "attention"): [
            "Look for tiling over sequence and online softmax signals; this is where most speedups come from.",
            "KV cache layout and memory indirection often dominates for decoding; confirm access patterns.",
        ],
        ("gpu", "elementwise"): [
            "Vectorize loads/stores (half2/float4) and fuse multiple pointwise ops into one kernel when possible.",
            "Avoid divergence; handle tails with predication.",
        ],
        ("gpu", "activation"): [
            "Activation kernels are usually bandwidth-bound; prioritize fusion and vectorization.",
            "For GELU/SILU, polynomial/approx variants change accuracy; ensure tests lock behavior.",
        ],
        ("gpu", "ffn"): [
            "FFN performance is usually GEMM-bound; fusion opportunities are in epilogues (bias, activation, gating).",
            "Check whether the repo uses CUTLASS/cublasLt epilogues or custom fused kernels.",
        ],
        ("gpu", "embedding"): [
            "Embedding/gather is bandwidth-bound; seek coalesced access, caching, and reuse of indices within a block.",
        ],
        ("gpu", "transpose"): [
            "Naive transpose is uncoalesced; shared-memory tiled transpose is the standard fix.",
        ],
        ("gpu", "quantization"): [
            "Quant/dequant is often best fused into producers/consumers (e.g., GEMM epilogue).",
        ],
        ("gpu", "reduce"): [
            "Reductions are often the bottleneck; prefer warp-level reductions + one final block reduction.",
            "If available, CUB primitives are a strong baseline; custom kernels win when fused into a larger op.",
        ],
        ("gpu", "argmax"): [
            "Argmax/argmin usually require tracking (value, index) pairs; avoid divergent comparisons where possible.",
            "Two-stage designs (per-block candidates + final reduction) are common for large dimensions.",
        ],
        ("gpu", "topk"): [
            "Top-k is selection, not full sort; keep k small and use hierarchical candidate reduction.",
            "For vocab-sized topk, a two-stage approach is typical: block-local topk then merge.",
        ],
        ("gpu", "sampling"): [
            "Sampling kernels combine softmax/filtering with RNG; correctness of RNG stream/state is non-negotiable.",
            "If top-p is used, cumulative probability often implies prefix-sum style logic or sorted candidates.",
        ],
        ("gpu", "kv_cache"): [
            "KV cache performance is about layout and coalescing; confirm which dimension is contiguous and aligned.",
            "Indirection tables (`slot_mapping` / `kv_cache_loc`) add latency; prefetch and vectorize head-dim loads.",
        ],
        ("gpu", "paged_attention"): [
            "Paged attention relies on page/block tables; correctness around partially filled blocks and masks is critical.",
            "Indirection overhead can dominate; look for caching of page_table entries and contiguous KV blocks per warp.",
        ],
        ("cpu", "conv"): [
            "Direct 3x3 and Winograd are common fast paths; choose by shape and numerical tolerance.",
            "Cache blocking + packing is often the deciding factor for larger convs / im2col-like paths.",
        ],
        ("cpu", "gemm"): [
            "High performance comes from packing + microkernels (NEON/SVE/ASM); check ukernel coverage by dtype.",
        ],
        ("cpu", "softmax"): [
            "Stable softmax is typically 2-3 passes; SIMD reductions and fast exp approximations matter.",
        ],
        ("cpu", "elementwise"): [
            "Vectorize and unroll; fuse chains of elementwise ops at the graph level if possible.",
        ],
        ("cpu", "activation"): [
            "Vectorization dominates; be explicit about approximation accuracy for GELU/SILU.",
        ],
        ("cpu", "norm"): [
            "Mean/var reductions benefit from SIMD; watch cache behavior for large hidden sizes.",
        ],
        ("cpu", "embedding"): [
            "Gather is latency/bandwidth bound; batching indices and improving locality helps more than math tricks.",
        ],
        ("cpu", "quantization"): [
            "Pay attention to rounding/saturation and per-channel scale handling; SIMD dotprod paths can help a lot.",
        ],
        ("cpu", "reduce"): [
            "SIMD reductions need careful horizontal reduction and tail handling; keep memory access contiguous.",
        ],
        ("cpu", "argmax"): [
            "Track indices alongside values; reduce in vector lanes then finalize with scalar compares.",
        ],
        ("cpu", "topk"): [
            "Prefer `nth_element`/heap-based partial selection; full sort is usually overkill for small k.",
        ],
        ("cpu", "sampling"): [
            "Sampling on CPU is branch/memory heavy; batching and cache-friendly data layout matter more than math tricks.",
        ],
    }

    see_also = [r for r in op_index.get((op.target, op.name), []) if r != repo_name]
    hints: List[str] = []
    file_lc = " ".join(op.files).lower()
    if op.target == "gpu":
        if op.opt_tag_counts.get("tensor_core", 0) > 0:
            hints.append("Tensor Core tokens present in matched CUDA sources; expect alignment/layout constraints and MMA tile sizing work.")
        if op.opt_tag_counts.get("async_copy_pipeline", 0) > 0:
            hints.append("Async copy / pipeline tokens present; likely sm80+ tuning opportunities and staging/double-buffering patterns.")
        if op.name == "attention" and ("flash" in file_lc or "scaled_dot_product" in file_lc or "sdpa" in file_lc):
            hints.append("File names suggest SDPA/FlashAttention-style kernels; focus on tiling + online softmax + masking correctness.")
        if op.name == "softmax" and "logsoftmax" in file_lc:
            hints.append("LogSoftmax appears in file names/tokens; check numerical stability and overflow behavior for extreme logits.")
        if op.name == "topk" and ("topk" in file_lc or "topkv2" in file_lc):
            hints.append("TopK-related CUDA sources detected; check whether it uses per-thread candidates + shared merge or full sort networks.")
        if op.name == "sampling" and ("curand" in file_lc or "philox" in file_lc):
            hints.append("RNG tokens detected; verify determinism requirements and seeding/offset handling (Philox counter management).")
        if op.name in ("kv_cache", "paged_attention") and ("page_table" in file_lc or "block_table" in file_lc):
            hints.append("Page/block table tokens detected; likely paged KV cache or paged attention. Focus on indirection/coalescing.")
    else:
        if op.opt_tag_counts.get("handwritten_asm", 0) > 0:
            hints.append("ASM tokens present; likely microkernel-style tuning where packing and MRxNR shapes matter.")
        if op.name == "conv" and op.opt_tag_counts.get("winograd", 0) > 0:
            hints.append("Winograd tokens present; choose transform sizes carefully and validate numeric error vs direct conv.")
        if op.name == "conv" and op.opt_tag_counts.get("im2col", 0) > 0:
            hints.append("im2col tokens present; performance will depend heavily on packing and cache blocking.")

    lines: List[str] = []
    lines.append(f"# {op.name} ({op.target}/{backend})")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- source_repo: `{repo_name}`")
    lines.append(f"- platform: `{platform}`")
    lines.append(f"- backend: `{backend}`")
    lines.append(f"- style_inference: `{scan.style}`")
    lines.append(f"- production_grade: `{scan.production_grade}`")
    lines.append(f"- quality_signals: tests={_bool_to_yesno(sig.tests_present)}, bench={_bool_to_yesno(sig.bench_present)}, ci={_bool_to_yesno(sig.ci_present)}")
    lines.append(f"- inferred_dtypes: `{scan.dtypes}`")
    lines.append(f"- inferred_layouts: `{scan.layouts}`")
    lines.append(f"- detection: hits={op.hits}, files={len(op.files)}")
    lines.append("")

    lines.append("## Key Files (Entry Points)")
    if not op.files:
        lines.append("- (none captured)")
        lines.append("")
    else:
        if op.target == "gpu":
            cuda_files = [f for f in op.files if f.endswith(".cu") or f.endswith(".cuh")]
            glue_files = [f for f in op.files if f not in cuda_files]
            if cuda_files:
                lines.append("### CUDA Sources")
                for f in cuda_files[:16]:
                    lines.append(f"- `{f}`")
            if glue_files:
                lines.append("### Glue / Operator Integration")
                for f in glue_files[:16]:
                    lines.append(f"- `{f}`")
            lines.append("")
        else:
            impl_files = [f for f in op.files if not (f.endswith(".h") or f.endswith(".hpp"))]
            hdr_files = [f for f in op.files if f not in impl_files]
            if impl_files:
                lines.append("### Implementation")
                for f in impl_files[:16]:
                    lines.append(f"- `{f}`")
            if hdr_files:
                lines.append("### Headers / Interfaces")
                for f in hdr_files[:16]:
                    lines.append(f"- `{f}`")
            lines.append("")
    lines.append("")

    lines.append("## Optimization Signals (Within These Files)")
    if op.opt_tag_counts:
        for k in sorted(op.opt_tag_counts.keys()):
            lines.append(f"- {k}: `{op.opt_tag_counts[k]}`")
    else:
        lines.append("- (unknown)")
    lines.append("")

    if hints:
        lines.append("## Repo-Specific Hints (Inferred)")
        for h in hints:
            lines.append(f"- {h}")
        lines.append("")

    lines.append("## Implementation Notes (Agent-Facing)")
    for n in guide.get((op.target, op.name), ["(no curated notes yet)"]):
        lines.append(f"- {n}")
    lines.append("")

    if pattern_links:
        lines.append("## Patterns To Read")
        for p in pattern_links:
            lines.append(f"- [{p}]({p})")
        lines.append("")

    if see_also:
        lines.append("## See Also (Same Op In Other Repos)")
        for other in see_also:
            lines.append(f"- [{other}]({other_repo_base}{other}/ops/{op.target}/{op.name}.md)")
        lines.append("")

    lines.append("## Scope Notes")
    if op.target == "gpu":
        lines.append("- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).")
    else:
        lines.append("- Out of scope: non-ARM CPU backends (x86).")
    lines.append("")
    return "\n".join(lines)


def generate_for_scan(
    scan: RepoScan,
    *,
    op_index: Dict[Tuple[str, str], List[str]],
    overwrite_readme: bool,
    overwrite_ops: bool,
) -> None:
    repo = scan.repo

    # patterns skeleton (create missing only)
    for rel, body in _patterns_templates():
        _ensure_pattern_file(repo / rel, body)

    # README.agent.md
    readme_agent = repo / "README.agent.md"
    if readme_agent.exists() and not overwrite_readme:
        pass
    else:
        content = _render_readme_agent(scan, op_index=op_index)
        _safe_write(readme_agent, content, backup_if_exists=readme_agent.exists())

    # ops pages
    for o in scan.ops:
        op_path = repo / "ops" / o.target / f"{o.name}.md"
        if op_path.exists() and not overwrite_ops:
            continue
        content = _render_op_page(scan, o, op_index=op_index)
        _safe_write(op_path, content, backup_if_exists=op_path.exists())


def _render_code_base_index(scans: List[RepoScan], op_index: Dict[Tuple[str, str], List[str]]) -> str:
    """
    code_base/README.agent.md: global index to jump across repos by op.
    """
    now = _dt.datetime.now().strftime("%Y-%m-%d")

    def _ref_id(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()

    ref_defs: Dict[str, str] = {}
    lines: List[str] = []
    lines.append("# code_base - README.agent.md")
    lines.append("")
    lines.append("**AUTO-GENERATED** global operator index across `code_base/*`.")
    lines.append("")
    lines.append(f"- scanned_at: `{now}`")
    lines.append("- scope:")
    lines.append("  - gpu: nvidia + cuda only")
    lines.append("  - cpu: arm only")
    lines.append("")

    # List ops in a stable, curated order
    ordered_ops = [s.name for s in _OPS]

    def _emit_section(target: str, title: str) -> None:
        lines.append(f"## {title}")
        for op in ordered_ops:
            repos = op_index.get((target, op), [])
            if not repos:
                continue
            parts: List[str] = []
            for r in repos:
                ref = f"idx_{_ref_id(r)}_{target}_{op}"
                ref_defs[ref] = f"{r}/ops/{target}/{op}.md"
                parts.append(f"[{r}][{ref}]")
            ref_op = f"idx_op_{target}_{op}"
            # Use the first repo link as the "op" anchor; users should pick repo anyway.
            ref_defs[ref_op] = f"{repos[0]}/ops/{target}/{op}.md"
            lines.append(f"- [{op}][{ref_op}]: " + ", ".join(parts))
        lines.append("")

    _emit_section("gpu", "GPU (NVIDIA CUDA)")
    _emit_section("cpu", "CPU (ARM)")

    if ref_defs:
        lines.append("## Link References")
        lines.append("")
        for k in sorted(ref_defs.keys()):
            lines.append(f"[{k}]: {ref_defs[k]}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="project root (default: repo root)")
    ap.add_argument(
        "--overwrite-readme",
        action="store_true",
        help="overwrite README.agent.md if it already exists (always makes a timestamped backup first)",
    )
    ap.add_argument(
        "--overwrite-ops",
        action="store_true",
        help="overwrite ops/*.md if they already exist (always makes a timestamped backup first)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    repos = _find_repos(root)
    if not repos:
        print("No repos found under code_base/. Nothing to do.")
        return 0

    scans = [scan_repo(r) for r in repos]
    # Build op->repos index
    op_index: Dict[Tuple[str, str], List[str]] = {}
    for s in scans:
        for o in s.ops:
            op_index.setdefault((o.target, o.name), []).append(s.name)
    for k in list(op_index.keys()):
        op_index[k] = sorted(set(op_index[k]), key=lambda x: x.lower())

    # Per-repo generation
    for s in scans:
        generate_for_scan(
            s,
            op_index=op_index,
            overwrite_readme=args.overwrite_readme,
            overwrite_ops=args.overwrite_ops,
        )
        print(f"[ok] generated: {s.repo}/README.agent.md, patterns/, ops/")

    # Global code_base index
    code_base_index = root / "code_base" / "README.agent.md"
    _safe_write(code_base_index, _render_code_base_index(scans, op_index), backup_if_exists=code_base_index.exists())
    print(f"[ok] generated: {code_base_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
