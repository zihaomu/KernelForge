# sgl-kernel triage report

## 扫描范围
- 仓库: `code_base/sglang/sgl-kernel`
- commit: `9a697ceabb2dce6885f6a1ab8993d9980ade86c1`
- 许可证定位: `code_base/sglang/sgl-kernel/LICENSE` (Apache-2.0), `THIRDPARTYNOTICES.txt`
- 总扫描文件数: `291`

## 纳入结果
- manifest: `code_base_agent_gen/sgl-kernel/code_base/manifests/sgl-kernel.yaml`
- snippets: `code_base_agent_gen/sgl-kernel/knowledge/snippets/sgl-kernel.jsonl`
- 纳入片段数: `26`（函数级/内核级，高信号）
- 覆盖 op: `gemm`, `moe`, `attention`, `quantization`, `allreduce`, `norm`, `speculative`, `cpu_simd`
- 覆盖 backend: `cuda`, `cutlass`, `cpu(avx512/vnni)`, `rocm/hip`

## 排除类型统计（按文件分类，互斥）
- kernel_candidate: `165`
- tests: `49`
- benchmark: `28`
- python: `29`
- build_config: `14`
- docs_meta: `3`
- other: `3`

## 主要纳入内容
- CUDA/CUTLASS 核心路径: `csrc/gemm/**`, `csrc/moe/**`, `csrc/attention/**`, `csrc/elementwise/**`, `csrc/allreduce/**`, `csrc/quantization/**`
- CPU 高性能路径: `csrc/cpu/gemm*.cpp`, `csrc/cpu/norm.cpp`, `csrc/cpu/mamba/conv.cpp`
- 关键模式:
  - tensorcore/shape dispatch
  - cp_async + ldmatrix + mma
  - fused routing/topk/norm
  - quantize/dequantize 在线路径
  - custom allreduce 信号同步
  - AVX512 VNNI packing + tinygemm/brgemm

## 主要排除内容与原因
- `tests/**`, `benchmark/**`: 用于验证和性能测量，不是内核实现主体。
- `python/**`: 多为调用封装与 API glue，非 kernel 关键路径。
- `cmake/**`, `pyproject*.toml`, `setup*.py`, `build.sh`, `Makefile`: 构建脚本，非算子实现。
- `README.md`, `LICENSE`, `THIRDPARTYNOTICES.txt`: 元信息，仅保留路径用于追溯。

## 风险与不确定项
- 架构绑定风险:
  - 多数高性能内核强绑定 NVIDIA SM 代际（SM90/SM100/SM120）或 AVX512 指令集。
- 数值风险:
  - FP8/INT4/INT8 路径中 scale/zero-point 与 pack layout 轻微偏差就会导致累积误差。
- 通信风险:
  - 自定义 allreduce 的 signal/IPC 协议对驱动与运行时版本敏感。
- 待复核清单:
  - `cutlass_mla_decode`（CUDA 版本门控分支）
  - `qserve_w4a8_*` kernels（硬编码 tile 对可迁移性影响）
  - `custom_all_reduce` / `quick_all_reduce`（跨节点和多拓扑场景）
  - `gptq/awq` 内核（位宽与布局一致性）
  - CPU AVX512-only fast path（非 AVX512 环境降级策略）

## 复现说明
- 使用 `manifest` 中 include/exclude globs 与关键词可复现同等筛选集合。
- 所有片段均记录了 `source_path + symbol + 行号`，并附带 `dependency_hint` 与 `license_hint`。
