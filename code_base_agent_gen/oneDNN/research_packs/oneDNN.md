# oneDNN research_pack

## 仓库定位
- 源路径: `code_base/oneDNN`
- 提交: `61359b68ffd7b2ac9d912fecdee278f5eed3a4eb`
- 许可证位置: `LICENSE`
- 高价值片段数: `20`

## 最有价值片段（10~30）
1. `jit_brgemm_kernel` (`src/cpu/aarch64/brgemm/jit_brgemm_kernel.cpp`) [gemm/cpu,simd]
可迁移建议: 迁移 brgemm/gemm 的 blocked dispatch 与 JIT 配置分层。

2. `jit_brdgmm_kernel` (`src/cpu/x64/brgemm/jit_brdgmm_kernel.hpp`) [gemm/cpu,simd]
可迁移建议: 迁移 brgemm/gemm 的 blocked dispatch 与 JIT 配置分层。

3. `jit_uni_dw_conv_bwd_weights_kernel` (`src/cpu/x64/jit_uni_dw_conv_kernel_utils.cpp`) [quantization/cpu,simd]
可迁移建议: 迁移 int8/bf16 路径的 scale 与数据布局约定。

4. `needsKLoopReset` (`src/gpu/intel/gemm/jit/generator/pieces/gemm_setup.cxx`) [gemm/gpu]
可迁移建议: 迁移 brgemm/gemm 的 blocked dispatch 与 JIT 配置分层。

5. `matmul_amx_blocking_params_micro_t::get_copied_data_reusage_scores` (`src/cpu/x64/matmul/amx_blocking_heuristics.cpp`) [gemm/cpu,simd]
可迁移建议: 迁移 brgemm/gemm 的 blocked dispatch 与 JIT 配置分层。

6. `jit_uni_dw_conv_bwd_data_kernel` (`src/cpu/aarch64/jit_uni_dw_conv_kernel_utils.hpp`) [quantization/cpu,simd]
可迁移建议: 迁移 int8/bf16 路径的 scale 与数据布局约定。

7. `jit_brdgmm_kernel` (`src/cpu/aarch64/brgemm/jit_brdgmm_kernel.hpp`) [gemm/cpu,simd]
可迁移建议: 迁移 brgemm/gemm 的 blocked dispatch 与 JIT 配置分层。

8. `useEltwiseInjector` (`src/gpu/intel/gemm/jit/generator/pieces/c_update.cxx`) [gemm/gpu]
可迁移建议: 迁移 brgemm/gemm 的 blocked dispatch 与 JIT 配置分层。

9. `jit_uni_reorder_kernel` (`src/cpu/aarch64/reorder/jit_uni_reorder_kernel.hpp`) [reorder/cpu]
可迁移建议: 迁移 layout/reorder 抽象，降低不同内核布局耦合。

10. `jit_single_blk_kernel` (`src/cpu/aarch64/reorder/jit_uni_reorder_kernel.hpp`) [reorder/cpu]
可迁移建议: 迁移 layout/reorder 抽象，降低不同内核布局耦合。

11. `jit_avx512_core_amx_fwd_kernel` (`src/cpu/x64/jit_avx512_core_amx_conv_kernel.hpp`) [kernel_misc/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

12. `jit_avx512_core_amx_bwd_data_copy_kernel` (`src/cpu/x64/jit_avx512_core_amx_conv_kernel.hpp`) [kernel_misc/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

13. `jit_avx512_core_amx_bwd_data_kernel` (`src/cpu/x64/jit_avx512_core_amx_conv_kernel.hpp`) [kernel_misc/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

14. `jit_avx512_core_amx_bwd_weights_kernel` (`src/cpu/x64/jit_avx512_core_amx_conv_kernel.hpp`) [kernel_misc/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

15. `jit_avx512_core_amx_bwd_bias_kernel` (`src/cpu/x64/jit_avx512_core_amx_conv_kernel.hpp`) [kernel_misc/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

16. `jit_avx512_core_bf16_fwd_kernel` (`src/cpu/x64/jit_avx512_core_bf16_conv_kernel.hpp`) [kernel_misc/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

17. `jit_uni_dw_conv_fwd_kernel` (`src/cpu/aarch64/jit_uni_dw_conv_kernel_f32.hpp`) [activation/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

18. `jit_sve_conv_fwd_kernel` (`src/cpu/aarch64/jit_sve_conv_kernel.hpp`) [activation/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

19. `jit_uni_subkernel` (`src/cpu/x64/jit_uni_eltwise_int.cpp`) [activation/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

20. `jit_uni_subkernel` (`src/cpu/aarch64/jit_uni_eltwise_int.cpp`) [activation/cpu,simd]
可迁移建议: 优先迁移该片段对应的调度层与配置结构体。

## 推荐先落地的 3 个实验方向
1. `实验A: gemm 配置与调度复用`
目标: 复用 oneDNN 的 conf/init/dispatch 分层思想。
成功标准: 同一 API 下不同形状性能更稳定。

2. `实验B: kernel_misc 后处理融合`
目标: 将 post-ops/fused 路径整合到主内核调用链。
成功标准: 减少中间写回并保持精度。

3. `实验C: activation 布局与量化协同`
目标: 对齐 reorder + quant 的数据流，降低额外搬运。
成功标准: 端到端延迟下降且数值可控。
