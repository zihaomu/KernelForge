# sgl-kernel research_pack

## 仓库定位
- 源路径: `code_base/sglang/sgl-kernel`
- 提交: `9a697ceabb2dce6885f6a1ab8993d9980ade86c1`
- 许可证: Apache-2.0 (`LICENSE`)
- 核心价值: LLM 推理 kernel 集合，覆盖 FP8/INT8/INT4 GEMM、MoE routing、Attention merge、Speculative sampling、自定义 allreduce、CPU AVX512 内核。

## 高价值片段（按优先级，供迁移）
1. `P1 sm90_fp8_dispatch_shape` (`csrc/gemm/fp8_gemm_kernel.cu`)
可迁移建议: 直接复用“按 M/架构切 scheduler”策略，把我们项目的 FP8 GEMM launch 层做成 shape-aware dispatch，而不是单一 kernel。

2. `P2 cutlass_int8_scaled_mm` (`csrc/gemm/int8_gemm_kernel.cu`)
可迁移建议: 借鉴 int8 + float scale + optional bias 的统一入口签名，减少量化路径接口分裂。

3. `P3 fp8_blockwise_scaled_grouped_mm` (`csrc/moe/fp8_blockwise_moe_kernel.cu`)
可迁移建议: 将 grouped GEMM 的指针表和 stride 表解耦，优先迁移“多 expert 批处理统一 launch”模式。

4. `P4 sm100_fp8_blockwise_dispatch_shape` (`csrc/gemm/fp8_blockwise_gemm_kernel.cu`)
可迁移建议: 把 tile shape 做成独立策略层（small-M 与 regular-M 分支），便于后续 autotune。

5. `P5 dense_kernel0(per-group)` (`csrc/gemm/qserve_w4a8_per_group_gemm.cu`)
可迁移建议: 迁移 `cp_async + ldmatrix + mma` 管线骨架，用于我们 W4A8 kernel 的 shared-memory 双缓冲版本。

6. `P6 dense_kernel0(per-channel)` (`csrc/gemm/qserve_w4a8_per_chn_gemm.cu`)
可迁移建议: 如果我们走 per-channel 量化，优先复用其“ascales/wscales + sum 修正”接口形态。

7. `P7 moe_fused_gate_kernel_dynamic` (`csrc/moe/moe_fused_gate.cu`)
可迁移建议: 迁移动态专家数参数化结构 `KernelParamsDynamic`，避免为不同 expert 数编译大量变体。

8. `P8 topkGatingSoftmax` (`csrc/moe/moe_topk_softmax_kernels.cu`)
可迁移建议: 借鉴 softmax + topk 融合，优先减少中间 logits 写回。

9. `P9 topkGatingSigmoid` (`csrc/moe/moe_topk_sigmoid_kernels.cu`)
可迁移建议: 作为 softmax gating 的可替代后端，保留同构 launch API，便于 A/B 对比。

10. `P10 fusedQKNormRopeKernel` (`csrc/moe/fused_qknorm_rope_kernel.cu`)
可迁移建议: 迁移“QK Norm + RoPE 一次读写”思路，优先落地到 decode 热路径。

11. `P11 merge_attn_states_kernel` (`csrc/attention/merge_attn_states.cu`)
可迁移建议: 使用 128-bit pack 读写 + LSE 合并模式，作为分段 attention merge 标准实现。

12. `P12 cutlass_mla_decode` (`csrc/attention/cutlass_mla_kernel.cu`, 待复核)
可迁移建议: 借鉴其接口设计（paged KV + workspace + split 参数），即使内核实现受 CUDA 版本限制也可先统一 ABI。

13. `P13 sgl_fused_add_rmsnorm` (`csrc/elementwise/fused_add_rms_norm_kernel.cu`)
可迁移建议: 迁移 residual + rmsnorm 融合前置检查与 shape guard，优先降低 glue code 复杂度。

14. `P14 topk_kernel` (`csrc/elementwise/topk.cu`)
可迁移建议: 复用“短序列 naive / 长序列 fast”双路径分派，而非强行统一单 kernel。

15. `P15 cross_device_reduce_2stage` (`csrc/allreduce/custom_all_reduce.cuh`, 待复核)
可迁移建议: 如果我们需要 bypass NCCL，可先实验其 two-stage reduce 模式和 signal 协议。

16. `P16 QuickReduce::allreduce` (`csrc/allreduce/quick_all_reduce.h`, 待复核)
可迁移建议: 借鉴 quantized allreduce 抽象层（INT8/INT6 切换），用于低带宽环境。

17. `P17 ggml_mul_mat_a8` (`csrc/quantization/gguf/gguf_kernel.cu`)
可迁移建议: 复用“输入先量化再 matmul”的路径分层，统一 GGUF/自定义量化权重执行入口。

18. `P18 TreeSpeculativeSamplingTargetOnly` (`csrc/speculative/speculative_sampling.cuh`, 待复核)
可迁移建议: 树式 speculative 的 accept/reject 合并在单 kernel 中，减少 host 侧循环。

19. `P19 per_token_group_quant_8bit_kernel` (`csrc/gemm/per_token_group_quant_8bit.cu`)
可迁移建议: 先迁移 group-wise per-token 量化，作为 INT8 GEMM 前处理标准组件。

20. `P20 per_token_quant_fp8_kernel` (`csrc/gemm/per_token_quant_fp8.cu`)
可迁移建议: 复用 warp-per-token 布局，为 FP8 路径提供轻量在线量化。

21. `P21 gemm_half_q_half_gptq_4bit_kernel` (`csrc/gemm/gptq/gptq_kernel.cu`, 待复核)
可迁移建议: 抽取“按 bitwidth 专用 kernel”模式（2/3/4/8bit），避免运行时分支过重。

22. `P22 dequantize_weights` (`csrc/gemm/awq_kernel.cu`, 待复核)
可迁移建议: 在 AWQ 路径中拆分为“预解量化/在线解量化”两套实现，再做吞吐对比。

23. `P23 tinygemm_kernel_nn<bf16>` (`csrc/cpu/gemm.cpp`, 待复核)
可迁移建议: 把 AVX512 VNNI pack + dpbf16 的 tinygemm 抽象成 CPU fallback 高性能路径。

24. `P24 fp8_scaled_mm_kernel_impl` (`csrc/cpu/gemm_fp8.cpp`, 待复核)
可迁移建议: 借鉴 FP8 unpack 到 BF16 再 GEMM 的 staged pipeline，便于无 GPU 环境测试。

25. `P25 causal_conv1d_fwd_cpu` (`csrc/cpu/mamba/conv.cpp`)
可迁移建议: 复用权重预打包 VNNI 策略，用于时序卷积/状态更新类算子。

26. `P26 bmm_fp8` (`csrc/gemm/bmm_fp8.cu`)
可迁移建议: 保留 cublas handle + workspace 的显式参数化，避免隐式全局状态。

## 推荐先落地的 3 个实验方向
1. `实验A: FP8 GEMM 调度层`
目标: 复现 `P1 + P4` 的 shape-aware dispatch，在同一 API 下选择不同 tile/scheduler。
成功标准: 至少在两个 M 档位（小 batch / 常规 batch）均优于单一 kernel 基线。

2. `实验B: MoE 路由与 grouped GEMM 融合链路`
目标: 结合 `P3 + P7 + P8/P9`，构建“gating -> reorder -> grouped mm”的最小闭环。
成功标准: 路由阶段中间写回减少，端到端 token 吞吐提升可测。

3. `实验C: 量化前处理在线化`
目标: 用 `P19 + P20 + P22` 建立统一 per-token/per-group 量化模块，驱动 INT8/FP8/W4A8 GEMM。
成功标准: 前处理耗时受控，量化误差在既定容差内，整体延迟下降。

## 迁移注意事项
- `待复核` 片段优先做正确性回归（数值误差 + determinism）后再性能调优。
- CUTLASS 路径对 CUDA 版本与 SM 架构约束较强，建议保留降级实现。
- CPU AVX512 片段应与 scalar/fallback 路径并存，避免平台覆盖不足。
