# MNN research_pack

## 仓库定位
- 源路径: `code_base/MNN`
- 提交: `6fe28bc7dc7441e12d01fe2918e72f76aa9f031f`
- 许可证位置: `LICENSE.txt`
- 已提纯高价值片段: `20`

## 最有价值片段（10~30）
1. `Im2Col_packC_16` (`source/backend/cuda/execution/int8/ConvInt8CutlassExecution.cu`) [quantization/gpu,cuda,simd]
可迁移建议: 优先迁移在线 quant/dequant 组件并固化 scale 契约。

2. `WeightInt8PackFill` (`source/backend/cuda/execution/int8/ConvInt8CutlassExecution.cu`) [quantization/gpu,cuda,simd]
可迁移建议: 优先迁移在线 quant/dequant 组件并固化 scale 契约。

3. `GEMM_FpAInt4B` (`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`) [quantization/gpu,cuda]
可迁移建议: 优先迁移在线 quant/dequant 组件并固化 scale 契约。

4. `GEMV_FpAInt4B` (`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`) [quantization/gpu,cuda]
可迁移建议: 优先迁移在线 quant/dequant 组件并固化 scale 契约。

5. `CONV_DW_INT8_` (`source/backend/cuda/execution/int8/DepthwiseConvInt8Execution.cu`) [quantization/gpu,cuda]
可迁移建议: 优先迁移在线 quant/dequant 组件并固化 scale 契约。

6. `CONV_DW3x3S1_INT8_OPT` (`source/backend/cuda/execution/int8/DepthwiseConvInt8Execution.cu`) [quantization/gpu,cuda]
可迁移建议: 优先迁移在线 quant/dequant 组件并固化 scale 契约。

7. `SOFTMAX` (`source/backend/cuda/execution/SoftmaxExecution.cu`) [attention/gpu,cuda]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

8. `compact_kv_cache_kernel` (`source/backend/cuda/execution/AttentionExecution.cu`) [attention/gpu,cuda]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

9. `softmax_kernel` (`source/backend/cuda/execution/AttentionExecution.cu`) [attention/gpu,cuda]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

10. `B2bGemm` (`source/backend/cuda/execution/plugin/FmhaCommon/fused_multi_head_attention/gemm/mma_from_smem.h`) [gemm/gpu,cuda,cpu]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

11. `AttentionKernel` (`source/backend/cuda/execution/plugin/FmhaCommon/fused_multi_head_attention/kernel_forward.h`) [kernel_misc/gpu,cuda,cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

12. `__launch_bounds__` (`source/backend/cuda/execution/plugin/FmhaCommon/fused_multi_head_attention/kernel_forward.h`) [kernel_misc/gpu,cuda,cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

13. `DequantizeInt8Weight` (`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

14. `QuantA` (`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

15. `GEMM_Int8` (`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

16. `Rearrange_Packed_Weight_Int4` (`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

17. `AttentionKernel` (`source/backend/cuda/execution/AttentionExecution.hpp`) [attention/gpu,cuda,metal,cpu]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

18. `LAYERNORM` (`source/backend/cuda/execution/LayerNormExecution.cu`) [norm/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

19. `groupNormNHWCSumKernel` (`source/backend/cuda/execution/plugin/GroupNorm/groupNormKernel.cu`) [norm/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

20. `DefaultGemm` (`source/backend/cuda/execution/plugin/FmhaCommon/fused_multi_head_attention/gemm_kernel_utils.h`) [gemm/gpu,cuda,cpu]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

## 推荐先落地的 3 个实验方向
1. `实验A: quantization 调度与分层`
目标: 在统一 API 下实现 shape/后端感知调度。
成功标准: 典型输入上优于基线且数值一致。

2. `实验B: kernel_misc 融合与访存优化`
目标: 降低中间张量读写和 launch 次数。
成功标准: 核心链路延迟下降，误差在容差内。

3. `实验C: attention 精度-性能协同`
目标: 建立可切换策略并做 A/B。
成功标准: 形成可复现的质量-吞吐 Pareto 结果。
