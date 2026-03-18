# oneflow research_pack

## 仓库定位
- 源路径: `code_base/oneflow`
- 提交: `25c8978c1c8b1371ef6aa4187dae4495bd233c35`
- 许可证位置: `LICENSE`
- 已提纯高价值片段: `20`

## 最有价值片段（10~30）
1. `SoftmaxWarpImpl` (`oneflow/core/cuda/softmax.cuh`) [attention/gpu,cuda,simd]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

2. `SoftmaxBlockSMemImpl` (`oneflow/core/cuda/softmax.cuh`) [attention/gpu,cuda,simd]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

3. `SoftmaxBlockUncachedImpl` (`oneflow/core/cuda/softmax.cuh`) [attention/gpu,cuda,simd]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

4. `SoftmaxGradWarpImpl` (`oneflow/core/cuda/softmax.cuh`) [attention/gpu,cuda,simd]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

5. `SoftmaxGradBlockSMemImpl` (`oneflow/core/cuda/softmax.cuh`) [attention/gpu,cuda,simd]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

6. `SoftmaxGradBlockUncachedImpl` (`oneflow/core/cuda/softmax.cuh`) [attention/gpu,cuda,simd]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

7. `ReduceMaxMinPerLayer` (`oneflow/user/kernels/moving_average_min_max_observer_kernel.cu`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

8. `ReduceMaxMinPerLayer` (`oneflow/user/kernels/min_max_observer_kernel.cu`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

9. `CountNotFiniteGpu` (`oneflow/user/kernels/count_not_finite_kernel.cu`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

10. `batch_norm_backward_reduce_kernel` (`oneflow/user/kernels/batch_norm_backward_reduce_kernel.cu`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

11. `ComputeDiffWithSoftmaxGpuHalf` (`oneflow/user/kernels/sparse_cross_entropy_kernel_util.cu`) [allreduce/gpu,cuda,simd]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

12. `GroupNormParamGradKernel` (`oneflow/user/kernels/group_norm_kernel.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

13. `BatchReduceGammaBetaGradKernel` (`oneflow/user/kernels/group_norm_kernel.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

14. `Dequantize3D` (`oneflow/user/kernels/groupwise_quantization_kernels.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

15. `DequantizeInnerSize1` (`oneflow/user/kernels/groupwise_quantization_kernels.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

16. `QuantizedMatmulBiasGroupN` (`oneflow/user/kernels/groupwise_quantization_kernels.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

17. `QuantizedMatmulBiasGroupK` (`oneflow/user/kernels/groupwise_quantization_kernels.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

18. `IntervalKernel` (`oneflow/user/kernels/fused_attention_kernels.cu`) [elementwise/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

19. `PlaneKernel` (`oneflow/user/kernels/fused_attention_kernels.cu`) [elementwise/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

20. `FusedVectorizedReluDropoutKernel` (`oneflow/user/kernels/fused_matmul_bias_add_relu_dropout.cu`) [activation/gpu,cuda]
可迁移建议: 优先迁移向量化骨架并补充 ISA 能力检测。

## 推荐先落地的 3 个实验方向
1. `实验A: attention 调度与分层`
目标: 在统一 API 下实现 shape/后端感知调度。
成功标准: 典型输入上优于基线且数值一致。

2. `实验B: kernel_misc 融合与访存优化`
目标: 降低中间张量读写和 launch 次数。
成功标准: 核心链路延迟下降，误差在容差内。

3. `实验C: allreduce 精度-性能协同`
目标: 建立可切换策略并做 A/B。
成功标准: 形成可复现的质量-吞吐 Pareto 结果。
