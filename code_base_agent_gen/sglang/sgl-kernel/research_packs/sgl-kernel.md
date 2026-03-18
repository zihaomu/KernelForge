# sglang/sgl-kernel research_pack

## 仓库定位
- 源路径: `code_base/sglang/sgl-kernel`
- 提交: `9a697ceabb2dce6885f6a1ab8993d9980ade86c1`
- 许可证位置: `LICENSE`
- 已提纯高价值片段: `20`

## 最有价值片段（10~30）
1. `moeSoftmax` (`csrc/moe/moe_topk_softmax_kernels.cu`) [attention/gpu,cuda]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

2. `topkGatingSoftmax` (`csrc/moe/moe_topk_softmax_kernels.cu`) [moe/gpu,cuda]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

3. `topkGatingSigmoid` (`csrc/moe/moe_topk_sigmoid_kernels.cu`) [moe/gpu,cuda]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

4. `per_token_group_quant_8bit_kernel` (`csrc/gemm/per_token_group_quant_8bit_v2.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

5. `dense_kernel0` (`csrc/gemm/qserve_w4a8_per_group_gemm.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

6. `dense_kernel0` (`csrc/gemm/qserve_w4a8_per_chn_gemm.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

7. `cross_device_reduce_1stage` (`csrc/allreduce/custom_all_reduce_hip.cuh`) [allreduce/gpu,cuda,rocm,hip]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

8. `cross_device_reduce_2stage` (`csrc/allreduce/custom_all_reduce_hip.cuh`) [allreduce/gpu,cuda,rocm,hip]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

9. `fusedQKNormRopeKernel` (`csrc/moe/fused_qknorm_rope_kernel.cu`) [attention/gpu,cuda]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

10. `per_token_quant_fp8_kernel` (`csrc/gemm/per_token_quant_fp8.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

11. `cross_device_reduce_1stage` (`csrc/allreduce/custom_all_reduce.cuh`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

12. `cross_device_reduce_2stage` (`csrc/allreduce/custom_all_reduce.cuh`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

13. `moe_sum_reduce_warp_per_token_vec_kernel` (`csrc/moe/moe_sum_reduce.cu`) [moe/gpu,cuda]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

14. `moe_sum_reduce_kernel_warp_token_topk` (`csrc/moe/moe_sum_reduce.cu`) [moe/gpu,cuda]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

15. `moe_sum_reduce_kernel_warp_token_general` (`csrc/moe/moe_sum_reduce.cu`) [moe/gpu,cuda]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

16. `per_token_group_quant_8bit_kernel` (`csrc/gemm/per_token_group_quant_8bit.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

17. `KernelParams` (`csrc/moe/moe_fused_gate.cu`) [moe/gpu,cuda]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

18. `Marlin` (`csrc/gemm/marlin/marlin_template.h`) [gemm/gpu,cuda,cpu]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

19. `allreduce_LL_1node` (`csrc/allreduce/mscclpp_allreduce.cuh`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

20. `allreduce_LL_2node` (`csrc/allreduce/mscclpp_allreduce.cuh`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移分块通信和量化通信策略，并做确定性验证。

## 推荐先落地的 3 个实验方向
1. `实验A: moe 调度与分层`
目标: 在统一 API 下实现 shape/后端感知调度。
成功标准: 典型输入上优于基线且数值一致。

2. `实验B: gemm 融合与访存优化`
目标: 降低中间张量读写和 launch 次数。
成功标准: 核心链路延迟下降，误差在容差内。

3. `实验C: allreduce 精度-性能协同`
目标: 建立可切换策略并做 A/B。
成功标准: 形成可复现的质量-吞吐 Pareto 结果。
