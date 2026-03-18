# MatmulTutorial research_pack

## 仓库定位
- 源路径: `code_base/MatmulTutorial`
- 提交: `70f70fa08b1d5095dfb500c39f56b0465fff05e0`
- 已提纯高价值片段: `11`

## 最有价值片段（10~30）
1. `matmul` (`examples/matmul/this-sm80/matmul-v17.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

2. `MatmulKernel` (`examples/matmul/this-sm90/this-fp8/matmul-fp8-v0.cu`) [gemm/gpu,cuda,simd]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

3. `matmul_fp8` (`examples/matmul/this-sm90/this-fp8/matmul-fp8-v0.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

4. `matmul` (`examples/matmul/this-sm80/matmul-for-perf.cu`) [gemm/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

5. `reduceKernel` (`examples/reduction/reduce-v1.cu`) [allreduce/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

6. `KernelParams` (`examples/atom/sm90-warpspecialized-barrier.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

7. `produce_consume` (`examples/atom/sm90-warpspecialized-barrier.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

8. `KernelParams` (`util/cutlass/test_tile_scheduler.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

9. `test_kernel` (`util/cutlass/test_tile_scheduler.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

10. `reference_gpu_gemm_kernel` (`include/reference.h`) [kernel_misc/gpu,cuda,cpu,simd]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

11. `GemmParams` (`include/common.h`) [kernel_misc/cpu]
可迁移建议: 优先迁移 tile/warp 调度与共享内存流水化骨架，再按目标架构细化。

## 推荐先落地的 3 个实验方向
1. `实验A: GEMM tile/scheduler 分层`
目标: 为不同 M/N/K 形状选择不同 block/warp 配置。
成功标准: 至少两档形状优于单策略基线。

2. `实验B: FP8 路径稳定性验证`
目标: 对齐 FP8 kernel 的 scale 与累加策略。
成功标准: 精度达标且吞吐不低于现有实现。

3. `实验C: reduction/producer-consumer 原语复用`
目标: 迁移 reduction 和 warp-specialized barrier 模式到核心算子。
成功标准: 端到端延迟下降且可复现。
