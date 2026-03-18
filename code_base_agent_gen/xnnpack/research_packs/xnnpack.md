# xnnpack research_pack

## 仓库定位
- 源路径: `code_base/xnnpack`
- 提交: `95af55e5c81dc88edfa48dec8f3bff0b275513d2`
- 许可证位置: `LICENSE`
- 已提纯高价值片段: `20`

## 最有价值片段（10~30）
1. `xnn_hmp_qp8gemm_ukernel` (`src/xnnpack/microfnptr.h`) [gemm/cpu,simd]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

2. `xnn_hmp_qp8gemm_bl_ukernel` (`src/xnnpack/microfnptr.h`) [gemm/cpu,simd]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

3. `gemm_fused_ukernel` (`src/xnnpack/microfnptr.h`) [gemm/cpu]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

4. `xnn_ukernel` (`src/operators/convolution-nhwc.c`) [conv/cpu]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

5. `xnn_hmp_igemm_ukernel` (`src/operators/convolution-nhwc.c`) [gemm/cpu]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

6. `xnn_ukernel` (`src/operators/deconvolution-nhwc.c`) [gemm/cpu,simd]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

7. `xnn_hmp_igemm_ukernel` (`src/operators/deconvolution-nhwc.c`) [gemm/cpu]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

8. `xnn_ukernel` (`src/operators/convolution-nchw.c`) [conv/cpu,simd]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

9. `xnn_hmp_dqgemm_ukernel` (`src/xnnpack/microfnptr.h`) [kernel_misc/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

10. `xnn_hmp_dqgemm_qc2w_ukernel` (`src/xnnpack/microfnptr.h`) [kernel_misc/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

11. `xnn_hmp_dqgemm_bl_ukernel` (`src/xnnpack/microfnptr.h`) [kernel_misc/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

12. `xnn_hmp_gemm_ukernel` (`src/xnnpack/microfnptr.h`) [kernel_misc/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

13. `xnn_hmp_dqigemm_ukernel` (`src/xnnpack/microfnptr.h`) [kernel_misc/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

14. `xnn_hmp_igemm_ukernel` (`src/xnnpack/microfnptr.h`) [kernel_misc/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

15. `AddOp` (`ynnpack/kernels/binary/binary.cc`) [elementwise/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

16. `SubtractOp` (`ynnpack/kernels/binary/binary.cc`) [elementwise/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

17. `MultiplyOp` (`ynnpack/kernels/binary/binary.cc`) [elementwise/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

18. `DivideOp` (`ynnpack/kernels/binary/binary.cc`) [elementwise/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

19. `MaxOp` (`ynnpack/kernels/binary/binary.cc`) [elementwise/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

20. `MinOp` (`ynnpack/kernels/binary/binary.cc`) [elementwise/cpu]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

## 推荐先落地的 3 个实验方向
1. `实验A: gemm 调度与分层`
目标: 在统一 API 下实现 shape/后端感知调度。
成功标准: 典型输入上优于基线且数值一致。

2. `实验B: kernel_misc 融合与访存优化`
目标: 降低中间张量读写和 launch 次数。
成功标准: 核心链路延迟下降，误差在容差内。

3. `实验C: elementwise 精度-性能协同`
目标: 建立可切换策略并做 A/B。
成功标准: 形成可复现的质量-吞吐 Pareto 结果。
