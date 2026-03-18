# ncnn research_pack

## 仓库定位
- 源路径: `code_base/ncnn`
- 提交: `7237643e4c0da870d8d022eddcf8730c501f1483`
- 许可证位置: `LICENSE.txt`
- 高价值片段数: `20`

## 最有价值片段（按优先级）
1. `InnerProduct_vulkan::create_pipeline` (`src/layer/vulkan/innerproduct_vulkan.cpp`) [gemm/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

2. `Deconvolution_vulkan::create_pipeline` (`src/layer/vulkan/deconvolution_vulkan.cpp`) [conv/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

3. `Convolution1D_vulkan::create_pipeline` (`src/layer/vulkan/convolution1d_vulkan.cpp`) [conv/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

4. `Gemm_vulkan::create_pipeline` (`src/layer/vulkan/gemm_vulkan.cpp`) [gemm/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

5. `DeconvolutionDepthWise_vulkan::create_pipeline` (`src/layer/vulkan/deconvolutiondepthwise_vulkan.cpp`) [conv/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

6. `Quantize_vulkan::create_pipeline` (`src/layer/vulkan/quantize_vulkan.cpp`) [quantization/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

7. `Convolution_vulkan::create_pipeline` (`src/layer/vulkan/convolution_vulkan.cpp`) [conv/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

8. `Requantize_vulkan::create_pipeline` (`src/layer/vulkan/requantize_vulkan.cpp`) [quantization/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

9. `MultiHeadAttention_vulkan::create_pipeline` (`src/layer/vulkan/multiheadattention_vulkan.cpp`) [attention/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

10. `SDPA_vulkan::create_pipeline` (`src/layer/vulkan/sdpa_vulkan.cpp`) [attention/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

11. `Dequantize_vulkan::create_pipeline` (`src/layer/vulkan/dequantize_vulkan.cpp`) [quantization/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

12. `ConvolutionDepthWise_vulkan::create_pipeline` (`src/layer/vulkan/convolutiondepthwise_vulkan.cpp`) [conv/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

13. `Softmax_vulkan::create_pipeline` (`src/layer/vulkan/softmax_vulkan.cpp`) [attention/gpu,vulkan]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

14. `Convolution1D_arm::create_pipeline_fp16s` (`src/layer/arm/convolution1d_arm_asimdhp.cpp`) [conv/cpu,simd,arm]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

15. `Packing_mips::forward_int8` (`src/layer/mips/packing_mips.cpp`) [quantization/cpu,simd,mips]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

16. `Packing_x86::forward_int8` (`src/layer/x86/packing_x86.cpp`) [quantization/cpu,simd,x86]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

17. `im2col_sgemm_pack4_msa` (`src/layer/mips/convolution_sgemm_pack4.h`) [gemm/cpu,simd,mips]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

18. `convolution_im2col_sgemm_pack4_msa` (`src/layer/mips/convolution_sgemm_pack4.h`) [gemm/cpu,simd,mips]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

19. `im2col_sgemm_int8_msa` (`src/layer/mips/convolution_sgemm_int8.h`) [gemm/cpu,simd,mips]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

20. `convolution_im2col_sgemm_transform_kernel_int8_msa` (`src/layer/mips/convolution_sgemm_int8.h`) [gemm/cpu,simd,mips]
可迁移建议: 先迁移该片段对应的算子调度/后端选择层，再替换为我们的高性能实现。

## 推荐先落地的 3 个实验方向
1. `实验A: 多 ISA 卷积路径统一调度`
目标: 将 x86/ARM/MIPS 的 conv/dwconv 快路径抽象为统一调度层。
成功标准: 至少两种 ISA 下吞吐提升且结果一致。

2. `实验B: Vulkan 后端 pipeline 复用`
目标: 迁移 create_pipeline/destroy_pipeline 机制到我们的 GPU 抽象层。
成功标准: 关键层在 GPU 后端稳定运行并优于 CPU 回退。

3. `实验C: 量化链路一体化`
目标: 对齐 int8 quant/dequant/requantize 与 packing 顺序。
成功标准: 误差受控且端到端时延下降。
