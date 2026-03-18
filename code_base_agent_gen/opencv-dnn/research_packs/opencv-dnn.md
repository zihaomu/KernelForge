# opencv-dnn research_pack

## 仓库定位
- 源路径: `code_base/opencv-dnn`
- 提交: `01e56607df6d74892c939ec36e454271a3c70e97`
- 许可证位置: `未在当前镜像中发现显式 LICENSE 文件`
- 已提纯高价值片段: `20`

## 最有价值片段（10~30）
1. `grid_nms` (`src/cuda/grid_nms.cu`) [moe/gpu,cuda,simd]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

2. `grid_nms_collect` (`src/cuda/grid_nms.cu`) [moe/gpu,cuda,simd]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

3. `kernel_channel_max` (`src/opencl/softmax.cl`) [attention/gpu,opencl]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

4. `kernel_channel_subtract` (`src/opencl/softmax.cl`) [attention/gpu,opencl]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

5. `kernel_channel_sum` (`src/opencl/softmax.cl`) [attention/gpu,opencl]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

6. `kernel_channel_div` (`src/opencl/softmax.cl`) [attention/gpu,opencl]
可迁移建议: 优先迁移减少中间读写的融合路径，先做数值回归。

7. `TransposeConv` (`src/cuda4dnn/primitives/transpose_convolution.hpp`) [conv/gpu,cuda,cpu]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

8. `fp32_to_fp16` (`src/cuda/fp_conversion.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移向量化骨架并补充 ISA 能力检测。

9. `fp16_to_fp32` (`src/cuda/fp_conversion.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移向量化骨架并补充 ISA 能力检测。

10. `convBlock4x24` (`src/layers/cpu_kernels/convolution.cpp`) [conv/cpu,simd]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

11. `OpConv::forward` (`src/vkcom/src/op_conv.cpp`) [kernel_misc/gpu,vulkan,cpu,simd]
可迁移建议: 优先迁移向量化骨架并补充 ISA 能力检测。

12. `FastConv` (`src/layers/cpu_kernels/convolution.hpp`) [conv/cpu]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

13. `OpMatMul::forward` (`src/vkcom/src/op_matmul.cpp`) [kernel_misc/gpu,vulkan,cpu,simd]
可迁移建议: 优先迁移向量化骨架并补充 ISA 能力检测。

14. `im2col` (`src/opencl/im2col.cl`) [conv/gpu,opencl]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

15. `reduce_sum_abs` (`src/cuda/normalize.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

16. `reciprocal` (`src/cuda/normalize.cu`) [kernel_misc/gpu,cuda]
可迁移建议: 优先迁移接口与调度层，再替换底层内核。

17. `ONNXImporter::parseGemm` (`src/onnx/onnx_importer.cpp`) [gemm/cpu]
可迁移建议: 优先迁移 shape-aware 调度与 tile 策略，保留回退路径。

18. `ONNXImporter::parseConv` (`src/onnx/onnx_importer.cpp`) [conv/cpu,simd]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

19. `ONNXImporter::parseConvTranspose` (`src/onnx/onnx_importer.cpp`) [conv/cpu,simd]
可迁移建议: 优先迁移 pack/im2col/winograd 分支与 ISA 选择逻辑。

20. `ONNXImporter::parseTopK` (`src/onnx/onnx_importer.cpp`) [moe/cpu]
可迁移建议: 优先迁移 routing 与 grouped compute 解耦接口。

## 推荐先落地的 3 个实验方向
1. `实验A: conv 调度与分层`
目标: 在统一 API 下实现 shape/后端感知调度。
成功标准: 典型输入上优于基线且数值一致。

2. `实验B: kernel_misc 融合与访存优化`
目标: 降低中间张量读写和 launch 次数。
成功标准: 核心链路延迟下降，误差在容差内。

3. `实验C: attention 精度-性能协同`
目标: 建立可切换策略并做 A/B。
成功标准: 形成可复现的质量-吞吐 Pareto 结果。
