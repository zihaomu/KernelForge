# Kernel

本项目旨在做一个AI自动生成的kernel池子，包括torch相关的GPU算子，XSIMD相关的CPU算子。

项目结构：
code_base中存放一些等待学习的kernel代码片段，从各个开源仓库搜集来的。
- sgl-kernel
- MatmulTutorial
- MNN
- ncnn
- oneflow
- opencv-dnn
- slglang/sgl-kernel
- xnnpack

code base的作用是给agent一个可以学习的方向。

kernel：中存放主要的核心实现，
- cpu_kernel中存放所有cpu算子
- gpu_kernel中存放所有gpu算子

其中每个算子一个文件夹:

CPU算子构成：
- conv_3x3
  - include 接口文件，输入的shape，kernel的shape，stride等信息，数据指针
  - src 实现文件，基于3rdpart中的 xsimd实现跨平台的simd加速
  - test 需要先实现python版本基于numpy的算子原型，然后将input和output数据写入test/data中，数据生成脚本也要保存到test中，然后C++部分基于libnpy实现加载python的验证数据，完成accuracy测试。
  - perf 就要include的头文件，实现一个算子的benchmark测试，测试不同规模数据，不同精度的速度。
- gemm 同理

GPU算子构成：
由于GPU有众多平台，可以选择
- gemm， 要求在4090上实现不同精度（fp32，fp16，int8，fp8），不同规模的gemm
  - cuda ： 基于cuda C++代码实现，使用pybind
  - triton： 基于triton实现 python的算子
  - test  ： 统一使用 torch 作为算子实现原型，将结果保存成npy数组到test的data文件夹中，accuracy测试时，从test data中读取不同规模的测试数据，从而保证我们的kernel在不同情况都能生成比较好的数据
  - perf： 性能的base是 torch实现的对应的算子。


算子列表[待补充]：
GPU算子：
- 
CPU算子：
- conv1x1
- conv3x3

