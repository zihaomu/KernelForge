# MNN triage report

## 扫描信息
- 总扫描文件数: `8175`
- 候选代码文件数: `4634`
- 纳入片段数: `20`
- commit: `6fe28bc7dc7441e12d01fe2918e72f76aa9f031f`
- 许可证定位: `LICENSE.txt, 3rd_party/half/LICENSE.txt, 3rd_party/protobuf/LICENSE`

## 排除类型统计
- other: `6597`
- tests: `603`
- third_party: `434`
- docs: `237`
- build_config_ci: `128`
- python_api_or_tooling: `124`
- benchmark_perf: `42`
- license_meta: `10`

## 纳入与排除说明
- 纳入: 与 kernel 优化直接相关的实现、调度、向量化、量化、通信与融合代码。
- 排除: 测试、benchmark、文档、构建脚本、CI 与通用封装层。

## 风险与不确定项
- 待复核片段数: `5`
- 主要风险: 量化数值稳定性、平台/ISA 绑定、融合路径可维护性、通信确定性。
- 已按 `待复核` 标记不确定片段。
