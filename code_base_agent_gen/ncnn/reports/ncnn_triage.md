# ncnn triage report

## 扫描信息
- 总扫描文件数: `3899`
- 候选符号行数: `1060`
- 纳入片段数: `20`
- commit: `7237643e4c0da870d8d022eddcf8730c501f1483`
- 许可证定位: `LICENSE.txt`

## 排除类型统计
- other: `2652`
- tests: `1031`
- docs: `85`
- python_api_or_tooling: `65`
- build_config_ci: `63`
- benchmark_perf: `2`
- license_meta: `1`

## 纳入与排除说明
- 纳入: `src/layer` 中与算子执行路径直接相关的函数签名和后端 pipeline 入口。
- 排除: 测试、benchmark、文档、构建脚本、CI 与通用工具层。

## 风险与不确定项
- 待复核片段数: `4`
- 风险: ISA 分支多、Vulkan 平台依赖、int8 量化路径数值敏感。
