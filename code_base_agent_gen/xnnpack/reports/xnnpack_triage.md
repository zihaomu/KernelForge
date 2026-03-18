# xnnpack triage report

## 扫描信息
- 总扫描文件数: `11552`
- 候选代码文件数: `10082`
- 纳入片段数: `20`
- commit: `95af55e5c81dc88edfa48dec8f3bff0b275513d2`
- 许可证定位: `LICENSE`

## 排除类型统计
- other: `10951`
- tests: `558`
- docs: `22`
- build_config_ci: `11`
- benchmark_perf: `9`
- license_meta: `1`

## 纳入与排除说明
- 纳入: 与 kernel 优化直接相关的实现、调度、向量化、量化、通信与融合代码。
- 排除: 测试、benchmark、文档、构建脚本、CI 与通用封装层。

## 风险与不确定项
- 待复核片段数: `9`
- 主要风险: 量化数值稳定性、平台/ISA 绑定、融合路径可维护性、通信确定性。
- 已按 `待复核` 标记不确定片段。
