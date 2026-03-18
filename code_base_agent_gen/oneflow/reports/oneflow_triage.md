# oneflow triage report

## 扫描信息
- 总扫描文件数: `4611`
- 候选代码文件数: `2012`
- 纳入片段数: `20`
- commit: `25c8978c1c8b1371ef6aa4187dae4495bd233c35`
- 许可证定位: `LICENSE, ci/check/run_license_format.py`

## 排除类型统计
- other: `3023`
- tests: `943`
- python_api_or_tooling: `523`
- build_config_ci: `56`
- third_party: `33`
- docs: `31`
- license_meta: `2`

## 纳入与排除说明
- 纳入: 与 kernel 优化直接相关的实现、调度、向量化、量化、通信与融合代码。
- 排除: 测试、benchmark、文档、构建脚本、CI 与通用封装层。

## 风险与不确定项
- 待复核片段数: `6`
- 主要风险: 量化数值稳定性、平台/ISA 绑定、融合路径可维护性、通信确定性。
- 已按 `待复核` 标记不确定片段。
