# oneDNN triage report

## 扫描信息
- 总扫描文件数: `4755`
- 候选代码文件数: `2853`
- 纳入片段数: `20`
- commit: `61359b68ffd7b2ac9d912fecdee278f5eed3a4eb`
- 许可证定位: `LICENSE, third_party/ngen/COPYRIGHT, third_party/ittnotify/LICENSE.BSD`

## 排除类型统计
- other: `2866`
- tests: `1571`
- docs: `220`
- build_config_ci: `92`
- license_meta: `6`

## 纳入与排除说明
- 纳入: `src/cpu` 与 `src/gpu` 下直接影响 kernel 关键路径的实现、JIT 配置和调度代码。
- 排除: tests/examples/docs/cmake/third_party 等非内核核心内容。

## 风险与不确定项
- 待复核片段数: `19`
- 主要风险: JIT/ISA 强绑定、layout 耦合、低精度路径的精度与性能权衡。
