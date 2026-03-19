# AutoResearch v2 -- Autonomous CPU GEMM Optimization Program

你是一个**只做算子级优化**的自主研究 Agent。  
当前版本只优化一个 OP：`CPU GEMM`。  
禁止上升到模型图、模型端到端推理、算子融合编排等模型层话题。

---

## 0. Scope / Non-Goals

### Scope (必须做)

1. 固定 OP 合约下的 CPU GEMM 内核优化。
2. 在固定评测器下执行 keep/revert 闭环。
3. 以可复现实验日志交付最终最优候选。

### Non-Goals (禁止做)

1. 不做模型 profiling，不做 top-k op 提取。
2. 不讨论 transformer/llm 层级优化策略。
3. 不修改评测协议来“制造”性能收益。

---

## 1. Program Architecture (X/E/D/L)

本方案遵循工程闭环：`X`(可变对象) / `E`(固定评测器) / `D`(决策器) / `L`(日志系统)。

### X: Mutable Artifact (唯一可变核心)

- `kernel_autoresearch_v2/workspace/gemm_candidate.json`
- `kernel_autoresearch_v2/workspace/gemm_impl.cc` (可选；当采用源码搜索时)

规则：

1. 每轮实验只允许一个“主改动点”（参数或代码二选一）。
2. 若是参数搜索，优先只改 `gemm_candidate.json`。
3. 若是源码搜索，必须记录“参数变化 + 代码变化 + 假设”。

### E: Evaluator (固定评测器，不可随意改)

- `kernel_autoresearch_v2/harness/reference.py`
- `kernel_autoresearch_v2/harness/bench.py`
- `kernel_autoresearch_v2/harness/manifest.json`
- `kernel_autoresearch_v2/cpp_runner/kc_gemm_runner` (或等价 runner)

规则：

1. 每轮运行前都校验 `manifest`，哈希不一致立即中止。
2. correctness gate 必须先于 performance gate。
3. benchmark 输出必须结构化并包含 `METRIC key=value`。

### D: Decision Policy (机械决策)

- `kernel_autoresearch_v2/python/decision_policy.py`

默认策略：

1. correctness 失败 -> `REVERT`
2. correctness 通过且主指标提升 `>= min_improve_ratio` -> `KEEP`
3. correctness 通过但提升不足 -> `REVERT`
4. 性能持平但代码明显更简单 -> 允许 `KEEP`（需 reason）

### L: Logbook (全量留痕)

- `kernel_autoresearch_v2/workspace/results/results.tsv`
- `kernel_autoresearch_v2/workspace/results/iter_XXXX.json`
- `kernel_autoresearch_v2/workspace/orchestration_state.json`
- `kernel_autoresearch_v2/workspace/run.log`
- `kernel_autoresearch_v2/workspace/final_report.md`

---

## 2. Optimization Objective

目标是提升 CPU GEMM 的稳定性能，不牺牲正确性。

### Primary Metric

- `score` (higher is better)

建议定义：

```text
throughput_ratio = gflops / baseline_gflops
latency_ratio    = baseline_latency_us / latency_us
score            = alpha * throughput_ratio + (1 - alpha) * latency_ratio
```

### Secondary Metrics

1. `gflops`
2. `latency_us`
3. `correctness_pass`
4. `l2_miss_rate` (可选)
5. `bandwidth_gbps` (可选)
6. `stability_cv` (重复运行变异系数，可选)

---

## 3. Search Space (CPU GEMM Example)

默认候选维度（可按机器裁剪）：

1. `kernel_variant`: `naive | blocked | blocked_pack | blocked_pack_simd`
2. `block_m`: `[16, 32, 64, 96, 128]`
3. `block_n`: `[16, 32, 64, 96, 128]`
4. `block_k`: `[16, 32, 64, 96, 128]`
5. `pack_a`: `[0, 1]`
6. `pack_b`: `[0, 1]`
7. `simd`: `[0, 1]`
8. `threads`: `[1, 2, 4, 8, 16]`
9. `unroll_k`: `[1, 2, 4]`

Shape bucket（建议）：

1. `small`: 小矩阵低延迟场景
2. `medium`: 通用吞吐场景
3. `large`: 大矩阵高吞吐场景

---

## 4. Correctness Gates (必须全部通过)

每轮实验执行顺序：

1. `schema gate`：候选 JSON 字段和类型合法。
2. `tiny-shape gate`：小规模 shape 快速精度校验（如 32x32x32）。
3. `reference gate`：与参考实现逐元素对比（atol/rtol 固定）。
4. `stability gate`：重复 2-5 次，确认结果无异常漂移。

任一 gate 失败，该轮实验状态必须记录为 `checks_failed` 或 `crash`，且执行 `REVERT`。

---

## 5. Phase A - Initialization (与人一次性对齐)

这一阶段允许和人交互；完成后进入全自动阶段。

### A1. 固定实验边界

1. 数据类型：`fp32`（v2 第一优先）
2. 输入布局：row-major（默认）
3. CPU 线程/绑核策略
4. 时间预算与最大迭代数

### A2. 初始化工作区

1. 创建分支：`git checkout -b autoresearch/cpu-gemm-<date>`
2. 生成初始 `gemm_candidate.json`
3. 生成 `results.tsv` 表头
4. 清空或归档上轮 `run.log`

### A3. 基线测量

1. 运行 baseline candidate
2. 写入 `iter_0000.json`
3. 固化 baseline 到 `orchestration_state.json`

### A4. 清单校验

1. 首次运行前刷新 `manifest`
2. 之后每轮校验 `manifest`，防止评测器漂移

---

## 6. Phase B - Autonomous Optimization Loop (全自动)

这一阶段**不需要人类持续介入**。  
除非发生不可恢复错误，否则持续循环直到满足停止条件。

### B1. 读取状态

1. 当前 bucket
2. 当前 best candidate
3. 最近 N 轮增益趋势

### B2. 提出下一候选

候选来源：

1. `rules_only`: 网格/启发式/局部扰动
2. `hybrid`: LLM 提案优先，失败回退规则
3. `agent_only`: LLM 主导，失败最小回退防停机

### B3. 单点改动

每轮只做一个主要假设，例如：

1. `block_k: 64 -> 96`
2. `threads: 8 -> 16`
3. `simd: 0 -> 1`
4. `pack_b: 0 -> 1`

### B4. 运行评测

统一通过 `uv` 触发，标准输出需含 `METRIC` 行。

```bash
uv run python -m kernel_autoresearch_v2.python.cli orchestrate \
  --config kernel_autoresearch_v2/configs/default.yaml
```

若仅执行单轮实验（可选）：

```bash
uv run python -m kernel_autoresearch_v2.python.cli run-once \
  --config kernel_autoresearch_v2/configs/default.yaml
```

### B5. 决策 keep/revert

按 D 策略执行：

1. `PASS + improve` -> keep，更新 best
2. `PASS + no improve` -> revert
3. `FAIL/CRASH` -> revert，并记录失败原因

### B6. 记录与推进

1. 写 `iter_XXXX.json`
2. 追加 `results.tsv`
3. 刷新 `orchestration_state.json`
4. 判定是否 move-on 到下一个 bucket

### B7. Stop Conditions

满足任一条件即可停止：

1. 达到 `max_iterations`
2. 连续 `patience_no_improve` 轮无提升
3. 全部 bucket 达到 plateau
4. 用户中断

---

## 7. Phase C - Final Verification & Delivery

### C1. 全量回归验证

1. 在所有目标 shape 上做 correctness 回归
2. 在所有目标 shape 上做性能复测（建议重复 5-10 次）

### C2. 产出最终候选

1. 输出 `best_config.json`
2. 导出最优实现（参数或源码）
3. 固化复现实验命令

### C3. 生成总结报告

`final_report.md` 至少包含：

1. baseline 与 final 对比表
2. 每个 bucket 的最优参数
3. 失败路径与无效尝试总结
4. 下一步可尝试但未执行的想法列表

---

## 8. Experiment Protocol (必须遵守)

1. 每轮实验前写一句 hypothesis。
2. 每轮实验只变一个主因素，避免归因污染。
3. 不允许跳过 correctness gate。
4. 不允许手改历史结果文件。
5. 失败实验也必须记录，不能静默丢弃。

---

## 9. Suggested File Layout (v2)

```text
kernel_autoresearch_v2/
  program.md
  configs/
    default.yaml
    shapes.yaml
  harness/
    bench.py
    reference.py
    manifest.json
  python/
    cli.py
    orchestration_loop.py
    decision_policy.py
    candidate_generator.py
    logbook.py
  cpp_runner/
    src/
    include/
    build/
  workspace/
    gemm_candidate.json
    orchestration_state.json
    run.log
    results/
      results.tsv
      iter_0001.json
      ...
```

---

## 10. Starter Commands (uv-first)

```bash
# 1) 安装依赖（如果后续补充了 requirements）
uv pip install -r kernel_autoresearch_v2/requirements.txt

# 2) 刷新评测器 manifest
uv run python -m kernel_autoresearch_v2.python.cli refresh-harness-manifest \
  --config kernel_autoresearch_v2/configs/default.yaml

# 3) 启动全自动闭环（推荐）
uv run python -m kernel_autoresearch_v2.python.cli orchestrate \
  --config kernel_autoresearch_v2/configs/default.yaml

# 4) 生成进展报告
uv run python -m kernel_autoresearch_v2.python.cli progress-report \
  --config kernel_autoresearch_v2/configs/default.yaml
```

---

## 11. Acceptance Criteria

当且仅当满足以下条件，可宣告一次 run 完成：

1. 最优候选在全部目标 shape correctness `PASS`
2. 相比 baseline 主指标有稳定提升（非单次抖动）
3. `results.tsv + iter_*.json + final_report.md` 完整可追溯
4. 可用一条 `uv run ...` 命令复现实验结果

---

## 12. Guiding Principle

这是一个算子级研究系统，不是模型级调度系统。  
任何提案都必须回答三件事：  
1) 改了什么；2) 为什么可能更快；3) 如何被固定评测器证实。
