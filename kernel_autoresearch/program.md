# CPU GEMM Autoresearch Program (X/E/D/L)

## 1. 设计目标

本实现将 CPU GEMM `autoresearch` 升级为严格工程闭环，采用 AutoKernel 风格的四段式映射：

- `X` (mutable artifact): `workspace/kernel_candidate.json`
- `E` (fixed evaluator): `harness/bench.py` + `harness/reference.py`
- `D` (keep/revert policy): `python/decision_policy.py`
- `L` (logbook): `workspace/results/results.tsv` + `workspace/orchestration_state.json` + `workspace/run.log`

## 2. X/E/D/L 映射

### X: 可变研究对象

- 文件：`kernel_autoresearch/workspace/kernel_candidate.json`
- 约束：
  - 每轮实验只修改这一份候选文件
  - 候选只描述 kernel 参数（variant、tile、pack、simd、threads、unroll、dtype）
  - `simd` 参数语义：`simd=1` 表示启用 xsimd 向量化路径（float32）
  - keep/revert 直接作用于该文件，回滚粒度明确
  - Agent 若启用，也只能输出该文件对应的参数 JSON，不能改 harness

### E: 固定评测器

- 文件：
  - `kernel_autoresearch/harness/bench.py`
  - `kernel_autoresearch/harness/reference.py`
  - `kernel_autoresearch/harness/manifest.json`（哈希清单）
- 约束：
  - 编排循环每轮评测前校验 `manifest`
  - 如果 `bench/reference` 哈希不一致，立即中止
- correctness gate：
  1. 候选 schema 校验（参数合法）
  2. deterministic 输入 checksum 校验（对齐 reference）
  3. 目标 shape 集 correctness（runner verify）
  4. 稳定性复跑 gate（可配置次数）

### D: 机械决策规则

- 文件：`kernel_autoresearch/python/decision_policy.py`
- 规则：
  - correctness FAIL => `revert`
  - correctness PASS 且 `score` 提升超过阈值 => `keep`
  - correctness PASS 但提升不足 => `revert`
- 阈值：
  - `min_improve_ratio`（默认 1%）

### Agent 提案层（可选）

- 文件：`kernel_autoresearch/python/agent_proposer.py`
- 模式：
  - `rules_only`：不使用 LLM
  - `hybrid`：LLM 提案优先，失败回退到启发式/规则候选
  - `agent_only`：尽量使用 LLM 提案，失败时最小回退避免停机
- 边界：
  - Agent 只负责“提案”，最终保留权仍由固定评测器 + 决策器决定
  - Agent 不能修改 `E`，且每轮前都校验 `harness manifest`

### L: 结构化记录

- 文件：
  - `workspace/results/results.tsv`: 每轮结果（含 score/decision/reason）
  - `workspace/orchestration_state.json`: 状态机快照（iteration、active bucket、cursor、best）
  - `workspace/run.log`: 文本日志（move-on、decision、异常）
  - `workspace/results/iter_XXXX.json`: 每轮详细评测数据

## 3. 状态机

- 入口：`python/orchestration_loop.py::run_orchestration`
- bucket 顺序：按配置的 `small -> medium -> large`（若该 bucket 有 shape）
- move-on 条件：
  - 当前 bucket 候选耗尽
  - 连续 `patience_no_improve` 轮无提升
  - 达到 `max_iterations`

## 4. 评分

每个 bucket 以 baseline 候选为参照（score=1）：

- 吞吐项：`avg_gflops / baseline_gflops`
- 延迟项：`baseline_latency / avg_latency`
- 总分：`alpha * throughput + (1-alpha) * latency`

`alpha` 来自 bucket 配置，体现吞吐/延迟平衡策略。

## 5. 运行命令

1. 刷新 harness 哈希清单（首次或有意更新评测器时）：

```bash
python -m kernel_autoresearch.python.cli \
  --config kernel_autoresearch/configs/default.yaml \
  refresh-harness-manifest
```

2. 启动闭环编排：

```bash
python -m kernel_autoresearch.python.cli \
  --config kernel_autoresearch/configs/default.yaml \
  orchestrate
```

3. 启动 Agent 增强模式：

```bash
python -m kernel_autoresearch.python.cli \
  --config kernel_autoresearch/configs/default.yaml \
  --agent-mode hybrid \
  --agent-model gpt-5.4 \
  orchestrate
```

## 6. 不变量

1. 评测不可变：`harness manifest` 不通过则停止。
2. 变更面最小：仅 `kernel_candidate.json` 在循环中可变。
3. correctness 优先：任何 correctness 失败都不能 keep。
4. 全量留痕：每轮输入、输出、决策、原因均写入结构化日志。
