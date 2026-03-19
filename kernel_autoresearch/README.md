# kernel_autoresearch

CPU GEMM 场景的 `autoresearch` 最小实现，目标是把以下两类范式融合成可执行实验：

- 本地提纯范式：`code_base_agent_gen/**` 下的 `manifest/snippets/research_pack/triage`
- 云端检索范式：基于查询词自动抓取网页并提炼优化关键词

该实现采用 `Python + 少量 C++ runner`，并按 `X/E/D/L` 工程闭环运行：

- Python 负责模式提取、候选生成、实验编排、平衡评分与报告输出
- C++ 负责执行 CPU GEMM 候选配置并返回性能/精度指标
- `X`: `workspace/kernel_candidate.json`
- `E`: `harness/bench.py + harness/reference.py`（由 `harness/manifest.json` 保护）
- `D`: `python/decision_policy.py` keep/revert 规则
- `L`: `workspace/results/results.tsv + workspace/orchestration_state.json + workspace/run.log`

CPU 向量化实现说明：
- 当前 `simd=1` 路径基于 `3rdparty/xsimd`（`xsimd::batch<float>`）实现。
- 不做手写 AVX/NEON 分支，统一通过 xsimd 抽象层适配底层 ISA。
- 当前优化目标限定为 `float32` GEMM。

## 环境

根据项目主 README，推荐环境：

```bash
conda activate py12_sgl
```

安装依赖：

```bash
pip install -r kernel_autoresearch/requirements.txt
```

## 快速开始

1. 首次或更新评测器后，刷新 harness 指纹：

```bash
python -m kernel_autoresearch.python.cli --config kernel_autoresearch/configs/default.yaml refresh-harness-manifest
```

2. 运行闭环 autoresearch（推荐）：

```bash
python -m kernel_autoresearch.python.cli orchestrate \
  --config kernel_autoresearch/configs/default.yaml
```

也可以直接用一键脚本：

```bash
bash kernel_autoresearch/quick_start.sh
```

常用变体：

```bash
# 快速配置（更短单轮时长）
bash kernel_autoresearch/quick_start.sh --quick

# INT8->INT32 快速探索
bash kernel_autoresearch/quick_start.sh --int8

# FP16 快速探索
bash kernel_autoresearch/quick_start.sh --fp16

# 实时看日志
bash kernel_autoresearch/quick_start.sh watch
```

3. 开启 Agent 模式（更强探索）：

```bash
python -m kernel_autoresearch.python.cli orchestrate \
  --config kernel_autoresearch/configs/default.yaml \
  --agent-mode hybrid \
  --agent-model gpt-5.4
```

说明：
- `rules_only`：只用规则候选（默认）
- `hybrid`：优先 agent 提案，失败自动回退规则/启发式
- `agent_only`：尽量只走 agent 提案，失败时最小回退防死锁
- 若要走 OpenAI 提案，请先设置 `OPENAI_API_KEY`：

```bash
export OPENAI_API_KEY="your_api_key"
```

4. 生成进展可视化（图像 + 时间戳日志对齐）：

```bash
python -m kernel_autoresearch.python.cli progress-report \
  --config kernel_autoresearch/configs/default.yaml
```

输出目录：
- `kernel_autoresearch/workspace/progress/index.html`（交互图 + 对照表）
- `kernel_autoresearch/workspace/progress/score_curve.svg`（静态图片）
- `kernel_autoresearch/workspace/progress/timeline.tsv`（可导出明细）

5. 兼容旧版一次性全量搜索：

```bash
python -m kernel_autoresearch.python.cli run \
  --config kernel_autoresearch/configs/default.yaml
```

6. 产物目录：

- `kernel_autoresearch/data/pattern_db/local_patterns.json`
- `kernel_autoresearch/data/pattern_db/cloud_patterns.json`
- `kernel_autoresearch/data/pattern_db/merged_patterns.json`
- `kernel_autoresearch/data/runs/<timestamp>/trials.jsonl`
- `kernel_autoresearch/data/runs/<timestamp>/best_config.json`
- `kernel_autoresearch/data/runs/<timestamp>/report.md`
- `kernel_autoresearch/workspace/kernel_candidate.json`
- `kernel_autoresearch/workspace/results/results.tsv`
- `kernel_autoresearch/workspace/orchestration_state.json`
- `kernel_autoresearch/workspace/run.log`

## C++ Runner 说明

runner 参数示例：

```bash
./kernel_autoresearch/cpp_runner/build/kc_gemm_runner \
  --m 1024 --n 1024 --k 1024 \
  --kernel_variant blocked_pack \
  --bm 64 --bn 64 --bk 64 \
  --pack_a 1 --pack_b 1 --simd 1 \
  --threads 8 --unroll_k 2 \
  --warmup 2 --iters 6 --verify 1 --json 1
```

runner 输出单行 JSON，可被 Python 自动采集。
