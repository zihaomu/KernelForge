# autoresearch

全算子自动优化控制面（CPU + GPU）。

该目录是统一的 autoresearch 框架，不替代 `kernel/` 中的算子源码目录。  
`kernel/` 继续承载实现，`autoresearch/` 负责自动实验、决策和可视化。

## 快速启动

```bash
bash autoresearch/run_autoresearch.sh
```

或：

```bash
uv run python -m autoresearch.core.cli run \
  --config autoresearch/configs/global.yaml \
  --portfolio autoresearch/configs/portfolios/all_ops.yaml
```

## 当前默认接入

1. `cpu_gemm`（可执行，复用 `kernel_autoresearch_v2` 的 C++ runner）
2. `gpu_gemm`（可执行；若无 CUDA/torch 自动标记 skipped）

## 产物

每次 run 会生成：

- `autoresearch/workspace/runs/<run_id>/run_summary.json`
- `autoresearch/workspace/runs/<run_id>/portfolio.tsv`
- `autoresearch/workspace/runs/<run_id>/index.md`
- `autoresearch/workspace/runs/<run_id>/<op_id>/results.tsv`
- `autoresearch/workspace/runs/<run_id>/<op_id>/best_candidate.json`

