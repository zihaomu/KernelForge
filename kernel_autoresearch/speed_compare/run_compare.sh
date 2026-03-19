#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/kernel_autoresearch/speed_compare/results_all"

python3 "${ROOT_DIR}/kernel_autoresearch/speed_compare/compare_gemm_vs_openblas.py" \
  --config "${ROOT_DIR}/kernel_autoresearch/configs/default.yaml" \
  --threads -1 \
  --warmup 5 \
  --iters 30 \
  --jobs 8 \
  --out-dir "${OUT_DIR}"

echo "[DONE] Report: ${OUT_DIR}/report.md"
