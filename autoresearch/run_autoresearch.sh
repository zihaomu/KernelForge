#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GLOBAL_CONFIG="${REPO_ROOT}/autoresearch/configs/global.yaml"
PORTFOLIO_CONFIG=""

usage() {
  cat <<'EOF'
Usage:
  bash autoresearch/run_autoresearch.sh [options]

Options:
  --config <path>      Global config path (default: autoresearch/configs/global.yaml)
  --portfolio <path>   Portfolio config path (default from global config)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      GLOBAL_CONFIG="$2"
      shift 2
      ;;
    --portfolio)
      PORTFOLIO_CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! "${GLOBAL_CONFIG}" = /* ]]; then
  GLOBAL_CONFIG="${REPO_ROOT}/${GLOBAL_CONFIG}"
fi

if [[ ! -f "${GLOBAL_CONFIG}" ]]; then
  echo "[ERROR] Global config not found: ${GLOBAL_CONFIG}" >&2
  exit 1
fi

CMD=(uv run python -m autoresearch.core.cli run --config "${GLOBAL_CONFIG}")
if [[ -n "${PORTFOLIO_CONFIG}" ]]; then
  if [[ ! "${PORTFOLIO_CONFIG}" = /* ]]; then
    PORTFOLIO_CONFIG="${REPO_ROOT}/${PORTFOLIO_CONFIG}"
  fi
  CMD+=(--portfolio "${PORTFOLIO_CONFIG}")
fi

cd "${REPO_ROOT}"
"${CMD[@]}"

