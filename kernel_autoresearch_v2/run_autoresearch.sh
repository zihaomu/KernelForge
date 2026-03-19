#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/kernel_autoresearch_v2/configs/default.yaml"

RUN_MODE="orchestrate" # orchestrate | run-once
REFRESH_MANIFEST=1
GENERATE_PROGRESS=1
AGENT_MODE=""

usage() {
  cat <<'EOF'
Usage:
  bash kernel_autoresearch_v2/run_autoresearch.sh [options]

Options:
  --config <path>                  Config path
  --mode <orchestrate|run-once>    Run full loop or single benchmark
  --agent-mode <mode>              rules_only|hybrid|agent_only
  --skip-refresh                   Skip refresh-harness-manifest
  --skip-progress                  Skip progress-report
  -h, --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --agent-mode)
      AGENT_MODE="$2"
      shift 2
      ;;
    --skip-refresh)
      REFRESH_MANIFEST=0
      shift
      ;;
    --skip-progress)
      GENERATE_PROGRESS=0
      shift
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

if [[ ! "${CONFIG_PATH}" = /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ "${REFRESH_MANIFEST}" -eq 1 ]]; then
  uv run python -m kernel_autoresearch_v2.python.cli \
    --config "${CONFIG_PATH}" \
    refresh-harness-manifest
fi

EXTRA_ARGS=()
if [[ -n "${AGENT_MODE}" ]]; then
  EXTRA_ARGS+=(--agent-mode "${AGENT_MODE}")
fi

uv run python -m kernel_autoresearch_v2.python.cli \
  --config "${CONFIG_PATH}" \
  "${EXTRA_ARGS[@]}" \
  "${RUN_MODE}"

if [[ "${GENERATE_PROGRESS}" -eq 1 ]]; then
  uv run python -m kernel_autoresearch_v2.python.cli \
    --config "${CONFIG_PATH}" \
    progress-report
fi

