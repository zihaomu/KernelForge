#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_autoresearch.sh"

DEFAULT_CONFIG="kernel_autoresearch/configs/default.yaml"
QUICK_CONFIG="kernel_autoresearch/configs/agent_hybrid_quick.yaml"
INT8_QUICK_CONFIG="kernel_autoresearch/configs/int8_hybrid_quick.yaml"
FP16_QUICK_CONFIG="kernel_autoresearch/configs/fp16_hybrid_quick.yaml"
DEFAULT_WORKSPACE="${REPO_ROOT}/kernel_autoresearch/workspace"
QUICK_WORKSPACE="${REPO_ROOT}/kernel_autoresearch/workspace_agent_quick"
INT8_QUICK_WORKSPACE="${REPO_ROOT}/kernel_autoresearch/workspace_int8_quick"
FP16_QUICK_WORKSPACE="${REPO_ROOT}/kernel_autoresearch/workspace_fp16_quick"

COMMAND="start"
PROFILE="default"
CONFIG_PATH=""
WORKSPACE_PATH=""

usage() {
  cat <<'EOF'
Usage:
  bash kernel_autoresearch/quick_start.sh [start] [options]
  bash kernel_autoresearch/quick_start.sh watch [options]
  bash kernel_autoresearch/quick_start.sh report [options]

Commands:
  start    One-command startup (default): refresh manifest + orchestrate + progress report
  watch    Follow run log in the workspace
  report   Generate progress dashboard once

Options:
  --quick                  Use quick config: kernel_autoresearch/configs/agent_hybrid_quick.yaml
  --int8                   Use int8 quick config: kernel_autoresearch/configs/int8_hybrid_quick.yaml
  --fp16                   Use fp16 quick config: kernel_autoresearch/configs/fp16_hybrid_quick.yaml
  --config <path>          Override config path
  --workspace <path>       Override workspace path (watch command only)
  -h, --help               Show this help

All extra options for `start` are forwarded to:
  bash kernel_autoresearch/run_autoresearch.sh

Examples:
  bash kernel_autoresearch/quick_start.sh
  bash kernel_autoresearch/quick_start.sh --quick --no-install
  bash kernel_autoresearch/quick_start.sh start --agent-mode hybrid
  bash kernel_autoresearch/quick_start.sh watch
  bash kernel_autoresearch/quick_start.sh watch --quick
  bash kernel_autoresearch/quick_start.sh --int8
  bash kernel_autoresearch/quick_start.sh --fp16
  bash kernel_autoresearch/quick_start.sh report --quick
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    start|watch|report)
      COMMAND="$1"
      shift
      ;;
  esac
fi

FORWARDED_ARGS=()
MODE_OVERRIDE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      PROFILE="quick"
      shift
      ;;
    --int8)
      PROFILE="int8_quick"
      shift
      ;;
    --fp16)
      PROFILE="fp16_quick"
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE_PATH="$2"
      shift 2
      ;;
    --mode)
      MODE_OVERRIDE=1
      FORWARDED_ARGS+=("$1" "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
  if [[ "${PROFILE}" == "quick" ]]; then
    CONFIG_PATH="${QUICK_CONFIG}"
  elif [[ "${PROFILE}" == "int8_quick" ]]; then
    CONFIG_PATH="${INT8_QUICK_CONFIG}"
  elif [[ "${PROFILE}" == "fp16_quick" ]]; then
    CONFIG_PATH="${FP16_QUICK_CONFIG}"
  else
    CONFIG_PATH="${DEFAULT_CONFIG}"
  fi
fi

if [[ "${COMMAND}" == "start" ]]; then
  if [[ "${MODE_OVERRIDE}" -eq 0 ]]; then
    FORWARDED_ARGS+=(--mode orchestrate)
  fi
  exec bash "${RUN_SCRIPT}" --config "${CONFIG_PATH}" "${FORWARDED_ARGS[@]}"
fi

if [[ "${COMMAND}" == "watch" ]]; then
  if [[ -z "${WORKSPACE_PATH}" ]]; then
    if [[ "${PROFILE}" == "quick" ]]; then
      WORKSPACE_PATH="${QUICK_WORKSPACE}"
    elif [[ "${PROFILE}" == "int8_quick" ]]; then
      WORKSPACE_PATH="${INT8_QUICK_WORKSPACE}"
    elif [[ "${PROFILE}" == "fp16_quick" ]]; then
      WORKSPACE_PATH="${FP16_QUICK_WORKSPACE}"
    else
      WORKSPACE_PATH="${DEFAULT_WORKSPACE}"
    fi
  elif [[ ! "${WORKSPACE_PATH}" = /* ]]; then
    WORKSPACE_PATH="${REPO_ROOT}/${WORKSPACE_PATH}"
  fi
  RUN_LOG="${WORKSPACE_PATH}/run.log"
  if [[ ! -f "${RUN_LOG}" ]]; then
    echo "[ERROR] run log not found: ${RUN_LOG}" >&2
    exit 1
  fi
  echo "[INFO] Following ${RUN_LOG}"
  exec tail -f "${RUN_LOG}"
fi

# report
if [[ ! "${CONFIG_PATH}" = /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
cd "${REPO_ROOT}"
python -m kernel_autoresearch.python.cli --config "${CONFIG_PATH}" progress-report
