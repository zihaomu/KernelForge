#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_REL="kernel_autoresearch/configs/default.yaml"
CONFIG_PATH="${REPO_ROOT}/${CONFIG_REL}"
PYTHON_BIN="python3"
MODE="orchestrate" # orchestrate | run
INSTALL_DEPS=1
REFRESH_MANIFEST=1
GENERATE_PROGRESS=1
CONDA_ENV=""
AGENT_MODE=""
AGENT_MODEL=""

usage() {
  cat <<'EOF'
Usage:
  bash kernel_autoresearch/run_autoresearch.sh [options]

Options:
  --config <path>         Config path (default: kernel_autoresearch/configs/default.yaml)
  --python <bin>          Python executable (default: python3)
  --mode <orchestrate|run>
                          orchestrate: X/E/D/L闭环主流程（推荐）
                          run: 兼容旧版全量搜索流程
  --agent-mode <mode>     Override agent mode: rules_only|hybrid|agent_only
  --agent-model <model>   Override LLM model for agent proposer
  --conda-env <name>      Optional conda env name (e.g. py12_sgl)
  --no-install            Skip pip install -r kernel_autoresearch/requirements.txt
  --skip-refresh          Skip refresh-harness-manifest
  --skip-progress         Skip progress-report generation
  -h, --help              Show this help

Examples:
  bash kernel_autoresearch/run_autoresearch.sh
  bash kernel_autoresearch/run_autoresearch.sh --conda-env py12_sgl
  bash kernel_autoresearch/run_autoresearch.sh --agent-mode hybrid
  bash kernel_autoresearch/run_autoresearch.sh --mode run --skip-progress
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --agent-mode)
      AGENT_MODE="$2"
      shift 2
      ;;
    --agent-model)
      AGENT_MODEL="$2"
      shift 2
      ;;
    --no-install)
      INSTALL_DEPS=0
      shift
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

if [[ "${MODE}" != "orchestrate" && "${MODE}" != "run" ]]; then
  echo "[ERROR] --mode must be one of: orchestrate, run" >&2
  exit 1
fi

if [[ ! "${CONFIG_PATH}" = /* ]]; then
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -n "${CONDA_ENV}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "[INFO] Activated conda env: ${CONDA_ENV}"
  else
    echo "[WARN] conda not found; skip activation for env=${CONDA_ENV}"
  fi
fi

cd "${REPO_ROOT}"
echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Using config: ${CONFIG_PATH}"
echo "[INFO] Python: ${PYTHON_BIN}"
if [[ -n "${AGENT_MODE}" ]]; then
  echo "[INFO] Agent mode override: ${AGENT_MODE}"
fi
if [[ -n "${AGENT_MODEL}" ]]; then
  echo "[INFO] Agent model override: ${AGENT_MODEL}"
fi

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
  echo "[STEP] Installing requirements..."
  "${PYTHON_BIN}" -m pip install -r kernel_autoresearch/requirements.txt
fi

if [[ "${REFRESH_MANIFEST}" -eq 1 ]]; then
  echo "[STEP] Refreshing harness manifest..."
  "${PYTHON_BIN}" -m kernel_autoresearch.python.cli \
    --config "${CONFIG_PATH}" \
    refresh-harness-manifest
fi

echo "[STEP] Running autoresearch mode=${MODE}..."
if [[ "${MODE}" == "orchestrate" ]]; then
  EXTRA_ARGS=()
  if [[ -n "${AGENT_MODE}" ]]; then
    EXTRA_ARGS+=(--agent-mode "${AGENT_MODE}")
  fi
  if [[ -n "${AGENT_MODEL}" ]]; then
    EXTRA_ARGS+=(--agent-model "${AGENT_MODEL}")
  fi
  "${PYTHON_BIN}" -m kernel_autoresearch.python.cli \
    --config "${CONFIG_PATH}" \
    "${EXTRA_ARGS[@]}" \
    orchestrate
else
  "${PYTHON_BIN}" -m kernel_autoresearch.python.cli \
    --config "${CONFIG_PATH}" \
    run
fi

if [[ "${GENERATE_PROGRESS}" -eq 1 ]]; then
  echo "[STEP] Generating progress dashboard..."
  "${PYTHON_BIN}" -m kernel_autoresearch.python.cli \
    --config "${CONFIG_PATH}" \
    progress-report
  WORKSPACE_DIR="$("${PYTHON_BIN}" - <<'PY' "${CONFIG_PATH}"
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
print(cfg["autoresearch"]["workspace_dir"])
PY
)"
  echo "[DONE] Open: ${REPO_ROOT}/${WORKSPACE_DIR}/progress/index.html"
else
  echo "[DONE] Autoresearch finished."
fi
