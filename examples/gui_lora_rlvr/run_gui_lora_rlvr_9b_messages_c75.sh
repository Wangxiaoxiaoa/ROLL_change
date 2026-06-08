#!/usr/bin/env bash
set -euo pipefail

ROLL_ROOT="${ROLL_ROOT:-/mnt/data0/xiao/RL/ROLL}"
AREAL_ROOT="${AREAL_ROOT:-/mnt/data0/xiao/RL/AReaL}"
PYTHON_BIN="${PYTHON_BIN:-${ROLL_ROOT}/.venv-roll-rl/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${AREAL_ROOT}/.venv-vllm/bin/python"
fi

CONFIG_NAME="${CONFIG_NAME:-roll_gui_lora_rlvr_9b_messages_c75}"
RUN_NAME="${RUN_NAME:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${ROLL_ROOT}:${ROLL_ROOT}/examples:${AREAL_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ROLL_ROOT}/.venv-roll-rl/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export DS_SKIP_CUDA_CHECK=1
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export RAY_DEDUP_LOGS=0
export TMPDIR="${TMPDIR:-/mnt/data0/raytmp_roll}"
export HF_HOME="${ROLL_ROOT}/examples/gui_lora_rlvr/hf_home"
export HF_DATASETS_CACHE="${ROLL_ROOT}/examples/gui_lora_rlvr/hf_datasets_cache"
export ROLL_RAY_TEMP_DIR="${ROLL_RAY_TEMP_DIR:-/mnt/data0/raytmp_roll}"
export GUI_ROLL_ROLLOUT_LOG="${ROLL_ROOT}/examples/gui_lora_rlvr/output/rollout_jsonl/${RUN_NAME}.jsonl"

mkdir -p \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${ROLL_RAY_TEMP_DIR}" \
  "${ROLL_ROOT}/examples/gui_lora_rlvr/output/logs/${RUN_NAME}" \
  "$(dirname "${GUI_ROLL_ROLLOUT_LOG}")"

if [[ "${ROLL_RESTART_RAY:-1}" == "1" ]]; then
  ray stop --force >/dev/null 2>&1 || true
fi

cd "${ROLL_ROOT}/examples"
exec "${PYTHON_BIN}" start_rlvr_vl_pipeline.py \
  --config_path gui_lora_rlvr \
  --config_name "${CONFIG_NAME}" \
  2>&1 | tee "${ROLL_ROOT}/examples/gui_lora_rlvr/output/logs/${RUN_NAME}/train.log"
