#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_PATH="${PYTHON_PATH:-python3}"
RUN_SIM="${RUN_SIM:-1}"
RUN_DATASET="${RUN_DATASET:-1}"
RUN_MAIN="${RUN_MAIN:-1}"
RUN_SINGLE="${RUN_SINGLE:-0}"
RUN_ABLATION="${RUN_ABLATION:-0}"
RUN_TRADITIONAL="${RUN_TRADITIONAL:-0}"
RUN_REASONING="${RUN_REASONING:-0}"

REPEAT_COUNT="${REPEAT_COUNT:-1}"
FINAL_TIME="${FINAL_TIME:-4.0}"
DT="${DT:-0.001}"
RANDOM_SEED="${RANDOM_SEED:-7}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_PATH=${PYTHON_PATH}"
echo "RUN_SIM=${RUN_SIM}"
echo "RUN_DATASET=${RUN_DATASET}"
echo "RUN_MAIN=${RUN_MAIN}"
echo "RUN_SINGLE=${RUN_SINGLE}"
echo "RUN_ABLATION=${RUN_ABLATION}"
echo "RUN_TRADITIONAL=${RUN_TRADITIONAL}"
echo "RUN_REASONING=${RUN_REASONING}"

if [[ "${RUN_SIM}" == "1" ]]; then
  echo "[1/7] Running servo simulation pipeline"
  "${PYTHON_PATH}" src/run_servo_pipeline_expanded.py \
    --repeat-count "${REPEAT_COUNT}" \
    --final-time "${FINAL_TIME}" \
    --dt "${DT}" \
    --random-seed "${RANDOM_SEED}"
fi

if [[ "${RUN_DATASET}" == "1" ]]; then
  echo "[2/7] Building handoff datasets"
  "${PYTHON_PATH}" src/build_servo_handoff_dataset.py
  "${PYTHON_PATH}" src/build_stage1_alignment_corpus.py
  "${PYTHON_PATH}" experiments_smoke_20260316/prepare_traditional_window_dataset.py
fi

if [[ "${RUN_MAIN}" == "1" ]]; then
  echo "[3/7] Running main four-modality experiment"
  PYTHON_PATH="${PYTHON_PATH}" bash experiments_smoke_20260316/run_exp1_decoupled_modality_tf_control_raw_thermal_full.sh
fi

if [[ "${RUN_SINGLE}" == "1" ]]; then
  echo "[4/7] Running single-modality experiments"
  PYTHON_PATH="${PYTHON_PATH}" bash experiments_smoke_20260316/run_exp1_decoupled_single_modality.sh
fi

if [[ "${RUN_ABLATION}" == "1" ]]; then
  echo "[5/7] Running modality ablation experiments"
  PYTHON_PATH="${PYTHON_PATH}" bash experiments_smoke_20260316/run_exp1_decoupled_modality_ablation.sh
fi

if [[ "${RUN_TRADITIONAL}" == "1" ]]; then
  echo "[6/7] Running traditional baseline comparisons"
  PYTHON_PATH="${PYTHON_PATH}" bash experiments_smoke_20260316/run_compare_all_current_modalities.sh
fi

if [[ "${RUN_REASONING}" == "1" ]]; then
  echo "[7/7] Running structured reasoning extension"
  if [[ -z "${QWEN_PATH:-}" ]]; then
    echo "RUN_REASONING=1 requires QWEN_PATH to be set." >&2
    exit 1
  fi
  PYTHON_PATH="${PYTHON_PATH}" QWEN_PATH="${QWEN_PATH}" \
    bash experiments_smoke_20260316/run_signal_prefix_qwen_reasoning_qwen25_1p5b_v6_raw_thermal_tuned.sh
fi

echo "Pipeline finished."
