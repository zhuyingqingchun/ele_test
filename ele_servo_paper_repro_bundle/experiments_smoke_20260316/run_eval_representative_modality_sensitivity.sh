#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH="${PYTHON_PATH:-python3}"
DEVICE="${DEVICE:-cuda}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v3_modality_tf_control/stage2_signal_fusion/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments_smoke_20260316/reports/representative_modality_sensitivity}"
SEED="${SEED:-7}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
VAL_RATIO="${VAL_RATIO:-0.15}"
TEST_RATIO="${TEST_RATIO:-0.15}"
FEATURE_MODE="${FEATURE_MODE:-}"

"${PYTHON_PATH}" experiments_smoke_20260316/eval_representative_modality_sensitivity.py \
  --dataset "${DATASET}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --max-samples "${MAX_SAMPLES}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --feature-mode "${FEATURE_MODE}" \
  "$@"
