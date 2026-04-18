#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH="${PYTHON_PATH:-/home/a/miniconda3/envs/swdtorch12/bin/python}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES

DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/derived_datasets/stage1_alignment_corpus.jsonl}"
SIGNAL_INIT="${SIGNAL_INIT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v3_modality_tf_control/stage2_signal_fusion/best.pt}"
QWEN_PATH="${QWEN_PATH:-/mnt/PRO6000_disk/models/Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments_smoke_20260316/models/signal_prefix_qwen_reasoning_qwen25_1p5b}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-7}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PREFIX_LENGTH="${PREFIX_LENGTH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-96}"
MAX_LENGTH="${MAX_LENGTH:-320}"

"${PYTHON_PATH}" experiments_smoke_20260316/train_signal_prefix_qwen_reasoning.py \
  --dataset "${DATASET}" \
  --corpus "${CORPUS}" \
  --signal-init "${SIGNAL_INIT}" \
  --qwen-path "${QWEN_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --feature-mode "${FEATURE_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --max-samples "${MAX_SAMPLES}" \
  --prefix-length "${PREFIX_LENGTH}" \
  --grad-accum "${GRAD_ACCUM}" \
  --gen-max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
  --max-length "${MAX_LENGTH}" \
  "$@"
