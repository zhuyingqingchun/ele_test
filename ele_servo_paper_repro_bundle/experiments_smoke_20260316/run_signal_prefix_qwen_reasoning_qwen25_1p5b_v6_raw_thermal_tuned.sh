#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH="${PYTHON_PATH:-python3}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES

DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/derived_datasets/stage1_alignment_corpus.jsonl}"
SIGNAL_INIT="${SIGNAL_INIT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v6_modality_tf_control_raw_thermal/stage2_signal_fusion/best.pt}"
QWEN_PATH="${QWEN_PATH:-/mnt/PRO6000_disk/models/Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiments_smoke_20260316/models/signal_prefix_qwen_reasoning_qwen25_1p5b_v6_raw_thermal_tuned}"

"${PYTHON_PATH}" experiments_smoke_20260316/train_signal_prefix_qwen_reasoning.py \
  --dataset "${DATASET}" \
  --corpus "${CORPUS}" \
  --signal-init "${SIGNAL_INIT}" \
  --qwen-path "${QWEN_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --feature-mode modality_tf \
  --batch-size 4 \
  --epochs 6 \
  --lr 2e-4 \
  --weight-decay 0.0 \
  --seed 7 \
  --max-samples 0 \
  --prefix-length 16 \
  --grad-accum 4 \
  --gen-max-new-tokens 96 \
  --max-length 320 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
