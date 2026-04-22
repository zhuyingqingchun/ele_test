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

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_topk_listwise_token_alignment_v1}"
DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/derived_datasets/stage1_alignment_corpus.jsonl}"
TEXT_BACKBONE="${TEXT_BACKBONE:-mock}"
QWEN_PATH="${QWEN_PATH:-/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-7}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"

STAGE3_INIT="${STAGE3_INIT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_fault_aware_gate_multiview_align_v1/stage2_signal_fusion/best.pt}"
STAGE3_DIR="${OUT_ROOT}/stage3_topk_listwise"
STAGE4_DIR="${OUT_ROOT}/stage4_topk_listwise"

mkdir -p "${OUT_ROOT}"

"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_topk_listwise_token_alignment_v1.py \
  --stage 3 \
  --dataset "${DATASET}" \
  --corpus "${CORPUS}" \
  --output-dir "${STAGE3_DIR}" \
  --init "${STAGE3_INIT}" \
  --device "${DEVICE}" \
  --epochs 12 \
  --batch-size "${BATCH_SIZE}" \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --text-backbone "${TEXT_BACKBONE}" \
  --qwen-path "${QWEN_PATH}" \
  --feature-mode "${FEATURE_MODE}" \
  --quality-aware-fusion \
  --quality-drop-prob 0.10

"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_topk_listwise_token_alignment_v1.py \
  --stage 4 \
  --dataset "${DATASET}" \
  --corpus "${CORPUS}" \
  --output-dir "${STAGE4_DIR}" \
  --init "${STAGE3_DIR}/best.pt" \
  --device "${DEVICE}" \
  --epochs 12 \
  --batch-size "${BATCH_SIZE}" \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --text-backbone "${TEXT_BACKBONE}" \
  --qwen-path "${QWEN_PATH}" \
  --feature-mode "${FEATURE_MODE}" \
  --quality-aware-fusion \
  --quality-drop-prob 0.10 \
  --pool attn
