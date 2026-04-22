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

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_topk_duallevel_align_v2}"
DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/derived_datasets/stage1_alignment_corpus.jsonl}"
TEXT_BACKBONE="${TEXT_BACKBONE:-mock}"
QWEN_PATH="${QWEN_PATH:-/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-7}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"

BASE_STAGE2_CKPT="${BASE_STAGE2_CKPT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_fault_aware_gate_multiview_align_v1/stage2_signal_fusion/best.pt}"

mkdir -p "${OUT_ROOT}"

STAGE3_DIR="${OUT_ROOT}/stage3_topk_duallevel_align"
STAGE4_DIR="${OUT_ROOT}/stage4_topk_duallevel_align"

"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_topk_duallevel_align_v2.py \
  --stage 3 --dataset "${DATASET}" --corpus "${CORPUS}" --output-dir "${STAGE3_DIR}" \
  --init "${BASE_STAGE2_CKPT}" --device "${DEVICE}" --epochs 16 --batch-size "${BATCH_SIZE}" \
  --lr 2e-4 --weight-decay 1e-4 --seed "${SEED}" --max-samples "${MAX_SAMPLES}" \
  --feature-mode "${FEATURE_MODE}" --text-backbone "${TEXT_BACKBONE}" --qwen-path "${QWEN_PATH}" \
  --text-batch-size 8 --quality-drop-prob 0.10 --lambda-combined 0.25 --lambda-evidence 0.35 \
  --lambda-mechanism 0.35 --lambda-contrast 0.15 --lambda-listwise 0.25 --lambda-hardneg 0.20 --lambda-proto 0.15

"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_topk_duallevel_align_v2.py \
  --stage 4 --dataset "${DATASET}" --corpus "${CORPUS}" --output-dir "${STAGE4_DIR}" \
  --init "${STAGE3_DIR}/best.pt" --device "${DEVICE}" --epochs 16 --batch-size "${BATCH_SIZE}" \
  --lr 2e-4 --weight-decay 1e-4 --seed "${SEED}" --max-samples "${MAX_SAMPLES}" \
  --feature-mode "${FEATURE_MODE}" --text-backbone "${TEXT_BACKBONE}" --qwen-path "${QWEN_PATH}" \
  --text-batch-size 8 --quality-drop-prob 0.10 --lambda-combined 0.20 --lambda-evidence 0.40 \
  --lambda-mechanism 0.40 --lambda-contrast 0.20 --lambda-listwise 0.30 --lambda-hardneg 0.25 --lambda-proto 0.15 \
  --llm-layers 4 --llm-heads 8 --llm-ff 768 --pool attn
