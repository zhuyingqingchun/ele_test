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

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_fault_aware_gate_multiview_align_v1}"
DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/derived_datasets/stage1_alignment_corpus.jsonl}"
TEXT_BACKBONE="${TEXT_BACKBONE:-mock}"
QWEN_PATH="${QWEN_PATH:-/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-7}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"
MODALITY_DROP_PROB="${MODALITY_DROP_PROB:-0.10}"

mkdir -p "${OUT_ROOT}"

COMMON_QUALITY_ARGS="--quality-aware-fusion --quality-hidden-dim 128 --quality-min-gate 0.10 --lambda-quality 0.10 --quality-drop-prob ${MODALITY_DROP_PROB}"
COMMON_ALIGN_ARGS="--fault-aware-gate --fault-gate-hidden-dim 128 --multiview-alignment --lambda-modality-align 0.15 --lambda-gate-prior 0.10 --export-counterfactual"

echo "[fault-aware-gate-multiview] Starting Stage 1..."
"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 1 --dataset "${DATASET}" --output-dir "${OUT_ROOT}/stage1_encoder_cls" --device "${DEVICE}" \
  --epochs 36 --batch-size "${BATCH_SIZE}" --lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.05 \
  --model-dim 128 --token-dim 256 --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
  ${COMMON_QUALITY_ARGS}

echo "[fault-aware-gate-multiview] Starting Stage 2..."
"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 2 --dataset "${DATASET}" --output-dir "${OUT_ROOT}/stage2_signal_fusion" --init "${OUT_ROOT}/stage1_encoder_cls/best.pt" --device "${DEVICE}" \
  --epochs 20 --batch-size "${BATCH_SIZE}" --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.05 \
  --model-dim 128 --token-dim 256 --fusion-layers 2 --fusion-heads 8 --fusion-ff 768 --pool attn \
  --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
  ${COMMON_QUALITY_ARGS}

echo "[fault-aware-gate-multiview] Starting Stage 3..."
"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 3 --dataset "${DATASET}" --corpus "${CORPUS}" --output-dir "${OUT_ROOT}/stage3_signal_text_align" --init "${OUT_ROOT}/stage2_signal_fusion/best.pt" --device "${DEVICE}" \
  --epochs 16 --batch-size "${BATCH_SIZE}" --lr 2e-4 --weight-decay 1e-4 --label-smoothing 0.03 --lambda-align 0.15 \
  --model-dim 128 --token-dim 256 --text-backbone "${TEXT_BACKBONE}" --qwen-path "${QWEN_PATH}" --text-batch-size 8 \
  --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
  ${COMMON_QUALITY_ARGS} ${COMMON_ALIGN_ARGS}

echo "[fault-aware-gate-multiview] Starting Stage 4..."
"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 4 --dataset "${DATASET}" --corpus "${CORPUS}" --output-dir "${OUT_ROOT}/stage4_signal_text_llm" --init "${OUT_ROOT}/stage3_signal_text_align/best.pt" --device "${DEVICE}" \
  --epochs 16 --batch-size "${BATCH_SIZE}" --lr 2e-4 --weight-decay 1e-4 --label-smoothing 0.03 --lambda-align 0.08 \
  --model-dim 128 --token-dim 256 --text-backbone "${TEXT_BACKBONE}" --qwen-path "${QWEN_PATH}" --text-batch-size 8 \
  --llm-layers 4 --llm-heads 8 --llm-ff 768 --pool attn --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
  ${COMMON_QUALITY_ARGS} ${COMMON_ALIGN_ARGS}

echo "[fault-aware-gate-multiview] All stages completed!"
echo "[fault-aware-gate-multiview] Results:"
echo "  ${OUT_ROOT}/stage2_signal_fusion/report.json"
echo "  ${OUT_ROOT}/stage3_signal_text_align/report.json"
echo "  ${OUT_ROOT}/stage4_signal_text_llm/report.json"
