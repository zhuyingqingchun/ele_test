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

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_evidence_mechanism_primary_alignment_v1}"
DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/derived_datasets/stage1_alignment_corpus.jsonl}"
TEXT_BACKBONE="${TEXT_BACKBONE:-mock}"
QWEN_PATH="${QWEN_PATH:-/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-7}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"
RUN_STAGE4="${RUN_STAGE4:-1}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-36}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-16}"
STAGE4_EPOCHS="${STAGE4_EPOCHS:-16}"

QUALITY_AWARE_FUSION="${QUALITY_AWARE_FUSION:-1}"
MODALITY_DROP_PROB="${MODALITY_DROP_PROB:-0.10}"

mkdir -p "${OUT_ROOT}"
STAGE1_DIR="${OUT_ROOT}/stage1_encoder_cls"
STAGE2_DIR="${OUT_ROOT}/stage2_signal_fusion"
STAGE3_DIR="${OUT_ROOT}/stage3_fault_aware_multiview"
STAGE4_DIR="${OUT_ROOT}/stage4_fault_aware_multiview_llm"

QUALITY_ARGS=""
if [[ "${QUALITY_AWARE_FUSION}" == "1" ]]; then
  QUALITY_ARGS="--quality-aware-fusion --quality-hidden-dim 128 --quality-min-gate 0.10 --lambda-quality 0.10"
fi
if [[ "$(${PYTHON_PATH} -c 'import sys; print(1 if float(sys.argv[1]) > 0 else 0)' "${MODALITY_DROP_PROB}")" == "1" ]]; then
  QUALITY_ARGS="${QUALITY_ARGS} --quality-drop-prob ${MODALITY_DROP_PROB}"
fi

echo "[Round 23] Starting Stage 1..."
"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 1 --dataset "${DATASET}" --output-dir "${STAGE1_DIR}" --device "${DEVICE}" \
  --epochs "${STAGE1_EPOCHS}" --batch-size "${BATCH_SIZE}" --lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.05 \
  --model-dim 128 --token-dim 256 --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
  ${QUALITY_ARGS}

echo "[Round 23] Starting Stage 2..."
"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 2 --dataset "${DATASET}" --output-dir "${STAGE2_DIR}" --init "${STAGE1_DIR}/best.pt" --device "${DEVICE}" \
  --epochs "${STAGE2_EPOCHS}" --batch-size "${BATCH_SIZE}" --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.05 \
  --model-dim 128 --token-dim 256 --fusion-layers 2 --fusion-heads 8 --fusion-ff 768 --pool attn \
  --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
  ${QUALITY_ARGS}

echo "[Round 23] Starting Stage 3 (Fault-aware Multi-view)..."
"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_fault_aware_gate_multiview.py \
  --stage 3 --dataset "${DATASET}" --corpus "${CORPUS}" --output-dir "${STAGE3_DIR}" --init "${STAGE2_DIR}/best.pt" --device "${DEVICE}" \
  --epochs "${STAGE3_EPOCHS}" --batch-size "${BATCH_SIZE}" --lr 2e-4 --weight-decay 1e-4 --label-smoothing 0.03 --lambda-align 0.15 \
  --model-dim 128 --token-dim 256 --text-backbone "${TEXT_BACKBONE}" --qwen-path "${QWEN_PATH}" --text-batch-size 8 \
  --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
  --fault-gate-hidden-dim 128 --fault-gate-min 0.15 \
  --lambda-combined 1.0 --lambda-evidence 1.0 --lambda-mechanism 1.0 --lambda-contrast 0.35 \
  --lambda-modality-align 0.25 --lambda-proto 0.10 --lambda-hard-negative 0.10 --hard-negative-margin 0.15 \
  ${QUALITY_ARGS}

if [[ "${RUN_STAGE4}" == "1" ]]; then
  echo "[Round 23] Starting Stage 4 (Fault-aware Multi-view LLM)..."
  "${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_fault_aware_gate_multiview.py \
    --stage 4 --dataset "${DATASET}" --corpus "${CORPUS}" --output-dir "${STAGE4_DIR}" --init "${STAGE3_DIR}/best.pt" --device "${DEVICE}" \
    --epochs "${STAGE4_EPOCHS}" --batch-size "${BATCH_SIZE}" --lr 2e-4 --weight-decay 1e-4 --label-smoothing 0.03 --lambda-align 0.10 \
    --model-dim 128 --token-dim 256 --text-backbone "${TEXT_BACKBONE}" --qwen-path "${QWEN_PATH}" --text-batch-size 8 \
    --fusion-layers 4 --fusion-heads 8 --fusion-ff 768 --pool text --seed "${SEED}" --max-samples "${MAX_SAMPLES}" --feature-mode "${FEATURE_MODE}" \
    --fault-gate-hidden-dim 128 --fault-gate-min 0.15 \
    --lambda-combined 1.0 --lambda-evidence 0.9 --lambda-mechanism 1.1 --lambda-contrast 0.35 \
    --lambda-modality-align 0.25 --lambda-proto 0.10 --lambda-hard-negative 0.10 --hard-negative-margin 0.15 \
    ${QUALITY_ARGS}
fi

echo "[Round 23] All stages completed!"
echo "Results:"
echo "  ${STAGE1_DIR}/report.json"
echo "  ${STAGE2_DIR}/report.json"
echo "  ${STAGE3_DIR}/report.json"
echo "  ${STAGE4_DIR}/report.json"
