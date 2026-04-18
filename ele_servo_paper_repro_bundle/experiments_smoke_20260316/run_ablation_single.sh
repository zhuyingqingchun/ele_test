#!/usr/bin/env bash
# 模态消融实验 - 单配置 Stage 1+2
# 用法: bash run_ablation_single.sh <name> <extra_flags> <gpu_id>
set -euo pipefail

NAME="${1:?Usage: $0 <name> <extra_flags> <gpu_id>}"
EXTRA_FLAGS="${2:-}"
GPU_ID="${3:-0}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH=.

DATASET="${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz"
ROOT_OUT_DIR="${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v4_ablation_raw_thermal"
BATCH_SIZE=16
SEED=7
DEVICE=cuda

mkdir -p "${ROOT_OUT_DIR}/${NAME}"

LOG_FILE="${ROOT_OUT_DIR}/${NAME}/ablation_stage12.log"

echo "[$(date)] Starting ${NAME} on GPU ${GPU_ID}" | tee "${LOG_FILE}"

echo "==============================" | tee -a "${LOG_FILE}"
echo "Running ${NAME} - Stage 1" | tee -a "${LOG_FILE}"
echo "==============================" | tee -a "${LOG_FILE}"

python3 experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 1 --dataset "${DATASET}" --output-dir "${ROOT_OUT_DIR}/${NAME}/stage1_encoder_cls" --device "${DEVICE}" \
  --epochs 24 --batch-size "${BATCH_SIZE}" --lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.05 \
  --model-dim 128 --token-dim 256 --seed "${SEED}" --max-samples 0 --feature-mode modality_tf ${EXTRA_FLAGS} 2>&1 | tee -a "${LOG_FILE}"

echo "==============================" | tee -a "${LOG_FILE}"
echo "Running ${NAME} - Stage 2" | tee -a "${LOG_FILE}"
echo "==============================" | tee -a "${LOG_FILE}"

python3 experiments_smoke_20260316/train_exp1_decoupled_stages.py \
  --stage 2 --dataset "${DATASET}" --output-dir "${ROOT_OUT_DIR}/${NAME}/stage2_signal_fusion" --init "${ROOT_OUT_DIR}/${NAME}/stage1_encoder_cls/best.pt" --device "${DEVICE}" \
  --epochs 20 --batch-size "${BATCH_SIZE}" --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.05 \
  --model-dim 128 --token-dim 256 --fusion-layers 2 --fusion-heads 8 --fusion-ff 768 --pool attn \
  --seed "${SEED}" --max-samples 0 --feature-mode modality_tf ${EXTRA_FLAGS} 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date)] ${NAME} completed!" | tee -a "${LOG_FILE}"
