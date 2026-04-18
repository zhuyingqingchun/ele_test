#!/usr/bin/env bash
# 模态消融实验 - 只跑 Stage 1+2
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH=.

DATASET="${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz"
ROOT_OUT_DIR="${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v4_ablation_raw_thermal"
BATCH_SIZE=16
SEED=7
DEVICE=cuda

mkdir -p "${ROOT_OUT_DIR}"

run_stage12 () {
  local name="$1"
  local extra_flags="$2"
  local out_root="${ROOT_OUT_DIR}/${name}"
  mkdir -p "${out_root}"

  echo "=============================="
  echo "Running ${name} - Stage 1"
  echo "=============================="
  python3 experiments_smoke_20260316/train_exp1_decoupled_stages.py \
    --stage 1 --dataset "${DATASET}" --output-dir "${out_root}/stage1_encoder_cls" --device "${DEVICE}" \
    --epochs 24 --batch-size "${BATCH_SIZE}" --lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.05 \
    --model-dim 128 --token-dim 256 --seed "${SEED}" --max-samples 0 --feature-mode modality_tf ${extra_flags}

  echo "=============================="
  echo "Running ${name} - Stage 2"
  echo "=============================="
  python3 experiments_smoke_20260316/train_exp1_decoupled_stages.py \
    --stage 2 --dataset "${DATASET}" --output-dir "${out_root}/stage2_signal_fusion" --init "${out_root}/stage1_encoder_cls/best.pt" --device "${DEVICE}" \
    --epochs 20 --batch-size "${BATCH_SIZE}" --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.05 \
    --model-dim 128 --token-dim 256 --fusion-layers 2 --fusion-heads 8 --fusion-ff 768 --pool attn \
    --seed "${SEED}" --max-samples 0 --feature-mode modality_tf ${extra_flags}
}

run_stage12 "A0_full" ""
run_stage12 "A1_no_position" "--zero-pos"
run_stage12 "A2_no_electrical" "--zero-electrical"
run_stage12 "A3_no_thermal" "--zero-thermal"
run_stage12 "A4_no_vibration" "--zero-vibration"

echo "=============================="
echo "All ablation experiments (Stage 1+2) completed!"
echo "Results in: ${ROOT_OUT_DIR}"
echo "=============================="
