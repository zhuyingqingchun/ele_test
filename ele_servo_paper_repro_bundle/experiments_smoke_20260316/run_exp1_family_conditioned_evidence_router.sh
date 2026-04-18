#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH="${PYTHON_PATH:-python3}"
DEVICE="${DEVICE:-cuda}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"
BASE_STAGE2="${BASE_STAGE2:-experiments_smoke_20260316/models/exp1_decoupled_v4_ablation_raw_thermal/A0_full/stage2_signal_fusion/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments_smoke_20260316/models/exp1_family_conditioned_evidence_router}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_PATH}" experiments_smoke_20260316/train_exp1_family_conditioned_evidence_router.py \
  --dataset derived_datasets/servo_multimodal_handoff_dataset.npz \
  --corpus derived_datasets/stage1_alignment_corpus.jsonl \
  --base-stage2 "${BASE_STAGE2}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --feature-mode "${FEATURE_MODE}" \
  --epochs "${EPOCHS:-6}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --seed "${SEED:-7}" \
  "$@"
