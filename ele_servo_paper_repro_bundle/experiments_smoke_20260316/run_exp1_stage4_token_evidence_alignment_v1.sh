#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_PATH="${PYTHON_PATH:-python}"
DEVICE="${DEVICE:-cuda}"

DATASET="${DATASET:-${PROJECT_ROOT}/ele_servo_paper_repro_bundle/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/ele_servo_paper_repro_bundle/derived_datasets/stage1_alignment_corpus.jsonl}"

BASE_DIR="${BASE_DIR:-${SCRIPT_DIR}/models/exp1_stage4_token_evidence_alignment_v1}"
STAGE3_INIT="${STAGE3_INIT:-${SCRIPT_DIR}/models/exp1_topk_duallevel_align_v2/stage3_topk_duallevel_align/best.pt}"
STAGE4_DIR="${BASE_DIR}/stage4_token_alignment"

mkdir -p "${STAGE4_DIR}"

"${PYTHON_PATH}" "${SCRIPT_DIR}/train_exp1_stage4_token_evidence_alignment_v1.py"   --dataset "${DATASET}"   --corpus "${CORPUS}"   --output-dir "${STAGE4_DIR}"   --init "${STAGE3_INIT}"   --device "${DEVICE}"   --epochs "${EPOCHS:-16}"   --batch-size "${BATCH_SIZE:-16}"   --lr "${LR:-2e-4}"   --weight-decay "${WEIGHT_DECAY:-1e-4}"   --seed "${SEED:-7}"   --model-dim "${MODEL_DIM:-128}"   --token-dim "${TOKEN_DIM:-256}"   --text-backbone "${TEXT_BACKBONE:-mock}"   --feature-mode "${FEATURE_MODE:-modality_tf}"   --quality-drop-prob "${QUALITY_DROP_PROB:-0.10}"   --quality-hidden-dim "${QUALITY_HIDDEN_DIM:-128}"   --fault-gate-hidden-dim "${FAULT_GATE_HIDDEN_DIM:-128}"   --llm-layers "${LLM_LAYERS:-4}"   --llm-heads "${LLM_HEADS:-8}"   --llm-ff "${LLM_FF:-768}"   --pool "${POOL:-attn}"   --lambda-combined "${LAMBDA_COMBINED:-0.20}"   --lambda-evidence "${LAMBDA_EVIDENCE:-0.35}"   --lambda-mechanism "${LAMBDA_MECHANISM:-0.35}"   --lambda-contrast "${LAMBDA_CONTRAST:-0.15}"   --lambda-listwise "${LAMBDA_LISTWISE:-0.20}"   --lambda-token "${LAMBDA_TOKEN:-0.25}"   --lambda-token-sparse "${LAMBDA_TOKEN_SPARSE:-0.05}"   --lambda-hardneg "${LAMBDA_HARDNEG:-0.20}"   --lambda-proto "${LAMBDA_PROTO:-0.12}"   --lambda-quality "${LAMBDA_QUALITY:-0.05}"
