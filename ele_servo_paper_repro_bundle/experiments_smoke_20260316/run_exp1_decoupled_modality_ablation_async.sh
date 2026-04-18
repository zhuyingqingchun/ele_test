#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH="${PYTHON_PATH:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

ROOT_OUT_DIR="${ROOT_OUT_DIR:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v4_ablation_raw_thermal_async}"
DATASET="${DATASET:-${PROJECT_ROOT}/derived_datasets/servo_multimodal_handoff_dataset.npz}"
CORPUS="${CORPUS:-${PROJECT_ROOT}/derived_datasets/stage1_alignment_corpus.jsonl}"
TEXT_BACKBONE="${TEXT_BACKBONE:-mock}"
QWEN_PATH="${QWEN_PATH:-/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-7}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"
GPU_MAP="${GPU_MAP:-0,1,2,3,0}"

mkdir -p "${ROOT_OUT_DIR}"
mkdir -p "${ROOT_OUT_DIR}/logs"

EXPERIMENT_NAMES=(
  "A0_full"
  "A1_no_position"
  "A2_no_electrical"
  "A3_no_thermal"
  "A4_no_vibration"
)

EXPERIMENT_FLAGS=(
  ""
  "--zero-pos"
  "--zero-electrical"
  "--zero-thermal"
  "--zero-vibration"
)

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_MAP}"

if [ "${#GPU_ARRAY[@]}" -lt "${#EXPERIMENT_NAMES[@]}" ]; then
  echo "GPU_MAP count (${#GPU_ARRAY[@]}) is smaller than experiments count (${#EXPERIMENT_NAMES[@]})." >&2
  exit 1
fi

echo "Launching ${#EXPERIMENT_NAMES[@]} async jobs"

for i in "${!EXPERIMENT_NAMES[@]}"; do
  name="${EXPERIMENT_NAMES[$i]}"
  flags="${EXPERIMENT_FLAGS[$i]}"
  gpu="${GPU_ARRAY[$i]}"
  out_root="${ROOT_OUT_DIR}/${name}"
  log_file="${ROOT_OUT_DIR}/logs/${name}.log"
  pid_file="${ROOT_OUT_DIR}/logs/${name}.pid"

  mkdir -p "${out_root}"

  echo "[launch] exp=${name} gpu=${gpu} log=${log_file}"

  nohup env CUDA_VISIBLE_DEVICES="${gpu}" bash -lc "
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=.
    '${PYTHON_PATH}' experiments_smoke_20260316/train_exp1_decoupled_stages.py \
      --stage 1 --dataset '${DATASET}' --output-dir '${out_root}/stage1_encoder_cls' --device cuda \
      --epochs 24 --batch-size '${BATCH_SIZE}' --lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.05 \
      --model-dim 128 --token-dim 256 --seed '${SEED}' --max-samples '${MAX_SAMPLES}' --feature-mode '${FEATURE_MODE}' ${flags} && \
    '${PYTHON_PATH}' experiments_smoke_20260316/train_exp1_decoupled_stages.py \
      --stage 2 --dataset '${DATASET}' --output-dir '${out_root}/stage2_signal_fusion' --init '${out_root}/stage1_encoder_cls/best.pt' --device cuda \
      --epochs 20 --batch-size '${BATCH_SIZE}' --lr 3e-4 --weight-decay 1e-4 --label-smoothing 0.05 \
      --model-dim 128 --token-dim 256 --fusion-layers 2 --fusion-heads 8 --fusion-ff 768 --pool attn \
      --seed '${SEED}' --max-samples '${MAX_SAMPLES}' --feature-mode '${FEATURE_MODE}' ${flags} && \
    '${PYTHON_PATH}' experiments_smoke_20260316/train_exp1_decoupled_stages.py \
      --stage 3 --dataset '${DATASET}' --corpus '${CORPUS}' --output-dir '${out_root}/stage3_signal_text_align' --init '${out_root}/stage2_signal_fusion/best.pt' --device cuda \
      --epochs 16 --batch-size '${BATCH_SIZE}' --lr 2e-4 --weight-decay 1e-4 --label-smoothing 0.03 --lambda-align 0.15 \
      --model-dim 128 --token-dim 256 --text-backbone '${TEXT_BACKBONE}' --qwen-path '${QWEN_PATH}' --text-batch-size 8 \
      --seed '${SEED}' --max-samples '${MAX_SAMPLES}' --feature-mode '${FEATURE_MODE}' ${flags} && \
    '${PYTHON_PATH}' experiments_smoke_20260316/train_exp1_decoupled_stages.py \
      --stage 4 --dataset '${DATASET}' --corpus '${CORPUS}' --output-dir '${out_root}/stage4_signal_text_llm' --init '${out_root}/stage3_signal_text_align/best.pt' --device cuda \
      --epochs 16 --batch-size '${BATCH_SIZE}' --lr 2e-4 --weight-decay 1e-4 --label-smoothing 0.03 --lambda-align 0.08 \
      --model-dim 128 --token-dim 256 --text-backbone '${TEXT_BACKBONE}' --qwen-path '${QWEN_PATH}' --text-batch-size 8 \
      --llm-layers 4 --llm-heads 8 --llm-ff 768 --pool attn --seed '${SEED}' --max-samples '${MAX_SAMPLES}' --feature-mode '${FEATURE_MODE}' ${flags}
  " > "${log_file}" 2>&1 < /dev/null &

  echo $! > "${pid_file}"
done

echo "Started all jobs."
echo "Logs: ${ROOT_OUT_DIR}/logs"
