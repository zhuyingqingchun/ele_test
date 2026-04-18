#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH="${PYTHON_PATH:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

DATASET="${DATASET:-${PROJECT_ROOT}/experiments_smoke_20260316/traditional/window_dataset}"
ROOT_SAVE_DIR="${ROOT_SAVE_DIR:-${PROJECT_ROOT}/experiments_smoke_20260316/models/cnn_tcn_modality_ablation_current_modalities_async}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-7}"
GPU_MAP="${GPU_MAP:-0,1,2,3,0}"
KEEP_COLUMNS="${KEEP_COLUMNS:-theta_meas_deg,theta_motor_meas_deg,omega_motor_meas_deg_s,current_q_meas_a,current_d_meas_a,bus_current_meas_a,phase_voltage_a_meas_v,phase_voltage_b_meas_v,phase_voltage_c_meas_v,available_bus_voltage_v,voltage_meas_v,winding_temp_c,housing_temp_c,vibration_accel_mps2}"

mkdir -p "${ROOT_SAVE_DIR}"
mkdir -p "${ROOT_SAVE_DIR}/logs"

EXPERIMENT_NAMES=(
  "cnn_tcn_full"
  "cnn_tcn_no_position"
  "cnn_tcn_no_electrical"
  "cnn_tcn_no_thermal"
  "cnn_tcn_no_vibration"
)

EXPERIMENT_ZERO_COLUMNS=(
  ""
  "theta_meas_deg,theta_motor_meas_deg,omega_motor_meas_deg_s"
  "current_q_meas_a,current_d_meas_a,bus_current_meas_a,phase_voltage_a_meas_v,phase_voltage_b_meas_v,phase_voltage_c_meas_v,available_bus_voltage_v,voltage_meas_v"
  "winding_temp_c,housing_temp_c"
  "vibration_accel_mps2"
)

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_MAP}"

if [ "${#GPU_ARRAY[@]}" -lt "${#EXPERIMENT_NAMES[@]}" ]; then
  echo "GPU_MAP count (${#GPU_ARRAY[@]}) is smaller than experiments count (${#EXPERIMENT_NAMES[@]})." >&2
  exit 1
fi

echo "Launching ${#EXPERIMENT_NAMES[@]} async jobs"

for i in "${!EXPERIMENT_NAMES[@]}"; do
  name="${EXPERIMENT_NAMES[$i]}"
  zero_columns="${EXPERIMENT_ZERO_COLUMNS[$i]}"
  gpu="${GPU_ARRAY[$i]}"
  save_dir="${ROOT_SAVE_DIR}/${name}"
  log_file="${ROOT_SAVE_DIR}/logs/${name}.log"
  pid_file="${ROOT_SAVE_DIR}/logs/${name}.pid"

  mkdir -p "${save_dir}"

  echo "[launch] exp=${name} gpu=${gpu} log=${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PYTHON_PATH}" experiments_smoke_20260316/train_timeseries_baseline_randomsplit.py \
    --dataset "${DATASET}" \
    --model cnn_tcn \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --seed "${SEED}" \
    --device "cuda" \
    --save-dir "${save_dir}" \
    --run-name "${name}" \
    --keep-columns "${KEEP_COLUMNS}" \
    --zero-columns "${zero_columns}" \
    > "${log_file}" 2>&1 &

  echo $! > "${pid_file}"
done

echo "Started all jobs."
echo "Logs: ${ROOT_SAVE_DIR}/logs"
