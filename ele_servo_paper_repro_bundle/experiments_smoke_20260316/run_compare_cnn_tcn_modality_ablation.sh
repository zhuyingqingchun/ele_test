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

DATASET="${DATASET:-${PROJECT_ROOT}/experiments_smoke_20260316/traditional/window_dataset}"
ROOT_SAVE_DIR="${ROOT_SAVE_DIR:-${PROJECT_ROOT}/experiments_smoke_20260316/models/cnn_tcn_modality_ablation_current_modalities}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-7}"
KEEP_COLUMNS="${KEEP_COLUMNS:-theta_meas_deg,theta_motor_meas_deg,omega_motor_meas_deg_s,current_q_meas_a,current_d_meas_a,bus_current_meas_a,phase_voltage_a_meas_v,phase_voltage_b_meas_v,phase_voltage_c_meas_v,available_bus_voltage_v,voltage_meas_v,winding_temp_c,housing_temp_c,vibration_accel_mps2}"

mkdir -p "${ROOT_SAVE_DIR}"

run_one () {
  local name="$1"
  local zero_columns="$2"
  local save_dir="${ROOT_SAVE_DIR}/${name}"
  mkdir -p "${save_dir}"

  echo "=============================="
  echo "Running ${name}"
  echo "=============================="

  "${PYTHON_PATH}" experiments_smoke_20260316/train_timeseries_baseline_randomsplit.py \
    --dataset "${DATASET}" \
    --model cnn_tcn \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --save-dir "${save_dir}" \
    --run-name "${name}" \
    --keep-columns "${KEEP_COLUMNS}" \
    --zero-columns "${zero_columns}"
}

run_one "cnn_tcn_full" ""
run_one "cnn_tcn_no_position" "theta_meas_deg,theta_motor_meas_deg,omega_motor_meas_deg_s"
run_one "cnn_tcn_no_electrical" "current_q_meas_a,current_d_meas_a,bus_current_meas_a,phase_voltage_a_meas_v,phase_voltage_b_meas_v,phase_voltage_c_meas_v,available_bus_voltage_v,voltage_meas_v"
run_one "cnn_tcn_no_thermal" "winding_temp_c,housing_temp_c"
run_one "cnn_tcn_no_vibration" "vibration_accel_mps2"
