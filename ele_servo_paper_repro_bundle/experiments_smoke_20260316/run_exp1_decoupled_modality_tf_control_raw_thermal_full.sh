#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v6_modality_tf_control_raw_thermal}"
REASONING_OUT="${REASONING_OUT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/signal_prefix_qwen_reasoning_qwen25_1p5b_v6_raw_thermal}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${REASONING_OUT}"

OUT_ROOT="${OUT_ROOT}" \
bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_exp1_decoupled_modality_tf_control_curriculum.sh"

SIGNAL_INIT="${OUT_ROOT}/stage2_signal_fusion/best.pt" \
OUTPUT_DIR="${REASONING_OUT}" \
bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_signal_prefix_qwen_reasoning.sh"
