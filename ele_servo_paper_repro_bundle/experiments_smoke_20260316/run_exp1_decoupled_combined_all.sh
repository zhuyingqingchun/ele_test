#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v14_combined_alignment}"
ABLATION_ROOT="${ABLATION_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v14_combined_alignment_ablation}"
FEATURE_MODE="${FEATURE_MODE:-modality_tf}"
QUALITY_AWARE_FUSION="${QUALITY_AWARE_FUSION:-1}"
MODALITY_DROP_PROB="${MODALITY_DROP_PROB:-0.10}"
RUN_ABLATION="${RUN_ABLATION:-1}"

echo "[combined] OUT_ROOT=${OUT_ROOT}"
echo "[combined] FEATURE_MODE=${FEATURE_MODE}"
echo "[combined] QUALITY_AWARE_FUSION=${QUALITY_AWARE_FUSION}"
echo "[combined] MODALITY_DROP_PROB=${MODALITY_DROP_PROB}"

OUT_ROOT="${OUT_ROOT}" \
FEATURE_MODE="${FEATURE_MODE}" \
QUALITY_AWARE_FUSION="${QUALITY_AWARE_FUSION}" \
MODALITY_DROP_PROB="${MODALITY_DROP_PROB}" \
bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_exp1_decoupled_modality_tf_control_curriculum.sh"

if [[ "${RUN_ABLATION}" == "1" ]]; then
  ROOT_OUT_DIR="${ABLATION_ROOT}" \
  FEATURE_MODE="${FEATURE_MODE}" \
  QUALITY_AWARE_FUSION="${QUALITY_AWARE_FUSION}" \
  MODALITY_DROP_PROB="${MODALITY_DROP_PROB}" \
  bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_exp1_decoupled_modality_ablation_async.sh"
fi
