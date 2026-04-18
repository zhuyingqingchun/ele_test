#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_compare_cnn_tcn_current_modalities.sh"
bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_compare_bilstm_current_modalities.sh"
bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_compare_resnet_fcn_current_modalities.sh"
bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_compare_transformer_encoder_current_modalities.sh"
bash "${PROJECT_ROOT}/experiments_smoke_20260316/run_compare_dual_branch_xattn_current_modalities.sh"
