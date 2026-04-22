#!/usr/bin/env bash
set -euo pipefail

PYTHON_PATH="${PYTHON_PATH:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=.

ROUND23_ROOT="${ROUND23_ROOT:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_evidence_mechanism_primary_alignment_v1}"
STAGE3_DIR="${STAGE3_DIR:-${ROUND23_ROOT}/stage3_fault_aware_multiview}"
STAGE4_DIR="${STAGE4_DIR:-${ROUND23_ROOT}/stage4_fault_aware_multiview_llm}"

for target in "${STAGE3_DIR}" "${STAGE4_DIR}"; do
  scores_csv="${target}/test_modality_evidence_scores.csv"
  if [[ ! -f "${scores_csv}" ]]; then
    echo "Missing input CSV: ${scores_csv}" >&2
    exit 1
  fi
  echo "[analyze] ${scores_csv}"
  "${PYTHON_PATH}" experiments_smoke_20260316/analyze_modality_evidence_topk.py \
    --scores-csv "${scores_csv}" \
    --output-dir "${target}"
done

printf 'done\n'
