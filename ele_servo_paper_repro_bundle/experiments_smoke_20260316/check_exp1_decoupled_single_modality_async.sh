#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
ROOT_OUT_DIR="${ROOT_OUT_DIR:-${PROJECT_ROOT}/experiments_smoke_20260316/models/exp1_decoupled_v4_single_modality_raw_thermal_async}"
LOG_DIR="${ROOT_OUT_DIR}/logs"

if [ ! -d "${LOG_DIR}" ]; then
  echo "Log dir not found: ${LOG_DIR}" >&2
  exit 1
fi

for pid_file in "${LOG_DIR}"/*.pid; do
  [ -e "${pid_file}" ] || continue
  name="$(basename "${pid_file}" .pid)"
  pid="$(cat "${pid_file}")"
  report_file="${ROOT_OUT_DIR}/${name}/stage4_signal_text_llm/report.json"
  if [ -f "${report_file}" ]; then
    status="DONE"
  elif ps -p "${pid}" > /dev/null 2>&1; then
    status="RUNNING"
  else
    status="FAILED_OR_EXITED"
  fi
  echo "${name} ${status} pid=${pid}"
done
