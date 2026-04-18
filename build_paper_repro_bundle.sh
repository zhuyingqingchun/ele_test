#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
MANIFEST="${PROJECT_ROOT}/handoff_packages/paper_repro_bundle_manifest.txt"
README_SRC="${PROJECT_ROOT}/handoff_packages/paper_repro_bundle_README.md"
REQ_SRC="${PROJECT_ROOT}/handoff_packages/paper_repro_bundle_requirements.txt"
OUT_PATH="${1:-${PROJECT_ROOT}/handoff_packages/ele_servo_paper_repro_bundle_$(date +%Y%m%d).tar.gz}"
BUNDLE_NAME="ele_servo_paper_repro_bundle"
TMP_DIR="$(mktemp -d)"
STAGE_DIR="${TMP_DIR}/${BUNDLE_NAME}"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${STAGE_DIR}"

while IFS= read -r line; do
  path="${line%%#*}"
  path="$(printf '%s' "${path}" | sed 's/[[:space:]]*$//')"
  [ -z "${path}" ] && continue
  src="${PROJECT_ROOT}/${path}"
  if [ ! -e "${src}" ]; then
    echo "Missing path in manifest: ${path}" >&2
    exit 1
  fi
  dst_dir="${STAGE_DIR}/$(dirname "${path}")"
  mkdir -p "${dst_dir}"
  cp -a "${src}" "${dst_dir}/"
done < "${MANIFEST}"

cp "${README_SRC}" "${STAGE_DIR}/README.md"
cp "${REQ_SRC}" "${STAGE_DIR}/requirements.txt"
cp "${MANIFEST}" "${STAGE_DIR}/FILELIST.txt"

find "${STAGE_DIR}" -type d -name __pycache__ -prune -exec rm -rf {} +
find "${STAGE_DIR}" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

mkdir -p "$(dirname "${OUT_PATH}")"
tar -czf "${OUT_PATH}" -C "${TMP_DIR}" "${BUNDLE_NAME}"

echo "bundle=${OUT_PATH}"
