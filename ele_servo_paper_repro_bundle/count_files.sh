#!/usr/bin/env bash
# 统计当前目录的文件、目录、大小等信息
set -euo pipefail

TARGET_DIR="${1:-.}"

echo "============================================"
echo " 目录统计报告: $(realpath "${TARGET_DIR}")"
echo "============================================"
echo ""

# 基本统计
FILE_COUNT=$(find "${TARGET_DIR}" -type f 2>/dev/null | wc -l)
DIR_COUNT=$(find "${TARGET_DIR}" -type d 2>/dev/null | wc -l)
LINK_COUNT=$(find "${TARGET_DIR}" -type l 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "${TARGET_DIR}" 2>/dev/null | cut -f1)

echo "📁 目录数量:    ${DIR_COUNT}"
echo "📄 文件数量:    ${FILE_COUNT}"
echo "🔗 符号链接:    ${LINK_COUNT}"
echo "💾 总大小:      ${TOTAL_SIZE}"
echo ""

# 按扩展名统计文件数量
echo "--------------------------------------------"
echo "  按文件扩展名统计 (Top 20)"
echo "--------------------------------------------"
find "${TARGET_DIR}" -type f 2>/dev/null | \
  sed 's/.*\.//' | \
  sort | uniq -c | sort -rn | head -20 | \
  awk '{printf "  %-20s %d\n", $2, $1}'
echo ""

# 按目录统计文件大小
echo "--------------------------------------------"
echo "  一级子目录大小"
echo "--------------------------------------------"
du -sh "${TARGET_DIR}"/*/ 2>/dev/null | sort -rh | \
  awk '{printf "  %-40s %s\n", $2, $1}'
echo ""

# 最大文件 Top 10
echo "--------------------------------------------"
echo "  最大文件 (Top 10)"
echo "--------------------------------------------"
find "${TARGET_DIR}" -type f -exec du -h {} + 2>/dev/null | \
  sort -rh | head -10 | \
  awk '{printf "  %-60s %s\n", $2, $1}'
echo ""

echo "============================================"
echo "  统计完成"
echo "============================================"
