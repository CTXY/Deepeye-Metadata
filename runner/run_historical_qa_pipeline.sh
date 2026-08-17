#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

run_step() {
  local rel="$1"
  echo ""
  echo "=========================================="
  echo "==> ${rel}"
  echo "=========================================="
  "${PYTHON}" "${rel}"
}

run_step "runner/run_memory_augmentation.py"
run_step "runner/run_sql_generation.py"
run_step "runner/run_sql_revision.py"
run_step "runner/run_sql_selection.py"

echo ""
echo "Historical QA pipeline finished successfully."
