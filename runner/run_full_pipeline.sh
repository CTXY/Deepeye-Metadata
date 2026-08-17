#!/usr/bin/env bash
# Run NL2SQL pipeline steps in order (from repository root).
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

run_step "runner/run_sql_generation.py"
run_step "runner/run_sql_revision.py"
run_step "runner/run_sql_selection.py"
run_step "runner/convert_pkl_to_sql_file.py"
run_step "runner/evaluation.py"

echo ""
echo "Full pipeline finished successfully."
