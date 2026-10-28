#!/usr/bin/env bash
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${1:?Usage: bash train.sh case_original|multi_trs|cem_original|kemp [--resume|--evaluate-only]}"
shift
case "$model" in case_original|multi_trs|cem_original|kemp) ;; *) exit 2 ;; esac
bundle_root="${BASELINE_BUNDLE_ROOT:-$(dirname "$tool_dir")}"
run_dir="${BASELINE_RUN_DIR:-$bundle_root/runs/formal_$model}"
mkdir -p "$(dirname "$run_dir")"
export PYTHONUTF8=1
"${BASELINE_PYTHON:-python}" -u "$tool_dir/train_baseline.py" \
  --config "$tool_dir/configs/$model.json" \
  --workspace-root "$bundle_root/workspace" \
  --data-dir "$bundle_root/data" \
  --output-dir "$run_dir" \
  --embedding-file "${BASELINE_GLOVE:-/root/autodl-tmp/CASE/vectors/glove.6B.300d.txt}" \
  "$@" 2>&1 | tee -a "$run_dir.log"
