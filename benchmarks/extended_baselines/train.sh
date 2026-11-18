#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${1:?Usage: train.sh iamm|iamm_no_situation|escm|seek [--resume|--evaluate-only]}"
shift
case "$model" in iamm|iamm_no_situation|escm|seek) ;; *) exit 2 ;; esac
bundle="${BASELINE_BUNDLE_ROOT:-$(dirname "$tool_dir")}"
run="${BASELINE_RUN_DIR:-$bundle/runs/formal_$model}"
mkdir -p "$(dirname "$run")"
export PYTHONUTF8=1
"${BASELINE_PYTHON:-python}" -u "$tool_dir/run.py" "$model" \
  --bundle-root "$bundle" --run-dir "$run" \
  --embedding-file "${BASELINE_GLOVE:-/root/autodl-tmp/CASE/vectors/glove.6B.300d.txt}" \
  "$@" 2>&1 | tee -a "$run.log"
