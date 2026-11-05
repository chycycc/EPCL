#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MODEL_DIR="${1:-results/cloud_archive/v10_trial7_pcmg_head}"
OUT_DIR="${2:-results/v10/v10_trial9_lexical_readout}"
PYTHON_BIN="${PYTHON:-python}"
EXPERT_DIR="artifacts/v10_trial9_lexical_readout"
mkdir -p "$OUT_DIR/baseline" "$OUT_DIR/candidate"
for variant in baseline candidate; do
  EXTRA=()
  if [ "$variant" = candidate ]; then
    EXTRA=(--lexical_expert "$EXPERT_DIR")
  fi
  "$PYTHON_BIN" src/scripts/eval_pipeline.py \
    --model_path "$MODEL_DIR" --eval_target both \
    --batch_size 1 --gpu 0 --use_pcmg_head \
    --lambda_epcl 0.03 --use_arc_margin --arc_margin 0.30 \
    --skip_manifold --gen_samples 500 \
    --exp_name "v10_trial9_lexical_${variant}" \
    --metrics_path "$OUT_DIR/$variant/eval_metrics.json" \
    --results_path "$OUT_DIR/$variant/eval.txt" \
    "${EXTRA[@]}" 2>&1 | tee "$OUT_DIR/$variant/run.log"
done
printf "Evaluation complete. Download: %s\n" "$OUT_DIR"
