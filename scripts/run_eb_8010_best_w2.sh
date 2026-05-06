#!/usr/bin/env bash
# Headline best-model EB 80/10/10 run with exact W_2.
# Trains MM+Linear (Euclidean baseline) and the winning spectral cell
# (alpha=0.5, beta=0) on each of 5 splits in parallel; eval reports
# MMD^2_otpfm AND W_2. Together with run_gom_otpfm9_best_w2.sh this
# gives the paper one per-stage W_2 table per "headline dataset".
# Uses max_cells_per_stage=500 so the per-split eval finishes quickly.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/eb_8010_best_w2_split${split}.log"
  outdir="outputs/eb_8010_best_w2_split${split}"

  echo "=== launching EB 80/10/10 best-model split=${split} ==="
  (
    .venv/bin/python -u -m surf.train \
      data=embryoid \
      +experiment=eb_8010_best \
      space=sphere \
      data.representation=sphere \
      data.max_cells_per_stage=500 \
      data.seed="${split}" \
      eval.n_seeds=1 \
      eval.mmd_protocol=otpfm \
      eval.compute_fgd=false \
      eval.compute_w2=true \
      training.batch_size=512 \
      training.n_iters=3000 \
      hydra.run.dir="${outdir}" \
      > "${log}" 2>&1
    echo "=== split ${split} finished ==="
  ) &
  pids+=($!)
done

echo "${#pids[@]} splits launched. PIDs: ${pids[*]}"
echo "Watch: tail -f logs/eb_8010_best_w2_split42.log"
wait
echo "=== ALL 5 SPLITS COMPLETE (EB 80/10/10 best-model + W_2) ==="
