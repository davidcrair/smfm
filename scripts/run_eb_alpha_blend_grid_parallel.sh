#!/usr/bin/env bash
# EB 80/10/10 alpha x blend grid sweep. Companion to the OTP-FM-holdout
# grid (run_eb_otpfm_alpha_blend_grid_parallel.sh): same 21-method grid
# (MM+Linear baseline + 5 alpha values x 4 blend values) but trained on
# the standard 80/10/10 stratified split across all 5 stages, no
# alternate-marginal holdout.
#
# This isolates the second precondition (graph connectivity) from the
# first (temporal sparsity) on EB: the OTP-FM grid asks 'can blend
# rescue spectral when stages 1, 3 are held out?', this grid asks 'with
# all stages observed, does blend buy anything beyond pure spectral?'.
#
# 21 methods x 5 splits = 105 method-seeds. Same parallelism budget
# (5-way) and same max_cells_per_stage=500 as the OTP-FM grid for
# direct apples-to-apples comparison of the heatmap surfaces.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

MAX_PARALLEL="${MAX_PARALLEL:-5}"
splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/eb_alpha_blend_grid_split${split}.log"
  outdir="outputs/eb_alpha_blend_grid_split${split}"

  echo "=== launching EB 80/10/10 grid split=${split} (background) ==="
  (
    .venv/bin/python -u -m surf.train \
      data=embryoid \
      +experiment=gom_alpha_blend_grid \
      space=sphere \
      data.representation=sphere \
      data.max_cells_per_stage=500 \
      data.seed="${split}" \
      eval.n_seeds=1 \
      eval.mmd_protocol=otpfm \
      eval.compute_fgd=false \
      training.batch_size=512 \
      training.n_iters=3000 \
      hydra.run.dir="${outdir}" \
      > "${log}" 2>&1
    echo "=== split ${split} finished ==="
  ) &
  pids+=($!)
done

echo
echo "${#pids[@]} splits launched in parallel. PIDs: ${pids[*]}"
echo "Watch progress: tail -f logs/eb_alpha_blend_grid_split42.log"
echo "Check completion: for s in 42 43 44 45 46; do echo \"split\$s:\"; tail -1 logs/eb_alpha_blend_grid_split\${s}.log; done"

wait
echo
echo "=== ALL 5 SPLITS COMPLETE (EB 80/10/10 grid) ==="
