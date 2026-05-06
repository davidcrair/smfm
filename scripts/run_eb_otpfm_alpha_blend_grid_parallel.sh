#!/usr/bin/env bash
# EB OTP-FM-holdout alpha x blend grid sweep. Tests whether the
# Euclidean blend rescues spectral OT under alternate-marginal holdout
# (stages 1, 3 held out, training on stages 0, 2, 4). Uses sphere
# encoding for parity with the GoM 80/10/10 grid sweep.
#
# 21 methods (MM+Linear baseline + 5 alpha values x 4 blend values)
# x 5 splits = 105 method-seeds. With max_cells_per_stage=2000 each
# split has ~6k train cells across 3 training stages.
#
# Two execution modes (set MAX_PARALLEL env var to switch):
#   MAX_PARALLEL=1   sequential, ~6-8 hours, most reliable
#   MAX_PARALLEL=5   default, parallel batch of 5, ~50-90 min
#
# A previous attempt at 10-way parallel killed eval mid-MMD on Colab,
# likely due to GPU contention across too many concurrent processes.
# 5 is the safer default for an A100; CPU-bound OT solver still
# benefits, GPU is less saturated for the eval pass.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

MAX_PARALLEL="${MAX_PARALLEL:-5}"
splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/eb_otpfm_alpha_blend_grid_split${split}.log"
  outdir="outputs/eb_otpfm_alpha_blend_grid_split${split}"

  echo "=== launching EB OTP-FM grid split=${split} (background) ==="
  (
    .venv/bin/python -u -m surf.train \
      data=embryoid \
      +experiment=gom_alpha_blend_grid \
      space=sphere \
      data.representation=sphere \
      data.otpfm_holdout=true \
      data.otpfm_split=true \
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
echo "Watch progress: tail -f logs/eb_otpfm_alpha_blend_grid_split42.log"
echo "Check completion: for s in 42 43 44 45 46; do echo \"split\$s:\"; tail -1 logs/eb_otpfm_alpha_blend_grid_split\${s}.log; done"

wait
echo
echo "=== ALL 5 SPLITS COMPLETE (EB OTP-FM grid) ==="
