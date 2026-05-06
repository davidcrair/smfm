#!/usr/bin/env bash
# GoM full alpha x blend grid under the 9-stage OTP-FM-style protocol:
# all 9 timepoints loaded; stages 1, 3, 5, 7 entirely held out from
# training; trained on stages 0, 2, 4, 6, 8. Tests whether the
# Euclidean blend rescues spectral OT when BOTH preconditions are
# violated (disconnected graph AND temporal sparsity).
#
# Each split runs in parallel on the same GPU (CPU-bound, ~2-3x
# speedup vs sequential).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/gom_otpfm9_alpha_blend_grid_split${split}.log"
  outdir="outputs/gom_otpfm9_alpha_blend_grid_split${split}"

  echo "=== launching GoM 9-stage OTP-FM grid split=${split} (background) ==="
  (
    .venv/bin/python -u -m surf.train \
      data=gom \
      +experiment=gom_alpha_blend_grid \
      data.n_stages=9 \
      data.otpfm_split=true \
      data.otpfm_holdout=true \
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
echo "All 5 splits launched in parallel. PIDs: ${pids[*]}"
echo "Watch progress with:"
echo "  tail -f logs/gom_otpfm9_alpha_blend_grid_split42.log"

wait
echo
echo "=== ALL 5 SPLITS COMPLETE (GoM OTP-FM 9-stage) ==="
