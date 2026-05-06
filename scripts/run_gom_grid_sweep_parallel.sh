#!/usr/bin/env bash
# Parallel version of run_gom_grid_sweep.sh: launches all 5 splits
# concurrently on the same GPU. GoM is tiny (~445 cells, 2-D), so the
# bottleneck is CPU (OT solver, kNN, eigendecomp); parallelism gives
# ~2-3x wall-clock speedup vs sequential.
#
# Memory footprint: each process has its own python interpreter +
# small CUDA context (~1-2 GB per process). 5 concurrent processes
# fit comfortably in 40-GB A100 memory.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/gom_alpha_blend_grid_split${split}.log"
  outdir="outputs/gom_alpha_blend_grid_split${split}"

  echo "=== launching GoM grid split=${split} (background) ==="
  (
    .venv/bin/python -u -m surf.train \
      data=gom \
      +experiment=gom_alpha_blend_grid \
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
echo "  tail -f logs/gom_alpha_blend_grid_split42.log"
echo "Or check completion with:"
echo "  for s in 42 43 44 45 46; do echo \"split\$s:\"; tail -1 logs/gom_alpha_blend_grid_split\${s}.log; done"
echo

wait
echo
echo "=== ALL 5 SPLITS COMPLETE ==="
