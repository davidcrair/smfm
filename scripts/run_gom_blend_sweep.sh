#!/usr/bin/env bash
# GoM blend sweep: tests whether a small Euclidean regularization on the
# spectral cost recovers usable OT pairings on the disconnected-graph
# vortex dataset (kNN union has ~6 components by connectivity diagnostic).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

for split in "${splits[@]}"; do
  log="logs/gom_blend_sweep_split${split}.log"
  outdir="outputs/gom_blend_sweep_split${split}"

  echo "=== GoM blend sweep, batch=512, split=${split} ==="
  .venv/bin/python -u -m surf.train \
    data=gom \
    +experiment=gom_blend_sweep \
    data.seed="${split}" \
    eval.n_seeds=1 \
    eval.mmd_protocol=both \
    eval.compute_fgd=false \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== GoM split ${split} done ==="
done
