#!/usr/bin/env bash
# GoM full alpha x blend grid sweep for the heatmap figure.
# 21 methods x 5 seeds; estimated ~10 min/split on A100, ~1 hr total.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

for split in "${splits[@]}"; do
  log="logs/gom_alpha_blend_grid_split${split}.log"
  outdir="outputs/gom_alpha_blend_grid_split${split}"

  echo "=== GoM alpha x blend grid, split=${split} ==="
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
    2>&1 | tee "${log}"

  echo "=== GoM grid split ${split} done ==="
done
