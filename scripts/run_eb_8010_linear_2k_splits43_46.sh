#!/usr/bin/env bash
# Splits 43-46 of EB 80/10/10 linear-trainer power-law sweep, with
# max_cells_per_stage=2000 subsampling for speed. Companion to the
# manually-run split 42 -- only run this *after* split 42 confirms
# spectral OT meaningfully helps on EB at this setup, otherwise it's
# wasted compute.
set -euo pipefail

mkdir -p logs

splits=(43 44 45 46)

for split in "${splits[@]}"; do
  log="logs/eb_8010_linear_2k_b512_split${split}.log"
  outdir="outputs/eb_8010_linear_2k_b512_split${split}"

  echo "=== EB 80/10/10 Linear-FM spectral sweep, max_cells=2000, split=${split} ==="
  .venv/bin/python -u -m surf.train \
    data=embryoid \
    +experiment=power_law_sweep_linear \
    data.max_cells_per_stage=2000 \
    data.seed="${split}" \
    eval.n_seeds=1 \
    eval.mmd_protocol=both \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== EB 80/10/10 linear split ${split} done ==="
done
