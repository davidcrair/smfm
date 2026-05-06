#!/usr/bin/env bash
# 6-way per-pair-vs-global spectral head-to-head on GoM 9-stage
# OTP-FM holdout, with 5 splits in parallel. Tests whether the
# theoretically-cleaner GLOBAL spectral cost (one Laplacian on the
# union of all training stages) outperforms the per-pair variant
# we've been using -- on both the Tong-style and Rohbeck-style
# MMFM scaffolds.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/gom_otpfm9_global_compare_split${split}.log"
  outdir="outputs/gom_otpfm9_global_compare_split${split}"

  echo "=== launching GoM 9-stage global-vs-per-pair split=${split} ==="
  (
    .venv/bin/python -u -m surf.train \
      data=gom \
      +experiment=gom_otpfm9_global_compare \
      data.n_stages=9 \
      data.otpfm_split=true \
      data.otpfm_holdout=true \
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
echo "Watch: tail -f logs/gom_otpfm9_global_compare_split42.log"
wait
echo "=== ALL 5 SPLITS COMPLETE (GoM global-vs-per-pair compare) ==="
