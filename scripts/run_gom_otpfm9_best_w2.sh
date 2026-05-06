#!/usr/bin/env bash
# Headline best-model GoM 9-stage OTP-FM-holdout run with exact W_2.
# Trains MM+Linear (Euclidean baseline) and the winning spectral-blend
# cell (alpha=0.5, beta=0.25) on each of 5 splits; eval reports
# MMD^2_otpfm AND exact W_2 (POT network simplex, fast on N=111). The
# downstream aggregator script extracts per-stage W_2 values for the
# held-out marginals and per-stage average W_2 for the train marginals.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/gom_otpfm9_best_w2_split${split}.log"
  outdir="outputs/gom_otpfm9_best_w2_split${split}"

  echo "=== launching GoM 9-stage OTP-FM best-model split=${split} ==="
  (
    .venv/bin/python -u -m surf.train \
      data=gom \
      +experiment=gom_otpfm9_best \
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

echo
echo "${#pids[@]} splits launched in parallel. PIDs: ${pids[*]}"
echo "Watch progress: tail -f logs/gom_otpfm9_best_w2_split42.log"

wait
echo
echo "=== ALL 5 SPLITS COMPLETE (GoM OTP-FM-9 best-model + W_2) ==="
