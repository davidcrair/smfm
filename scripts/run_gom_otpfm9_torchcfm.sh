#!/usr/bin/env bash
# 6-way GoM 9-stage OTP-FM-holdout run that adds torchcfm-backed MMFM
# variants to the existing MM+Linear / MM+MMFM-OTPFM / MM+MMFM-Rohbeck
# baselines. 5 splits in parallel; each split fits all six methods on
# the same train marginals and evaluates W_2 at the four held-out
# stages so the numbers are directly comparable to OTP-FM Table 2.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/gom_otpfm9_torchcfm_split${split}.log"
  outdir="outputs/gom_otpfm9_torchcfm_split${split}"

  echo "=== launching GoM 9-stage torchcfm split=${split} ==="
  (
    .venv/bin/python -u -m surf.train \
      data=gom \
      +experiment=gom_otpfm9_torchcfm \
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
echo "Watch: tail -f logs/gom_otpfm9_torchcfm_split42.log"
wait
echo "=== ALL 5 SPLITS COMPLETE (GoM torchcfm MMFM) ==="
