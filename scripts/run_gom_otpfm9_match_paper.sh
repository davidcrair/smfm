#!/usr/bin/env bash
# Faithful OTP-FM/Rohbeck MMFM reproduction on GoM 9-stage holdout.
# Uses FlowNetV2 (10x128 + residuals + layernorm + sin-t-emb) and the
# OTP-FM training schedule (lr=1e-3, batch=64, ema=0.99, no grad clip,
# 18k iters, sigma_scale=1.0) so MMFM-Rohbeck here is the closest
# in-codebase analogue to OTP-FM Table 2's reported MMFM W_2 ~ 0.15.
#
# 5 splits in parallel. Significantly slower than the standard runs:
# at batch=64 the per-iter step is small but n_iters=18000 = 6x more
# steps, so ~30-90 min per split on a single GPU depending on amp.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

pids=()
for split in "${splits[@]}"; do
  log="logs/gom_otpfm9_match_paper_split${split}.log"
  outdir="outputs/gom_otpfm9_match_paper_split${split}"

  echo "=== launching GoM 9-stage match-paper split=${split} ==="
  (
    .venv/bin/python -u -m surf.train \
      data=gom \
      +experiment=gom_otpfm9_match_paper \
      data.n_stages=9 \
      data.otpfm_split=true \
      data.otpfm_holdout=true \
      data.seed="${split}" \
      eval.n_seeds=1 \
      eval.mmd_protocol=otpfm \
      eval.compute_fgd=false \
      eval.compute_w2=true \
      hydra.run.dir="${outdir}" \
      > "${log}" 2>&1
    echo "=== split ${split} finished ==="
  ) &
  pids+=($!)
done

echo "${#pids[@]} splits launched. PIDs: ${pids[*]}"
echo "Watch: tail -f logs/gom_otpfm9_match_paper_split42.log"
wait
echo "=== ALL 5 SPLITS COMPLETE (GoM match-paper MMFM) ==="
