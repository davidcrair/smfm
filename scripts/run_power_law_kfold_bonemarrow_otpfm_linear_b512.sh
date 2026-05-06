#!/usr/bin/env bash
# Bonemarrow erythroid OTP-FM-style holdout ablation: same 6 methods as the
# regular bonemarrow sweep, but with stages 1 (HSC_late) and 3 (Ery_early)
# entirely held out from training. Tests whether the "spectral OT degrades
# under alternate-marginal holdout" pattern (already observed on
# pancreas-OTPFM and EB-OTPFM) also holds on bonemarrow -- which would
# strengthen the temporal-sampling-precondition story for the writeup.
set -euo pipefail

# Resolve project root from script location so it works on Mac, Colab, etc.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

splits=(42 43 44 45 46)

for split in "${splits[@]}"; do
  log="logs/bonemarrow_otpfm_linear_b512_split${split}.log"
  outdir="outputs/bonemarrow_otpfm_linear_b512_split${split}"

  echo "=== bonemarrow OTP-FM holdout, batch=512, split=${split} ==="
  .venv/bin/python -u -m surf.train \
    data=bonemarrow \
    +experiment=power_law_sweep_linear \
    data.otpfm_holdout=true \
    data.otpfm_split=true \
    data.seed="${split}" \
    eval.n_seeds=1 \
    eval.mmd_protocol=both \
    eval.compute_fgd=false \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== bonemarrow OTP-FM split ${split} done ==="
done
