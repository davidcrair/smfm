#!/usr/bin/env bash
# Linear-trainer power-law spectral OT cost sweep on EB at the log1p HVG
# representation (embryoid_body.h5ad), with OTP-FM holdout (stages 1, 3) and
# OTP-FM-style full-marginal eval. Companion to run_power_law_kfold_eb_otpfm
# which runs the same sweep in 100-PC space.
set -euo pipefail

# Derive project root from script location so this works on local + Colab.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Use whichever python is in PATH if .venv isn't present (e.g., on Colab).
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi
mkdir -p logs

splits=(42 43 44 45 46)

for split in "${splits[@]}"; do
  log="logs/eb_hvg_otpfm_power_law_linear_b512_split${split}.log"
  outdir="outputs/eb_hvg_otpfm_power_law_linear_b512_split${split}"

  echo "=== EB-HVG-OTPFM Linear-FM spectral sweep, batch=512, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=embryoid \
    +experiment=power_law_sweep_linear_eb_hvg_otpfm \
    data.seed="${split}" \
    eval.n_seeds=1 \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== EB-HVG-OTPFM split ${split} done ==="
done
