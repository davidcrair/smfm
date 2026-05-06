#!/usr/bin/env bash
# Linear-trainer power-law spectral OT cost sweep on OTP-FM-style EB
# (TrajectoryNet eb_velocity_v5.npz, top-100 PCs, StandardScaler).
# Same 5-split protocol as the pancreas sweeps; produces directly comparable
# numbers to MMFM / OTP-FM published EB tables.
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
  log="logs/eb_otpfm_power_law_linear_b512_split${split}.log"
  outdir="outputs/eb_otpfm_power_law_linear_b512_split${split}"

  echo "=== EB-OTPFM Linear-FM spectral sweep, batch=512, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=eb_otpfm \
    +experiment=power_law_sweep_linear_otpfm \
    data.seed="${split}" \
    eval.n_seeds=1 \
    eval.mmd_protocol=both \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== EB-OTPFM split ${split} done ==="
done
