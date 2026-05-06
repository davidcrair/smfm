#!/usr/bin/env bash
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
  log="logs/power_law_b512_split${split}.log"
  outdir="outputs/power_law_b512_split${split}"

  echo "=== power-law spectral sweep, batch=512, split=${split} ==="
  "${PY}" -u -m surf.train \
    +experiment=power_law_sweep \
    data.seed="${split}" \
    eval.n_seeds=1 \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== split ${split} done ==="
done
