#!/usr/bin/env bash
# Power-law spectral OT sweep on the pancreatic endocrinogenesis dataset.
# Mirrors run_power_law_kfold_b512.sh (the embryoid-body version) so the
# resulting CHAINED EVAL / PER-SEGMENT EVAL tables can be aggregated the same
# way (mean +/- std over data.seed splits 42..46, eval.n_seeds=1 per split).
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
  log="logs/pancreas_power_law_b512_split${split}.log"
  outdir="outputs/pancreas_power_law_b512_split${split}"

  echo "=== pancreas power-law spectral sweep, batch=512, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=pancreas \
    +experiment=power_law_sweep \
    data.seed="${split}" \
    eval.n_seeds=1 \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== pancreas split ${split} done ==="
done
