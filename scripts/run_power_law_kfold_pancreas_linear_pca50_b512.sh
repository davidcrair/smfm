#!/usr/bin/env bash
# Power-law spectral OT cost sweep, Linear/Euclidean trainer, PCA-50 latent
# space, on pancreatic endocrinogenesis. Companion to
# run_power_law_kfold_pancreas_linear_b512.sh (sphere-ambient). Identical
# 5-split protocol; produces the data needed for the cross-space ablation
# table (PCA-50 vs sphere-ambient) in the writeup.
set -euo pipefail

# Derive project root from script location so this works on local + Colab.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
mkdir -p logs

# Use whichever python is in PATH if .venv isn't present (e.g., on Colab).
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

splits=(42 43 44 45 46)

for split in "${splits[@]}"; do
  log="logs/pancreas_power_law_linear_pca50_b512_split${split}.log"
  outdir="outputs/pancreas_power_law_linear_pca50_b512_split${split}"

  # Per-split torch.compile / triton cache dirs. Without these, parallel splits
  # race on the same /tmp/torchinductor_root/ files and crash with
  # FileNotFoundError mid-compile.
  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== pancreas Linear-FM PCA-50 spectral sweep, batch=512, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=pancreas \
    +experiment=power_law_sweep_linear_pca50 \
    data.seed="${split}" \
    eval.n_seeds=1 \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== pancreas Linear-FM PCA-50 split ${split} done ==="
done
