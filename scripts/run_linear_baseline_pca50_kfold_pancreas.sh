#!/usr/bin/env bash
# MM+Linear in 50-dim PCA latent space, on pancreatic endocrinogenesis.
# Encodes log1p -> PCA50 for training, decodes PCA50 -> log1p via inverse
# transform, then evaluates in the same compositional space as the sphere /
# spectral methods. Same data splits 42..46, eval.n_seeds=1.
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
  log="logs/pancreas_linear_baseline_pca50_split${split}.log"
  outdir="outputs/pancreas_linear_baseline_pca50_split${split}"

  echo "=== pancreas linear baseline pca50, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=pancreas \
    +experiment=linear_baseline_pca50 \
    data.seed="${split}" \
    eval.n_seeds=1 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"

  echo "=== pancreas linear pca50 split ${split} done ==="
done
