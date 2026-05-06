#!/usr/bin/env bash
# Power-law spectral OT cost sweep, Linear/Euclidean trainer, PCA-50 latent
# space, on bonemarrow erythroid trajectory. Companion to
# run_power_law_kfold_bonemarrow_linear_b512.sh (sphere-ambient). Same 5-split
# protocol; produces the data needed for the BM PCA-50 vs sphere comparison.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

splits=(42 43 44 45 46)
if [[ -n "${SPLITS_OVERRIDE:-}" ]]; then
  IFS=',' read -ra splits <<< "${SPLITS_OVERRIDE}"
fi
N_SEEDS="${N_SEEDS:-5}"
PARALLEL_SPLITS="${PARALLEL_SPLITS:-1}"

THREADS_PER_PROC="${THREADS_PER_PROC:-3}"
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_PROC}"

run_split() {
  local split="$1"
  local log="logs/bonemarrow_power_law_linear_pca50_b512_split${split}.log"
  local outdir="outputs/bonemarrow_power_law_linear_pca50_b512_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== bonemarrow PCA-50 split ${split}: log already complete for n_seeds=${N_SEEDS}, skipping ==="
      return 0
    fi
    echo "=== bonemarrow PCA-50 split ${split}: existing log is single-seed; re-running ==="
  fi

  # Per-split torch.compile / triton cache dirs. Without these, parallel splits
  # race on the same /tmp/torchinductor_root/ files and crash with
  # FileNotFoundError mid-compile.
  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== bonemarrow Linear-FM PCA-50 sweep, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=bonemarrow \
    +experiment=power_law_sweep_linear_pca50_full \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    eval.mmd_protocol=both \
    eval.compute_fgd=false \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== bonemarrow PCA-50 split ${split} done ==="
}

for split in "${splits[@]}"; do
  if (( PARALLEL_SPLITS <= 1 )); then
    run_split "$split"
  else
    run_split "$split" &
    while (( $(jobs -r | wc -l) >= PARALLEL_SPLITS )); do wait -n || true; done
  fi
done
wait
