#!/usr/bin/env bash
# BM log1p ambient sweep with Euclidean kNN graph for the spectral cost.
# Tests whether the BM headline (sphere -> -44%) recovers in log1p when
# we stop throwing away the magnitude axis via cosine kNN.
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

# THE KEY DIFFERENCE: Euclidean kNN graph for the spectral Laplacian.
# Default (cosine) throws away magnitude information that log1p ambient
# preserves; Euclidean kNN keeps it.
export SMFM_KNN_METRIC=euclidean

run_split() {
  local split="$1"
  local log="logs/bonemarrow_power_law_linear_log1p_eucknn_b512_split${split}.log"
  local outdir="outputs/bonemarrow_power_law_linear_log1p_eucknn_b512_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== bonemarrow log1p+EucKNN split ${split}: log already complete, skipping ==="
      return 0
    fi
  fi

  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== BM log1p ambient + Euclidean-kNN sweep, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  echo "    SMFM_KNN_METRIC=${SMFM_KNN_METRIC}"
  "${PY}" -u -m surf.train \
    data=bonemarrow \
    +experiment=power_law_sweep_linear_log1p \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== BM log1p+EucKNN split ${split} done ==="
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
