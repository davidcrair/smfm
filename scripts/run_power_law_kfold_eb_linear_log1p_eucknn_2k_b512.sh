#!/usr/bin/env bash
# EB 80/10/10 split, log1p HVG ambient + Euclidean kNN, subsampled to
# 2000 cells per marginal. Mirrors the BM/PE log1p+EucKNN protocol but
# on EB (and with the cell cap to keep wallclock manageable on the
# largest dataset).
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

# Use Euclidean kNN -- correct for log1p HVG ambient.
export SMFM_KNN_METRIC=euclidean

run_split() {
  local split="$1"
  local log="logs/eb_8010_linear_log1p_eucknn_2k_b512_split${split}.log"
  local outdir="outputs/eb_8010_linear_log1p_eucknn_2k_b512_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== EB 80/10/10 log1p+EucKNN+2k split ${split}: log already complete, skipping ==="
      return 0
    fi
  fi

  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== EB 80/10/10 log1p ambient + Euclidean-kNN + 2k cells/stage, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  echo "    SMFM_KNN_METRIC=${SMFM_KNN_METRIC}"
  "${PY}" -u -m surf.train \
    data=embryoid \
    +experiment=power_law_sweep_linear_log1p \
    data.seed="${split}" \
    data.max_cells_per_stage=2000 \
    eval.n_seeds="${N_SEEDS}" \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== EB 80/10/10 log1p+EucKNN+2k split ${split} done ==="
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
