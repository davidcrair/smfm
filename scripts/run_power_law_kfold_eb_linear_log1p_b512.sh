#!/usr/bin/env bash
# Power-law spectral OT cost sweep, Linear/Euclidean trainer, **raw log1p
# HVG ambient space** (no sphere encoding) on EB 80/10/10. Mirror of the
# Paul15 / PE / BM log1p sweeps for cross-dataset comparison under the
# OTP-FM-comparable ambient + PCA-50 dual eval.
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
  local log="logs/eb_8010_linear_log1p_b512_split${split}.log"
  local outdir="outputs/eb_8010_linear_log1p_b512_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== EB log1p split ${split}: log already complete for n_seeds=${N_SEEDS}, skipping ==="
      return 0
    fi
  fi

  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== EB 80/10/10 log1p ambient sweep, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=embryoid \
    +experiment=power_law_sweep_linear_log1p \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== EB log1p split ${split} done ==="
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
