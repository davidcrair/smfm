#!/usr/bin/env bash
# GoM 9-stage OTP-FM hold-out alpha-sweep with beta=0 (pure spectral),
# 5 init seeds × 5 data splits, exact W_2 as primary metric.
#
# Reports per-stage W_2 (held-out + train) for each (alpha, split, init)
# combination. Aggregator scripts read the eval blocks downstream.
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
  local log="logs/gom_otpfm9_alpha_sweep_split${split}.log"
  local outdir="outputs/gom_otpfm9_alpha_sweep_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== GoM-9 alpha-sweep split ${split}: log already complete for n_seeds=${N_SEEDS}, skipping ==="
      return 0
    fi
  fi

  # Per-split torch.compile / triton cache dirs to avoid race in parallel runs.
  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== GoM-9 OTP-FM alpha-sweep (beta=0), batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=gom \
    +experiment=gom_otpfm9_alpha_sweep_beta0 \
    data.n_stages=9 \
    data.otpfm_split=true \
    data.otpfm_holdout=true \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== GoM-9 alpha-sweep split ${split} done ==="
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
