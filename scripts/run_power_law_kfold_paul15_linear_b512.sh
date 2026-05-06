#!/usr/bin/env bash
# Power-law spectral OT cost sweep with the LINEAR / Euclidean trainer on
# Paul15 hematopoiesis (erythroid trajectory, 5 stages, ~1.2k cells).
# Mirrors run_power_law_kfold_pancreas_linear_b512.sh.
set -euo pipefail

# Resolve project root from script location so this works on local + Colab + Modal.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

# Use whichever python is in PATH if .venv isn't present (e.g., on Modal).
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

# Cap BLAS thread count per process so parallel splits don't oversubscribe CPU.
THREADS_PER_PROC="${THREADS_PER_PROC:-3}"
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_PROC}"

run_split() {
  local split="$1"
  local log="logs/paul15_power_law_linear_b512_split${split}.log"
  local outdir="outputs/paul15_power_law_linear_b512_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== Paul15 split ${split}: log already complete for n_seeds=${N_SEEDS}, skipping ==="
      return 0
    fi
    echo "=== Paul15 split ${split}: existing log is single-seed; re-running for n_seeds=${N_SEEDS} ==="
  fi

  # Per-split torch.compile / triton cache dirs. Without these, parallel splits
  # race on the same /tmp/torchinductor_root/ files and crash with
  # FileNotFoundError mid-compile.
  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== Paul15 Linear-FM sweep, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=paul15 \
    +experiment=power_law_sweep_linear \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    eval.mmd_protocol=both \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== Paul15 split ${split} done ==="
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
