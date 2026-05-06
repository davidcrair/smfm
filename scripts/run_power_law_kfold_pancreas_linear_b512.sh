#!/usr/bin/env bash
# Power-law spectral OT cost sweep with the LINEAR / Euclidean trainer on
# pancreatic endocrinogenesis. Companion to
# run_power_law_kfold_pancreas_b512.sh (which uses the sphere trainer).
# Identical 5-split protocol; produces the data needed for the "spectral cost
# is trainer-agnostic" ablation in the writeup.
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
N_SEEDS="${N_SEEDS:-5}"
PARALLEL_SPLITS="${PARALLEL_SPLITS:-1}"

# Cap BLAS / numpy / OpenMP thread count per process so parallel splits don't
# all try to grab every core. With 12-vCPU Colab and PARALLEL_SPLITS=3, we
# want each process limited to ~3-4 threads (12 / 3 splits + headroom).
THREADS_PER_PROC="${THREADS_PER_PROC:-3}"
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_PROC}"

run_split() {
  local split="$1"
  local log="logs/pancreas_power_law_linear_b512_split${split}.log"
  local outdir="outputs/pancreas_power_law_linear_b512_split${split}"

  # Resume support: skip only if the existing log was already produced with
  # the requested n_seeds (multi-seed runs encode "mean±std" in the cells;
  # single-seed runs do not). This avoids re-using stale single-seed logs.
  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== pancreas split ${split}: log already complete for n_seeds=${N_SEEDS}, skipping ==="
      return 0
    fi
    echo "=== pancreas split ${split}: existing log is single-seed; re-running for n_seeds=${N_SEEDS} ==="
  fi

  # Per-split torch.compile / triton cache dirs. Without these, parallel splits
  # race on the same /tmp/torchinductor_root/ files and crash with
  # FileNotFoundError mid-compile.
  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== pancreas Linear-FM sweep, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=pancreas \
    +experiment=power_law_sweep_linear \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== pancreas split ${split} done ==="
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
