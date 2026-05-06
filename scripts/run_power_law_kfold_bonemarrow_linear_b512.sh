#!/usr/bin/env bash
# Power-law spectral OT cost sweep with the LINEAR / Euclidean trainer on
# bonemarrow (Setty 2019 / scVelo) erythroid trajectory, 5 stages, ~3.7k cells.
# Mirrors the pancreas / EB / paul15 scripts. Adds a third dataset row to the
# cross-dataset spectral-OT story with statistical power similar to pancreas.
set -euo pipefail

# Resolve project root from script location so it works on any machine
# (Mac, Colab, cluster). Override by `cd <project>` before invoking, or
# by setting PROJECT_ROOT.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python"
fi

splits=(42 43 44 45 46)
# Override the splits list via env var, e.g. SPLITS_OVERRIDE="42" to rerun
# only one split. Comma-separated.
if [[ -n "${SPLITS_OVERRIDE:-}" ]]; then
  IFS=',' read -ra splits <<< "${SPLITS_OVERRIDE}"
fi
N_SEEDS="${N_SEEDS:-5}"
PARALLEL_SPLITS="${PARALLEL_SPLITS:-1}"

# Cap BLAS thread count per process so parallel splits don't oversubscribe
# the Colab CPU pool (12 vCPUs). With THREADS_PER_PROC=3 and PARALLEL_SPLITS=3
# you use ~9 cores, leaving headroom for OS overhead and a stable load avg.
THREADS_PER_PROC="${THREADS_PER_PROC:-3}"
export OMP_NUM_THREADS="${THREADS_PER_PROC}"
export MKL_NUM_THREADS="${THREADS_PER_PROC}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_PROC}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}"
export VECLIB_MAXIMUM_THREADS="${THREADS_PER_PROC}"

run_split() {
  local split="$1"
  local log="logs/bonemarrow_power_law_linear_b512_split${split}.log"
  local outdir="outputs/bonemarrow_power_law_linear_b512_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== bonemarrow split ${split}: log already complete for n_seeds=${N_SEEDS}, skipping ==="
      return 0
    fi
    echo "=== bonemarrow split ${split}: existing log is single-seed; re-running for n_seeds=${N_SEEDS} ==="
  fi

  # Per-split torch.compile / triton cache dirs. Without these, parallel splits
  # race on the same /tmp/torchinductor_root/ files and crash with
  # FileNotFoundError mid-compile.
  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== bonemarrow Linear-FM sweep, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  "${PY}" -u -m surf.train \
    data=bonemarrow \
    +experiment=power_law_sweep_linear \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    eval.mmd_protocol=both \
    eval.compute_fgd=false \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    2>&1 | tee "${log}"
  echo "=== bonemarrow split ${split} done ==="
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
