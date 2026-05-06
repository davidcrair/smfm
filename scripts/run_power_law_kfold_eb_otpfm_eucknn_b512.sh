#!/usr/bin/env bash
# EB OTP-FM hold-out alpha sweep in 100-D PCA latent (TrajectoryNet's
# eb_velocity_v5.npz pipeline) with **Euclidean kNN** for the spectral
# Laplacian, 5 init seeds × 5 data splits.
#
# Differences vs scripts/run_power_law_kfold_eb_otpfm_b512.sh:
#   - eval.n_seeds=5 (not 1) -> proper init-variance averaging
#   - SMFM_KNN_METRIC=euclidean -> right metric for raw 100-PC R^D data
#   - PARALLEL_SPLITS support, thread caps, per-split torch caches
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

# Use Euclidean kNN -- correct for raw PCA-100 R^D data. Cosine in 100-D
# tends to concentrate near pi/2 (curse of dimensionality) and discards
# magnitude information that the PCA projection preserves.
export SMFM_KNN_METRIC=euclidean

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
LOG_SUFFIX="${LOG_SUFFIX:-}"

run_split() {
  local split="$1"
  local log="logs/eb_otpfm_pca100_eucknn_b512${LOG_SUFFIX}_split${split}.log"
  local outdir="outputs/eb_otpfm_pca100_eucknn_b512${LOG_SUFFIX}_split${split}"

  if [[ -f "$log" ]] && grep -q "PER-SEGMENT EVAL" "$log"; then
    if [[ "${N_SEEDS}" == "1" ]] || grep -q "±" "$log"; then
      echo "=== EB-OTPFM PCA100+EucKNN split ${split}: log already complete, skipping ==="
      return 0
    fi
  fi

  export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_split${split}_$$"
  export TRITON_CACHE_DIR="/tmp/triton_split${split}_$$"

  echo "=== EB-OTPFM PCA-100 + Euclidean-kNN sweep, batch=512, n_seeds=${N_SEEDS}, split=${split} ==="
  echo "    SMFM_KNN_METRIC=${SMFM_KNN_METRIC}, EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-(none)}"
  "${PY}" -u -m surf.train \
    data=eb_otpfm \
    +experiment=power_law_sweep_linear_otpfm \
    data.seed="${split}" \
    eval.n_seeds="${N_SEEDS}" \
    eval.mmd_protocol=both \
    eval.compute_fgd=false \
    training.batch_size=512 \
    training.n_iters=3000 \
    hydra.run.dir="${outdir}" \
    ${EXTRA_OVERRIDES} \
    2>&1 | tee "${log}"
  echo "=== EB-OTPFM PCA100+EucKNN split ${split} done ==="
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
