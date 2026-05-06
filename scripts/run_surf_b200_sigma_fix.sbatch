#!/usr/bin/env bash
#SBATCH --job-name=surf_b200_sigma_fix
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=gpu_b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# SURF on B200 with score-σ alignment fix.
#
# Runs after commit that aligns score net training sampling (lognormal
# peak at σ=0.7) to match the inference query σ=0.7. This should restore
# the Stark K=20 improvement that regressed from −20% to −7% when the
# score net improvements were added with the default lognormal peak at
# σ=0.3.
#
# Task 0: Stark experiment, 3 seeds (writes stark_sigma_fix_results.txt)
# Task 1: MM embryoid body, 3 seeds, full ablation (writes mm_sigma_fix_results.txt)
#
# The output filenames are DIFFERENT from run_surf_b200.sbatch so you can
# compare old vs new runs side-by-side.

set -euo pipefail

ROOT_DIR="${SURF_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"
echo "Git commit: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

CONFIGS=(
  "stark --stark-iters 50000 --stark-seeds 3 --batch-size 2048"
  "mm --n-iters 3000 --mm-score-iters 3000 --ot-subsample 5000 --batch-size 2048 --mm-score-net-sigma 0.7 --mm-alpha 0.005 --mm-seeds 3"
)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error: SLURM_ARRAY_TASK_ID not set -- submit with sbatch" >&2
  exit 1
fi

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
mode=$(echo "$entry" | awk '{print $1}')

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "mode=$mode"
echo "args=$entry"

srun --chdir="$ROOT_DIR" $PYTHON_BIN main.py $entry 2>&1 | tee "${mode}_sigma_fix_results.txt"

echo "== task $SLURM_ARRAY_TASK_ID done =="
