#!/usr/bin/env bash
#SBATCH --job-name=surf_rtx6000
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# SURF on rtx6000 (scavenge partition).
#
# Task 0: Stark experiment, 3 seeds
# Task 1: MM embryoid body, 3 seeds, full ablation
#
# Submit with:
#   sbatch scripts/run_surf_rtx6000.sbatch
#
# If your cluster uses a different scavenge partition name or rtx6000 gres
# constraint, adjust --partition and --gres above. Common variants:
#   --partition=scavenge       --gres=gpu:1 --constraint=rtx6000
#   --partition=scavenge_gpu   --gres=gpu:rtx6000:1
#   --partition=gpu_rtx6000       --gres=gpu:1

set -euo pipefail

ROOT_DIR="${SURF_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"
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

srun --chdir="$ROOT_DIR" $PYTHON_BIN main.py $entry 2>&1 | tee "${mode}_rtx6000_results.txt"

echo "== task $SLURM_ARRAY_TASK_ID done =="
