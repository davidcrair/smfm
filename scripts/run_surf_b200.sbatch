#!/usr/bin/env bash
#SBATCH --job-name=surf_b200
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=gpu_b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --array=0-1

# SURF on B200 (Nvidia Blackwell, ~2250 bf16 TFLOPS, 180+ GB HBM3e).
#
# Task 0: Stark experiment, 3 seeds, batch 2048
# Task 1: MM embryoid body, 3 seeds, batch 2048, full ablation
#
# Submit with:
#   sbatch scripts/run_surf_b200.sbatch
#
# Array task filters:
#   sbatch --array=0 scripts/run_surf_b200.sbatch     # Stark only
#   sbatch --array=1 scripts/run_surf_b200.sbatch     # MM only
#
# If your cluster uses a different partition name or gres constraint,
# adjust --partition and --gres above. Common variants:
#   --partition=b200          --gres=gpu:1 --constraint=b200
#   --partition=scavenge_gpu  --gres=gpu:b200:1
#   --partition=gpu_b200      --gres=gpu:1
#
# Estimated runtime on B200 (~3x faster than RTX 6000 Pro):
#   Stark (array=0): ~20-30 min
#   MM    (array=1): ~15-20 min
# vs wall time of 3 hours, so plenty of headroom.

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

srun --chdir="$ROOT_DIR" $PYTHON_BIN main.py $entry 2>&1 | tee "${mode}_b200_results.txt"

echo "== task $SLURM_ARRAY_TASK_ID done =="
