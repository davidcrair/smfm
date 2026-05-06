#!/usr/bin/env bash
#SBATCH --job-name=surf
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-1

# SURF: Score-regularized Unitary Riemannian Flows
#
# Array task 0: Stark et al. toy experiment (Figure 3 reproduction)
# Array task 1: MM embryoid body final ablation
#
# usage:
#   sbatch --partition=gpu_rtx6000 scripts/run_surf.sbatch

set -euo pipefail

ROOT_DIR="${SURF_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

CONFIGS=(
  "stark --stark-iters 50000 --stark-seeds 3 --batch-size 1024"
  "mm --n-iters 3000 --mm-score-iters 3000 --ot-subsample 5000 --batch-size 1024 --mm-score-net-sigma 0.7 --mm-alpha 0.005"
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

srun --chdir="$ROOT_DIR" $PYTHON_BIN main.py $entry 2>&1 | tee "${mode}_results.txt"

echo "== task $SLURM_ARRAY_TASK_ID done =="
