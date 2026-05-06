#!/usr/bin/env bash
# Simple sbatch wrapper for surf.train.
#
# Usage:
#   ./scripts/submit.sh experiment=otpfm
#   ./scripts/submit.sh experiment=otpfm eval.n_seeds=12
#   ./scripts/submit.sh methods='[MM+SLERP,MM+SI]' training.si_sigma=0.05 eval.n_seeds=3
#
# Submits a single SLURM job that runs `python -m surf.train <args>`.
# For sweeps, use Hydra's --multirun with the slurm launcher instead:
#   python -m surf.train --multirun hydra/launcher=slurm experiment=si_sweep

set -euo pipefail

# Defaults — override via environment variables
PARTITION="${PARTITION:-gpu_b200}"
GRES="${GRES:-gpu:b200:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-3:00:00}"

# Build a readable job name from the first Hydra arg
FIRST_ARG="${1:-default}"
JOB_NAME="surf_${FIRST_ARG//[^a-zA-Z0-9_]/_}"

# Resolve ROOT_DIR as the directory containing the scripts/ folder,
# using the script's own location (not the caller's cwd).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python not found at $PYTHON_BIN" >&2
  exit 1
fi

echo "Submitting: python -m surf.train $*"
echo "  root=$ROOT_DIR"
echo "  partition=$PARTITION, gres=$GRES, cpus=$CPUS, mem=$MEM, time=$TIME"

sbatch \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --job-name="$JOB_NAME" \
  --output="logs/slurm-%x-%j.out" \
  --error="logs/slurm-%x-%j.err" \
  --requeue \
  --chdir="$ROOT_DIR" \
  --wrap="$PYTHON_BIN -u -m surf.train $*"
