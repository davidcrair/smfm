#!/usr/bin/env bash
# Parallelized PE + BM 5-seed run for an A100-class GPU.
#
# Two levels of parallelism:
#  (1) PE and BM are launched concurrently as separate processes (always on).
#  (2) Within each dataset, ${PARALLEL_SPLITS} of the 5 data splits run in
#      parallel (default 1 = sequential within dataset; set to 2-5 to dial up).
#
# Each (dataset, split) gets its own log file via the existing per-split tee,
# so partial-completion is recoverable and the parser/builder doesn't change.
#
# Usage:
#   bash scripts/run_pe_bm_5seeds_parallel.sh                         # safe: 2 procs total
#   PARALLEL_SPLITS=2 bash scripts/run_pe_bm_5seeds_parallel.sh       # 4 procs
#   PARALLEL_SPLITS=5 bash scripts/run_pe_bm_5seeds_parallel.sh       # max: 10 procs
#
#   # Keep alive across Colab tab close:
#   nohup bash scripts/run_pe_bm_5seeds_parallel.sh > logs/par_run.log 2>&1 &
#
# Notes on parallelism limits:
#   - The MLP forward/backward is small relative to A100 capacity, so multiple
#     concurrent runs share the GPU via CUDA's time-slicing without wallclock
#     loss until you saturate memory (~80 GB) or CPU (OT/eigendecomp is CPU-
#     bound). For your 4x256 MLP + 5000-cell pancreas, ~10 parallel processes
#     fit comfortably; CPU contention shows up first.
#   - If you see CUDA OOM, lower PARALLEL_SPLITS. If wallclock isn't improving
#     past PARALLEL_SPLITS=2, CPU is the bottleneck (OT/eigendecomp).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export N_SEEDS="${N_SEEDS:-5}"
export PARALLEL_SPLITS="${PARALLEL_SPLITS:-1}"
mkdir -p logs

START_TS=$(date +%s)
echo "=================================================================="
echo "Starting parallel PE + BM run with N_SEEDS=${N_SEEDS}, PARALLEL_SPLITS=${PARALLEL_SPLITS}"
echo "Started at: $(date)"
echo "=================================================================="

# Launch both datasets concurrently.
bash scripts/run_power_law_kfold_pancreas_linear_b512.sh > logs/pe_run.log 2>&1 &
PE_PID=$!
echo "  PE launched (pid=${PE_PID}); tail -f logs/pe_run.log"

bash scripts/run_power_law_kfold_bonemarrow_linear_b512.sh > logs/bm_run.log 2>&1 &
BM_PID=$!
echo "  BM launched (pid=${BM_PID}); tail -f logs/bm_run.log"
echo

# Wait on each, capture exit codes and elapsed times.
wait ${PE_PID}
PE_EXIT=$?
PE_TS=$(date +%s)
echo "------------------------------------------------------------------"
echo "PE finished at $(date)  exit=${PE_EXIT}  elapsed=$(( (PE_TS - START_TS) / 60 )) min"

wait ${BM_PID}
BM_EXIT=$?
BM_TS=$(date +%s)
echo "BM finished at $(date)  exit=${BM_EXIT}  elapsed=$(( (BM_TS - START_TS) / 60 )) min"

echo "=================================================================="
echo "ALL DONE  total=$(( (BM_TS - START_TS) / 60 )) min"
if (( PE_EXIT != 0 )); then echo "  WARNING: PE exited with code ${PE_EXIT}"; fi
if (( BM_EXIT != 0 )); then echo "  WARNING: BM exited with code ${BM_EXIT}"; fi
echo "=================================================================="
