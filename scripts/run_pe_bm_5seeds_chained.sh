#!/usr/bin/env bash
# Chained PE -> BM 5-seed-per-split run, designed for an unattended Colab
# session. Each split's log is written separately, and any split whose log
# already contains the PER-SEGMENT EVAL block is skipped, so the script is
# resumable after disconnects or partial failures.
#
# Usage:
#   bash scripts/run_pe_bm_5seeds_chained.sh
#   # or, in background so closing the Colab tab doesn't kill it:
#   nohup bash scripts/run_pe_bm_5seeds_chained.sh > logs/chained_run.log 2>&1 &
#
# Override seed count via env (e.g. for a faster sanity check):
#   N_SEEDS=3 bash scripts/run_pe_bm_5seeds_chained.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export N_SEEDS="${N_SEEDS:-5}"

START_TS=$(date +%s)
echo "=================================================================="
echo "Starting chained PE -> BM run with N_SEEDS=${N_SEEDS}"
echo "Started at: $(date)"
echo "=================================================================="

echo
echo ">>> Stage 1/2: pancreas (PE)"
bash scripts/run_power_law_kfold_pancreas_linear_b512.sh

PE_TS=$(date +%s)
echo
echo "=================================================================="
echo "PE done at: $(date)  (elapsed: $(( (PE_TS - START_TS) / 60 )) min)"
echo "=================================================================="

echo
echo ">>> Stage 2/2: bonemarrow (BM)"
bash scripts/run_power_law_kfold_bonemarrow_linear_b512.sh

END_TS=$(date +%s)
echo
echo "=================================================================="
echo "ALL DONE at: $(date)"
echo "Total elapsed: $(( (END_TS - START_TS) / 60 )) min"
echo "PE elapsed:    $(( (PE_TS - START_TS) / 60 )) min"
echo "BM elapsed:    $(( (END_TS - PE_TS) / 60 )) min"
echo "=================================================================="
