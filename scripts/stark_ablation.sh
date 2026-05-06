#!/usr/bin/env bash
# Stark low-K ablation: isolate OT solver and score net changes.
#
# Creates 4 git worktrees with different code states, runs each for K=20
# only, 3 seeds each, 50k iters. Compares final KL.
#
# Usage:
#   bash scripts/stark_ablation.sh
#
# Expects to be run from the repo root on a machine with a GPU.
# Run time: ~4×10 min = ~40 min on one GPU, or in parallel via SLURM.
#
# After running, compare:
#   grep "FF (OT)\|FF (OT) + Score" stark_ablation_*/stark_ablation_results.txt

set -euo pipefail

# Commits to test
CURRENT_COMMIT="$(git rev-parse HEAD)"
OLD_SINKHORN_OLD_SCORE="2f1c6bf"  # pre-improvements: Sinkhorn OT + old score net
NEW_EMD_NEW_SCORE="$CURRENT_COMMIT"  # current: EMD-at-low-K + new score net

WORKTREE_ROOT="/tmp/surf_ablation_$$"
mkdir -p "$WORKTREE_ROOT"

# State 1: old code baseline
echo "== Setting up worktree: old (Sinkhorn + old score) =="
git worktree add "$WORKTREE_ROOT/old" "$OLD_SINKHORN_OLD_SCORE" 2>/dev/null || true

# State 2: current code
echo "== Setting up worktree: new (EMD + new score) =="
git worktree add "$WORKTREE_ROOT/new" "$NEW_EMD_NEW_SCORE" 2>/dev/null || true

# Hardlink embryoid_body.h5ad into each worktree (avoid copy)
if [ -f "embryoid_body.h5ad" ]; then
    ln "embryoid_body.h5ad" "$WORKTREE_ROOT/old/" 2>/dev/null || cp "embryoid_body.h5ad" "$WORKTREE_ROOT/old/"
    ln "embryoid_body.h5ad" "$WORKTREE_ROOT/new/" 2>/dev/null || cp "embryoid_body.h5ad" "$WORKTREE_ROOT/new/"
fi

run_stark_k20() {
    local worktree="$1"
    local label="$2"
    local out="${label}_stark_k20.txt"
    echo "== Running Stark K=20 in $label =="
    cd "$worktree"
    uv run python main.py stark --stark-iters 50000 --stark-seeds 3 --stark-K 20 --batch-size 1024 > "../../${out}" 2>&1
    cd - > /dev/null
    echo "   → ${out}"
}

run_stark_k20 "$WORKTREE_ROOT/old" "old"
run_stark_k20 "$WORKTREE_ROOT/new" "new"

echo ""
echo "== Summary =="
echo ""
echo "OLD CODE (Sinkhorn + old score net):"
grep -E "KL = |^K=" old_stark_k20.txt || true
echo ""
echo "NEW CODE (EMD at K≤50 + improved score net):"
grep -E "KL = |^K=" new_stark_k20.txt || true
echo ""
echo "Worktrees in $WORKTREE_ROOT — remove with:"
echo "  git worktree remove $WORKTREE_ROOT/old"
echo "  git worktree remove $WORKTREE_ROOT/new"
