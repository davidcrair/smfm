# Check 12-seed B200 runs

Two jobs submitted 2026-04-11 on B200 partition to test whether the
biharmonic / score-net effects survive at higher seed count after the
3-seed results flipped across GPUs.

## Files to read when done

- `mm_12seed_b200.txt` — embryoid body (3 methods, 12 seeds, ~3h)
- `stark_12seed_b200.txt` — Stark synthetic (K ∈ {20,40,60,80}, 12 seeds, ~2h)

Check job status: `squeue -u $USER`

## mm_12seed_b200.txt — what to look for

Scroll to the `CHAINED EVAL` and `PER-SEGMENT EVAL` tables at the bottom.
Three rows:

1. `MM+SLERP` — baseline
2. `MM+SLERP+Biharmonic` — isolates OT-cost-only effect
3. `MM+Score_learned+Biharmonic` — full combination

**Decision rule** (chained mean column):
- Biharmonic rows beat SLERP by more than `std/√12 ≈ std/3.5` with
  non-overlapping bars → biharmonic is real, 3-seed runs were
  underpowered. Paper keeps biharmonic as headline.
- Means overlap within ±std/√12 → confirmed null. Paper pivots to
  honest negative result.

Compare against:
- `mm_b200_results.txt` (old 3-seed B200, biharmonic was −21%)
- `mm_h200_results.txt` (old 3-seed H200, biharmonic was +20% worse)
- `timed_bihar.txt` (f3f7747 3-seed B200, biharmonic was −5.5%)

## stark_12seed_b200.txt — what to look for

Per-K table at the bottom. Four methods × four K values. The question:
does `FF (OT) + Score` consistently beat `FF (OT)` at K=20 (where EMD
kicks in) or is that effect also seed noise?

K=20 is the key cell — biggest historical delta. Compare against
`stark_b200_results.txt` per-K means.

## If you want me to run the analysis

Paste both final tables (just the CHAINED/PER-SEGMENT for MM and the
per-K summary for Stark) into the chat and I'll compute the std-error
bars and the GPU-noise comparison against prior runs.
