#!/usr/bin/env python
"""
Compare two Stark experiment output files side-by-side.

Usage:
  python scripts/compare_runs.py stark_b200_results.txt stark_sigma_fix_results.txt
  python scripts/compare_runs.py stark_run_old.txt stark_run_new.txt [--method "FF (OT) + Score"]

Extracts per-K, per-seed KL values from both files and prints a table
showing old vs new vs delta for each (K, seed, method) combination.
Also prints a per-K mean comparison.
"""

import re
import sys
from collections import defaultdict


def parse_stark_file(path):
    """Return dict {(K, seed, method): KL}."""
    results = {}
    current_K = None
    current_seed = None
    with open(path) as f:
        for line in f:
            m = re.match(r"K=(\d+),\s+seed=(\d+)", line)
            if m:
                current_K = int(m.group(1))
                current_seed = int(m.group(2))
                continue
            m = re.match(r"\s*(FF \(No OT\)|FF \(OT\)|FF \(OT\) \+ Score|Linear):\s*KL\s*=\s*([\d.]+)", line)
            if m and current_K is not None:
                method = m.group(1)
                kl = float(m.group(2))
                results[(current_K, current_seed, method)] = kl
    return results


def summarize(results, label):
    """Return dict {(K, method): (mean, [per-seed KLs])}."""
    by_k_method = defaultdict(list)
    for (K, seed, method), kl in results.items():
        by_k_method[(K, method)].append((seed, kl))
    summary = {}
    for (K, method), rows in by_k_method.items():
        rows.sort()
        kls = [kl for _, kl in rows]
        mean = sum(kls) / len(kls)
        summary[(K, method)] = (mean, kls)
    return summary


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    path_old = sys.argv[1]
    path_new = sys.argv[2]

    old = parse_stark_file(path_old)
    new = parse_stark_file(path_new)

    sum_old = summarize(old, "old")
    sum_new = summarize(new, "new")

    all_K = sorted({K for (K, _) in set(sum_old) | set(sum_new)})
    methods = ["FF (No OT)", "FF (OT)", "FF (OT) + Score", "Linear"]

    print(f"\n{'='*78}")
    print(f"OLD: {path_old}")
    print(f"NEW: {path_new}")
    print("=" * 78)

    for K in all_K:
        print(f"\nK={K}")
        print(f"  {'Method':<22}  {'OLD mean':>10}  {'NEW mean':>10}  {'Δ':>10}  {'Δ%':>8}")
        print("  " + "-" * 68)
        for method in methods:
            key = (K, method)
            old_row = sum_old.get(key)
            new_row = sum_new.get(key)
            if old_row is None and new_row is None:
                continue
            old_mean = old_row[0] if old_row else None
            new_mean = new_row[0] if new_row else None
            if old_mean is not None and new_mean is not None:
                delta = new_mean - old_mean
                pct = (delta / old_mean * 100) if old_mean else 0.0
                old_s = f"{old_mean:10.4f}"
                new_s = f"{new_mean:10.4f}"
                delta_s = f"{delta:+10.4f}"
                pct_s = f"{pct:+7.1f}%"
            else:
                old_s = f"{old_mean:10.4f}" if old_mean is not None else "      —   "
                new_s = f"{new_mean:10.4f}" if new_mean is not None else "      —   "
                delta_s = "      —   "
                pct_s = "      —"
            print(f"  {method:<22}  {old_s}  {new_s}  {delta_s}  {pct_s}")

        # Score-vs-OT improvement for each run
        ot_old = sum_old.get((K, "FF (OT)"), (None, None))[0]
        score_old = sum_old.get((K, "FF (OT) + Score"), (None, None))[0]
        ot_new = sum_new.get((K, "FF (OT)"), (None, None))[0]
        score_new = sum_new.get((K, "FF (OT) + Score"), (None, None))[0]
        if None not in (ot_old, score_old, ot_new, score_new):
            gain_old = (score_old - ot_old) / ot_old * 100
            gain_new = (score_new - ot_new) / ot_new * 100
            print(f"  {'→ Score vs OT':<22}  {gain_old:+9.1f}%  {gain_new:+9.1f}%")


if __name__ == "__main__":
    main()
