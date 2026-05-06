r"""
Aggregate EB 80/10/10 split logs (42..46) into a LaTeX table mirroring
build_pancreas_table.py's format.

Reads:
  - logs/power_law_b512_split{42..46}.log  (sphere-trainer sweep:
      MM+SLERP and MM+SLERP+SquaredSpectral@alpha={0,0.5,1,1.5,2})
  - logs/eb_8010_linear_2k_b512_split{42..46}.log  (linear-trainer sweep
      with max_cells_per_stage=2000:
      MM+Linear and MM+Linear+SquaredSpectral@alpha={0,0.5,1,1.5,2})

Computes mean +/- std across 5 splits, marks per-column best (\textbf) and
second-best (\underline), and computes Delta vs FisherFlow (MM+SLERP) for the
chained-mean column. Emits a `.tex` snippet matching the pancreas table style.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]

SPHERE_LOGS = [LOG_DIR / f"power_law_b512_split{s}.log" for s in SPLITS]
LINEAR_LOGS = [LOG_DIR / f"eb_8010_linear_2k_b512_split{s}.log" for s in SPLITS]

METHOD_MAP = {
    "MM+Linear":         ("Linear FM",          0, False),
    "MM+Linear+Random":  ("Random OT",          1, False),
    "MM+SLERP":          ("FisherFlow",         2, False),
    # Sphere trainer + spectral cost.
    "MM+SLERP+SquaredSpectral@alpha=0":   (r"\ourmodel + sphere, $\alpha=0$",   3, True),
    "MM+SLERP+SquaredSpectral@alpha=0.5": (r"\ourmodel + sphere, $\alpha=0.5$", 4, True),
    "MM+SLERP+SquaredSpectral@alpha=1":   (r"\ourmodel + sphere, $\alpha=1$",   5, True),
    "MM+SLERP+SquaredSpectral@alpha=1.5": (r"\ourmodel + sphere, $\alpha=1.5$", 6, True),
    "MM+SLERP+SquaredSpectral@alpha=2":   (r"\ourmodel + sphere, $\alpha=2$",   7, True),
    # Linear (Euclidean) trainer + spectral cost.
    "MM+Linear+SquaredSpectral@alpha=0":   (r"\ourmodel $\alpha=0$",   8, True),
    "MM+Linear+SquaredSpectral@alpha=0.5": (r"\ourmodel $\alpha=0.5$", 9, True),
    "MM+Linear+SquaredSpectral@alpha=1":   (r"\ourmodel $\alpha=1$",   10, True),
    "MM+Linear+SquaredSpectral@alpha=1.5": (r"\ourmodel $\alpha=1.5$", 11, True),
    "MM+Linear+SquaredSpectral@alpha=2":   (r"\ourmodel $\alpha=2$",   12, True),
}

N_HOP_COLS = 4
BASELINE_KEY = "MM+SLERP"  # default: FisherFlow as baseline; --linear-only switches to MM+Linear

LINEAR_ONLY_KEYS = {
    "MM+Linear",
    "MM+Linear+Random",
    "MM+Linear+SquaredSpectral@alpha=0",
    "MM+Linear+SquaredSpectral@alpha=0.5",
    "MM+Linear+SquaredSpectral@alpha=1",
    "MM+Linear+SquaredSpectral@alpha=1.5",
    "MM+Linear+SquaredSpectral@alpha=2",
}


def parse_block(text: str, block_title_regex: str) -> dict[str, np.ndarray]:
    """
    Returns method-name -> (n_hop_cols + 1,) array. Last column is row mean.
    """
    pat = re.compile(
        r"={80}\s*\n\s*" + block_title_regex + r"\s*\n={80}\s*\n"
        r"(?P<body>.*?)(?=\n={80}|\n\s*Metric conventions:|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        return {}
    rows: dict[str, np.ndarray] = {}
    for line in m.group("body").splitlines():
        toks = line.split()
        if len(toks) < N_HOP_COLS + 2:
            continue
        if line.lstrip().startswith(("Method", "----")):
            continue
        method = toks[0]
        if method not in METHOD_MAP:
            continue
        try:
            vals = [float(x) for x in toks[-(N_HOP_COLS + 1):]]
        except ValueError:
            continue
        rows[method] = np.array(vals)
    return rows


def collect(logs: list[Path], block_title_regex: str) -> dict[str, np.ndarray]:
    """Collect (n_splits, n_hop_cols+1) array per method across the given logs."""
    out: dict[str, list[np.ndarray]] = {}
    for log in logs:
        if not log.exists():
            print(f"  WARNING: missing {log.name}")
            continue
        rows = parse_block(log.read_text(), block_title_regex)
        for m, arr in rows.items():
            out.setdefault(m, []).append(arr)
    return {m: np.stack(lst) for m, lst in out.items() if len(lst) == len(logs)}


def fmt_cell(mean: float, std: float, is_best: bool, is_second: bool) -> str:
    base = f"{mean:.3f}"
    err = rf"$\,\pm\,{std:.3f}$"
    if is_best:
        return rf"\textbf{{{base}}}{err}"
    if is_second:
        return rf"\underline{{{base}}}{err}"
    return f"{base}{err}"


def build_block(stats_dict: dict[str, np.ndarray], block_title: str,
                show_baseline: str = BASELINE_KEY) -> str:
    """Build one LaTeX subsection for a given metric block (chained or per-segment)."""
    ordered = sorted(stats_dict.keys(), key=lambda k: METHOD_MAP[k][1])
    means = np.stack([stats_dict[m].mean(axis=0) for m in ordered])
    stds = np.stack([stats_dict[m].std(axis=0, ddof=1) for m in ordered])
    n_methods, n_cols = means.shape

    # Best / second-best per column (across all methods).
    best_idx = {}
    second_idx = {}
    for c in range(n_cols):
        col = means[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            best_idx[c] = -1
            second_idx[c] = -1
            continue
        order = np.argsort(np.where(valid, col, np.inf))
        best_idx[c] = int(order[0])
        second_idx[c] = int(order[1])

    # Delta vs baseline (FisherFlow / MM+SLERP) on the row-mean column.
    if show_baseline in ordered:
        base_idx = ordered.index(show_baseline)
        base_mean = means[base_idx, -1]
    else:
        base_idx = None
        base_mean = None

    rows_out = []  # section emph header suppressed per paper style
    rows_out.append(r"Method & $t{=}0.25$ & $t{=}0.50$ & $t{=}0.75$ & $t{=}1.00$ & mean & $\Delta$ \\")
    for i, m in enumerate(ordered):
        cells = []
        for c in range(n_cols):
            cells.append(fmt_cell(
                means[i, c], stds[i, c],
                is_best=(best_idx[c] == i),
                is_second=(second_idx[c] == i),
            ))
        if base_idx is not None and i == base_idx:
            delta_str = "---"
        elif base_mean is not None and not np.isnan(means[i, -1]):
            d = (means[i, -1] / base_mean - 1.0) * 100
            sign = "-" if d < 0 else "+"
            delta_str = rf"${sign}{abs(d):.1f}\%$"
        else:
            delta_str = ""
        line = METHOD_MAP[m][0]
        for c in cells:
            line += "\n  & " + c
        line += "\n  & " + delta_str + r" \\"
        rows_out.append(line)
    return "\n".join(rows_out)


def paired_significance(linear_logs: list[Path], block_title: str) -> dict[str, tuple[float, int]]:
    """
    For the linear-trainer sweep, compute paired Wilcoxon p-value (one-sided,
    'less') vs MM+Linear on the row-mean column. Returns method -> (p, n_wins).
    """
    base_vals: list[float] = []
    method_vals: dict[str, list[float]] = {}
    for log in linear_logs:
        if not log.exists():
            continue
        rows = parse_block(log.read_text(), block_title)
        if "MM+Linear" not in rows:
            continue
        base_vals.append(rows["MM+Linear"][-1])
        for m in rows:
            if m.startswith("MM+Linear+SquaredSpectral"):
                method_vals.setdefault(m, []).append(rows[m][-1])
    base_arr = np.array(base_vals)
    out = {}
    for m, vals in method_vals.items():
        v = np.array(vals)
        diffs = v - base_arr
        n_wins = int((diffs < 0).sum())
        try:
            _, p = stats.wilcoxon(diffs, alternative="less")
        except ValueError:
            p = float("nan")
        out[m] = (p, n_wins)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Drop sphere-trainer rows (MM+SLERP and \\ourmodel + sphere). "
             "Use Linear FM (MM+Linear) as the baseline for delta. Output to "
             "eb_8010_spectral_alpha_sweep_linear_only.tex.",
    )
    args = parser.parse_args()

    print("Parsing 5 sphere-trainer logs and 5 linear-trainer logs...")
    chained_sphere = collect(SPHERE_LOGS, r"CHAINED EVAL \[MMD\^2\] \(1 seed\)")
    chained_linear = collect(LINEAR_LOGS, r"CHAINED EVAL \[MMD\^2\] \(1 seed\)")
    persegm_sphere = collect(SPHERE_LOGS, r"PER-SEGMENT EVAL \[MMD\^2\] \(1 seed\)")
    persegm_linear = collect(LINEAR_LOGS, r"PER-SEGMENT EVAL \[MMD\^2\] \(1 seed\)")

    # Merge sphere + linear stats for joint table
    chained = {**chained_sphere, **chained_linear}
    persegm = {**persegm_sphere, **persegm_linear}

    if args.linear_only:
        chained = {k: v for k, v in chained.items() if k in LINEAR_ONLY_KEYS}
        persegm = {k: v for k, v in persegm.items() if k in LINEAR_ONLY_KEYS}
        baseline = "MM+Linear"
    else:
        baseline = BASELINE_KEY

    if baseline not in chained:
        raise RuntimeError(
            f"Baseline {baseline} not found in parsed logs. "
            f"Got methods: {list(chained.keys())}"
        )
    expected = list(METHOD_MAP)
    missing = [m for m in expected if m not in chained]
    if missing:
        print(f"  NOTE: missing methods (will be omitted from table): {missing}")

    # Paired significance for the linear-trainer half (same metric, MMD^2 mean).
    sig = paired_significance(LINEAR_LOGS, r"CHAINED EVAL \[MMD\^2\] \(1 seed\)")
    sig_otpfm = paired_significance(LINEAR_LOGS, r"CHAINED EVAL \[MMD\^2_otpfm\] \(1 seed\)")

    chained_block = build_block(chained, r"Chained evaluation (integrate from $t{=}0$)",
                                show_baseline=baseline)
    persegm_block = build_block(persegm, r"Per-segment evaluation (each 1-hop transition)",
                                show_baseline=baseline)

    if args.linear_only:
        out_filename = "eb_table.tex"
        label = "tab:eb_8010_spectral_alpha_sweep_linear_only"
    else:
        out_filename = "eb_8010_spectral_alpha_sweep.tex"
        label = "tab:eb_8010_spectral_alpha_sweep"

    table = (
        r"\begin{table*}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{}" + "\n"
        + r"\label{" + label + r"}" + "\n"
        + r"\renewcommand{\arraystretch}{1.15}" + "\n"
        + r"\setlength{\tabcolsep}{4pt}" + "\n"
        + r"\begin{tabular}{lcccccc}" + "\n"
        + r"\toprule" + "\n"
        + chained_block + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table*}" + "\n"
    )

    out_path = PROJ / "surf_latex" / "final_report" / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {out_path}\n")

    # Plain-text echo for inspection
    def print_text(stats_dict, title):
        ordered = sorted(stats_dict.keys(), key=lambda k: METHOD_MAP[k][1])
        means = np.stack([stats_dict[m].mean(axis=0) for m in ordered])
        stds = np.stack([stats_dict[m].std(axis=0, ddof=1) for m in ordered])
        base_mean = means[ordered.index(baseline), -1] if baseline in ordered else None
        print(f"\n{title}")
        print("-" * len(title))
        print(f"{'Method':<42} {'t=0.25':>10} {'t=0.50':>10} {'t=0.75':>10} {'t=1.00':>10}   {'mean':>14}   Δ")
        for i, m in enumerate(ordered):
            mean_str = f"{means[i,-1]:.4f}±{stds[i,-1]:.4f}"
            if base_mean is not None and m != baseline:
                d = (means[i, -1] / base_mean - 1.0) * 100
                d_str = f"{('+' if d >= 0 else '')}{d:.1f}%"
            elif m == baseline:
                d_str = "---"
            else:
                d_str = ""
            cells = " ".join(f"{means[i,c]:.4f}±{stds[i,c]:.4f}" for c in range(N_HOP_COLS))
            print(f"{METHOD_MAP[m][0]:<42} {cells}   {mean_str:>14}   {d_str}")

    print_text(chained, "Chained evaluation [MMD^2]")
    print_text(persegm, "Per-segment evaluation [MMD^2]")

    # Print paired significance for the linear-trainer half
    if sig:
        print("\nPaired Wilcoxon (linear-trainer vs MM+Linear, n=5, one-sided 'less'):")
        for k in sorted(sig, key=lambda x: METHOD_MAP[x][1]):
            p, n_wins = sig[k]
            label = METHOD_MAP[k][0]
            sig_marker = "★" if p < 0.05 else ("." if p < 0.10 else "")
            print(f"  {label:<44}  MMD²: wins={n_wins}/5  p={p:.3f} {sig_marker}")
    if sig_otpfm:
        print("\nPaired Wilcoxon on MMD²_otpfm (linear-trainer vs MM+Linear, n=5, one-sided 'less'):")
        for k in sorted(sig_otpfm, key=lambda x: METHOD_MAP[x][1]):
            p, n_wins = sig_otpfm[k]
            label = METHOD_MAP[k][0]
            sig_marker = "★" if p < 0.05 else ("." if p < 0.10 else "")
            print(f"  {label:<44}  MMD²_otpfm: wins={n_wins}/5  p={p:.3f} {sig_marker}")


if __name__ == "__main__":
    main()
