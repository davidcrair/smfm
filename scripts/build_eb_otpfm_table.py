r"""
Aggregate EB-OTPFM Linear-trainer 5-split logs into a LaTeX table that puts
Ours alongside published OTP-FM / MMFM Table 2 numbers.

Reads CHAINED EVAL [MMD^2_otpfm] (the multi-scale convention OTP-FM uses) +
optionally CHAINED EVAL [MMD^2] (median heuristic) + FGD + SWD blocks per
split, computes mean +/- std across splits, marks per-column best (\textbf)
and second-best (\underline), and emits LaTeX.

Output:
  - surf_latex/final_report/eb_otpfm_alpha_sweep.tex
  - Plain-text echo to stdout for immediate inspection.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]
LOGS = [LOG_DIR / f"eb_otpfm_power_law_linear_b512_split{s}.log" for s in SPLITS]

# Column convention: stages 0..4 -> times {0, 0.25, 0.5, 0.75, 1.0}; with
# OTP-FM holdout, stages 1, 3 are HO (held-out) and 2, 4 are TR (training).
HO_COLS = [0, 2]   # t=0.25, t=0.75
TR_COLS = [1, 3]   # t=0.50, t=1.00
N_HOP_COLS = 4

# Method ordering for the table.
METHOD_MAP = {
    "MM+Linear":                              ("Linear FM (Euclidean OT)",     0),
    "MM+Linear+Random":                       ("Linear FM + Random pairing",   1),
    "MM+Linear+SquaredSpectral@alpha=0":      (r"\ourmodel, $\alpha=0$",       2),
    "MM+Linear+SquaredSpectral@alpha=0.5":    (r"\ourmodel, $\alpha=0.5$",     3),
    "MM+Linear+SquaredSpectral@alpha=1":      (r"\ourmodel, $\alpha=1$",       4),
    "MM+Linear+SquaredSpectral@alpha=1.5":    (r"\ourmodel, $\alpha=1.5$",     5),
    "MM+Linear+SquaredSpectral@alpha=2":      (r"\ourmodel, $\alpha=2$ (biharmonic)", 6),
}

# Published OTP-FM Table 2 baselines (extracted by hand from the paper).
PUBLISHED = {
    # method_label: {(metric, col_or_aggregate): (mean, std)}
    "MMFM (Rohbeck 2025, reproduced)": {
        "mmd_t1": (0.207, 0.005), "mmd_t3": (0.16, 0.01),
        "mmd_rest": (0.020, 0.004), "fgd_t1": (6.22, 0.03),
        "fgd_t3": (6.27, 0.07), "fgd_rest": (3.53, 0.06),
        "swd_t1": (0.44, 0.02), "swd_t3": (0.46, 0.01), "swd_rest": (0.20, 0.01),
    },
    r"OTP-FM ($\mathcal{W}_2^2$, their best)": {
        "mmd_t1": (0.194, 0.007), "mmd_t3": (0.082, 0.005),
        "mmd_rest": (0.006, 0.001), "fgd_t1": (6.30, 0.05),
        "fgd_t3": (5.6, 0.1), "fgd_rest": (2.5, 0.2),
        "swd_t1": (0.45, 0.03), "swd_t3": (0.41, 0.04), "swd_rest": (0.18, 0.02),
    },
}


def parse_block(text: str, title_regex: str) -> dict[str, np.ndarray]:
    pat = re.compile(
        r"={80}\s*\n\s*" + title_regex + r"\s*\n={80}\s*\n"
        r"(?P<body>.*?)(?=\n={80}|\n\s*Metric conventions:|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        return {}
    rows: dict[str, np.ndarray] = {}
    for line in m.group("body").splitlines():
        toks = line.split()
        if len(toks) < 6 or line.lstrip().startswith(("Method", "----")):
            continue
        method = toks[0]
        if method not in METHOD_MAP:
            continue
        vals = []
        for x in toks[1:]:
            if x == "-":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(x))
                except ValueError:
                    vals.append(np.nan)
        rows[method] = np.array(vals, dtype=np.float64)
    return rows


def collect(title_regex: str) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for log in LOGS:
        rows = parse_block(log.read_text(), title_regex)
        for m, arr in rows.items():
            out.setdefault(m, []).append(arr)
    return {m: np.stack(lst, 0) for m, lst in out.items() if len(lst) == len(LOGS)}


def fmt_cell(mean: float, std: float, is_best: bool, is_second: bool) -> str:
    base = f"{mean:.3f}"
    err = f"$\\,\\pm\\,{std:.3f}$"
    if is_best:
        return f"\\textbf{{{base}}}{err}"
    if is_second:
        return f"\\underline{{{base}}}{err}"
    return f"{base}{err}"


def build_block(stats: dict[str, np.ndarray], title: str, baseline_idx: int):
    """
    Build one LaTeX subsection for a single metric. Each row's columns are
    t1[HO], t2[TR], t3[HO], t4[TR], HO_mean, TR_mean, Delta_vs_baseline.

    `stats[method]` is a (5, 7)-shaped array per the trainer's table format:
    columns 0..3 are t=0.25, 0.50, 0.75, 1.00; col 4 is row-mean over hops;
    cols 5, 6 are hold_mean and train_mean (per the train.py reporter).
    """
    ordered = sorted(stats.keys(), key=lambda k: METHOD_MAP[k][1])
    n_methods = len(ordered)

    # mean / std per method per column over splits
    means = np.stack([stats[m].mean(axis=0) for m in ordered], axis=0)
    stds_recovered = np.stack([stats[m].std(axis=0, ddof=1) for m in ordered], axis=0)

    # Recompute HO / TR aggregate stds from per-split per-method aggregates
    # rather than trusting the table's per-row aggregate column, which is a
    # different reduction order.
    for k, cols in [(5, HO_COLS), (6, TR_COLS)]:
        per_split = np.stack([stats[m][:, cols].mean(axis=1) for m in ordered], axis=0)
        means[:, k] = per_split.mean(axis=1)
        stds_recovered[:, k] = per_split.std(axis=1, ddof=1)

    # Best / second-best per column index in [0, 1, 2, 3, 5, 6] (skip col 4)
    display_cols = [0, 1, 2, 3, 5, 6]
    best_idx_per_col = {}
    second_idx_per_col = {}
    for c in display_cols:
        order = np.argsort(means[:, c])
        best_idx_per_col[c] = int(order[0])
        second_idx_per_col[c] = int(order[1])

    # Delta vs baseline -> use HO_mean (col 5)
    base_ho = means[baseline_idx, 5]
    deltas = (means[:, 5] / base_ho - 1.0) * 100.0

    rows: list[str] = []
    if title:
        rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\")
    for i, m in enumerate(ordered):
        cells = []
        for c in display_cols:
            cells.append(fmt_cell(
                means[i, c], stds_recovered[i, c],
                is_best=(best_idx_per_col[c] == i),
                is_second=(second_idx_per_col[c] == i),
            ))
        if i == baseline_idx:
            delta_str = "---"
        else:
            sign = "-" if deltas[i] < 0 else "+"
            delta_str = rf"${sign}{abs(deltas[i]):.1f}\%$"
        line = METHOD_MAP[m][0]
        for c in cells:
            line += "\n  & " + c
        line += "\n  & " + delta_str + r" \\"
        rows.append(line)
    return "\n".join(rows)


def main():
    print("Parsing 5 split logs...")
    cm  = collect(r"CHAINED EVAL \[MMD\^2\] \(1 seed\)")
    co  = collect(r"CHAINED EVAL \[MMD\^2_otpfm\] \(1 seed\)")
    cf  = collect(r"CHAINED EVAL \[FGD\] \(1 seed\)")
    cs  = collect(r"CHAINED EVAL \[SWD\] \(1 seed\)")

    if not co:
        raise RuntimeError(
            "No `CHAINED EVAL [MMD^2_otpfm]` blocks found -- did the rerun "
            "actually use eval.mmd_protocol=both?"
        )

    ordered = sorted(METHOD_MAP, key=lambda k: METHOD_MAP[k][1])
    if any(m not in co for m in ordered):
        missing = [m for m in ordered if m not in co]
        raise RuntimeError(f"Missing methods after parsing: {missing}")
    baseline_idx = ordered.index("MM+Linear")

    # Only emit MMD^2_{otpfm} block to keep the table focused (the protocol-
    # native metric directly comparable to OTP-FM Table 4). FGD / SWD /
    # median-heuristic MMD blocks are still computed but no longer rendered.
    block = build_block(co, "", baseline_idx)

    table = (
        r"""\begin{table*}[t]
\centering
\caption{}
\label{tab:eb_otpfm_alpha_sweep}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
& $t_1$[HO] & $t_2$[TR] & $t_3$[HO] & $t_4$[TR] & HO mean & TR mean & $\Delta$ \\
\midrule
"""
        + block + "\n"
        + r"""\bottomrule
\end{tabular}
\end{table*}
"""
    )

    out_path = PROJ / "surf_latex" / "final_report" / "eb_otpfm_alpha_sweep.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {out_path}\n")

    # ---- Plain-text echo for immediate inspection ----
    def print_text(stats, title):
        ordered = sorted(stats.keys(), key=lambda k: METHOD_MAP[k][1])
        means = np.stack([stats[m].mean(axis=0) for m in ordered], axis=0)
        stds = np.stack([stats[m].std(axis=0, ddof=1) for m in ordered], axis=0)
        for k, cols in [(5, HO_COLS), (6, TR_COLS)]:
            per_split = np.stack([stats[m][:, cols].mean(axis=1) for m in ordered], axis=0)
            means[:, k] = per_split.mean(axis=1)
            stds[:, k] = per_split.std(axis=1, ddof=1)
        base_ho = means[ordered.index("MM+Linear"), 5]
        deltas = (means[:, 5] / base_ho - 1.0) * 100.0

        print(f"\n{title}")
        print("-" * len(title))
        cols_disp = [0, 1, 2, 3, 5, 6]
        col_labels = ["t1[HO]", "t2[TR]", "t3[HO]", "t4[TR]", "HO mean", "TR mean"]
        head = f"{'Method':<32} " + " ".join(f"{c:>13}" for c in col_labels) + "  Delta"
        print(head)
        for i, m in enumerate(ordered):
            cells = " ".join(f"{means[i,c]:>5.3f}±{stds[i,c]:<5.3f}" for c in cols_disp)
            d = "---" if m == "MM+Linear" else f"{('+' if deltas[i]>=0 else '')}{deltas[i]:5.1f}%"
            print(f"{METHOD_MAP[m][0]:<32} {cells}  {d}")

    print_text(co, "MMD^2_otpfm (multi-scale; comparable to OTP-FM published)")
    print_text(cm, "MMD^2 (median-heuristic single kernel)")
    print_text(cf, "FGD")
    print_text(cs, "SWD")

    # Anchor against published numbers
    print()
    print("Published reference (OTP-FM Table 2):")
    for label, vals in PUBLISHED.items():
        v = vals
        print(f"  {label}")
        print(f"    MMD^2_otpfm  t1={v['mmd_t1'][0]:.3f}±{v['mmd_t1'][1]:.3f}, "
              f"t3={v['mmd_t3'][0]:.3f}±{v['mmd_t3'][1]:.3f}, "
              f"Rest={v['mmd_rest'][0]:.3f}±{v['mmd_rest'][1]:.3f}")
        print(f"    FGD          t1={v['fgd_t1'][0]:.2f}±{v['fgd_t1'][1]:.2f}, "
              f"t3={v['fgd_t3'][0]:.2f}±{v['fgd_t3'][1]:.2f}, "
              f"Rest={v['fgd_rest'][0]:.2f}±{v['fgd_rest'][1]:.2f}")
        print(f"    SWD          t1={v['swd_t1'][0]:.2f}±{v['swd_t1'][1]:.2f}, "
              f"t3={v['swd_t3'][0]:.2f}±{v['swd_t3'][1]:.2f}, "
              f"Rest={v['swd_rest'][0]:.2f}±{v['swd_rest'][1]:.2f}")


if __name__ == "__main__":
    main()
