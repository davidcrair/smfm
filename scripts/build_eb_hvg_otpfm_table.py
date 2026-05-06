r"""
Aggregate EB-HVG-OTPFM Linear-trainer 5-split logs into a LaTeX table.

This is the 2000-D log1p HVG counterpart of build_eb_otpfm_table.py (which
runs on 100-PC OTP-FM-preprocessed data). The OTP-FM Table 2 published
baselines do *not* apply here -- they were measured on PCA-100, and this
script reports on raw log1p HVG -- so the published-comparison block is
dropped. Same OTP-FM holdout (stages 1, 3) and full-marginal eval.

Reads logs/eb_hvg_otpfm_power_law_linear_b512_split{42..46}.log and emits
surf_latex/final_report/eb_hvg_otpfm_alpha_sweep.tex plus a stdout summary.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]
LOGS = [LOG_DIR / f"eb_hvg_otpfm_power_law_linear_b512_split{s}.log" for s in SPLITS]

HO_COLS = [0, 2]   # t=0.25, t=0.75 held-out
TR_COLS = [1, 3]   # t=0.50, t=1.00 training
N_HOP_COLS = 4

METHOD_MAP = {
    "MM+Linear":                              ("Linear FM (Euclidean OT)",     0),
    "MM+Linear+Random":                       ("Linear FM + Random pairing",   1),
    "MM+Linear+SquaredSpectral@alpha=0":      (r"\ourmodel, $\alpha=0$",       2),
    "MM+Linear+SquaredSpectral@alpha=0.5":    (r"\ourmodel, $\alpha=0.5$",     3),
    "MM+Linear+SquaredSpectral@alpha=1":      (r"\ourmodel, $\alpha=1$",       4),
    "MM+Linear+SquaredSpectral@alpha=1.5":    (r"\ourmodel, $\alpha=1.5$",     5),
    "MM+Linear+SquaredSpectral@alpha=2":      (r"\ourmodel, $\alpha=2$ (biharmonic)", 6),
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
    found_per_log = []
    for log in LOGS:
        if not log.exists():
            print(f"  WARNING: missing {log.name}")
            found_per_log.append(0)
            continue
        rows = parse_block(log.read_text(), title_regex)
        found_per_log.append(len(rows))
        for m, arr in rows.items():
            out.setdefault(m, []).append(arr)
    n_complete = min((len(LOGS), *(len(v) for v in out.values())))
    return {m: np.stack(lst, 0) for m, lst in out.items() if len(lst) == len(LOGS)}


def fmt_cell(mean: float, std: float, is_best: bool, is_second: bool) -> str:
    base = f"{mean:.3f}"
    err = f"{{\\scriptsize\\,$\\pm${std:.3f}}}"
    if is_best:
        return f"\\textbf{{{base}}}{err}"
    if is_second:
        return f"\\underline{{{base}}}{err}"
    return f"{base}{err}"


def build_block(stats: dict[str, np.ndarray], title: str, baseline_idx: int):
    ordered = sorted(stats.keys(), key=lambda k: METHOD_MAP[k][1])
    means = np.stack([stats[m].mean(axis=0) for m in ordered], axis=0)
    stds_recovered = np.stack([stats[m].std(axis=0, ddof=1) for m in ordered], axis=0)

    for k, cols in [(5, HO_COLS), (6, TR_COLS)]:
        per_split = np.stack([stats[m][:, cols].mean(axis=1) for m in ordered], axis=0)
        means[:, k] = per_split.mean(axis=1)
        stds_recovered[:, k] = per_split.std(axis=1, ddof=1)

    display_cols = [0, 1, 2, 3, 5, 6]
    best_idx_per_col = {}
    second_idx_per_col = {}
    for c in display_cols:
        valid = ~np.isnan(means[:, c])
        if valid.sum() < 2:
            best_idx_per_col[c] = -1
            second_idx_per_col[c] = -1
            continue
        order = np.argsort(np.where(valid, means[:, c], np.inf))
        best_idx_per_col[c] = int(order[0])
        second_idx_per_col[c] = int(order[1])

    base_ho = means[baseline_idx, 5]
    deltas = (means[:, 5] / base_ho - 1.0) * 100.0

    rows: list[str] = []
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
    print("Parsing 5 split logs (EB-HVG-OTPFM)...")
    cm = collect(r"CHAINED EVAL \[MMD\^2\] \(1 seed\)")
    co = collect(r"CHAINED EVAL \[MMD\^2_otpfm\] \(1 seed\)")
    cf = collect(r"CHAINED EVAL \[FGD\] \(1 seed\)")
    cs = collect(r"CHAINED EVAL \[SWD\] \(1 seed\)")

    if not co:
        raise RuntimeError(
            "No `CHAINED EVAL [MMD^2_otpfm]` blocks found -- did the rerun "
            "actually use eval.mmd_protocol=both?"
        )

    ordered = sorted(METHOD_MAP, key=lambda k: METHOD_MAP[k][1])
    if any(m not in co for m in ordered):
        missing = [m for m in ordered if m not in co]
        raise RuntimeError(
            f"Missing methods after parsing across all 5 splits: {missing}"
        )
    baseline_idx = ordered.index("MM+Linear")

    blocks = []
    blocks.append(build_block(co, r"MMD$^2_{\mathrm{otpfm}}$ (multi-scale)", baseline_idx))
    blocks.append(build_block(cm, r"MMD$^2$ (median-heuristic single kernel)", baseline_idx))
    blocks.append(build_block(cf, r"FGD (only the final-time column $t{=}1.00$ has values)", baseline_idx))
    blocks.append(build_block(cs, r"Sliced Wasserstein Distance", baseline_idx))

    table = (
        r"""\begin{table*}[t]
\centering
\caption{Embryoid Body at the OTP-FM holdout protocol on 2000-D log1p HVG (no PCA compression). Hold-out at $t_1$ and $t_3$, full-marginal eval. Mean $\pm$ std over five data splits (42--46); lower is better. \textbf{Bold} marks the column-best, \underline{underline} the second-best. $\Delta$ is relative to Linear FM (Euclidean OT) on the held-out mean (HO).}
\label{tab:eb_hvg_otpfm_alpha_sweep}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccccc}
\toprule
& $t_1$[HO] & $t_2$[TR] & $t_3$[HO] & $t_4$[TR] & HO mean & TR mean & $\Delta$ \\
\midrule
"""
        + blocks[0] + "\n\\midrule\n"
        + blocks[1] + "\n\\midrule\n"
        + blocks[2] + "\n\\midrule\n"
        + blocks[3]
        + "\n"
        + r"""\bottomrule
\end{tabular}
\end{table*}
"""
    )

    out_path = PROJ / "surf_latex" / "final_report" / "eb_hvg_otpfm_alpha_sweep.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {out_path}\n")

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

    print_text(co, "MMD^2_otpfm (multi-scale)")
    print_text(cm, "MMD^2 (median-heuristic single kernel)")
    print_text(cf, "FGD")
    print_text(cs, "SWD")


if __name__ == "__main__":
    main()
