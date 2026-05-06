r"""
Aggregate Paul15 (Paul 2015) erythroid 5-stage trajectory linear-trainer
sweep logs into a LaTeX table mirroring the EB 80/10/10 linear-only format.

Reads logs/bonemarrow_power_law_linear_b512_split{42..46}.log, parses CHAINED
and PER-SEGMENT MMD^2 / MMD^2_otpfm / SWD blocks, computes per-method
mean +/- std across 5 splits, marks per-column best (\textbf) and
second-best (\underline), and emits a `.tex` snippet to
surf_latex/final_report/bonemarrow_spectral_alpha_sweep.tex.

Also reports paired Wilcoxon p-values for each spectral-OT row vs the
MM+Linear baseline on the chained-mean column.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import stats


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]
LOGS = [LOG_DIR / f"bonemarrow_power_law_linear_b512_split{s}.log" for s in SPLITS]

METHOD_MAP = {
    "MM+Linear":                              ("Linear FM",                       0),
    "MM+Linear+Random":                       ("Random OT",                       1),
    "MM+Linear+SquaredSpectral@alpha=0":      (r"\ourmodel $\alpha=0$",           2),
    "MM+Linear+SquaredSpectral@alpha=0.5":    (r"\ourmodel $\alpha=0.5$",         3),
    "MM+Linear+SquaredSpectral@alpha=1":      (r"\ourmodel $\alpha=1$",           4),
    "MM+Linear+SquaredSpectral@alpha=1.5":    (r"\ourmodel $\alpha=1.5$",         5),
    "MM+Linear+SquaredSpectral@alpha=2":      (r"\ourmodel $\alpha=2$",           6),
}
N_HOP_COLS = 4
BASELINE = "MM+Linear"


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
        if len(toks) < N_HOP_COLS + 2:
            continue
        if line.lstrip().startswith(("Method", "----")):
            continue
        method = toks[0]
        if method not in METHOD_MAP:
            continue
        try:
            # Multi-seed runs format cells as "mean±std" (single token);
            # single-seed runs format as plain "mean". Strip the std off if
            # present so we work with the per-split mean either way.
            vals = [
                float(x.split("±")[0]) if x != "-" else np.nan
                for x in toks[-(N_HOP_COLS + 1):]
            ]
        except ValueError:
            continue
        rows[method] = np.array(vals, dtype=np.float64)
    return rows


def collect(title_regex: str) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for log in LOGS:
        if not log.exists():
            print(f"  WARNING: missing {log.name}")
            continue
        rows = parse_block(log.read_text(), title_regex)
        for m, arr in rows.items():
            out.setdefault(m, []).append(arr)
    return {m: np.stack(lst) for m, lst in out.items() if len(lst) == len(LOGS)}


def fmt_cell(mean: float, std: float, is_best: bool, is_second: bool) -> str:
    base = f"{mean:.3f}"
    err = rf"$\,\pm\,{std:.3f}$"
    if is_best:
        return rf"\textbf{{{base}}}{err}"
    if is_second:
        return rf"\underline{{{base}}}{err}"
    return f"{base}{err}"


def build_block(stats_dict: dict[str, np.ndarray], title: str) -> str:
    ordered = sorted(stats_dict.keys(), key=lambda k: METHOD_MAP[k][1])
    means = np.stack([stats_dict[m].mean(axis=0) for m in ordered])
    stds = np.stack([stats_dict[m].std(axis=0, ddof=1) for m in ordered])
    n_methods, n_cols = means.shape

    best_idx, second_idx = {}, {}
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

    base_idx = ordered.index(BASELINE) if BASELINE in ordered else None
    base_mean = means[base_idx, -1] if base_idx is not None else None

    rows_out = []  # section emph header suppressed per paper style
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


def paired_wilcoxon(title_regex: str) -> dict[str, tuple[float, int]]:
    base_vals: list[float] = []
    method_vals: dict[str, list[float]] = {}
    for log in LOGS:
        if not log.exists():
            continue
        rows = parse_block(log.read_text(), title_regex)
        if BASELINE not in rows:
            continue
        base_vals.append(rows[BASELINE][-1])
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
    print("Parsing 5 bonemarrow split logs...")
    chained_med = collect(r"CHAINED EVAL \[MMD\^2\] \(\d+ seeds?\)")
    chained_otpfm = collect(r"CHAINED EVAL \[MMD\^2_otpfm\] \(\d+ seeds?\)")
    persegm_med = collect(r"PER-SEGMENT EVAL \[MMD\^2\] \(\d+ seeds?\)")
    persegm_otpfm = collect(r"PER-SEGMENT EVAL \[MMD\^2_otpfm\] \(\d+ seeds?\)")

    if BASELINE not in chained_med:
        raise RuntimeError(f"Missing {BASELINE} baseline in parsed logs")

    chained_block = build_block(chained_med, r"Chained evaluation (integrate from $t{=}0$)")

    table = (
        r"\begin{table*}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{}" + "\n"
        + r"\label{tab:bonemarrow_spectral_alpha_sweep}" + "\n"
        + r"\renewcommand{\arraystretch}{1.15}" + "\n"
        + r"\setlength{\tabcolsep}{4pt}" + "\n"
        + r"\begin{tabular}{lcccccc}" + "\n"
        + r"\toprule" + "\n"
        + r"& \multicolumn{4}{c}{MMD$^2$ at evaluation time / stage} & & \\" + "\n"
        + r"\cmidrule(lr){2-5}" + "\n"
        + r"Method & $t{=}0.25$ & $t{=}0.50$ & $t{=}0.75$ & $t{=}1.00$ & mean & $\Delta$ \\" + "\n"
        + r"\midrule" + "\n"
        + chained_block + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table*}" + "\n"
    )

    out_path = PROJ / "surf_latex" / "final_report" / "bm_table.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {out_path}\n")

    # Plain-text echo
    def print_text(stats_dict, title):
        ordered = sorted(stats_dict.keys(), key=lambda k: METHOD_MAP[k][1])
        means = np.stack([stats_dict[m].mean(axis=0) for m in ordered])
        stds = np.stack([stats_dict[m].std(axis=0, ddof=1) for m in ordered])
        base_mean = means[ordered.index(BASELINE), -1]
        print(f"\n{title}")
        print("-" * len(title))
        print(f"{'Method':<42} {'mean over 5 splits':>22}   Δ")
        for i, m in enumerate(ordered):
            mu, sd = means[i, -1], stds[i, -1]
            d = (mu / base_mean - 1.0) * 100
            d_str = "---" if m == BASELINE else f"{('+' if d >= 0 else '')}{d:.1f}%"
            print(f"{METHOD_MAP[m][0]:<42} {mu:.4f}±{sd:.4f}      {d_str}")

    print_text(chained_med, "Chained MMD² (median)")
    print_text(chained_otpfm, "Chained MMD²_otpfm (multi-scale)")

    print("\nPaired Wilcoxon (n=5, one-sided 'less' vs MM+Linear, chained-mean column):")
    for label, sig_dict in [("MMD²", paired_wilcoxon(r"CHAINED EVAL \[MMD\^2\] \(\d+ seeds?\)")),
                            ("MMD²_otpfm", paired_wilcoxon(r"CHAINED EVAL \[MMD\^2_otpfm\] \(\d+ seeds?\)"))]:
        print(f"\n  -- {label} --")
        for m in sorted(sig_dict, key=lambda k: METHOD_MAP[k][1]):
            p, n_wins = sig_dict[m]
            sig = " ★" if p < 0.05 else (" ." if p < 0.10 else "")
            print(f"    {METHOD_MAP[m][0]:<46}  wins={n_wins}/5  p={p:.3f}{sig}")


if __name__ == "__main__":
    main()
