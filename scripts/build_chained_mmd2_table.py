r"""
Parametric chained-MMD^2 table builder.

Given a list of split logs from the linear-trainer sweep
(MM+Linear, MM+Linear+Random, MM+Linear+SquaredSpectral@alpha={0,0.5,1,1.5,2}),
aggregate the CHAINED EVAL [MMD^2] block across splits and emit a 4-stage
+ mean + Delta LaTeX table.

Usage (CLI):
  python scripts/build_chained_mmd2_table.py \
      --logs 'logs/bonemarrow_power_law_linear_log1p_eucknn_b512_split{s}.log' \
      --out  surf_latex/final_report/tables/bm_log1p_eucknn.tex \
      --label tab:bm_log1p_eucknn \
      --caption "Bone marrow, log1p HVG ambient + Euclidean kNN."
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


METHOD_MAP = {
    "MM+Linear":                              ("Linear FM",                       0),
    "MM+Linear+Random":                       ("Random OT",                       1),
    "MM+Linear+SquaredSpectral@alpha=0":      (r"\ourmodel $\alpha=0$",           2),
    "MM+Linear+SquaredSpectral@alpha=0.5":    (r"\ourmodel $\alpha=0.5$",         3),
    "MM+Linear+SquaredSpectral@alpha=1":      (r"\ourmodel $\alpha=1$",           4),
    "MM+Linear+SquaredSpectral@alpha=1.5":    (r"\ourmodel $\alpha=1.5$",         5),
    "MM+Linear+SquaredSpectral@alpha=2":      (r"\ourmodel $\alpha=2$",           6),
}
SPHERE_METHOD_MAP = {
    "MM+Linear":                              ("Linear FM",                       0),
    "MM+Linear+Random":                       ("Random OT",                       1),
    "MM+SLERP":                               ("FisherFlow",                      2),
    "MM+Linear+SquaredSpectral@alpha=0":      (r"\ourmodel $\alpha=0$",           3),
    "MM+Linear+SquaredSpectral@alpha=0.5":    (r"\ourmodel $\alpha=0.5$",         4),
    "MM+Linear+SquaredSpectral@alpha=1":      (r"\ourmodel $\alpha=1$",           5),
    "MM+Linear+SquaredSpectral@alpha=1.5":    (r"\ourmodel $\alpha=1.5$",         6),
    "MM+Linear+SquaredSpectral@alpha=2":      (r"\ourmodel $\alpha=2$",           7),
}
N_HOP_COLS = 4
BASELINE = "MM+Linear"


def parse_block(text: str, title_regex: str, method_map: dict) -> dict[str, np.ndarray]:
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
        if method not in method_map:
            continue
        try:
            vals = [float(x.split("±")[0]) if x != "-" else np.nan
                    for x in toks[-(N_HOP_COLS + 1):]]
        except ValueError:
            continue
        rows[method] = np.array(vals, dtype=np.float64)
    return rows


def collect(logs: list[Path], title_regex: str, method_map: dict) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for log in logs:
        if not log.exists():
            print(f"  WARNING: missing {log.name}")
            continue
        rows = parse_block(log.read_text(), title_regex, method_map)
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


def build_block(stats_dict: dict[str, np.ndarray], method_map: dict) -> str:
    ordered = sorted(stats_dict.keys(), key=lambda k: method_map[k][1])
    means = np.stack([stats_dict[m].mean(axis=0) for m in ordered])
    stds = np.stack([stats_dict[m].std(axis=0, ddof=1) for m in ordered])
    n_methods, n_cols = means.shape

    best_idx, second_idx = {}, {}
    for c in range(n_cols):
        col = means[:, c]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            best_idx[c], second_idx[c] = -1, -1
            continue
        order = np.argsort(np.where(valid, col, np.inf))
        best_idx[c], second_idx[c] = int(order[0]), int(order[1])

    base_idx = ordered.index(BASELINE) if BASELINE in ordered else None
    base_mean = means[base_idx, -1] if base_idx is not None else None

    rows_out = []
    for i, m in enumerate(ordered):
        cells = [fmt_cell(means[i, c], stds[i, c],
                          best_idx[c] == i, second_idx[c] == i)
                 for c in range(n_cols)]
        if base_idx is not None and i == base_idx:
            delta_str = "---"
        elif base_mean is not None and not np.isnan(means[i, -1]):
            d = (means[i, -1] / base_mean - 1.0) * 100
            sign = "-" if d < 0 else "+"
            delta_str = rf"${sign}{abs(d):.1f}\%$"
        else:
            delta_str = ""
        line = method_map[m][0]
        for c in cells:
            line += "\n  & " + c
        line += "\n  & " + delta_str + r" \\"
        rows_out.append(line)
    return "\n".join(rows_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True,
                   help="Glob pattern with {s} placeholder for split id")
    p.add_argument("--splits", default="42,43,44,45,46")
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--title", choices=["table", "table*"], default="table*")
    p.add_argument("--metric", default="MMD^2",
                   help="Metric name in CHAINED EVAL block (e.g. MMD^2 or MMD^2_otpfm)")
    p.add_argument("--metric-display", default="MMD$^2$",
                   help="LaTeX rendering for the metric in the column header")
    p.add_argument("--include-fisher-flow", action="store_true")
    args = p.parse_args()

    splits = [int(s) for s in args.splits.split(",")]
    logs = [Path(args.logs.format(s=s)) for s in splits]

    method_map = SPHERE_METHOD_MAP if args.include_fisher_flow else METHOD_MAP

    title_regex = rf"CHAINED EVAL \[{re.escape(args.metric)}\] \(\d+ seeds?\)"
    chained = collect(logs, title_regex, method_map)
    if BASELINE not in chained:
        raise RuntimeError(f"Missing {BASELINE} baseline in parsed logs")

    block = build_block(chained, method_map)

    out_lines = [
        rf"\begin{{{args.title}}}[t]",
        r"\centering",
        rf"\caption{{{args.caption}}}",
        rf"\label{{{args.label}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        rf"& \multicolumn{{4}}{{c}}{{{args.metric_display} at evaluation time / stage}} & & \\",
        r"\cmidrule(lr){2-5}",
        r"Method & $t{=}0.25$ & $t{=}0.50$ & $t{=}0.75$ & $t{=}1.00$ & mean & $\Delta$ \\",
        r"\midrule",
        block,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\end{{{args.title}}}",
        "",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    print(f"Wrote {out_path}")

    # Plain-text echo of chained mean column
    ordered = sorted(chained.keys(), key=lambda k: method_map[k][1])
    means = np.stack([chained[m].mean(axis=0) for m in ordered])
    stds = np.stack([chained[m].std(axis=0, ddof=1) for m in ordered])
    base_mean = means[ordered.index(BASELINE), -1]
    print(f"\n{'Method':<28} {'chained mean':>20}    Δ")
    for i, m in enumerate(ordered):
        mu, sd = means[i, -1], stds[i, -1]
        d = (mu / base_mean - 1.0) * 100
        d_str = "---" if m == BASELINE else f"{('+' if d >= 0 else '')}{d:.1f}%"
        print(f"{method_map[m][0]:<28} {mu:.4f}±{sd:.4f}    {d_str}")


if __name__ == "__main__":
    main()
