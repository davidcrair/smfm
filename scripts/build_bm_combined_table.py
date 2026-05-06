r"""
Stacked MMD^2 + SWD chained-evaluation table for bonemarrow.

Reads logs/bonemarrow_power_law_linear_b512_split{N}.log, parses CHAINED
EVAL [MMD^2] and CHAINED EVAL [SWD] blocks (multi-seed ``mean±std`` cells),
aggregates per-stage means + per-row mean across splits, and emits

    surf_latex/final_report/bm_combined_table.tex

with two sub-blocks (MMD^2 on top, SWD on bottom) sharing a single header
row. Per-column best is bold; second-best is underlined; the "Δ" column
reports % change vs MM+Linear on the row mean.

Usage:
    .venv/bin/python scripts/build_bm_combined_table.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]   # script will skip any missing/incomplete log

LOG_PATTERN = "bonemarrow_power_law_linear_b512_split{s}.log"

METHOD_MAP = {
    "MM+Linear":                            ("Linear FM",                  0),
    "MM+Linear+Random":                     ("Random OT",                  1),
    "MM+Linear+SquaredSpectral@alpha=0":    (r"\ourmodel $\alpha=0$",      2),
    "MM+Linear+SquaredSpectral@alpha=0.5":  (r"\ourmodel $\alpha=0.5$",    3),
    "MM+Linear+SquaredSpectral@alpha=1":    (r"\ourmodel $\alpha=1$",      4),
    "MM+Linear+SquaredSpectral@alpha=1.5":  (r"\ourmodel $\alpha=1.5$",    5),
    "MM+Linear+SquaredSpectral@alpha=2":    (r"\ourmodel $\alpha=2$",      6),
}
N_HOP_COLS = 4
BASELINE = "MM+Linear"


def parse_block(text: str, title_regex: str) -> dict[str, np.ndarray]:
    """Parse one ``<TITLE> EVAL [METRIC] (n seeds)`` block.

    Returns dict mapping method-name -> array of length N_HOP_COLS+1
    (per-stage means followed by the printed row mean).
    """
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
        if len(toks) < N_HOP_COLS + 2 or line.lstrip().startswith(("Method", "----")):
            continue
        method = toks[0]
        if method not in METHOD_MAP:
            continue
        try:
            vals = [
                float(x.split("±")[0]) if x != "-" else np.nan
                for x in toks[-(N_HOP_COLS + 1):]
            ]
        except ValueError:
            continue
        rows[method] = np.array(vals, dtype=np.float64)
    return rows


def collect(title_regex: str) -> dict[str, np.ndarray]:
    """For each split log, parse the block; stack per-method arrays into
    (n_splits_present, N_HOP_COLS+1) arrays. Splits whose log is missing or
    has no parsable block are silently dropped."""
    out: dict[str, list[np.ndarray]] = {}
    n_splits_seen = 0
    for s in SPLITS:
        f = LOG_DIR / LOG_PATTERN.format(s=s)
        if not f.exists():
            continue
        rows = parse_block(f.read_text(), title_regex)
        if not rows:
            continue
        n_splits_seen += 1
        for m, arr in rows.items():
            out.setdefault(m, []).append(arr)
    # only keep methods that appear in every split we actually parsed
    return {m: np.stack(v) for m, v in out.items() if len(v) == n_splits_seen}


def fmt_cell(mean: float, std: float, is_best: bool, is_second: bool,
             precision: int) -> str:
    base = f"{mean:.{precision}f}"
    err = rf"$\,\pm\,{std:.{precision}f}$"
    if is_best:
        return rf"\textbf{{{base}}}{err}"
    if is_second:
        return rf"\underline{{{base}}}{err}"
    return f"{base}{err}"


def render_block(stats: dict[str, np.ndarray], header_label: str,
                 precision: int) -> str:
    """Render rows for one metric block. Header_label is shown above the
    method rows via a multicolumn-emph line. Per-column best/second-best is
    marked. Δ column = % change of row-mean vs BASELINE row-mean."""
    ordered = sorted(stats.keys(), key=lambda k: METHOD_MAP[k][1])
    means = np.stack([stats[m].mean(axis=0) for m in ordered])
    stds = np.stack([stats[m].std(axis=0, ddof=1) for m in ordered])
    n_methods, n_cols = means.shape

    best_idx, second_idx = {}, {}
    for c in range(n_cols):
        col = means[:, c]
        order = np.argsort(col)
        best_idx[c] = int(order[0])
        second_idx[c] = int(order[1])

    base_idx = ordered.index(BASELINE) if BASELINE in ordered else None
    base_mean = means[base_idx, -1] if base_idx is not None else None

    lines = [rf"\multicolumn{{7}}{{l}}{{\emph{{{header_label}}}}} \\"]
    for i, m in enumerate(ordered):
        cells = [
            fmt_cell(means[i, c], stds[i, c],
                     is_best=(best_idx[c] == i),
                     is_second=(second_idx[c] == i),
                     precision=precision)
            for c in range(n_cols)
        ]
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
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    print("Parsing chained MMD^2 and SWD eval blocks...")
    mmd2 = collect(r"CHAINED EVAL \[MMD\^2\] \(\d+ seeds?\)")
    swd = collect(r"CHAINED EVAL \[SWD\] \(\d+ seeds?\)")

    if BASELINE not in mmd2 or BASELINE not in swd:
        raise RuntimeError("Baseline missing from one of the metric blocks")

    n_mmd_splits = next(iter(mmd2.values())).shape[0]
    n_swd_splits = next(iter(swd.values())).shape[0]
    print(f"  MMD^2 aggregated over {n_mmd_splits} splits")
    print(f"  SWD   aggregated over {n_swd_splits} splits")

    mmd_block = render_block(mmd2,  r"MMD$^2$ in compositional space", precision=3)
    swd_block = render_block(swd,   r"Sliced Wasserstein distance (SWD)", precision=4)

    table = (
        r"\begin{table*}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{}" + "\n"
        + r"\label{tab:bm_combined}" + "\n"
        + r"\renewcommand{\arraystretch}{1.15}" + "\n"
        + r"\setlength{\tabcolsep}{4pt}" + "\n"
        + r"\begin{tabular}{lcccccc}" + "\n"
        + r"\toprule" + "\n"
        + r"& \multicolumn{4}{c}{Per-stage held-out marginal} & & \\" + "\n"
        + r"\cmidrule(lr){2-5}" + "\n"
        + r"Method & $t{=}0.25$ & $t{=}0.50$ & $t{=}0.75$ & $t{=}1.00$ & mean & $\Delta$ \\" + "\n"
        + r"\midrule" + "\n"
        + mmd_block + "\n"
        + r"\midrule" + "\n"
        + swd_block + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table*}" + "\n"
    )

    out_path = PROJ / "surf_latex" / "final_report" / "bm_combined_table.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {out_path}\n")
    print(table)


if __name__ == "__main__":
    main()
