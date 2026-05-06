r"""
Aggregate pancreas split logs (42..46) into a LaTeX table matching the
embryoid-body spectral_alpha_sweep table format.

Reads CHAINED EVAL [MMD^2] and PER-SEGMENT EVAL [MMD^2] blocks from each
per-split log, computes mean +/- std across splits, marks per-column best
(\textbf) and second-best (\underline), and computes Delta vs FisherFlow.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


LINEAR_ONLY_KEYS = {
    "MM+Linear@sphere",
    "MM+Linear+Random",
    "MM+Linear+SquaredSpectral@alpha=0",
    "MM+Linear+SquaredSpectral@alpha=0.5",
    "MM+Linear+SquaredSpectral@alpha=1",
    "MM+Linear+SquaredSpectral@alpha=1.5",
    "MM+Linear+SquaredSpectral@alpha=2",
}


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]

POWER_LAW_LOGS = [LOG_DIR / f"pancreas_power_law_b512_split{s}.log" for s in SPLITS]
LINEAR_LOGS = [LOG_DIR / f"pancreas_linear_baseline_sphere_split{s}.log" for s in SPLITS]
LINEAR_PCA50_LOGS = [
    LOG_DIR / f"pancreas_linear_baseline_pca50_split{s}.log" for s in SPLITS
]
LINEAR_SPECTRAL_LOGS = [
    LOG_DIR / f"pancreas_power_law_linear_b512_split{s}.log" for s in SPLITS
]

# Method-name mapping: synthetic key -> (display label, ordering index, is_ourmodel).
# Sphere-trainer and Linear-trainer methods are kept distinct via the
# `MM+Linear+SquaredSpectral@...` vs `MM+SLERP+SquaredSpectral@...` keys.
# The bare `MM+Linear` row appears in two experiment configs so we
# synthetically rename via the `rename_mm_linear` parser hook.
METHOD_MAP = {
    "MM+Linear@sphere": ("Linear FM",                0, False),
    "MM+Linear+Random": ("Random OT",                1, False),
    "MM+Linear@pca50":  ("Linear FM (PCA-50)",       2, False),
    "MM+SLERP":         ("FisherFlow",               3, False),
    # Sphere trainer + spectral cost (existing power_law_sweep).
    "MM+SLERP+SquaredSpectral@alpha=0":   (r"\ourmodel + sphere, $\alpha=0$",   4, True),
    "MM+SLERP+SquaredSpectral@alpha=0.5": (r"\ourmodel + sphere, $\alpha=0.5$", 5, True),
    "MM+SLERP+SquaredSpectral@alpha=1":   (r"\ourmodel + sphere, $\alpha=1$",   6, True),
    "MM+SLERP+SquaredSpectral@alpha=1.5": (r"\ourmodel + sphere, $\alpha=1.5$", 7, True),
    "MM+SLERP+SquaredSpectral@alpha=2":   (r"\ourmodel + sphere, $\alpha=2$",   8, True),
    # Linear (Euclidean) trainer + spectral cost (power_law_sweep_linear).
    "MM+Linear+SquaredSpectral@alpha=0":   (r"\ourmodel $\alpha=0$",   9, True),
    "MM+Linear+SquaredSpectral@alpha=0.5": (r"\ourmodel $\alpha=0.5$", 10, True),
    "MM+Linear+SquaredSpectral@alpha=1":   (r"\ourmodel $\alpha=1$",   11, True),
    "MM+Linear+SquaredSpectral@alpha=1.5": (r"\ourmodel $\alpha=1.5$", 12, True),
    "MM+Linear+SquaredSpectral@alpha=2":   (r"\ourmodel $\alpha=2$",   13, True),
}

# Methods we expect in each block. n_cols = 4 hop columns + 1 mean = 5 floats.
N_HOP_COLS = 4


def parse_block(text: str, block_title_regex: str,
                rename_mm_linear: str | None = None,
                skip_mm_linear: bool = False) -> dict[str, np.ndarray]:
    """
    Locate a block whose section title matches ``block_title_regex`` and return
    a dict mapping method-name -> (n_hop_cols + 1,) array of floats.

    ``rename_mm_linear`` lets the caller disambiguate the bare "MM+Linear" key
    that appears in both the sphere-baseline and pca50-baseline logs --
    parsers for those logs pass a synthetic key like ``MM+Linear@pca50``.

    ``skip_mm_linear`` drops the bare "MM+Linear" row entirely; useful when a
    log file (e.g., the Linear-trainer spectral sweep) repeats the
    `MM+Linear` baseline that another log already contributes, to avoid
    double-counting it across split parses.
    """
    pat = re.compile(
        r"={80}\s*\n\s*" + block_title_regex + r"\s*\n={80}\s*\n"
        r"(?P<body>.*?)(?=\n={80}|\n\s*Metric conventions:|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        raise RuntimeError(f"Block {block_title_regex!r} not found")
    body = m.group("body")

    rows: dict[str, np.ndarray] = {}
    for line in body.splitlines():
        line = line.rstrip()
        if not line.strip() or line.lstrip().startswith(("Method", "----", "----  ")):
            continue
        toks = line.split()
        if len(toks) < N_HOP_COLS + 2:
            continue
        method = toks[0]
        if method == "MM+Linear" and skip_mm_linear:
            continue
        if method == "MM+Linear" and rename_mm_linear is not None:
            method = rename_mm_linear
        if method not in METHOD_MAP:
            continue
        try:
            # Multi-seed runs format cells as "mean±std" (single token);
            # single-seed runs format as plain "mean". Strip the std off if
            # present so we work with the per-split mean either way.
            vals = [
                float(x.split("±")[0]) if x != "-" else float("nan")
                for x in toks[-(N_HOP_COLS + 1):]
            ]
        except ValueError:
            continue
        rows[method] = np.array(vals, dtype=np.float64)
    return rows


def collect(logs, block_regex, rename_mm_linear=None, skip_mm_linear=False):
    """For each split log, parse the block and return dict[method] -> (n_splits, n_cols)."""
    per_split: dict[str, list[np.ndarray]] = {}
    for log in logs:
        text = log.read_text()
        rows = parse_block(text, block_regex,
                           rename_mm_linear=rename_mm_linear,
                           skip_mm_linear=skip_mm_linear)
        for method, arr in rows.items():
            per_split.setdefault(method, []).append(arr)
    out = {}
    for method, lst in per_split.items():
        if len(lst) != len(logs):
            print(f"  [warn] {method}: only {len(lst)}/{len(logs)} splits parsed")
        out[method] = np.stack(lst, axis=0)
    return out


def merge(*dicts):
    """Combine several method->arr dicts into one. Later dicts override earlier."""
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = v
    return out


def fmt_mean_std(mean, std, is_best=False, is_second=False):
    s = f"{mean:.3f}".lstrip("0") if mean < 1 else f"{mean:.3f}"
    s_full = f"{mean:.3f}"  # keep leading 0 to match EB style ('0.039', not '.039')
    val = f"{mean:.3f}"
    std_part = f"$\\,\\pm\\,{std:.3f}$"
    if is_best:
        return f"\\textbf{{{val}}}{std_part}"
    if is_second:
        return f"\\underline{{{val}}}{std_part}"
    return f"{val}{std_part}"


def render_block(title, ordered_methods, mean_arr, std_arr, fisherflow_idx, hop_labels):
    """
    Render one half of the table (chained or per-segment).

    mean_arr, std_arr are (n_methods, n_hop_cols + 1) -- last col is the row
    mean across hops. Best (per column) is bold, second-best is underline.
    Delta is computed against ``fisherflow_idx``'s mean column.
    """
    n_methods, n_cols = mean_arr.shape
    assert n_cols == N_HOP_COLS + 1

    # Per-column best/second-best
    rank_best = np.argsort(mean_arr, axis=0)
    best_idx = rank_best[0]
    second_idx = rank_best[1]

    # Delta vs FisherFlow on the mean column
    ff_mean = mean_arr[fisherflow_idx, -1]
    deltas = mean_arr[:, -1] / ff_mean - 1.0

    # Best (most-negative) Delta gets bold
    delta_rank = np.argsort(deltas)
    delta_best = delta_rank[0]
    # second-best Delta is the next non-FisherFlow row (FisherFlow is "---")
    delta_second = next(i for i in delta_rank[1:] if i != fisherflow_idx)

    out_lines = []
    # Section emph header dropped per paper-style decision; per-segment block
    # still needs the hop-label sub-header when present.
    if hop_labels is not None:
        cells = " & ".join(hop_labels)
        out_lines.append(rf"& {cells} & mean & $\Delta$ \\")
    for i, method in enumerate(ordered_methods):
        cells = []
        for j in range(N_HOP_COLS + 1):
            is_best = (j != n_cols - 1 and best_idx[j] == i) or (j == n_cols - 1 and best_idx[j] == i)
            is_second = (j != n_cols - 1 and second_idx[j] == i) or (j == n_cols - 1 and second_idx[j] == i)
            cells.append(fmt_mean_std(mean_arr[i, j], std_arr[i, j], is_best=is_best, is_second=is_second))
        if i == fisherflow_idx:
            delta_str = "---"
        else:
            sign = "-" if deltas[i] < 0 else "+"
            d = abs(deltas[i]) * 100
            inner = rf"${sign}{d:.1f}\%$"
            if i == delta_best:
                delta_str = rf"\textbf{{{inner}}}"
            else:
                delta_str = inner
        line = METHOD_MAP[method][0] + "\n" + "\n".join(f"& {c}" for c in cells)
        line += "\n& " + delta_str + r" \\"
        out_lines.append(line)
    return "\n".join(out_lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Drop sphere-trainer rows (FisherFlow, \\ourmodel + sphere) and "
             "the PCA-50 baseline. Keep only Linear FM as baseline "
             "and \\ourmodel + linear rows. Use Linear FM for "
             "delta. Output to pancreas_spectral_alpha_sweep_linear_only.tex.",
    )
    args = parser.parse_args()

    pl_logs_present = all(p.exists() for p in POWER_LAW_LOGS)
    if pl_logs_present:
        print("Parsing power-law sweep logs (5 splits)...")
        chained_pl = collect(POWER_LAW_LOGS, r"CHAINED EVAL \[MMD\^2\] \(\d+ seeds?\)")
        perseg_pl = collect(POWER_LAW_LOGS, r"PER-SEGMENT EVAL \[MMD\^2\] \(\d+ seeds?\)")
    else:
        print("  [info] Sphere-trainer power-law logs not found; skipping FisherFlow rows.")
        chained_pl = {}
        perseg_pl = {}

    lin_logs_present = all(p.exists() for p in LINEAR_LOGS)
    if lin_logs_present:
        print("Parsing linear (sphere) baseline logs (5 splits)...")
        chained_lin = collect(
            LINEAR_LOGS, r"CHAINED EVAL \[MMD\^2\] \(\d+ seeds?\)",
            rename_mm_linear="MM+Linear@sphere",
        )
        perseg_lin = collect(
            LINEAR_LOGS, r"PER-SEGMENT EVAL \[MMD\^2\] \(\d+ seeds?\)",
            rename_mm_linear="MM+Linear@sphere",
        )
    else:
        print("  [info] Linear sphere baseline logs not found; using MM+Linear from spectral sweep instead.")
        chained_lin = {}
        perseg_lin = {}

    print("Parsing linear (PCA-50) baseline logs (5 splits)...")
    pca50_logs_present = all(p.exists() for p in LINEAR_PCA50_LOGS)
    if pca50_logs_present:
        chained_pca50 = collect(
            LINEAR_PCA50_LOGS, r"CHAINED EVAL \[MMD\^2\] \(\d+ seeds?\)",
            rename_mm_linear="MM+Linear@pca50",
        )
        perseg_pca50 = collect(
            LINEAR_PCA50_LOGS, r"PER-SEGMENT EVAL \[MMD\^2\] \(\d+ seeds?\)",
            rename_mm_linear="MM+Linear@pca50",
        )
    else:
        print("  [info] PCA-50 logs not found; skipping that row.")
        chained_pca50 = {}
        perseg_pca50 = {}

    print("Parsing linear-trainer spectral sweep logs (5 splits)...")
    lin_spec_present = all(p.exists() for p in LINEAR_SPECTRAL_LOGS)
    if lin_spec_present:
        # If the dedicated LINEAR_LOGS sweep wasn't run, pick up MM+Linear from
        # this combined log instead and rename it to the sphere-baseline key
        # so METHOD_MAP resolves it correctly.
        rename = None if lin_logs_present else "MM+Linear@sphere"
        chained_lin_spec = collect(
            LINEAR_SPECTRAL_LOGS, r"CHAINED EVAL \[MMD\^2\] \(\d+ seeds?\)",
            skip_mm_linear=lin_logs_present,
            rename_mm_linear=rename,
        )
        perseg_lin_spec = collect(
            LINEAR_SPECTRAL_LOGS, r"PER-SEGMENT EVAL \[MMD\^2\] \(\d+ seeds?\)",
            skip_mm_linear=lin_logs_present,
            rename_mm_linear=rename,
        )
    else:
        print("  [info] Linear-trainer spectral logs not found; skipping those rows.")
        chained_lin_spec = {}
        perseg_lin_spec = {}

    chained = merge(chained_pl, chained_lin, chained_pca50, chained_lin_spec)
    perseg = merge(perseg_pl, perseg_lin, perseg_pca50, perseg_lin_spec)

    if args.linear_only:
        chained = {k: v for k, v in chained.items() if k in LINEAR_ONLY_KEYS}
        perseg = {k: v for k, v in perseg.items() if k in LINEAR_ONLY_KEYS}
        baseline_key = "MM+Linear@sphere"
        baseline_label = "Linear FM"
    else:
        baseline_key = "MM+SLERP"
        baseline_label = "FisherFlow"

    ordered_methods = sorted(
        [m for m in METHOD_MAP if m in chained and m in perseg],
        key=lambda k: METHOD_MAP[k][1],
    )
    if baseline_key not in ordered_methods:
        raise RuntimeError(f"Baseline {baseline_key} missing from parsed logs")
    fisherflow_idx = ordered_methods.index(baseline_key)

    def stack(stats_dict):
        means = np.stack([stats_dict[m].mean(axis=0) for m in ordered_methods], axis=0)
        stds = np.stack([stats_dict[m].std(axis=0, ddof=1) for m in ordered_methods], axis=0)
        return means, stds

    chained_mean, chained_std = stack(chained)
    perseg_mean, perseg_std = stack(perseg)

    # The mean column from the log is already the row mean across hops,
    # so we just propagate it. Recompute std for the mean column from per-split
    # row means to avoid sqrt-of-mean-of-variances bias:
    for stats_dict, mean_arr, std_arr in [
        (chained, chained_mean, chained_std),
        (perseg, perseg_mean, perseg_std),
    ]:
        for i, m in enumerate(ordered_methods):
            row_means = stats_dict[m][:, :-1].mean(axis=1)
            mean_arr[i, -1] = row_means.mean()
            std_arr[i, -1] = row_means.std(ddof=1)

    perseg_hop_labels = [r"$1\to 2$", r"$2\to 3$", r"$3\to 4$", r"$4\to 5$"]

    chained_block = render_block(
        "Chained evaluation (integrate from $t{=}0$)",
        ordered_methods, chained_mean, chained_std, fisherflow_idx,
        hop_labels=None,
    )
    perseg_block = render_block(
        "Per-segment evaluation (each 1-hop transition)",
        ordered_methods, perseg_mean, perseg_std, fisherflow_idx,
        hop_labels=perseg_hop_labels,
    )

    if args.linear_only:
        out_filename = "pe_table.tex"
        label = "tab:pancreas_spectral_alpha_sweep_linear_only"
    else:
        out_filename = "pancreas_spectral_alpha_sweep.tex"
        label = "tab:pancreas_spectral_alpha_sweep"

    table = (
        r"\begin{table*}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{}" + "\n"
        + r"\label{" + label + r"}" + "\n"
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

    out_path = PROJ / "surf_latex" / "final_report" / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {out_path}")
    print("\n" + "=" * 80)
    print(table)


if __name__ == "__main__":
    main()
