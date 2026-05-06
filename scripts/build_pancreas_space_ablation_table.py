r"""
Cross-space (preprocessing) ablation table for pancreas: trains the same
3 methods (Linear FM, SMFM alpha=0, SMFM alpha=1) in two preprocessing spaces
-- 50-dim PCA latent vs sphere-ambient -- and evaluates both in compositional
gene space. Demonstrates that ambient-space training only helps when paired
with a geometry-aware OT cost.

Reads chained-MMD^2 from:
  - logs/pancreas_power_law_linear_pca50_b512_split{42..46}.log    (PCA-50)
  - logs/pancreas_power_law_linear_b512_split{42..46}.log          (sphere)

Aggregates the chained-mean across 4 stages per split, then mean +/- std over
5 splits, marks per-column best (\textbf), and writes
surf_latex/final_report/pe_space_ablation.tex.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]

PCA_LOGS = [LOG_DIR / f"pancreas_power_law_linear_pca50_b512_split{s}.log" for s in SPLITS]
SPHERE_LOGS = [LOG_DIR / f"pancreas_power_law_linear_b512_split{s}.log" for s in SPLITS]

# Order: baseline first, then random-pairing control, then spectral methods.
METHOD_KEYS = [
    ("MM+Linear",                          "Linear FM"),
    ("MM+Linear+Random",                   "Random OT"),
    ("MM+Linear+SquaredSpectral@alpha=0",  r"\ourmodel $\alpha=0$"),
    ("MM+Linear+SquaredSpectral@alpha=1",  r"\ourmodel $\alpha=1$"),
]
N_HOP_COLS = 4
BLOCK_REGEX = r"CHAINED EVAL \[MMD\^2\] \(1 seed\)"


def parse_block(text: str) -> dict[str, np.ndarray]:
    pat = re.compile(
        r"={80}\s*\n\s*" + BLOCK_REGEX + r"\s*\n={80}\s*\n"
        r"(?P<body>.*?)(?=\n={80}|\n\s*Metric conventions:|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        return {}
    rows: dict[str, np.ndarray] = {}
    target_keys = {k for k, _ in METHOD_KEYS}
    for line in m.group("body").splitlines():
        toks = line.split()
        if len(toks) < N_HOP_COLS + 2:
            continue
        if line.lstrip().startswith(("Method", "----")):
            continue
        method = toks[0]
        if method not in target_keys:
            continue
        try:
            vals = [float(x) if x != "-" else np.nan for x in toks[-(N_HOP_COLS + 1):]]
        except ValueError:
            continue
        rows[method] = np.array(vals, dtype=np.float64)
    return rows


def collect(logs: list[Path]) -> dict[str, np.ndarray]:
    """Return dict[method] -> (n_splits,) array of chained-mean MMD^2."""
    out: dict[str, list[float]] = {}
    for log in logs:
        if not log.exists():
            print(f"  [warn] missing {log.name}")
            continue
        rows = parse_block(log.read_text())
        for method, arr in rows.items():
            chained_mean = float(np.nanmean(arr[:-1]))
            out.setdefault(method, []).append(chained_mean)
    return {m: np.array(v, dtype=np.float64) for m, v in out.items()}


def fmt_cell(mean: float, std: float, is_best: bool) -> str:
    if not np.isfinite(mean):
        return "-"
    base = f"{mean:.3f}"
    err = rf"$\,\pm\,{std:.3f}$"
    if is_best:
        return rf"\textbf{{{base}}}{err}"
    return f"{base}{err}"


def main() -> None:
    print("Parsing PCA-50 logs (5 splits)...")
    pca = collect(PCA_LOGS)
    print("Parsing sphere logs (5 splits)...")
    sphere = collect(SPHERE_LOGS)

    if not pca:
        raise RuntimeError(
            "No PCA-50 logs parsed. Run "
            "scripts/run_power_law_kfold_pancreas_linear_pca50_b512.sh first."
        )
    if not sphere:
        raise RuntimeError(
            "No sphere logs parsed. Expected "
            "logs/pancreas_power_law_linear_b512_split{42..46}.log to exist."
        )

    means_pca = np.array([np.nanmean(pca.get(k, [np.nan])) for k, _ in METHOD_KEYS])
    stds_pca = np.array([np.nanstd(pca.get(k, [np.nan]), ddof=1) for k, _ in METHOD_KEYS])
    means_sph = np.array([np.nanmean(sphere.get(k, [np.nan])) for k, _ in METHOD_KEYS])
    stds_sph = np.array([np.nanstd(sphere.get(k, [np.nan]), ddof=1) for k, _ in METHOD_KEYS])

    best_pca = int(np.nanargmin(means_pca))
    best_sph = int(np.nanargmin(means_sph))

    rows = []
    for i, (_, label) in enumerate(METHOD_KEYS):
        line = (
            f"{label}\n"
            f"  & {fmt_cell(means_pca[i], stds_pca[i], i == best_pca)}\n"
            f"  & {fmt_cell(means_sph[i], stds_sph[i], i == best_sph)} \\\\"
        )
        rows.append(line)

    table = (
        r"\begin{table}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{}" + "\n"
        + r"\label{tab:pancreas_space_ablation}" + "\n"
        + r"\renewcommand{\arraystretch}{1.15}" + "\n"
        + r"\setlength{\tabcolsep}{6pt}" + "\n"
        + r"\begin{tabular}{lcc}" + "\n"
        + r"\toprule" + "\n"
        + r"& \multicolumn{2}{c}{Chained MMD$^2$ in gene space} \\" + "\n"
        + r"\cmidrule(lr){2-3}" + "\n"
        + r"Method & PCA-50 $\to$ genes & Sphere-ambient \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table}" + "\n"
    )

    out_path = PROJ / "surf_latex" / "final_report" / "pe_space_ablation.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"\nWrote {out_path}\n")
    print("=" * 70)
    print(table)
    print("=" * 70)

    # Plain-text echo
    print("\nChained MMD^2 (mean across 4 stages, then mean +/- std over 5 splits):")
    print(f"  {'Method':<32} {'PCA-50 -> genes':>20}   {'Sphere-ambient':>20}")
    for i, (_, label) in enumerate(METHOD_KEYS):
        clean_label = label.replace(r"\ourmodel", "SMFM").replace("$", "")
        print(
            f"  {clean_label:<32} "
            f"{means_pca[i]:.4f}+/-{stds_pca[i]:.4f}    "
            f"{means_sph[i]:.4f}+/-{stds_sph[i]:.4f}"
        )


if __name__ == "__main__":
    main()
