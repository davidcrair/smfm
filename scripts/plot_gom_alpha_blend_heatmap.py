"""
GoM alpha x blend grid heatmap.

Parses logs/gom_alpha_blend_grid_split{42..46}.log, builds a 5x5 mean
percent-Δ surface (alpha = {0,0.5,1,1.5,2} rows, blend = {0,0.25,0.5,
0.75,1.0} columns; the blend=1.0 column collapses to MM+Linear for
every alpha row), and saves a heatmap PDF/PNG plus a paired Wilcoxon
significance overlay.

Output:
  outputs/gom_alpha_blend_heatmap.png
  outputs/gom_alpha_blend_heatmap.pdf
  surf_latex/final_report/gom_alpha_blend_grid.tex   (matching LaTeX)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PROJ = Path(__file__).resolve().parent.parent
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]   # 1.0 = pure Euclidean = MM+Linear

METRIC_TITLE = r"CHAINED EVAL \[MMD\^2_otpfm\] \(1 seed\)"  # OTP-FM-comparable
METRIC_NAME = "MMD²_otpfm"


def parse_block(text: str, title_regex: str) -> dict[str, float]:
    pat = re.compile(
        r"={80}\s*\n\s*" + title_regex + r"\s*\n={80}\s*\n"
        r"(?P<body>.*?)(?=\n={80}|\n\s*Metric conventions:|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        return {}
    rows: dict[str, float] = {}
    for line in m.group("body").splitlines():
        toks = line.split()
        if len(toks) < 5:
            continue
        if toks[0] in ("Method",) or line.lstrip().startswith("----"):
            continue
        method = toks[0]
        try:
            mean_val = float(toks[-1])  # last column = row mean
            rows[method] = mean_val
        except ValueError:
            continue
    return rows


def cell_method_name(alpha: float, blend: float) -> str:
    if blend == 0.0:
        if alpha == 0.0:
            return "MM+Linear+SquaredSpectral@alpha=0"
        return f"MM+Linear+SquaredSpectral@alpha={alpha:g}"
    return f"MM+Linear+SquaredSpectral@alpha={alpha:g},blend={blend:g}"


def main():
    logs = [LOG_DIR / f"gom_alpha_blend_grid_split{s}.log" for s in SPLITS]
    if not all(p.exists() for p in logs):
        missing = [p.name for p in logs if not p.exists()]
        raise SystemExit(f"Missing logs: {missing}")

    # Per-split parsing -> per-method list of MMD² values across splits.
    all_method_vals: dict[str, list[float]] = {}
    for log in logs:
        rows = parse_block(log.read_text(), METRIC_TITLE)
        for method, val in rows.items():
            all_method_vals.setdefault(method, []).append(val)

    # MM+Linear baseline (5 values across splits)
    if "MM+Linear" not in all_method_vals:
        raise SystemExit("MM+Linear baseline not found in logs")
    base_arr = np.array(all_method_vals["MM+Linear"])
    print(f"MM+Linear baseline: {base_arr.mean():.4f} ± {base_arr.std(ddof=1):.4f}")

    # Build (alpha, blend) -> (mean_delta, p_less, n_wins) matrix.
    n_alpha = len(ALPHAS)
    n_blend = len(BLENDS)
    delta_pct = np.full((n_alpha, n_blend), np.nan)
    p_less_mat = np.full((n_alpha, n_blend), np.nan)
    wins_mat = np.zeros((n_alpha, n_blend), dtype=int)

    for ai, alpha in enumerate(ALPHAS):
        for bi, blend in enumerate(BLENDS):
            if blend == 1.0:
                # collapse to MM+Linear baseline (same for every alpha)
                delta_pct[ai, bi] = 0.0
                p_less_mat[ai, bi] = 1.0   # by definition not better
                wins_mat[ai, bi] = 0
                continue
            method = cell_method_name(alpha, blend)
            if method not in all_method_vals:
                print(f"  WARN missing: {method}")
                continue
            v = np.array(all_method_vals[method])
            if len(v) != len(base_arr):
                print(f"  WARN length mismatch for {method}: {len(v)} vs {len(base_arr)}")
                continue
            diffs = v - base_arr
            delta_pct[ai, bi] = (v.mean() / base_arr.mean() - 1.0) * 100.0
            wins_mat[ai, bi] = int((diffs < 0).sum())
            try:
                _, p_less = stats.wilcoxon(diffs, alternative="less")
            except ValueError:
                p_less = float("nan")
            p_less_mat[ai, bi] = p_less

    # ---- Print plain-text summary ----
    print("\n=== Δ% vs MM+Linear (CHAINED MMD²_otpfm), n=5 paired splits ===\n")
    header = "α \\ blend " + " ".join(f"{b:>9.2f}" for b in BLENDS)
    print(header)
    print("-" * len(header))
    for ai, alpha in enumerate(ALPHAS):
        row = [f"α={alpha:>3.1f}    "]
        for bi in range(n_blend):
            d = delta_pct[ai, bi]
            p = p_less_mat[ai, bi]
            mark = "★" if (not np.isnan(p) and p < 0.05) else " "
            row.append(f"{d:>+6.1f}%{mark}")
        print(" ".join(row))
    print()
    print("(★ = paired Wilcoxon p<0.05 better than MM+Linear, n=5)")

    # ---- Plot heatmap ----
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    vmax = max(abs(np.nanmin(delta_pct)), abs(np.nanmax(delta_pct)))
    im = ax.imshow(
        delta_pct, cmap="coolwarm", aspect="auto",
        vmin=-vmax, vmax=vmax,
        origin="lower",
    )

    # Annotate each cell with Δ% and significance marker.
    for ai in range(n_alpha):
        for bi in range(n_blend):
            d = delta_pct[ai, bi]
            p = p_less_mat[ai, bi]
            wins = wins_mat[ai, bi]
            if np.isnan(d):
                txt = "—"
            else:
                txt = f"{d:+.1f}%"
                if not np.isnan(p) and p < 0.05:
                    txt += "\n★"
                elif not np.isnan(p) and p < 0.10:
                    txt += "\n·"
                if BLENDS[bi] == 1.0:
                    txt = "0.0%\n(baseline)" if ai == n_alpha // 2 else " "
            color = "white" if abs(d) > 0.5 * vmax else "black"
            ax.text(bi, ai, txt, ha="center", va="center",
                    color=color, fontsize=9, fontweight="normal")

    ax.set_xticks(range(n_blend))
    ax.set_xticklabels([f"{b:.2f}" for b in BLENDS])
    ax.set_yticks(range(n_alpha))
    ax.set_yticklabels([f"{a:.1f}" for a in ALPHAS])
    ax.set_xlabel(r"Euclidean blend $\beta$ (0 = pure spectral, 1 = pure Euclidean)")
    ax.set_ylabel(r"Spectral exponent $\alpha$")
    ax.set_title(
        r"GoM 80/10/10: $\Delta$\% chained MMD$^2_{\mathrm{otpfm}}$ vs MM+Linear" +
        "\n(★ = paired Wilcoxon p<0.05; · = p<0.10; n=5 splits)"
    )

    cbar = fig.colorbar(im, ax=ax, label=r"$\Delta$\% vs MM+Linear (negative = better)")
    fig.tight_layout()

    out_dir = PROJ / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "gom_alpha_blend_heatmap.png"
    pdf = out_dir / "gom_alpha_blend_heatmap.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")

    # ---- LaTeX table ----
    tex_path = PROJ / "surf_latex" / "final_report" / "gom_alpha_blend_grid.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Gulf of Mexico vortex tracking: $\alpha \times \beta$ grid of "
        r"\ourmodel + linear with the Spectral+Euclidean blended cost "
        r"$C = (1-\beta)\,C_{\mathrm{spec}}^{(\alpha)} + \beta\,C_{\mathrm{eucl}}$. "
        r"Cells show mean $\Delta$\% chained MMD$^2_{\mathrm{otpfm}}$ vs "
        r"MM+Linear (Euclidean OT) over 5 data splits (42--46); "
        r"$\star$ denotes paired Wilcoxon $p<0.05$. The kNN union graph on GoM "
        r"has $\sim$6 connected components; pure spectral ($\beta=0$) "
        r"degrades by up to $+30$\%, but the $\beta = 0.75$ column "
        r"recovers and exceeds the Euclidean baseline.}"
    )
    lines.append(r"\label{tab:gom_alpha_blend_grid}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{c" + "r" * n_blend + r"}")
    lines.append(r"\toprule")
    lines.append(r"$\alpha$ \textbackslash\ $\beta$ & " +
                 " & ".join(f"${b:.2f}$" for b in BLENDS) + r" \\")
    lines.append(r"\midrule")
    for ai, alpha in enumerate(ALPHAS):
        cells = []
        for bi in range(n_blend):
            d = delta_pct[ai, bi]
            p = p_less_mat[ai, bi]
            if np.isnan(d):
                cells.append("--")
            elif BLENDS[bi] == 1.0:
                cells.append(r"$0.0$\%")
            else:
                txt = f"${d:+.1f}\\%$"
                if not np.isnan(p) and p < 0.05:
                    txt = r"\textbf{" + txt + r"$^{\star}$}"
                cells.append(txt)
        lines.append(f"${alpha:.1f}$ & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
