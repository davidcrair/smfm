"""
Generic alpha x blend grid heatmap aggregator.

Parses 5 split logs of the form ``logs/<prefix>_split{42..46}.log``,
builds a 5x5 mean percent-delta surface (alpha rows, blend columns;
blend=1.0 collapses to MM+Linear for every alpha row), and saves a
heatmap PNG/PDF plus a matching LaTeX table.

Usage:
  # GoM 80/10/10 (default)
  .venv/bin/python scripts/plot_alpha_blend_heatmap.py \
      --prefix gom_alpha_blend_grid \
      --tag gom \
      --dataset "Gulf of Mexico"

  # EB OTP-FM holdout
  .venv/bin/python scripts/plot_alpha_blend_heatmap.py \
      --prefix eb_otpfm_alpha_blend_grid \
      --tag eb_otpfm \
      --dataset "Embryoid (OTP-FM holdout)"
"""
from __future__ import annotations

import argparse
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

METRIC_TITLE = r"CHAINED EVAL \[MMD\^2_otpfm\] \(1 seed\)"
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
    # First locate the column index of 'mean' from the header line. The
    # header reads: 'Method <t-cols...> mean [hold_mean train_mean]'.
    body = m.group("body").splitlines()
    header_idx = None
    for i, line in enumerate(body):
        if line.lstrip().startswith("Method"):
            header_idx = i
            break
    if header_idx is None:
        return rows
    header_toks = body[header_idx].split()
    try:
        mean_col = header_toks.index("mean")
    except ValueError:
        return rows
    for line in body[header_idx + 1:]:
        toks = line.split()
        if len(toks) <= mean_col:
            continue
        if toks[0] == "Method" or line.lstrip().startswith("----"):
            continue
        try:
            rows[toks[0]] = float(toks[mean_col])
        except (ValueError, IndexError):
            continue
    return rows


def cell_method_name(alpha: float, blend: float) -> str:
    if blend == 0.0:
        if alpha == 0.0:
            return "MM+Linear+SquaredSpectral@alpha=0"
        return f"MM+Linear+SquaredSpectral@alpha={alpha:g}"
    return f"MM+Linear+SquaredSpectral@alpha={alpha:g},blend={blend:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="log filename prefix; logs are read at "
                         "logs/<prefix>_split{42..46}.log")
    ap.add_argument("--tag", required=True,
                    help="output tag; outputs go to outputs/<tag>_alpha_blend_heatmap.{png,pdf} "
                         "and surf_latex/final_report/<tag>_alpha_blend_grid.tex")
    ap.add_argument("--dataset", default="dataset",
                    help="dataset display name for figure title and LaTeX caption")
    args = ap.parse_args()

    logs = [LOG_DIR / f"{args.prefix}_split{s}.log" for s in SPLITS]
    missing = [p.name for p in logs if not p.exists()]
    if missing:
        raise SystemExit(f"Missing logs: {missing}")

    all_method_vals: dict[str, list[float]] = {}
    for log in logs:
        rows = parse_block(log.read_text(), METRIC_TITLE)
        for method, val in rows.items():
            all_method_vals.setdefault(method, []).append(val)

    if "MM+Linear" not in all_method_vals:
        raise SystemExit("MM+Linear baseline not found in logs")
    base_arr = np.array(all_method_vals["MM+Linear"])
    print(f"MM+Linear baseline ({METRIC_NAME}): "
          f"{base_arr.mean():.4f} ± {base_arr.std(ddof=1):.4f}  (n={len(base_arr)} splits)")

    n_alpha = len(ALPHAS)
    n_blend = len(BLENDS)
    delta_pct = np.full((n_alpha, n_blend), np.nan)
    p_less_mat = np.full((n_alpha, n_blend), np.nan)
    wins_mat = np.zeros((n_alpha, n_blend), dtype=int)

    for ai, alpha in enumerate(ALPHAS):
        for bi, blend in enumerate(BLENDS):
            if blend == 1.0:
                delta_pct[ai, bi] = 0.0
                p_less_mat[ai, bi] = 1.0
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

    # ---- Plain-text summary ----
    print(f"\n=== {args.dataset}: Δ% vs MM+Linear (CHAINED {METRIC_NAME}), "
          f"n={len(base_arr)} paired splits ===\n")
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
    print("(★ = paired Wilcoxon p<0.05 better than MM+Linear)")

    # ---- Heatmap ----
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    finite = delta_pct[np.isfinite(delta_pct)]
    vmax = max(abs(finite.min()), abs(finite.max())) if finite.size else 1.0
    im = ax.imshow(
        delta_pct, cmap="coolwarm", aspect="auto",
        vmin=-vmax, vmax=vmax,
        origin="lower",
    )

    for ai in range(n_alpha):
        for bi in range(n_blend):
            d = delta_pct[ai, bi]
            p = p_less_mat[ai, bi]
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
        f"{args.dataset}: " +
        r"$\Delta$\% chained MMD$^2_{\mathrm{otpfm}}$ vs MM+Linear" +
        f"\n(★ = paired Wilcoxon p<0.05; · = p<0.10; n={len(base_arr)} splits)"
    )
    fig.colorbar(im, ax=ax, label=r"$\Delta$\% vs MM+Linear (negative = better)")
    fig.tight_layout()

    out_dir = PROJ / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.tag}_alpha_blend_heatmap.png"
    pdf = out_dir / f"{args.tag}_alpha_blend_heatmap.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")

    # ---- LaTeX ----
    tex_path = PROJ / "surf_latex" / "final_report" / f"{args.tag}_alpha_blend_grid.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{{args.dataset}: $\alpha \times \beta$ grid of "
        r"\ourmodel + linear with the Spectral+Euclidean blended cost "
        r"$C = (1-\beta)\,C_{\mathrm{spec}}^{(\alpha)} + \beta\,C_{\mathrm{eucl}}$. "
        r"Cells show mean $\Delta$\% chained MMD$^2_{\mathrm{otpfm}}$ vs "
        rf"MM+Linear (Euclidean OT) over {len(base_arr)} data splits; "
        r"$\star$ denotes paired Wilcoxon $p<0.05$.}"
    )
    lines.append(rf"\label{{tab:{args.tag}_alpha_blend_grid}}")
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
