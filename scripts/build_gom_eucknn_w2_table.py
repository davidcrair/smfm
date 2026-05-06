"""
Aggregate GoM OTP-FM 9-stage Euclidean-kNN spectral sweep (multi-seed)
into the per-stage W_2 LaTeX table at tables/gom_ho_w_2.tex.

Reads logs/gom_otpfm9_alpha_sweep_eucknn_split{42..46}.log, picks the
winning alpha row (default alpha=0.5, beta=0), and emits a per-stage
W_2 table with paired Wilcoxon p-values vs MM+Linear.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import stats


PROJ = Path(__file__).resolve().parent.parent
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]
LOG_TMPL = "gom_otpfm9_alpha_sweep_eucknn_split{s}.log"

WINNER = "MM+Linear+SquaredSpectral@alpha=0.5"
WINNER_LABEL = r"\ourmodel{} ($\alpha{=}0.5$, $\beta{=}0$, EucKNN)"
BASELINE = "MM+Linear"

STAGE_TIMES = [0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88, 1.00]
HELD = {1, 3, 5, 7}
TRAIN = {0, 2, 4, 6, 8}


def parse_chained_w2(text: str):
    pat = re.compile(
        r"={80}\s*\nCHAINED EVAL \[W_2\] \(\d+ seeds?\)\s*\n={80}\s*\n(?P<body>.*?)(?=\n={80}|\nMetric conventions|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        return {}
    rows = {}
    for line in m.group("body").splitlines():
        toks = line.split()
        if len(toks) < 12:
            continue
        method = toks[0]
        cells = toks[1:]
        try:
            vals = [float(t.split("±")[0]) for t in cells]
        except ValueError:
            continue
        # Layout: 8 stage cols (t1..t8) + mean + hold_mean + train_mean
        if len(vals) >= 11:
            rows[method] = vals[:11]
    return rows


def main():
    logs = [LOG_DIR / LOG_TMPL.format(s=s) for s in SPLITS]
    missing = [p.name for p in logs if not p.exists()]
    if missing:
        raise SystemExit(f"Missing logs: {missing}")

    base_per_stage = [[] for _ in range(8)]
    win_per_stage = [[] for _ in range(8)]
    for log in logs:
        rows = parse_chained_w2(log.read_text(errors="replace"))
        if BASELINE not in rows or WINNER not in rows:
            print(f"  WARN missing methods in {log.name}")
            continue
        for st in range(8):
            base_per_stage[st].append(rows[BASELINE][st])
            win_per_stage[st].append(rows[WINNER][st])

    print(f"\n=== GoM OTP-FM 9-stage Euclidean-kNN ({WINNER}) per-stage W_2 ===\n")
    print(f"  Stage | t    | role  | {BASELINE:<22} | {'spectral':<22} | Δ%       | p_paired")
    print(f"  ------+------+-------+------------------------+------------------------+----------+---------")

    held_b, held_w = [], []
    train_b, train_w = [], []
    for stage_idx in range(8):
        actual = stage_idx + 1
        role = "HELD" if actual in HELD else "TRAIN"
        b = np.array(base_per_stage[stage_idx])
        w = np.array(win_per_stage[stage_idx])
        diff = w - b
        d_pct = (w.mean() / b.mean() - 1) * 100
        try:
            _, p = stats.wilcoxon(diff, alternative="less")
        except ValueError:
            p = float("nan")
        print(f"  s={actual}    | {STAGE_TIMES[actual]:.2f} | {role:<5} | "
              f"{b.mean():>9.4f} ± {b.std(ddof=1):>5.4f}    | "
              f"{w.mean():>9.4f} ± {w.std(ddof=1):>5.4f}    | "
              f"{d_pct:>+6.1f}%  | {p:.4f}")
        if role == "HELD":
            held_b.append(b); held_w.append(w)
        else:
            train_b.append(b); train_w.append(w)

    def role_avg(arrs):
        return np.mean(np.vstack(arrs), axis=0) if arrs else None

    hb, hw = role_avg(held_b), role_avg(held_w)
    tb, tw = role_avg(train_b), role_avg(train_w)
    print()
    if hb is not None:
        d = hw - hb
        try: _, p = stats.wilcoxon(d, alternative="less")
        except ValueError: p = float("nan")
        print(f"  AVG held-out  | {hb.mean():>9.4f} ± {hb.std(ddof=1):>5.4f}    | "
              f"{hw.mean():>9.4f} ± {hw.std(ddof=1):>5.4f}    | "
              f"{(hw.mean()/hb.mean()-1)*100:>+6.1f}%  | {p:.4f}")
    if tb is not None:
        d = tw - tb
        try: _, p = stats.wilcoxon(d, alternative="less")
        except ValueError: p = float("nan")
        print(f"  AVG train     | {tb.mean():>9.4f} ± {tb.std(ddof=1):>5.4f}    | "
              f"{tw.mean():>9.4f} ± {tw.std(ddof=1):>5.4f}    | "
              f"{(tw.mean()/tb.mean()-1)*100:>+6.1f}%  | {p:.4f}")

    # LaTeX table
    tex_path = PROJ / "surf_latex" / "final_report" / "tables" / "gom_ho_w_2.tex"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Per-stage chained $W_2$ on the Gulf of Mexico (GoM) 9-stage protocol with held-out marginals $\{{1,3,5,7\}}$. SMFM uses $\alpha=0.5$, $\beta=0$ with the Euclidean-kNN graph; Linear FM is the multi-marginal Euclidean baseline. Cells are mean$\,\pm\,$std over 5 splits $\times$ 5 init seeds; $\star$ marks $p<0.05$ on a paired one-sided Wilcoxon vs the baseline.}}",
        r"\label{tab:gom_otpfm9_w2}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Stage & $t$ & role & MM+Linear $W_2$ & SMFM $W_2$ & $\Delta\%$ & paired $p$ \\",
        r"\midrule",
    ]
    for stage_idx in range(8):
        actual = stage_idx + 1
        role = "held" if actual in HELD else "train"
        b = np.array(base_per_stage[stage_idx])
        w = np.array(win_per_stage[stage_idx])
        diff = w - b
        d_pct = (w.mean() / b.mean() - 1) * 100
        try: _, p = stats.wilcoxon(diff, alternative="less")
        except ValueError: p = float("nan")
        cell_w = f"${w.mean():.3f}\\pm{w.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            f"$s_{{{actual}}}$ & ${STAGE_TIMES[actual]:.2f}$ & {role} & "
            f"${b.mean():.3f}\\pm{b.std(ddof=1):.3f}$ & {cell_w} & "
            f"${d_pct:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    lines.append(r"\midrule")
    if hb is not None:
        d = hw - hb
        try: _, p = stats.wilcoxon(d, alternative="less")
        except ValueError: p = float("nan")
        cell_w = f"${hw.mean():.3f}\\pm{hw.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            r"\multicolumn{3}{l}{\emph{avg.\ over held-out marginals (1,3,5,7)}} & "
            f"${hb.mean():.3f}\\pm{hb.std(ddof=1):.3f}$ & {cell_w} & "
            f"${(hw.mean()/hb.mean()-1)*100:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    if tb is not None:
        d = tw - tb
        try: _, p = stats.wilcoxon(d, alternative="less")
        except ValueError: p = float("nan")
        cell_w = f"${tw.mean():.3f}\\pm{tw.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            r"\multicolumn{3}{l}{\emph{avg.\ over train marginals (0,2,4,6,8)}} & "
            f"${tb.mean():.3f}\\pm{tb.std(ddof=1):.3f}$ & {cell_w} & "
            f"${(tw.mean()/tb.mean()-1)*100:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {tex_path}")


if __name__ == "__main__":
    main()
