"""
Aggregate per-stage W_2 (and MMD^2_otpfm) for the GoM OTP-FM 9-stage
best-model rerun across 5 splits. Produces:

  - per-held-out-stage W_2 mean +/- std for MM+Linear and the winning
    spectral cell
  - average W_2 across the 5 train marginals (sanity check that the
    model fits the observed stages well)
  - paired Wilcoxon vs MM+Linear at each held-out stage
  - LaTeX table at surf_latex/final_report/gom_otpfm9_w2_table.tex

Usage: .venv/bin/python scripts/build_gom_otpfm9_w2_table.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import stats


PROJ = Path(__file__).resolve().parent.parent
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]
LOG_PREFIX = "gom_otpfm9_best_w2"

WINNER = "MM+Linear+SquaredSpectral@alpha=0.5,blend=0.25"
BASELINE = "MM+Linear"

# 9-stage protocol with stages 1, 3, 5, 7 held out, 0/2/4/6/8 trained.
STAGE_TIMES = [0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88, 1.00]
HELD = [1, 3, 5, 7]
TRAIN = [0, 2, 4, 6, 8]

# Chained eval predicts forward from stage 0 to each later stage; the
# resulting per-row layout has 8 stage columns (t1..t8), then mean,
# hold_mean, train_mean.


def parse_chained_block(text: str, metric_name: str):
    title_regex = rf"CHAINED EVAL \[{re.escape(metric_name)}\] \(1 seed\)"
    pat = re.compile(
        r"={80}\s*\n\s*" + title_regex + r"\s*\n={80}\s*\n"
        r"(?P<body>.*?)(?=\n={80}|\n\s*Metric conventions:|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        return None, {}
    body = m.group("body").splitlines()
    hdr_idx = next((i for i, l in enumerate(body) if l.lstrip().startswith("Method")), None)
    if hdr_idx is None:
        return None, {}
    header = body[hdr_idx].split()
    rows = {}
    for line in body[hdr_idx + 1:]:
        toks = line.split()
        if len(toks) <= 2:
            continue
        if toks[0] == "Method" or line.lstrip().startswith("----"):
            continue
        nums = []
        for t in toks[1:]:
            try:
                nums.append(float(t))
            except ValueError:
                pass
        rows[toks[0]] = nums
    return header, rows


def main():
    logs = [LOG_DIR / f"{LOG_PREFIX}_split{s}.log" for s in SPLITS]
    missing = [p.name for p in logs if not p.exists()]
    if missing:
        raise SystemExit(f"Missing logs: {missing}")

    # Per-stage W_2 across splits, per method.
    # Chained-eval row has 8 t-cols (predicts to stage_idx 1..8) + 3 summary cols.
    # nums[stage_idx-1] is the W_2 at that stage.
    w2_per_method_stage = {BASELINE: [[None] * 5 for _ in range(8)],
                           WINNER:   [[None] * 5 for _ in range(8)]}
    mmd_per_method_stage = {BASELINE: [[None] * 5 for _ in range(8)],
                            WINNER:   [[None] * 5 for _ in range(8)]}

    for split_idx, log in enumerate(logs):
        text = log.read_text(errors="replace")
        for metric_name, target in [("W_2", w2_per_method_stage),
                                    ("MMD^2_otpfm", mmd_per_method_stage)]:
            _, rows = parse_chained_block(text, metric_name)
            for method in (BASELINE, WINNER):
                if method not in rows:
                    print(f"  WARN missing {method!r} for {metric_name} in split {SPLITS[split_idx]}")
                    continue
                vals = rows[method]
                for stage_idx in range(8):
                    if stage_idx < len(vals):
                        target[method][stage_idx][split_idx] = vals[stage_idx]

    # ---- Build summary statistics ----
    print(f"\n=== GoM OTP-FM 9-stage (n={len(SPLITS)} splits): per-stage W_2 ===\n")
    print(f"  Stage | t   | role  | {BASELINE:<25} | {'spectral (α=0.5, β=0.25)':<28} | Δ%       | p_paired")
    print(f"  ------+-----+-------+---------------------------+------------------------------+----------+---------")

    held_w2_base = []
    held_w2_win = []
    train_w2_base = []
    train_w2_win = []

    for stage_idx in range(8):  # stages 1..8 (predicted from stage 0)
        actual_stage = stage_idx + 1
        role = "HELD" if actual_stage in HELD else "TRAIN"
        b = np.array(w2_per_method_stage[BASELINE][stage_idx], dtype=float)
        w = np.array(w2_per_method_stage[WINNER][stage_idx], dtype=float)
        if np.any(np.isnan(b)) or np.any(np.isnan(w)):
            continue
        diff = w - b
        delta_pct = (w.mean() / b.mean() - 1.0) * 100.0
        try:
            _, p = stats.wilcoxon(diff, alternative="less")
        except ValueError:
            p = float("nan")
        print(f"  s={actual_stage}   | {STAGE_TIMES[actual_stage]:.2f} | {role:<5} | "
              f"{b.mean():>10.4f} ± {b.std(ddof=1):>5.4f}      | "
              f"{w.mean():>10.4f} ± {w.std(ddof=1):>5.4f}        | "
              f"{delta_pct:>+6.1f}%  | {p:.4f}")
        if role == "HELD":
            held_w2_base.append(b)
            held_w2_win.append(w)
        else:
            train_w2_base.append(b)
            train_w2_win.append(w)

    # Average across roles (per-split, then aggregate)
    def role_avg(arrays):
        # arrays: list over stages, each (n_splits,)
        if not arrays:
            return None
        per_split = np.mean(np.vstack(arrays), axis=0)  # average across stages within split
        return per_split

    held_b = role_avg(held_w2_base)
    held_w = role_avg(held_w2_win)
    train_b = role_avg(train_w2_base)
    train_w = role_avg(train_w2_win)

    print()
    if held_b is not None:
        d = held_w - held_b
        try:
            _, p = stats.wilcoxon(d, alternative="less")
        except ValueError:
            p = float("nan")
        print(f"  AVG over held-out  | {held_b.mean():>10.4f} ± {held_b.std(ddof=1):>5.4f}      | "
              f"{held_w.mean():>10.4f} ± {held_w.std(ddof=1):>5.4f}        | "
              f"{(held_w.mean()/held_b.mean()-1)*100:>+6.1f}%  | {p:.4f}")
    if train_b is not None:
        d = train_w - train_b
        try:
            _, p = stats.wilcoxon(d, alternative="less")
        except ValueError:
            p = float("nan")
        print(f"  AVG over train     | {train_b.mean():>10.4f} ± {train_b.std(ddof=1):>5.4f}      | "
              f"{train_w.mean():>10.4f} ± {train_w.std(ddof=1):>5.4f}        | "
              f"{(train_w.mean()/train_b.mean()-1)*100:>+6.1f}%  | {p:.4f}")

    # ---- LaTeX table ----
    tex_path = PROJ / "surf_latex" / "final_report" / "gom_otpfm9_w2_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{}")
    lines.append(r"\label{tab:gom_otpfm9_w2}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Stage & $t$ & role & MM+Linear (Eucl.) $W_2$ & Spectral$+$blend $W_2$ & $\Delta\%$ & paired $p$ \\")
    lines.append(r"\midrule")
    for stage_idx in range(8):
        actual_stage = stage_idx + 1
        role = "held" if actual_stage in HELD else "train"
        b = np.array(w2_per_method_stage[BASELINE][stage_idx], dtype=float)
        w = np.array(w2_per_method_stage[WINNER][stage_idx], dtype=float)
        if np.any(np.isnan(b)) or np.any(np.isnan(w)):
            continue
        diff = w - b
        delta_pct = (w.mean() / b.mean() - 1.0) * 100.0
        try:
            _, p = stats.wilcoxon(diff, alternative="less")
        except ValueError:
            p = float("nan")
        cell_w = f"${w.mean():.3f}\\pm{w.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            f"$s_{{{actual_stage}}}$ & ${STAGE_TIMES[actual_stage]:.2f}$ & {role} & "
            f"${b.mean():.3f}\\pm{b.std(ddof=1):.3f}$ & {cell_w} & "
            f"${delta_pct:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    lines.append(r"\midrule")
    if held_b is not None:
        d = held_w - held_b
        try:
            _, p = stats.wilcoxon(d, alternative="less")
        except ValueError:
            p = float("nan")
        cell_w = f"${held_w.mean():.3f}\\pm{held_w.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            r"\multicolumn{3}{l}{\emph{avg.\ over held-out marginals (1,3,5,7)}} & " +
            f"${held_b.mean():.3f}\\pm{held_b.std(ddof=1):.3f}$ & {cell_w} & " +
            f"${(held_w.mean()/held_b.mean()-1)*100:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    if train_b is not None:
        d = train_w - train_b
        try:
            _, p = stats.wilcoxon(d, alternative="less")
        except ValueError:
            p = float("nan")
        cell_w = f"${train_w.mean():.3f}\\pm{train_w.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            r"\multicolumn{3}{l}{\emph{avg.\ over train marginals (0,2,4,6,8)}} & " +
            f"${train_b.mean():.3f}\\pm{train_b.std(ddof=1):.3f}$ & {cell_w} & " +
            f"${(train_w.mean()/train_b.mean()-1)*100:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {tex_path}")


if __name__ == "__main__":
    main()
