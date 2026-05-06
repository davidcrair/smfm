"""
Generic per-stage W_2 aggregator for any 'best-model' rerun that has
two methods (Euclidean baseline + 1 spectral winner) trained on 5
splits with eval.compute_w2=true.

Outputs:
  - console summary with per-stage W_2, paired Wilcoxon p-values
  - LaTeX table at surf_latex/final_report/<tag>_w2_table.tex

Usage:
  .venv/bin/python scripts/build_best_w2_table.py \
      --prefix gom_otpfm9_best_w2 --tag gom_otpfm9 \
      --winner "MM+Linear+SquaredSpectral@alpha=0.5,blend=0.25" \
      --held-stages 1,3,5,7

  .venv/bin/python scripts/build_best_w2_table.py \
      --prefix eb_8010_best_w2 --tag eb_8010 \
      --winner "MM+Linear+SquaredSpectral@alpha=0.5" \
      --held-stages ""    # 80/10/10 has no held stages; all are train
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats


PROJ = Path(__file__).resolve().parent.parent
LOG_DIR = PROJ / "logs"
SPLITS = [42, 43, 44, 45, 46]
BASELINE = "MM+Linear"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="log filename prefix; reads logs/<prefix>_split{42..46}.log")
    ap.add_argument("--tag", required=True, help="output tag for surf_latex/final_report/<tag>_w2_table.tex")
    ap.add_argument("--winner", required=True, help="full method name of the spectral winner")
    ap.add_argument("--held-stages", default="", help="comma-separated 1-indexed predicted-stage indices that are held out (empty for 80/10/10)")
    args = ap.parse_args()

    held = set()
    if args.held_stages.strip():
        held = set(int(x) for x in args.held_stages.split(","))

    logs = [LOG_DIR / f"{args.prefix}_split{s}.log" for s in SPLITS]
    missing = [p.name for p in logs if not p.exists()]
    if missing:
        raise SystemExit(f"Missing logs: {missing}")

    # Discover number of stages from first log's header
    sample = logs[0].read_text(errors="replace")
    hdr, _ = parse_chained_block(sample, "W_2")
    if hdr is None:
        raise SystemExit(f"No CHAINED EVAL [W_2] block found in {logs[0]}; "
                         "did you run with eval.compute_w2=true?")
    # Count time columns: tokens after 'Method' before 'mean'
    t_cols = []
    for h in hdr[1:]:
        if h == "mean":
            break
        t_cols.append(h)
    n_stages_pred = len(t_cols)

    # gather W_2 values
    w2 = {BASELINE: [[None]*len(SPLITS) for _ in range(n_stages_pred)],
          args.winner: [[None]*len(SPLITS) for _ in range(n_stages_pred)]}
    for si, log in enumerate(logs):
        text = log.read_text(errors="replace")
        _, rows = parse_chained_block(text, "W_2")
        for m in (BASELINE, args.winner):
            if m not in rows:
                continue
            vals = rows[m]
            for stage in range(min(n_stages_pred, len(vals))):
                w2[m][stage][si] = vals[stage]

    print(f"\n=== {args.prefix}: per-stage W_2 (n=5 paired splits) ===\n")
    print(f"  pred-stage | t-col           | role  | {BASELINE:<22} | spectral winner             | Δ%       | p_paired")
    print(f"  -----------+-----------------+-------+------------------------+-----------------------------+----------+---------")

    held_b, held_w, train_b, train_w = [], [], [], []
    for stage in range(n_stages_pred):
        actual_stage = stage + 1
        role = "HELD" if actual_stage in held else "TRAIN"
        b = np.array(w2[BASELINE][stage], dtype=float)
        ww = np.array(w2[args.winner][stage], dtype=float)
        if np.any(np.isnan(b)) or np.any(np.isnan(ww)):
            continue
        diff = ww - b
        delta = (ww.mean()/b.mean()-1)*100
        try:
            _, p = stats.wilcoxon(diff, alternative="less")
        except ValueError:
            p = float("nan")
        print(f"  s={actual_stage:<7}  | {t_cols[stage]:<15} | {role:<5} | "
              f"{b.mean():>10.4f} ± {b.std(ddof=1):>5.4f}  | "
              f"{ww.mean():>10.4f} ± {ww.std(ddof=1):>5.4f}    | "
              f"{delta:>+6.1f}%  | {p:.4f}")
        if role == "HELD":
            held_b.append(b); held_w.append(ww)
        else:
            train_b.append(b); train_w.append(ww)

    def role_avg(arrays):
        if not arrays: return None
        return np.mean(np.vstack(arrays), axis=0)

    def report(label, b, w):
        if b is None: return None
        d = w - b
        try: _, p = stats.wilcoxon(d, alternative="less")
        except ValueError: p = float("nan")
        delta = (w.mean()/b.mean()-1)*100
        print(f"  {label:<11}            |                 |       | "
              f"{b.mean():>10.4f} ± {b.std(ddof=1):>5.4f}  | "
              f"{w.mean():>10.4f} ± {w.std(ddof=1):>5.4f}    | "
              f"{delta:>+6.1f}%  | {p:.4f}")
        return p

    print()
    p_held = report("AVG held", role_avg(held_b), role_avg(held_w)) if held else None
    p_train = report("AVG train", role_avg(train_b), role_avg(train_w))

    # ---- LaTeX ----
    tex_path = PROJ / "surf_latex" / "final_report" / f"{args.tag}_w2_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{}",
        rf"\label{{tab:{args.tag}_w2}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Stage & $t$ & role & MM+Linear $W_2$ & spectral winner $W_2$ & $\Delta\%$ & paired $p$ \\",
        r"\midrule",
    ]
    for stage in range(n_stages_pred):
        actual_stage = stage + 1
        role = "held" if actual_stage in held else "train"
        b = np.array(w2[BASELINE][stage], dtype=float)
        ww = np.array(w2[args.winner][stage], dtype=float)
        if np.any(np.isnan(b)) or np.any(np.isnan(ww)):
            continue
        diff = ww - b
        delta = (ww.mean()/b.mean()-1)*100
        try: _, p = stats.wilcoxon(diff, alternative="less")
        except ValueError: p = float("nan")
        cell_w = f"${ww.mean():.3f}\\pm{ww.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            f"$s_{{{actual_stage}}}$ & ${t_cols[stage]}$ & {role} & "
            f"${b.mean():.3f}\\pm{b.std(ddof=1):.3f}$ & {cell_w} & "
            f"${delta:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    lines.append(r"\midrule")
    if held:
        b, w = role_avg(held_b), role_avg(held_w)
        d = w - b
        try: _, p = stats.wilcoxon(d, alternative="less")
        except ValueError: p = float("nan")
        delta = (w.mean()/b.mean()-1)*100
        cell_w = f"${w.mean():.3f}\\pm{w.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        lines.append(
            r"\multicolumn{3}{l}{\emph{avg.\ over held-out marginals}} & " +
            f"${b.mean():.3f}\\pm{b.std(ddof=1):.3f}$ & {cell_w} & " +
            f"${delta:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    b, w = role_avg(train_b), role_avg(train_w)
    if b is not None:
        d = w - b
        try: _, p = stats.wilcoxon(d, alternative="less")
        except ValueError: p = float("nan")
        delta = (w.mean()/b.mean()-1)*100
        cell_w = f"${w.mean():.3f}\\pm{w.std(ddof=1):.3f}$"
        if not np.isnan(p) and p < 0.05:
            cell_w = r"\textbf{" + cell_w + r"}$^{\star}$"
        role_label = "train marginals" if held else "all evaluated marginals"
        lines.append(
            rf"\multicolumn{{3}}{{l}}{{\emph{{avg.\ over {role_label}}}}} & " +
            f"${b.mean():.3f}\\pm{b.std(ddof=1):.3f}$ & {cell_w} & " +
            f"${delta:+.1f}\\%$ & ${p:.3f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {tex_path}")


if __name__ == "__main__":
    main()
