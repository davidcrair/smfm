"""Plot per-stage W_2 for the GoM 9-stage OTP-FM hold-out best-model rerun.

Reads the same five split logs as scripts/build_gom_otpfm9_w2_table.py
and emits a publication-style figure to
surf_latex/final_report/figures/gom_otpfm9_w2.{png,pdf}.

The figure shows mean +/- std W_2 across 5 splits at each of the 8
predicted stages (t1..t8), with two lines (MM+Linear baseline vs
spectral winner) and held/train marker styles.

Usage: .venv/bin/python scripts/plot_gom_otpfm9_w2.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJ = Path(__file__).resolve().parent.parent
LOG_DIR = PROJ / "logs"
FIG_DIR = PROJ / "surf_latex" / "final_report" / "figures"
SPLITS = [42, 43, 44, 45, 46]
LOG_PREFIX = "gom_otpfm9_best_w2"

WINNER = "MM+Linear+SquaredSpectral@alpha=0.5,blend=0.25"
BASELINE = "MM+Linear"

STAGE_TIMES = [0.00, 0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88, 1.00]
PRED_TIMES = STAGE_TIMES[1:]                      # 8 predicted stages t1..t8
HELD = {1, 3, 5, 7}                                # 1-indexed predicted stage idx


def parse_chained_block(text: str, metric_name: str):
    title_regex = rf"CHAINED EVAL \[{re.escape(metric_name)}\] \(1 seed\)"
    pat = re.compile(
        r"={80}\s*\n\s*" + title_regex + r"\s*\n={80}\s*\n"
        r"(?P<body>.*?)(?=\n={80}|\n\s*Metric conventions:|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        return {}
    body = m.group("body").splitlines()
    hdr_idx = next((i for i, l in enumerate(body) if l.lstrip().startswith("Method")), None)
    if hdr_idx is None:
        return {}
    rows = {}
    for line in body[hdr_idx + 1:]:
        toks = line.split()
        if len(toks) < 9:
            continue
        name = toks[0]
        try:
            vals = [float(x) for x in toks[1:9]]
        except ValueError:
            continue
        rows[name] = vals
    return rows


def collect():
    """Returns w2[method] = ndarray (n_splits, 8 stages)."""
    out = {BASELINE: [], WINNER: []}
    for s in SPLITS:
        log = LOG_DIR / f"{LOG_PREFIX}_split{s}.log"
        if not log.exists():
            print(f"  warn: missing {log}")
            continue
        rows = parse_chained_block(log.read_text(), "W_2")
        for method in out:
            if method in rows:
                out[method].append(rows[method])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def main():
    w2 = collect()
    if any(v.ndim != 2 or v.shape[0] != len(SPLITS) for v in w2.values()):
        raise SystemExit(f"Unexpected shapes: {[v.shape for v in w2.values()]}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    times = np.asarray(PRED_TIMES)
    means_b = w2[BASELINE].mean(axis=0)
    stds_b = w2[BASELINE].std(axis=0, ddof=1)
    means_w = w2[WINNER].mean(axis=0)
    stds_w = w2[WINNER].std(axis=0, ddof=1)
    held_mask = np.array([(i + 1) in HELD for i in range(8)])

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    # Lines first, then per-stage markers (open=train, filled=held).
    color_b, color_w = "#9b9b9b", "#1f77b4"
    ax.plot(times, means_b, "-", color=color_b, lw=1.4, alpha=0.9, zorder=2)
    ax.plot(times, means_w, "-", color=color_w, lw=1.6, zorder=3)
    ax.fill_between(times, means_b - stds_b, means_b + stds_b,
                    color=color_b, alpha=0.18, zorder=1)
    ax.fill_between(times, means_w - stds_w, means_w + stds_w,
                    color=color_w, alpha=0.20, zorder=1)

    for i, t in enumerate(times):
        face_b = color_b if held_mask[i] else "white"
        face_w = color_w if held_mask[i] else "white"
        ax.plot(t, means_b[i], marker="o", ms=6, mec=color_b, mfc=face_b,
                mew=1.3, zorder=4)
        ax.plot(t, means_w[i], marker="o", ms=6.5, mec=color_w, mfc=face_w,
                mew=1.4, zorder=5)

    # Held-stage shading
    for i, t in enumerate(times):
        if held_mask[i]:
            ax.axvspan(t - 0.06, t + 0.06, color="0.92", zorder=0)

    ax.set_xlabel("predicted time $t$")
    ax.set_ylabel("Wasserstein-2 distance")
    ax.set_xticks(times)
    ax.set_xticklabels([f"{t:.2f}" for t in times], fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend with held/train marker key
    h_line_b = plt.Line2D([0], [0], color=color_b, lw=1.4,
                          marker="o", mfc=color_b, mec=color_b,
                          ms=6, label="MM+Linear (Eucl.~OT)")
    h_line_w = plt.Line2D([0], [0], color=color_w, lw=1.6,
                          marker="o", mfc=color_w, mec=color_w,
                          ms=6.5, label=r"Ours: spectral, $\alpha{=}0.5,\beta{=}0.25$")
    h_held = plt.Line2D([0], [0], color="black", lw=0,
                        marker="o", mfc="black", mec="black",
                        ms=6, label="held-out")
    h_train = plt.Line2D([0], [0], color="black", lw=0,
                         marker="o", mfc="white", mec="black",
                         ms=6, label="train")
    ax.legend(handles=[h_line_b, h_line_w, h_held, h_train],
              loc="upper left", frameon=False, fontsize=8, ncol=2,
              handletextpad=0.5, columnspacing=0.8)

    plt.tight_layout()
    out_png = FIG_DIR / "gom_otpfm9_w2.png"
    out_pdf = FIG_DIR / "gom_otpfm9_w2.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")

    # Console summary
    print()
    print(f"{'pred-stage':<11}{'t':<7}{'role':<7}{'Linear W_2':<22}{'Spectral W_2':<22}{'Δ%':>7}")
    print("-" * 76)
    for i, t in enumerate(times):
        role = "held" if held_mask[i] else "train"
        delta = (means_w[i] - means_b[i]) / means_b[i] * 100
        print(f"s={i+1:<9}{t:<7.2f}{role:<7}"
              f"{means_b[i]:.4f} ± {stds_b[i]:.4f}     "
              f"{means_w[i]:.4f} ± {stds_w[i]:.4f}    {delta:+5.1f}%")
    held_b = w2[BASELINE][:, held_mask].mean(axis=1)
    held_w = w2[WINNER][:, held_mask].mean(axis=1)
    train_b = w2[BASELINE][:, ~held_mask].mean(axis=1)
    train_w = w2[WINNER][:, ~held_mask].mean(axis=1)
    print("-" * 76)
    print(f"avg HELD          {held_b.mean():.4f} ± {held_b.std(ddof=1):.4f}     "
          f"{held_w.mean():.4f} ± {held_w.std(ddof=1):.4f}    "
          f"{(held_w.mean() / held_b.mean() - 1) * 100:+5.1f}%")
    print(f"avg TRAIN         {train_b.mean():.4f} ± {train_b.std(ddof=1):.4f}     "
          f"{train_w.mean():.4f} ± {train_w.std(ddof=1):.4f}    "
          f"{(train_w.mean() / train_b.mean() - 1) * 100:+5.1f}%")


if __name__ == "__main__":
    main()
