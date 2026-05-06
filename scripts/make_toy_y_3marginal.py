"""Generate a 2D three-marginal 'Y' toy dataset and plot it.

Class 0 (t=0): bottom of the Y (stem base)
Class 1 (t=1): junction where the Y splits
Class 2 (t=2): the two tips of the Y (left + right branches, pooled)

Usage:
    uv run python scripts/make_toy_y_3marginal.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_Y = -2.0
BASE_HALF_WIDTH = 0.8
STEM_TOP_Y = -0.5  # top of the vertical piece of the base serif (closer to junction)
JUNCTION = np.array([0.0, 0.0])
LEFT_TIP = np.array([-1.2, 1.5])
RIGHT_TIP = np.array([1.2, 1.5])
TIP_HALF_WIDTH = 0.45  # half-width of the top serif at each branch tip
TIP_STUB_FRAC = 0.75   # fraction of the tip->junction segment for the slanted stub
JUNCTION_ARM_FRAC = 0.55  # fraction of each branch used by the middle marginal
JUNCTION_STEM_Y_END = -1.3  # how far down the stem the middle marginal extends


def sample_blob(center, n, sigma, rng):
    return center + sigma * rng.standard_normal(size=(n, 2))


def sample_segment(p0, p1, n, sigma, rng):
    u = rng.uniform(0.0, 1.0, size=(n, 1))
    pts = (1.0 - u) * p0 + u * p1
    return pts + sigma * rng.standard_normal(size=(n, 2))


def make_dataset(n_per_class=300, sigma=0.08, seed=0):
    rng = np.random.default_rng(seed)
    # t=0: upside-down T (Yale-Y serif foot) — horizontal bar + short vertical stub.
    n_bar = int(0.65 * n_per_class)
    n_stub = n_per_class - n_bar
    bar = sample_segment(
        np.array([-BASE_HALF_WIDTH, BASE_Y]),
        np.array([BASE_HALF_WIDTH, BASE_Y]),
        n_bar, sigma, rng,
    )
    stub = sample_segment(
        np.array([0.0, BASE_Y]),
        np.array([0.0, STEM_TOP_Y]),
        n_stub, sigma, rng,
    )
    m1 = np.vstack([bar, stub])
    # t=1: middle marginal spanning the junction, extending along all three
    # arms (down the stem and up each branch).
    left_arm_end = JUNCTION + JUNCTION_ARM_FRAC * (LEFT_TIP - JUNCTION)
    right_arm_end = JUNCTION + JUNCTION_ARM_FRAC * (RIGHT_TIP - JUNCTION)
    stem_arm_end = np.array([0.0, JUNCTION_STEM_Y_END])
    per_arm = n_per_class // 3
    remainder = n_per_class - 3 * per_arm
    m2 = np.vstack([
        sample_segment(JUNCTION, stem_arm_end, per_arm + remainder, sigma, rng),
        sample_segment(JUNCTION, left_arm_end, per_arm, sigma, rng),
        sample_segment(JUNCTION, right_arm_end, per_arm, sigma, rng),
    ])

    # t=2: two right-side-up T's at the tips, each with its vertical stub
    # slanted toward the junction (matching the Yale-Y serifs at the branches).
    n_tip = n_per_class // 2
    m3 = np.vstack([
        _sample_tip_T(LEFT_TIP, n_tip, sigma, rng),
        _sample_tip_T(RIGHT_TIP, n_per_class - n_tip, sigma, rng),
    ])
    return m1, m2, m3


def _sample_tip_T(tip, n, sigma, rng):
    n_bar = int(0.6 * n)
    n_stub = n - n_bar
    bar = sample_segment(
        tip + np.array([-TIP_HALF_WIDTH, 0.0]),
        tip + np.array([TIP_HALF_WIDTH, 0.0]),
        n_bar, sigma, rng,
    )
    stub_end = tip + TIP_STUB_FRAC * (JUNCTION - tip)
    stub = sample_segment(tip, stub_end, n_stub, sigma, rng)
    return np.vstack([bar, stub])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="toy_y_3marginal.png")
    parser.add_argument("--npz", default="toy_y_3marginal.npz")
    args = parser.parse_args()

    m1, m2, m3 = make_dataset(args.n_per_class, args.sigma, args.seed)

    fig, ax = plt.subplots(figsize=(5, 6))
    # Y skeleton for reference
    base_left = np.array([-BASE_HALF_WIDTH, BASE_Y])
    base_right = np.array([BASE_HALF_WIDTH, BASE_Y])
    stub_top = np.array([0.0, STEM_TOP_Y])
    ax.plot(*np.vstack([base_left, base_right]).T, color="0.75", lw=1, zorder=0)
    ax.plot(*np.vstack([np.array([0.0, BASE_Y]), stub_top]).T, color="0.75", lw=1, zorder=0)
    ax.plot(*np.vstack([stub_top, JUNCTION]).T, color="0.75", lw=1, zorder=0)
    ax.plot(*np.vstack([JUNCTION, LEFT_TIP]).T, color="0.75", lw=1, zorder=0)
    ax.plot(*np.vstack([JUNCTION, RIGHT_TIP]).T, color="0.75", lw=1, zorder=0)
    for tip in (LEFT_TIP, RIGHT_TIP):
        bar_l = tip + np.array([-TIP_HALF_WIDTH, 0.0])
        bar_r = tip + np.array([TIP_HALF_WIDTH, 0.0])
        ax.plot(*np.vstack([bar_l, bar_r]).T, color="0.75", lw=1, zorder=0)

    for pts, color, label in [
        (m1, "#1f77b4", "t=0 (stem base)"),
        (m2, "#2ca02c", "t=1 (junction)"),
        (m3, "#d62728", "t=2 (tips)"),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color, alpha=0.7, label=label)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Y-shaped 3-marginal toy")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(f"  Wrote {out.resolve()}")

    np.savez(args.npz, m1=m1, m2=m2, m3=m3)
    print(f"  Wrote {Path(args.npz).resolve()}  shapes: {m1.shape}, {m2.shape}, {m3.shape}")


if __name__ == "__main__":
    main()
