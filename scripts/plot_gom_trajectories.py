"""Visualize predicted trajectories on GoM (2D) for the spectral vs Euclidean
OT-FM comparison.

Trains MM+Linear (Euclidean OT) and our headline spectral model
(MM+Linear+SquaredSpectral, alpha=0.5, beta=0, Euclidean kNN graph)
on a single seed of the GoM 9-stage OTP-FM hold-out protocol, integrates
a sample of source cells forward in time, and produces a side-by-side
scatter plot showing each method's predicted trajectories overlaid on
the ground-truth marginals (one color per stage).

Usage:
  .venv/bin/python scripts/plot_gom_trajectories.py
  .venv/bin/python scripts/plot_gom_trajectories.py --refit  # retrain even if cache exists
  .venv/bin/python scripts/plot_gom_trajectories.py --n-traj 80 --integration-steps 200
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force Euclidean kNN for the spectral graph (matches the headline result).
os.environ.setdefault("SMFM_KNN_METRIC", "euclidean")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from surf.runtime import setup as setup_runtime
from surf.data.gom import load_gom
from surf.training.euclidean_flow_trainer import train_multi_marginal_euclidean_flow
from surf.ot.costs import (
    compute_euclidean_cost_matrix,
    make_spectral_plus_euclidean_cost_fn,
)


PROJ = Path(__file__).resolve().parent.parent
FIG_DIR = PROJ / "surf_latex" / "final_report" / "figures"
CACHE = PROJ / "outputs" / "gom_trajectories_cache.npz"

ALL_TIMES = np.linspace(0.0, 1.0, 9)             # 9 stages spanning [0, 1]
HELD_IDX = {1, 3, 5, 7}                          # OTP-FM-9 holdout stages


def integrate_with_traj(model, x0, n_steps, device):
    """Euler integrate from t=0 to t=1, returning trajectory of shape (n_steps+1, B, D)."""
    model.eval()
    dt = 1.0 / n_steps
    xt = x0.clone().to(device)
    traj = [xt.cpu().numpy().copy()]
    with torch.no_grad():
        for step in range(n_steps):
            t_val = torch.full((len(xt),), step * dt, device=device)
            v = model(xt, t_val)
            xt = xt + dt * v
            traj.append(xt.cpu().numpy().copy())
    return np.stack(traj, axis=0)                # (n_steps+1, B, D)


def train_or_load(args):
    if CACHE.exists() and not args.refit:
        print(f"Loading cache from {CACHE}")
        data = dict(np.load(CACHE, allow_pickle=True))
        return data

    setup_runtime(device="cpu")
    print("Loading GoM (9 stages, otpfm hold-out, seed=42)...")
    data = load_gom(
        n_stages=9, otpfm_split=True, holdout_stages=[1, 3, 5, 7],
        seed=42, normalize=True,
    )

    stages = data["train"]["stages"]                              # 9 stage names t0..t8
    all_cells = []
    train_cells_only = []
    for i, s in enumerate(stages):
        train_arr = data["train"]["cells"][s].numpy()
        test_arr = data["test"]["cells"][s].numpy() if data["test"]["cells"][s].numel() else None
        # Held marginals live in 'test' split; train marginals in 'train'.
        all_cells.append(train_arr if len(train_arr) > 0 else test_arr)
        if i not in HELD_IDX and len(train_arr) > 0:
            train_cells_only.append(train_arr)

    train_times = [ALL_TIMES[i] for i in range(len(stages)) if i not in HELD_IDX]
    train_cells_t = [torch.from_numpy(c) for c in train_cells_only]
    print(f"  Training on {len(train_cells_t)} marginals at t={train_times}")

    # Train Euclidean baseline
    print("\nTraining MM+Linear (Euclidean OT)...")
    model_eucl, _ = train_multi_marginal_euclidean_flow(
        train_cells_t, train_times, D=2,
        n_iters=args.n_iters, batch_size=512, lr=3e-4,
        ot_subsample=5000, label="MM+Linear",
        cost_fn=compute_euclidean_cost_matrix,
        coupling_mode="joint", use_ema=True,
    )

    # Train spectral winner: alpha=2, beta=0 (pure spectral, biharmonic) with Euclidean kNN.
    spectral_cost = make_spectral_plus_euclidean_cost_fn(
        blend=0.0, knn=15, n_eig=50,
        spectral_family="power", weight_power=1.0,     # alpha=2 -> weight_power=1.0
    )
    print(f"\nTraining MM+Linear+SquaredSpectral (alpha=2, beta=0, EucKNN)...")
    model_spec, _ = train_multi_marginal_euclidean_flow(
        train_cells_t, train_times, D=2,
        n_iters=args.n_iters, batch_size=512, lr=3e-4,
        ot_subsample=5000, label="MM+Linear+Spec",
        cost_fn=spectral_cost,
        coupling_mode="joint", use_ema=True,
    )

    # Pick source cells from t=0 (first train marginal)
    rng = np.random.default_rng(0)
    src = train_cells_only[0]
    sub_idx = rng.choice(len(src), size=min(args.n_traj, len(src)), replace=False)
    x0 = torch.from_numpy(src[sub_idx]).float()

    print(f"\nIntegrating {len(x0)} source cells forward with {args.integration_steps} steps...")
    traj_eucl = integrate_with_traj(model_eucl, x0, args.integration_steps, "cpu")
    traj_spec = integrate_with_traj(model_spec, x0, args.integration_steps, "cpu")

    # Save cache (numpy array of arrays)
    cache_data = {
        "all_cells": np.array(all_cells, dtype=object),
        "all_times": np.asarray(ALL_TIMES),
        "held_idx": np.asarray(sorted(HELD_IDX)),
        "traj_eucl": traj_eucl,
        "traj_spec": traj_spec,
        "x0_subset": x0.numpy(),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **cache_data)
    print(f"Cached results to {CACHE}")
    return cache_data


def plot(data, args):
    all_cells = data["all_cells"]
    all_times = data["all_times"]
    held_idx = set(int(i) for i in data["held_idx"])
    traj_eucl = data["traj_eucl"]
    traj_spec = data["traj_spec"]
    n_traj_show = min(args.n_traj, traj_eucl.shape[1], traj_spec.shape[1])
    traj_eucl = traj_eucl[:, :n_traj_show, :]
    traj_spec = traj_spec[:, :n_traj_show, :]

    # Match the per-stage palette used in plot_pca_predictions.py:
    #   t=0 red, t=0.25 orange, t=0.5 limegreen, t=0.75 lightskyblue, t=1.0 purple.
    # GoM has 9 evenly-spaced stages, so build a smooth interpolation in RGB
    # through those anchor colors and sample at each stage time.
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    anchors = [(0.0, "red"), (0.25, "orange"), (0.5, "limegreen"),
               (0.75, "lightskyblue"), (1.0, "purple")]
    cmap = LinearSegmentedColormap.from_list(
        "smfm_stages", [(t, to_rgba(c)) for t, c in anchors]
    )
    stage_colors = [cmap(t) for t in all_times]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=True, sharey=True)
    titles = ["Linear FM", r"SMFM, $\alpha{=}2,\beta{=}0$"]
    trajs = [traj_eucl, traj_spec]

    # Compute axis limits across all marginals
    all_pts = np.concatenate([c for c in all_cells], axis=0)
    pad = 0.5
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    for ax, title, traj in zip(axes, titles, trajs):
        # Background marginals
        for i, (cells, t, color) in enumerate(zip(all_cells, all_times, stage_colors)):
            if cells is None:
                continue
            edge = "black" if i in held_idx else "none"
            lw = 0.4 if i in held_idx else 0
            ax.scatter(cells[:, 0], cells[:, 1], s=14, c=[color],
                       edgecolors=edge, linewidth=lw, alpha=0.55, zorder=2)

        # Trajectories: one line per source cell, colored by progress
        # Use a thin grey line so individual paths are legible without
        # overpowering the marginal cloud.
        n_steps_total = traj.shape[0]
        for j in range(traj.shape[1]):
            ax.plot(traj[:, j, 0], traj[:, j, 1], color="0.15",
                    lw=0.45, alpha=0.55, zorder=3)

        # Highlight the source cells
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=22,
                   c=[stage_colors[0]], edgecolors="none", linewidth=0,
                   zorder=4, label="source ($t{=}0$)")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.7")

    # Discrete stage legend. Held-out marginals keep the same black outline
    # convention used in the scatter plots.
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="black" if i in held_idx else color,
            markeredgewidth=0.9 if i in held_idx else 0.0,
            label=f"{t:.2f}",
        )
        for i, (t, color) in enumerate(zip(all_times, stage_colors))
    ]
    fig.subplots_adjust(bottom=0.20, wspace=0.05)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        title="stage time $t$ (black outline = held out)",
        fontsize=8,
        title_fontsize=9,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    out_png = FIG_DIR / "gom_trajectories.png"
    out_pdf = FIG_DIR / "gom_trajectories.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-traj", type=int, default=25,
                        help="Number of source cells to integrate or show from cache (default 25)")
    parser.add_argument("--integration-steps", type=int, default=200,
                        help="Euler integration steps t in [0,1] (default 200)")
    parser.add_argument("--n-iters", type=int, default=3000,
                        help="Training iterations per model (default 3000)")
    parser.add_argument("--refit", action="store_true",
                        help="Retrain even if cache exists")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = train_or_load(args)
    plot(data, args)


if __name__ == "__main__":
    main()
