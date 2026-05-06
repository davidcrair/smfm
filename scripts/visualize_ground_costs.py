#!/usr/bin/env python
"""Presentation-oriented visualization of sphere vs biharmonic ground costs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from surf.data.embryoid import load_embryoid_body
from surf.geometry.sphere import compute_sphere_cost_matrix, normalize_sphere, to_compositional, to_orthant
from surf.ot.costs import compute_biharmonic_cost_matrix
from surf.ot.coupling import ot_coupling


def _resolve_stage(stages: list[str], stage_arg: str | None, stage_idx: int | None, flag_name: str) -> tuple[int, str]:
    if stage_arg is not None:
        if stage_arg not in stages:
            raise ValueError(f"{flag_name}={stage_arg!r} not in stages={stages}")
        idx = stages.index(stage_arg)
        return idx, stage_arg
    if stage_idx is None:
        raise ValueError(f"Need either {flag_name}-stage or {flag_name}-idx")
    if not (0 <= stage_idx < len(stages)):
        raise ValueError(f"{flag_name}-idx={stage_idx} out of range [0, {len(stages) - 1}]")
    return stage_idx, stages[stage_idx]


def _subsample(X: torch.Tensor, n: int, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(X) <= n:
        return X, np.arange(len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], idx


def _fit_embedding(X: np.ndarray, method: str) -> np.ndarray:
    if method == "phate":
        try:
            import phate
        except ImportError as exc:
            raise RuntimeError("PHATE not installed; use --embedding pca") from exc
        op = phate.PHATE(n_components=2, knn=15, t="auto", verbose=0, random_state=42)
        return op.fit_transform(X)

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(X)

    raise ValueError(f"Unknown embedding method: {method!r}")


def _row_rank01(cost_row: np.ndarray) -> np.ndarray:
    order = np.argsort(cost_row)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=len(cost_row), endpoint=True, dtype=np.float32)
    return ranks


def _choose_anchor(cost_sphere: np.ndarray, cost_biharm: np.ndarray, top_k: int) -> int:
    n_src, n_tgt = cost_sphere.shape
    top_k = min(top_k, n_tgt)
    best_idx = 0
    best_score = -1.0
    for i in range(n_src):
        s_top = set(np.argsort(cost_sphere[i])[:top_k].tolist())
        b_top = set(np.argsort(cost_biharm[i])[:top_k].tolist())
        overlap = len(s_top & b_top) / max(len(s_top | b_top), 1)
        score = 1.0 - overlap
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _sample_plan(cost: np.ndarray, n_show: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    return ot_coupling(cost, n_show)


def _plot_anchor_cost_panel(ax, xy_src, xy_tgt, anchor_idx, rank01, title):
    scatter = ax.scatter(
        xy_tgt[:, 0],
        xy_tgt[:, 1],
        c=rank01,
        cmap="viridis_r",
        s=24,
        alpha=0.95,
        edgecolor="none",
    )
    ax.scatter(xy_src[:, 0], xy_src[:, 1], c="#b3b3b3", s=16, alpha=0.25, edgecolor="none")
    ax.scatter(
        xy_src[anchor_idx, 0],
        xy_src[anchor_idx, 1],
        c="#d62728",
        s=120,
        marker="*",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )

    top10 = np.argsort(rank01)[:10]
    ax.scatter(
        xy_tgt[top10, 0],
        xy_tgt[top10, 1],
        facecolor="none",
        edgecolor="white",
        s=70,
        linewidth=1.0,
        zorder=4,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter


def _plot_coupling_panel(ax, xy_src, xy_tgt, src_idx, tgt_idx, title):
    ax.scatter(xy_src[:, 0], xy_src[:, 1], c="#8c8c8c", s=18, alpha=0.35, edgecolor="none", label="source")
    ax.scatter(xy_tgt[:, 0], xy_tgt[:, 1], c="#1f77b4", s=18, alpha=0.45, edgecolor="none", label="target")
    for s, t in zip(src_idx, tgt_idx):
        ax.plot(
            [xy_src[s, 0], xy_tgt[t, 0]],
            [xy_src[s, 1], xy_tgt[t, 1]],
            c="#111111",
            alpha=0.18,
            linewidth=0.7,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_rank_panel(ax, sphere_rank01, biharm_rank01):
    ax.scatter(
        sphere_rank01,
        biharm_rank01,
        c=np.maximum(np.abs(sphere_rank01 - biharm_rank01), 0.05),
        cmap="magma",
        s=18,
        alpha=0.85,
        edgecolor="none",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Sphere cost rank")
    ax.set_ylabel("Biharmonic cost rank")
    ax.set_title("How Much The Two Costs Reorder Targets", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="embryoid_body.h5ad")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--src-stage", default="0-1")
    parser.add_argument("--tgt-stage", default="4-5")
    parser.add_argument("--src-idx", type=int)
    parser.add_argument("--tgt-idx", type=int)
    parser.add_argument("--n-src", type=int, default=250)
    parser.add_argument("--n-tgt", type=int, default=250)
    parser.add_argument("--embedding", default="phate", choices=["phate", "pca"])
    parser.add_argument("--anchor-mode", default="auto", choices=["auto", "index"])
    parser.add_argument("--anchor-index", type=int, default=0)
    parser.add_argument("--anchor-top-k", type=int, default=20)
    parser.add_argument("--show-couplings", type=int, default=80)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--n-eig", type=int, default=50)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = load_embryoid_body(args.data_path, n_hvg=args.n_hvg)
    stages = data[args.split]["stages"]
    src_idx, src_label = _resolve_stage(stages, args.src_stage, args.src_idx, "src")
    tgt_idx, tgt_label = _resolve_stage(stages, args.tgt_stage, args.tgt_idx, "tgt")
    if src_idx == tgt_idx:
        raise ValueError("Source and target stages must differ")

    src_log1p = data[args.split]["cells"][src_label]
    tgt_log1p = data[args.split]["cells"][tgt_label]
    src_sphere_full = normalize_sphere(to_orthant(to_compositional(src_log1p)))
    tgt_sphere_full = normalize_sphere(to_orthant(to_compositional(tgt_log1p)))

    src_sphere, src_sel = _subsample(src_sphere_full, args.n_src, args.seed)
    tgt_sphere, tgt_sel = _subsample(tgt_sphere_full, args.n_tgt, args.seed + 1)
    src_log1p = src_log1p[src_sel]
    tgt_log1p = tgt_log1p[tgt_sel]

    cost_sphere = compute_sphere_cost_matrix(src_sphere, tgt_sphere)
    cost_biharm = compute_biharmonic_cost_matrix(
        src_sphere,
        tgt_sphere,
        knn=args.knn,
        n_eig=args.n_eig,
        weight_power=args.weight_power,
    )

    src_np = src_log1p.numpy()
    tgt_np = tgt_log1p.numpy()
    embedding = _fit_embedding(np.vstack([src_np, tgt_np]), args.embedding)
    xy_src = embedding[: len(src_np)]
    xy_tgt = embedding[len(src_np) :]

    if args.anchor_mode == "auto":
        anchor_idx = _choose_anchor(cost_sphere, cost_biharm, args.anchor_top_k)
    else:
        if not (0 <= args.anchor_index < len(src_sphere)):
            raise ValueError(f"--anchor-index must be in [0, {len(src_sphere) - 1}]")
        anchor_idx = args.anchor_index

    sphere_rank01 = _row_rank01(cost_sphere[anchor_idx])
    biharm_rank01 = _row_rank01(cost_biharm[anchor_idx])
    sphere_src, sphere_tgt = _sample_plan(cost_sphere, args.show_couplings, args.seed)
    biharm_src, biharm_tgt = _sample_plan(cost_biharm, args.show_couplings, args.seed + 1)

    default_out = f"ground_costs_{src_label}_to_{tgt_label}_{args.embedding}.png".replace("/", "-")
    out_path = Path(args.out or default_out)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle(
        (
            f"Great-Circle vs Biharmonic Ground Cost"
            f"\n{src_label} -> {tgt_label} ({args.split} split, {len(src_sphere)}x{len(tgt_sphere)} cells)"
        ),
        fontsize=15,
    )

    axes[0, 0].scatter(xy_src[:, 0], xy_src[:, 1], c="#8c8c8c", s=22, alpha=0.45, edgecolor="none", label=f"source {src_label}")
    axes[0, 0].scatter(xy_tgt[:, 0], xy_tgt[:, 1], c="#1f77b4", s=22, alpha=0.45, edgecolor="none", label=f"target {tgt_label}")
    axes[0, 0].scatter(
        xy_src[anchor_idx, 0],
        xy_src[anchor_idx, 1],
        c="#d62728",
        s=130,
        marker="*",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label="anchor source cell",
    )
    axes[0, 0].set_title("Union Embedding + Chosen Anchor", fontsize=11)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].legend(loc="best", fontsize=8, frameon=True)

    c1 = _plot_anchor_cost_panel(axes[0, 1], xy_src, xy_tgt, anchor_idx, sphere_rank01, "Sphere / Great-Circle Cost To Anchor")
    c2 = _plot_anchor_cost_panel(axes[0, 2], xy_src, xy_tgt, anchor_idx, biharm_rank01, "Biharmonic Cost To Anchor")
    cb1 = fig.colorbar(c1, ax=axes[0, 1], fraction=0.046, pad=0.02)
    cb2 = fig.colorbar(c2, ax=axes[0, 2], fraction=0.046, pad=0.02)
    cb1.set_label("cheap  <->  expensive")
    cb2.set_label("cheap  <->  expensive")

    _plot_coupling_panel(axes[1, 0], xy_src, xy_tgt, sphere_src, sphere_tgt, f"OT Pairs Under Sphere Cost ({args.show_couplings} samples)")
    _plot_coupling_panel(axes[1, 1], xy_src, xy_tgt, biharm_src, biharm_tgt, f"OT Pairs Under Biharmonic Cost ({args.show_couplings} samples)")
    _plot_rank_panel(axes[1, 2], sphere_rank01, biharm_rank01)

    overlap = len(set(np.argsort(sphere_rank01)[: args.anchor_top_k]) & set(np.argsort(biharm_rank01)[: args.anchor_top_k]))
    fig.text(
        0.5,
        0.01,
        (
            f"Anchor auto-selected for disagreement. Top-{args.anchor_top_k} cheapest-target overlap: "
            f"{overlap}/{args.anchor_top_k}. "
            f"Diagonal rank panel means the two costs agree; off-diagonal points are targets reordered by manifold geometry."
        ),
        ha="center",
        fontsize=10,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
