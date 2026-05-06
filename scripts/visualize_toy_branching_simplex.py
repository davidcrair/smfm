#!/usr/bin/env python
"""Visualize a branching toy dataset on the simplex and positive-orthant sphere.

This script builds a five-marginal "near-touching Y" dataset on the 2-simplex,
maps it to the positive orthant of S^2 via x = sqrt(p), and compares:

- sphere/geodesic OT cost
- biharmonic OT cost

It is designed to make the mechanism behind branch-aware couplings visible:
the two leaves curl back toward each other ambiently, so geodesic cost is
tempted to shortcut across branches while biharmonic cost remains topology-aware
on the sample graph.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from surf.geometry.sphere import compute_sphere_cost_matrix, normalize_sphere, to_orthant
from scipy.spatial.distance import cdist

from surf.ot.costs import compute_global_biharmonic_embedding
from surf.ot.coupling import ot_coupling


SQRT3 = np.sqrt(3.0)


@dataclass
class Marginal:
    name: str
    simplex: np.ndarray
    branch: np.ndarray


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-8, None)
    return p / p.sum()


def _quad_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    a = (1.0 - u)[:, None] ** 2 * p0[None, :]
    b = 2.0 * (1.0 - u)[:, None] * u[:, None] * p1[None, :]
    c = u[:, None] ** 2 * p2[None, :]
    out = a + b + c
    out = np.clip(out, 1e-8, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _dirichlet_cloud(centers: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    alpha = np.clip(kappa * centers, 1e-3, None)
    return np.stack([rng.dirichlet(a) for a in alpha], axis=0)


def make_branching_dataset(
    n_per_stage: int = 300,
    kappa: float = 220.0,
    seed: int = 42,
) -> list[Marginal]:
    """Return five simplex marginals for a branching toy process.

    Branch labels:
    - 0 = trunk / branchpoint
    - 1 = left branch
    - 2 = right branch
    """
    rng = np.random.default_rng(seed)

    p0 = _normalize_prob(np.array([0.84, 0.08, 0.08]))
    p2 = _normalize_prob(np.array([0.40, 0.30, 0.30]))
    q_left = _normalize_prob(np.array([0.20, 0.76, 0.04]))
    q_right = _normalize_prob(np.array([0.20, 0.04, 0.76]))
    p4_left = _normalize_prob(np.array([0.08, 0.52, 0.40]))
    p4_right = _normalize_prob(np.array([0.08, 0.40, 0.52]))

    trunk0 = np.repeat(p0[None, :], n_per_stage, axis=0)
    trunk1 = np.repeat(((p0 + p2) / 2.0)[None, :], n_per_stage, axis=0)
    branchpoint = np.repeat(p2[None, :], n_per_stage, axis=0)

    half = n_per_stage // 2
    rem = n_per_stage - 2 * half

    u3_left = rng.uniform(0.45, 0.65, size=half)
    u3_right = rng.uniform(0.45, 0.65, size=half + rem)
    u4_left = np.clip(rng.normal(loc=1.0, scale=0.03, size=half), 0.88, 1.0)
    u4_right = np.clip(rng.normal(loc=1.0, scale=0.03, size=half + rem), 0.88, 1.0)

    t3_left = _quad_bezier(p2, q_left, p4_left, u3_left)
    t3_right = _quad_bezier(p2, q_right, p4_right, u3_right)
    t4_left = _quad_bezier(p2, q_left, p4_left, u4_left)
    t4_right = _quad_bezier(p2, q_right, p4_right, u4_right)

    stages = [
        Marginal("t0_root", _dirichlet_cloud(trunk0, kappa, rng), np.zeros(n_per_stage, dtype=np.int64)),
        Marginal("t1_trunk", _dirichlet_cloud(trunk1, kappa, rng), np.zeros(n_per_stage, dtype=np.int64)),
        Marginal("t2_branch", _dirichlet_cloud(branchpoint, kappa, rng), np.zeros(n_per_stage, dtype=np.int64)),
        Marginal(
            "t3_split",
            _dirichlet_cloud(np.vstack([t3_left, t3_right]), kappa, rng),
            np.concatenate([np.ones(half, dtype=np.int64), np.full(half + rem, 2, dtype=np.int64)]),
        ),
        Marginal(
            "t4_leaves",
            _dirichlet_cloud(np.vstack([t4_left, t4_right]), kappa, rng),
            np.concatenate([np.ones(half, dtype=np.int64), np.full(half + rem, 2, dtype=np.int64)]),
        ),
    ]
    return stages


def simplex_to_xy(p: np.ndarray) -> np.ndarray:
    """Barycentric simplex coordinates in 2D."""
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
    ])
    return p @ vertices


def branch_sort_key(xy: np.ndarray, branch: np.ndarray) -> np.ndarray:
    """Stable ordering for cost-matrix visualization."""
    key = xy[:, 0] + 0.15 * xy[:, 1]
    return np.lexsort((key, branch))


def plot_simplex_overview(stages: list[Marginal], out_path: Path) -> None:
    colors = {0: "#666666", 1: "#1f77b4", 2: "#d62728"}
    fig, axes = plt.subplots(1, len(stages), figsize=(18, 3.8), constrained_layout=True)

    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [0.0, 0.0],
    ])

    for ax, marginal in zip(axes, stages):
        xy = simplex_to_xy(marginal.simplex)
        for branch_id in np.unique(marginal.branch):
            mask = marginal.branch == branch_id
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                s=10, alpha=0.7, c=colors[int(branch_id)], edgecolor="none",
            )
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.0)
        ax.set_title(marginal.name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")

    fig.suptitle("Branching Toy Dataset on the 2-Simplex", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_simplex_all_steps(stages: list[Marginal], out_path: Path) -> None:
    time_colors = ["#111111", "#2c7fb8", "#41ab5d", "#f16913", "#cb181d"]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)

    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [0.0, 0.0],
    ])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.2)

    for i, marginal in enumerate(stages):
        xy = simplex_to_xy(marginal.simplex)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=14,
            alpha=0.58,
            c=time_colors[i],
            edgecolor="none",
            label=marginal.name,
        )

    ax.set_title("Entire Toy Branching Dataset on the 2-Simplex")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sphere_overview(stages: list[Marginal], out_path: Path) -> None:
    colors = {0: "#666666", 1: "#1f77b4", 2: "#d62728"}
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(len(stages))]

    for ax, marginal in zip(axes, stages):
        sphere = np.sqrt(np.clip(marginal.simplex, 1e-8, None))
        for branch_id in np.unique(marginal.branch):
            mask = marginal.branch == branch_id
            ax.scatter(
                sphere[mask, 0], sphere[mask, 1], sphere[mask, 2],
                s=10, alpha=0.75, c=colors[int(branch_id)], depthshade=False,
            )
        ax.set_title(marginal.name)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_xlabel("sqrt(p1)")
        ax.set_ylabel("sqrt(p2)")
        ax.set_zlabel("sqrt(p3)")
        ax.view_init(elev=24, azim=40)

    fig.suptitle("Positive-Orthant Sphere Embedding x = sqrt(p)", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def coupling_cross_branch_rate(src_branch: np.ndarray, tgt_branch: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> float:
    src_lbl = src_branch[src_idx]
    tgt_lbl = tgt_branch[tgt_idx]
    valid = (src_lbl > 0) & (tgt_lbl > 0)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(src_lbl[valid] != tgt_lbl[valid]))


def plot_cost_and_couplings(
    stages: list[Marginal],
    src: Marginal,
    tgt: Marginal,
    out_path: Path,
    sample_pairs: int,
    seed: int,
) -> None:
    src_xy = simplex_to_xy(src.simplex)
    tgt_xy = simplex_to_xy(tgt.simplex)

    src_t = normalize_sphere(to_orthant(torch.tensor(src.simplex, dtype=torch.float32)))
    tgt_t = normalize_sphere(to_orthant(torch.tensor(tgt.simplex, dtype=torch.float32)))

    cost_sphere = compute_sphere_cost_matrix(src_t, tgt_t)
    all_stage_tensors = [
        normalize_sphere(to_orthant(torch.tensor(m.simplex, dtype=torch.float32)))
        for m in stages
    ]
    global_embeds = compute_global_biharmonic_embedding(all_stage_tensors)
    src_embed = global_embeds[3]
    tgt_embed = global_embeds[4]
    cost_biharm = cdist(src_embed, tgt_embed, metric="sqeuclidean").astype(np.float32, copy=False)

    src_order = branch_sort_key(src_xy, src.branch)
    tgt_order = branch_sort_key(tgt_xy, tgt.branch)
    sphere_sorted = cost_sphere[np.ix_(src_order, tgt_order)]
    biharm_sorted = cost_biharm[np.ix_(src_order, tgt_order)]

    np.random.seed(seed)
    sphere_src_idx, sphere_tgt_idx = ot_coupling(cost_sphere, sample_pairs)
    np.random.seed(seed)
    biharm_src_idx, biharm_tgt_idx = ot_coupling(cost_biharm, sample_pairs)

    sphere_cross = coupling_cross_branch_rate(src.branch, tgt.branch, sphere_src_idx, sphere_tgt_idx)
    biharm_cross = coupling_cross_branch_rate(src.branch, tgt.branch, biharm_src_idx, biharm_tgt_idx)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_cost_sphere = fig.add_subplot(gs[0, 0])
    ax_cost_biharm = fig.add_subplot(gs[0, 1])
    ax_cpl_sphere = fig.add_subplot(gs[1, 0])
    ax_cpl_biharm = fig.add_subplot(gs[1, 1])

    im0 = ax_cost_sphere.imshow(sphere_sorted, aspect="auto", cmap="viridis")
    ax_cost_sphere.set_title("Sphere / Great-Circle OT Cost")
    ax_cost_sphere.set_xlabel("target points (ordered by branch)")
    ax_cost_sphere.set_ylabel("source points (ordered by branch)")
    fig.colorbar(im0, ax=ax_cost_sphere, fraction=0.046, pad=0.04)

    im1 = ax_cost_biharm.imshow(biharm_sorted, aspect="auto", cmap="viridis")
    ax_cost_biharm.set_title("Global Biharmonic OT Cost")
    ax_cost_biharm.set_xlabel("target points (ordered by branch)")
    ax_cost_biharm.set_ylabel("source points (ordered by branch)")
    fig.colorbar(im1, ax=ax_cost_biharm, fraction=0.046, pad=0.04)

    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [0.0, 0.0],
    ])
    colors = {1: "#1f77b4", 2: "#d62728"}

    for ax, src_idx, tgt_idx, title, cross_rate in [
        (ax_cpl_sphere, sphere_src_idx, sphere_tgt_idx, "OT Samples Under Sphere Cost", sphere_cross),
        (ax_cpl_biharm, biharm_src_idx, biharm_tgt_idx, "OT Samples Under Global Biharmonic Cost", biharm_cross),
    ]:
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.0)
        ax.scatter(src_xy[:, 0], src_xy[:, 1], s=10, c="#8c8c8c", alpha=0.35, edgecolor="none", label="t3 source")
        for branch_id in (1, 2):
            mask = tgt.branch == branch_id
            ax.scatter(
                tgt_xy[mask, 0], tgt_xy[mask, 1],
                s=11, c=colors[branch_id], alpha=0.55, edgecolor="none",
            )
        for s, t in zip(src_idx, tgt_idx):
            branch = int(tgt.branch[t])
            color = colors.get(branch, "#444444")
            ax.plot(
                [src_xy[s, 0], tgt_xy[t, 0]],
                [src_xy[s, 1], tgt_xy[t, 1]],
                color=color,
                alpha=0.16,
                linewidth=0.8,
            )
        ax.set_title(f"{title}\nCross-branch sampled pairs: {100 * cross_rate:.1f}%")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")

    fig.suptitle(
        "Why Global Biharmonic Cost Can Help on a Branching Simplex/Sphere Toy",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"{src.name} -> {tgt.name}")
    print(f"  sphere sampled cross-branch rate     : {100 * sphere_cross:.2f}%")
    print(f"  biharmonic sampled cross-branch rate : {100 * biharm_cross:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-stage", type=int, default=300)
    parser.add_argument("--kappa", type=float, default=220.0)
    parser.add_argument("--sample-pairs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="plots/toy_branching_simplex")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = make_branching_dataset(
        n_per_stage=args.n_per_stage,
        kappa=args.kappa,
        seed=args.seed,
    )

    np.savez_compressed(
        out_dir / "toy_branching_data.npz",
        **{m.name: m.simplex for m in stages},
        **{f"{m.name}_branch": m.branch for m in stages},
    )

    plot_simplex_overview(stages, out_dir / "toy_branching_simplex_overview.png")
    plot_simplex_all_steps(stages, out_dir / "toy_branching_simplex_all_steps.png")
    plot_sphere_overview(stages, out_dir / "toy_branching_sphere_overview.png")
    plot_cost_and_couplings(
        stages,
        stages[3],
        stages[4],
        out_dir / "toy_branching_costs_and_couplings.png",
        sample_pairs=args.sample_pairs,
        seed=args.seed,
    )

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
