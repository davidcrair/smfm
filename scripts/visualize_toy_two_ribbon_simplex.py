#!/usr/bin/env python
"""Two-ribbon simplex/sphere toy where local biharmonic can beat sphere cost.

The source and target marginals each lie on two nearby curved ribbons inside the
2-simplex. True transport moves forward along the same ribbon, but ambient
sphere/geodesic distance is tempted to shortcut across ribbons because the lane
gap is smaller than the along-ribbon displacement. Local biharmonic distance,
built on the union of the two marginals only, can stay branch-aware.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from surf.geometry.sphere import compute_sphere_cost_matrix, normalize_sphere, to_orthant
from surf.ot.costs import compute_biharmonic_cost_matrix
from surf.ot.coupling import ot_coupling


SQRT3 = np.sqrt(3.0)


@dataclass
class RibbonMarginal:
    name: str
    simplex: np.ndarray
    branch: np.ndarray
    u: np.ndarray


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-8, None)
    return p / p.sum()


def simplex_to_xy(p: np.ndarray) -> np.ndarray:
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
    ])
    return p @ vertices


def _curve_center(u: np.ndarray) -> np.ndarray:
    p1 = 0.64 - 0.12 * u + 0.02 * np.sin(2 * np.pi * u)
    p2 = 0.16 + 0.52 * u
    p3 = 1.0 - p1 - p2
    center = np.stack([p1, p2, p3], axis=1)
    center = np.clip(center, 1e-8, None)
    center /= center.sum(axis=1, keepdims=True)
    return center


def _simplex_normal(center: np.ndarray) -> np.ndarray:
    # Direction within the simplex plane (sum = 0), chosen to separate the
    # two ribbons mostly in the p2/p3 coordinates.
    base = np.tile(np.array([0.0, 1.0, -1.0]), (len(center), 1))
    base -= (base * center).sum(axis=1, keepdims=True) * center
    base /= np.linalg.norm(base, axis=1, keepdims=True).clip(min=1e-8)
    return base


def _dirichlet_cloud(centers: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    alpha = np.clip(kappa * centers, 1e-3, None)
    return np.stack([rng.dirichlet(a) for a in alpha], axis=0)


def make_two_ribbon_dataset(
    n_per_ribbon: int = 250,
    kappa: float = 450.0,
    ribbon_eps: float = 0.040,
    delta_u: float = 0.14,
    seed: int = 42,
) -> tuple[RibbonMarginal, RibbonMarginal]:
    """Create source/target marginals with two nearby curved ribbons."""
    rng = np.random.default_rng(seed)

    u_src_left = rng.uniform(0.10, 0.72, size=n_per_ribbon)
    u_src_right = rng.uniform(0.10, 0.72, size=n_per_ribbon)
    u_tgt_left = np.clip(u_src_left + delta_u, 0.18, 0.92)
    u_tgt_right = np.clip(u_src_right + delta_u, 0.18, 0.92)

    def ribbon_points(u: np.ndarray, sign: float) -> np.ndarray:
        center = _curve_center(u)
        normal = _simplex_normal(center)
        pts = center + sign * ribbon_eps * normal
        pts = np.clip(pts, 1e-8, None)
        pts /= pts.sum(axis=1, keepdims=True)
        return pts

    src_left = ribbon_points(u_src_left, +1.0)
    src_right = ribbon_points(u_src_right, -1.0)
    tgt_left = ribbon_points(u_tgt_left, +1.0)
    tgt_right = ribbon_points(u_tgt_right, -1.0)

    src = RibbonMarginal(
        "t0_source",
        _dirichlet_cloud(np.vstack([src_left, src_right]), kappa, rng),
        np.concatenate([np.ones(n_per_ribbon, dtype=np.int64), np.full(n_per_ribbon, 2, dtype=np.int64)]),
        np.concatenate([u_src_left, u_src_right]),
    )
    tgt = RibbonMarginal(
        "t1_target",
        _dirichlet_cloud(np.vstack([tgt_left, tgt_right]), kappa, rng),
        np.concatenate([np.ones(n_per_ribbon, dtype=np.int64), np.full(n_per_ribbon, 2, dtype=np.int64)]),
        np.concatenate([u_tgt_left, u_tgt_right]),
    )
    return src, tgt


def _branch_sort_key(xy: np.ndarray, branch: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.lexsort((u, branch))


def _cross_branch_rate(src_branch: np.ndarray, tgt_branch: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> float:
    return float(np.mean(src_branch[src_idx] != tgt_branch[tgt_idx]))


def plot_dataset(src: RibbonMarginal, tgt: RibbonMarginal, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [0.0, 0.0],
    ])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.2)

    for marginal, color in [(src, "#1f77b4"), (tgt, "#d62728")]:
        xy = simplex_to_xy(marginal.simplex)
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=14, alpha=0.50, c=color, edgecolor="none", label=marginal.name,
        )

    ax.set_title("Two-Ribbon Toy Dataset on the 2-Simplex")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_costs_and_couplings(
    src: RibbonMarginal,
    tgt: RibbonMarginal,
    out_path: Path,
    sample_pairs: int,
    seed: int,
    knn: int,
    n_eig: int,
) -> tuple[float, float, float, float]:
    src_xy = simplex_to_xy(src.simplex)
    tgt_xy = simplex_to_xy(tgt.simplex)

    src_t = normalize_sphere(to_orthant(torch.tensor(src.simplex, dtype=torch.float32)))
    tgt_t = normalize_sphere(to_orthant(torch.tensor(tgt.simplex, dtype=torch.float32)))

    cost_sphere = compute_sphere_cost_matrix(src_t, tgt_t)
    cost_biharm = compute_biharmonic_cost_matrix(src_t, tgt_t, knn=knn, n_eig=n_eig, weight_power=0.5)

    src_order = _branch_sort_key(src_xy, src.branch, src.u)
    tgt_order = _branch_sort_key(tgt_xy, tgt.branch, tgt.u)
    sphere_sorted = cost_sphere[np.ix_(src_order, tgt_order)]
    biharm_sorted = cost_biharm[np.ix_(src_order, tgt_order)]

    np.random.seed(seed)
    sphere_src_idx, sphere_tgt_idx = ot_coupling(cost_sphere, sample_pairs)
    np.random.seed(seed)
    biharm_src_idx, biharm_tgt_idx = ot_coupling(cost_biharm, sample_pairs)

    sphere_cross = _cross_branch_rate(src.branch, tgt.branch, sphere_src_idx, sphere_tgt_idx)
    biharm_cross = _cross_branch_rate(src.branch, tgt.branch, biharm_src_idx, biharm_tgt_idx)
    sphere_argmin_cross = _cross_branch_rate(src.branch, tgt.branch, np.arange(len(src.branch)), cost_sphere.argmin(axis=1))
    biharm_argmin_cross = _cross_branch_rate(src.branch, tgt.branch, np.arange(len(src.branch)), cost_biharm.argmin(axis=1))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_cost_sphere = fig.add_subplot(gs[0, 0])
    ax_cost_biharm = fig.add_subplot(gs[0, 1])
    ax_cpl_sphere = fig.add_subplot(gs[1, 0])
    ax_cpl_biharm = fig.add_subplot(gs[1, 1])

    im0 = ax_cost_sphere.imshow(sphere_sorted, aspect="auto", cmap="viridis")
    ax_cost_sphere.set_title("Sphere / Great-Circle OT Cost")
    ax_cost_sphere.set_xlabel("target points (sorted by ribbon and arclength)")
    ax_cost_sphere.set_ylabel("source points (sorted by ribbon and arclength)")
    fig.colorbar(im0, ax=ax_cost_sphere, fraction=0.046, pad=0.04)

    im1 = ax_cost_biharm.imshow(biharm_sorted, aspect="auto", cmap="viridis")
    ax_cost_biharm.set_title("Local Biharmonic OT Cost")
    ax_cost_biharm.set_xlabel("target points (sorted by ribbon and arclength)")
    ax_cost_biharm.set_ylabel("source points (sorted by ribbon and arclength)")
    fig.colorbar(im1, ax=ax_cost_biharm, fraction=0.046, pad=0.04)

    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
        [0.0, 0.0],
    ])
    ribbon_colors = {1: "#1f77b4", 2: "#d62728"}

    for ax, src_idx, tgt_idx, title, cross_rate in [
        (ax_cpl_sphere, sphere_src_idx, sphere_tgt_idx, "OT Samples Under Sphere Cost", sphere_cross),
        (ax_cpl_biharm, biharm_src_idx, biharm_tgt_idx, "OT Samples Under Local Biharmonic Cost", biharm_cross),
    ]:
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.0)
        ax.scatter(src_xy[:, 0], src_xy[:, 1], s=10, c="#8c8c8c", alpha=0.35, edgecolor="none")
        for branch_id in (1, 2):
            mask = tgt.branch == branch_id
            ax.scatter(
                tgt_xy[mask, 0], tgt_xy[mask, 1],
                s=11, c=ribbon_colors[branch_id], alpha=0.55, edgecolor="none",
            )
        for s, t in zip(src_idx, tgt_idx):
            color = ribbon_colors[int(tgt.branch[t])]
            ax.plot(
                [src_xy[s, 0], tgt_xy[t, 0]],
                [src_xy[s, 1], tgt_xy[t, 1]],
                color=color,
                alpha=0.14,
                linewidth=0.8,
            )
        ax.set_title(f"{title}\nCross-ribbon sampled pairs: {100 * cross_rate:.1f}%")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")

    fig.suptitle("Two-Ribbon Toy: Local Biharmonic vs Sphere Cost", fontsize=15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return sphere_cross, biharm_cross, sphere_argmin_cross, biharm_argmin_cross


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-ribbon", type=int, default=250)
    parser.add_argument("--kappa", type=float, default=450.0)
    parser.add_argument("--ribbon-eps", type=float, default=0.040)
    parser.add_argument("--delta-u", type=float, default=0.14)
    parser.add_argument("--sample-pairs", type=int, default=180)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--n-eig", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="plots/toy_two_ribbon_simplex")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src, tgt = make_two_ribbon_dataset(
        n_per_ribbon=args.n_per_ribbon,
        kappa=args.kappa,
        ribbon_eps=args.ribbon_eps,
        delta_u=args.delta_u,
        seed=args.seed,
    )

    np.savez_compressed(
        out_dir / "toy_two_ribbon_data.npz",
        t0_source=src.simplex,
        t1_target=tgt.simplex,
        t0_source_branch=src.branch,
        t1_target_branch=tgt.branch,
        t0_source_u=src.u,
        t1_target_u=tgt.u,
    )

    plot_dataset(src, tgt, out_dir / "toy_two_ribbon_simplex.png")
    sphere_cross, biharm_cross, sphere_argmin_cross, biharm_argmin_cross = plot_costs_and_couplings(
        src,
        tgt,
        out_dir / "toy_two_ribbon_costs_and_couplings.png",
        sample_pairs=args.sample_pairs,
        seed=args.seed,
        knn=args.knn,
        n_eig=args.n_eig,
    )

    print(f"{src.name} -> {tgt.name}")
    print(f"  sphere argmin cross-ribbon rate      : {100 * sphere_argmin_cross:.2f}%")
    print(f"  local biharmonic argmin cross-rate   : {100 * biharm_argmin_cross:.2f}%")
    print(f"  sphere sampled cross-ribbon rate     : {100 * sphere_cross:.2f}%")
    print(f"  local biharmonic sampled cross-rate  : {100 * biharm_cross:.2f}%")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
