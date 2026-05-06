#!/usr/bin/env python
"""Curved-X simplex/sphere toy for local biharmonic vs sphere OT cost.

The toy lives on the 2-simplex (visualized as a triangle) and consists of two
disjoint curved diagonal lanes:

- branch 1: bottom-left -> top-right
- branch 2: bottom-right -> top-left

The source marginal t=0 occupies the lower half of each lane, while the target
marginal t=1 occupies the upper half. In direct ambient geometry, points are
often closer to the wrong "same-side" target blob. A local biharmonic cost,
built on the union of only these two marginals, can instead prefer travel along
the sampled lane geometry.
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
TRIANGLE_VERTICES = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2.0],
    ],
    dtype=np.float64,
)


@dataclass
class ToyMarginal:
    name: str
    simplex: np.ndarray
    branch: np.ndarray
    u: np.ndarray
    xy: np.ndarray


def simplex_to_xy(p: np.ndarray) -> np.ndarray:
    return p @ TRIANGLE_VERTICES


def xy_to_simplex(xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    p3 = y / (SQRT3 / 2.0)
    p2 = x - 0.5 * p3
    p1 = 1.0 - p2 - p3
    simplex = np.stack([p1, p2, p3], axis=1)
    simplex = np.clip(simplex, 1e-8, None)
    simplex /= simplex.sum(axis=1, keepdims=True)
    return simplex


def _bezier(control_points: np.ndarray, u: np.ndarray) -> np.ndarray:
    one_minus = 1.0 - u[:, None]
    return (
        (one_minus ** 2) * control_points[0][None, :]
        + 2.0 * one_minus * u[:, None] * control_points[1][None, :]
        + (u[:, None] ** 2) * control_points[2][None, :]
    )


def _curve_tangent(control_points: np.ndarray, u: np.ndarray) -> np.ndarray:
    p0, p1, p2 = control_points
    tangent = 2.0 * (1.0 - u[:, None]) * (p1 - p0)[None, :] + 2.0 * u[:, None] * (p2 - p1)[None, :]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1e-8)
    return tangent


def _curve_normal(control_points: np.ndarray, u: np.ndarray) -> np.ndarray:
    tangent = _curve_tangent(control_points, u)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(min=1e-8)
    return normal


def _sample_lane(
    control_points: np.ndarray,
    u_min: float,
    u_max: float,
    n: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(u_min, u_max, size=n)
    center = _bezier(control_points, u)
    normal = _curve_normal(control_points, u)
    tangent = _curve_tangent(control_points, u)
    noise_normal = rng.normal(scale=noise_scale, size=(n, 1))
    noise_tangent = rng.normal(scale=0.45 * noise_scale, size=(n, 1))
    xy = center + noise_normal * normal + noise_tangent * tangent
    simplex = xy_to_simplex(xy)
    xy = simplex_to_xy(simplex)
    return simplex, u


def make_curved_x_dataset(
    n_per_branch: int = 260,
    noise_scale: float = 0.020,
    gap: float = 0.10,
    source_u_max: float = 0.42,
    target_u_min: float = 0.58,
    seed: int = 42,
) -> tuple[ToyMarginal, ToyMarginal]:
    """Build a two-marginal curved-X toy inside the simplex."""
    rng = np.random.default_rng(seed)

    # Two diagonal lanes that come close near the center but remain disjoint.
    branch1 = np.array(
        [
            [0.18, 0.12],
            [0.46, 0.52 + 0.5 * gap],
            [0.75, 0.67],
        ],
        dtype=np.float64,
    )
    branch2 = np.array(
        [
            [0.82, 0.12],
            [0.54, 0.34 - 0.5 * gap],
            [0.25, 0.67],
        ],
        dtype=np.float64,
    )

    src1, u_src1 = _sample_lane(branch1, 0.02, source_u_max, n_per_branch, noise_scale, rng)
    src2, u_src2 = _sample_lane(branch2, 0.02, source_u_max, n_per_branch, noise_scale, rng)
    tgt1, u_tgt1 = _sample_lane(branch1, target_u_min, 0.98, n_per_branch, noise_scale, rng)
    tgt2, u_tgt2 = _sample_lane(branch2, target_u_min, 0.98, n_per_branch, noise_scale, rng)

    src_simplex = np.vstack([src1, src2])
    tgt_simplex = np.vstack([tgt1, tgt2])
    src_branch = np.concatenate(
        [
            np.ones(n_per_branch, dtype=np.int64),
            np.full(n_per_branch, 2, dtype=np.int64),
        ]
    )
    tgt_branch = np.concatenate(
        [
            np.ones(n_per_branch, dtype=np.int64),
            np.full(n_per_branch, 2, dtype=np.int64),
        ]
    )
    src_u = np.concatenate([u_src1, u_src2])
    tgt_u = np.concatenate([u_tgt1, u_tgt2])

    return (
        ToyMarginal("t0_source", src_simplex, src_branch, src_u, simplex_to_xy(src_simplex)),
        ToyMarginal("t1_target", tgt_simplex, tgt_branch, tgt_u, simplex_to_xy(tgt_simplex)),
    )


def _sort_key(branch: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.lexsort((u, branch))


def _confusion_rate(src_branch: np.ndarray, tgt_branch: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> float:
    return float(np.mean(src_branch[src_idx] != tgt_branch[tgt_idx]))


def plot_dataset(src: ToyMarginal, tgt: ToyMarginal, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6), constrained_layout=True)
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.2)

    ax.scatter(src.xy[:, 0], src.xy[:, 1], s=14, c="#1f77b4", alpha=0.55, edgecolor="none", label="t=0")
    ax.scatter(tgt.xy[:, 0], tgt.xy[:, 1], s=14, c="#d62728", alpha=0.55, edgecolor="none", label="t=1")

    ax.set_title("Curved-X Toy Dataset on the 2-Simplex")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")
    ax.legend(loc="upper center", frameon=False, ncol=2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_branch_overlay(src: ToyMarginal, tgt: ToyMarginal, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.6), constrained_layout=True)
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.2)

    branch_colors = {1: "#1f77b4", 2: "#2ca02c"}
    for branch_id, marker, marginal in [
        (1, "o", src),
        (2, "o", src),
        (1, "^", tgt),
        (2, "^", tgt),
    ]:
        mask = marginal.branch == branch_id
        ax.scatter(
            marginal.xy[mask, 0],
            marginal.xy[mask, 1],
            s=14,
            alpha=0.55,
            c=branch_colors[branch_id],
            marker=marker,
            edgecolor="none",
        )

    ax.set_title("Curved-X Toy by Lane Identity")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_costs_and_couplings(
    src: ToyMarginal,
    tgt: ToyMarginal,
    out_path: Path,
    sample_pairs: int,
    seed: int,
    knn: int,
    n_eig: int,
    weight_power: float,
) -> dict[str, float]:
    src_t = normalize_sphere(to_orthant(torch.tensor(src.simplex, dtype=torch.float32)))
    tgt_t = normalize_sphere(to_orthant(torch.tensor(tgt.simplex, dtype=torch.float32)))

    cost_sphere = compute_sphere_cost_matrix(src_t, tgt_t)
    cost_biharm = compute_biharmonic_cost_matrix(src_t, tgt_t, knn=knn, n_eig=n_eig, weight_power=weight_power)

    src_order = _sort_key(src.branch, src.u)
    tgt_order = _sort_key(tgt.branch, tgt.u)
    sphere_sorted = cost_sphere[np.ix_(src_order, tgt_order)]
    biharm_sorted = cost_biharm[np.ix_(src_order, tgt_order)]

    np.random.seed(seed)
    sphere_src_idx, sphere_tgt_idx = ot_coupling(cost_sphere, sample_pairs)
    np.random.seed(seed)
    biharm_src_idx, biharm_tgt_idx = ot_coupling(cost_biharm, sample_pairs)

    metrics = {
        "sphere_argmin_confusion": _confusion_rate(src.branch, tgt.branch, np.arange(len(src.branch)), cost_sphere.argmin(axis=1)),
        "biharm_argmin_confusion": _confusion_rate(src.branch, tgt.branch, np.arange(len(src.branch)), cost_biharm.argmin(axis=1)),
        "sphere_sampled_confusion": _confusion_rate(src.branch, tgt.branch, sphere_src_idx, sphere_tgt_idx),
        "biharm_sampled_confusion": _confusion_rate(src.branch, tgt.branch, biharm_src_idx, biharm_tgt_idx),
    }

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_cost_sphere = fig.add_subplot(gs[0, 0])
    ax_cost_biharm = fig.add_subplot(gs[0, 1])
    ax_cpl_sphere = fig.add_subplot(gs[1, 0])
    ax_cpl_biharm = fig.add_subplot(gs[1, 1])

    im0 = ax_cost_sphere.imshow(sphere_sorted, aspect="auto", cmap="viridis")
    ax_cost_sphere.set_title("Sphere / Great-Circle OT Cost")
    ax_cost_sphere.set_xlabel("target points (sorted by lane and arclength)")
    ax_cost_sphere.set_ylabel("source points (sorted by lane and arclength)")
    fig.colorbar(im0, ax=ax_cost_sphere, fraction=0.046, pad=0.04)

    im1 = ax_cost_biharm.imshow(biharm_sorted, aspect="auto", cmap="viridis")
    ax_cost_biharm.set_title("Local Biharmonic OT Cost")
    ax_cost_biharm.set_xlabel("target points (sorted by lane and arclength)")
    ax_cost_biharm.set_ylabel("source points (sorted by lane and arclength)")
    fig.colorbar(im1, ax=ax_cost_biharm, fraction=0.046, pad=0.04)

    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    lane_colors = {1: "#1f77b4", 2: "#2ca02c"}

    for ax, src_idx, tgt_idx, title, confusion_key in [
        (ax_cpl_sphere, sphere_src_idx, sphere_tgt_idx, "OT Samples Under Sphere Cost", "sphere_sampled_confusion"),
        (ax_cpl_biharm, biharm_src_idx, biharm_tgt_idx, "OT Samples Under Local Biharmonic Cost", "biharm_sampled_confusion"),
    ]:
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.0)
        ax.scatter(src.xy[:, 0], src.xy[:, 1], s=10, c="#8c8c8c", alpha=0.30, edgecolor="none")
        for branch_id in (1, 2):
            mask = tgt.branch == branch_id
            ax.scatter(
                tgt.xy[mask, 0],
                tgt.xy[mask, 1],
                s=11,
                c=lane_colors[branch_id],
                alpha=0.55,
                edgecolor="none",
            )
        for s_idx, t_idx in zip(src_idx, tgt_idx):
            ax.plot(
                [src.xy[s_idx, 0], tgt.xy[t_idx, 0]],
                [src.xy[s_idx, 1], tgt.xy[t_idx, 1]],
                color=lane_colors[int(tgt.branch[t_idx])],
                alpha=0.14,
                linewidth=0.8,
            )
        ax.set_title(f"{title}\nCross-lane OT samples: {100 * metrics[confusion_key]:.1f}%")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")

    fig.suptitle("Curved-X Toy: Local Biharmonic vs Sphere Cost", fontsize=15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-branch", type=int, default=260)
    parser.add_argument("--noise-scale", type=float, default=0.020)
    parser.add_argument("--gap", type=float, default=0.10)
    parser.add_argument("--source-u-max", type=float, default=0.42)
    parser.add_argument("--target-u-min", type=float, default=0.58)
    parser.add_argument("--sample-pairs", type=int, default=180)
    parser.add_argument("--knn", type=int, default=12)
    parser.add_argument("--n-eig", type=int, default=40)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="plots/toy_x_simplex")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src, tgt = make_curved_x_dataset(
        n_per_branch=args.n_per_branch,
        noise_scale=args.noise_scale,
        gap=args.gap,
        source_u_max=args.source_u_max,
        target_u_min=args.target_u_min,
        seed=args.seed,
    )

    np.savez_compressed(
        out_dir / "toy_x_dataset.npz",
        t0_source=src.simplex,
        t1_target=tgt.simplex,
        t0_branch=src.branch,
        t1_branch=tgt.branch,
        t0_u=src.u,
        t1_u=tgt.u,
    )
    plot_dataset(src, tgt, out_dir / "toy_x_simplex.png")
    plot_branch_overlay(src, tgt, out_dir / "toy_x_branch_overlay.png")
    metrics = plot_costs_and_couplings(
        src,
        tgt,
        out_dir / "toy_x_costs_and_couplings.png",
        sample_pairs=args.sample_pairs,
        seed=args.seed,
        knn=args.knn,
        n_eig=args.n_eig,
        weight_power=args.weight_power,
    )

    print("Curved-X toy summary")
    print(f"  output dir                       : {out_dir}")
    print(f"  sphere argmin cross-lane rate    : {100 * metrics['sphere_argmin_confusion']:.2f}%")
    print(f"  biharm argmin cross-lane rate    : {100 * metrics['biharm_argmin_confusion']:.2f}%")
    print(f"  sphere OT sample cross-lane rate : {100 * metrics['sphere_sampled_confusion']:.2f}%")
    print(f"  biharm OT sample cross-lane rate : {100 * metrics['biharm_sampled_confusion']:.2f}%")


if __name__ == "__main__":
    main()
