#!/usr/bin/env python
"""Visualize a two-horseshoe toy dataset on the simplex and positive orthant.

This is only a dataset/geometry visualization. It does not run flow matching.

Construction:
- Two disjoint horseshoe-shaped curves in 2D, embedded inside the 2-simplex.
- Each horseshoe is split into a source half (t=0) and target half (t=1).
- The inner tip of one horseshoe is intentionally close to the inner tip of the
  other in ambient geometry, setting up the intended ambiguity for later OT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
class Marginal:
    name: str
    simplex: np.ndarray
    horseshoe: np.ndarray
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


def _rotation_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _horseshoe_points(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    u: np.ndarray,
    rotation: float = 0.0,
) -> np.ndarray:
    """Rotated lower semicircle traversal from right tip -> bottom -> left tip."""
    theta = -np.pi * u
    rel = np.stack([radius_x * np.cos(theta), radius_y * np.sin(theta)], axis=1)
    if rotation != 0.0:
        rel = rel @ _rotation_matrix(rotation).T
    return rel + np.array([center_x, center_y], dtype=np.float64)[None, :]


def _horseshoe_tangent(u: np.ndarray, radius_x: float, radius_y: float, rotation: float = 0.0) -> np.ndarray:
    theta = -np.pi * u
    dtheta_du = -np.pi
    tangent = np.stack(
        [
            -radius_x * np.sin(theta) * dtheta_du,
            radius_y * np.cos(theta) * dtheta_du,
        ],
        axis=1,
    )
    if rotation != 0.0:
        tangent = tangent @ _rotation_matrix(rotation).T
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1e-8)
    return tangent


def _sample_horseshoe_half(
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    rotation: float,
    u_min: float,
    u_max: float,
    n: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(u_min, u_max, size=n)
    xy = _horseshoe_points(center_x, center_y, radius_x, radius_y, u, rotation=rotation)
    tangent = _horseshoe_tangent(u, radius_x, radius_y, rotation=rotation)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    xy = (
        xy
        + rng.normal(scale=noise_scale, size=(n, 1)) * normal
        + rng.normal(scale=0.35 * noise_scale, size=(n, 1)) * tangent
    )
    simplex = xy_to_simplex(xy)
    xy = simplex_to_xy(simplex)
    return simplex, u, xy


def make_horseshoe_dataset(
    n_per_half: int = 260,
    noise_scale: float = 0.010,
    seed: int = 42,
) -> tuple[Marginal, Marginal]:
    """Build two side-by-side horseshoes inside the simplex triangle.

    Horseshoe A: centered left. Its inner tip is the right tip.
    Horseshoe B: centered right. Its inner tip is the left tip.

    Time orientation is chosen so that:
    - the inner tip of A belongs to t=0
    - the inner tip of B belongs to t=1
    """
    rng = np.random.default_rng(seed)

    # Geometry chosen to stay well inside the simplex while making the two
    # inner tips nearly touch.
    left_center_x = 0.49
    right_center_x = 0.51
    left_center_y = 0.29
    right_center_y = 0.38
    radius_x = 0.10
    radius_y = 0.14
    left_rotation = +1.5 * np.pi
    right_rotation = +0.5 * np.pi

    # Horseshoe A: switch back to the original time orientation.
    a_t0_simplex, a_t0_u, a_t0_xy = _sample_horseshoe_half(
        left_center_x, left_center_y, radius_x, radius_y, left_rotation,
        u_min=0.02, u_max=0.48, n=n_per_half, noise_scale=noise_scale, rng=rng,
    )
    a_t1_simplex, a_t1_u, a_t1_xy = _sample_horseshoe_half(
        left_center_x, left_center_y, radius_x, radius_y, left_rotation,
        u_min=0.52, u_max=0.98, n=n_per_half, noise_scale=noise_scale, rng=rng,
    )

    # Horseshoe B: keep the original time orientation.
    b_t1_simplex, b_t1_u, b_t1_xy = _sample_horseshoe_half(
        right_center_x, right_center_y, radius_x, radius_y, right_rotation,
        u_min=0.52, u_max=0.98, n=n_per_half, noise_scale=noise_scale, rng=rng,
    )
    b_t0_simplex, b_t0_u, b_t0_xy = _sample_horseshoe_half(
        right_center_x, right_center_y, radius_x, radius_y, right_rotation,
        u_min=0.02, u_max=0.48, n=n_per_half, noise_scale=noise_scale, rng=rng,
    )

    t0_simplex = np.vstack([a_t0_simplex, b_t0_simplex])
    t1_simplex = np.vstack([a_t1_simplex, b_t1_simplex])
    t0_xy = np.vstack([a_t0_xy, b_t0_xy])
    t1_xy = np.vstack([a_t1_xy, b_t1_xy])
    t0_u = np.concatenate([a_t0_u, b_t0_u])
    t1_u = np.concatenate([a_t1_u, b_t1_u])
    t0_horseshoe = np.concatenate([np.ones(n_per_half, dtype=np.int64), np.full(n_per_half, 2, dtype=np.int64)])
    t1_horseshoe = np.concatenate([np.ones(n_per_half, dtype=np.int64), np.full(n_per_half, 2, dtype=np.int64)])

    return (
        Marginal("t0_source", t0_simplex, t0_horseshoe, t0_u, t0_xy),
        Marginal("t1_target", t1_simplex, t1_horseshoe, t1_u, t1_xy),
    )


def plot_simplex(t0: Marginal, t1: Marginal, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.2)

    horseshoe_colors = {1: "#1f77b4", 2: "#2ca02c"}
    for marginal, marker, alpha in [(t0, "o", 0.62), (t1, "^", 0.62)]:
        for horseshoe_id in (1, 2):
            mask = marginal.horseshoe == horseshoe_id
            ax.scatter(
                marginal.xy[mask, 0],
                marginal.xy[mask, 1],
                s=16,
                marker=marker,
                alpha=alpha,
                c=horseshoe_colors[horseshoe_id],
                edgecolor="none",
                label=f"{marginal.name} / horseshoe {horseshoe_id}" if horseshoe_id == 1 else None,
            )

    ax.set_title("Two Interlocking Horseshoes on the 2-Simplex")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_time_overlay(t0: Marginal, t1: Marginal, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.2)

    ax.scatter(t0.xy[:, 0], t0.xy[:, 1], s=14, c="#1f77b4", alpha=0.55, edgecolor="none", label="t=0")
    ax.scatter(t1.xy[:, 0], t1.xy[:, 1], s=14, c="#d62728", alpha=0.55, edgecolor="none", label="t=1")

    ax.set_title("Time Overlay on the 2-Simplex")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_orthant(t0: Marginal, t1: Marginal, out_path: Path) -> None:
    fig = plt.figure(figsize=(10.5, 8.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    horseshoe_colors = {1: "#1f77b4", 2: "#2ca02c"}
    for marginal, marker, alpha in [(t0, "o", 0.58), (t1, "^", 0.58)]:
        sphere = np.sqrt(np.clip(marginal.simplex, 1e-8, None))
        for horseshoe_id in (1, 2):
            mask = marginal.horseshoe == horseshoe_id
            ax.scatter(
                sphere[mask, 0],
                sphere[mask, 1],
                sphere[mask, 2],
                s=16,
                marker=marker,
                alpha=alpha,
                c=horseshoe_colors[horseshoe_id],
                depthshade=False,
            )

    ax.set_title("Two Interlocking Horseshoes on the Positive Orthant of $S^2$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.view_init(elev=26, azim=36)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-half", type=int, default=260)
    parser.add_argument("--noise-scale", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="plots/toy_horseshoes_simplex")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0, t1 = make_horseshoe_dataset(
        n_per_half=args.n_per_half,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    np.savez_compressed(
        out_dir / "toy_horseshoes_dataset.npz",
        t0_source=t0.simplex,
        t1_target=t1.simplex,
        t0_horseshoe=t0.horseshoe,
        t1_horseshoe=t1.horseshoe,
        t0_u=t0.u,
        t1_u=t1.u,
    )

    plot_simplex(t0, t1, out_dir / "toy_horseshoes_simplex.png")
    plot_time_overlay(t0, t1, out_dir / "toy_horseshoes_time_overlay.png")
    plot_orthant(t0, t1, out_dir / "toy_horseshoes_orthant.png")

    print("Saved horseshoe toy plots to", out_dir)


if __name__ == "__main__":
    main()
