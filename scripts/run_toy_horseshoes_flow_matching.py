#!/usr/bin/env python
"""Run flow matching on the horseshoe toy.

Trains two sphere-based flow-matching models on the same toy dataset:

- standard Fisher flow with sphere/geodesic OT cost
- Fisher flow with local biharmonic OT cost

Outputs:
- endpoint plots on the simplex
- optional multi-marginal stage overlay
- 3D positive-orthant endpoint plots
- a small metrics JSON/NPZ summary

This is intentionally standalone and does not go through the full Hydra
training entrypoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from surf.runtime import setup
from surf.training.flow_trainer import train_multi_marginal_flow
from surf.evaluation.generation import generate_fisher_flow, generate_fisher_flow_trajectory
from surf.geometry.sphere import to_orthant, normalize_sphere, from_orthant, compute_sphere_cost_matrix
from surf.ot.costs import compute_biharmonic_cost_matrix
from surf.ot.coupling import ot_coupling
from surf.evaluation.metrics import mmd_rbf, fgd


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


def subset_marginal(marginal: Marginal, horseshoe_id: int | None) -> Marginal:
    if horseshoe_id is None:
        return marginal
    mask = marginal.horseshoe == horseshoe_id
    return Marginal(
        name=f"{marginal.name}_horse{horseshoe_id}",
        simplex=marginal.simplex[mask],
        horseshoe=marginal.horseshoe[mask],
        u=marginal.u[mask],
        xy=marginal.xy[mask],
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    rng = np.random.default_rng(seed)

    left_center_x = 0.49
    right_center_x = 0.51
    left_center_y = 0.29
    right_center_y = 0.38
    radius_x = 0.10
    radius_y = 0.14
    left_rotation = +1.5 * np.pi
    right_rotation = +0.5 * np.pi

    a_t0_simplex, a_t0_u, a_t0_xy = _sample_horseshoe_half(
        left_center_x, left_center_y, radius_x, radius_y, left_rotation,
        u_min=0.02, u_max=0.48, n=n_per_half, noise_scale=noise_scale, rng=rng,
    )
    a_t1_simplex, a_t1_u, a_t1_xy = _sample_horseshoe_half(
        left_center_x, left_center_y, radius_x, radius_y, left_rotation,
        u_min=0.52, u_max=0.98, n=n_per_half, noise_scale=noise_scale, rng=rng,
    )

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


def _horseshoe_geometry() -> dict[str, float]:
    return {
        "left_center_x": 0.49,
        "right_center_x": 0.51,
        "left_center_y": 0.29,
        "right_center_y": 0.38,
        "radius_x": 0.10,
        "radius_y": 0.14,
        "left_rotation": +1.5 * np.pi,
        "right_rotation": +0.5 * np.pi,
    }


def make_horseshoe_multimarginal_dataset(
    n_per_stage_per_horseshoe: int = 260,
    noise_scale: float = 0.010,
    seed: int = 42,
    n_marginals: int = 5,
    window_width: float = 0.24,
) -> list[Marginal]:
    if n_marginals < 2:
        raise ValueError("n_marginals must be at least 2")

    rng = np.random.default_rng(seed)
    geom = _horseshoe_geometry()
    centers = np.linspace(0.10, 0.90, n_marginals)
    half_width = 0.5 * window_width
    marginals: list[Marginal] = []

    for stage_idx, center in enumerate(centers):
        u_min = max(0.02, center - half_width)
        u_max = min(0.98, center + half_width)
        if u_max - u_min < 0.06:
            slack = 0.03
            u_min = max(0.02, center - slack)
            u_max = min(0.98, center + slack)

        a_simplex, a_u, a_xy = _sample_horseshoe_half(
            geom["left_center_x"], geom["left_center_y"], geom["radius_x"], geom["radius_y"], geom["left_rotation"],
            u_min=u_min, u_max=u_max, n=n_per_stage_per_horseshoe, noise_scale=noise_scale, rng=rng,
        )
        b_simplex, b_u, b_xy = _sample_horseshoe_half(
            geom["right_center_x"], geom["right_center_y"], geom["radius_x"], geom["radius_y"], geom["right_rotation"],
            u_min=u_min, u_max=u_max, n=n_per_stage_per_horseshoe, noise_scale=noise_scale, rng=rng,
        )

        simplex = np.vstack([a_simplex, b_simplex])
        u = np.concatenate([a_u, b_u])
        xy = np.vstack([a_xy, b_xy])
        horseshoe = np.concatenate(
            [np.ones(n_per_stage_per_horseshoe, dtype=np.int64), np.full(n_per_stage_per_horseshoe, 2, dtype=np.int64)]
        )
        marginals.append(Marginal(f"t{stage_idx}", simplex, horseshoe, u, xy))

    return marginals


def train_test_split_indices(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_train = min(max(n_train, 1), n - 1)
    return perm[:n_train], perm[n_train:]


def split_marginal(marginal: Marginal, train_frac: float, seed: int) -> tuple[Marginal, Marginal]:
    train_idx, test_idx = train_test_split_indices(len(marginal.simplex), train_frac, seed)
    train = Marginal(
        f"{marginal.name}_train",
        marginal.simplex[train_idx],
        marginal.horseshoe[train_idx],
        marginal.u[train_idx],
        marginal.xy[train_idx],
    )
    test = Marginal(
        f"{marginal.name}_test",
        marginal.simplex[test_idx],
        marginal.horseshoe[test_idx],
        marginal.u[test_idx],
        marginal.xy[test_idx],
    )
    return train, test


def plot_endpoints(
    t0_test: Marginal,
    t1_test: Marginal,
    pred_by_method: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2 + len(pred_by_method), figsize=(5.2 * (2 + len(pred_by_method)), 5.4), constrained_layout=True)
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])

    def style_axis(ax, title: str) -> None:
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")

    style_axis(axes[0], "t=0 Test Source")
    axes[0].scatter(t0_test.xy[:, 0], t0_test.xy[:, 1], s=14, c="#1f77b4", alpha=0.55, edgecolor="none")

    style_axis(axes[1], "t=1 Test Target")
    axes[1].scatter(t1_test.xy[:, 0], t1_test.xy[:, 1], s=14, c="#d62728", alpha=0.55, edgecolor="none")

    colors = {"sphere": "#444444", "biharmonic": "#e6194b", "premetric_spectral_ot": "#00a08a"}
    for ax, (name, pred_simplex) in zip(axes[2:], pred_by_method.items()):
        pred_xy = simplex_to_xy(pred_simplex)
        style_axis(ax, f"Predicted t=1\n{name}")
        ax.scatter(t1_test.xy[:, 0], t1_test.xy[:, 1], s=10, c="#d9d9d9", alpha=0.30, edgecolor="none")
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=14, c=colors.get(name, "#17becf"), alpha=0.58, edgecolor="none")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_overlay(
    marginals: list[Marginal],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    cmap = plt.get_cmap("viridis")
    stage_colors = [cmap(i / max(len(marginals) - 1, 1)) for i in range(len(marginals))]
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.1)
    for idx, marginal in enumerate(marginals):
        ax.scatter(
            marginal.xy[:, 0],
            marginal.xy[:, 1],
            s=12,
            c=[stage_colors[idx]],
            alpha=0.42,
            edgecolor="none",
            label=f"{marginal.name}",
        )
    ax.set_title("Toy Horseshoe Marginals")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_endpoints_orthant(
    t1_test_simplex: np.ndarray,
    pred_by_method: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(5.4 * (1 + len(pred_by_method)), 5.6), constrained_layout=True)
    n_cols = 1 + len(pred_by_method)
    test_sphere = np.sqrt(np.clip(t1_test_simplex, 1e-8, None))

    colors = {"sphere": "#444444", "biharmonic": "#e6194b", "premetric_spectral_ot": "#00a08a"}
    ax = fig.add_subplot(1, n_cols, 1, projection="3d")
    ax.scatter(test_sphere[:, 0], test_sphere[:, 1], test_sphere[:, 2], s=14, c="#d62728", alpha=0.55, depthshade=False)
    ax.set_title("t=1 Test Target")
    for j in range(3):
        pass
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=26, azim=36)

    for col, (name, pred_simplex) in enumerate(pred_by_method.items(), start=2):
        pred_sphere = np.sqrt(np.clip(pred_simplex, 1e-8, None))
        ax = fig.add_subplot(1, n_cols, col, projection="3d")
        ax.scatter(test_sphere[:, 0], test_sphere[:, 1], test_sphere[:, 2], s=8, c="#d9d9d9", alpha=0.22, depthshade=False)
        ax.scatter(pred_sphere[:, 0], pred_sphere[:, 1], pred_sphere[:, 2], s=14, c=colors.get(name, "#17becf"), alpha=0.55, depthshade=False)
        ax.set_title(f"Predicted t=1\n{name}")
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_zlim(0.0, 1.0)
        ax.view_init(elev=26, azim=36)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _sort_key(horseshoe: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.lexsort((u, horseshoe))


def _cross_rate(src_horse: np.ndarray, tgt_horse: np.ndarray, src_idx: np.ndarray, tgt_idx: np.ndarray) -> float:
    return float(np.mean(src_horse[src_idx] != tgt_horse[tgt_idx]))


def _solve_plan(cost_matrix: np.ndarray) -> np.ndarray:
    import ot

    n, m = cost_matrix.shape
    a = np.ones(n) / n
    b = np.ones(m) / m
    if max(n, m) <= 15000:
        T = ot.emd(a, b, cost_matrix)
    else:
        eps = 0.005 * cost_matrix.max()
        T = ot.sinkhorn(
            a, b, cost_matrix, reg=eps,
            method="sinkhorn_stabilized",
            numItermax=2000, stopThr=1e-7,
        )
    T = np.maximum(T, 0)
    return T / T.sum()


def _log_tangent_at_source(y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
    cos = np.clip(np.sum(y0 * y1, axis=1, keepdims=True), -1.0 + 1e-6, 1.0 - 1e-6)
    omega = np.arccos(cos)
    sin_omega = np.sin(omega).clip(min=1e-8)
    unit_tan = (y1 - cos * y0) / sin_omega
    return omega * unit_tan


def _local_velocity_variance(y: np.ndarray, v: np.ndarray, knn: int) -> np.ndarray:
    dists = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    nn_idx = np.argsort(dists, axis=1)[:, 1:knn + 1]
    var = np.zeros(len(y), dtype=np.float64)
    for i in range(len(y)):
        nbr_v = v[nn_idx[i]]
        mean_v = nbr_v.mean(axis=0, keepdims=True)
        var[i] = np.mean(np.sum((nbr_v - mean_v) ** 2, axis=1))
    return var


def plot_velocity_targets_and_variance(
    t0_train: Marginal,
    t1_train: Marginal,
    out_dir: Path,
    knn_cost: int,
    n_eig: int,
    knn_var: int,
    arrow_scale: float,
) -> dict[str, float]:
    src_sphere = normalize_sphere(to_orthant(torch.tensor(t0_train.simplex, dtype=torch.float32)))
    tgt_sphere = normalize_sphere(to_orthant(torch.tensor(t1_train.simplex, dtype=torch.float32)))
    src_np = src_sphere.numpy()
    tgt_np = tgt_sphere.numpy()

    cost_sphere = compute_sphere_cost_matrix(src_sphere, tgt_sphere)
    cost_biharm = compute_biharmonic_cost_matrix(
        src_sphere, tgt_sphere, knn=knn_cost, n_eig=n_eig, weight_power=0.5
    )

    plan_sphere = _solve_plan(cost_sphere)
    plan_biharm = _solve_plan(cost_biharm)
    tgt_idx_sphere = plan_sphere.argmax(axis=1)
    tgt_idx_biharm = plan_biharm.argmax(axis=1)

    v_sphere = _log_tangent_at_source(src_np, tgt_np[tgt_idx_sphere])
    v_biharm = _log_tangent_at_source(src_np, tgt_np[tgt_idx_biharm])

    # Visualize target vectors by taking a small step on the sphere and mapping
    # both points back to simplex coordinates.
    step_sphere = normalize_sphere(torch.tensor(src_np + arrow_scale * v_sphere, dtype=torch.float32)).numpy()
    step_biharm = normalize_sphere(torch.tensor(src_np + arrow_scale * v_biharm, dtype=torch.float32)).numpy()
    src_xy = simplex_to_xy(t0_train.simplex)
    sphere_step_xy = simplex_to_xy(from_orthant(torch.tensor(step_sphere)).numpy())
    biharm_step_xy = simplex_to_xy(from_orthant(torch.tensor(step_biharm)).numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), constrained_layout=True)
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    for ax, step_xy, title in [
        (axes[0], sphere_step_xy, "Target Vectors at Source\nSphere OT"),
        (axes[1], biharm_step_xy, "Target Vectors at Source\nLocal Biharmonic OT"),
    ]:
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.1)
        ax.scatter(src_xy[:, 0], src_xy[:, 1], s=10, c="#8c8c8c", alpha=0.28, edgecolor="none")
        for i in range(len(src_xy)):
            ax.plot(
                [src_xy[i, 0], step_xy[i, 0]],
                [src_xy[i, 1], step_xy[i, 1]],
                color="#1f77b4" if ax is axes[0] else "#e6194b",
                alpha=0.16,
                linewidth=0.8,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")
    fig.savefig(out_dir / "toy_horseshoes_velocity_targets.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    var_sphere = _local_velocity_variance(src_np, v_sphere, knn=knn_var)
    var_biharm = _local_velocity_variance(src_np, v_biharm, knn=knn_var)

    vmax = max(float(np.quantile(var_sphere, 0.95)), float(np.quantile(var_biharm, 0.95)), 1e-8)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), constrained_layout=True)
    for ax, var, title in [
        (axes[0, 0], var_sphere, "Local Target-Velocity Variance\nSphere OT"),
        (axes[0, 1], var_biharm, "Local Target-Velocity Variance\nLocal Biharmonic OT"),
    ]:
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.1)
        sc = ax.scatter(
            src_xy[:, 0], src_xy[:, 1], c=var, cmap="magma", vmin=0.0, vmax=vmax,
            s=18, alpha=0.88, edgecolor="none",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    axes[1, 0].hist(var_sphere, bins=40, color="#444444", alpha=0.85)
    axes[1, 0].set_title("Variance Histogram\nSphere OT")
    axes[1, 1].hist(var_biharm, bins=40, color="#e6194b", alpha=0.85)
    axes[1, 1].set_title("Variance Histogram\nLocal Biharmonic OT")
    for ax in axes[1]:
        ax.set_xlabel("Local target-velocity variance")
        ax.set_ylabel("Count")
    fig.savefig(out_dir / "toy_horseshoes_velocity_variance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "sphere_local_variance_mean": float(var_sphere.mean()),
        "sphere_local_variance_median": float(np.median(var_sphere)),
        "biharmonic_local_variance_mean": float(var_biharm.mean()),
        "biharmonic_local_variance_median": float(np.median(var_biharm)),
    }


def plot_ot_pairings(
    t0_train: Marginal,
    t1_train: Marginal,
    out_path: Path,
    sample_pairs: int,
    seed: int,
    knn: int,
    n_eig: int,
) -> dict[str, float]:
    src_sphere = normalize_sphere(to_orthant(torch.tensor(t0_train.simplex, dtype=torch.float32)))
    tgt_sphere = normalize_sphere(to_orthant(torch.tensor(t1_train.simplex, dtype=torch.float32)))

    cost_sphere = compute_sphere_cost_matrix(src_sphere, tgt_sphere)
    cost_biharm = compute_biharmonic_cost_matrix(
        src_sphere, tgt_sphere, knn=knn, n_eig=n_eig, weight_power=0.5
    )

    src_order = _sort_key(t0_train.horseshoe, t0_train.u)
    tgt_order = _sort_key(t1_train.horseshoe, t1_train.u)
    sphere_sorted = cost_sphere[np.ix_(src_order, tgt_order)]
    biharm_sorted = cost_biharm[np.ix_(src_order, tgt_order)]

    np.random.seed(seed)
    sphere_src_idx, sphere_tgt_idx = ot_coupling(cost_sphere, sample_pairs)
    np.random.seed(seed)
    biharm_src_idx, biharm_tgt_idx = ot_coupling(cost_biharm, sample_pairs)

    metrics = {
        "sphere_argmin_cross_horseshoe": _cross_rate(
            t0_train.horseshoe, t1_train.horseshoe, np.arange(len(t0_train.horseshoe)), cost_sphere.argmin(axis=1)
        ),
        "biharmonic_argmin_cross_horseshoe": _cross_rate(
            t0_train.horseshoe, t1_train.horseshoe, np.arange(len(t0_train.horseshoe)), cost_biharm.argmin(axis=1)
        ),
        "sphere_sampled_cross_horseshoe": _cross_rate(
            t0_train.horseshoe, t1_train.horseshoe, sphere_src_idx, sphere_tgt_idx
        ),
        "biharmonic_sampled_cross_horseshoe": _cross_rate(
            t0_train.horseshoe, t1_train.horseshoe, biharm_src_idx, biharm_tgt_idx
        ),
    }

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_cost_sphere = fig.add_subplot(gs[0, 0])
    ax_cost_biharm = fig.add_subplot(gs[0, 1])
    ax_cpl_sphere = fig.add_subplot(gs[1, 0])
    ax_cpl_biharm = fig.add_subplot(gs[1, 1])

    im0 = ax_cost_sphere.imshow(sphere_sorted, aspect="auto", cmap="viridis")
    ax_cost_sphere.set_title("Sphere / Great-Circle OT Cost")
    ax_cost_sphere.set_xlabel("target points (sorted by horseshoe and arclength)")
    ax_cost_sphere.set_ylabel("source points (sorted by horseshoe and arclength)")
    fig.colorbar(im0, ax=ax_cost_sphere, fraction=0.046, pad=0.04)

    im1 = ax_cost_biharm.imshow(biharm_sorted, aspect="auto", cmap="viridis")
    ax_cost_biharm.set_title("Local Biharmonic OT Cost")
    ax_cost_biharm.set_xlabel("target points (sorted by horseshoe and arclength)")
    ax_cost_biharm.set_ylabel("source points (sorted by horseshoe and arclength)")
    fig.colorbar(im1, ax=ax_cost_biharm, fraction=0.046, pad=0.04)

    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    horse_colors = {1: "#1f77b4", 2: "#2ca02c"}
    for ax, src_idx, tgt_idx, title, cross_key in [
        (ax_cpl_sphere, sphere_src_idx, sphere_tgt_idx, "OT Samples Under Sphere Cost", "sphere_sampled_cross_horseshoe"),
        (ax_cpl_biharm, biharm_src_idx, biharm_tgt_idx, "OT Samples Under Local Biharmonic Cost", "biharmonic_sampled_cross_horseshoe"),
    ]:
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.0)
        ax.scatter(t0_train.xy[:, 0], t0_train.xy[:, 1], s=10, c="#8c8c8c", alpha=0.28, edgecolor="none")
        for horse_id in (1, 2):
            mask = t1_train.horseshoe == horse_id
            ax.scatter(
                t1_train.xy[mask, 0], t1_train.xy[mask, 1],
                s=11, c=horse_colors[horse_id], alpha=0.55, edgecolor="none",
            )
        for s_idx, t_idx in zip(src_idx, tgt_idx):
            ax.plot(
                [t0_train.xy[s_idx, 0], t1_train.xy[t_idx, 0]],
                [t0_train.xy[s_idx, 1], t1_train.xy[t_idx, 1]],
                color=horse_colors[int(t1_train.horseshoe[t_idx])],
                alpha=0.14,
                linewidth=0.8,
            )
        ax.set_title(f"{title}\nCross-horseshoe OT samples: {100 * metrics[cross_key]:.1f}%")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")

    fig.suptitle("Toy Horseshoe OT Pairings", fontsize=15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_trajectories_simplex(
    t0_test: Marginal,
    t1_test: Marginal,
    traj_by_method: dict[str, list[tuple[float, torch.Tensor]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(traj_by_method), figsize=(5.8 * len(traj_by_method), 5.6), constrained_layout=True)
    if len(traj_by_method) == 1:
        axes = [axes]
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    colors = {"sphere": "#444444", "biharmonic": "#e6194b", "premetric_spectral_ot": "#00a08a"}

    for ax, (name, traj) in zip(axes, traj_by_method.items()):
        ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.1)
        ax.scatter(t0_test.xy[:, 0], t0_test.xy[:, 1], s=10, c="#1f77b4", alpha=0.18, edgecolor="none")
        ax.scatter(t1_test.xy[:, 0], t1_test.xy[:, 1], s=10, c="#d62728", alpha=0.18, edgecolor="none")
        color = colors.get(name, "#17becf")
        n_cells = traj[0][1].shape[0]
        xy_ckpts = [simplex_to_xy(pos.numpy()) for _, pos in traj]
        for ci in range(n_cells):
            xs = [xy[ci, 0] for xy in xy_ckpts]
            ys = [xy[ci, 1] for xy in xy_ckpts]
            ax.plot(xs, ys, color=color, alpha=0.28, linewidth=1.0)
        ax.set_title(f"Generated trajectories\n{name}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
        ax.set_aspect("equal")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_orthant(
    t0_test_simplex: np.ndarray,
    t1_test_simplex: np.ndarray,
    traj_by_method: dict[str, list[tuple[float, torch.Tensor]]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(5.8 * len(traj_by_method), 5.8), constrained_layout=True)
    colors = {"sphere": "#444444", "biharmonic": "#e6194b", "premetric_spectral_ot": "#00a08a"}
    src_sphere = np.sqrt(np.clip(t0_test_simplex, 1e-8, None))
    tgt_sphere = np.sqrt(np.clip(t1_test_simplex, 1e-8, None))

    for col, (name, traj) in enumerate(traj_by_method.items(), start=1):
        ax = fig.add_subplot(1, len(traj_by_method), col, projection="3d")
        ax.scatter(src_sphere[:, 0], src_sphere[:, 1], src_sphere[:, 2], s=8, c="#1f77b4", alpha=0.12, depthshade=False)
        ax.scatter(tgt_sphere[:, 0], tgt_sphere[:, 1], tgt_sphere[:, 2], s=8, c="#d62728", alpha=0.12, depthshade=False)
        color = colors.get(name, "#17becf")
        sphere_ckpts = [np.sqrt(np.clip(pos.numpy(), 1e-8, None)) for _, pos in traj]
        n_cells = sphere_ckpts[0].shape[0]
        for ci in range(n_cells):
            xs = [pts[ci, 0] for pts in sphere_ckpts]
            ys = [pts[ci, 1] for pts in sphere_ckpts]
            zs = [pts[ci, 2] for pts in sphere_ckpts]
            ax.plot(xs, ys, zs, color=color, alpha=0.24, linewidth=1.0)
        ax.set_title(f"Generated trajectories\n{name}")
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_zlim(0.0, 1.0)
        ax.view_init(elev=26, azim=36)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-half", type=int, default=260)
    parser.add_argument("--noise-scale", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ot-subsample", type=int, default=1024)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--n-eig", type=int, default=40)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-traj", type=int, default=80)
    parser.add_argument("--n-checkpoints", type=int, default=20)
    parser.add_argument(
        "--include-premetric-spectral-ot",
        action="store_true",
        help="Also train MM+PremetricBiharmonic-SpectralOT and include its generated paths.",
    )
    parser.add_argument("--premetric-ode-steps", type=int, default=16)
    parser.add_argument("--premetric-extension-k", type=int, default=64)
    parser.add_argument("--premetric-softmax-beta", type=float, default=10.0)
    parser.add_argument(
        "--premetric-trajectory-mode",
        choices=["graph_geodesic", "spectral_decode", "spectral_decode_arclength", "ode"],
        default="graph_geodesic",
        help="Premetric teacher path construction for the spectral OT method.",
    )
    parser.add_argument("--premetric-decode-k", type=int, default=64)
    parser.add_argument("--premetric-decode-beta", type=float, default=10.0)
    parser.add_argument("--premetric-velocity-fd-eps", type=float, default=0.02)
    parser.add_argument("--premetric-time-cap", type=float, default=0.95)
    parser.add_argument("--premetric-grad-norm-floor", type=float, default=0.05)
    parser.add_argument("--premetric-max-drive-scale", type=float, default=50.0)
    parser.add_argument("--premetric-spectral-family", choices=["power", "diffusion"], default="power")
    parser.add_argument("--premetric-weight-power", type=float, default=0.5)
    parser.add_argument("--premetric-diffusion-time", type=float, default=1.0)
    parser.add_argument(
        "--premetric-diagnostics",
        action="store_true",
        help="Write premetric diagnostic CSV/JSON artifacts during toy training.",
    )
    parser.add_argument("--n-marginals", type=int, default=2, help="Number of time marginals. Use >2 for the multi-stage horseshoe toy.")
    parser.add_argument(
        "--window-width",
        type=float,
        default=0.24,
        help="Arclength window width per stage when n-marginals > 2.",
    )
    parser.add_argument(
        "--diagnostic-interval-idx",
        type=int,
        default=0,
        help="Which adjacent interval to use for OT/velocity diagnostics in multi-marginal mode.",
    )
    parser.add_argument(
        "--horseshoe-config",
        choices=["both", "single_left", "single_right"],
        default="both",
        help="Train/evaluate on both horseshoes or a single horseshoe only.",
    )
    parser.add_argument("--out-dir", default="plots/toy_horseshoes_flow")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rt = setup()

    horseshoe_id = None
    if args.horseshoe_config == "single_left":
        horseshoe_id = 1
    elif args.horseshoe_config == "single_right":
        horseshoe_id = 2

    if args.n_marginals == 2:
        t0, t1 = make_horseshoe_dataset(
            n_per_half=args.n_per_half,
            noise_scale=args.noise_scale,
            seed=args.seed,
        )
        marginals = [subset_marginal(t0, horseshoe_id), subset_marginal(t1, horseshoe_id)]
    else:
        marginals = [
            subset_marginal(m, horseshoe_id)
            for m in make_horseshoe_multimarginal_dataset(
                n_per_stage_per_horseshoe=args.n_per_half,
                noise_scale=args.noise_scale,
                seed=args.seed,
                n_marginals=args.n_marginals,
                window_width=args.window_width,
            )
        ]
        plot_stage_overlay(marginals, out_dir / "toy_horseshoes_stage_overlay.png")

    train_marginals: list[Marginal] = []
    test_marginals: list[Marginal] = []
    for idx, marginal in enumerate(marginals):
        train_m, test_m = split_marginal(marginal, args.train_frac, args.seed + idx)
        train_marginals.append(train_m)
        test_marginals.append(test_m)

    stage_cells = [
        normalize_sphere(to_orthant(torch.tensor(m.simplex, dtype=torch.float32))).to(rt.device)
        for m in train_marginals
    ]
    stage_times = np.linspace(0.0, 1.0, len(train_marginals)).tolist()
    test_source = normalize_sphere(to_orthant(torch.tensor(test_marginals[0].simplex, dtype=torch.float32))).to(rt.device)

    diag_iv = min(max(args.diagnostic_interval_idx, 0), len(train_marginals) - 2)
    diag_src = train_marginals[diag_iv]
    diag_tgt = train_marginals[diag_iv + 1]

    coupling_metrics = plot_ot_pairings(
        diag_src,
        diag_tgt,
        out_dir / "toy_horseshoes_ot_pairings.png",
        sample_pairs=min(240, len(diag_src.simplex)),
        seed=args.seed,
        knn=args.knn,
        n_eig=args.n_eig,
    )
    velocity_metrics = plot_velocity_targets_and_variance(
        diag_src,
        diag_tgt,
        out_dir=out_dir,
        knn_cost=args.knn,
        n_eig=args.n_eig,
        knn_var=12,
        arrow_scale=0.08,
    )

    print("\nTraining sphere-cost FM on horseshoe toy...")
    model_sphere, losses_sphere = train_multi_marginal_flow(
        stage_cells=stage_cells,
        stage_times=stage_times,
        D=3,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        label="ToyHorseshoeSphere",
        ot_subsample=args.ot_subsample,
    )

    print("\nTraining biharmonic-cost FM on horseshoe toy...")
    model_biharm, losses_biharm = train_multi_marginal_flow(
        stage_cells=stage_cells,
        stage_times=stage_times,
        D=3,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        label="ToyHorseshoeBiharmonic",
        ot_subsample=args.ot_subsample,
        cost_fn=lambda Y0, Y1: compute_biharmonic_cost_matrix(
            Y0, Y1, knn=args.knn, n_eig=args.n_eig, weight_power=0.5
        ),
    )

    model_premetric = None
    losses_premetric = None
    if args.include_premetric_spectral_ot:
        print("\nTraining MM+PremetricBiharmonic-SpectralOT on horseshoe toy...")
        model_premetric, losses_premetric = train_multi_marginal_flow(
            stage_cells=stage_cells,
            stage_times=stage_times,
            D=3,
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            lr=args.lr,
            label="ToyHorseshoePremetricSpectralOT",
            ot_subsample=args.ot_subsample,
            premetric_type="biharmonic",
            premetric_ot_cost=True,
            premetric_extension_k=args.premetric_extension_k,
            premetric_softmax_beta=args.premetric_softmax_beta,
            premetric_ode_steps=args.premetric_ode_steps,
            premetric_trajectory_mode=args.premetric_trajectory_mode,
            premetric_decode_k=args.premetric_decode_k,
            premetric_decode_beta=args.premetric_decode_beta,
            premetric_velocity_fd_eps=args.premetric_velocity_fd_eps,
            premetric_knn=args.knn,
            premetric_n_eig=args.n_eig,
            premetric_spectral_family=args.premetric_spectral_family,
            premetric_weight_power=args.premetric_weight_power,
            premetric_diffusion_time=args.premetric_diffusion_time,
            premetric_time_cap=args.premetric_time_cap,
            premetric_grad_norm_floor=args.premetric_grad_norm_floor,
            premetric_max_drive_scale=args.premetric_max_drive_scale,
            premetric_diagnostics=args.premetric_diagnostics,
            premetric_diagnostic_samples=256,
        )

    pred_sphere = from_orthant(generate_fisher_flow(model_sphere, test_source, n_steps=args.n_steps, t_start=0.0, t_end=1.0))
    pred_biharm = from_orthant(generate_fisher_flow(model_biharm, test_source, n_steps=args.n_steps, t_start=0.0, t_end=1.0))
    pred_by_method = {
        "sphere": pred_sphere.numpy(),
        "biharmonic": pred_biharm.numpy(),
    }
    if model_premetric is not None:
        pred_premetric = from_orthant(
            generate_fisher_flow(model_premetric, test_source, n_steps=args.n_steps, t_start=0.0, t_end=1.0)
        )
        pred_by_method["premetric_spectral_ot"] = pred_premetric.numpy()

    traj_rng = np.random.default_rng(args.seed)
    traj_idx = traj_rng.choice(len(test_source), size=min(args.n_traj, len(test_source)), replace=False)
    traj_source = test_source[traj_idx]
    traj_sphere = generate_fisher_flow_trajectory(
        model_sphere, traj_source, n_steps=args.n_steps, t_start=0.0, t_end=1.0,
        n_checkpoints=args.n_checkpoints,
    )
    traj_biharm = generate_fisher_flow_trajectory(
        model_biharm, traj_source, n_steps=args.n_steps, t_start=0.0, t_end=1.0,
        n_checkpoints=args.n_checkpoints,
    )
    traj_by_method = {
        "sphere": traj_sphere,
        "biharmonic": traj_biharm,
    }
    if model_premetric is not None:
        traj_by_method["premetric_spectral_ot"] = generate_fisher_flow_trajectory(
            model_premetric, traj_source, n_steps=args.n_steps, t_start=0.0, t_end=1.0,
            n_checkpoints=args.n_checkpoints,
        )

    t0_test = test_marginals[0]
    t1_test = test_marginals[-1]
    t1_test_simplex = torch.tensor(t1_test.simplex, dtype=torch.float32)

    metrics = {
        "couplings": coupling_metrics,
        "velocity_diagnostics": velocity_metrics,
        "config": {
            "horseshoe_config": args.horseshoe_config,
            "n_marginals": len(marginals),
            "diagnostic_interval_idx": diag_iv,
        },
        "sphere": {
            "mmd2": mmd_rbf(pred_sphere, t1_test_simplex),
            "fgd": fgd(pred_sphere, t1_test_simplex),
            "final_loss": float(losses_sphere[-1]),
        },
        "biharmonic": {
            "mmd2": mmd_rbf(pred_biharm, t1_test_simplex),
            "fgd": fgd(pred_biharm, t1_test_simplex),
            "final_loss": float(losses_biharm[-1]),
        },
    }
    if model_premetric is not None:
        metrics["premetric_spectral_ot"] = {
            "mmd2": mmd_rbf(torch.tensor(pred_by_method["premetric_spectral_ot"], dtype=torch.float32), t1_test_simplex),
            "fgd": fgd(torch.tensor(pred_by_method["premetric_spectral_ot"], dtype=torch.float32), t1_test_simplex),
            "final_loss": float(losses_premetric[-1]),
            "spectral_family": args.premetric_spectral_family,
            "weight_power": args.premetric_weight_power,
            "diffusion_time": args.premetric_diffusion_time,
        }

    plot_endpoints(
        t0_test,
        t1_test,
        pred_by_method,
        out_dir / "toy_horseshoes_flow_endpoints.png",
    )
    plot_endpoints_orthant(
        t1_test.simplex,
        pred_by_method,
        out_dir / "toy_horseshoes_flow_orthant.png",
    )
    plot_trajectories_simplex(
        t0_test,
        t1_test,
        traj_by_method,
        out_dir / "toy_horseshoes_flow_trajectories_simplex.png",
    )
    plot_trajectories_orthant(
        t0_test.simplex,
        t1_test.simplex,
        traj_by_method,
        out_dir / "toy_horseshoes_flow_trajectories_orthant.png",
    )

    output_arrays = dict(
        stage_times=np.asarray(stage_times, dtype=np.float32),
        t0_test=t0_test.simplex,
        t1_test=t1_test.simplex,
        pred_sphere=pred_by_method["sphere"],
        pred_biharmonic=pred_by_method["biharmonic"],
        traj_sphere=np.stack([pos.numpy() for _, pos in traj_sphere], axis=0),
        traj_biharmonic=np.stack([pos.numpy() for _, pos in traj_biharm], axis=0),
        losses_sphere=np.asarray(losses_sphere, dtype=np.float32),
        losses_biharmonic=np.asarray(losses_biharm, dtype=np.float32),
    )
    if model_premetric is not None:
        output_arrays.update(
            pred_premetric_spectral_ot=pred_by_method["premetric_spectral_ot"],
            traj_premetric_spectral_ot=np.stack(
                [pos.numpy() for _, pos in traj_by_method["premetric_spectral_ot"]], axis=0
            ),
            losses_premetric_spectral_ot=np.asarray(losses_premetric, dtype=np.float32),
        )
    np.savez_compressed(out_dir / "toy_horseshoes_flow_outputs.npz", **output_arrays)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nToy horseshoe FM summary")
    print(f"  config    : {args.horseshoe_config}")
    print(f"  marginals : {len(marginals)}")
    print(f"  diagnostic interval : {diag_iv} ({stage_times[diag_iv]:.2f} -> {stage_times[diag_iv + 1]:.2f})")
    print(
        "  pairings  : "
        f"sphere_ot_cross={100 * coupling_metrics['sphere_sampled_cross_horseshoe']:.1f}%  "
        f"biharm_ot_cross={100 * coupling_metrics['biharmonic_sampled_cross_horseshoe']:.1f}%"
    )
    print(
        "  local var : "
        f"sphere_mean={velocity_metrics['sphere_local_variance_mean']:.5f}  "
        f"biharm_mean={velocity_metrics['biharmonic_local_variance_mean']:.5f}"
    )
    for name, vals in metrics.items():
        if name in {"couplings", "velocity_diagnostics", "config"}:
            continue
        print(f"  {name:10s} MMD^2={vals['mmd2']:.5f}  FGD={vals['fgd']:.5f}  final_loss={vals['final_loss']:.5f}")
    print(f"  outputs: {out_dir}")


if __name__ == "__main__":
    main()
