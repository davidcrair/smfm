#!/usr/bin/env python
"""Train flow matching models on a two-marginal Y toy and visualize trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
import ot

from surf.runtime import setup
from surf.training.flow_trainer import train_multi_marginal_flow
from surf.training.spectral_jvp_trainer import train_spectral_path_jvp_flow
from surf.evaluation.generation import generate_fisher_flow, generate_fisher_flow_trajectory
from surf.geometry.sphere import to_orthant, normalize_sphere, from_orthant
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
ROOT = np.array([0.50, 0.06], dtype=np.float64)
JUNCTION = np.array([0.50, 0.36], dtype=np.float64)
LEFT_LEAF = np.array([0.34, 0.58], dtype=np.float64)
RIGHT_LEAF = np.array([0.66, 0.58], dtype=np.float64)


@dataclass
class Marginal:
    name: str
    simplex: np.ndarray
    xy: np.ndarray
    branch: np.ndarray


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


def _line_points(
    start: np.ndarray,
    end: np.ndarray,
    u: np.ndarray,
    *,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    xy = (1.0 - u)[:, None] * start[None, :] + u[:, None] * end[None, :]
    tangent = end - start
    tangent = tangent / np.linalg.norm(tangent).clip(min=1e-8)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    xy = (
        xy
        + rng.normal(scale=noise, size=(len(u), 1)) * normal[None, :]
        + rng.normal(scale=0.25 * noise, size=(len(u), 1)) * tangent[None, :]
    )
    return xy


def make_y_dataset(
    *,
    n_source: int,
    n_target_per_branch: int,
    noise: float,
    seed: int,
    target_mode: str = "both",
) -> tuple[Marginal, Marginal]:
    rng = np.random.default_rng(seed)
    source_u = rng.uniform(0.0, 1.0, size=n_source)
    source_xy = _line_points(ROOT, JUNCTION, source_u, noise=noise, rng=rng)
    source_simplex = xy_to_simplex(source_xy)
    source = Marginal(
        name="t0_trunk",
        simplex=source_simplex,
        xy=simplex_to_xy(source_simplex),
        branch=np.zeros(n_source, dtype=np.int64),
    )

    target_parts = []
    branch_parts = []
    if target_mode in ("both", "left"):
        left_u = rng.uniform(0.0, 1.0, size=n_target_per_branch)
        left_xy = _line_points(JUNCTION, LEFT_LEAF, left_u, noise=noise, rng=rng)
        target_parts.append(left_xy)
        branch_parts.append(np.ones(n_target_per_branch, dtype=np.int64))
    if target_mode in ("both", "right"):
        right_u = rng.uniform(0.0, 1.0, size=n_target_per_branch)
        right_xy = _line_points(JUNCTION, RIGHT_LEAF, right_u, noise=noise, rng=rng)
        target_parts.append(right_xy)
        branch_parts.append(np.full(n_target_per_branch, 2, dtype=np.int64))
    if not target_parts:
        raise ValueError(f"Unknown target_mode={target_mode!r}; expected 'both', 'left', or 'right'.")

    target_xy = np.vstack(target_parts)
    target_simplex = xy_to_simplex(target_xy)
    target = Marginal(
        name="t1_branches",
        simplex=target_simplex,
        xy=simplex_to_xy(target_simplex),
        branch=np.concatenate(branch_parts),
    )
    return source, target


def split_marginal(marginal: Marginal, train_frac: float, seed: int) -> tuple[Marginal, Marginal]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(marginal.simplex))
    n_train = min(max(int(round(train_frac * len(perm))), 1), len(perm) - 1)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    train = Marginal(
        f"{marginal.name}_train",
        marginal.simplex[train_idx],
        marginal.xy[train_idx],
        marginal.branch[train_idx],
    )
    test = Marginal(
        f"{marginal.name}_test",
        marginal.simplex[test_idx],
        marginal.xy[test_idx],
        marginal.branch[test_idx],
    )
    return train, test


def _target_segments(target_mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if target_mode == "left":
        return [(JUNCTION, LEFT_LEAF)]
    if target_mode == "right":
        return [(JUNCTION, RIGHT_LEAF)]
    if target_mode == "both":
        return [(JUNCTION, LEFT_LEAF), (JUNCTION, RIGHT_LEAF)]
    raise ValueError(f"Unknown target_mode={target_mode!r}")


def _point_to_segment_distance(xy: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-12:
        return np.linalg.norm(xy - start[None, :], axis=1)
    u = ((xy - start[None, :]) @ seg) / denom
    u = np.clip(u, 0.0, 1.0)
    proj = start[None, :] + u[:, None] * seg[None, :]
    return np.linalg.norm(xy - proj, axis=1)


def _distance_to_target_manifold(xy: np.ndarray, target_mode: str) -> np.ndarray:
    dists = [
        _point_to_segment_distance(xy, start, end)
        for start, end in _target_segments(target_mode)
    ]
    return np.min(np.stack(dists, axis=1), axis=1)


def _cov_eigvals(cov: np.ndarray) -> np.ndarray:
    cov = np.atleast_2d(cov)
    eigvals = np.linalg.eigvalsh(cov)
    return np.sort(np.real_if_close(eigvals))[::-1]


def _frechet_terms(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    sigma_x = np.atleast_2d(np.cov(X, rowvar=False))
    sigma_y = np.atleast_2d(np.cov(Y, rowvar=False))
    mean_term = float((mu_x - mu_y).dot(mu_x - mu_y))
    covmean = linalg.sqrtm(sigma_x.dot(sigma_y))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    cov_term = float(
        np.trace(sigma_x) + np.trace(sigma_y) - 2.0 * np.trace(covmean)
    )
    return mean_term, cov_term


def empirical_w2_sq(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    *,
    cost: str,
) -> float:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    a = np.ones(len(X), dtype=np.float64) / len(X)
    b = np.ones(len(Y), dtype=np.float64) / len(Y)

    if cost == "simplex_l2":
        C = ot.dist(X, Y, metric="sqeuclidean")
    elif cost == "sphere_arc":
        Xs = np.sqrt(np.clip(X, 1e-8, None))
        Ys = np.sqrt(np.clip(Y, 1e-8, None))
        Xs = Xs / np.linalg.norm(Xs, axis=1, keepdims=True).clip(min=1e-8)
        Ys = Ys / np.linalg.norm(Ys, axis=1, keepdims=True).clip(min=1e-8)
        cos = np.clip(Xs @ Ys.T, -1.0 + 1e-8, 1.0 - 1e-8)
        arc = np.arccos(cos)
        C = arc ** 2
    else:
        raise ValueError(f"Unknown OT cost={cost!r}; expected 'simplex_l2' or 'sphere_arc'.")
    return float(ot.emd2(a, b, C))


def endpoint_diagnostics(
    pred_simplex: np.ndarray,
    target_simplex: np.ndarray,
    *,
    target_mode: str,
) -> dict[str, object]:
    pred_xy = simplex_to_xy(pred_simplex)
    target_xy = simplex_to_xy(target_simplex)
    pred_mean_xy = pred_xy.mean(axis=0)
    target_mean_xy = target_xy.mean(axis=0)
    pred_cov_xy = np.atleast_2d(np.cov(pred_xy, rowvar=False))
    target_cov_xy = np.atleast_2d(np.cov(target_xy, rowvar=False))
    pred_manifold_dist = _distance_to_target_manifold(pred_xy, target_mode)
    target_manifold_dist = _distance_to_target_manifold(target_xy, target_mode)
    fgd_mean_term, fgd_cov_term = _frechet_terms(pred_simplex, target_simplex)
    w2_sq_simplex = empirical_w2_sq(pred_simplex, target_simplex, cost="simplex_l2")
    w2_sq_sphere_arc = empirical_w2_sq(pred_simplex, target_simplex, cost="sphere_arc")
    return {
        "mean_xy": pred_mean_xy.tolist(),
        "target_mean_xy": target_mean_xy.tolist(),
        "mean_error_xy": float(np.linalg.norm(pred_mean_xy - target_mean_xy)),
        "cov_trace_xy": float(np.trace(pred_cov_xy)),
        "target_cov_trace_xy": float(np.trace(target_cov_xy)),
        "cov_eigvals_xy": _cov_eigvals(pred_cov_xy).tolist(),
        "target_cov_eigvals_xy": _cov_eigvals(target_cov_xy).tolist(),
        "manifold_dist_mean_xy": float(pred_manifold_dist.mean()),
        "manifold_dist_q95_xy": float(np.quantile(pred_manifold_dist, 0.95)),
        "target_manifold_dist_mean_xy": float(target_manifold_dist.mean()),
        "target_manifold_dist_q95_xy": float(np.quantile(target_manifold_dist, 0.95)),
        "fgd_sq_mean_term": float(fgd_mean_term),
        "fgd_sq_cov_term": float(max(fgd_cov_term, 0.0)),
        "w2_simplex_l2_sq": float(w2_sq_simplex),
        "w2_simplex_l2": float(np.sqrt(max(w2_sq_simplex, 0.0))),
        "w2_sphere_arc_sq": float(w2_sq_sphere_arc),
        "w2_sphere_arc": float(np.sqrt(max(w2_sq_sphere_arc, 0.0))),
    }


def print_endpoint_diagnostics(method: str, diag: dict[str, object]) -> None:
    pred_mean = np.asarray(diag["mean_xy"])
    target_mean = np.asarray(diag["target_mean_xy"])
    pred_eigs = np.asarray(diag["cov_eigvals_xy"])
    target_eigs = np.asarray(diag["target_cov_eigvals_xy"])
    print(
        f"  {method}: "
        f"mean_err_xy={diag['mean_error_xy']:.4f}, "
        f"pred_mean=({pred_mean[0]:.4f}, {pred_mean[1]:.4f}), "
        f"target_mean=({target_mean[0]:.4f}, {target_mean[1]:.4f})"
    )
    print(
        "    "
        f"cov_trace_xy={diag['cov_trace_xy']:.4f} "
        f"(target {diag['target_cov_trace_xy']:.4f}), "
        f"cov_eigs_xy={np.array2string(pred_eigs, precision=4)} "
        f"(target {np.array2string(target_eigs, precision=4)})"
    )
    print(
        "    "
        f"branch_dist_mean_xy={diag['manifold_dist_mean_xy']:.4f} "
        f"(target {diag['target_manifold_dist_mean_xy']:.4f}), "
        f"branch_dist_q95_xy={diag['manifold_dist_q95_xy']:.4f} "
        f"(target {diag['target_manifold_dist_q95_xy']:.4f})"
    )
    print(
        "    "
        f"FGD^2 mean_term={diag['fgd_sq_mean_term']:.6f}, "
        f"cov_term={diag['fgd_sq_cov_term']:.6f}"
    )
    print(
        "    "
        f"W2(simplex l2)={diag['w2_simplex_l2']:.4f} "
        f"[sq={diag['w2_simplex_l2_sq']:.6f}], "
        f"W2(sphere arc)={diag['w2_sphere_arc']:.4f} "
        f"[sq={diag['w2_sphere_arc_sq']:.6f}]"
    )


def _style_simplex(ax, title: str) -> None:
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")


def plot_endpoints(
    source_test: Marginal,
    target_test: Marginal,
    pred_by_method: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    fig, axes = plt.subplots(
        1,
        2 + len(pred_by_method),
        figsize=(5.2 * (2 + len(pred_by_method)), 5.2),
        constrained_layout=True,
    )
    _style_simplex(axes[0], "t=0 Test Source")
    axes[0].scatter(source_test.xy[:, 0], source_test.xy[:, 1], s=14, c="#555555", alpha=0.60, edgecolor="none")

    _style_simplex(axes[1], "t=1 Test Target")
    for branch_id in np.unique(target_test.branch):
        color = branch_colors.get(int(branch_id), "#cc79a7")
        mask = target_test.branch == branch_id
        axes[1].scatter(target_test.xy[mask, 0], target_test.xy[mask, 1], s=14, c=color, alpha=0.60, edgecolor="none")

    colors = {
        "slerp": "#444444",
        "premetric": "#00a08a",
        "spectral_jvp": "#7b3294",
    }
    for ax, (name, pred_simplex) in zip(axes[2:], pred_by_method.items()):
        _style_simplex(ax, f"Predicted t=1\n{name}")
        ax.scatter(target_test.xy[:, 0], target_test.xy[:, 1], s=10, c="#dddddd", alpha=0.26, edgecolor="none")
        pred_xy = simplex_to_xy(pred_simplex)
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=14, c=colors.get(name, "#cc79a7"), alpha=0.62, edgecolor="none")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_simplex(
    source_test: Marginal,
    target_test: Marginal,
    traj_by_method: dict[str, list[tuple[float, torch.Tensor]]],
    out_path: Path,
    *,
    rollout_group_size: int = 1,
) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    fig, axes = plt.subplots(1, len(traj_by_method), figsize=(6.2 * len(traj_by_method), 5.6), constrained_layout=True)
    if len(traj_by_method) == 1:
        axes = [axes]
    for ax, (name, traj) in zip(axes, traj_by_method.items()):
        _style_simplex(ax, f"Learned Trajectories\n{name}")
        ax.scatter(source_test.xy[:, 0], source_test.xy[:, 1], s=12, c="#555555", alpha=0.20, edgecolor="none")
        for branch_id in np.unique(target_test.branch):
            color = branch_colors.get(int(branch_id), "#cc79a7")
            mask = target_test.branch == branch_id
            ax.scatter(target_test.xy[mask, 0], target_test.xy[mask, 1], s=12, c=color, alpha=0.20, edgecolor="none")

        traj_np = [pos.numpy() for _, pos in traj]
        for sample_idx in range(traj_np[0].shape[0]):
            path_simplex = np.stack([arr[sample_idx] for arr in traj_np], axis=0)
            path_xy = simplex_to_xy(path_simplex)
            alpha = 0.48 if rollout_group_size <= 1 else 0.26
            linewidth = 1.1 if rollout_group_size <= 1 else 0.95
            color = "#00a08a" if name == "premetric" else "#7b3294" if name == "spectral_jvp" else "#444444"
            ax.plot(path_xy[:, 0], path_xy[:, 1], color=color, alpha=alpha, linewidth=linewidth)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_orthant(
    source_test: Marginal,
    target_test: Marginal,
    traj_by_method: dict[str, list[tuple[float, torch.Tensor]]],
    out_path: Path,
    *,
    rollout_group_size: int = 1,
) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    fig = plt.figure(figsize=(6.0 * len(traj_by_method), 5.8), constrained_layout=True)
    source_sphere = np.sqrt(np.clip(source_test.simplex, 1e-8, None))
    target_sphere = np.sqrt(np.clip(target_test.simplex, 1e-8, None))
    for col, (name, traj) in enumerate(traj_by_method.items(), start=1):
        ax = fig.add_subplot(1, len(traj_by_method), col, projection="3d")
        ax.scatter(source_sphere[:, 0], source_sphere[:, 1], source_sphere[:, 2], s=10, c="#555555", alpha=0.18, depthshade=False)
        for branch_id in np.unique(target_test.branch):
            color = branch_colors.get(int(branch_id), "#cc79a7")
            mask = target_test.branch == branch_id
            ax.scatter(target_sphere[mask, 0], target_sphere[mask, 1], target_sphere[mask, 2], s=10, c=color, alpha=0.18, depthshade=False)
        traj_np = [np.sqrt(np.clip(pos.numpy(), 1e-8, None)) for _, pos in traj]
        for sample_idx in range(traj_np[0].shape[0]):
            path = np.stack([arr[sample_idx] for arr in traj_np], axis=0)
            alpha = 0.48 if rollout_group_size <= 1 else 0.26
            linewidth = 1.0 if rollout_group_size <= 1 else 0.9
            color = "#00a08a" if name == "premetric" else "#7b3294" if name == "spectral_jvp" else "#444444"
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, alpha=alpha, linewidth=linewidth)
        ax.set_title(name)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)
        ax.view_init(elev=24, azim=38)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-source", type=int, default=220)
    parser.add_argument("--n-target-per-branch", type=int, default=110)
    parser.add_argument("--target-mode", choices=["both", "left", "right"], default="both")
    parser.add_argument("--noise", type=float, default=0.008)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ot-subsample", type=int, default=220)
    parser.add_argument("--n-steps", type=int, default=120)
    parser.add_argument("--n-traj", type=int, default=40)
    parser.add_argument("--n-checkpoints", type=int, default=80)
    parser.add_argument("--eval-rollouts", type=int, default=1,
                        help="Number of stochastic endpoint rollouts per held-out source cell.")
    parser.add_argument("--traj-rollouts", type=int, default=1,
                        help="Number of stochastic trajectory rollouts per plotted source cell.")
    parser.add_argument("--inf-sigma", type=float, default=0.0,
                        help="Inference-time Euler-Maruyama noise scale on the sphere.")
    parser.add_argument(
        "--manifold-project-mode",
        choices=["none", "nearest", "soft"],
        default="none",
        help="Optional support projection after each inference step.",
    )
    parser.add_argument(
        "--manifold-project-every",
        type=int,
        default=1,
        help="Project every N inference steps when manifold projection is enabled.",
    )
    parser.add_argument(
        "--manifold-project-k",
        type=int,
        default=32,
        help="Support neighbors for soft manifold projection.",
    )
    parser.add_argument(
        "--manifold-project-beta",
        type=float,
        default=20.0,
        help="Inverse temperature for soft support projection.",
    )
    parser.add_argument(
        "--manifold-project-mix",
        type=float,
        default=1.0,
        help="Blend weight between current point and projected point.",
    )
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--n-eig", type=int, default=32)
    parser.add_argument("--spectral-family", choices=["power", "diffusion"], default="power")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--diffusion-time", type=float, default=1.0)
    parser.add_argument(
        "--trajectory-mode",
        choices=["graph_geodesic", "spectral_decode", "spectral_decode_arclength", "ode"],
        default="graph_geodesic",
    )
    parser.add_argument("--decode-k", type=int, default=64)
    parser.add_argument("--decode-beta", type=float, default=10.0)
    parser.add_argument("--velocity-fd-eps", type=float, default=0.02)
    parser.add_argument("--extension-k", type=int, default=64)
    parser.add_argument("--softmax-beta", type=float, default=10.0)
    parser.add_argument("--ode-steps", type=int, default=16)
    parser.add_argument("--time-cap", type=float, default=0.95)
    parser.add_argument("--grad-norm-floor", type=float, default=0.05)
    parser.add_argument("--max-drive-scale", type=float, default=50.0)
    parser.add_argument(
        "--include-spectral-jvp",
        action="store_true",
        help="Also train the spectral-path JVP flow baseline.",
    )
    parser.add_argument("--spectral-jvp-encoder-iters", type=int, default=1000)
    parser.add_argument("--spectral-jvp-encoder-batch-size", type=int, default=0)
    parser.add_argument("--spectral-jvp-encoder-lr", type=float, default=3e-4)
    parser.add_argument("--spectral-jvp-encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--spectral-jvp-encoder-depth", type=int, default=4)
    parser.add_argument("--spectral-jvp-interp-fraction", type=float, default=0.5)
    parser.add_argument("--spectral-jvp-decode-k", type=int, default=64)
    parser.add_argument("--spectral-jvp-decode-tau", default="auto")
    parser.add_argument("--spectral-jvp-decode-chunk-size", type=int, default=64)
    parser.add_argument("--spectral-jvp-velocity-norm-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="plots/toy_y_flow_matching")
    args = parser.parse_args()

    rt = setup(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source, target = make_y_dataset(
        n_source=args.n_source,
        n_target_per_branch=args.n_target_per_branch,
        noise=args.noise,
        seed=args.seed,
        target_mode=args.target_mode,
    )
    source_train, source_test = split_marginal(source, args.train_frac, args.seed)
    target_train, target_test = split_marginal(target, args.train_frac, args.seed + 1)

    stage_cells = [
        normalize_sphere(to_orthant(torch.tensor(source_train.simplex, dtype=torch.float32))).to(rt.device),
        normalize_sphere(to_orthant(torch.tensor(target_train.simplex, dtype=torch.float32))).to(rt.device),
    ]
    stage_times = [0.0, 1.0]
    test_source = normalize_sphere(to_orthant(torch.tensor(source_test.simplex, dtype=torch.float32))).to(rt.device)
    test_target_comp = torch.tensor(target_test.simplex, dtype=torch.float32)
    eval_rollouts = max(int(args.eval_rollouts), 1)
    traj_rollouts = max(int(args.traj_rollouts), 1)
    support_cells = torch.cat(stage_cells, dim=0)

    print("\nTraining MM+SLERP on Y toy...")
    model_slerp, losses_slerp = train_multi_marginal_flow(
        stage_cells=stage_cells,
        stage_times=stage_times,
        D=3,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        label="ToyYSLERP",
        ot_subsample=args.ot_subsample,
    )

    print("\nTraining MM+PremetricBiharmonic-SpectralOT on Y toy...")
    model_premetric, losses_premetric = train_multi_marginal_flow(
        stage_cells=stage_cells,
        stage_times=stage_times,
        D=3,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        label="ToyYPremetricSpectralOT",
        ot_subsample=args.ot_subsample,
        premetric_type="biharmonic",
        premetric_ot_cost=True,
        premetric_extension_k=args.extension_k,
        premetric_softmax_beta=args.softmax_beta,
        premetric_ode_steps=args.ode_steps,
        premetric_trajectory_mode=args.trajectory_mode,
        premetric_decode_k=args.decode_k,
        premetric_decode_beta=args.decode_beta,
        premetric_velocity_fd_eps=args.velocity_fd_eps,
        premetric_knn=args.knn,
        premetric_n_eig=args.n_eig,
        premetric_spectral_family=args.spectral_family,
        premetric_weight_power=args.weight_power,
        premetric_diffusion_time=args.diffusion_time,
        premetric_time_cap=args.time_cap,
        premetric_grad_norm_floor=args.grad_norm_floor,
        premetric_max_drive_scale=args.max_drive_scale,
    )

    model_spectral_jvp = None
    losses_spectral_jvp = None
    if args.include_spectral_jvp:
        print("\nTraining MM+SpectralPathJVP on Y toy...")
        encoder_batch_size = (
            args.spectral_jvp_encoder_batch_size
            if args.spectral_jvp_encoder_batch_size > 0
            else args.batch_size
        )
        model_spectral_jvp, losses_spectral_jvp = train_spectral_path_jvp_flow(
            stage_cells=stage_cells,
            stage_times=stage_times,
            D=3,
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            lr=args.lr,
            label="ToyYSpectralPathJVP",
            ot_subsample=args.ot_subsample,
            spectral_knn=args.knn,
            spectral_n_eig=args.n_eig,
            spectral_family=args.spectral_family,
            spectral_weight_power=args.weight_power,
            spectral_diffusion_time=args.diffusion_time,
            decode_k=args.spectral_jvp_decode_k,
            decode_tau=args.spectral_jvp_decode_tau,
            decode_chunk_size=args.spectral_jvp_decode_chunk_size,
            encoder_iters=args.spectral_jvp_encoder_iters,
            encoder_batch_size=encoder_batch_size,
            encoder_lr=args.spectral_jvp_encoder_lr,
            encoder_hidden_dim=args.spectral_jvp_encoder_hidden_dim,
            encoder_depth=args.spectral_jvp_encoder_depth,
            encoder_interp_fraction=args.spectral_jvp_interp_fraction,
            velocity_norm_weight=args.spectral_jvp_velocity_norm_weight,
        )

    eval_source = test_source.repeat_interleave(eval_rollouts, dim=0)
    torch.manual_seed(args.seed + 101)
    pred_slerp = from_orthant(
        generate_fisher_flow(
            model_slerp,
            eval_source,
            n_steps=args.n_steps,
            t_start=0.0,
            t_end=1.0,
            inf_sigma=args.inf_sigma,
            support_cells=support_cells,
            manifold_project_mode=args.manifold_project_mode,
            manifold_project_every=args.manifold_project_every,
            manifold_project_k=args.manifold_project_k,
            manifold_project_beta=args.manifold_project_beta,
            manifold_project_mix=args.manifold_project_mix,
        )
    )
    torch.manual_seed(args.seed + 202)
    pred_premetric = from_orthant(
        generate_fisher_flow(
            model_premetric,
            eval_source,
            n_steps=args.n_steps,
            t_start=0.0,
            t_end=1.0,
            inf_sigma=args.inf_sigma,
            support_cells=support_cells,
            manifold_project_mode=args.manifold_project_mode,
            manifold_project_every=args.manifold_project_every,
            manifold_project_k=args.manifold_project_k,
            manifold_project_beta=args.manifold_project_beta,
            manifold_project_mix=args.manifold_project_mix,
        )
    )
    pred_by_method = {
        "slerp": pred_slerp.numpy(),
        "premetric": pred_premetric.numpy(),
    }
    if model_spectral_jvp is not None:
        torch.manual_seed(args.seed + 303)
        pred_spectral_jvp = from_orthant(
            generate_fisher_flow(
                model_spectral_jvp,
                eval_source,
                n_steps=args.n_steps,
                t_start=0.0,
                t_end=1.0,
                inf_sigma=args.inf_sigma,
                support_cells=support_cells,
                manifold_project_mode=args.manifold_project_mode,
                manifold_project_every=args.manifold_project_every,
                manifold_project_k=args.manifold_project_k,
                manifold_project_beta=args.manifold_project_beta,
                manifold_project_mix=args.manifold_project_mix,
            )
        )
        pred_by_method["spectral_jvp"] = pred_spectral_jvp.numpy()

    traj_rng = np.random.default_rng(args.seed)
    traj_idx = traj_rng.choice(len(test_source), size=min(args.n_traj, len(test_source)), replace=False)
    traj_source_base = test_source[traj_idx]
    traj_source = traj_source_base.repeat_interleave(traj_rollouts, dim=0)
    traj_by_method = {
        "slerp": generate_fisher_flow_trajectory(
            model_slerp,
            traj_source,
            n_steps=args.n_steps,
            t_start=0.0,
            t_end=1.0,
            n_checkpoints=args.n_checkpoints,
            inf_sigma=args.inf_sigma,
            support_cells=support_cells,
            manifold_project_mode=args.manifold_project_mode,
            manifold_project_every=args.manifold_project_every,
            manifold_project_k=args.manifold_project_k,
            manifold_project_beta=args.manifold_project_beta,
            manifold_project_mix=args.manifold_project_mix,
        ),
        "premetric": generate_fisher_flow_trajectory(
            model_premetric,
            traj_source,
            n_steps=args.n_steps,
            t_start=0.0,
            t_end=1.0,
            n_checkpoints=args.n_checkpoints,
            inf_sigma=args.inf_sigma,
            support_cells=support_cells,
            manifold_project_mode=args.manifold_project_mode,
            manifold_project_every=args.manifold_project_every,
            manifold_project_k=args.manifold_project_k,
            manifold_project_beta=args.manifold_project_beta,
            manifold_project_mix=args.manifold_project_mix,
        ),
    }
    if model_spectral_jvp is not None:
        traj_by_method["spectral_jvp"] = generate_fisher_flow_trajectory(
            model_spectral_jvp,
            traj_source,
            n_steps=args.n_steps,
            t_start=0.0,
            t_end=1.0,
            n_checkpoints=args.n_checkpoints,
            inf_sigma=args.inf_sigma,
            support_cells=support_cells,
            manifold_project_mode=args.manifold_project_mode,
            manifold_project_every=args.manifold_project_every,
            manifold_project_k=args.manifold_project_k,
            manifold_project_beta=args.manifold_project_beta,
            manifold_project_mix=args.manifold_project_mix,
        )

    metrics = {
        "config": {
            "target_mode": args.target_mode,
            "trajectory_mode": args.trajectory_mode,
            "spectral_family": args.spectral_family,
            "weight_power": args.weight_power,
            "diffusion_time": args.diffusion_time,
            "inf_sigma": args.inf_sigma,
            "manifold_project_mode": args.manifold_project_mode,
            "manifold_project_every": int(args.manifold_project_every),
            "manifold_project_k": int(args.manifold_project_k),
            "manifold_project_beta": float(args.manifold_project_beta),
            "manifold_project_mix": float(args.manifold_project_mix),
            "eval_rollouts": eval_rollouts,
            "traj_rollouts": traj_rollouts,
            "include_spectral_jvp": bool(args.include_spectral_jvp),
            "spectral_jvp_velocity_norm_weight": float(args.spectral_jvp_velocity_norm_weight),
        },
        "slerp": {
            "mmd2": float(mmd_rbf(pred_slerp, test_target_comp)),
            "fgd": float(fgd(pred_slerp, test_target_comp)),
            "final_loss": float(losses_slerp[-1]),
        },
        "premetric": {
            "mmd2": float(mmd_rbf(pred_premetric, test_target_comp)),
            "fgd": float(fgd(pred_premetric, test_target_comp)),
            "final_loss": float(losses_premetric[-1]),
        },
    }
    if model_spectral_jvp is not None:
        metrics["spectral_jvp"] = {
            "mmd2": float(mmd_rbf(pred_spectral_jvp, test_target_comp)),
            "fgd": float(fgd(pred_spectral_jvp, test_target_comp)),
            "final_loss": float(losses_spectral_jvp[-1]),
            "encoder_diagnostics": getattr(model_spectral_jvp, "spectral_jvp_diagnostics", {}),
        }
    metrics["slerp"]["endpoint_diagnostics"] = endpoint_diagnostics(
        pred_by_method["slerp"],
        target_test.simplex,
        target_mode=args.target_mode,
    )
    metrics["premetric"]["endpoint_diagnostics"] = endpoint_diagnostics(
        pred_by_method["premetric"],
        target_test.simplex,
        target_mode=args.target_mode,
    )
    if model_spectral_jvp is not None:
        metrics["spectral_jvp"]["endpoint_diagnostics"] = endpoint_diagnostics(
            pred_by_method["spectral_jvp"],
            target_test.simplex,
            target_mode=args.target_mode,
        )

    plot_endpoints(source_test, target_test, pred_by_method, out_dir / "toy_y_flow_endpoints.png")
    plot_trajectories_simplex(
        source_test,
        target_test,
        traj_by_method,
        out_dir / "toy_y_flow_trajectories_simplex.png",
        rollout_group_size=traj_rollouts,
    )
    plot_trajectories_orthant(
        source_test,
        target_test,
        traj_by_method,
        out_dir / "toy_y_flow_trajectories_orthant.png",
        rollout_group_size=traj_rollouts,
    )

    save_payload = {
        "source_test": source_test.simplex,
        "target_test": target_test.simplex,
        "pred_slerp": pred_by_method["slerp"],
        "pred_premetric": pred_by_method["premetric"],
        "traj_slerp": np.stack([pos.numpy() for _, pos in traj_by_method["slerp"]], axis=0),
        "traj_premetric": np.stack([pos.numpy() for _, pos in traj_by_method["premetric"]], axis=0),
        "losses_slerp": np.asarray(losses_slerp, dtype=np.float32),
        "losses_premetric": np.asarray(losses_premetric, dtype=np.float32),
    }
    if model_spectral_jvp is not None:
        save_payload.update(
            {
                "pred_spectral_jvp": pred_by_method["spectral_jvp"],
                "traj_spectral_jvp": np.stack(
                    [pos.numpy() for _, pos in traj_by_method["spectral_jvp"]],
                    axis=0,
                ),
                "losses_spectral_jvp": np.asarray(losses_spectral_jvp, dtype=np.float32),
            }
        )
    np.savez_compressed(out_dir / "toy_y_flow_outputs.npz", **save_payload)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nToy Y FM summary")
    print(f"  SLERP:      MMD^2={metrics['slerp']['mmd2']:.4f}  FGD={metrics['slerp']['fgd']:.4f}")
    print(f"  Premetric:  MMD^2={metrics['premetric']['mmd2']:.4f}  FGD={metrics['premetric']['fgd']:.4f}")
    if model_spectral_jvp is not None:
        print(
            "  SpectralJVP:"
            f" MMD^2={metrics['spectral_jvp']['mmd2']:.4f}"
            f"  FGD={metrics['spectral_jvp']['fgd']:.4f}"
        )
    print("\nEndpoint diagnostics (xy / branch manifold)")
    print_endpoint_diagnostics("SLERP", metrics["slerp"]["endpoint_diagnostics"])
    print_endpoint_diagnostics("Premetric", metrics["premetric"]["endpoint_diagnostics"])
    if model_spectral_jvp is not None:
        print_endpoint_diagnostics(
            "SpectralJVP",
            metrics["spectral_jvp"]["endpoint_diagnostics"],
        )
    print(f"  Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
