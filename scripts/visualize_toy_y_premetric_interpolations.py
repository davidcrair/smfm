#!/usr/bin/env python
"""Visualize spectral-OT premetric interpolations on a two-marginal Y toy.

This script does not train a flow model. It builds a Y-shaped dataset on the
2-simplex, maps it to the positive orthant with x=sqrt(p), solves OT using the
same global spectral premetric cost used by
MM+PremetricBiharmonic-SpectralOT, and visualizes the resulting trajectories.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from surf.geometry.premetric import SpectralPremetric
from surf.geometry.sphere import from_orthant, normalize_sphere, to_orthant
from surf.runtime import setup


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
class YToy:
    source_simplex: np.ndarray
    target_simplex: np.ndarray
    source_xy: np.ndarray
    target_xy: np.ndarray
    target_branch: np.ndarray


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


def make_y_toy(
    *,
    n_source: int,
    n_target_per_branch: int,
    noise: float,
    seed: int,
) -> YToy:
    """Create a two-marginal Y toy inside the 2-simplex.

    The source marginal is the bottom trunk. The target marginal consists of
    the two upper branches and includes points near the junction so the kNN
    graph is connected through the Y.
    """
    rng = np.random.default_rng(seed)

    root = np.array([0.50, 0.06], dtype=np.float64)
    junction = np.array([0.50, 0.36], dtype=np.float64)
    left_leaf = np.array([0.34, 0.58], dtype=np.float64)
    right_leaf = np.array([0.66, 0.58], dtype=np.float64)

    source_u = rng.uniform(0.00, 1.00, size=n_source)
    source_xy = _line_points(root, junction, source_u, noise=noise, rng=rng)

    left_u = rng.uniform(0.00, 1.00, size=n_target_per_branch)
    right_u = rng.uniform(0.00, 1.00, size=n_target_per_branch)
    left_xy = _line_points(junction, left_leaf, left_u, noise=noise, rng=rng)
    right_xy = _line_points(junction, right_leaf, right_u, noise=noise, rng=rng)

    target_xy = np.vstack([left_xy, right_xy])
    target_branch = np.concatenate(
        [
            np.ones(n_target_per_branch, dtype=np.int64),
            np.full(n_target_per_branch, 2, dtype=np.int64),
        ]
    )

    source_simplex = xy_to_simplex(source_xy)
    target_simplex = xy_to_simplex(target_xy)
    return YToy(
        source_simplex=source_simplex,
        target_simplex=target_simplex,
        source_xy=simplex_to_xy(source_simplex),
        target_xy=simplex_to_xy(target_simplex),
        target_branch=target_branch,
    )


def _solve_transport_plan(cost: np.ndarray, *, num_iter_max: int) -> np.ndarray:
    import ot

    n, m = cost.shape
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    plan = ot.emd(a, b, cost.astype(np.float64), numItermax=num_iter_max)
    plan = np.maximum(plan, 0.0)
    return plan / plan.sum()


def sample_plan_pairs(
    plan: np.ndarray,
    *,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat = plan.reshape(-1)
    flat = flat / flat.sum()
    flat_idx = rng.choice(len(flat), size=n_paths, replace=True, p=flat)
    src_idx = flat_idx // plan.shape[1]
    tgt_idx = flat_idx % plan.shape[1]
    return src_idx.astype(np.int64), tgt_idx.astype(np.int64)


def slerp_paths(y0: torch.Tensor, y1: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    paths = []
    for tau in t_grid:
        t = torch.full((len(y0), 1), float(tau.item()), device=y0.device)
        cos_omega = (y0 * y1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        z = (
            torch.sin((1.0 - t) * omega) / sin_omega * y0
            + torch.sin(t * omega) / sin_omega * y1
        )
        paths.append(normalize_sphere(z))
    return torch.stack(paths, dim=1)


def premetric_paths(
    premetric: SpectralPremetric,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    t_grid: torch.Tensor,
    *,
    ode_steps: int,
    path_cache=None,
) -> torch.Tensor:
    paths = []
    pair_indices = np.arange(len(src_idx), dtype=np.int64)
    for tau in t_grid:
        t = torch.full((len(src_idx), 1), float(tau.item()), device=premetric.device)
        z_t, _ = premetric.sample_trajectory(
            0,
            src_idx,
            1,
            tgt_idx,
            t,
            n_steps=ode_steps,
            path_cache=path_cache,
            pair_indices=pair_indices,
        )
        paths.append(z_t)
    return torch.stack(paths, dim=1)


def legacy_spectral_decode_reference(
    premetric: SpectralPremetric,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Reference path: interpolate in spectral coordinates and decode by NN.

    This is not the training target. It is a visual sanity check for the graph
    premetric itself before the Chen-Lipman ODE approximation is involved.
    """
    src_idx_t = torch.as_tensor(src_idx, device=premetric.device, dtype=torch.long)
    tgt_idx_t = torch.as_tensor(tgt_idx, device=premetric.device, dtype=torch.long)
    e0 = premetric.stage_embeddings[0][src_idx_t]
    e1 = premetric.stage_embeddings[1][tgt_idx_t]
    paths = []
    for tau in t_grid:
        e_tau = (1.0 - tau) * e0 + tau * e1
        dist = torch.cdist(e_tau, premetric.all_embeddings)
        nn_idx = dist.argmin(dim=1)
        paths.append(premetric.all_cells[nn_idx])
    return torch.stack(paths, dim=1)

def sphere_paths_to_simplex(paths: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        simplex = from_orthant(paths).detach().cpu().numpy()
    simplex = np.clip(simplex, 0.0, None)
    simplex /= simplex.sum(axis=-1, keepdims=True).clip(min=1e-12)
    return simplex


def paths_to_xy(paths: torch.Tensor) -> np.ndarray:
    simplex = sphere_paths_to_simplex(paths)
    flat_xy = simplex_to_xy(simplex.reshape(-1, 3))
    return flat_xy.reshape(simplex.shape[0], simplex.shape[1], 2)


def _draw_simplex(ax) -> None:
    triangle = np.vstack([TRIANGLE_VERTICES, TRIANGLE_VERTICES[0]])
    ax.plot(triangle[:, 0], triangle[:, 1], c="black", linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, SQRT3 / 2.0 + 0.04)
    ax.set_aspect("equal")


def _draw_dataset(ax, toy: YToy, *, source_alpha: float = 0.35, target_alpha: float = 0.35) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    ax.scatter(
        toy.source_xy[:, 0],
        toy.source_xy[:, 1],
        s=13,
        c="#555555",
        alpha=source_alpha,
        edgecolor="none",
        label="source trunk",
    )
    for branch_id in (1, 2):
        mask = toy.target_branch == branch_id
        ax.scatter(
            toy.target_xy[mask, 0],
            toy.target_xy[mask, 1],
            s=13,
            c=branch_colors[branch_id],
            alpha=target_alpha,
            edgecolor="none",
            label=f"target branch {branch_id}",
        )


def plot_dataset(toy: YToy, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    _draw_simplex(ax)
    _draw_dataset(ax, toy, source_alpha=0.70, target_alpha=0.70)
    ax.set_title("Two-Marginal Y Toy")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pairings(
    toy: YToy,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    out_path: Path,
) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    fig, ax = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    _draw_simplex(ax)
    _draw_dataset(ax, toy, source_alpha=0.35, target_alpha=0.55)
    for s, t in zip(src_idx, tgt_idx):
        color = branch_colors[int(toy.target_branch[t])]
        ax.plot(
            [toy.source_xy[s, 0], toy.target_xy[t, 0]],
            [toy.source_xy[s, 1], toy.target_xy[t, 1]],
            color=color,
            alpha=0.20,
            linewidth=0.8,
        )
    ax.set_title("Sampled Spectral OT Pairs")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_interpolation_comparison(
    toy: YToy,
    slerp_xy: np.ndarray,
    premetric_xy: np.ndarray,
    tgt_idx: np.ndarray,
    out_path: Path,
    *,
    legacy_xy: np.ndarray | None = None,
    premetric_title: str,
) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    panels = [(slerp_xy, "SLERP on Same Spectral OT Pairs")]
    if legacy_xy is not None:
        panels.append((legacy_xy, "Spectral Linear Decode\nReference Only"))
    panels.append((premetric_xy, premetric_title))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.8), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (path_xy, title) in zip(axes, panels):
        _draw_simplex(ax)
        _draw_dataset(ax, toy, source_alpha=0.18, target_alpha=0.24)
        for i in range(path_xy.shape[0]):
            color = branch_colors[int(toy.target_branch[tgt_idx[i]])]
            ax.plot(path_xy[i, :, 0], path_xy[i, :, 1], color=color, alpha=0.34, linewidth=1.0)
            ax.scatter(path_xy[i, 0, 0], path_xy[i, 0, 1], s=10, c="#444444", alpha=0.35, edgecolor="none")
            ax.scatter(path_xy[i, -1, 0], path_xy[i, -1, 1], s=10, c=color, alpha=0.45, edgecolor="none")
        ax.set_title(title)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_premetric_time_slices(
    toy: YToy,
    premetric_xy: np.ndarray,
    t_grid: np.ndarray,
    tgt_idx: np.ndarray,
    out_path: Path,
) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    keep = np.linspace(0, len(t_grid) - 1, min(6, len(t_grid)), dtype=int)
    fig, axes = plt.subplots(1, len(keep), figsize=(3.2 * len(keep), 3.6), constrained_layout=True)
    if len(keep) == 1:
        axes = [axes]
    for ax, step_idx in zip(axes, keep):
        _draw_simplex(ax)
        _draw_dataset(ax, toy, source_alpha=0.10, target_alpha=0.12)
        for branch_id in (1, 2):
            mask = toy.target_branch[tgt_idx] == branch_id
            ax.scatter(
                premetric_xy[mask, step_idx, 0],
                premetric_xy[mask, step_idx, 1],
                s=18,
                c=branch_colors[branch_id],
                alpha=0.72,
                edgecolor="none",
            )
        ax.set_title(f"tau={t_grid[step_idx]:.2f}")
    fig.suptitle("Premetric Interpolation Time Slices", fontsize=13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_orthant_paths(
    source: torch.Tensor,
    target: torch.Tensor,
    premetric_paths_t: torch.Tensor,
    tgt_idx: np.ndarray,
    toy: YToy,
    out_path: Path,
) -> None:
    branch_colors = {1: "#0072b2", 2: "#d55e00"}
    paths_np = premetric_paths_t.detach().cpu().numpy()
    source_np = source.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    min_coord = min(float(paths_np.min()), 0.0)

    fig = plt.figure(figsize=(7.0, 6.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(source_np[:, 0], source_np[:, 1], source_np[:, 2], s=10, c="#555555", alpha=0.22, depthshade=False)
    for branch_id in (1, 2):
        mask = toy.target_branch == branch_id
        ax.scatter(
            target_np[mask, 0],
            target_np[mask, 1],
            target_np[mask, 2],
            s=10,
            c=branch_colors[branch_id],
            alpha=0.30,
            depthshade=False,
        )
    for i in range(paths_np.shape[0]):
        color = branch_colors[int(toy.target_branch[tgt_idx[i]])]
        ax.plot(paths_np[i, :, 0], paths_np[i, :, 1], paths_np[i, :, 2], color=color, alpha=0.38, linewidth=1.0)
    lo = min(-0.05, min_coord - 0.02)
    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)
    ax.set_zlim(lo, 1.02)
    ax.set_xlabel("sqrt(p1)")
    ax.set_ylabel("sqrt(p2)")
    ax.set_zlabel("sqrt(p3)")
    ax.view_init(elev=26, azim=38)
    ax.set_title("Premetric Paths on the Orthant Sphere")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_decay_diagnostics(
    premetric: SpectralPremetric,
    y0: torch.Tensor,
    tgt_idx: np.ndarray,
    paths: torch.Tensor,
    t_grid: torch.Tensor,
) -> dict[str, float]:
    target_idx = torch.as_tensor(tgt_idx, device=premetric.device, dtype=torch.long)
    target_emb = premetric.target_embeddings(1, target_idx)
    ratios = []
    with torch.no_grad():
        d0 = premetric.distance(y0, target_emb).clamp(min=premetric.eps)
        for step in range(paths.shape[1]):
            d_tau = premetric.distance(paths[:, step, :], target_emb)
            ratios.append((d_tau / d0).squeeze(-1).detach().cpu().numpy())
    ratio = np.stack(ratios, axis=1)
    ideal = 1.0 - t_grid.detach().cpu().numpy()[None, :]
    abs_error = np.abs(ratio - ideal)
    paths_np = paths.detach().cpu().numpy()
    return {
        "rho_decay_abs_error_mean": float(abs_error.mean()),
        "rho_decay_abs_error_q95": float(np.quantile(abs_error, 0.95)),
        "min_coord_min": float(paths_np.min()),
        "min_coord_q01": float(np.quantile(paths_np, 0.01)),
        "negative_fraction": float(np.mean(paths_np < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-source", type=int, default=220)
    parser.add_argument("--n-target-per-branch", type=int, default=110)
    parser.add_argument("--noise", type=float, default=0.008)
    parser.add_argument("--n-paths", type=int, default=90)
    parser.add_argument("--n-time", type=int, default=30)
    parser.add_argument("--max-t", type=float, default=0.95)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--n-eig", type=int, default=32)
    parser.add_argument("--spectral-family", choices=["power", "diffusion"], default="power")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--diffusion-time", type=float, default=1.0)
    parser.add_argument("--extension-k", type=int, default=64)
    parser.add_argument("--softmax-beta", type=float, default=10.0)
    parser.add_argument(
        "--trajectory-mode",
        choices=["graph_geodesic", "spectral_decode", "spectral_decode_arclength", "ode"],
        default="graph_geodesic",
    )
    parser.add_argument("--decode-k", type=int, default=64)
    parser.add_argument("--decode-beta", type=float, default=10.0)
    parser.add_argument("--velocity-fd-eps", type=float, default=0.02)
    parser.add_argument("--ode-steps", type=int, default=32)
    parser.add_argument("--time-cap", type=float, default=0.95)
    parser.add_argument("--grad-norm-floor", type=float, default=0.05)
    parser.add_argument("--max-drive-scale", type=float, default=50.0)
    parser.add_argument("--emd-num-iter", type=int, default=1000000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="plots/toy_y_premetric")
    args = parser.parse_args()

    rt = setup(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    toy = make_y_toy(
        n_source=args.n_source,
        n_target_per_branch=args.n_target_per_branch,
        noise=args.noise,
        seed=args.seed,
    )
    source = normalize_sphere(to_orthant(torch.tensor(toy.source_simplex, dtype=torch.float32))).to(rt.device)
    target = normalize_sphere(to_orthant(torch.tensor(toy.target_simplex, dtype=torch.float32))).to(rt.device)

    premetric = SpectralPremetric(
        [source, target],
        knn_graph=args.knn,
        n_eig=args.n_eig,
        spectral_family=args.spectral_family,
        weight_power=args.weight_power,
        diffusion_time=args.diffusion_time,
        extension_k=args.extension_k,
        softmax_beta=args.softmax_beta,
        trajectory_mode=args.trajectory_mode,
        decode_k=args.decode_k,
        decode_beta=args.decode_beta,
        velocity_fd_eps=args.velocity_fd_eps,
        time_cap=args.time_cap,
        grad_norm_floor=args.grad_norm_floor,
        max_drive_scale=args.max_drive_scale,
    )

    cost = premetric.stage_pair_cost(0, 1)
    plan = _solve_transport_plan(cost, num_iter_max=args.emd_num_iter)
    src_idx, tgt_idx = sample_plan_pairs(plan, n_paths=args.n_paths, seed=args.seed + 1)
    path_cache = None
    if args.trajectory_mode == "graph_geodesic":
        path_cache = premetric.prepare_interval_paths(0, 1, src_idx, tgt_idx)

    pair_source = source[torch.as_tensor(src_idx, device=rt.device, dtype=torch.long)]
    pair_target = target[torch.as_tensor(tgt_idx, device=rt.device, dtype=torch.long)]
    t_grid = torch.linspace(0.0, min(args.max_t, args.time_cap), args.n_time, device=rt.device)

    with torch.no_grad():
        slerp = slerp_paths(pair_source, pair_target, t_grid)
        premetric_path = premetric_paths(
            premetric,
            src_idx,
            tgt_idx,
            t_grid,
            ode_steps=args.ode_steps,
            path_cache=path_cache,
        )
        legacy_reference = None
        if args.trajectory_mode in ("graph_geodesic", "spectral_decode_arclength", "ode"):
            legacy_reference = legacy_spectral_decode_reference(
                premetric, src_idx, tgt_idx, t_grid
            )

    slerp_xy = paths_to_xy(slerp)
    premetric_xy = paths_to_xy(premetric_path)
    diagnostics = compute_decay_diagnostics(premetric, pair_source, tgt_idx, premetric_path, t_grid)
    diagnostics.update(
        {
            "n_source": int(args.n_source),
            "n_target": int(2 * args.n_target_per_branch),
            "n_paths": int(args.n_paths),
            "left_target_pair_fraction": float(np.mean(toy.target_branch[tgt_idx] == 1)),
            "right_target_pair_fraction": float(np.mean(toy.target_branch[tgt_idx] == 2)),
            "cost_min": float(cost.min()),
            "cost_median": float(np.median(cost)),
            "cost_max": float(cost.max()),
            "trajectory_mode": args.trajectory_mode,
            "decode_k": int(args.decode_k),
            "decode_beta": float(args.decode_beta),
            "velocity_fd_eps": float(args.velocity_fd_eps),
        }
    )

    plot_dataset(toy, out_dir / "toy_y_dataset.png")
    plot_pairings(toy, src_idx, tgt_idx, out_dir / "toy_y_spectral_ot_pairings.png")
    plot_interpolation_comparison(
        toy,
        slerp_xy,
        premetric_xy,
        tgt_idx,
        out_dir / "toy_y_slerp_vs_premetric_interpolations.png",
        legacy_xy=(paths_to_xy(legacy_reference) if legacy_reference is not None else None),
        premetric_title=(
            "Premetric Trajectory, Graph Geodesic"
            if args.trajectory_mode == "graph_geodesic"
            else (
                "Premetric Trajectory, Spectral Decode Arc-Length"
                if args.trajectory_mode == "spectral_decode_arclength"
                else (
                "Premetric Trajectory, Spectral OT"
                if args.trajectory_mode == "spectral_decode"
                else "Premetric ODE, Spectral OT"
                )
            )
        ),
    )
    plot_premetric_time_slices(
        toy,
        premetric_xy,
        t_grid.detach().cpu().numpy(),
        tgt_idx,
        out_dir / "toy_y_premetric_time_slices.png",
    )
    plot_orthant_paths(
        source,
        target,
        premetric_path,
        tgt_idx,
        toy,
        out_dir / "toy_y_premetric_orthant_paths.png",
    )

    np.savez_compressed(
        out_dir / "toy_y_premetric_outputs.npz",
        source_simplex=toy.source_simplex,
        target_simplex=toy.target_simplex,
        target_branch=toy.target_branch,
        spectral_ot_cost=cost,
        spectral_ot_plan=plan,
        src_idx=src_idx,
        tgt_idx=tgt_idx,
        t_grid=t_grid.detach().cpu().numpy(),
        slerp_simplex=sphere_paths_to_simplex(slerp),
        premetric_simplex=sphere_paths_to_simplex(premetric_path),
        premetric_sphere=premetric_path.detach().cpu().numpy(),
    )
    with (out_dir / "diagnostics.json").open("w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Saved outputs to {out_dir}")
    print(
        "Diagnostics: "
        f"rho_decay_abs_error_mean={diagnostics['rho_decay_abs_error_mean']:.4f}, "
        f"rho_decay_abs_error_q95={diagnostics['rho_decay_abs_error_q95']:.4f}, "
        f"min_coord_min={diagnostics['min_coord_min']:.4f}, "
        f"negative_fraction={diagnostics['negative_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
