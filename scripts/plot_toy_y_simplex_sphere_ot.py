"""Visualize the 3-marginal Y toy on the 2D simplex, the positive orthant of
the 2-sphere, and OT couplings / learned trajectories for sphere, squared
spectral, and true biharmonic Fisher-flow variants.

Usage:
    uv run python scripts/plot_toy_y_simplex_sphere_ot.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import plotly.graph_objects as go
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_toy_y_3marginal import make_dataset  # noqa: E402

from surf.evaluation.generation import generate_fisher_flow, generate_fisher_flow_trajectory  # noqa: E402
from surf.evaluation.metrics import fgd, mmd_rbf, swd  # noqa: E402
from surf.geometry.sphere import compute_sphere_cost_matrix, from_orthant, to_orthant  # noqa: E402
from surf.ot.costs import make_biharmonic_cost_fn, make_spectral_cost_fn  # noqa: E402
from surf.runtime import setup  # noqa: E402
from surf.training.flow_trainer import train_multi_marginal_flow  # noqa: E402


SQRT3 = np.sqrt(3.0)
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3 / 2.0]])

# Affine map from raw Y xy into the simplex triangle; chosen so the tips,
# junction and base serif all land strictly inside the triangle.
SIMPLEX_CX = 0.5
SIMPLEX_CY = 0.26
SIMPLEX_S = 0.115

COLORS = {"t=0": "#1f77b4", "t=0.5": "#2ca02c", "t=1": "#d62728"}

PRESENTATION_RC = {
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

TRAIN_LOG_EVERY = 200


def raw_to_triangle(xy):
    out = np.empty_like(xy)
    out[:, 0] = SIMPLEX_CX + SIMPLEX_S * xy[:, 0]
    out[:, 1] = SIMPLEX_CY + SIMPLEX_S * xy[:, 1]
    return out


def triangle_to_simplex(xy):
    x, y = xy[:, 0], xy[:, 1]
    p3 = y / (SQRT3 / 2.0)
    p2 = x - 0.5 * p3
    p1 = 1.0 - p2 - p3
    p = np.stack([p1, p2, p3], axis=1)
    p = np.clip(p, 1e-8, None)
    return p / p.sum(axis=1, keepdims=True)


def draw_triangle(ax):
    tri = np.vstack([TRIANGLE, TRIANGLE[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.25", lw=2.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, SQRT3 / 2.0 + 0.05)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_octant_sphere(ax):
    u = np.linspace(0, np.pi / 2, 40)
    v = np.linspace(0, np.pi / 2, 40)
    uu, vv = np.meshgrid(u, v)
    x = np.sin(vv) * np.cos(uu)
    y = np.sin(vv) * np.sin(uu)
    z = np.cos(vv)
    ax.plot_surface(x, y, z, color="0.9", alpha=0.25, linewidth=0, antialiased=True)
    for c in np.linspace(0, np.pi / 2, 7):
        ax.plot(
            np.sin(c) * np.cos(u), np.sin(c) * np.sin(u), np.full_like(u, np.cos(c)),
            color="0.7", lw=0.5,
        )
        ax.plot(
            np.sin(u) * np.cos(c), np.sin(u) * np.sin(c), np.cos(u),
            color="0.7", lw=0.5,
        )
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("$\\sqrt{p_1}$")
    ax.set_ylabel("$\\sqrt{p_2}$")
    ax.set_zlabel("$\\sqrt{p_3}$")
    ax.view_init(elev=25, azim=35)


def slerp_arc(y0, y1, n_points=30):
    cos_om = np.clip(y0 @ y1, -1 + 1e-6, 1 - 1e-6)
    om = np.arccos(cos_om)
    if om < 1e-6:
        return np.tile(y0, (n_points, 1))
    s = np.linspace(0.0, 1.0, n_points)[:, None]
    a = np.sin((1 - s) * om) / np.sin(om)
    b = np.sin(s * om) / np.sin(om)
    return a * y0[None, :] + b * y1[None, :]


def _sample_slerp_tangents(y0_arr, y1_arr, T, n_per_arc=25):
    """Sample (xy, unit_tangent) pairs from every coupled SLERP interpolant."""
    idx0, idx1 = np.where(T > 1e-12)
    xys, vs = [], []
    for i, j in zip(idx0, idx1):
        arc = slerp_arc(y0_arr[i], y1_arr[j], n_points=n_per_arc + 1)
        xy = (arc ** 2) @ TRIANGLE
        dxy = np.diff(xy, axis=0)
        mid = 0.5 * (xy[:-1] + xy[1:])
        xys.append(mid)
        vs.append(dxy)
    if not xys:
        return np.empty((0, 2)), np.empty((0, 2))
    xys = np.vstack(xys)
    vs = np.vstack(vs)
    norms = np.linalg.norm(vs, axis=1, keepdims=True).clip(min=1e-12)
    return xys, vs / norms


def _inside_triangle(xy, eps=0.01):
    p3 = xy[..., 1] / (SQRT3 / 2.0)
    p2 = xy[..., 0] - 0.5 * p3
    p1 = 1.0 - p2 - p3
    return (p1 >= eps) & (p2 >= eps) & (p3 >= eps)


def _draw_arc_backdrop(
    ax, T_list, max_lines_per_hop=120, n_arc=40,
    arrow_t_range=(0.25, 0.8), arrow_rng_seed=0,
):
    """Draw the raw SLERP interpolants with a direction arrow at a jittered t."""
    rng = np.random.default_rng(arrow_rng_seed)
    for y0_arr, y1_arr, T in T_list:
        idx0, idx1 = np.where(T > 1e-12)
        if len(idx0) > max_lines_per_hop:
            weights = T[idx0, idx1]
            keep = np.argsort(weights)[-max_lines_per_hop:]
            idx0, idx1 = idx0[keep], idx1[keep]
        t_lo, t_hi = arrow_t_range
        ts = rng.uniform(t_lo, t_hi, size=len(idx0))
        for i, j, t in zip(idx0, idx1, ts):
            arc = slerp_arc(y0_arr[i], y1_arr[j], n_points=n_arc)
            xy = (arc ** 2) @ TRIANGLE
            ax.plot(xy[:, 0], xy[:, 1], color="0.1", lw=1.0, alpha=0.7, zorder=1)

            k = int(np.clip(t * (n_arc - 1), 1, n_arc - 2))
            ax.annotate(
                "", xy=xy[k + 1], xytext=xy[k - 1],
                arrowprops=dict(
                    arrowstyle="-|>", color="0.1", lw=1.1,
                    mutation_scale=14, alpha=0.45,
                ),
                zorder=3,
            )


def draw_flow_field(
    ax, y0_arr, y1_arr, T_list, *,
    n_bins=24, min_count=3, min_coherence=0.35, arrow_length=0.035,
):
    """Grid-averaged SLERP velocity field colored by direction (HSV hue = angle).

    Each coupled (y0_i, y1_j) pair contributes (t, x_t, v_t) samples along
    the SLERP interpolant — the very tuples that flow matching trains on.
    We bin those samples onto a grid, take the mean unit tangent per cell,
    and plot one arrow per cell colored by its angle.
    """
    xys_all, vs_all = [], []
    for y0, y1, T in T_list:
        xys, vs = _sample_slerp_tangents(y0, y1, T)
        xys_all.append(xys)
        vs_all.append(vs)
    xys = np.vstack(xys_all)
    vs = np.vstack(vs_all)

    x_edges = np.linspace(-0.05, 1.05, n_bins + 1)
    y_edges = np.linspace(-0.05, SQRT3 / 2.0 + 0.05, n_bins + 1)
    ix = np.clip(np.searchsorted(x_edges, xys[:, 0]) - 1, 0, n_bins - 1)
    iy = np.clip(np.searchsorted(y_edges, xys[:, 1]) - 1, 0, n_bins - 1)

    grid_sum = np.zeros((n_bins, n_bins, 2))
    counts = np.zeros((n_bins, n_bins))
    np.add.at(grid_sum, (iy, ix), vs)
    np.add.at(counts, (iy, ix), 1)
    with np.errstate(invalid="ignore"):
        mean_vec = np.where(counts[..., None] > 0, grid_sum / counts[..., None].clip(min=1), 0.0)
    coherence = np.linalg.norm(mean_vec, axis=-1)

    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(xc, yc)
    grid_xy = np.stack([X, Y], axis=-1)

    mask = (counts >= min_count) & (coherence >= min_coherence) & _inside_triangle(grid_xy)
    x_pts, y_pts = X[mask], Y[mask]
    u, v = mean_vec[..., 0][mask], mean_vec[..., 1][mask]
    u_dir = u / np.sqrt(u * u + v * v).clip(min=1e-8) * arrow_length
    v_dir = v / np.sqrt(u * u + v * v).clip(min=1e-8) * arrow_length

    # Color arrows by angle (hue) so same-direction regions share color.
    import matplotlib.colors as mcolors
    ang = np.arctan2(v, u)
    hues = (ang + np.pi) / (2 * np.pi)
    sat = 0.75 * (coherence[mask] - min_coherence) / (1.0 - min_coherence + 1e-8)
    sat = np.clip(0.4 + sat, 0.4, 0.95)
    rgb = mcolors.hsv_to_rgb(np.stack([hues, sat, np.full_like(hues, 0.9)], axis=-1))

    ax.quiver(
        x_pts, y_pts, u_dir, v_dir,
        color=rgb, angles="xy", scale_units="xy", scale=1.0,
        width=0.006, headwidth=3.5, headlength=4.5, zorder=1,
    )


def draw_predicted_paths(ax, trajectory, max_paths=80, rng_seed=0):
    """Draw integrated flow-model trajectories on the simplex.

    trajectory: list of (t, positions_comp) tuples where positions_comp is a
    (n_cells, D) torch/numpy tensor of simplex probabilities. Generated by
    ``generate_fisher_flow_trajectory``.
    """
    ckpts = [(p.numpy() if hasattr(p, "numpy") else np.asarray(p)) for _, p in trajectory]
    stacked = np.stack(ckpts, axis=1)  # (n_cells, n_ckpt, D)
    xy_per_cell = stacked @ TRIANGLE  # (n_cells, n_ckpt, 2)

    rng = np.random.default_rng(rng_seed)
    n_cells = xy_per_cell.shape[0]
    if n_cells > max_paths:
        idx = rng.choice(n_cells, max_paths, replace=False)
        xy_per_cell = xy_per_cell[idx]

    for xy in xy_per_cell:
        ax.plot(xy[:, 0], xy[:, 1], color="0.1", lw=1.0, alpha=0.55, zorder=4)
        n_pts = len(xy)
        t = rng.uniform(0.25, 0.8)
        k = int(np.clip(t * (n_pts - 1), 1, n_pts - 2))
        ax.annotate(
            "", xy=xy[k + 1], xytext=xy[k - 1],
            arrowprops=dict(
                arrowstyle="-|>", color="0.1", lw=1.1,
                mutation_scale=14, alpha=0.6,
            ),
            zorder=5,
        )


def write_loss_curves_plotly(loss_map, out_path):
    """Write training loss curves to an interactive Plotly HTML file."""
    fig = go.Figure()
    colors = {
        "MM+SLERP": "#111111",
        "MM+SLERP+SquaredSpectral": "#d62728",
        "MM+SLERP+Biharmonic": "#ff7f0e",
    }

    for name, losses in loss_map.items():
        if not losses:
            continue
        iters = np.arange(len(losses), dtype=np.int64) * TRAIN_LOG_EVERY
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=losses,
                mode="lines+markers",
                name=name,
                line=dict(color=colors.get(name, "#666666"), width=3),
                marker=dict(size=8),
                hovertemplate="iter=%{x}<br>loss=%{y:.6f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title="Toy Y Training Loss Curves",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        template="plotly_white",
        width=1000,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=70, r=30, t=80, b=70),
    )
    fig.write_html(out_path)
    print(f"  Wrote {Path(out_path).resolve()}")


def ot_plan(cost_matrix):
    n0, n1 = cost_matrix.shape
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    M = np.ascontiguousarray(cost_matrix, dtype=np.float64)
    return ot.emd(a, b, M)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=120)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--biharm-knn", type=int, default=15)
    parser.add_argument("--biharm-neig", type=int, default=30)
    parser.add_argument("--spectral-weight-power", type=float, default=0.5,
                        help="Weight power for MM+SLERP+SquaredSpectral.")
    parser.add_argument("--out", default="toy_y_simplex_sphere_ot.png")
    parser.add_argument("--loss-out", default=None,
                        help="Optional Plotly HTML path for training loss curves.")
    parser.add_argument("--skip-predictions", action="store_true",
                        help="Skip training + predicted-path row 3 (faster).")
    parser.add_argument("--train-iters", type=int, default=1500)
    parser.add_argument("--n-viz-paths", type=int, default=80)
    parser.add_argument("--n-gen-steps", type=int, default=50)
    parser.add_argument("--si-sigma", type=float, default=0.05,
                        help="MM+SI training-time Brownian arc length (radians).")
    parser.add_argument("--si-inf-sigmas", default="0.025,0.05",
                        help="Comma-separated SDE inference noises to visualize "
                             "alongside the deterministic ODE rollout. Keep these "
                             "small relative to the data's sphere spread (~0.3 rad).")
    parser.add_argument("--si-stochastic-sources", type=int, default=8,
                        help="Number of source cells used for stochastic-replicate panels.")
    parser.add_argument("--si-stochastic-replicates", type=int, default=12,
                        help="Stochastic samples per source cell in the SDE panels.")
    parser.add_argument("--skip-si", action="store_true",
                        help="Skip MM+SI training and the row 4 panels.")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip MMD^2/FGD/SWD evaluation.")
    args = parser.parse_args()

    plt.rcParams.update(PRESENTATION_RC)

    m0, m1, m2 = make_dataset(args.n_per_class, args.sigma, args.seed)
    tri0, tri1, tri2 = [raw_to_triangle(m) for m in (m0, m1, m2)]
    p0, p1, p2 = [triangle_to_simplex(t) for t in (tri0, tri1, tri2)]
    y0, y1, y2 = [to_orthant(torch.from_numpy(p).float()) for p in (p0, p1, p2)]

    # --- OT couplings on two hops ---------------------------------------
    print("  Computing SLERP (sphere arc^2) cost matrices...")
    C_slerp_01 = compute_sphere_cost_matrix(y0, y1)
    C_slerp_12 = compute_sphere_cost_matrix(y1, y2)
    T_slerp_01 = ot_plan(C_slerp_01)
    T_slerp_12 = ot_plan(C_slerp_12)

    print("  Computing squared spectral cost matrices...")
    spectral_cost_fn = make_spectral_cost_fn(
        knn=args.biharm_knn,
        n_eig=args.biharm_neig,
        spectral_family="power",
        weight_power=args.spectral_weight_power,
    )
    C_spec_01 = spectral_cost_fn(y0, y1)
    C_spec_12 = spectral_cost_fn(y1, y2)
    T_spec_01 = ot_plan(C_spec_01)
    T_spec_12 = ot_plan(C_spec_12)

    print("  Computing true biharmonic cost matrices...")
    biharm_cost_fn = make_biharmonic_cost_fn(
        knn=args.biharm_knn,
        n_eig=args.biharm_neig,
    )
    C_bi_01 = biharm_cost_fn(y0, y1)
    C_bi_12 = biharm_cost_fn(y1, y2)
    T_bi_01 = ot_plan(C_bi_01)
    T_bi_12 = ot_plan(C_bi_12)

    # --- Train MM+SLERP / MM+SLERP+SquaredSpectral / MM+SLERP+Biharmonic --
    slerp_traj = spec_traj = bi_traj = None
    losses_slerp = losses_spec = losses_biharm = None
    si_ode_traj = None
    si_sde_trajs = []
    losses_si = None
    do_si = (not args.skip_predictions) and (not args.skip_si)
    si_inf_sigmas = [float(s) for s in args.si_inf_sigmas.split(",") if s.strip()] if do_si else []
    if not args.skip_predictions:
        import torch as _torch
        rt = setup("auto")
        sphere_train = [y.to(rt.device) for y in (y0, y1, y2)]
        D = sphere_train[0].shape[1]
        print("\n  Training MM+SLERP for predicted paths...")
        slerp_model, losses_slerp = train_multi_marginal_flow(
            sphere_train, [0.0, 0.5, 1.0], D,
            n_iters=args.train_iters, batch_size=512, lr=3e-4,
            ot_subsample=512, label="MM+SLERP",
        )
        print("\n  Training MM+SLERP+SquaredSpectral for predicted paths...")
        spec_model, losses_spec = train_multi_marginal_flow(
            sphere_train, [0.0, 0.5, 1.0], D,
            n_iters=args.train_iters, batch_size=512, lr=3e-4,
            ot_subsample=512, cost_fn=spectral_cost_fn, label="MM+SLERP+SquaredSpectral",
        )
        print("\n  Training MM+SLERP+Biharmonic for predicted paths...")
        bi_model, losses_biharm = train_multi_marginal_flow(
            sphere_train, [0.0, 0.5, 1.0], D,
            n_iters=args.train_iters, batch_size=512, lr=3e-4,
            ot_subsample=512, cost_fn=biharm_cost_fn, label="MM+SLERP+Biharmonic",
        )
        n_viz = min(args.n_viz_paths, len(y0))
        rng = np.random.default_rng(args.seed)
        viz_idx = rng.choice(len(y0), n_viz, replace=False)
        src = y0[viz_idx].to(rt.device)
        slerp_traj = generate_fisher_flow_trajectory(
            slerp_model, src, n_steps=60, t_start=0.0, t_end=1.0, n_checkpoints=30,
        )
        spec_traj = generate_fisher_flow_trajectory(
            spec_model, src, n_steps=60, t_start=0.0, t_end=1.0, n_checkpoints=30,
        )
        bi_traj = generate_fisher_flow_trajectory(
            bi_model, src, n_steps=60, t_start=0.0, t_end=1.0, n_checkpoints=30,
        )

        if do_si:
            print(f"\n  Training MM+SI (si_sigma={args.si_sigma}) for predicted paths...")
            si_model, losses_si = train_multi_marginal_flow(
                sphere_train, [0.0, 0.5, 1.0], D,
                n_iters=args.train_iters, batch_size=512, lr=3e-4,
                ot_subsample=512, label="MM+SI",
                si_sigma=args.si_sigma,
            )
            # Deterministic ODE rollout (matches the other 3 panels).
            si_ode_traj = generate_fisher_flow_trajectory(
                si_model, src, n_steps=60, t_start=0.0, t_end=1.0,
                n_checkpoints=30, inf_sigma=0.0,
            )
            # Stochastic SDE rollouts: replicate each of K source cells R times so
            # a single starting cell spawns distinct trajectories — this is the
            # visualization the deterministic ODE panels structurally cannot show.
            n_sde_src = min(args.si_stochastic_sources, len(viz_idx))
            sde_src_idx = viz_idx[:n_sde_src]
            sde_src = y0[sde_src_idx].to(rt.device)
            sde_src_rep = sde_src.repeat_interleave(args.si_stochastic_replicates, dim=0)
            for inf_sigma in si_inf_sigmas:
                _torch.manual_seed(args.seed)
                traj = generate_fisher_flow_trajectory(
                    si_model, sde_src_rep, n_steps=60, t_start=0.0, t_end=1.0,
                    n_checkpoints=30, inf_sigma=inf_sigma,
                )
                si_sde_trajs.append((inf_sigma, traj))

        if not args.skip_metrics:
            print("\n  Evaluating MMD^2 / FGD / SWD on held-out test seed...")
            em0, em1, em2 = make_dataset(args.n_per_class, args.sigma, args.seed + 1000)
            test_tri = [raw_to_triangle(m) for m in (em0, em1, em2)]
            test_simp = [triangle_to_simplex(t) for t in test_tri]
            test_comp = [_torch.from_numpy(p).float() for p in test_simp]
            test_sphere = [to_orthant(c) for c in test_comp]

            TIMES = [0.0, 0.5, 1.0]
            METRICS = {"MMD^2": mmd_rbf, "FGD": fgd, "SWD": swd}

            def eval_model(model, name):
                src_test = test_sphere[0].to(rt.device)
                chained, per_seg = [], []
                for i in range(1, len(TIMES)):
                    pred = generate_fisher_flow(
                        model, src_test, n_steps=args.n_gen_steps,
                        t_start=0.0, t_end=TIMES[i],
                    )
                    chained.append((from_orthant(pred), test_comp[i]))
                for i in range(len(TIMES) - 1):
                    pred = generate_fisher_flow(
                        model, test_sphere[i].to(rt.device),
                        n_steps=args.n_gen_steps,
                        t_start=TIMES[i], t_end=TIMES[i + 1],
                    )
                    per_seg.append((from_orthant(pred), test_comp[i + 1]))
                return chained, per_seg

            method_preds = {
                "MM+SLERP": eval_model(slerp_model, "MM+SLERP"),
                "MM+SLERP+SquaredSpectral": eval_model(spec_model, "MM+SLERP+SquaredSpectral"),
                "MM+SLERP+Biharmonic": eval_model(bi_model, "MM+SLERP+Biharmonic"),
            }

            def _fmt_table(title, cols, rows):
                w = 14
                print("\n" + "=" * 72)
                print(title)
                print("=" * 72)
                print(f"  {'Method':<24}" + "  ".join(f"{c:>{w}}" for c in cols))
                print("  " + "-" * (24 + (w + 2) * len(cols)))
                for name, vals in rows.items():
                    cells = "  ".join(f"{v:.4f}".rjust(w) for v in vals)
                    print(f"  {name:<24}{cells}")

            chained_cols = [f"0->{t}" for t in TIMES[1:]]
            per_seg_cols = [f"{TIMES[i]}->{TIMES[i+1]}" for i in range(len(TIMES) - 1)]

            for metric_name, metric_fn in METRICS.items():
                chained_rows = {
                    m: [float(metric_fn(p, t)) for p, t in chained]
                    for m, (chained, _) in method_preds.items()
                }
                per_seg_rows = {
                    m: [float(metric_fn(p, t)) for p, t in per_seg]
                    for m, (_, per_seg) in method_preds.items()
                }
                _fmt_table(f"CHAINED {metric_name}", chained_cols, chained_rows)
                _fmt_table(f"PER-SEGMENT {metric_name}", per_seg_cols, per_seg_rows)

    # --- Figure ----------------------------------------------------------
    if args.skip_predictions:
        n_rows = 2
    elif do_si:
        n_rows = 4
    else:
        n_rows = 3
    n_cols = 3
    fig = plt.figure(figsize=(24, 7 * n_rows))
    ax_simplex = fig.add_subplot(n_rows, n_cols, 1)
    ax_sphere = fig.add_subplot(n_rows, n_cols, 2, projection="3d")
    ax_note = fig.add_subplot(n_rows, n_cols, 3)
    ax_slerp = fig.add_subplot(n_rows, n_cols, 4)
    ax_spec = fig.add_subplot(n_rows, n_cols, 5)
    ax_bi = fig.add_subplot(n_rows, n_cols, 6)
    if not args.skip_predictions:
        ax_slerp_pred = fig.add_subplot(n_rows, n_cols, 7)
        ax_spec_pred = fig.add_subplot(n_rows, n_cols, 8)
        ax_bi_pred = fig.add_subplot(n_rows, n_cols, 9)
    if do_si:
        ax_si_ode = fig.add_subplot(n_rows, n_cols, 10)
        ax_si_sde_axes = [fig.add_subplot(n_rows, n_cols, 11 + i) for i in range(min(2, len(si_sde_trajs)))]

    scatter_kw_2d = dict(s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
    scatter_kw_3d = dict(s=55, alpha=0.95, depthshade=False, edgecolors="white", linewidths=0.5)

    # (a) 2D simplex scatter
    draw_triangle(ax_simplex)
    for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
        ax_simplex.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], label=label, **scatter_kw_2d)
    ax_simplex.set_title("2D simplex")
    ax_simplex.legend(loc="upper right", frameon=False, markerscale=1.2)

    # (b) positive orthant of 2-sphere
    draw_octant_sphere(ax_sphere)
    for y, label in [(y0, "t=0"), (y1, "t=0.5"), (y2, "t=1")]:
        pts = y.numpy()
        ax_sphere.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2], c=COLORS[label], label=label, **scatter_kw_3d
        )
    ax_sphere.set_title("Positive orthant of $S^2$")
    ax_sphere.legend(loc="upper left", frameon=False, markerscale=1.2)

    ax_note.axis("off")
    ax_note.text(
        0.0,
        0.95,
        "Compared OT costs",
        fontsize=20,
        fontweight="bold",
        va="top",
    )
    ax_note.text(
        0.0,
        0.72,
        "MM+SLERP\nsphere arc$^2$ cost",
        fontsize=17,
        va="top",
    )
    ax_note.text(
        0.0,
        0.48,
        "MM+SLERP+SquaredSpectral\n"
        f"power spectral cost, weight_power={args.spectral_weight_power:g}",
        fontsize=17,
        va="top",
    )
    ax_note.text(
        0.0,
        0.22,
        "MM+SLERP+Biharmonic\ntrue biharmonic cost ($\\lambda^{-2}$)",
        fontsize=17,
        va="top",
    )

    # (c) sphere-cost OT + SLERP interpolants
    y0_np, y1_np, y2_np = y0.numpy(), y1.numpy(), y2.numpy()
    draw_triangle(ax_slerp)
    slerp_hops = [(y0_np, y1_np, T_slerp_01), (y1_np, y2_np, T_slerp_12)]
    _draw_arc_backdrop(ax_slerp, slerp_hops)
    for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
        ax_slerp.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
    ax_slerp.set_title("Sphere-cost OT + SLERP interpolants")

    # (d) squared-spectral-cost OT + SLERP interpolants
    draw_triangle(ax_spec)
    spec_hops = [(y0_np, y1_np, T_spec_01), (y1_np, y2_np, T_spec_12)]
    _draw_arc_backdrop(ax_spec, spec_hops)
    for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
        ax_spec.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
    ax_spec.set_title("Squared-spectral-cost OT + SLERP interpolants")

    # (e) biharmonic-cost OT + SLERP interpolants
    draw_triangle(ax_bi)
    bi_hops = [(y0_np, y1_np, T_bi_01), (y1_np, y2_np, T_bi_12)]
    _draw_arc_backdrop(ax_bi, bi_hops)
    for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
        ax_bi.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
    ax_bi.set_title("Biharmonic-cost OT + SLERP interpolants")

    if not args.skip_predictions:
        draw_triangle(ax_slerp_pred)
        draw_predicted_paths(ax_slerp_pred, slerp_traj, max_paths=args.n_viz_paths)
        for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
            ax_slerp_pred.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
        ax_slerp_pred.set_title("MM+SLERP predicted paths")

        draw_triangle(ax_spec_pred)
        draw_predicted_paths(ax_spec_pred, spec_traj, max_paths=args.n_viz_paths)
        for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
            ax_spec_pred.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
        ax_spec_pred.set_title("MM+SLERP+SquaredSpectral predicted paths")

        draw_triangle(ax_bi_pred)
        draw_predicted_paths(ax_bi_pred, bi_traj, max_paths=args.n_viz_paths)
        for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
            ax_bi_pred.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
        ax_bi_pred.set_title("MM+SLERP+Biharmonic predicted paths")

    if do_si:
        # Deterministic-ODE panel for MM+SI: same source cells and rollout as
        # the row-3 panels, only the training objective differs.
        draw_triangle(ax_si_ode)
        draw_predicted_paths(ax_si_ode, si_ode_traj, max_paths=args.n_viz_paths)
        for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
            ax_si_ode.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
        ax_si_ode.set_title(
            f"MM+SI predicted paths (deterministic ODE, $\\sigma_\\mathrm{{train}}={args.si_sigma}$)"
        )

        # Stochastic-replicate panels: each of K source cells spawns R sample
        # paths, so a single starting cell visibly fans out toward distinct tips.
        for ax, (inf_sigma, traj) in zip(ax_si_sde_axes, si_sde_trajs):
            draw_triangle(ax)
            n_paths = args.si_stochastic_sources * args.si_stochastic_replicates
            draw_predicted_paths(ax, traj, max_paths=n_paths)
            for tri, label in [(tri0, "t=0"), (tri1, "t=0.5"), (tri2, "t=1")]:
                ax.scatter(tri[:, 0], tri[:, 1], c=COLORS[label], zorder=2, **scatter_kw_2d)
            ax.set_title(
                f"MM+SI predicted paths (SDE, $\\sigma_\\mathrm{{inf}}={inf_sigma}$, "
                f"{args.si_stochastic_replicates} samples / source)"
            )

    fig.tight_layout(pad=2.5, w_pad=2.5, h_pad=2.5)
    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(f"  Wrote {out.resolve()}")

    if not args.skip_predictions:
        loss_out = Path(args.loss_out) if args.loss_out is not None else out.with_name(
            f"{out.stem}_loss_curves.html"
        )
        loss_map = {
            "MM+SLERP": losses_slerp,
            "MM+SLERP+SquaredSpectral": losses_spec,
            "MM+SLERP+Biharmonic": losses_biharm,
        }
        if losses_si is not None:
            loss_map["MM+SI"] = losses_si
        write_loss_curves_plotly(loss_map, loss_out)


if __name__ == "__main__":
    main()
