"""
Multi-marginal Fisher Flow Matching training on the positive orthant.

Trains a single FlowNet velocity field jointly across S developmental stages,
with optional score regularization, stochastic interpolant noise, and
biharmonic spectral velocity/waypoint blending.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from surf.runtime import get as get_runtime
from surf.geometry.premetric import SpectralPremetric
from surf.geometry.sphere import normalize_sphere, compute_sphere_cost_matrix, sphere_brownian_perturb
from surf.models.flow_net import FlowNet
from surf.models.score_net import TimedRiemannianScoreNet
from surf.ot.coupling import ot_coupling


def _safe_label(label):
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


def _quantile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _write_premetric_diagnostics(
    *,
    label,
    premetric,
    stage_cells_sub,
    adj_couplings,
    interval_path_caches,
    n_samples,
    t_values,
    n_steps,
):
    """Sample conditional premetric paths and save Chen-Lipman diagnostics."""
    if premetric is None:
        return
    if t_values is None:
        t_values = [0.25, 0.5, 0.75, 0.9]

    rows = []
    rng = np.random.default_rng(123)
    n_intervals = max(len(stage_cells_sub) - 1, 1)
    n_each = max(1, int(n_samples) // max(n_intervals * len(t_values), 1))

    with torch.no_grad():
        for interval_idx, (ot_src, ot_tgt) in enumerate(adj_couplings):
            Y0 = stage_cells_sub[interval_idx]
            pair_count = len(ot_src)
            if pair_count == 0:
                continue
            for t_value in t_values:
                choice = rng.choice(pair_count, size=n_each, replace=pair_count < n_each)
                src_idx = torch.as_tensor(ot_src[choice], device=premetric.device, dtype=torch.long)
                tgt_idx = torch.as_tensor(ot_tgt[choice], device=premetric.device, dtype=torch.long)
                y0 = Y0[src_idx]
                target_emb = premetric.target_embeddings(interval_idx + 1, tgt_idx)
                t = torch.full((len(choice), 1), float(t_value), device=premetric.device)
                if premetric.trajectory_mode == "ode":
                    tau = premetric.clamp_time(t)
                else:
                    tau = premetric.clamp_path_time(t)

                rho0 = premetric.distance(y0, target_emb)
                x_t, u_t = premetric.sample_trajectory(
                    interval_idx,
                    src_idx,
                    interval_idx + 1,
                    tgt_idx,
                    tau,
                    n_steps=n_steps,
                    path_cache=(
                        None if interval_path_caches is None else interval_path_caches[interval_idx]
                    ),
                    pair_indices=choice,
                )
                rho_t, grad_tan = premetric.distance_and_gradient(x_t, target_emb)

                rho_linear = (1.0 - tau) * rho0
                abs_error = (rho_t - rho_linear).abs()
                rel_error = abs_error / rho0.clamp(min=premetric.eps)
                grad_norm = grad_tan.norm(dim=-1, keepdim=True)
                velocity_norm = u_t.norm(dim=-1, keepdim=True)
                drive_scale = rho_t / grad_norm.clamp(min=premetric.grad_norm_floor).square()
                if premetric.max_drive_scale is not None:
                    drive_scale = drive_scale.clamp(max=premetric.max_drive_scale)

                cos_nn = (x_t @ premetric.all_cells.T).max(dim=1).values.clamp(-1 + 1e-6, 1 - 1e-6)
                nn_arc = torch.acos(cos_nn).unsqueeze(-1)
                sphere_norm_error = (x_t.norm(dim=-1, keepdim=True) - 1.0).abs()
                min_coord = x_t.min(dim=1, keepdim=True).values
                negative_fraction = (x_t < 0).float().mean(dim=1, keepdim=True)

                tensors = {
                    "rho_initial": rho0,
                    "rho_t": rho_t,
                    "rho_target_linear": rho_linear,
                    "rho_decay_abs_error": abs_error,
                    "rho_decay_rel_error": rel_error,
                    "velocity_norm": velocity_norm,
                    "grad_norm": grad_norm,
                    "drive_scale": drive_scale,
                    "nearest_data_sphere_arc": nn_arc,
                    "min_coordinate": min_coord,
                    "negative_fraction": negative_fraction,
                    "sphere_norm_error": sphere_norm_error,
                }
                values = {
                    name: tensor.detach().cpu().numpy().reshape(-1)
                    for name, tensor in tensors.items()
                }
                for row_idx in range(len(choice)):
                    row = {
                        "method": label,
                        "interval": interval_idx,
                        "t": float(t_value),
                    }
                    for name, arr in values.items():
                        row[name] = float(arr[row_idx])
                    rows.append(row)

    safe = _safe_label(label)
    csv_path = Path(f"premetric_diagnostics_{safe}.csv")
    json_path = Path(f"premetric_diagnostics_{safe}.json")
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "method": label,
        "n_rows": len(rows),
        "decay_rel_error_mean": float(np.mean([r["rho_decay_rel_error"] for r in rows])) if rows else float("nan"),
        "decay_rel_error_q95": _quantile([r["rho_decay_rel_error"] for r in rows], 0.95),
        "velocity_norm_q95": _quantile([r["velocity_norm"] for r in rows], 0.95),
        "grad_norm_q05": _quantile([r["grad_norm"] for r in rows], 0.05),
        "drive_scale_q95": _quantile([r["drive_scale"] for r in rows], 0.95),
        "nearest_data_sphere_arc_q50": _quantile([r["nearest_data_sphere_arc"] for r in rows], 0.50),
        "min_coordinate_q01": _quantile([r["min_coordinate"] for r in rows], 0.01),
        "negative_fraction_mean": float(np.mean([r["negative_fraction"] for r in rows])) if rows else float("nan"),
        "sphere_norm_error_q95": _quantile([r["sphere_norm_error"] for r in rows], 0.95),
        "csv": str(csv_path),
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Premetric diagnostics:")
    print(
        "    "
        f"decay_rel_mean={summary['decay_rel_error_mean']:.4f}, "
        f"decay_rel_q95={summary['decay_rel_error_q95']:.4f}, "
        f"vel_q95={summary['velocity_norm_q95']:.4f}, "
        f"grad_q05={summary['grad_norm_q05']:.4f}, "
        f"nn_arc_q50={summary['nearest_data_sphere_arc_q50']:.4f}, "
        f"min_coord_q01={summary['min_coordinate_q01']:.4e}"
    )
    print(f"    wrote {csv_path} and {json_path}")


def _build_graph_smooth_edges(stage_cells_sub, knn, sigma_scale, device):
    """Build a symmetric heat-weighted kNN graph over the training sphere cells."""
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    all_cells = torch.cat(stage_cells_sub, dim=0)
    n_cells = len(all_cells)
    if n_cells < 2:
        return all_cells, None, None, None

    k_neighbors = min(int(knn) + 1, n_cells)
    cells_np = all_cells.detach().cpu().numpy()
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine")
    nn.fit(cells_np)
    dists, inds = nn.kneighbors(cells_np)
    if k_neighbors <= 1:
        return all_cells, None, None, None

    neigh_inds = inds[:, 1:]
    neigh_dists = dists[:, 1:]
    cos_sim = np.clip(1.0 - neigh_dists, -1.0 + 1e-6, 1.0 - 1e-6)
    arc = np.arccos(cos_sim)
    positive_arc = arc[arc > 1e-8]
    sigma = float(np.median(positive_arc)) if positive_arc.size else 1.0
    sigma = max(sigma * float(sigma_scale), 1e-6)
    weights = np.exp(-(arc ** 2) / (2.0 * sigma ** 2)).astype(np.float32, copy=False)

    rows = np.repeat(np.arange(n_cells), k_neighbors - 1)
    cols = neigh_inds.reshape(-1)
    data = weights.reshape(-1)
    graph = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    graph = graph.maximum(graph.T).tocoo()
    mask = (graph.row != graph.col) & (graph.data > 0)
    src = torch.as_tensor(graph.row[mask], device=device, dtype=torch.long)
    dst = torch.as_tensor(graph.col[mask], device=device, dtype=torch.long)
    edge_w = torch.as_tensor(graph.data[mask], device=device, dtype=torch.float32)
    return all_cells, src, dst, edge_w


def train_multi_marginal_flow(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    label="MultiMarginal",
    ot_subsample=2000,
    score_net=None,
    score_nets_per_interval=None,
    alpha=0.0,
    score_net_sigma=0.1,
    cost_fn=None,
    graph_smooth_lambda=0.0,
    graph_smooth_knn=15,
    graph_smooth_batch_edges=512,
    graph_smooth_sigma_scale=1.0,
    si_sigma=0.0,
    si_brownian_steps=3,
    biharm_beta=0.0,
    biharm_waypoints=False,
    global_biharm_embeddings=None,
    premetric_type=None,
    premetric_extension_k=64,
    premetric_softmax_beta=10.0,
    premetric_ode_steps=16,
    premetric_trajectory_mode="spectral_decode",
    premetric_decode_k=64,
    premetric_decode_beta=10.0,
    premetric_decode_chunk_size=64,
    premetric_velocity_fd_eps=0.02,
    premetric_knn=15,
    premetric_n_eig=50,
    premetric_spectral_family="power",
    premetric_weight_power=0.5,
    premetric_diffusion_time=1.0,
    premetric_time_cap=0.9,
    premetric_grad_norm_floor=0.05,
    premetric_max_drive_scale=50.0,
    premetric_ot_cost=False,
    premetric_diagnostics=False,
    premetric_diagnostic_samples=512,
    premetric_diagnostic_t_values=None,
    use_ema=True,
):
    """
    Multi-marginal Fisher Flow Matching. Trains a single FlowNet that passes
    through S developmental stages as marginals at times stage_times[0..S-1].

    At each training step:
      1. Pick a random adjacent pair of stages (i, i+1).
      2. Sample t ~ Uniform(stage_times[i], stage_times[i+1]).
      3. Rescale to local [0, 1] within the interval and SLERP between
         OT-coupled pairs from stages i and i+1.
      4. Compute v_target in the global time parameterization (SLERP velocity
         scaled by 1/(t_{i+1} - t_i)), optionally augmented by alpha*score.

    This generalizes the pairwise Fisher Flow to a single velocity field
    continuous across all stages. Unlike LOO, every stage is supervised
    directly, so we test the method's ability to match held-out *cells*
    within each stage rather than reconstruct an unseen intermediate stage.

    Parameters
    ----------
    stage_cells: list of (N_i, D) torch tensors on DEVICE, one per stage,
        already mapped to the positive orthant.
    stage_times: list of floats in [0,1], one per stage, strictly increasing.
    score_net: optional single RiemannianScoreNet (global, all stages).
    score_nets_per_interval: optional list of S-1 RiemannianScoreNets, one
        per interval. When provided, interval i uses score_nets_per_interval[i]
        instead of score_net. This enables forward-directed score regularization
        where each interval's score only sees future stages.
    alpha: strength of the score regularizer.

    Returns
    -------
    (model, losses)
    """
    rt = get_runtime()

    assert len(stage_cells) == len(stage_times)
    S = len(stage_cells)
    assert S >= 2
    for i in range(S - 1):
        assert stage_times[i] < stage_times[i + 1]

    model = FlowNet(D).to(rt.device)
    if rt.use_amp:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"  torch.compile failed ({e}); running uncompiled")
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)

    # EMA of model weights for stability (0.999 decay)
    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    warmup_iters = min(500, n_iters // 10)

    # Precompute OT couplings for each adjacent pair, once (stable over training).
    # We subsample each stage to ot_subsample cells for OT tractability.
    print(f"  Computing OT couplings for {S - 1} adjacent pairs...")
    stage_cells_sub = []
    rng = np.random.default_rng(0)
    for i, cells in enumerate(stage_cells):
        if len(cells) > ot_subsample:
            idx = rng.choice(len(cells), size=ot_subsample, replace=False)
            stage_cells_sub.append(cells[idx])
        else:
            stage_cells_sub.append(cells)
        print(f"    stage {i}: {len(stage_cells_sub[i])} cells (t={stage_times[i]:.2f})")

    use_premetric = premetric_type is not None
    premetric = None
    if use_premetric or premetric_ot_cost:
        if premetric_type not in ("biharmonic", "spectral"):
            raise ValueError(f"Unsupported premetric_type={premetric_type!r}")
        premetric = SpectralPremetric(
            stage_cells_sub,
            knn_graph=premetric_knn,
            n_eig=premetric_n_eig,
            spectral_family=premetric_spectral_family,
            weight_power=premetric_weight_power,
            diffusion_time=premetric_diffusion_time,
            extension_k=premetric_extension_k,
            softmax_beta=premetric_softmax_beta,
            trajectory_mode=premetric_trajectory_mode,
            decode_k=premetric_decode_k,
            decode_beta=premetric_decode_beta,
            decode_chunk_size=premetric_decode_chunk_size,
            velocity_fd_eps=premetric_velocity_fd_eps,
            time_cap=premetric_time_cap,
            grad_norm_floor=premetric_grad_norm_floor,
            max_drive_scale=premetric_max_drive_scale,
        )
        print(
            "  Premetric target: "
            f"type={premetric_type}, family={premetric_spectral_family}, "
            f"weight_power={premetric_weight_power:.3f}, "
            f"diffusion_time={premetric_diffusion_time:.3f}, "
            f"trajectory={premetric_trajectory_mode}, "
            f"extension_k={premetric_extension_k}, beta={premetric_softmax_beta:.1f}, "
            f"decode_k={premetric_decode_k}, decode_beta={premetric_decode_beta:.1f}, "
            f"decode_chunk={premetric_decode_chunk_size}, "
            f"decode_device={'cpu' if premetric.decode_on_cpu else str(rt.device)}, "
            f"fd_eps={premetric_velocity_fd_eps:.3f}, "
            f"ode_steps={premetric_ode_steps}, time_cap={premetric_time_cap:.2f}, "
            f"grad_floor={premetric_grad_norm_floor:.3f}, "
            f"max_drive={premetric_max_drive_scale:.1f}"
        )

    cost_label = "sphere W2"
    if premetric_ot_cost:
        cost_label = f"premetric_{premetric_spectral_family}_global"
    if cost_fn is not None:
        cost_label = cost_fn.__name__ if hasattr(cost_fn, "__name__") else "custom"
    print(f"  OT cost: {cost_label}")
    adj_couplings = []  # list of (ot_src, ot_tgt) numpy arrays
    for i in range(S - 1):
        Y0, Y1 = stage_cells_sub[i], stage_cells_sub[i + 1]
        if premetric_ot_cost:
            cost = premetric.stage_pair_cost(i, i + 1)
        elif cost_fn is not None:
            cost = cost_fn(Y0, Y1)
        else:
            cost = compute_sphere_cost_matrix(Y0, Y1)
        n_pool = min(20000, len(Y0) * len(Y1))
        os_, ot_ = ot_coupling(cost, n_pool)
        adj_couplings.append((os_, ot_))

    interval_path_caches = None
    if use_premetric and premetric.trajectory_mode == "graph_geodesic":
        print("  Precomputing graph-geodesic paths for OT pairs...")
        interval_path_caches = []
        for i, (ot_src, ot_tgt) in enumerate(adj_couplings):
            path_cache = premetric.prepare_interval_paths(i, i + 1, ot_src, ot_tgt)
            interval_path_caches.append(path_cache)
            path_nodes = np.asarray([len(meta["nodes"]) for meta in path_cache], dtype=np.float64)
            print(
                f"    interval {i}->{i+1}: "
                f"mean_nodes={path_nodes.mean():.1f}, p95_nodes={np.quantile(path_nodes, 0.95):.1f}"
            )

    adj_couplings_device = [
        (
            torch.as_tensor(ot_src, device=rt.device, dtype=torch.long),
            torch.as_tensor(ot_tgt, device=rt.device, dtype=torch.long),
        )
        for ot_src, ot_tgt in adj_couplings
    ]

    use_global_score = alpha > 0.0 and score_net is not None and score_nets_per_interval is None
    use_interval_score = alpha > 0.0 and score_nets_per_interval is not None
    use_score = use_global_score or use_interval_score

    log_sigma_tensor = torch.full((batch_size,), float(np.log(score_net_sigma)), device=rt.device) if use_score else None

    if use_global_score:
        score_net.eval()
        print(f"  Score regularization: alpha={alpha}, sigma={score_net_sigma:.4f} (global)")
    elif use_interval_score:
        for sn in score_nets_per_interval:
            if isinstance(sn, nn.Module):
                sn.eval()
        print(f"  Score regularization: alpha={alpha}, sigma={score_net_sigma:.4f} (per-interval forward-directed)")

    # Interpret si_sigma as the target Brownian arc length (radians on the
    # sphere). sphere_brownian_perturb's `sigma` is a per-component scale
    # applied to an ambient Gaussian whose tangent projection has norm
    # ~sqrt(D-1), so the effective arc length is per_component_sigma *
    # sqrt(D-1). Invert this to get the per-component scale that produces
    # the requested arc. This makes si_sigma dimensionally interpretable
    # as "typical arc distance the noise should cover", independent of D.
    si_per_component_sigma = 0.0
    if si_sigma > 0.0:
        si_per_component_sigma = float(si_sigma) / float(np.sqrt(max(D - 1, 1)))
        print(f"  Stochastic interpolant: si_sigma={si_sigma:.4f} rad (target arc), "
              f"per_component={si_per_component_sigma:.6f}, n_brownian_steps={si_brownian_steps}")

    graph_smooth_cells = graph_smooth_src = graph_smooth_dst = graph_smooth_w = None
    use_graph_smooth = graph_smooth_lambda > 0.0
    if use_graph_smooth:
        graph_smooth_cells, graph_smooth_src, graph_smooth_dst, graph_smooth_w = (
            _build_graph_smooth_edges(
                stage_cells_sub,
                knn=graph_smooth_knn,
                sigma_scale=graph_smooth_sigma_scale,
                device=rt.device,
            )
        )
        if graph_smooth_src is None or len(graph_smooth_src) == 0:
            use_graph_smooth = False
            print("  Graph velocity smoothness: disabled (no graph edges)")
        else:
            n_smooth_edges = (
                batch_size
                if graph_smooth_batch_edges is None or int(graph_smooth_batch_edges) <= 0
                else int(graph_smooth_batch_edges)
            )
            graph_smooth_batch_edges = n_smooth_edges
            print(
                "  Graph velocity smoothness: "
                f"lambda={graph_smooth_lambda:.4g}, knn={graph_smooth_knn}, "
                f"edges={len(graph_smooth_src)}, batch_edges={graph_smooth_batch_edges}, "
                f"sigma_scale={graph_smooth_sigma_scale:.3f}"
            )

    # Biharmonic velocity blending (Approach A): blend SLERP velocity with
    # the biharmonic spectral-gradient direction from Chen & Lipman (2024).
    # The spectral embedding maps each training cell to R^k; for a SLERP
    # midpoint x_t we extend the embedding via kNN interpolation, then
    # compute the spectral direction toward the target x_1.
    use_biharm_vel = biharm_beta > 0.0 and global_biharm_embeddings is not None
    # Biharmonic waypoints (Approach B): piecewise SLERP through a
    # biharmonic-midpoint cell, doubling the number of SLERP segments.
    use_biharm_wp = biharm_waypoints and global_biharm_embeddings is not None

    biharm_emb_tensors = None
    biharm_knn_idx = None
    biharm_knn_weights = None
    waypoint_idx_per_interval = None

    if use_biharm_vel or use_biharm_wp:
        biharm_emb_tensors = [torch.from_numpy(e.astype(np.float32)).to(rt.device)
                              for e in global_biharm_embeddings]
        # Build kNN index for out-of-sample spectral embedding extension
        # (needed for Approach A to compute Phi(x_t) at SLERP midpoints)
        if use_biharm_vel:
            all_sub_cells = torch.cat(stage_cells_sub, dim=0)
            all_sub_emb = torch.cat(biharm_emb_tensors, dim=0)
            print(f"  Biharmonic velocity blend: beta={biharm_beta:.3f}")

        if use_biharm_wp:
            # For each adjacent pair, precompute the biharmonic midpoint cell
            # for every OT-coupled pair. The midpoint is the training cell
            # whose spectral embedding is closest to the average of the
            # source and target embeddings.
            all_train = torch.cat(stage_cells_sub, dim=0)
            all_emb = torch.cat(biharm_emb_tensors, dim=0)
            waypoint_idx_per_interval = []
            for iv in range(S - 1):
                os_, ot_ = adj_couplings[iv]
                e0 = biharm_emb_tensors[iv][os_]  # (n_pairs, k)
                e1 = biharm_emb_tensors[iv + 1][ot_]
                mid_emb = (e0 + e1) / 2  # (n_pairs, k)
                # Find closest training cell in spectral space
                dists = torch.cdist(mid_emb, all_emb)  # (n_pairs, N_all)
                wp_idx = dists.argmin(dim=1)  # (n_pairs,)
                waypoint_idx_per_interval.append(wp_idx)
            print(f"  Biharmonic waypoints: piecewise SLERP through spectral midpoints")

    losses = []
    for it in range(n_iters):
        # Pick a random adjacent pair for each batch element. Simplest: uniform
        # over intervals, then fill the batch from that single interval. This
        # keeps the SLERP math batched.
        i = np.random.randint(0, S - 1)
        Y0 = stage_cells_sub[i]
        Y1 = stage_cells_sub[i + 1]
        ot_src_i, ot_tgt_i = adj_couplings_device[i]
        t_start = stage_times[i]
        t_end = stage_times[i + 1]
        dt_interval = t_end - t_start

        pair_idx = torch.randint(len(ot_src_i), (batch_size,), device=rt.device)
        s_idx_t = ot_src_i[pair_idx]
        g_idx_t = ot_tgt_i[pair_idx]

        # Local SLERP parameter in [0, 1] and global t
        s_local = torch.rand(batch_size, 1, device=rt.device)
        t_global = t_start + dt_interval * s_local  # (B, 1) in [t_start, t_end]

        y0 = Y0[s_idx_t]
        y1 = Y1[g_idx_t]

        # True Chen-Lipman-style target on a spectral premetric.
        if use_premetric:
            if premetric.trajectory_mode == "ode":
                s_sub = premetric.clamp_time(s_local)
            else:
                s_sub = premetric.clamp_path_time(s_local)
            t_global = t_start + dt_interval * s_sub
            with torch.no_grad():
                z_t, v_local = premetric.sample_trajectory(
                    i,
                    s_idx_t,
                    i + 1,
                    g_idx_t,
                    s_sub,
                    n_steps=premetric_ode_steps,
                    path_cache=None if interval_path_caches is None else interval_path_caches[i],
                    pair_indices=pair_idx,
                )
                v_target = v_local / dt_interval
        # Approach B: piecewise SLERP through biharmonic midpoint waypoint
        elif use_biharm_wp:
            all_train_cells = torch.cat(stage_cells_sub, dim=0)
            wp_global_idx = waypoint_idx_per_interval[i][pair_idx]
            y_mid = all_train_cells[wp_global_idx]
            # Two segments: [y0->y_mid] for s_local<0.5, [y_mid->y1] for s_local>=0.5
            first_half = (s_local < 0.5).squeeze(-1)
            # Remap s_local to sub-segment local parameter
            s_sub = torch.where(first_half.unsqueeze(-1), 2 * s_local, 2 * s_local - 1)
            seg_start = torch.where(first_half.unsqueeze(-1), y0, y_mid)
            seg_end = torch.where(first_half.unsqueeze(-1), y_mid, y1)
            # Effective interval dt: each sub-segment covers half the interval
            dt_sub = dt_interval / 2.0
        else:
            s_sub = s_local
            seg_start = y0
            seg_end = y1
            dt_sub = dt_interval

        if not use_premetric:
            # SLERP position + local velocity (magnitude = omega in s_local time)
            cos_omega = (seg_start * seg_end).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
            omega = torch.acos(cos_omega)
            sin_omega = torch.sin(omega).clamp(min=1e-8)
            w0 = torch.sin((1 - s_sub) * omega) / sin_omega
            w1 = torch.sin(s_sub * omega) / sin_omega
            z_t = w0 * seg_start + w1 * seg_end
            z_t = z_t / z_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Stochastic interpolant: perturb the SLERP position with Riemannian
        # Brownian noise so the velocity field is supervised on a neighborhood
        # of the clean SLERP arc, not just the arc itself. First-order
        # parallel transport of v_target is handled by the existing
        # `v_local - (v_local * z_t) * z_t` tangent projection below, which
        # now runs at the noisy z_t.
        if si_sigma > 0.0:
            with torch.no_grad():
                z_t_clean = z_t
                z_t = sphere_brownian_perturb(z_t, si_per_component_sigma, n_steps=si_brownian_steps)
                if it == 0:
                    cos_disp = (z_t * z_t_clean).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
                    arc_displacement = torch.acos(cos_disp)
                    ref_cells = Y0[:256]
                    arc_to_nearest_clean = torch.acos(
                        (z_t_clean @ ref_cells.T).clamp(-1 + 1e-6, 1 - 1e-6)
                    ).min(dim=-1).values
                    arc_to_nearest_noisy = torch.acos(
                        (z_t @ ref_cells.T).clamp(-1 + 1e-6, 1 - 1e-6)
                    ).min(dim=-1).values
                    print(f"    [diag@iter0, SI] arc_displacement(clean->noisy)  "
                          f"mean={arc_displacement.mean():.4f}  "
                          f"p95={arc_displacement.quantile(0.95):.4f}")
                    print(f"    [diag@iter0, SI] arc-to-nearest(clean)  "
                          f"mean={arc_to_nearest_clean.mean():.4f}  "
                          f"p95={arc_to_nearest_clean.quantile(0.95):.4f}")
                    print(f"    [diag@iter0, SI] arc-to-nearest(noisy)  "
                          f"mean={arc_to_nearest_noisy.mean():.4f}  "
                          f"p95={arc_to_nearest_noisy.quantile(0.95):.4f}")

        if not use_premetric:
            v_local = omega * (
                -torch.cos((1 - s_sub) * omega) / sin_omega * seg_start
                + torch.cos(s_sub * omega) / sin_omega * seg_end
            )
            v_local = v_local - (v_local * z_t).sum(dim=-1, keepdim=True) * z_t
            v_target = v_local / dt_sub

        # Approach A: blend SLERP velocity with biharmonic spectral direction.
        # The biharmonic direction at z_t toward y1 is computed as the
        # sphere log-map from z_t to y1, scaled to match the SLERP velocity
        # magnitude. This teaches the velocity network to follow the data
        # manifold geometry (via the spectral embedding) rather than the
        # great-circle arc.
        if use_biharm_vel:
            with torch.no_grad():
                # log-map from z_t to y1: the geodesic tangent pointing at y1
                cos_zy = (z_t * y1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
                theta_zy = torch.acos(cos_zy)
                sin_zy = torch.sin(theta_zy).clamp(min=1e-8)
                log_zt_y1 = (theta_zy / sin_zy) * (y1 - cos_zy * z_t)
                # Scale to match SLERP velocity magnitude for stable blending
                v_slerp_norm = v_target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                log_norm = log_zt_y1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                v_biharm_dir = log_zt_y1 * (v_slerp_norm / log_norm)
                v_target = (1 - biharm_beta) * v_target + biharm_beta * v_biharm_dir

        if use_score:
            with torch.no_grad():
                cur_score_net = score_nets_per_interval[i] if use_interval_score else score_net
                if isinstance(cur_score_net, TimedRiemannianScoreNet):
                    score = cur_score_net(z_t, log_sigma_tensor, t_global.squeeze(1))
                else:
                    score = cur_score_net(z_t, log_sigma_tensor)

                if it == 0:
                    v_slerp_norm = v_target.norm(dim=-1)
                    score_norm = (alpha * score).norm(dim=-1)
                    v_slerp_hat = v_target / v_slerp_norm.unsqueeze(-1).clamp(min=1e-8)
                    score_hat = score / score.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    cos_align = (v_slerp_hat * score_hat).sum(dim=-1)
                    any_cells = Y0[:256]
                    cos_to_cells = (z_t @ any_cells.T).clamp(-1 + 1e-6, 1 - 1e-6)
                    arc_to_nearest = torch.acos(cos_to_cells).min(dim=-1).values
                    mode = "forward-interval" if use_interval_score else "global"
                    print(f"    [diag@iter0, {mode}] ||v_slerp||  mean={v_slerp_norm.mean():.3f}  "
                          f"p95={v_slerp_norm.quantile(0.95):.3f}")
                    print(f"    [diag@iter0, {mode}] ||alpha*score|| mean={score_norm.mean():.3f}  "
                          f"p95={score_norm.quantile(0.95):.3f}  "
                          f"ratio={score_norm.mean() / v_slerp_norm.mean():.3f}")
                    print(f"    [diag@iter0, {mode}] cos(v_slerp, score)  "
                          f"mean={cos_align.mean():.3f}  std={cos_align.std():.3f}")
                    print(f"    [diag@iter0, {mode}] z_t arc-to-nearest  "
                          f"mean={arc_to_nearest.mean():.3f}  p95={arc_to_nearest.quantile(0.95):.3f}")

            v_target = v_target + alpha * score

        smooth_loss = None
        with torch.autocast(device_type=rt.device.type, dtype=rt.amp_dtype, enabled=rt.use_amp):
            v_pred_raw = model(z_t.detach(), t_global.squeeze(1))
            v_pred = v_pred_raw - (v_pred_raw * z_t).sum(dim=-1, keepdim=True) * z_t
            fm_loss = ((v_pred - v_target.detach()) ** 2).mean()
            loss = fm_loss
            if use_graph_smooth:
                edge_idx = torch.randint(
                    len(graph_smooth_src),
                    (graph_smooth_batch_edges,),
                    device=rt.device,
                )
                smooth_src = graph_smooth_src[edge_idx]
                smooth_dst = graph_smooth_dst[edge_idx]
                smooth_w = graph_smooth_w[edge_idx]
                ya = graph_smooth_cells[smooth_src]
                yb = graph_smooth_cells[smooth_dst]
                t_smooth = (
                    stage_times[0]
                    + (stage_times[-1] - stage_times[0])
                    * torch.rand(graph_smooth_batch_edges, device=rt.device)
                )
                va_raw = model(ya, t_smooth)
                vb_raw = model(yb, t_smooth)
                va = va_raw - (va_raw * ya).sum(dim=-1, keepdim=True) * ya
                vb = vb_raw - (vb_raw * yb).sum(dim=-1, keepdim=True) * yb
                smooth_sq = (va - vb).square().sum(dim=-1)
                smooth_loss = (smooth_w * smooth_sq).sum() / smooth_w.sum().clamp(min=1e-8)
                loss = loss + graph_smooth_lambda * smooth_loss
        opt.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # LR warmup
        if it < warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr * (it + 1) / warmup_iters
        elif it == warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr
        opt.step()
        # EMA update
        with torch.no_grad():
            if it == warmup_iters:
                # Polyak-style: snap EMA to live weights at end of warmup, so
                # the iter-0 random init never contaminates the EMA average.
                # (At decay=0.999 over 3000 iters, this would otherwise be
                # ~5% of the final state.)
                for k, v in model.state_dict().items():
                    ema_state[k].copy_(v)
            else:
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

        if it % 200 == 0:
            losses.append(loss.item())
            if smooth_loss is None:
                print(f"  {label:22s} iter {it:4d}  loss={loss.item():.4f}")
            else:
                print(
                    f"  {label:22s} iter {it:4d}  loss={loss.item():.4f}  "
                    f"fm={fm_loss.item():.4f}  smooth={smooth_loss.item():.4f}"
                )

    # Load EMA weights for evaluation (or skip if use_ema=False to keep
    # live weights). At decay 0.999 over 3000 iters, ~5% of the EMA still
    # comes from random init; live weights may give better absolute MMD.
    if use_ema:
        model.load_state_dict(ema_state)
    if use_premetric and premetric_diagnostics:
        _write_premetric_diagnostics(
            label=label,
            premetric=premetric,
            stage_cells_sub=stage_cells_sub,
            adj_couplings=adj_couplings,
            interval_path_caches=interval_path_caches,
            n_samples=premetric_diagnostic_samples,
            t_values=premetric_diagnostic_t_values,
            n_steps=premetric_ode_steps,
        )
    return model, losses
