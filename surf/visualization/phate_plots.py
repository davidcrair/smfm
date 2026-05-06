"""
PHATE-based visualization routines for multi-marginal flow models.

Contains:
- visualize_trajectories_phate: trajectory lines on PHATE embedding
- visualize_endpoints_phate: predicted vs ground-truth endpoint scatter
- visualize_nn_distance_histograms: nearest-neighbor distance distributions
- visualize_pca_vector_field: projected instantaneous vector-field snapshots
"""

import numpy as np
import torch

from surf.evaluation.generation import (
    generate_euclidean_flow_trajectory,
    generate_fisher_flow,
    generate_fisher_flow_trajectory,
    generate_sphere_mean_flow_trajectory,
)
from surf.geometry.sphere import from_orthant, normalize_sphere


def visualize_trajectories_phate(models_dict, test_stage_log1p, test_stage_sphere,
                                 stage_times, stages, n_traj=75, n_steps=100,
                                 out_path="trajectories_phate.html",
                                 model_representations=None,
                                 model_spaces=None):
    """Generate PHATE-embedded trajectory visualization as interactive Plotly HTML.

    Fits PHATE on log1p test cells (matching the endpoint visualization and
    the standard PHATE tutorial), integrates each model from t=0 test cells
    to t=1, maps intermediate positions into log1p space, and plots
    trajectories over the empirical cell landscape.

    Sphere-based models integrate on the sphere; their compositional outputs
    are mapped to log1p via the median-library-size inverse transform.
    Euclidean models (e.g. MM+Linear) integrate in the configured Euclidean
    training space and are decoded back to log1p before PHATE projection.
    """
    import phate
    import plotly.graph_objects as go

    S = len(stages)
    all_test_log1p = torch.cat(test_stage_log1p, dim=0)

    print(f"\n  Fitting PHATE on {len(all_test_log1p)} test cells (log1p) for trajectory visualization...")
    ph = phate.PHATE(n_components=2, knn=15, t="auto", verbose=0, random_state=42)
    emb_all = ph.fit_transform(all_test_log1p.numpy())

    all_counts = torch.expm1(all_test_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig = go.Figure()

    # Ground-truth cells — each timepoint is its own toggleable legend entry
    offset = 0
    for i, s in enumerate(stages):
        n_s = len(test_stage_log1p[i])
        idx = slice(offset, offset + n_s)
        fig.add_trace(go.Scatter(
            x=emb_all[idx, 0], y=emb_all[idx, 1],
            mode="markers",
            marker=dict(size=7, color=stage_colors[i % len(stage_colors)], opacity=0.55),
            name=f"GT {s} (t={stage_times[i]:.2f})",
            legendgroup=f"gt_{i}",
            legendgrouptitle_text="Ground Truth" if i == 0 else None,
            showlegend=True,
        ))
        offset += n_s

    rng = np.random.default_rng(0)
    n_source = len(test_stage_sphere[0])
    traj_idx = rng.choice(n_source, size=min(n_traj, n_source), replace=False)
    sphere_source = test_stage_sphere[0][traj_idx]
    log1p_source = test_stage_log1p[0][traj_idx]

    method_colors = {"MM+SLERP": "#000000", "MM+SLERP+SquaredSpectral": "#e6194b",
                     "MM+SLERP+Biharmonic": "#e6194b",
                     "MM+SLERP+GlobalBiharmonic": "#ff6600",
                     "MM+PremetricBiharmonic": "#00a08a",
                     "MM+SI": "#3cb44b", "MM+SI+Biharmonic": "#4363d8",
                     "MM+Score_learned+Biharmonic": "#f58231",
                     "MM+Linear": "#000000", "MM+Linear+Biharmonic": "#808000"}

    for mname, model in models_dict.items():
        representation = (model_representations or {}).get(mname, "sphere")
        state_space = (model_spaces or {}).get(mname)
        print(f"    Generating {n_traj} trajectories for {mname} ({representation})...")
        if representation == "euclidean":
            source = state_space.encode(log1p_source) if state_space is not None else log1p_source
            traj = generate_euclidean_flow_trajectory(
                model, source, n_steps=n_steps,
                t_start=0.0, t_end=1.0, n_checkpoints=20,
            )
            traj_log1p_all = []
            for _, pos in traj:
                if state_space is None:
                    traj_log1p_all.append(pos.numpy())
                else:
                    decoded = state_space.decode(pos)
                    traj_log1p_all.append(
                        decoded.numpy() if hasattr(decoded, "numpy") else np.asarray(decoded)
                    )
        elif representation == "sphere_mean":
            traj = generate_sphere_mean_flow_trajectory(
                model, sphere_source, n_steps=n_steps,
                t_start=0.0, t_end=1.0, n_checkpoints=20,
            )
            traj_log1p_all = [
                torch.log1p(pos_comp * median_libsize).numpy() for _, pos_comp in traj
            ]
        else:
            traj = generate_fisher_flow_trajectory(
                model, sphere_source, n_steps=n_steps,
                t_start=0.0, t_end=1.0, n_checkpoints=20,
            )
            # Sphere trajectory returns compositional positions -> map to log1p.
            traj_log1p_all = [
                torch.log1p(pos_comp * median_libsize).numpy() for _, pos_comp in traj
            ]
        traj_log1p_cat = np.concatenate(traj_log1p_all, axis=0)
        traj_emb = ph.transform(traj_log1p_cat)

        n_cells = len(traj_idx)
        n_ckpt = len(traj)
        color = method_colors.get(mname, "#888888")

        # 10 evenly-spaced arrows per trajectory (skip t=0)
        n_arrows = min(10, max(1, n_ckpt - 1))
        arrow_indices = set(
            int(round(x)) for x in np.linspace(1, n_ckpt - 1, n_arrows)
        )
        sizes = [13 if k in arrow_indices else 0 for k in range(n_ckpt)]

        for ci in range(n_cells):
            xs = [traj_emb[k * n_cells + ci, 0] for k in range(n_ckpt)]
            ys = [traj_emb[k * n_cells + ci, 1] for k in range(n_ckpt)]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color=color, width=2.0),
                marker=dict(
                    size=sizes,
                    symbol="arrow",
                    angleref="previous",
                    color=color,
                    line=dict(width=0),
                ),
                opacity=0.4,
                showlegend=(ci == 0),
                name=mname,
                legendgroup=mname,
                legendgrouptitle_text=mname if ci == 0 else None,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(
            text="Predicted Trajectories on PHATE Embedding (log1p space)",
            font=dict(size=26, color="#111"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="PHATE 1", font=dict(size=22)),
            tickfont=dict(size=16),
            showgrid=True, gridcolor="#eee", zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="PHATE 2", font=dict(size=22)),
            tickfont=dict(size=16),
            showgrid=True, gridcolor="#eee", zeroline=False,
        ),
        width=1500, height=1050,
        margin=dict(l=80, r=40, t=90, b=70),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            font=dict(size=16),
            groupclick="toggleitem",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#bbb",
            borderwidth=1,
        ),
    )
    fig.write_html(out_path)
    print(f"    Saved interactive visualization to {out_path}")


def visualize_endpoints_phate(models_dict, test_stage_log1p, test_stage_sphere,
                              stage_times, stages, out_path="endpoints_phate.html"):
    """PHATE scatter of ground-truth vs predicted cell distributions at each stage.

    PHATE is fitted on the raw log1p-normalized test cells (matching the
    standard PHATE tutorial pipeline). Flow predictions are mapped from
    the sphere back to log1p space via:
        sphere -> from_orthant (y^2) -> compositional -> * median_libsize -> log1p
    then projected through phate.transform().

    Legend is grouped so clicking "Ground Truth" toggles all GT stages at once,
    and clicking a method name toggles all that method's predicted stages.
    """
    import phate
    import plotly.graph_objects as go

    S = len(stages)

    # Fit PHATE on raw log1p test cells (matching the standard tutorial)
    all_test_log1p = torch.cat(test_stage_log1p, dim=0)
    print(f"\n  Fitting PHATE on {len(all_test_log1p)} test cells (log1p space)...")
    ph = phate.PHATE(n_components=2, knn=15, t="auto", verbose=0, random_state=42)
    emb_all = ph.fit_transform(all_test_log1p.numpy())

    # Compute median library size from all test cells for inverse transform
    all_counts = torch.expm1(all_test_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())
    print(f"    Median library size (test cells): {median_libsize:.1f}")

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig = go.Figure()

    # Ground-truth cells per stage — each stage is its own legend entry
    offset = 0
    for i, s in enumerate(stages):
        n_s = len(test_stage_log1p[i])
        idx = slice(offset, offset + n_s)
        fig.add_trace(go.Scatter(
            x=emb_all[idx, 0], y=emb_all[idx, 1],
            mode="markers",
            marker=dict(size=5, color=stage_colors[i % len(stage_colors)], opacity=0.4),
            name=f"GT {s} (t={stage_times[i]:.2f})",
        ))
        offset += n_s

    method_markers = {"MM+SLERP": "diamond", "MM+SLERP+SquaredSpectral": "x",
                      "MM+SLERP+Biharmonic": "x",
                      "MM+SLERP+GlobalBiharmonic": "triangle-up",
                      "MM+PremetricBiharmonic": "hexagon",
                      "MM+SI": "cross", "MM+SI+Biharmonic": "star",
                      "MM+BiharmonicVel": "square", "MM+BiharmonicWaypoint": "pentagon",
                      "MM+Score_learned+Biharmonic": "triangle-down"}

    source = test_stage_sphere[0]
    for mname, model in models_dict.items():
        print(f"    Generating chained predictions for {mname}...")
        marker_sym = method_markers.get(mname, "circle")
        for i in range(1, S):
            t_target = stage_times[i]
            n_steps = max(10, int(50 * t_target))
            pred_sphere = generate_fisher_flow(
                model, source, n_steps=n_steps, t_start=0.0, t_end=t_target,
            )
            pred_comp = from_orthant(pred_sphere)
            pred_counts = pred_comp * median_libsize
            pred_log1p = torch.log1p(pred_counts).numpy()
            pred_emb = ph.transform(pred_log1p)
            fig.add_trace(go.Scatter(
                x=pred_emb[:, 0], y=pred_emb[:, 1],
                mode="markers",
                marker=dict(size=7, symbol=marker_sym, opacity=0.7,
                            color=stage_colors[i % len(stage_colors)],
                            line=dict(width=0.5, color="black")),
                name=f"{mname} t={t_target:.2f}",
            ))

    fig.update_layout(
        title="Ground Truth vs Predicted Endpoints (PHATE, log1p space)",
        xaxis_title="PHATE 1", yaxis_title="PHATE 2",
        width=1100, height=850,
        legend=dict(font=dict(size=10)),
    )
    fig.write_html(out_path)
    print(f"    Saved endpoint visualization to {out_path}")


def visualize_nn_distance_histograms(models_dict, test_stage_log1p, test_stage_sphere,
                                     stage_times, stages, out_path="nn_distances.html"):
    """Nearest-neighbor distance histograms: predicted -> ground-truth per stage.

    For each predicted cell at stage i, computes its Euclidean distance (in
    log1p space) to the closest ground-truth cell at that stage. Overlapping
    histograms for each method let you compare how well predictions match the
    true cell distribution.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    S = len(stages)

    # Compute median library size for inverse transform
    all_log1p = torch.cat(test_stage_log1p, dim=0)
    all_counts = torch.expm1(all_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())

    method_colors = {"MM+SLERP": "rgba(0,0,0,0.5)", "MM+SLERP+SquaredSpectral": "rgba(230,25,75,0.5)",
                     "MM+SLERP+Biharmonic": "rgba(230,25,75,0.5)",
                     "MM+PremetricBiharmonic": "rgba(0,160,138,0.5)",
                     "MM+SI": "rgba(60,180,75,0.5)", "MM+SI+Biharmonic": "rgba(67,99,216,0.5)",
                     "MM+Score_learned+Biharmonic": "rgba(245,130,49,0.5)"}

    fig = make_subplots(
        rows=1, cols=S - 1,
        subplot_titles=[f"t={stage_times[i]:.2f} ({stages[i]})" for i in range(1, S)],
        shared_yaxes=True,
    )

    source = test_stage_sphere[0]
    for mname, model in models_dict.items():
        print(f"    Computing NN distances for {mname}...")
        color = method_colors.get(mname, "rgba(128,128,128,0.5)")
        for col_idx, i in enumerate(range(1, S)):
            t_target = stage_times[i]
            n_steps = max(10, int(50 * t_target))
            pred_sphere = generate_fisher_flow(
                model, source, n_steps=n_steps, t_start=0.0, t_end=t_target,
            )
            pred_comp = from_orthant(pred_sphere)
            pred_log1p = torch.log1p(pred_comp * median_libsize)
            gt_log1p = test_stage_log1p[i]

            dists = torch.cdist(pred_log1p, gt_log1p)
            nn_dists = dists.min(dim=1).values.numpy()

            fig.add_trace(
                go.Histogram(
                    x=nn_dists, nbinsx=50,
                    name=mname,
                    marker_color=color,
                    legendgroup=mname,
                    showlegend=(col_idx == 0),
                ),
                row=1, col=col_idx + 1,
            )

    fig.update_layout(
        title="Nearest-Neighbor Distance to Ground Truth (log1p Euclidean)",
        barmode="overlay",
        width=300 * (S - 1) + 100, height=400,
        legend=dict(font=dict(size=11)),
    )
    for col_idx in range(S - 1):
        fig.update_xaxes(title_text="NN distance" if col_idx == 0 else "", row=1, col=col_idx + 1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.write_html(out_path)
    print(f"    Saved NN distance histograms to {out_path}")


def visualize_endpoints_pca(models_dict, test_stage_log1p, test_stage_sphere,
                            stage_times, stages, out_path="endpoints_pca.html"):
    """PCA scatter: ground truth vs predicted endpoints at t=1.00.

    Fits PCA on the final-stage (t=1.00) ground-truth test cells only,
    then projects the chained predictions from each method (integrated
    from t=0 to t=1) into the same PCA space. Clean 3-trace plot:
    GT (gray), SLERP (blue), Biharmonic (red), etc.
    """
    from sklearn.decomposition import PCA
    import plotly.graph_objects as go

    S = len(stages)

    # Fit PCA on the FINAL stage ground truth only
    gt_final_log1p = test_stage_log1p[-1]
    print(f"\n  Fitting PCA on {len(gt_final_log1p)} test cells at t=1.00 ({stages[-1]})...")
    pca = PCA(n_components=2, random_state=42)
    gt_emb = pca.fit_transform(gt_final_log1p.numpy())
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    all_counts = torch.expm1(torch.cat(test_stage_log1p, dim=0)).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())

    method_colors = {"MM+SLERP": "#1f77b4", "MM+SLERP+SquaredSpectral": "#d62728",
                     "MM+SLERP+Biharmonic": "#d62728",
                     "MM+SLERP+GlobalBiharmonic": "#ff7f0e",
                     "MM+PremetricBiharmonic": "#00a08a",
                     "MM+SI": "#2ca02c", "MM+SI+Biharmonic": "#9467bd",
                     "MM+BiharmonicVel": "#8c564b", "MM+BiharmonicWaypoint": "#e377c2",
                     "MM+Score_learned+Biharmonic": "#bcbd22"}

    fig = go.Figure()

    # Ground truth at t=1.00
    fig.add_trace(go.Scatter(
        x=gt_emb[:, 0], y=gt_emb[:, 1],
        mode="markers",
        marker=dict(size=5, color="#888888", opacity=0.4),
        name=f"Ground Truth {stages[-1]}",
    ))

    # Each method's chained prediction from t=0 → t=1.00
    source = test_stage_sphere[0]
    for mname, model in models_dict.items():
        print(f"    Generating t=1.00 predictions for {mname}...")
        n_steps = 50
        pred_sphere = generate_fisher_flow(
            model, source, n_steps=n_steps, t_start=0.0, t_end=1.0,
        )
        pred_comp = from_orthant(pred_sphere)
        pred_log1p = torch.log1p(pred_comp * median_libsize).numpy()
        pred_emb = pca.transform(pred_log1p)
        color = method_colors.get(mname, "#17becf")
        fig.add_trace(go.Scatter(
            x=pred_emb[:, 0], y=pred_emb[:, 1],
            mode="markers",
            marker=dict(size=6, opacity=0.6, color=color),
            name=mname,
        ))

    fig.update_layout(
        title="Chained Predictions at t=1.00 vs Ground Truth (PCA of final stage)",
        xaxis_title="PC 1", yaxis_title="PC 2",
        width=1000, height=750,
        legend=dict(font=dict(size=12)),
    )
    fig.write_html(out_path)
    print(f"    Saved PCA endpoint visualization to {out_path}")


def visualize_pca_grid(models_dict, test_stage_log1p, test_stage_sphere,
                       stage_times, stages, out_path="pca_grid.html"):
    """Per-method PCA subplots: PC1 vs PC2 and PC3 vs PC4 for each method.

    Fits PCA(n_components=4) on ALL ground-truth test cells. For each method
    (plus ground truth), shows two subplots: PC1 vs PC2 and PC3 vs PC4.
    Each timepoint is color-coded consistently across all panels. Chained
    predictions are integrated from t=0 test cells to each stage time.
    """
    from sklearn.decomposition import PCA
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    S = len(stages)
    all_test_log1p = torch.cat(test_stage_log1p, dim=0)

    print(f"\n  Fitting PCA-4 on {len(all_test_log1p)} test cells for grid visualization...")
    pca = PCA(n_components=4, random_state=42)
    emb_all = pca.fit_transform(all_test_log1p.numpy())
    var = pca.explained_variance_ratio_
    print(f"    Explained variance: PC1={var[0]:.3f} PC2={var[1]:.3f} PC3={var[2]:.3f} PC4={var[3]:.3f}")

    all_counts = torch.expm1(all_test_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    method_names = ["Ground Truth"] + list(models_dict.keys())
    n_methods = len(method_names)

    fig = make_subplots(
        rows=n_methods, cols=2,
        subplot_titles=[f"{m} — PC1 vs PC2" if c == 0 else f"{m} — PC3 vs PC4"
                        for m in method_names for c in range(2)],
        horizontal_spacing=0.06, vertical_spacing=0.08,
    )

    source = test_stage_sphere[0]

    # Generate all predictions once
    pred_embs = {}
    for mname, model in models_dict.items():
        print(f"    Generating chained predictions for {mname}...")
        method_embs = {}
        for i in range(S):
            if i == 0:
                # t=0 is the source — just project the test cells directly
                src_comp = from_orthant(test_stage_sphere[0].cpu())
                src_log1p = torch.log1p(src_comp * median_libsize).numpy()
                method_embs[i] = pca.transform(src_log1p)
            else:
                t_target = stage_times[i]
                n_steps = max(10, int(50 * t_target))
                pred_sphere = generate_fisher_flow(
                    model, source, n_steps=n_steps, t_start=0.0, t_end=t_target,
                )
                pred_comp = from_orthant(pred_sphere)
                pred_log1p = torch.log1p(pred_comp * median_libsize).numpy()
                method_embs[i] = pca.transform(pred_log1p)
        pred_embs[mname] = method_embs

    # Split GT embeddings by stage
    gt_embs = {}
    offset = 0
    for i in range(S):
        n_s = len(test_stage_log1p[i])
        gt_embs[i] = emb_all[offset:offset + n_s]
        offset += n_s

    # Plot: one row per method, two columns (PC1v2, PC3v4)
    for row_idx, mname in enumerate(method_names):
        is_gt = (mname == "Ground Truth")
        embs = gt_embs if is_gt else pred_embs[mname]

        for i in range(S):
            e = embs[i]
            color = stage_colors[i % len(stage_colors)]
            label = f"{stages[i]} (t={stage_times[i]:.2f})"
            show_legend = (row_idx == 0)

            # PC1 vs PC2
            fig.add_trace(go.Scatter(
                x=e[:, 0], y=e[:, 1],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.35),
                name=label,
                legendgroup=label,
                showlegend=show_legend,
            ), row=row_idx + 1, col=1)

            # PC3 vs PC4
            fig.add_trace(go.Scatter(
                x=e[:, 2], y=e[:, 3],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.35),
                name=label,
                legendgroup=label,
                showlegend=False,
            ), row=row_idx + 1, col=2)

    fig.update_layout(
        title="PCA Projections: Ground Truth vs Predicted Endpoints per Method",
        width=1200, height=350 * n_methods,
        legend=dict(font=dict(size=11)),
    )
    for r in range(1, n_methods + 1):
        fig.update_xaxes(title_text="PC 1", row=r, col=1)
        fig.update_yaxes(title_text="PC 2", row=r, col=1)
        fig.update_xaxes(title_text="PC 3", row=r, col=2)
        fig.update_yaxes(title_text="PC 4", row=r, col=2)

    fig.write_html(out_path)
    print(f"    Saved PCA grid visualization to {out_path}")


def _sphere_to_log1p(y_sphere: torch.Tensor, median_libsize: float) -> np.ndarray:
    comp = from_orthant(y_sphere.cpu())
    return torch.log1p(comp * median_libsize).numpy()


def visualize_pca_vector_field(
    models_dict,
    test_stage_log1p,
    test_stage_sphere,
    stage_times,
    stages,
    out_path="pca_vector_field.html",
    n_cells=250,
    arrow_dt=0.04,
):
    """Visualize instantaneous velocity snapshots after projection into PCA space.

    For each selected evaluation time, the routine:
    1. Integrates source cells from t=0 to t_eval.
    2. Evaluates the learned tangent velocity v(z_t, t_eval).
    3. Takes one small Euler step on the sphere, z_t -> z_t+dt.
    4. Projects both states into a PCA basis fit on all ground-truth test cells.

    This gives a practical projected vector-field view without depending on
    PHATE's out-of-sample transform.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    all_test_log1p = torch.cat(test_stage_log1p, dim=0)
    print(f"\n  Fitting PCA-2 on {len(all_test_log1p)} test cells for vector-field visualization...")
    pca = PCA(n_components=2, random_state=42)
    emb_all = pca.fit_transform(all_test_log1p.numpy())
    print(f"    Explained variance (PC1+PC2): {pca.explained_variance_ratio_.sum():.3f}")

    all_counts = torch.expm1(all_test_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())

    source_all = test_stage_sphere[0]
    rng = np.random.default_rng(0)
    subset_idx = rng.choice(len(source_all), size=min(n_cells, len(source_all)), replace=False)
    source_subset = source_all[subset_idx]

    # Show the learned field near each evaluation stage except the initial source.
    eval_indices = list(range(1, len(stages)))
    n_rows = len(models_dict)
    n_cols = len(eval_indices)
    subplot_titles = [
        f"{mname} @ t={stage_times[i]:.2f}"
        for mname in models_dict.keys()
        for i in eval_indices
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    # Background empirical embedding split per stage.
    stage_embs = []
    offset = 0
    for stage_cells in test_stage_log1p:
        n_s = len(stage_cells)
        stage_embs.append(emb_all[offset:offset + n_s])
        offset += n_s

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    model_colors = {
        "MM+SLERP": "#111111",
        "MM+SLERP+SquaredSpectral": "#d62728",
        "MM+SLERP+Biharmonic": "#d62728",
        "MM+SLERP+GlobalBiharmonic": "#ff7f0e",
        "MM+PremetricBiharmonic": "#00a08a",
        "MM+SI": "#2ca02c",
        "MM+SI+Biharmonic": "#4363d8",
        "MM+Score_learned+Biharmonic": "#f58231",
    }

    for row_idx, (mname, model) in enumerate(models_dict.items(), start=1):
        color = model_colors.get(mname, "#444444")
        print(f"    Computing PCA vector-field snapshots for {mname}...")
        for col_idx, stage_idx in enumerate(eval_indices, start=1):
            t_eval = stage_times[stage_idx]
            n_steps = max(10, int(50 * t_eval))
            z_t = generate_fisher_flow(
                model, source_subset, n_steps=n_steps, t_start=0.0, t_end=t_eval,
            )
            with torch.no_grad():
                t_vec = torch.full((len(z_t),), t_eval, device=z_t.device)
                vel = model(z_t.to(next(model.parameters()).device), t_vec)
                z_dev = z_t.to(vel.device)
                vel = vel - (vel * z_dev).sum(dim=-1, keepdim=True) * z_dev
                z_next = normalize_sphere(z_dev + arrow_dt * vel).cpu()

            z_t_emb = pca.transform(_sphere_to_log1p(z_t, median_libsize))
            z_next_emb = pca.transform(_sphere_to_log1p(z_next, median_libsize))

            gt_stage = stage_embs[stage_idx]
            fig.add_trace(
                go.Scatter(
                    x=gt_stage[:, 0],
                    y=gt_stage[:, 1],
                    mode="markers",
                    marker=dict(size=3, color=stage_colors[(stage_idx - 1) % len(stage_colors)], opacity=0.18),
                    name=f"GT t={t_eval:.2f}",
                    legendgroup=f"gt_{stage_idx}",
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=z_t_emb[:, 0],
                    y=z_t_emb[:, 1],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.50),
                    name=mname,
                    legendgroup=mname,
                    showlegend=(col_idx == 1),
                ),
                row=row_idx,
                col=col_idx,
            )

            for start, end in zip(z_t_emb, z_next_emb):
                fig.add_trace(
                    go.Scatter(
                        x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        mode="lines",
                        line=dict(color=color, width=1.0),
                        opacity=0.28,
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    fig.update_layout(
        title="Projected Vector-Field Snapshots in PCA Space",
        width=max(360 * n_cols, 900),
        height=max(320 * n_rows, 420),
        legend=dict(font=dict(size=11)),
    )
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            fig.update_xaxes(title_text="PC 1", row=row, col=col)
            fig.update_yaxes(title_text="PC 2", row=row, col=col)

    fig.write_html(out_path)
    print(f"    Saved PCA vector-field visualization to {out_path}")
