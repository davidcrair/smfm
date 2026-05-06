"""PCA visualization of flow matching on embryoid body.

Three side-by-side panels:
  1. Ground truth test cells
  2. MM+Linear (Euclidean FM in 1500-HVG log1p space)
  3. MM+SLERP (Fisher Flow on the positive orthant of the sphere)

PCA is fit once on the TRAIN log1p cells only (pooled over all stages);
all panels use the same 2D basis and the same per-timepoint color so they
are directly comparable. Sphere-based predictions are converted back to
log1p via the median library size of the test data before projection.

Per-timepoint legend entries toggle the corresponding trace in every panel.

Usage:
    uv run python scripts/plot_linear_fm_pca.py
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from surf.data.embryoid import load_embryoid_body
from surf.evaluation.generation import generate_euclidean_flow, generate_fisher_flow
from surf.evaluation.metrics import mmd_rbf
from surf.geometry.sphere import (
    from_orthant,
    normalize_sphere,
    to_compositional,
    to_orthant,
)
from surf.runtime import setup
from surf.training.euclidean_flow_trainer import train_multi_marginal_euclidean_flow
from surf.training.flow_trainer import train_multi_marginal_flow


STAGES = ["0-1", "2-3", "4-5", "6-7", "8-9"]
TIMES = [0.0, 0.25, 0.50, 0.75, 1.0]
STAGE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def sphere_to_log1p(y_sphere, median_libsize):
    """Map positive-orthant sphere coords back to log1p gene space."""
    comp = from_orthant(y_sphere)  # simplex (sum to 1)
    counts = comp * median_libsize
    return torch.log1p(counts.clamp(min=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="embryoid_body.h5ad")
    parser.add_argument("--n-hvg", type=int, default=1500)
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ot-subsample", type=int, default=5000)
    parser.add_argument("--n-gen-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="flow_matching_train_pca.html")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rt = setup("auto")

    split = load_embryoid_body(args.data, n_hvg=args.n_hvg, seed=args.seed)

    # Log1p-space tensors (as stored in the h5ad).
    train_log1p = [split["train"]["cells"][s].to(rt.device) for s in STAGES]
    test_log1p = [split["test"]["cells"][s] for s in STAGES]
    D = train_log1p[0].shape[1]

    # Sphere-space tensors (compositional -> sqrt -> normalize).
    train_sphere = [
        normalize_sphere(to_orthant(to_compositional(split["train"]["cells"][s]))).to(
            rt.device
        )
        for s in STAGES
    ]
    test_sphere = [
        normalize_sphere(to_orthant(to_compositional(split["test"]["cells"][s])))
        for s in STAGES
    ]
    print(f"  Gene dim: {D}")

    # PCA on train log1p only.
    X_train_all = torch.cat([c.cpu() for c in train_log1p], dim=0).numpy()
    print(f"\n  Fitting PCA-2 on {len(X_train_all)} TRAIN cells (all stages)...")
    pca = PCA(n_components=2, random_state=args.seed).fit(X_train_all)
    var = pca.explained_variance_ratio_
    print(f"    Explained variance (PC1+PC2): {var.sum():.3f}")

    # Median library size computed from test log1p, matching phate_plots.py.
    all_counts = torch.expm1(torch.cat(test_log1p, dim=0)).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())
    print(f"    median library size: {median_libsize:.1f}")

    # --- Train MM+Linear (Euclidean FM in log1p space) ---------------------
    print("\n  Training MM+Linear (Euclidean FM)...")
    linear_model, _ = train_multi_marginal_euclidean_flow(
        train_log1p,
        TIMES,
        D,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        ot_subsample=args.ot_subsample,
        label="MM+Linear",
    )
    src_log1p = test_log1p[0].to(rt.device)
    linear_preds = {}
    for i, t in enumerate(TIMES[1:], start=1):
        print(f"    Integrating MM+Linear to t={t:.2f} ({STAGES[i]})...")
        linear_preds[STAGES[i]] = generate_euclidean_flow(
            linear_model, src_log1p, n_steps=args.n_gen_steps, t_start=0.0, t_end=t
        )

    # --- Train MM+SLERP (Fisher Flow on the sphere) ------------------------
    print("\n  Training MM+SLERP (Fisher Flow)...")
    slerp_model, _ = train_multi_marginal_flow(
        train_sphere,
        TIMES,
        D,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        ot_subsample=args.ot_subsample,
        label="MM+SLERP",
    )
    src_sphere = test_sphere[0].to(rt.device)
    slerp_preds = {}
    for i, t in enumerate(TIMES[1:], start=1):
        print(f"    Integrating MM+SLERP to t={t:.2f} ({STAGES[i]})...")
        pred_sphere = generate_fisher_flow(
            slerp_model, src_sphere, n_steps=args.n_gen_steps, t_start=0.0, t_end=t
        )
        slerp_preds[STAGES[i]] = sphere_to_log1p(pred_sphere, median_libsize)

    # --- Evaluate chained + per-segment MMD^2 in SPHERE space -------------
    # Both methods are scored on the positive-orthant unit sphere (the space
    # MM+SLERP trains in). MM+Linear's log1p predictions are mapped back via
    # log1p -> counts -> compositional -> sqrt -> normalize, which is the
    # inverse of the sphere construction the SLERP family uses.
    S = len(STAGES)
    test_log1p_dev = [c.to(rt.device) for c in test_log1p]
    test_sphere_dev = [c.to(rt.device) for c in test_sphere]

    def log1p_to_sphere(x_log1p):
        comp = to_compositional(x_log1p)
        return normalize_sphere(to_orthant(comp))

    def eval_chained_sphere(generate_fn, src, to_sphere_fn):
        out = []
        for i in range(1, S):
            t_target = TIMES[i]
            n_steps = max(10, int(50 * t_target))
            pred_state = generate_fn(src, n_steps=n_steps, t_start=0.0, t_end=t_target)
            pred_sphere = to_sphere_fn(pred_state)
            out.append(float(mmd_rbf(pred_sphere, test_sphere_dev[i])))
        return np.array(out)

    def eval_per_segment_sphere(generate_fn, segment_sources, to_sphere_fn):
        out = []
        for i in range(S - 1):
            pred_state = generate_fn(
                segment_sources[i], n_steps=50, t_start=TIMES[i], t_end=TIMES[i + 1],
            )
            pred_sphere = to_sphere_fn(pred_state)
            out.append(float(mmd_rbf(pred_sphere, test_sphere_dev[i + 1])))
        return np.array(out)

    print("\n  Evaluating MM+Linear in sphere space...")
    lin_chained = eval_chained_sphere(
        lambda src, **kw: generate_euclidean_flow(linear_model, src, **kw),
        src_log1p,
        log1p_to_sphere,
    )
    lin_per_seg = eval_per_segment_sphere(
        lambda src, **kw: generate_euclidean_flow(linear_model, src, **kw),
        test_log1p_dev,
        log1p_to_sphere,
    )

    print("\n  Evaluating MM+SLERP in sphere space...")
    slerp_chained = eval_chained_sphere(
        lambda src, **kw: generate_fisher_flow(slerp_model, src, **kw),
        src_sphere,
        lambda y: y,  # already on the sphere
    )
    slerp_per_seg = eval_per_segment_sphere(
        lambda src, **kw: generate_fisher_flow(slerp_model, src, **kw),
        test_sphere_dev,
        lambda y: y,
    )

    chained_cols = [f"0->{i}" for i in range(1, S)]
    seg_cols = [f"{i}->{i + 1}" for i in range(S - 1)]

    def print_eval_table(title, cols, linear_vals, slerp_vals):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        header = f"  {'Method':<12}" + "  ".join(f"{c:>10}" for c in cols) + f"  {'mean':>10}"
        print(header)
        print("  " + "-" * (12 + 12 * (len(cols) + 1)))
        for name, vals in [("MM+Linear", linear_vals), ("MM+SLERP", slerp_vals)]:
            body = "  ".join(f"{v:>10.4f}" for v in vals)
            print(f"  {name:<12}{body}  {vals.mean():>10.4f}")

    print_eval_table(
        "Chained MMD^2 in sphere space (integrate 0 -> t_i)",
        chained_cols, lin_chained, slerp_chained,
    )
    print_eval_table(
        "Per-segment MMD^2 in sphere space (integrate i -> i+1)",
        seg_cols, lin_per_seg, slerp_per_seg,
    )

    # --- Project everything through the shared PCA -------------------------
    gt_emb_per_stage = [
        pca.transform(test_log1p[i].cpu().numpy()) for i in range(len(STAGES))
    ]
    lin_emb_per_stage = {
        s: pca.transform(linear_preds[s].cpu().numpy()) for s in STAGES[1:]
    }
    slerp_emb_per_stage = {
        s: pca.transform(slerp_preds[s].cpu().numpy()) for s in STAGES[1:]
    }

    all_pts = np.vstack(
        gt_emb_per_stage
        + list(lin_emb_per_stage.values())
        + list(slerp_emb_per_stage.values())
    )
    pad = 0.05 * (all_pts.max(axis=0) - all_pts.min(axis=0))
    xr = [all_pts[:, 0].min() - pad[0], all_pts[:, 0].max() + pad[0]]
    yr = [all_pts[:, 1].min() - pad[1], all_pts[:, 1].max() + pad[1]]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Ground truth (test)",
            "MM+Linear (Euclidean FM)",
            "MM+SLERP (Fisher Flow)",
        ),
        horizontal_spacing=0.06,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    def add(panel_col, emb, legend_label, group, color, show_legend):
        fig.add_trace(
            go.Scattergl(
                x=emb[:, 0],
                y=emb[:, 1],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.45),
                name=legend_label,
                legendgroup=group,
                showlegend=show_legend,
            ),
            row=1,
            col=panel_col,
        )

    for i, s in enumerate(STAGES):
        color = STAGE_COLORS[i]
        group = f"t={TIMES[i]:.2f}"
        legend_label = f"t={TIMES[i]:.2f} ({s})"

        add(1, gt_emb_per_stage[i], legend_label, group, color, show_legend=True)

        if i == 0:
            # t=0 is the source for both flow models — show it on all panels.
            add(2, gt_emb_per_stage[0], legend_label, group, color, show_legend=False)
            add(3, gt_emb_per_stage[0], legend_label, group, color, show_legend=False)
            continue

        add(2, lin_emb_per_stage[s], legend_label, group, color, show_legend=False)
        add(3, slerp_emb_per_stage[s], legend_label, group, color, show_legend=False)

    for col in (1, 2, 3):
        fig.update_xaxes(
            title_text=f"PC 1 ({var[0] * 100:.1f}%)", range=xr, row=1, col=col
        )
        fig.update_yaxes(range=yr, row=1, col=col)
    fig.update_yaxes(title_text=f"PC 2 ({var[1] * 100:.1f}%)", row=1, col=1)

    fig.update_layout(
        title=(
            f"Flow matching on embryoid body ({D}-HVG) — "
            f"PCA fit on TRAIN cells (var={var.sum():.2f})"
        ),
        width=2000,
        height=750,
        legend=dict(font=dict(size=12), title_text="Timepoint"),
    )

    out_path = Path(args.out)
    fig.write_html(out_path)
    print(f"\n  Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
