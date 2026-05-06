"""
PC1 vs PC2 visualization of model predictions across a trajectory dataset.

Supports pancreas, bonemarrow, paul15, and embryoid. For each method in
the linear-trainer table (MM+Linear baseline + five
MM+Linear+SquaredSpectral@alpha values), this script:

  1. Loads the dataset's split (sphere-encoded training cells).
  2. Trains the method's flow model for ~3000 iterations.
  3. Integrates from test-set source cells (stage 0) forward to each
     downstream stage time.
  4. Projects all model predictions + ground-truth test cells into the
     same 2-D PCA basis (fit on the joint training cell cloud).
  5. Plots one row of panels: leftmost = ground truth test cells, then
     one panel per method, all sharing the same axes.

Output: outputs/{dataset}_pca_predictions_split{seed}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def apply_paper_style(style: str = "paper"):
    """Apply a publication-ready matplotlib style.

    Uses the ``scienceplots`` package (Nature/IEEE-style preset) when
    available, falling back to manual rcParams otherwise. Overrides
    font sizes on top of the base style for visibility at presentation
    and ICML two-column scales.

    style:
      - "paper":        clean serif-like, medium-large fonts, grid on.
      - "presentation": same family but bigger, thicker, no minor ticks.
      - "default":      no style change (whatever matplotlib defaults to).
    """
    if style == "default":
        return 18.0
    try:
        import scienceplots  # noqa: F401  (registers `science` styles)
        base = ["science", "no-latex"]            # sans-serif clones
    except ImportError:
        base = ["seaborn-v0_8-paper"]
    plt.style.use(base)

    if style == "presentation":
        size = dict(font=22, title=26, label=22, tick=18, legend=20,
                    line=2.5, marker=56)
    else:  # "paper"
        size = dict(font=18, title=22, label=20, tick=14, legend=18,
                    line=1.6, marker=36)

    mpl.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        size["font"],
        "axes.titlesize":   size["title"],
        "axes.labelsize":   size["label"],
        "axes.titleweight": "bold",
        "xtick.labelsize":  size["tick"],
        "ytick.labelsize":  size["tick"],
        "legend.fontsize":  size["legend"],
        "lines.linewidth":  size["line"],
        "axes.linewidth":   1.0,
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "grid.linewidth":   0.6,
        "figure.dpi":       100,
        "savefig.dpi":      200,
        "savefig.bbox":     "tight",
    })
    return float(size["marker"])

from surf.runtime import setup as runtime_setup, get as get_runtime
from surf.data.pancreas import load_pancreas
from surf.data.bonemarrow import load_bonemarrow
from surf.data.paul15 import load_paul15
from surf.data.embryoid import load_embryoid_body
from surf.geometry.sphere import to_compositional, to_orthant, normalize_sphere
from surf.training.euclidean_flow_trainer import train_multi_marginal_euclidean_flow
from surf.evaluation.eval_runner import _generate_prediction, _predict_to_compositional
from surf.ot.costs import compute_euclidean_cost_matrix, make_spectral_cost_fn


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")


def _load_pancreas(n_hvg, seed, max_cells_per_stage):
    return load_pancreas(n_hvg=n_hvg, seed=seed,
                         max_cells_per_stage=max_cells_per_stage)


def _load_bonemarrow(n_hvg, seed, max_cells_per_stage):
    return load_bonemarrow(n_hvg=n_hvg, seed=seed,
                           max_cells_per_stage=max_cells_per_stage)


def _load_paul15(n_hvg, seed, max_cells_per_stage):
    return load_paul15(n_hvg=n_hvg, seed=seed,
                       max_cells_per_stage=max_cells_per_stage)


def _load_embryoid(n_hvg, seed, max_cells_per_stage):
    return load_embryoid_body(
        path=str(PROJ / "embryoid_body.h5ad"),
        n_hvg=n_hvg, seed=seed,
        max_cells_per_stage=max_cells_per_stage,
    )


DATASET_LOADERS = {
    "pancreas":   _load_pancreas,
    "bonemarrow": _load_bonemarrow,
    "paul15":     _load_paul15,
    "embryoid":   _load_embryoid,
}

# Stage colors per the user's request: t=0 red, t=0.25 orange, t=0.5 light green,
# t=0.75 light blue, t=1.0 purple, plus a fallback black for >5 stages.
STAGE_COLORS = ["red", "orange", "limegreen", "lightskyblue", "purple",
                "black", "brown", "magenta"]


def cost_fn_for_method(name: str):
    """Return the OT cost-function callable for a given method name."""
    if name == "MM+Linear":
        return compute_euclidean_cost_matrix
    if name.startswith("MM+Linear+SquaredSpectral@alpha="):
        alpha = float(name.split("=", 1)[1])
        # method_registry's convention: w_effective(lambda) = lambda^{-2*weight_power},
        # so alpha = 2 * weight_power -> weight_power = alpha / 2.
        return make_spectral_cost_fn(
            knn=15,
            n_eig=50,
            spectral_family="power",
            weight_power=alpha / 2.0,
            diffusion_time=1.0,
        )
    raise ValueError(f"Unsupported method: {name!r}")


def display_label(name: str) -> str:
    if name == "MM+Linear":
        return "Linear FM"
    if name.startswith("MM+Linear+SquaredSpectral@alpha="):
        alpha = name.split("=", 1)[1]
        return rf"SMFM, $\alpha={alpha}$"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=list(DATASET_LOADERS.keys()),
                        default="pancreas", help="Dataset to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Data split seed")
    parser.add_argument("--n-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--max-cells-per-stage", type=int, default=None,
                        help="Cap cells per stage (default: dataset default; "
                             "useful for embryoid where full size is ~5000/stage).")
    parser.add_argument("--style", choices=["paper", "presentation", "default"],
                        default="paper",
                        help="Matplotlib styling preset (default: paper)")
    args = parser.parse_args()
    marker_size = apply_paper_style(args.style)

    runtime_setup(device="auto")
    rt = get_runtime()
    print(f"Device: {rt.device}")

    loader = DATASET_LOADERS[args.data]
    print(f"\nLoading {args.data} split seed={args.seed}...")
    data = loader(
        n_hvg=args.n_hvg,
        seed=args.seed,
        max_cells_per_stage=args.max_cells_per_stage,
    )
    stages = data["train"]["stages"]
    S = len(stages)
    stage_times = [i / (S - 1) for i in range(S)]

    # Sphere-encoded train cells (this is what the linear trainer consumes).
    train_log1p = [data["train"]["cells"][s].to(rt.device) for s in stages]
    train_sphere = [
        normalize_sphere(to_orthant(to_compositional(c))).to(rt.device)
        for c in train_log1p
    ]
    test_log1p = [data["test"]["cells"][s] for s in stages]
    test_comp = [to_compositional(c) for c in test_log1p]   # log1p HVG, what we'll PCA on
    test_sphere = [normalize_sphere(to_orthant(tc)).to(rt.device) for tc in test_comp]

    D = train_sphere[0].shape[1]
    methods = [
        "MM+Linear",
        "MM+Linear+SquaredSpectral@alpha=0",
        "MM+Linear+SquaredSpectral@alpha=0.5",
        "MM+Linear+SquaredSpectral@alpha=1",
        "MM+Linear+SquaredSpectral@alpha=1.5",
        "MM+Linear+SquaredSpectral@alpha=2",
    ]

    # Fit PCA on the joint training cell cloud (in log1p HVG space) so all
    # panels share a common 2-D coordinate frame.
    print("\nFitting shared PCA on training cells...")
    train_cells_np = np.concatenate([c.cpu().numpy() for c in train_log1p], axis=0)
    pca = PCA(n_components=2)
    pca.fit(train_cells_np)
    print(f"  explained variance ratio: {pca.explained_variance_ratio_}")

    # Train each method, generate predictions at every downstream stage time.
    predictions: dict[str, list[np.ndarray]] = {}
    for method in methods:
        print(f"\n{'='*60}\nTRAINING {method}\n{'='*60}")
        cost_fn = cost_fn_for_method(method)
        model, _losses = train_multi_marginal_euclidean_flow(
            stage_cells=train_sphere,
            stage_times=stage_times,
            D=D,
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            label=method,
            cost_fn=cost_fn,
        )
        model.eval()

        source_sphere = test_sphere[0]
        per_stage_pred_comp = []
        for i in range(S):
            if i == 0:
                # ground-truth source cells (no flow integration)
                pred_comp_np = test_comp[0].cpu().numpy()
            else:
                t_target = stage_times[i]
                n_steps = max(10, int(50 * t_target))
                with torch.no_grad():
                    pred_state = _generate_prediction(
                        model, source_sphere, "sphere", n_steps, 0.0, t_target,
                        score_n=None, a=0.0, score_net_sigma=0.0, inf_sigma=0.0,
                    )
                pred_comp = _predict_to_compositional(pred_state, "sphere")
                pred_comp_np = pred_comp.detach().cpu().numpy()
            per_stage_pred_comp.append(pred_comp_np)
        predictions[method] = per_stage_pred_comp

    # Plot
    print("\nRendering figure...")
    test_proj_per_stage = [pca.transform(test_comp[i].cpu().numpy()) for i in range(S)]

    # Compute global axis limits over GT test + all predictions
    all_proj = np.concatenate(
        [test_proj_per_stage[i] for i in range(S)] +
        [pca.transform(predictions[m][i]) for m in methods for i in range(S)],
        axis=0,
    )
    pad = 0.05 * (all_proj.max(axis=0) - all_proj.min(axis=0))
    xlim = (all_proj[:, 0].min() - pad[0], all_proj[:, 0].max() + pad[0])
    ylim = (all_proj[:, 1].min() - pad[1], all_proj[:, 1].max() + pad[1])

    # Single row: leftmost = ground truth, then one panel per method.
    n_panels = 1 + len(methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 5.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    # Panel 0: ground truth test cells, colored by stage.
    # Scatter points are rasterized=True so the SVG export embeds them
    # as PNG (small file size on dense scatters) while everything else
    # (axes, labels, ticks, legend) stays vector.
    ax = axes[0]
    for i in range(S):
        color = STAGE_COLORS[i % len(STAGE_COLORS)]
        ax.scatter(
            test_proj_per_stage[i][:, 0], test_proj_per_stage[i][:, 1],
            c=color, s=marker_size, alpha=0.85, marker="o",
            edgecolors="none",
            rasterized=True,
        )
    ax.set_title("Ground truth (test cells)")

    # Remaining panels: per-method predictions (no GT background).
    for ax, method in zip(axes[1:], methods):
        for i in range(S):
            color = STAGE_COLORS[i % len(STAGE_COLORS)]
            proj = pca.transform(predictions[method][i])
            ax.scatter(
                proj[:, 0], proj[:, 1],
                c=color, s=marker_size, alpha=0.85, marker="o",
                edgecolors="none",
                rasterized=True,
            )
        ax.set_title(display_label(method))

    # Common axis cosmetic styling. Drop tick labels and variance % from
    # axis labels so the axes act purely as a visual reference frame
    # (the PCA basis is fit-on-train, so absolute coordinates carry no
    # interpretable meaning).
    for j, ax in enumerate(axes):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("PC1")
        if j == 0:
            ax.set_ylabel("PC2")
        ax.set_xticks([])
        ax.set_yticks([])

    # Single shared legend below the row
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=STAGE_COLORS[i],
                   markersize=mpl.rcParams["legend.fontsize"],
                   markeredgecolor='none',
                   label=f"{stages[i]} (t={stage_times[i]:.2f})")
        for i in range(S)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=S,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               handletextpad=0.4, columnspacing=1.2)

    fig.suptitle(
        f"{args.data.capitalize()} predictions, PC1 vs PC2 (split seed={args.seed}, "
        f"{args.n_iters} iters, sphere $\\mathbb{{R}}^{{{D}}}$)",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    out_dir = PROJ / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.data}_pca_predictions_split{args.seed}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {out_path}")

    # SVG with rasterized scatter (vector axes/labels, PNG-embedded points).
    # dpi controls the resolution of the embedded raster.
    svg_path = out_dir / f"{args.data}_pca_predictions_split{args.seed}.svg"
    fig.savefig(svg_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {svg_path}")

    # PDF (also picks up rasterized=True hint -> vector axes + PNG-embedded
    # scatter; matches the SVG output but is what LaTeX wants natively).
    pdf_path = out_dir / f"{args.data}_pca_predictions_split{args.seed}.pdf"
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    print(f"  Wrote {pdf_path}")


if __name__ == "__main__":
    main()
