from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def select_repo_hvgs(adata: ad.AnnData, n_hvg: int = 2000) -> tuple[ad.AnnData, np.ndarray]:
    """Match the repo loader's HVG selection logic."""
    if "highly_variable" not in adata.var.columns:
        return adata, np.arange(adata.n_vars)

    hvg_mask = adata.var["highly_variable"].to_numpy()
    hvg_idx = np.where(hvg_mask)[0]
    if len(hvg_idx) == 0:
        return adata, np.arange(adata.n_vars)

    if len(hvg_idx) > n_hvg:
        disp = adata.var["dispersions_norm"].to_numpy()[hvg_idx]
        top_idx = hvg_idx[np.argsort(disp)[::-1][:n_hvg]]
    else:
        top_idx = hvg_idx

    return adata[:, top_idx].copy(), top_idx


def to_dense_float32(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def repo_probability_preprocessing(log1p_expr: np.ndarray) -> np.ndarray:
    """Mirror surf.geometry.sphere.to_compositional -> to_orthant."""
    counts = np.expm1(log1p_expr).clip(min=0.0)
    counts = counts + 1e-8
    counts = counts / counts.sum(axis=1, keepdims=True)
    orthant = np.sqrt(counts)
    orthant = orthant / np.linalg.norm(orthant, axis=1, keepdims=True).clip(min=1e-8)
    return orthant.astype(np.float32, copy=False)


def fit_pca(data: np.ndarray, n_components: int, random_state: int) -> PCA:
    n_components = min(n_components, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(data)
    return pca


def plot_scree(
    explained_variance_ratio: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
) -> None:
    dims = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative = np.cumsum(explained_variance_ratio)

    fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax1.bar(dims, explained_variance_ratio * 100.0, color="#4878cf", alpha=0.85, width=0.9)
    ax1.plot(dims, explained_variance_ratio * 100.0, color="#1f3b73", linewidth=1.5)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)", color="#1f3b73")
    ax1.tick_params(axis="y", labelcolor="#1f3b73")
    ax1.set_xlim(0.5, len(dims) + 0.5)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    ax2 = ax1.twinx()
    ax2.plot(dims, cumulative * 100.0, color="#d65f5f", linewidth=2.2)
    ax2.set_ylabel("Cumulative explained variance (%)", color="#d65f5f")
    ax2.tick_params(axis="y", labelcolor="#d65f5f")
    ax2.set_ylim(0, min(100.0, max(100.0, float(cumulative[-1] * 100.0) + 2.0)))

    ax1.set_title(title)
    fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=10, color="#444444")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="embryoid_body.h5ad")
    parser.add_argument("--outdir", default="plots")
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    adata = ad.read_h5ad(input_path)
    adata_hvg, _ = select_repo_hvgs(adata, n_hvg=args.n_hvg)
    x_log1p = to_dense_float32(adata_hvg.X)

    # Paper-style interpretation: use the preprocessed AnnData matrix and the
    # same PCA settings recorded in the file metadata (HVGs + zero-centered PCA).
    paper_pca = fit_pca(x_log1p, n_components=args.n_components, random_state=args.seed)
    paper_subtitle = (
        f"Input: log1p matrix on {x_log1p.shape[1]} HVGs "
        f"(matching AnnData PCA metadata: zero-centered, HVG-only). "
        f"Paper also z-scores the PC scores after PCA."
    )
    plot_scree(
        paper_pca.explained_variance_ratio_,
        outdir / "embryoid_pca_scree_paper_style.png",
        "Embryoid Body PCA Scree: Paper-Style Preprocessing",
        paper_subtitle,
    )

    repo_prob = repo_probability_preprocessing(x_log1p)
    repo_pca = fit_pca(repo_prob, n_components=args.n_components, random_state=args.seed)
    repo_subtitle = (
        f"Input: repo probability preprocessing on {x_log1p.shape[1]} HVGs "
        f"(expm1 -> row-normalize -> sqrt positive orthant)."
    )
    plot_scree(
        repo_pca.explained_variance_ratio_,
        outdir / "embryoid_pca_scree_repo_probdist.png",
        "Embryoid Body PCA Scree: Repo Probability Preprocessing",
        repo_subtitle,
    )

    print("Wrote:")
    print(outdir / "embryoid_pca_scree_paper_style.png")
    print(outdir / "embryoid_pca_scree_repo_probdist.png")
    print()
    print(
        f"Paper-style cumulative variance at {len(paper_pca.explained_variance_ratio_)} PCs: "
        f"{paper_pca.explained_variance_ratio_.sum():.4f}"
    )
    print(
        f"Repo-prob cumulative variance at {len(repo_pca.explained_variance_ratio_)} PCs: "
        f"{repo_pca.explained_variance_ratio_.sum():.4f}"
    )


if __name__ == "__main__":
    main()
