#!/usr/bin/env python
"""Plot the first nontrivial graph-Laplacian eigenvectors on the toy Y data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_toy_y_premetric_interpolations import (  # noqa: E402
    SQRT3,
    TRIANGLE_VERTICES,
    _draw_simplex,
    make_y_toy,
)


def build_normalized_laplacian(xy: np.ndarray, knn: int):
    from scipy.sparse import csr_matrix, diags
    from sklearn.neighbors import NearestNeighbors

    n = len(xy)
    n_neighbors = min(int(knn) + 1, n)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(xy)
    dists, inds = nn.kneighbors(xy)
    local_dists = dists[:, 1:]
    sigma = float(np.median(local_dists[local_dists > 1e-12]))
    sigma = max(sigma, 1e-8)

    weights = np.exp(-(dists[:, 1:] ** 2) / (2.0 * sigma**2))
    rows = np.repeat(np.arange(n), n_neighbors - 1)
    cols = inds[:, 1:].reshape(-1)
    data = weights.reshape(-1)
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)

    degree = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    L = diags(np.ones(n)) - diags(d_inv_sqrt) @ W @ diags(d_inv_sqrt)
    return L, W, degree, sigma


def compute_eigenvectors(xy: np.ndarray, *, knn: int, n_eig: int):
    from scipy.sparse.linalg import eigsh

    L, W, degree, sigma = build_normalized_laplacian(xy, knn)
    k = min(int(n_eig) + 1, len(xy) - 2)
    eigvals, eigvecs = eigsh(L, k=k, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs, W, degree, sigma


def plot_eigenvectors(
    xy: np.ndarray,
    labels: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    out_path: Path,
    *,
    n_show: int,
) -> None:
    n_show = min(n_show, eigvecs.shape[1] - 1)
    fig, axes = plt.subplots(
        1,
        n_show,
        figsize=(4.2 * n_show, 4.5),
        constrained_layout=True,
    )
    if n_show == 1:
        axes = [axes]

    for j, ax in enumerate(axes, start=1):
        _draw_simplex(ax)
        values = eigvecs[:, j]
        vmax = float(np.quantile(np.abs(values), 0.98))
        vmax = max(vmax, 1e-8)
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=values,
            s=18,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            edgecolor="none",
            alpha=0.92,
        )
        ax.scatter(
            xy[labels == 0, 0],
            xy[labels == 0, 1],
            s=7,
            c="black",
            alpha=0.25,
            edgecolor="none",
        )
        ax.set_title(f"eigenvector {j}\n$\\lambda$={eigvals[j]:.4g}")
        fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.01)

    fig.suptitle("First Nontrivial Normalized Graph-Laplacian Eigenvectors", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eigenmap(
    eigvecs: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    colors = np.asarray(["#555555", "#0072b2", "#d55e00"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for ax, dims in zip(axes, [(1, 2), (2, 3)]):
        ax.scatter(
            eigvecs[:, dims[0]],
            eigvecs[:, dims[1]],
            s=18,
            c=colors[labels],
            alpha=0.85,
            edgecolor="none",
        )
        ax.set_xlabel(f"eigenvector {dims[0]}")
        ax.set_ylabel(f"eigenvector {dims[1]}")
        ax.set_title(f"Laplacian eigenmap ({dims[0]}, {dims[1]})")
        ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(eigvals: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(np.arange(len(eigvals)), eigvals, marker="o", linewidth=1.4)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("normalized Laplacian eigenvalue")
    ax.set_title("Smallest Graph-Laplacian Eigenvalues")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-source", type=int, default=220)
    parser.add_argument("--n-target-per-branch", type=int, default=110)
    parser.add_argument("--noise", type=float, default=0.008)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--n-eig", type=int, default=8)
    parser.add_argument("--n-show", type=int, default=5)
    parser.add_argument("--out-dir", default="plots/toy_y_laplacian_eigenvectors")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    toy = make_y_toy(
        n_source=args.n_source,
        n_target_per_branch=args.n_target_per_branch,
        noise=args.noise,
        seed=args.seed,
    )
    xy = np.vstack([toy.source_xy, toy.target_xy])
    labels = np.concatenate(
        [
            np.zeros(len(toy.source_xy), dtype=np.int64),
            toy.target_branch.astype(np.int64),
        ]
    )

    eigvals, eigvecs, W, degree, sigma = compute_eigenvectors(
        xy,
        knn=args.knn,
        n_eig=max(args.n_eig, args.n_show),
    )
    plot_eigenvectors(
        xy,
        labels,
        eigvals,
        eigvecs,
        out_dir / "toy_y_laplacian_top5_on_simplex.png",
        n_show=args.n_show,
    )
    plot_eigenmap(eigvecs, labels, out_dir / "toy_y_laplacian_eigenmap.png")
    plot_spectrum(eigvals, out_dir / "toy_y_laplacian_spectrum.png")

    metadata = {
        "n_points": int(len(xy)),
        "knn": int(args.knn),
        "sigma": float(sigma),
        "n_edges": int(W.nnz // 2),
        "degree_min": float(degree.min()),
        "degree_median": float(np.median(degree)),
        "degree_max": float(degree.max()),
        "eigenvalues": [float(x) for x in eigvals.tolist()],
        "triangle_vertices": TRIANGLE_VERTICES.tolist(),
        "simplex_height": float(SQRT3 / 2.0),
    }
    (out_dir / "laplacian_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved Laplacian eigenvector plots to {out_dir}")
    print("First nontrivial eigenvalues:", ", ".join(f"{x:.5g}" for x in eigvals[1: args.n_show + 1]))


if __name__ == "__main__":
    main()
