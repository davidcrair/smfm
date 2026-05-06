from __future__ import annotations

import argparse
import csv
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import PCA

from surf.evaluation.metrics import fgd, mmd_rbf, swd


def select_top_hvgs(adata: ad.AnnData, n_hvg: int) -> tuple[ad.AnnData, np.ndarray]:
    if "highly_variable" not in adata.var.columns:
        raise ValueError("AnnData is missing `highly_variable` annotations")

    hvg_mask = adata.var["highly_variable"].to_numpy()
    hvg_idx = np.where(hvg_mask)[0]
    if len(hvg_idx) < n_hvg:
        raise ValueError(
            f"Requested {n_hvg} HVGs but only found {len(hvg_idx)} highly variable genes"
        )

    disp = adata.var["dispersions_norm"].to_numpy()[hvg_idx]
    ranked_hvg_idx = hvg_idx[np.argsort(disp)[::-1]]
    selected = ranked_hvg_idx[:n_hvg]
    return adata[:, selected].copy(), selected


def to_dense_float32(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def centered_total_variance(x: np.ndarray) -> float:
    return float(x.var(axis=0, ddof=0).sum())


def centered_total_variance_any(x) -> float:
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
        return float(var.sum())
    arr = np.asarray(x, dtype=np.float32)
    return centered_total_variance(arr)


def gaussian_w2_same_basis(x: np.ndarray, x_recon: np.ndarray) -> tuple[float, float]:
    """Empirical Fréchet/W2 between two clouds using full mean/cov estimates."""
    mu_x = x.mean(axis=0)
    mu_y = x_recon.mean(axis=0)
    xc = x - mu_x
    yc = x_recon - mu_y
    cov_x = (xc.T @ xc) / x.shape[0]
    cov_y = (yc.T @ yc) / x_recon.shape[0]

    evals_x, evecs_x = np.linalg.eigh(cov_x)
    evals_x = np.clip(evals_x, 0.0, None)
    sqrt_cov_x = (evecs_x * np.sqrt(evals_x)) @ evecs_x.T
    middle = sqrt_cov_x @ cov_y @ sqrt_cov_x
    evals_mid = np.linalg.eigvalsh(middle)
    evals_mid = np.clip(evals_mid, 0.0, None)

    mean_term = float(np.sum((mu_x - mu_y) ** 2))
    w2_sq = mean_term + float(np.trace(cov_x) + np.trace(cov_y) - 2.0 * np.sum(np.sqrt(evals_mid)))
    w2_sq = max(w2_sq, 0.0)
    return float(np.sqrt(w2_sq)), w2_sq


def sample_metric_estimates(
    x: np.ndarray,
    x_recon: np.ndarray,
    sample_size: int,
    repeats: int,
    base_seed: int,
    swd_projections: int,
) -> dict[str, float]:
    rng = np.random.default_rng(base_seed)
    n = x.shape[0]
    sample_size = min(sample_size, n)

    mmd_vals: list[float] = []
    swd_vals: list[float] = []
    for _ in range(repeats):
        idx = rng.choice(n, size=sample_size, replace=False)
        xt = torch.from_numpy(x[idx])
        yt = torch.from_numpy(x_recon[idx])
        mmd_vals.append(float(mmd_rbf(xt, yt)))
        swd_vals.append(float(swd(xt, yt, n_projections=swd_projections)))

    return {
        "mmd2_mean": float(np.mean(mmd_vals)),
        "mmd2_std": float(np.std(mmd_vals, ddof=0)),
        "swd_mean": float(np.mean(swd_vals)),
        "swd_std": float(np.std(swd_vals, ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="embryoid_body.h5ad")
    parser.add_argument("--n-hvg", type=int, default=1500)
    parser.add_argument("--dims", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swd-projections", type=int, default=50)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input)
    full_total_var = centered_total_variance_any(adata.X)

    adata_hvg, selected_idx = select_top_hvgs(adata, args.n_hvg)
    x = to_dense_float32(adata_hvg.X)
    ambient_total_var = centered_total_variance(x)
    ambient_frac_of_full = ambient_total_var / full_total_var

    rows: list[dict[str, float | int | str]] = []
    identity_row: dict[str, float | int | str] = {
        "representation": f"{args.n_hvg}_HVG_identity",
        "dims": args.n_hvg,
        "ambient_cve": 1.0,
        "full_gene_variance_fraction": ambient_frac_of_full,
        "gaussian_w2": 0.0,
        "gaussian_w2_sq": 0.0,
        "fgd": 0.0,
        "mmd2_mean": 0.0,
        "mmd2_std": 0.0,
        "swd_mean": 0.0,
        "swd_std": 0.0,
    }
    rows.append(identity_row)

    max_dim = max(args.dims)
    pca = PCA(n_components=max_dim, svd_solver="randomized", random_state=args.seed)
    z = pca.fit_transform(x)
    vr = pca.explained_variance_ratio_

    for k in args.dims:
        z_k = z[:, :k]
        x_recon = (z_k @ pca.components_[:k]) + pca.mean_

        w2, w2_sq = gaussian_w2_same_basis(x, x_recon)
        approx_fgd = float(fgd(x, x_recon))
        sample_metrics = sample_metric_estimates(
            x,
            x_recon.astype(np.float32, copy=False),
            sample_size=args.sample_size,
            repeats=args.repeats,
            base_seed=args.seed + k,
            swd_projections=args.swd_projections,
        )

        rows.append(
            {
                "representation": f"pca_{k}_reconstruction",
                "dims": k,
                "ambient_cve": float(vr[:k].sum()),
                "full_gene_variance_fraction": float(vr[:k].sum()) * ambient_frac_of_full,
                "gaussian_w2": w2,
                "gaussian_w2_sq": w2_sq,
                "fgd": approx_fgd,
                **sample_metrics,
            }
        )

    csv_path = outdir / f"embryoid_pca_oracle_metrics_hvg{args.n_hvg}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "representation",
                "dims",
                "ambient_cve",
                "full_gene_variance_fraction",
                "gaussian_w2",
                "gaussian_w2_sq",
                "fgd",
                "mmd2_mean",
                "mmd2_std",
                "swd_mean",
                "swd_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Selected top {args.n_hvg} HVGs from {adata.n_vars} genes")
    print(f"Ambient HVG variance fraction of full gene matrix: {ambient_frac_of_full:.6f}")
    print(
        f"MMD/SWD estimated with sample_size={args.sample_size}, "
        f"repeats={args.repeats}, swd_projections={args.swd_projections}"
    )
    print(f"Wrote {csv_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
