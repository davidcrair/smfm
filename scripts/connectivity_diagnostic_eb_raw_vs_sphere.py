"""
EB-only connectivity + spectral-gap diagnostic comparing the *sphere-encoded*
HVG representation (where MM+Linear+SquaredSpectral wins on pancreas) against
the *raw log1p HVG* representation (where the EB-HVG-OTPFM sweep currently
loses to MM+Linear).

For each representation we:
  1. Build the symmetrized kNN union graph at varying k.
  2. Count connected components.
  3. Compute the smallest 5 non-trivial eigenvalues of the symmetrically
     normalized Laplacian (Lsym = I - D^{-1/2} W D^{-1/2}). lambda_2 is the
     spectral-gap proxy: small lambda_2 means 1/lambda^alpha for alpha>=1
     blows up the contribution of one near-trivial / near-disconnected mode,
     which is exactly what would make the spectral cost worse than Linear.

Output: prints a table; writes CSV under surf_latex/final_report/figures/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, eye as speye, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
K_VALUES = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
K_FOR_SPECTRAL = 15  # the kNN value where the spectral cost is built


def load_eb_raw_and_sphere():
    """Return (raw_marginals, sphere_marginals) per training stage."""
    from surf.data.embryoid import load_embryoid_body
    from surf.geometry.sphere import to_orthant, to_compositional, normalize_sphere

    data = load_embryoid_body(str(PROJ / "embryoid_body.h5ad"), n_hvg=2000, seed=42)
    stages = data["train"]["stages"]
    raw, sph = [], []
    for s in stages:
        x = data["train"]["cells"][s]
        # raw log1p HVG (this is what `representation: euclidean_raw` consumes)
        raw.append(x.numpy() if hasattr(x, "numpy") else np.asarray(x))
        # Fisher-Rao sphere (this is what `representation: sphere` / pancreas consumes)
        sph.append(normalize_sphere(to_orthant(to_compositional(x))).numpy())
    return raw, sph


def build_knn_union(marginals, k: int, metric: str) -> csr_matrix:
    combined = np.vstack(marginals)
    n = len(combined)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(combined)
    _, inds = nn.kneighbors(combined)
    rows = np.repeat(np.arange(n), k + 1)
    cols = inds.reshape(-1)
    data = np.ones_like(rows, dtype=np.float64)
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    # zero the diagonal
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def n_components(W: csr_matrix) -> int:
    n_comp, _ = connected_components(W, directed=False)
    return n_comp


def smallest_eigs_lsym(W: csr_matrix, k_eig: int = 6):
    """Smallest k_eig eigenvalues of the symmetrically-normalized Laplacian."""
    deg = np.asarray(W.sum(axis=1)).ravel()
    # guard against isolated nodes; for connected graph deg>0 already
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = diags(1.0 / np.sqrt(deg_safe))
    n = W.shape[0]
    Lsym = speye(n) - d_inv_sqrt @ W @ d_inv_sqrt
    # eigsh with sigma=0, mode='normal' or shift-invert; SM is fine but slow.
    # Use shift-invert with sigma=-1e-6 to land near 0 robustly.
    vals, _ = eigsh(Lsym, k=k_eig, sigma=-1e-6, which="LM")
    vals.sort()
    return vals


def main():
    print("Loading EB at HVG and computing both representations...")
    raw, sph = load_eb_raw_and_sphere()
    sizes = [len(m) for m in raw]
    print(f"  5 marginals, sizes={sizes}, total={sum(sizes)}")
    print(f"  raw dim={raw[0].shape[1]}  sphere dim={sph[0].shape[1]}")
    print()

    print("--- Connected components in symmetrized kNN union graph ---")
    print(f"{'k':>4}  {'raw HVG (Eucl)':>18}  {'raw HVG (cos)':>16}  {'sphere (cos)':>16}")
    rows = []
    for k in K_VALUES:
        W_raw_eu = build_knn_union(raw, k, metric="euclidean")
        W_raw_co = build_knn_union(raw, k, metric="cosine")
        W_sph    = build_knn_union(sph, k, metric="cosine")
        c_raw_eu = n_components(W_raw_eu)
        c_raw_co = n_components(W_raw_co)
        c_sph    = n_components(W_sph)
        rows.append((k, c_raw_eu, c_raw_co, c_sph))
        print(f"{k:>4}  {c_raw_eu:>18}  {c_raw_co:>16}  {c_sph:>16}")
    print()

    print(f"--- Smallest Laplacian eigenvalues (Lsym), k_knn={K_FOR_SPECTRAL} ---")
    print("  smaller lambda_2 => 1/lambda^alpha blows up that mode")
    print()
    eig_results = {}
    for label, marginals, metric in [
        ("raw HVG  (Eucl)", raw, "euclidean"),
        ("raw HVG  (cos) ", raw, "cosine"),
        ("sphere   (cos) ", sph, "cosine"),
    ]:
        W = build_knn_union(marginals, K_FOR_SPECTRAL, metric=metric)
        vals = smallest_eigs_lsym(W, k_eig=6)
        eig_results[label] = vals
        formatted = "  ".join(f"{v:.6f}" for v in vals)
        print(f"  {label}: {formatted}")
        # also report 1/lambda^alpha for alpha=1, alpha=2 (avoiding lambda_1 ~ 0)
        if len(vals) >= 3:
            l2, l3 = vals[1], vals[2]
            print(f"    1/lambda_2 ={1.0/max(l2,1e-12):.2f}  "
                  f"1/lambda_2^2 ={1.0/max(l2,1e-12)**2:.2f}  "
                  f"1/lambda_3 ={1.0/max(l3,1e-12):.2f}")
    print()

    out_dir = PROJ / "surf_latex" / "final_report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "connectivity_diagnostic_eb_raw_vs_sphere.csv"
    with open(csv_path, "w") as f:
        f.write("k,raw_HVG_Eucl,raw_HVG_cos,sphere_cos\n")
        for k, a, b, c in rows:
            f.write(f"{k},{a},{b},{c}\n")
    print(f"Wrote {csv_path}")

    eig_csv = out_dir / "spectral_gap_eb_raw_vs_sphere.csv"
    with open(eig_csv, "w") as f:
        f.write("representation,lambda_1,lambda_2,lambda_3,lambda_4,lambda_5,lambda_6\n")
        for label, vals in eig_results.items():
            f.write(f"{label.strip()}," + ",".join(f"{v:.8f}" for v in vals) + "\n")
    print(f"Wrote {eig_csv}")


if __name__ == "__main__":
    main()
