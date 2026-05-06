"""
Connectivity / spectral viability check for the GoM and Beijing PM2.5 datasets.

For each dataset:
  - Per-marginal sample counts and dimensionality.
  - Inter-marginal centroid distance vs intra-marginal spread.
  - kNN union graph at multiple k: connected components, sizes, mixed/pure
    composition by marginal, count of cross-marginal edges.
  - First few normalized-Laplacian eigenvalues (zeros = #components, then gap).

Spectral OT (compute_global_biharmonic_embedding in surf/ot/costs.py) builds a
single kNN graph on the union of all marginals. If that graph is disconnected
or only barely connected through accidental edges, the bottom eigenvectors
collapse to component indicators and the 1/lambda^p weighting blows up
inter-component costs -- making the OT plan refuse to cross marginals. So this
script's purpose is one binary verdict: union graph behaves like a single
manifold, or not.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
DOWNLOADS = Path("/Users/davidcrair/Downloads")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_gom() -> list[np.ndarray]:
    gom = _load_module("gom_data", DOWNLOADS / "data.py")
    bundle = gom.load_gom_data(data_dir=PROJ / "data", normalize=True, ot_coupling=False)
    return bundle["marginals_list"]


def load_beijing() -> list[np.ndarray]:
    beij = _load_module("beijing_data", DOWNLOADS / "data-2.py")
    bundle = beij.load_beijing_data(
        data_dir=PROJ / "data" / "beijing",
        station="Dingling",
        normalize=True,
        ot_coupling=False,
    )
    return bundle["marginals_list"]


def summarize_marginals(name: str, marginals: list[np.ndarray]) -> None:
    print(f"\n=== {name}: marginals ===")
    sizes = [len(m) for m in marginals]
    dim = marginals[0].shape[1]
    print(f"n_marginals={len(marginals)}  dim={dim}  sizes={sizes}  total={sum(sizes)}")
    centroids = np.array([m.mean(axis=0) for m in marginals])
    spreads = np.array([m.std(axis=0).mean() for m in marginals])
    pairwise = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    upper = pairwise[np.triu_indices(len(marginals), k=1)]
    print(f"intra-marginal mean stdev: {spreads.mean():.4f}  (per-marginal: {np.round(spreads,3).tolist()})")
    print(f"inter-marginal centroid distance: mean={upper.mean():.4f}  median={np.median(upper):.4f}  max={upper.max():.4f}")
    print(f"  ratio inter/intra (centroid_dist / mean_intra_std): {upper.mean()/spreads.mean():.3f}")


def union_kNN_diagnostics(
    name: str,
    marginals: list[np.ndarray],
    k_list: list[int] = (5, 15, 30),
) -> None:
    print(f"\n=== {name}: kNN union graph diagnostics ===")
    sizes = [len(m) for m in marginals]
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    combined = np.vstack(marginals)
    N = len(combined)
    print(f"N={N}, marginal labels in [0..{len(marginals)-1}]")

    for knn in k_list:
        if knn + 1 >= N:
            print(f"  knn={knn}: too few points")
            continue
        nn = NearestNeighbors(n_neighbors=knn + 1, metric="euclidean")
        nn.fit(combined)
        _, inds = nn.kneighbors(combined)

        rows = np.repeat(np.arange(N), knn + 1)
        cols = inds.reshape(-1)
        data = np.ones_like(rows, dtype=np.float64)
        W = csr_matrix((data, (rows, cols)), shape=(N, N))
        W = W.maximum(W.T)  # symmetrize as undirected
        n_comp, comp_labels = connected_components(W, directed=False)

        # Per-component composition
        comp_sizes = np.bincount(comp_labels)
        order = np.argsort(-comp_sizes)
        composition = []
        for c in order[:5]:
            mask = comp_labels == c
            ms = labels[mask]
            counts = np.bincount(ms, minlength=len(marginals))
            present = [(int(t), int(n)) for t, n in enumerate(counts) if n > 0]
            composition.append((int(comp_sizes[c]), present))

        # Cross-marginal edges (count edges between points with different labels)
        coo = W.tocoo()
        ii, jj = coo.row, coo.col
        mask = ii < jj  # unique edges
        ii, jj = ii[mask], jj[mask]
        cross = (labels[ii] != labels[jj]).sum()
        total_edges = len(ii)

        print(f"  knn={knn}: components={n_comp}  largest_sizes={comp_sizes[order[:5]].tolist()}")
        print(f"           top components composition (size, [(marginal,count),...]):")
        for sz, comp in composition:
            print(f"             size={sz}  composition={comp}")
        print(f"           total undirected edges={total_edges}  cross-marginal edges={cross}  "
              f"ratio={cross/total_edges:.3f}")

        # Spectrum on largest component (or the whole graph if connected)
        if n_comp == 1:
            target = W
            sub_labels = labels
        else:
            largest = order[0]
            keep = np.where(comp_labels == largest)[0]
            target = W[keep][:, keep]
            sub_labels = labels[keep]
        n_sub = target.shape[0]
        d = np.asarray(target.sum(axis=1)).flatten()
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        D_inv_sqrt = diags(d_inv_sqrt)
        L = diags(np.ones(n_sub)) - D_inv_sqrt @ target @ D_inv_sqrt
        try:
            keig = min(8, n_sub - 2)
            ev, _ = eigsh(L, k=keig, which="SM")
            ev = np.sort(ev)
            print(f"           bottom Laplacian eigenvalues (largest component, n={n_sub}): "
                  f"{np.round(ev, 5).tolist()}")
        except Exception as e:
            print(f"           eigsh failed: {e}")


if __name__ == "__main__":
    print("Loading GoM ocean-vortex data...")
    gom_marg = load_gom()
    summarize_marginals("GoM", gom_marg)
    union_kNN_diagnostics("GoM", gom_marg, k_list=[5, 10, 15, 30])

    print("\nLoading Beijing PM2.5 (Dingling) data...")
    bej_marg = load_beijing()
    summarize_marginals("Beijing PM2.5", bej_marg)
    union_kNN_diagnostics("Beijing PM2.5", bej_marg, k_list=[5, 15, 30])
