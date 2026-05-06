"""
OT cost matrix computations: biharmonic (spectral) and PHATE-based costs.
"""
import os

import numpy as np
import torch


def _knn_metric() -> str:
    """kNN-graph metric for spectral cost construction.

    Default 'cosine' is appropriate for sphere-encoded (Hellinger) data where
    cells lie on the unit hypersphere. For log1p HVG ambient training, set
    SMFM_KNN_METRIC=euclidean -- cosine throws away the magnitude axis that
    log1p preserves and that the spectral eigendecomposition needs to
    recover the trajectory.
    """
    return os.environ.get("SMFM_KNN_METRIC", "cosine")


def _knn_distances_to_weights(dists: np.ndarray, metric: str):
    """Convert per-edge distances from sklearn.NearestNeighbors into Gaussian
    weights, handling cosine vs Euclidean conventions consistently.

    Returns (effective_distances, sigma, weights) for downstream use.
    """
    if metric == "cosine":
        # sklearn cosine distance = 1 - cos(theta); convert to arc-length.
        cos_sim = np.clip(1.0 - dists, -1 + 1e-6, 1 - 1e-6)
        d = np.arccos(cos_sim)  # great-circle arc, well-defined and bounded
    else:
        d = dists  # already a metric distance (e.g., Euclidean)
    sigma = float(np.median(d[:, 1:])) if d.shape[1] > 1 else 1.0
    sigma = max(sigma, 1e-8)
    weights = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return d, sigma, weights


def _spectral_embedding_from_eigs(
    eigvals,
    eigvecs,
    *,
    spectral_family="power",
    weight_power=0.5,
    diffusion_time=1.0,
):
    """Apply spectral-distance weights to graph Laplacian eigenvectors."""
    eigvals = np.maximum(eigvals, 1e-8)
    if spectral_family == "power":
        weights = 1.0 / eigvals ** weight_power
    elif spectral_family == "diffusion":
        weights = np.exp(-float(diffusion_time) * eigvals)
    else:
        raise ValueError(
            f"Unknown spectral_family={spectral_family!r}; "
            "expected 'power' or 'diffusion'."
        )
    return eigvecs * weights[None, :]


def compute_biharmonic_cost_matrix(
    Y0,
    Y1,
    knn=15,
    n_eig=50,
    weight_power=0.5,
    spectral_family="power",
    diffusion_time=1.0,
    augmentation_eps=0.0,
    augmentation_sigma_scale=2.0,
):
    """
    OT cost matrix using spectral distance on a kNN graph built from the union
    of Y0 and Y1. Power-family distance is sum of
    (u_k(i) - u_k(j))^2 / lambda_k^(2*weight_power). Diffusion-family distance
    uses exp(-2 * diffusion_time * lambda_k).

    weight_power controls how aggressively low-frequency eigenvectors are
    amplified:
      - weight_power=1 -> true biharmonic (1/lambda^2), can collapse to 1-2 eigs
      - weight_power=0.5 -> 1/lambda weighting (diffusion distance-like)
      - weight_power=0 -> uniform weighting over the top-k eigenvectors
    Default 0.5 gives a stable balance between global structure and noise.

    This is a smoothed, global version of graph shortest-path distance --
    more robust than raw Dijkstra and captures manifold structure via the
    spectrum of the Laplacian.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    Y0_np = Y0.detach().cpu().numpy()
    Y1_np = Y1.detach().cpu().numpy()
    n0, n1 = len(Y0_np), len(Y1_np)
    N = n0 + n1
    combined = np.vstack([Y0_np, Y1_np])

    # kNN graph with Gaussian weights. Metric controlled by SMFM_KNN_METRIC
    # env (default 'cosine'; set to 'euclidean' for log1p HVG ambient data).
    metric = _knn_metric()
    nn = NearestNeighbors(n_neighbors=knn + 1, metric=metric)
    nn.fit(combined)
    dists, inds = nn.kneighbors(combined)
    _, sigma, weights = _knn_distances_to_weights(dists, metric)

    rows = np.repeat(np.arange(N), knn + 1)
    cols = inds.reshape(-1)
    data = weights.reshape(-1)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W.maximum(W.T)  # symmetrize

    # Optional dense Gaussian-Euclidean augmentation: adds a low-weight global
    # graph on top of the kNN edges so the union graph becomes connected even
    # when the kNN structure fragments. The Gaussian falloff preserves the
    # local-neighborhood bias of the kNN cost; only cross-component pairs
    # inherit a non-trivial weight that breaks the indicator-eigenvector
    # degeneracy diagnosed in the disconnected case.
    if augmentation_eps and augmentation_eps > 0.0:
        # Pairwise squared Euclidean on the original (un-normalized) inputs;
        # using `combined` keeps the augmentation graph in the same coordinate
        # system the kNN was built in.
        from scipy.spatial.distance import cdist as _cdist
        d2_full = _cdist(combined, combined, metric="sqeuclidean").astype(np.float32, copy=False)
        median_d = float(np.sqrt(np.median(d2_full[d2_full > 0]))) if (d2_full > 0).any() else 1.0
        sigma_aug = max(augmentation_sigma_scale * median_d, 1e-6)
        W_aug = augmentation_eps * np.exp(-d2_full / (2.0 * sigma_aug ** 2))
        np.fill_diagonal(W_aug, 0.0)
        # Add to existing kNN weights and re-symmetrize.
        W = csr_matrix(W.toarray() + W_aug)
        W = W.maximum(W.T)

    # Normalized Laplacian
    d = np.asarray(W.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = diags(d_inv_sqrt)
    L = diags(np.ones(N)) - D_inv_sqrt @ W @ D_inv_sqrt

    # Smallest n_eig+1 eigenpairs (skip the first trivial 0 eigenvalue)
    k_eig = min(n_eig + 1, N - 2)
    eigvals, eigvecs = eigsh(L, k=k_eig, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order][1:]  # skip trivial
    eigvecs = eigvecs[:, order][:, 1:]

    # Spectral embedding with configurable weighting.
    biharm_emb = _spectral_embedding_from_eigs(
        eigvals,
        eigvecs,
        spectral_family=spectral_family,
        weight_power=weight_power,
        diffusion_time=diffusion_time,
    )

    e0 = biharm_emb[:n0]
    e1 = biharm_emb[n0:]

    # Avoid materializing the full (n0, n1, n_eig) broadcasted tensor.
    from scipy.spatial.distance import cdist

    return cdist(e0, e1, metric="sqeuclidean").astype(np.float32, copy=False)


def make_spectral_cost_fn(
    knn=15,
    n_eig=50,
    spectral_family="power",
    weight_power=0.5,
    diffusion_time=1.0,
    augmentation_eps=0.0,
    augmentation_sigma_scale=2.0,
):
    """Create a named pairwise spectral cost function with bound parameters."""

    def spectral_cost(Y0, Y1):
        return compute_biharmonic_cost_matrix(
            Y0,
            Y1,
            knn=knn,
            n_eig=n_eig,
            spectral_family=spectral_family,
            weight_power=weight_power,
            diffusion_time=diffusion_time,
            augmentation_eps=augmentation_eps,
            augmentation_sigma_scale=augmentation_sigma_scale,
        )

    aug_tag = f"_aug{augmentation_eps:g}" if augmentation_eps else ""
    if spectral_family == "power":
        spectral_cost.__name__ = f"spectral_power_{weight_power:g}_cost{aug_tag}"
    else:
        spectral_cost.__name__ = f"spectral_diffusion_t{diffusion_time:g}_cost{aug_tag}"
    return spectral_cost


def make_spectral_plus_euclidean_cost_fn(
    blend=0.5,
    knn=15,
    n_eig=50,
    spectral_family="power",
    weight_power=0.5,
    diffusion_time=1.0,
    augmentation_eps=0.0,
    augmentation_sigma_scale=2.0,
):
    """
    Convex blend of spectral and Euclidean OT cost matrices, normalized by
    each cost's mean before mixing so `blend` is interpretable across scales.

    blend in [0, 1]:
      blend=0 -> pure spectral cost (matches make_spectral_cost_fn).
      blend=1 -> pure Euclidean (squared L2) cost.
      0<blend<1 -> (1-blend) * C_spectral_norm + blend * C_euclidean_norm.

    Motivation: when the kNN union graph is disconnected (Belkin-Niyogi
    consistency precondition fails), the bottom Laplacian eigenvectors
    degenerate to component-indicator functions and the spectral cost
    becomes uninformative within a component. Adding a small Euclidean
    regularization (blend > 0) recovers a meaningful within-component
    pairing structure without giving up the manifold-aware between-
    component penalty entirely.

    Used to test on the GoM vortex dataset where the kNN union graph
    has ~6 connected components (see scripts/connectivity_diagnostic.py).
    """
    spec_fn = make_spectral_cost_fn(
        knn=knn, n_eig=n_eig, spectral_family=spectral_family,
        weight_power=weight_power, diffusion_time=diffusion_time,
        augmentation_eps=augmentation_eps,
        augmentation_sigma_scale=augmentation_sigma_scale,
    )

    def combined_cost(Y0, Y1):
        c_spec = spec_fn(Y0, Y1)
        c_eucl = compute_euclidean_cost_matrix(Y0, Y1)
        # Mean-normalize each cost so the blend is unit-agnostic.
        spec_scale = max(float(np.mean(c_spec)), 1e-12)
        eucl_scale = max(float(np.mean(c_eucl)), 1e-12)
        c_spec_n = (c_spec / spec_scale).astype(np.float32)
        c_eucl_n = (c_eucl / eucl_scale).astype(np.float32)
        return ((1.0 - blend) * c_spec_n + blend * c_eucl_n).astype(np.float32)

    if spectral_family == "power":
        spec_label = f"spectral_power_{weight_power:g}"
    else:
        spec_label = f"spectral_diffusion_t{diffusion_time:g}"
    combined_cost.__name__ = f"{spec_label}+euclidean_blend{blend:g}_cost"
    return combined_cost


def make_biharmonic_cost_fn(
    knn=15,
    n_eig=50,
):
    """Create a cost function that always means true biharmonic distance."""

    cost_fn = make_spectral_cost_fn(
        knn=knn,
        n_eig=n_eig,
        spectral_family="power",
        weight_power=1.0,
        diffusion_time=1.0,
    )
    cost_fn.__name__ = "biharmonic_cost"
    return cost_fn


def compute_global_biharmonic_embedding(
    stage_cells,
    knn=15,
    n_eig=50,
    weight_power=0.5,
    spectral_family="power",
    diffusion_time=1.0,
):
    """Compute a single spectral embedding over ALL stages.

    Builds one kNN graph on the union of all stage cells, computes the
    normalized Laplacian, and returns the weighted spectral embedding.
    All per-hop OT couplings then use distances in this shared coordinate
    system, guaranteeing genuine global coherence across intervals.

    Returns (embeddings_per_stage, stage_sizes) where embeddings_per_stage[i]
    is the (N_i, n_eig) embedding for stage i.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    all_np = [Y.detach().cpu().numpy() for Y in stage_cells]
    sizes = [len(Y) for Y in all_np]
    combined = np.vstack(all_np)
    N = len(combined)

    metric = _knn_metric()
    nn = NearestNeighbors(n_neighbors=knn + 1, metric=metric)
    nn.fit(combined)
    dists, inds = nn.kneighbors(combined)
    _, sigma, weights = _knn_distances_to_weights(dists, metric)

    rows = np.repeat(np.arange(N), knn + 1)
    cols = inds.reshape(-1)
    data = weights.reshape(-1)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W.maximum(W.T)
    d = np.asarray(W.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = diags(d_inv_sqrt)
    L = diags(np.ones(N)) - D_inv_sqrt @ W @ D_inv_sqrt

    k_eig = min(n_eig + 1, N - 2)
    eigvals, eigvecs = eigsh(L, k=k_eig, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order][1:]
    eigvecs = eigvecs[:, order][:, 1:]

    biharm_emb = _spectral_embedding_from_eigs(
        eigvals,
        eigvecs,
        spectral_family=spectral_family,
        weight_power=weight_power,
        diffusion_time=diffusion_time,
    )

    embeddings = []
    offset = 0
    for sz in sizes:
        embeddings.append(biharm_emb[offset:offset + sz])
        offset += sz
    return embeddings


def make_global_biharmonic_cost_fn(stage_cells):
    """Return a cost_fn(Y0, Y1) that uses a precomputed global embedding.

    The returned function matches the signature of compute_biharmonic_cost_matrix
    but looks up rows from the precomputed global spectral embedding instead
    of building a per-pair graph. When called with subsampled tensors (as
    train_multi_marginal_flow does internally), each cell is matched to its
    nearest neighbor in the full embedding via cosine distance.
    """
    print("  Computing global biharmonic embedding across all stages...")
    embeddings = compute_global_biharmonic_embedding(stage_cells)

    all_cells_np = np.vstack([Y.detach().cpu().numpy() for Y in stage_cells])
    all_emb = np.vstack(embeddings)

    from sklearn.neighbors import NearestNeighbors
    nn_index = NearestNeighbors(n_neighbors=1, metric=_knn_metric())
    nn_index.fit(all_cells_np)

    def global_biharmonic_cost(Y0, Y1):
        Y0_np = Y0.detach().cpu().numpy()
        Y1_np = Y1.detach().cpu().numpy()
        idx0 = nn_index.kneighbors(Y0_np, return_distance=False).ravel()
        idx1 = nn_index.kneighbors(Y1_np, return_distance=False).ravel()
        e0 = all_emb[idx0]
        e1 = all_emb[idx1]
        diff = e0[:, None, :] - e1[None, :, :]
        return (diff ** 2).sum(axis=-1).astype(np.float32)

    global_biharmonic_cost.__name__ = "global_biharmonic_cost"
    return global_biharmonic_cost


# ────────────────────────────────────────────────────────────────────────
# Global spectral OT cost (one Laplacian over the union of ALL stages)
# ────────────────────────────────────────────────────────────────────────

# Module-level cache keyed by (id(stage_cells), knn, n_eig). Lets us re-use
# the eigendecomposition across an alpha x blend grid sweep instead of
# recomputing for every method-with-different-weight_power.
_GLOBAL_SPECTRAL_CACHE: dict = {}


def compute_global_spectral_eigs(stage_cells, knn=15, n_eig=50):
    """Compute raw (eigvals, eigvecs, sizes) of the global Laplacian.

    Builds one cosine-kNN graph on the union of all stage cells and
    returns the bottom n_eig non-trivial Laplacian eigenpairs without
    applying any weighting. Cached by (id(stage_cells), knn, n_eig) so
    parameter sweeps over weight_power / diffusion_time / blend share
    the eigendecomposition.
    """
    cache_key = (id(stage_cells), knn, n_eig)
    if cache_key in _GLOBAL_SPECTRAL_CACHE:
        return _GLOBAL_SPECTRAL_CACHE[cache_key]

    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    all_np = [
        Y.detach().cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        for Y in stage_cells
    ]
    sizes = [len(Y) for Y in all_np]
    combined = np.vstack(all_np)
    N = len(combined)

    metric = _knn_metric()
    nn = NearestNeighbors(n_neighbors=knn + 1, metric=metric)
    nn.fit(combined)
    dists, inds = nn.kneighbors(combined)
    _, sigma, weights = _knn_distances_to_weights(dists, metric)

    rows = np.repeat(np.arange(N), knn + 1)
    cols = inds.reshape(-1)
    data = weights.reshape(-1)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W.maximum(W.T)
    d = np.asarray(W.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = diags(d_inv_sqrt)
    L = diags(np.ones(N)) - D_inv_sqrt @ W @ D_inv_sqrt

    k_eig = min(n_eig + 1, N - 2)
    eigvals, eigvecs = eigsh(L, k=k_eig, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order][1:]
    eigvecs = eigvecs[:, order][:, 1:]

    _GLOBAL_SPECTRAL_CACHE[cache_key] = (eigvals, eigvecs, sizes, all_np)
    return _GLOBAL_SPECTRAL_CACHE[cache_key]


def make_global_spectral_cost_fn(
    stage_cells,
    *,
    knn=15,
    n_eig=50,
    spectral_family="power",
    weight_power=0.5,
    diffusion_time=1.0,
    blend=0.0,
):
    """Return a cost_fn(Y0, Y1) that uses the GLOBAL Laplacian eigenbasis.

    Same shape as ``make_spectral_cost_fn`` (per-pair) but the Laplacian
    is built on the union of ALL training stages, so every adjacent-pair
    OT solve sees the same eigenvectors (same coordinate system).
    Theoretically aligns with the Belkin-Niyogi consistency precondition,
    which is about the union graph -- not the 2-stage subgraph.

    Parameters mirror ``make_spectral_cost_fn``; ``blend`` adds a
    mean-normalised Euclidean cost on top, exactly like
    ``make_spectral_plus_euclidean_cost_fn`` does for the per-pair variant.
    """
    eigvals, eigvecs, sizes, all_cells_np = compute_global_spectral_eigs(
        stage_cells, knn=knn, n_eig=n_eig,
    )
    biharm_emb = _spectral_embedding_from_eigs(
        eigvals, eigvecs,
        spectral_family=spectral_family,
        weight_power=weight_power,
        diffusion_time=diffusion_time,
    )

    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist

    all_combined = np.vstack(all_cells_np)
    nn_index = NearestNeighbors(n_neighbors=1, metric=_knn_metric())
    nn_index.fit(all_combined)

    def global_spectral_cost(Y0, Y1):
        Y0_np = Y0.detach().cpu().numpy() if isinstance(Y0, torch.Tensor) else np.asarray(Y0)
        Y1_np = Y1.detach().cpu().numpy() if isinstance(Y1, torch.Tensor) else np.asarray(Y1)
        idx0 = nn_index.kneighbors(Y0_np, return_distance=False).ravel()
        idx1 = nn_index.kneighbors(Y1_np, return_distance=False).ravel()
        e0 = biharm_emb[idx0]
        e1 = biharm_emb[idx1]
        c_spec = cdist(e0, e1, metric="sqeuclidean").astype(np.float32, copy=False)
        if blend and blend > 0.0:
            c_eucl = cdist(Y0_np, Y1_np, metric="sqeuclidean").astype(np.float32, copy=False)
            spec_scale = max(float(np.mean(c_spec)), 1e-12)
            eucl_scale = max(float(np.mean(c_eucl)), 1e-12)
            return (
                (1.0 - blend) * (c_spec / spec_scale)
                + blend * (c_eucl / eucl_scale)
            ).astype(np.float32, copy=False)
        return c_spec

    blend_tag = f"_blend{blend:g}" if blend and blend > 0 else ""
    if spectral_family == "power":
        global_spectral_cost.__name__ = (
            f"global_spectral_power_{weight_power:g}{blend_tag}_cost"
        )
    else:
        global_spectral_cost.__name__ = (
            f"global_spectral_diffusion_t{diffusion_time:g}{blend_tag}_cost"
        )
    return global_spectral_cost


def _pairwise_sqeuclidean(X0, X1, chunk_size=512):
    """Squared Euclidean distance without materializing a 3D broadcast."""
    X0 = np.asarray(X0, dtype=np.float32, order="C")
    X1 = np.asarray(X1, dtype=np.float32, order="C")

    cost = np.empty((len(X0), len(X1)), dtype=np.float32)
    x1_norm = np.einsum("ij,ij->i", X1, X1, dtype=np.float32)
    for start in range(0, len(X0), chunk_size):
        stop = min(start + chunk_size, len(X0))
        block = X0[start:stop]
        block_norm = np.einsum("ij,ij->i", block, block, dtype=np.float32)
        block_cost = block_norm[:, None] + x1_norm[None, :] - 2.0 * (block @ X1.T)
        np.maximum(block_cost, 0.0, out=block_cost)
        cost[start:stop] = block_cost
    return cost


def _sphere_arc_distance_matrix(X, chunk_size=512):
    """All-pairs great-circle distances for row-normalized sphere points."""
    X = np.asarray(X, dtype=np.float32, order="C")
    dist = np.empty((len(X), len(X)), dtype=np.float32)
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        block = X[start:stop] @ X.T
        np.clip(block, -1.0, 1.0, out=block)
        np.arccos(block, out=block)
        dist[start:stop] = block
    np.fill_diagonal(dist, 0.0)
    return dist


def _phate_potential_coordinates(ph, n_samples):
    """Return one PHATE potential row per original sample.

    PHATE computes the diffusion potential before MDS. For large inputs it
    uses a LandmarkGraph, where the potential is landmark-by-landmark; in that
    case PHATE's own interpolation gives each original sample coordinates in
    the landmark potential basis.
    """
    landmark_potential = np.asarray(ph._calculate_potential(), dtype=np.float32)
    if landmark_potential.shape[0] == n_samples:
        return landmark_potential

    graph = ph.graph
    if not hasattr(graph, "interpolate"):
        raise RuntimeError(
            "PHATE returned a landmark potential but the graph cannot "
            "interpolate it back to samples."
        )
    return np.asarray(graph.interpolate(landmark_potential), dtype=np.float32)


def compute_phate_cost_matrix(
    Y0,
    Y1,
    n_components=10,
    knn=15,
    decay=40,
    n_landmark=2000,
    graph_metric="euclidean",
):
    """
    OT cost matrix using squared PHATE potential distance.

    PHATE first builds a diffusion operator on an adaptive kNN graph and maps
    each point to its diffusion potential. The PHATE embedding is obtained only
    afterwards by MDS. This cost intentionally uses distances between the
    pre-MDS potential rows, not Euclidean distances in a low-dimensional PHATE
    embedding.

    graph_metric controls the geometry used to build PHATE's kNN graph:
    "euclidean" uses PHATE's default input-space distance, while "sphere_arc"
    passes a precomputed great-circle distance matrix. For large OT subsamples,
    PHATE uses landmarks. We keep that approximation and interpolate the
    landmark potential rows back to the original samples before computing
    pairwise costs.
    """
    import phate

    Y0_np = Y0.detach().cpu().numpy()
    Y1_np = Y1.detach().cpu().numpy()
    n0, n1 = len(Y0_np), len(Y1_np)
    combined = np.vstack([Y0_np, Y1_np])

    if graph_metric == "euclidean":
        phate_input = combined
        knn_dist = "euclidean"
    elif graph_metric == "sphere_arc":
        phate_input = _sphere_arc_distance_matrix(combined)
        knn_dist = "precomputed_distance"
    else:
        raise ValueError(
            f"Unknown PHATE graph_metric={graph_metric!r}; "
            "expected 'euclidean' or 'sphere_arc'."
        )

    ph = phate.PHATE(n_components=n_components, knn=knn, decay=decay,
                     n_landmark=n_landmark, t="auto", knn_dist=knn_dist,
                     verbose=0, random_state=42, n_jobs=1)
    ph.fit(phate_input)
    potential = _phate_potential_coordinates(ph, n_samples=n0 + n1)
    e0 = potential[:n0]
    e1 = potential[n0:]
    return _pairwise_sqeuclidean(e0, e1)


def make_phate_cost_fn(
    n_components=10,
    knn=15,
    decay=40,
    n_landmark=2000,
    graph_metric="euclidean",
):
    """Create a named PHATE cost function with bound Hydra parameters."""

    def phate_cost(Y0, Y1):
        return compute_phate_cost_matrix(
            Y0,
            Y1,
            n_components=n_components,
            knn=knn,
            decay=decay,
            n_landmark=n_landmark,
            graph_metric=graph_metric,
        )

    phate_cost.__name__ = f"phate_potential_{graph_metric}"
    return phate_cost


def compute_euclidean_cost_matrix(Y0, Y1):
    """Squared Euclidean cost matrix between two Euclidean point clouds.

    Uses SciPy's pairwise-distance kernel rather than materializing the full
    broadcasted (n0, n1, D) difference tensor, which becomes prohibitively
    large for 1500-HVG OT subsamples.
    """
    from scipy.spatial.distance import cdist

    Y0_np = Y0.detach().cpu().numpy()
    Y1_np = Y1.detach().cpu().numpy()
    return cdist(Y0_np, Y1_np, metric="sqeuclidean").astype(np.float32, copy=False)


_RANDOM_COST_RNG = np.random.default_rng(0xCAFEC0DE)


def compute_random_cost_matrix(Y0, Y1):
    """Uniform-random cost matrix; the resulting OT plan is essentially
    a random matching that respects marginals.

    Use as a 'do-nothing' baseline alongside Euclidean / sphere-W2 / spectral
    OT costs. The plan changes batch-to-batch so the trainer sees noisy
    pseudo-pairs, which is the relevant negative control: any method that
    fails to beat this is not exploiting cross-marginal structure at all.

    Uses a module-private Generator instance (not the global numpy RNG) so
    that running this method does not perturb the seed sequence visible to
    other methods sharing the same training-loop process.
    """
    n0, n1 = len(Y0), len(Y1)
    return _RANDOM_COST_RNG.random((n0, n1)).astype(np.float32, copy=False)
