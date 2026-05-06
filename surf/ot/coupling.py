"""
Optimal transport coupling: solve OT and sample coupled pairs.
"""

import numpy as np


def ot_coupling(cost_matrix, n_samples, emd_max_pool=15000):
    """
    Solve OT problem and sample coupled pairs.
    Returns indices (src_idx, tgt_idx) of coupled pairs.

    Uses exact EMD when max(n, m) <= emd_max_pool (sharp permutation plan,
    best training signal), falls back to Sinkhorn above that threshold for
    tractability. Historically this always used Sinkhorn, which introduces
    entropic smoothing into the coupling -- fine at high dim/N where EMD is
    intractable, but degrades training signal when EMD would have worked.
    """
    try:
        import ot
        HAS_POT = True
    except ImportError:
        HAS_POT = False

    n, m = cost_matrix.shape
    if not HAS_POT:
        # Uniform random pairing fallback
        src = np.random.randint(0, n, n_samples)
        tgt = np.random.randint(0, m, n_samples)
        return src, tgt

    a = np.ones(n) / n
    b = np.ones(m) / m

    if max(n, m) <= emd_max_pool:
        # Exact EMD: sharp permutation plan
        T = ot.emd(a, b, cost_matrix)
    else:
        # Stabilized Sinkhorn at small eps: near-EMD sharpness, log-domain
        # stable, much better at high K than regular Sinkhorn with large eps.
        eps = 0.005 * cost_matrix.max()
        T = ot.sinkhorn(
            a, b, cost_matrix, reg=eps,
            method="sinkhorn_stabilized",
            numItermax=2000, stopThr=1e-7,
        )
    T = T / T.sum()  # normalize to joint distribution

    # Sample from the transport plan
    T_flat = T.flatten()
    T_flat = np.maximum(T_flat, 0)
    T_flat = T_flat / T_flat.sum()
    flat_idx = np.random.choice(len(T_flat), size=n_samples, p=T_flat)
    src_idx = flat_idx // m
    tgt_idx = flat_idx % m
    return src_idx, tgt_idx
