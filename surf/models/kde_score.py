"""
Non-parametric (KDE-based) Riemannian score estimation on the sphere.

Contains:
- estimate_rbf_sigma: bandwidth selection via median k-NN arc distance
- rbf_spherical_score: closed-form RBF score on the sphere
- _KDEScoreWrapper: adapter matching the RiemannianScoreNet interface
"""

import torch
import numpy as np


def estimate_rbf_sigma(cells, k=50):
    """
    Bandwidth for an RBF kernel on the sphere: median arc distance to the
    k-th nearest neighbor among the training cells. Much more robust in
    high dimension than Silverman's rule.
    """
    from sklearn.neighbors import NearestNeighbors
    A = cells.detach().cpu().numpy()
    k_eff = min(k + 1, len(A))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(A)
    dists, _ = nn.kneighbors(A)
    cos_sim = np.clip(1.0 - dists[:, 1:], -1 + 1e-6, 1 - 1e-6)
    arcs = np.arccos(cos_sim)  # (N, k)
    return float(np.median(arcs[:, -1]))


def rbf_spherical_score(z, cells, sigma):
    """
    Non-parametric Riemannian score of the data density on the sphere at
    query positions z. Uses an isotropic RBF kernel on arc-length distance:

        p(z) ~ sum_i exp(-arc(z, c_i)^2 / (2*sigma^2))
        score(z) = nabla log p(z) = (1/sigma^2) * sum_i w_i(z) * log_z(c_i)

    where w_i are softmax-normalized kernel weights and log_z is the sphere
    log map. The score is zero on the ribbon (cells surround z symmetrically)
    and points toward the ribbon when z drifts off it -- crucially, no pull
    toward the global mode because on-ribbon gradient flattens to zero.

    Factored implementation avoids an (B, N, D) intermediate tensor:
        score(z) = (1/sigma^2) * [ (w*a) @ cells  -  ((w*a*cos_w).sum) * z ]

    where a = w/sin(w) is the exact log-map scale.

    Parameters
    ----------
    z: (B, D) unit vectors on the sphere (tangent basepoints)
    cells: (N, D) training cells on the sphere
    sigma: scalar bandwidth (arc-length units)

    Returns
    -------
    score: (B, D) tangent vectors at each z
    """
    cos_sim = (z @ cells.T).clamp(-1 + 1e-6, 1 - 1e-6)  # (B, N)
    arc = torch.acos(cos_sim)  # (B, N)

    # Softmax-normalized RBF weights (numerically stable)
    log_w = -(arc ** 2) / (2 * sigma * sigma)
    w = torch.softmax(log_w, dim=-1)  # (B, N)

    # Exact log-map scale a = omega / sin(omega); limit 1 at omega = 0
    sin_arc = torch.sin(arc)
    a = torch.where(arc < 1e-6, torch.ones_like(arc), arc / sin_arc.clamp(min=1e-8))

    wa = w * a  # (B, N)
    weighted_cells = wa @ cells  # (B, D)
    scalar_term = (wa * cos_sim).sum(dim=-1, keepdim=True)  # (B, 1)
    score = (weighted_cells - scalar_term * z) / (sigma * sigma)

    # Tangent projection (numerical cleanup)
    score = score - (score * z).sum(dim=-1, keepdim=True) * z
    return score


class _KDEScoreWrapper:
    """
    Adapter exposing the same (z, log_sigma) -> score signature as
    RiemannianScoreNet, backed by the non-parametric RBF KDE. Lets the
    multi-marginal trainer switch between KDE and learned score via a single
    `score_net` argument.
    """

    def __init__(self, cells, sigma):
        self.cells = cells
        self.sigma = sigma

    def eval(self):
        return self

    def __call__(self, z, log_sigma):
        return rbf_spherical_score(z, self.cells, self.sigma)
