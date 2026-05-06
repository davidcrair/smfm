"""
Positive-orthant / sphere geometry utilities: mapping between the simplex
and the positive orthant of the unit sphere, cost matrices, and Brownian
perturbation.
"""

import torch
import numpy as np

from surf.runtime import get as get_runtime


def to_orthant(p):
    """Map compositional data p (simplex) -> sqrt(p) (positive orthant of S^d)."""
    return torch.sqrt(p.clamp(min=1e-8))


def from_orthant(y):
    """Map positive orthant y -> y^2 (back to simplex). y is already unit-norm."""
    return y ** 2


def normalize_sphere(y):
    """Project onto unit sphere."""
    return y / y.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def sphere_log_map(x, y):
    """Riemannian log map on the unit sphere: tangent at x pointing to y."""
    cos_theta = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-8)
    v = (theta / sin_theta) * (y - cos_theta * x)
    return v - (v * x).sum(dim=-1, keepdim=True) * x


def sphere_exp_map(x, v):
    """Riemannian exp map on the unit sphere from base point x along tangent v."""
    v_norm = v.norm(dim=-1, keepdim=True)
    direction = v / v_norm.clamp(min=1e-8)
    moved = torch.cos(v_norm) * x + torch.sin(v_norm) * direction
    near_zero = v_norm < 1e-8
    moved = torch.where(near_zero, x, moved)
    return normalize_sphere(moved)


def to_compositional(X):
    """
    Convert log1p-normalized expression to a probability distribution over genes.
    X: (n, D) tensor of log1p(library-size-normalized counts).
    """
    counts = torch.expm1(X).clamp(min=0)
    counts = counts + 1e-8  # avoid exact zeros in downstream sqrt
    return counts / counts.sum(dim=-1, keepdim=True)


def compute_sphere_cost_matrix(Y0, Y1):
    """Squared geodesic cost matrix between all pairs on the positive orthant.

    Fisher Flow (Davis et al., Prop. 2) uses c(x,y) = d²(x,y) for optimal
    transport -- the W2 cost -- which gives the unique constant-speed geodesic
    interpolant. d(x,y) = arccos(<x,y>) on the unit sphere.
    Computed on whatever device the inputs live on; returned as numpy for POT.
    """
    cos_sim = (Y0 @ Y1.T).clamp(-1 + 1e-6, 1 - 1e-6)
    arc = torch.acos(cos_sim)
    return (arc ** 2).cpu().numpy()


def sphere_brownian_perturb(c, sigma, n_steps=3):
    """
    Multi-step Euler-Maruyama approximation to Brownian motion on the unit
    sphere, starting at c with total noise scale sigma. More faithful to the
    heat kernel (de Bortoli et al. 2022) than a single Gaussian-tangent
    retraction, especially at larger sigma where the single-step approximation
    biases samples toward the tangent plane.

    Each of n_steps applies a Gaussian tangent perturbation with variance
    sigma^2/n_steps and retracts via Exp. The total tangent-space variance
    accumulates to sigma^2 (approximately; manifold curvature introduces a
    small correction for large sigma).

    c: (B, D) unit vectors
    sigma: (B, 1) per-sample noise scale
    Returns z: (B, D) perturbed points on the sphere.
    """
    z = c
    # sqrt(n_steps) so that K steps of variance sigma^2/K accumulate to sigma^2
    step_sigma = sigma / (n_steps ** 0.5)
    for _ in range(n_steps):
        eps = torch.randn_like(z)
        eps = eps - (eps * z).sum(dim=-1, keepdim=True) * z  # tangent at z
        v = step_sigma * eps
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z = torch.cos(v_norm) * z + torch.sin(v_norm) * v / v_norm
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return z
