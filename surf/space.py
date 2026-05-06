"""Configurable Euclidean training spaces for flow models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _like_tensor(arr, ref):
    if isinstance(ref, torch.Tensor):
        return torch.from_numpy(np.asarray(arr)).to(device=ref.device, dtype=ref.dtype)
    return np.asarray(arr)


@dataclass
class IdentitySpace:
    """No-op Euclidean space corresponding to raw log1p coordinates."""

    input_dim: int

    @property
    def name(self):
        return "log1p"

    @property
    def output_dim(self):
        return self.input_dim

    def encode(self, x):
        return x

    def decode(self, z):
        return z

    def summary(self):
        return f"log1p ambient ({self.output_dim}D)"


class PCASpace:
    """Linear latent space backed by a reversible PCA transform."""

    def __init__(self, pca: PCA):
        self.pca = pca

    @property
    def name(self):
        return "pca"

    @property
    def input_dim(self):
        return int(self.pca.n_features_in_)

    @property
    def output_dim(self):
        return int(self.pca.n_components_)

    @classmethod
    def fit(cls, train_cells, *, n_components=10, random_state=42):
        all_cells = np.concatenate([_as_numpy(x) for x in train_cells], axis=0)
        max_rank = int(min(all_cells.shape[0], all_cells.shape[1]))
        if n_components > max_rank:
            raise ValueError(
                f"PCA n_components={n_components} exceeds max rank {max_rank} "
                f"for training matrix shape {all_cells.shape}."
            )
        pca = PCA(
            n_components=int(n_components),
            svd_solver="randomized",
            random_state=random_state,
        )
        pca.fit(all_cells)
        return cls(pca)

    def encode(self, x):
        emb = self.pca.transform(_as_numpy(x))
        return _like_tensor(emb, x)

    def decode(self, z):
        rec = self.pca.inverse_transform(_as_numpy(z))
        return _like_tensor(rec, z)

    def summary(self):
        var = float(self.pca.explained_variance_ratio_.sum())
        return f"PCA latent ({self.output_dim}D, explained_var={var:.3f})"


@dataclass
class SphereSpace:
    """Sphere-encoded Euclidean space: log1p -> compositional -> sqrt(p) on the
    positive orthant of S^{D-1}, but treated as ordinary Euclidean coordinates
    by the Euclidean flow trainer. This isolates the *interpolation* geometry
    (straight-line vs SLERP) while keeping the *training point cloud* identical
    to what MM+SLERP sees -- a controlled fairness ablation."""

    input_dim: int

    @property
    def name(self):
        return "sphere"

    @property
    def output_dim(self):
        return self.input_dim

    def encode(self, x):
        # x: log1p tensor (B, D). Reproduce the MM+SLERP preprocessing.
        from surf.geometry.sphere import normalize_sphere, to_compositional, to_orthant

        return normalize_sphere(to_orthant(to_compositional(x)))

    def decode(self, z):
        # The Euclidean trainer outputs ambient R^D vectors that we *interpret*
        # as sphere coordinates. To plug into the standard eval pipeline
        # (`to_compositional(decoded.clamp(min=0))`), we return log1p(y_+^2):
        #   expm1(log1p(y_+^2)) = y_+^2
        # which then normalizes to the simplex inside `to_compositional`.
        # Negative entries are clamped (off-orthant projection); ‖y‖₂ need not
        # equal 1, since the simplex normalization handles the rescaling.
        return torch.log1p(z.clamp(min=0.0) ** 2)

    def summary(self):
        return f"sphere (positive orthant of S^{self.output_dim - 1}, Euclidean trainer)"


def build_space(cfg, train_cells):
    """Construct the configured Euclidean state space from train log1p cells."""
    name = str(cfg.name)
    input_dim = int(train_cells[0].shape[1])
    if name == "log1p":
        return IdentitySpace(input_dim=input_dim)
    if name == "sphere":
        return SphereSpace(input_dim=input_dim)
    if name == "pca":
        return PCASpace.fit(
            train_cells,
            n_components=int(cfg.n_components),
            random_state=int(cfg.random_state),
        )
    raise ValueError(f"Unknown space.name={name!r}; expected 'log1p', 'sphere', or 'pca'.")
