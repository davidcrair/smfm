"""Spectral-path helpers for Laplacian-space flow supervision."""

from __future__ import annotations

import numpy as np
import torch

from surf.geometry.sphere import normalize_sphere


class SpectralKNNDecoder:
    """Detached kNN barycentric decoder from spectral coordinates to the sphere."""

    def __init__(
        self,
        all_embeddings,
        all_cells,
        *,
        k=64,
        tau="auto",
        chunk_size=64,
        eps=1e-8,
        decode_device=None,
    ):
        if len(all_embeddings) != len(all_cells):
            raise ValueError("all_embeddings and all_cells must have the same length")
        if len(all_embeddings) == 0:
            raise ValueError("cannot build a decoder with no cells")

        source_device = all_cells.device
        if decode_device is None:
            decode_device = torch.device("cpu") if source_device.type == "mps" else source_device
        else:
            decode_device = torch.device(decode_device)

        self.output_device = source_device
        self.decode_device = decode_device
        self.k = min(int(k), len(all_embeddings))
        self.chunk_size = max(1, int(chunk_size))
        self.eps = float(eps)

        self.all_embeddings = all_embeddings.detach().to(decode_device, dtype=torch.float32)
        self.all_cells = all_cells.detach().to(decode_device, dtype=torch.float32)
        self.all_embedding_norm_sq = (self.all_embeddings * self.all_embeddings).sum(dim=-1)
        self.tau = self._resolve_tau(tau)

    def _resolve_tau(self, tau):
        if tau is None or str(tau).lower() == "auto":
            return self._estimate_tau()
        tau_float = float(tau)
        if tau_float <= 0:
            raise ValueError(f"decode tau must be positive, got {tau!r}")
        return tau_float

    def _estimate_tau(self):
        from sklearn.neighbors import NearestNeighbors

        emb = self.all_embeddings.detach().cpu().numpy()
        n = len(emb)
        if n <= 1:
            return 1.0
        n_query = min(n, 2048)
        rng = np.random.default_rng(0)
        query_idx = rng.choice(n, size=n_query, replace=False)
        n_neighbors = min(self.k + 1, n)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(emb)
        dists, _ = nn.kneighbors(emb[query_idx])
        if n_neighbors > 1:
            local_sq = dists[:, 1:] ** 2
        else:
            local_sq = dists ** 2
        positive = local_sq[local_sq > 1e-12]
        if positive.size == 0:
            return 1.0
        return float(max(np.median(positive), 1e-8))

    @torch.no_grad()
    def decode(self, z):
        """Decode a batch of spectral coordinates to positive-orthant sphere points."""
        z_decode = z.detach().to(self.decode_device, dtype=torch.float32)
        decoded_chunks = []
        for z_chunk in z_decode.split(self.chunk_size, dim=0):
            z_norm_sq = (z_chunk * z_chunk).sum(dim=-1, keepdim=True)
            dist_sq = z_norm_sq + self.all_embedding_norm_sq.unsqueeze(0) - 2.0 * (
                z_chunk @ self.all_embeddings.T
            )
            dist_sq = dist_sq.clamp(min=0.0)
            top_dist_sq, top_idx = torch.topk(
                dist_sq,
                k=self.k,
                dim=1,
                largest=False,
            )
            weights = torch.softmax(-top_dist_sq / self.tau, dim=1)
            flat_idx = top_idx.reshape(-1)
            neigh_cells = self.all_cells.index_select(0, flat_idx).reshape(
                z_chunk.shape[0],
                self.k,
                self.all_cells.shape[1],
            )
            x_tilde = (weights.unsqueeze(-1) * neigh_cells).sum(dim=1)
            decoded_chunks.append(normalize_sphere(x_tilde.clamp(min=self.eps)))
        decoded = torch.cat(decoded_chunks, dim=0)
        return decoded.to(self.output_device)

    def diagnostics(self, z, x):
        """Return simple decode sanity statistics for a decoded batch."""
        with torch.no_grad():
            return {
                "decode_tau": float(self.tau),
                "decode_k": int(self.k),
                "min_coordinate": float(x.min().detach().cpu()),
                "sphere_norm_error_mean": float(
                    (x.norm(dim=-1) - 1.0).abs().mean().detach().cpu()
                ),
                "spectral_query_norm_mean": float(z.norm(dim=-1).mean().detach().cpu()),
            }
