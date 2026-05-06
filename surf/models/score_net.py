"""
Score networks for denoising score matching on the sphere.

Contains:
- SigmaEmbedding: Random Fourier features for noise level conditioning
- RiemannianScoreNet: Parametric score estimator (z, log_sigma) -> tangent
- TimeEmbedding: Random Fourier features for trajectory time t
- TimedRiemannianScoreNet: Time-conditioned score (z, log_sigma, t) -> tangent
"""

import torch
import torch.nn as nn
import numpy as np


class SigmaEmbedding(nn.Module):
    """
    Random Fourier features for noise level conditioning, a la DDPM/EDM.

    Replaces raw log_sigma scalar input with a rich sinusoidal embedding.
    Frequencies are log-spaced over the range typical for log_sigma values
    in [-4, 0] (i.e., sigma in [0.018, 1.0]).

    Input: log_sigma, shape (B,) or (B, 1)
    Output: (B, emb_dim) embedding vector
    """

    def __init__(self, emb_dim=64, num_freqs=None):
        super().__init__()
        num_freqs = num_freqs or (emb_dim // 2)
        # Log-spaced frequencies, roughly [0.5, 50] rad/unit
        freqs = 2 * np.pi * torch.exp(
            torch.linspace(float(np.log(0.1)), float(np.log(50)), num_freqs)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * num_freqs, emb_dim)

    def forward(self, log_sigma):
        if log_sigma.dim() == 0:
            log_sigma = log_sigma.view(1)
        elif log_sigma.dim() == 2:
            log_sigma = log_sigma.squeeze(-1)
        # log_sigma: (B,)
        s = log_sigma.unsqueeze(-1) * self.freqs.view(1, -1)  # (B, num_freqs)
        feats = torch.cat([s.sin(), s.cos()], dim=-1)  # (B, 2*num_freqs)
        return self.proj(feats)  # (B, emb_dim)


class RiemannianScoreNet(nn.Module):
    """
    Parametric score estimator on the sphere. Input: (z, log_sigma). Output:
    tangent vector at z approximating nabla log p_sigma(z), where p_sigma is the noise-
    convolved density of the training cell cloud at scale sigma.

    Improvements (vs the original Song & Ermon NCSN-style MLP):
      - SigmaEmbedding (Fourier features) instead of raw log_sigma scalar
      - Wider multi-scale range [sigma=0.02, sigma=1.0]
      - Compatible with lognormal sigma sampling (EDM schedule)

    Tangent projection is applied inside forward() so the output is always
    a valid tangent vector at z.
    """

    def __init__(self, D, hidden=256, depth=4, sigma_emb_dim=64):
        super().__init__()
        self.sigma_emb = SigmaEmbedding(emb_dim=sigma_emb_dim)
        layers = [nn.Linear(D + sigma_emb_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)
        self.D = D

    def forward(self, z, log_sigma):
        if log_sigma.dim() == 0:
            log_sigma = log_sigma.expand(z.shape[0])
        emb = self.sigma_emb(log_sigma)  # (B, emb_dim)
        x = torch.cat([z, emb], dim=-1)
        raw = self.net(x)
        # Tangent projection at z
        return raw - (raw * z).sum(dim=-1, keepdim=True) * z


class TimeEmbedding(nn.Module):
    """
    Random Fourier features for trajectory time t in [0, 1]. Mirror of
    SigmaEmbedding but with a lower frequency range appropriate for unit
    intervals: log-spaced from 0.5 to 16 cycles per unit. The lowest
    harmonic resolves the 0.25-wide developmental stage intervals.

    Input: t, shape (B,), (B, 1), or scalar. Output: (B, emb_dim).
    """

    def __init__(self, emb_dim=64, num_freqs=None):
        super().__init__()
        num_freqs = num_freqs or (emb_dim // 2)
        freqs = 2 * np.pi * torch.exp(
            torch.linspace(float(np.log(0.5)), float(np.log(16.0)), num_freqs)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * num_freqs, emb_dim)

    def forward(self, t):
        if t.dim() == 0:
            t = t.view(1)
        elif t.dim() == 2:
            t = t.squeeze(-1)
        # t: (B,)
        s = t.unsqueeze(-1) * self.freqs.view(1, -1)  # (B, num_freqs)
        feats = torch.cat([s.sin(), s.cos()], dim=-1)  # (B, 2*num_freqs)
        return self.proj(feats)  # (B, emb_dim)


class TimedRiemannianScoreNet(nn.Module):
    """
    Time-conditioned parametric score on the sphere.

    forward(z, log_sigma, t) -> tangent at z approximating
        nabla log p_sigma(z | developmental stage = t)

    Addresses the core limitation of RiemannianScoreNet: the unconditional
    score pulls toward the cell cloud centroid (marginal density gradient)
    regardless of which developmental stage the flow should be at. The
    time-conditioned variant learns stage-specific densities so the score
    points toward the correct target stage during multi-marginal flow
    training.

    Same tangent projection and MLP architecture as RiemannianScoreNet,
    but with an additional TimeEmbedding concatenated into the input.
    """

    def __init__(self, D, hidden=256, depth=4, sigma_emb_dim=64, time_emb_dim=64):
        super().__init__()
        self.sigma_emb = SigmaEmbedding(emb_dim=sigma_emb_dim)
        self.time_emb = TimeEmbedding(emb_dim=time_emb_dim)
        in_dim = D + sigma_emb_dim + time_emb_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)
        self.D = D

    def forward(self, z, log_sigma, t):
        if log_sigma.dim() == 0:
            log_sigma = log_sigma.expand(z.shape[0])
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        elif t.dim() == 2:
            t = t.squeeze(-1)
        emb_s = self.sigma_emb(log_sigma)  # (B, sigma_emb_dim)
        emb_t = self.time_emb(t)  # (B, time_emb_dim)
        x = torch.cat([z, emb_s, emb_t], dim=-1)
        raw = self.net(x)
        return raw - (raw * z).sum(dim=-1, keepdim=True) * z
