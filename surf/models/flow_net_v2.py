"""FlowNetV2: faithful re-implementation of OTP-FM's ``FlowNetMLP`` architecture.

This network replicates the architecture used by Atanackovic et al. 2024
(OTP-FM) and Rohbeck et al. 2025 (MMFM) for tabular cell-level flow
matching:

  - Linear x-embedding (D -> x_emb_dim).
  - Sinusoidal positional t-embedding (concatenated cos/sin), then
    Linear projection to 2 * t_emb_dim.
  - Main MLP: ``num_hidden_layers`` blocks of [LayerNorm -> SiLU -> Linear],
    with a residual skip every ``residual_every`` layers. Pre-activation
    style.
  - Output projection: LayerNorm -> SiLU -> Linear -> D-dim velocity.

Defaults match ``OTP-FM/configs/gom/defaults.json`` so MMFM trained on
this network is directly comparable to OTP-FM Table 2's reported numbers.

The ``forward(x, t)`` signature matches ``surf.models.FlowNet`` so it can
be swapped in transparently anywhere FlowNet is used. We do NOT take a
``dt`` arg (that's an OTP-FM mean-flow concept that doesn't apply to
plain CFM / MMFM training).
"""

import torch
import torch.nn as nn


class _PositionalEmbedding(nn.Module):
    """Sinusoidal timestep embedding (DDPM++/EDM/MeanFlow style)."""

    def __init__(self, num_channels, max_positions=10000, endpoint=True):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2,
            dtype=torch.float32, device=x.device,
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.view(-1).outer(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class _ResidualMLP(nn.Module):
    """Pre-activation MLP with optional residual connections every N layers."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers,
                 layernorm=True, residual_every=2):
        super().__init__()
        self.residual_every = residual_every
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            mods = []
            if layernorm:
                mods.append(nn.LayerNorm(hidden_dim))
            mods.append(nn.SiLU())
            mods.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.Sequential(*mods))
        out_mods = []
        if layernorm:
            out_mods.append(nn.LayerNorm(hidden_dim))
        out_mods.append(nn.SiLU())
        out_mods.append(nn.Linear(hidden_dim, output_dim))
        self.output_proj = nn.Sequential(*out_mods)

    def forward(self, x):
        h = self.input_proj(x)
        if self.residual_every > 0:
            h_res = h
            for i, layer in enumerate(self.hidden_layers):
                h = layer(h)
                if (i + 1) % self.residual_every == 0:
                    h = h + h_res
                    h_res = h
        else:
            for layer in self.hidden_layers:
                h = layer(h)
        return self.output_proj(h)


class FlowNetV2(nn.Module):
    """Velocity predictor matching OTP-FM's ``FlowNetMLP`` for tabular data.

    Defaults match ``OTP-FM/configs/gom/defaults.json``.

    Parameters
    ----------
    D : int
        Data dimension.
    hidden_dim : int, default 128
        Width of the main MLP.
    num_hidden_layers : int, default 10
        Depth of the main MLP.
    x_emb_dim : int, default 64
        Linear x-embedding output dim.
    t_emb_dim : int, default 64
        Sinusoidal t-embedding base dim (output is 2 * t_emb_dim).
    residual_every : int, default 2
        Residual connection every N hidden layers.
    layernorm : bool, default True
        Pre-activation LayerNorm in each hidden block.
    """

    def __init__(self, D, hidden_dim=128, num_hidden_layers=10,
                 x_emb_dim=64, t_emb_dim=64, residual_every=2, layernorm=True):
        super().__init__()
        self.D = D
        self.x_emb = nn.Linear(D, x_emb_dim)
        self.t_pos_emb = _PositionalEmbedding(t_emb_dim)
        self.t_emb = nn.Linear(t_emb_dim, 2 * t_emb_dim)
        self.v = _ResidualMLP(
            input_dim=x_emb_dim + 2 * t_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=D,
            num_hidden_layers=num_hidden_layers,
            layernorm=layernorm,
            residual_every=residual_every,
        )

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.dim() > 1:
            t = t.view(-1)
        x_e = self.x_emb(x)
        t_e = self.t_emb(self.t_pos_emb(t))
        return self.v(torch.cat([x_e, t_e], dim=-1))


def build_flow_net(D, arch="v1"):
    """Factory: ``arch="v1"`` -> :class:`FlowNet` (4x256 plain MLP),
    ``arch="v2"`` -> :class:`FlowNetV2` (OTP-FM 10x128 with residuals)."""
    if arch == "v1":
        from surf.models.flow_net import FlowNet
        return FlowNet(D)
    if arch == "v2":
        return FlowNetV2(D)
    raise ValueError(f"Unknown flow-net arch {arch!r}; choose 'v1' or 'v2'")
