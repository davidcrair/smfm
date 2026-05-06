"""Differentiable encoder from Fisher-sphere coordinates to spectral coordinates."""

import torch.nn as nn


class SpectralEncoderNet(nn.Module):
    """MLP encoder E_eta(x) used for spectral-space velocity supervision."""

    def __init__(self, input_dim, output_dim, hidden=256, depth=4):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(self, x):
        return self.net(x)
