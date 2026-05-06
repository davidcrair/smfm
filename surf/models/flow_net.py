"""
FlowNet: MLP backbone predicting a tangent velocity vector on the sphere.
"""

import torch
import torch.nn as nn


class FlowNet(nn.Module):
    """
    MLP backbone predicting a tangent velocity vector. Input: (z_t, t)
    concatenated. Output is projected onto the tangent space at z_t inside
    the training loop.
    """

    def __init__(self, D, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(D + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)
        self.D = D

    def forward(self, zt, t):
        # t: scalar or (B,) -> (B, 1)
        if t.dim() == 0:
            t = t.expand(zt.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        x = torch.cat([zt, t], dim=-1)
        return self.net(x)
