"""
MeanFlowNet: endpoint/average-velocity baseline on sphere coordinates.
"""

import torch
import torch.nn as nn


class MeanFlowNet(nn.Module):
    """
    MLP predicting an average tangent velocity from (x, t_start, t_end).

    The output is projected onto the tangent space at x by the trainer and
    generator, so the network can share the same simple MLP style as FlowNet.
    """

    def __init__(self, D, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(D + 2, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)
        self.D = D

    def forward(self, x, t_start, t_end):
        if t_start.dim() == 0:
            t_start = t_start.expand(x.shape[0], 1)
        elif t_start.dim() == 1:
            t_start = t_start.unsqueeze(1)
        if t_end.dim() == 0:
            t_end = t_end.expand(x.shape[0], 1)
        elif t_end.dim() == 1:
            t_end = t_end.unsqueeze(1)
        return self.net(torch.cat([x, t_start, t_end], dim=-1))
