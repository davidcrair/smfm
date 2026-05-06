"""
Interpolants on the sphere for flow matching training.
"""

import torch

from surf.geometry.sphere import normalize_sphere


class SLERPInterpolant:
    """
    Great-circle SLERP interpolant between coupled source/target cells.
    This is the vanilla Fisher Flow Matching interpolant -- a single geodesic
    arc on the sphere. Fast, closed-form, but ignores the data manifold.
    """

    def __init__(self, Y0, Y1):
        self.Y0 = Y0
        self.Y1 = Y1
        self.n0 = len(Y0)
        self.n1 = len(Y1)
        self.device = Y0.device

    def sample(self, src_idx, tgt_idx, t):
        """
        src_idx, tgt_idx: (B,) numpy int arrays (local indices into Y0, Y1)
        t: (B, 1) torch tensor on device
        Returns: z_t (B, D), v_t (B, D) on device
        """
        y0 = self.Y0[src_idx]
        y1 = self.Y1[tgt_idx]

        cos_omega = (y0 * y1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)

        w0 = torch.sin((1 - t) * omega) / sin_omega
        w1 = torch.sin(t * omega) / sin_omega
        z_t = w0 * y0 + w1 * y1
        z_t = z_t / z_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # d/dt SLERP = omega * [-cos((1-t)w)/sin(w) * y0 + cos(tw)/sin(w) * y1]
        v_t = omega * (
            -torch.cos((1 - t) * omega) / sin_omega * y0
            + torch.cos(t * omega) / sin_omega * y1
        )
        # Project to tangent space at z_t (handles numerical drift)
        v_t = v_t - (v_t * z_t).sum(dim=-1, keepdim=True) * z_t

        return z_t, v_t
