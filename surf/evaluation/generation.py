"""
Flow integration routines for generating predictions from trained models.

Contains:
- generate_fisher_flow: single-shot Euler integration on the sphere
- generate_euclidean_flow: single-shot Euler integration in Euclidean space
- generate_fisher_flow_trajectory: trajectory integration with checkpoints
"""

import numpy as np
import torch
import torch.nn as nn

from surf.runtime import get as get_runtime
from surf.geometry.sphere import normalize_sphere, from_orthant, sphere_exp_map


def _project_to_support(
    z,
    *,
    support_cells=None,
    manifold_project_mode="none",
    manifold_project_k=32,
    manifold_project_beta=20.0,
    manifold_project_mix=1.0,
):
    """Project a sphere point back toward observed support cells.

    Parameters
    ----------
    z : torch.Tensor
        Current sphere points, shape (B, D).
    support_cells : torch.Tensor | None
        Support cloud on the sphere, shape (N, D). When omitted, projection is
        disabled regardless of mode.
    manifold_project_mode : {"none", "nearest", "soft"}
        ``nearest`` snaps to the closest support cell by cosine similarity.
        ``soft`` takes a soft kNN barycenter in ambient sphere coordinates.
    manifold_project_k : int
        Number of nearest support cells to use for ``soft`` projection.
    manifold_project_beta : float
        Inverse temperature for soft kNN weighting.
    manifold_project_mix : float
        Blend between the current point and projected point. 1.0 means fully
        project, 0.0 leaves the point unchanged.
    """
    if support_cells is None or manifold_project_mode == "none":
        return z

    mode = str(manifold_project_mode)
    if mode not in ("nearest", "soft"):
        raise ValueError(
            f"Unknown manifold_project_mode={mode!r}; expected 'none', 'nearest', or 'soft'."
        )

    support = support_cells.to(device=z.device, dtype=z.dtype)
    mix = float(np.clip(manifold_project_mix, 0.0, 1.0))
    if mix <= 0.0:
        return z

    sims = z @ support.T
    if mode == "nearest":
        idx = sims.argmax(dim=1)
        projected = support[idx]
    else:
        k = min(int(manifold_project_k), support.shape[0])
        top_sims, top_idx = torch.topk(sims, k=k, dim=1)
        neigh = support[top_idx]
        weights = torch.softmax(manifold_project_beta * top_sims, dim=1)
        projected = (weights.unsqueeze(-1) * neigh).sum(dim=1)
        projected = normalize_sphere(projected.clamp(min=1e-8))

    if mix < 1.0:
        projected = normalize_sphere(
            ((1.0 - mix) * z + mix * projected).clamp(min=1e-8)
        )
    return projected


def generate_fisher_flow(model, Y0_test, n_steps=50, t_start=0.0, t_end=1.0,
                         score_net=None, alpha=0.0, score_net_sigma=0.1,
                         inf_sigma=0.0, support_cells=None,
                         manifold_project_mode="none",
                         manifold_project_every=1,
                         manifold_project_k=32,
                         manifold_project_beta=20.0,
                         manifold_project_mix=1.0):
    """
    Integrate the learned tangent velocity field with Euler steps on the
    sphere, from t_start to t_end. Defaults to the full [0, 1] interval;
    call with t_end < 1.0 for intermediate-timepoint eval.

    When score_net and alpha>0 are supplied, each Euler step is augmented
    with a manifold-correcting score term:

        v_total = v_flow(z, t) + alpha * score(z)

    When inf_sigma > 0, integration switches from deterministic Euler to a
    first-order Euler-Maruyama SDE on the sphere: at each step a tangent-
    space Gaussian increment `sqrt(dt) * inf_sigma * eps_tan` is added on
    top of the drift, giving DDPM-style stochastic sampling. With
    inf_sigma = 0 this reduces exactly to deterministic Euler.

    When support_cells and manifold_project_mode are supplied, each step is
    optionally projected back toward the observed support cloud. This is a
    pragmatic manifold-correction for noisy rollouts that would otherwise
    drift into low-density sphere regions.
    """
    rt = get_runtime()
    model.eval()
    use_score = alpha > 0.0 and score_net is not None
    if use_score and isinstance(score_net, nn.Module):
        score_net.eval()
    with torch.no_grad():
        zt = Y0_test.clone().to(rt.device)
        dt = (t_end - t_start) / n_steps
        log_sigma = float(np.log(score_net_sigma)) if use_score else 0.0
        sqrt_dt = dt ** 0.5
        project_every = max(int(manifold_project_every), 1)
        for step in range(n_steps):
            t_val = torch.full((len(zt),), t_start + step * dt, device=rt.device)
            v = model(zt, t_val)
            v = v - (v * zt).sum(dim=-1, keepdim=True) * zt
            if use_score:
                log_sig_t = torch.full((len(zt),), log_sigma, device=rt.device)
                s = score_net(zt, log_sig_t)
                v = v + alpha * s
            if inf_sigma > 0.0:
                eps = torch.randn_like(zt) * inf_sigma
                eps = eps - (eps * zt).sum(dim=-1, keepdim=True) * zt
                zt = normalize_sphere(zt + dt * v + sqrt_dt * eps)
            else:
                zt = normalize_sphere(zt + dt * v)
            if (step + 1) % project_every == 0 or step == n_steps - 1:
                zt = _project_to_support(
                    zt,
                    support_cells=support_cells,
                    manifold_project_mode=manifold_project_mode,
                    manifold_project_k=manifold_project_k,
                    manifold_project_beta=manifold_project_beta,
                    manifold_project_mix=manifold_project_mix,
                )
    return zt.cpu()


def generate_euclidean_flow(model, X0_test, n_steps=50, t_start=0.0, t_end=1.0,
                            score_net=None, alpha=0.0, score_net_sigma=0.1,
                            inf_sigma=0.0):
    """Integrate a Euclidean velocity field with Euler steps."""
    rt = get_runtime()
    model.eval()
    use_score = alpha > 0.0 and score_net is not None
    if use_score and isinstance(score_net, nn.Module):
        score_net.eval()
    with torch.no_grad():
        xt = X0_test.clone().to(rt.device)
        dt = (t_end - t_start) / n_steps
        log_sigma = float(np.log(score_net_sigma)) if use_score else 0.0
        sqrt_dt = dt ** 0.5
        for step in range(n_steps):
            t_val = torch.full((len(xt),), t_start + step * dt, device=rt.device)
            v = model(xt, t_val)
            if use_score:
                log_sig_t = torch.full((len(xt),), log_sigma, device=rt.device)
                v = v + alpha * score_net(xt, log_sig_t)
            if inf_sigma > 0.0:
                xt = xt + dt * v + sqrt_dt * inf_sigma * torch.randn_like(xt)
            else:
                xt = xt + dt * v
    return xt.cpu()


def generate_sphere_mean_flow(model, Y0_test, n_steps=1, t_start=0.0, t_end=1.0,
                              score_net=None, alpha=0.0, score_net_sigma=0.1,
                              inf_sigma=0.0):
    """Apply a one-step endpoint mean-flow map on the sphere."""
    del n_steps, score_net, alpha, score_net_sigma, inf_sigma
    rt = get_runtime()
    model.eval()
    with torch.no_grad():
        x0 = Y0_test.clone().to(rt.device)
        t0 = torch.full((len(x0),), float(t_start), device=rt.device)
        t1 = torch.full((len(x0),), float(t_end), device=rt.device)
        v_raw = model(x0, t0, t1)
        v = v_raw - (v_raw * x0).sum(dim=-1, keepdim=True) * x0
        xt = sphere_exp_map(x0, (float(t_end) - float(t_start)) * v)
    return xt.cpu()


def generate_euclidean_flow_trajectory(model, X0_test, n_steps=50, t_start=0.0, t_end=1.0,
                                        n_checkpoints=20, score_net=None, alpha=0.0,
                                        score_net_sigma=0.1, inf_sigma=0.0):
    """Like generate_euclidean_flow but returns intermediate positions for visualization.

    Returns list of (t, positions_cpu) tuples of length n_checkpoints+1 (including
    the starting position at t_start). Positions are in the same Euclidean space
    the model was trained in (log1p for the MM+Linear baselines).
    """
    rt = get_runtime()
    model.eval()
    use_score = alpha > 0.0 and score_net is not None
    if use_score and isinstance(score_net, nn.Module):
        score_net.eval()
    save_every = max(1, n_steps // n_checkpoints)
    with torch.no_grad():
        xt = X0_test.clone().to(rt.device)
        dt = (t_end - t_start) / n_steps
        log_sigma = float(np.log(score_net_sigma)) if use_score else 0.0
        sqrt_dt = dt ** 0.5
        trajectory = [(t_start, xt.cpu().clone())]
        for step in range(n_steps):
            t_val = torch.full((len(xt),), t_start + step * dt, device=rt.device)
            v = model(xt, t_val)
            if use_score:
                log_sig_t = torch.full((len(xt),), log_sigma, device=rt.device)
                v = v + alpha * score_net(xt, log_sig_t)
            if inf_sigma > 0.0:
                xt = xt + dt * v + sqrt_dt * inf_sigma * torch.randn_like(xt)
            else:
                xt = xt + dt * v
            if (step + 1) % save_every == 0 or step == n_steps - 1:
                trajectory.append((t_start + (step + 1) * dt, xt.cpu().clone()))
    return trajectory


def generate_fisher_flow_trajectory(model, Y0_test, n_steps=50, t_start=0.0, t_end=1.0,
                                    n_checkpoints=20, score_net=None, alpha=0.0,
                                    score_net_sigma=0.1, inf_sigma=0.0,
                                    support_cells=None,
                                    manifold_project_mode="none",
                                    manifold_project_every=1,
                                    manifold_project_k=32,
                                    manifold_project_beta=20.0,
                                    manifold_project_mix=1.0):
    """Like generate_fisher_flow but returns intermediate positions for visualization.

    Returns list of (t, positions_cpu) tuples of length n_checkpoints+1
    (including the starting position at t_start).
    """
    rt = get_runtime()
    model.eval()
    use_score = alpha > 0.0 and score_net is not None
    if use_score and isinstance(score_net, nn.Module):
        score_net.eval()
    save_every = max(1, n_steps // n_checkpoints)
    with torch.no_grad():
        zt = Y0_test.clone().to(rt.device)
        dt = (t_end - t_start) / n_steps
        log_sigma = float(np.log(score_net_sigma)) if use_score else 0.0
        sqrt_dt = dt ** 0.5
        project_every = max(int(manifold_project_every), 1)
        trajectory = [(t_start, from_orthant(zt).cpu())]
        for step in range(n_steps):
            t_val = torch.full((len(zt),), t_start + step * dt, device=rt.device)
            v = model(zt, t_val)
            v = v - (v * zt).sum(dim=-1, keepdim=True) * zt
            if use_score:
                log_sig_t = torch.full((len(zt),), log_sigma, device=rt.device)
                s = score_net(zt, log_sig_t)
                v = v + alpha * s
            if inf_sigma > 0.0:
                eps = torch.randn_like(zt) * inf_sigma
                eps = eps - (eps * zt).sum(dim=-1, keepdim=True) * zt
                zt = normalize_sphere(zt + dt * v + sqrt_dt * eps)
            else:
                zt = normalize_sphere(zt + dt * v)
            if (step + 1) % project_every == 0 or step == n_steps - 1:
                zt = _project_to_support(
                    zt,
                    support_cells=support_cells,
                    manifold_project_mode=manifold_project_mode,
                    manifold_project_k=manifold_project_k,
                    manifold_project_beta=manifold_project_beta,
                    manifold_project_mix=manifold_project_mix,
                )
            if (step + 1) % save_every == 0 or step == n_steps - 1:
                trajectory.append((t_start + (step + 1) * dt, from_orthant(zt).cpu()))
    return trajectory


def generate_sphere_mean_flow_trajectory(model, Y0_test, n_steps=50, t_start=0.0, t_end=1.0,
                                         n_checkpoints=20, score_net=None, alpha=0.0,
                                         score_net_sigma=0.1, inf_sigma=0.0):
    """Endpoint mean-flow checkpoints from the same source to each checkpoint."""
    del n_steps, score_net, alpha, score_net_sigma, inf_sigma
    rt = get_runtime()
    model.eval()
    times = np.linspace(float(t_start), float(t_end), int(n_checkpoints) + 1)
    with torch.no_grad():
        x0 = Y0_test.clone().to(rt.device)
        trajectory = []
        for t in times:
            if t == float(t_start):
                xt = x0
            else:
                t0 = torch.full((len(x0),), float(t_start), device=rt.device)
                t1 = torch.full((len(x0),), float(t), device=rt.device)
                v_raw = model(x0, t0, t1)
                v = v_raw - (v_raw * x0).sum(dim=-1, keepdim=True) * x0
                xt = sphere_exp_map(x0, (float(t) - float(t_start)) * v)
            trajectory.append((float(t), from_orthant(xt).cpu()))
    return trajectory
