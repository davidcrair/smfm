"""
Training routines for Riemannian score networks on the sphere.

Contains:
- sample_log_sigma_lognormal: EDM-style lognormal noise schedule sampling
- train_riemannian_score: unconditional denoising score matching
- train_timed_riemannian_score: time-conditioned denoising score matching
- train_forward_score_nets: per-interval forward-directed score training
"""

import numpy as np
import torch

from surf.runtime import get as get_runtime
from surf.geometry.sphere import sphere_brownian_perturb, normalize_sphere
from surf.models.score_net import RiemannianScoreNet, TimedRiemannianScoreNet


def sample_log_sigma_lognormal(batch_size, device, sigma_min, sigma_max,
                                log_mean=-1.2, log_std=1.2):
    """
    EDM-style lognormal sigma sampling, clipped to [sigma_min, sigma_max].
    Concentrates training samples near sigma ~ exp(log_mean) where DSM is
    most informative, rather than log-uniform across the whole range.

    Default (mean=-1.2, std=1.2) peaks at sigma ~ 0.3.

    Returns log_sigma of shape (batch_size, 1).
    """
    log_sigma = torch.randn(batch_size, 1, device=device) * log_std + log_mean
    return log_sigma.clamp(float(np.log(sigma_min)), float(np.log(sigma_max)))


def train_riemannian_score(
    cells,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    sigma_min=0.02,
    sigma_max=1.0,
    lognormal_mean=-1.2,
    lognormal_std=1.2,
    n_brownian_steps=3,
    hidden=256,
    depth=4,
    label="ScoreNet",
):
    """
    Denoising score matching on the sphere (Song & Ermon, de Bortoli et al.).

    Improvements vs naive DSM:
      1. Wider sigma range [sigma_min=0.02, sigma_max=1.0] -- network sees
         multi-scale behavior instead of a narrow band.
      2. Lognormal sigma sampling (EDM, Karras et al. 2022) with
         log_sigma ~ N(log_mean, log_std^2) clipped to [sigma_min, sigma_max].
         Concentrates samples near sigma ~ 0.3 where DSM is most informative.
      3. Multi-step Brownian perturbation (n_brownian_steps Euler steps)
         instead of a single Gaussian-tangent retraction. More faithful to
         the heat kernel on the sphere at large sigma.
      4. Sinusoidal (Fourier) embedding of log_sigma via SigmaEmbedding,
         inside RiemannianScoreNet. Richer conditioning than raw scalar.

    Target: log_z(c) / sigma^2 (Riemannian DSM). Loss: sigma^2-weighted MSE so the
    training signal is balanced across noise scales.
    """
    rt = get_runtime()
    model = RiemannianScoreNet(D, hidden=hidden, depth=depth).to(rt.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)

    N = len(cells)
    cells_dev = cells.to(rt.device)

    losses = []
    for i in range(n_iters):
        idx = np.random.randint(0, N, size=batch_size)
        c = cells_dev[idx]  # (B, D)

        # Lognormal sigma sampling (clipped to valid range)
        log_sigma = sample_log_sigma_lognormal(
            batch_size, rt.device, sigma_min, sigma_max,
            log_mean=lognormal_mean, log_std=lognormal_std,
        )
        sigma = log_sigma.exp()  # (B, 1)

        # Multi-step Brownian perturbation (approx heat kernel)
        z = sphere_brownian_perturb(c, sigma, n_steps=n_brownian_steps)

        # Target: log_z(c) / sigma^2
        cos_omega = (z * c).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        log_z_to_c = (omega / sin_omega) * (c - cos_omega * z)
        log_z_to_c = log_z_to_c - (log_z_to_c * z).sum(dim=-1, keepdim=True) * z
        target = log_z_to_c / (sigma * sigma)

        s_pred = model(z, log_sigma.squeeze(1))

        # sigma^2-weighted MSE: cancels the 1/sigma^2 target scale
        loss = (((s_pred - target) * sigma) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {i:4d}  loss={loss.item():.4f}")

    model.eval()
    return model, losses


def train_timed_riemannian_score(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    sigma_min=0.02,
    sigma_max=1.0,
    lognormal_mean=-1.2,
    lognormal_std=1.2,
    n_brownian_steps=3,
    hidden=256,
    depth=4,
    label="TimedScoreNet",
):
    """
    Time-conditioned denoising score matching on the sphere.

    At each step:
      1. Pick a stage interval i in [0, S-2] uniformly.
      2. Sample s_local ~ U(0, 1) per batch element.
      3. t = stage_times[i] + s_local * (stage_times[i+1] - stage_times[i]).
      4. Draw each batch element's clean cell c from stage i with prob
         (1 - s_local), else from stage i+1. This gives a valid mixture
         between the two marginals that the flow must pass through.
      5. Sample sigma via lognormal, perturb c via multi-step sphere Brownian.
      6. Target: log_z(c) / sigma^2, same as train_riemannian_score.
      7. Predict via TimedRiemannianScoreNet(z, log_sigma, t), sigma^2-weighted MSE.

    At inference (multi-marginal flow training), the timed score net is
    queried at (z_t, log_sigma_tensor, t_global) so it pulls z_t toward
    cells at the correct developmental stage, not the pooled centroid.

    Returns (model, losses) with model in eval mode.
    """
    rt = get_runtime()
    S = len(stage_cells)
    assert S >= 2, "need at least 2 stages for a multi-marginal time-conditioned score"
    stage_cells_dev = [c.to(rt.device) for c in stage_cells]

    model = TimedRiemannianScoreNet(D, hidden=hidden, depth=depth).to(rt.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)

    losses = []
    for it in range(n_iters):
        # Pick an interval and sample s_local ~ U(0, 1)
        i = np.random.randint(0, S - 1)
        s_local = torch.rand(batch_size, 1, device=rt.device)  # (B, 1)
        t_start = stage_times[i]
        t_end = stage_times[i + 1]
        t_batch = t_start + s_local * (t_end - t_start)  # (B, 1)

        # Per-element coin flip: stage i with prob (1 - s_local), else stage i+1
        use_next = (torch.rand(batch_size, device=rt.device) < s_local.squeeze(1))
        n_next = int(use_next.sum().item())
        n_this = batch_size - n_next

        cells_this = stage_cells_dev[i]
        cells_next = stage_cells_dev[i + 1]
        idx_this = torch.randint(0, len(cells_this), (n_this,), device=rt.device)
        idx_next = torch.randint(0, len(cells_next), (n_next,), device=rt.device)

        c = torch.empty(batch_size, D, device=rt.device, dtype=cells_this.dtype)
        c[~use_next] = cells_this[idx_this]
        c[use_next] = cells_next[idx_next]

        # Lognormal sigma
        log_sigma = sample_log_sigma_lognormal(
            batch_size, rt.device, sigma_min, sigma_max,
            log_mean=lognormal_mean, log_std=lognormal_std,
        )
        sigma = log_sigma.exp()  # (B, 1)

        # Multi-step Brownian perturbation
        z = sphere_brownian_perturb(c, sigma, n_steps=n_brownian_steps)

        # DSM target: log_z(c) / sigma^2
        cos_omega = (z * c).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        log_z_to_c = (omega / sin_omega) * (c - cos_omega * z)
        log_z_to_c = log_z_to_c - (log_z_to_c * z).sum(dim=-1, keepdim=True) * z
        target = log_z_to_c / (sigma * sigma)

        s_pred = model(z, log_sigma.squeeze(1), t_batch.squeeze(1))

        loss = (((s_pred - target) * sigma) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {it:4d}  loss={loss.item():.4f}")

    model.eval()
    return model, losses


def train_forward_score_nets(
    stage_cells,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    sigma_min=0.05,
    sigma_max=0.7,
):
    """
    Train one forward-directed score net per stage interval.

    For interval i (between stages i and i+1), the score net is trained on
    cells from stages i+1, i+2, ..., S-1 only -- the "future" stages. This
    ensures the score always pulls the trajectory forward along the
    developmental direction, never backward toward already-visited stages.

    Returns a list of S-1 trained RiemannianScoreNets (one per interval).
    """
    S = len(stage_cells)
    nets = []
    for i in range(S - 1):
        future_cells = torch.cat(stage_cells[i + 1:], dim=0)
        print(f"\n  Interval {i} (t in [{i/(S-1):.2f}, {(i+1)/(S-1):.2f}]): "
              f"training on {len(future_cells)} cells from stages {i+1}..{S-1}")
        net, _ = train_riemannian_score(
            future_cells, D, n_iters=n_iters, batch_size=batch_size,
            lr=lr, sigma_min=sigma_min, sigma_max=sigma_max,
            label=f"ScoreNet[interval{i}]",
        )
        nets.append(net)
    return nets
