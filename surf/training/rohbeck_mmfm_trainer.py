"""Rohbeck et al. 2025 (ICLR) Multi-Marginal Flow Matching trainer.

Faithful re-implementation of the framework described in §3 of
"Modeling Complex System Dynamics with Flow Matching Across Time and
Conditions" (Rohbeck et al., ICLR 2025). Three substantive differences
from the simpler pairwise-MMFM trainer in euclidean_flow_trainer.py:

  1. Cubic-spline interpolant μ_t(z) through ALL observed marginals
     (Eq. 9, natural cubic spline minimising ∫||γ''||² subject to
     γ(t_k)=x_k; Holladay's theorem). Replaces piecewise-linear
     interpolation between adjacent pairs.

  2. Time-dependent variance σ_t(z) = 4(t_{k+1}-t)(t-t_k)/(t_{k+1}-t_k)²
     (Eq. 10). Vanishes at observed times, peaks mid-interval at
     σ=1. The conditional path is p_t(x|z) = N(μ_t(z), σ_t(z)²I).

  3. MMOT joint coupling (Eq. 11). For EMD pairwise plans (sharp
     permutation matrices when |X|=|Y|, which is our setting on GoM
     and the sphere-encoded scRNA-seq datasets), MMOT samples reduce
     to deterministic chains: c_0 -> argmax(T_0[c_0,:]) -> ...
     Sampling z from MMOT is equivalent to sampling a chain index.

The conditional vector field follows Lipman et al. 2023 Theorem 3:
  u_t(x|z) = σ'_t(z)/σ_t(z) · (x - μ_t(z)) + μ'_t(z)
With x = μ_t + σ_t · ε, ε ~ N(0,I), this simplifies to:
  u_t = σ'_t · ε + μ'_t

The cost_fn argument is the pairwise OT cost matrix builder. Pass
``compute_euclidean_cost_matrix`` for vanilla Rohbeck-MMFM, or pass a
spectral or blended cost via the registry to get spectral-MMFM-Rohbeck.
"""

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from surf.runtime import get as get_runtime
from surf.models.flow_net_v2 import build_flow_net
from surf.ot.costs import compute_euclidean_cost_matrix


def _build_argmax_mmot_chains(stage_cells_np, cost_fn):
    """Return chains[c, k] = index in stage k along the c-th MMOT chain.

    cost_fn typically expects torch tensors (it calls .detach().cpu()
    inside), so we wrap numpy arrays into CPU tensors before each call.
    """
    import ot
    K = len(stage_cells_np) - 1
    n0 = len(stage_cells_np[0])
    chains = np.zeros((n0, K + 1), dtype=np.int64)
    chains[:, 0] = np.arange(n0)
    for k in range(K):
        X0 = torch.from_numpy(stage_cells_np[k])
        X1 = torch.from_numpy(stage_cells_np[k + 1])
        cost = np.asarray(cost_fn(X0, X1), dtype=np.float64)
        n_src, n_tgt = cost.shape
        a = np.ones(n_src) / n_src
        b = np.ones(n_tgt) / n_tgt
        T = ot.emd(a, b, cost)
        mapping = T.argmax(axis=1)
        chains[:, k + 1] = mapping[chains[:, k]]
    return chains


def _gather_chain_data(stage_cells_np, chains):
    """chain_data[c, k, :] = stage_cells_np[k][chains[c, k]]."""
    n0, K1 = chains.shape
    D = stage_cells_np[0].shape[-1]
    out = np.zeros((n0, K1, D), dtype=np.float32)
    for k in range(K1):
        out[:, k, :] = stage_cells_np[k][chains[:, k]]
    return out


def _spline_coeffs_per_chain(chain_data, stage_times):
    """Precompute natural-cubic-spline coefficients for every chain.

    For each (chain, interval) pair the spline is a cubic polynomial
    p(t) = c0 + c1*(t-t_k) + c2*(t-t_k)^2 + c3*(t-t_k)^3 in each
    dimension. Returns coeffs shaped (n_chains, K, 4, D) so that
    coeffs[c, k, p, d] is the coefficient of (t - t_k)^p.
    """
    n_chains, K1, D = chain_data.shape
    K = K1 - 1
    coeffs = np.zeros((n_chains, K, 4, D), dtype=np.float32)
    for c in range(n_chains):
        cs = CubicSpline(stage_times, chain_data[c], bc_type='natural', axis=0)
        # scipy's cs.c shape is (4, K, D): row 0 is the cubic coefficient
        # (i.e. (t-t_k)^3), row 3 is the constant term. Reorder so axis 0
        # ascends in power.
        coeffs[c] = cs.c[::-1, :, :].transpose(1, 0, 2)
    return coeffs


def train_rohbeck_mmfm(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    label="MMFM-Rohbeck",
    ot_subsample=2000,
    cost_fn=None,
    use_ema=True,
    sigma_scale=1.0,
    flow_net_arch="v1",
    ema_decay=0.999,
    grad_clip=1.0,
    epoch_based=False,
    n_epochs=None,
    warmup_iters=None,
):
    """Train Rohbeck-MMFM on the given training stages.

    Parameters
    ----------
    stage_cells : list of torch.Tensor or np.ndarray, length S = K + 1
        Cells at each observed time, one tensor per stage (n_k, D).
    stage_times : list of float, length S
        Observation times t_0 < t_1 < ... < t_K.
    D : int
        Ambient dimension.
    cost_fn : callable(X0, X1) -> np.ndarray, optional
        OT cost-matrix builder for the pairwise plans that MMOT chains
        ride. Defaults to ``compute_euclidean_cost_matrix``. Pass a
        spectral cost (e.g. via ``make_spectral_cost_fn``) to stack
        spectral OT inside the Rohbeck-MMFM scaffold.
    sigma_scale : float, default 1.0
        Multiplier on the time-varying noise schedule. Paper spec is
        1.0 (peak σ=1 mid-interval).
    """
    rt = get_runtime()
    S = len(stage_cells)
    K = S - 1
    assert S >= 2
    for i in range(K):
        assert stage_times[i] < stage_times[i + 1]

    if cost_fn is None:
        cost_fn = compute_euclidean_cost_matrix

    cost_label = "euclidean" if cost_fn is compute_euclidean_cost_matrix else getattr(
        cost_fn, "__name__", "custom_cost")
    print(f"  Rohbeck-MMFM: {S} stages, OT cost={cost_label}, sigma_scale={sigma_scale}")

    # Subsample to numpy arrays
    rng = np.random.default_rng(0)
    stage_cells_np = []
    for i, cells in enumerate(stage_cells):
        arr = cells.detach().cpu().numpy() if isinstance(cells, torch.Tensor) else np.asarray(cells)
        if len(arr) > ot_subsample:
            idx = rng.choice(len(arr), size=ot_subsample, replace=False)
            arr = arr[idx]
        stage_cells_np.append(arr.astype(np.float32))
        print(f"    stage {i}: {len(arr)} cells (t={stage_times[i]:.2f})")

    print(f"  Building MMOT argmax chains across {S} stages...")
    chains = _build_argmax_mmot_chains(stage_cells_np, cost_fn)
    chain_data = _gather_chain_data(stage_cells_np, chains)
    n_chains = chain_data.shape[0]

    print(f"  Fitting {n_chains} natural cubic splines...")
    coeffs_np = _spline_coeffs_per_chain(chain_data, stage_times)

    coeffs = torch.tensor(coeffs_np, device=rt.device)              # (n_chains, K, 4, D)
    stage_times_t = torch.tensor(stage_times, device=rt.device, dtype=torch.float32)
    dks_t = stage_times_t[1:] - stage_times_t[:-1]                  # (K,)
    t0 = float(stage_times[0])
    tK = float(stage_times[-1])

    # Resolve epoch-based training. When epoch_based=True, compute the
    # iteration budget from n_epochs and the chain count, and sample
    # batches via per-epoch reshuffle (matches Rohbeck/OTP-FM's
    # `reshuffle_each_epoch=true` exactly).
    if epoch_based:
        if n_epochs is None:
            raise ValueError("epoch_based=True requires n_epochs")
        # We'll set the actual loop length once we know n_chains.

    model = build_flow_net(D, arch=flow_net_arch).to(rt.device)
    print(f"  flow_net_arch={flow_net_arch}, ema_decay={ema_decay}, grad_clip={grad_clip}, "
          f"epoch_based={epoch_based}, n_epochs={n_epochs}, warmup_iters={warmup_iters}")
    if rt.use_amp:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"  torch.compile failed ({e}); running uncompiled")
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)

    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    if warmup_iters is None:
        warmup_iters = min(500, n_iters // 10)

    if epoch_based:
        batches_per_epoch = max(1, n_chains // batch_size)
        n_iters = n_epochs * batches_per_epoch
        print(f"  epoch_based: {n_epochs} epochs x {batches_per_epoch} batches = {n_iters} iters")
        chain_perm = torch.randperm(n_chains, device=rt.device)
        perm_pos = 0

    losses = []
    for it in range(n_iters):
        # Sample chain index + uniform t in [t_0, t_K]
        if epoch_based:
            if perm_pos + batch_size > n_chains:
                chain_perm = torch.randperm(n_chains, device=rt.device)
                perm_pos = 0
            chain_idx = chain_perm[perm_pos:perm_pos + batch_size]
            perm_pos += batch_size
        else:
            chain_idx = torch.randint(0, n_chains, (batch_size,), device=rt.device)
        t_b = torch.rand(batch_size, device=rt.device) * (tK - t0) + t0

        # Find interval k_b such that t_k <= t < t_{k+1}; clamp to [0, K-1]
        k_b = torch.searchsorted(stage_times_t, t_b, right=True) - 1
        k_b = k_b.clamp(0, K - 1)

        # Gather spline coefficients for these (chain, interval) pairs
        c_sel = coeffs[chain_idx, k_b]                               # (B, 4, D)
        tk = stage_times_t[k_b]                                       # (B,)
        tkp1 = stage_times_t[k_b + 1]                                 # (B,)
        dk = dks_t[k_b]                                               # (B,)
        dt = (t_b - tk).unsqueeze(-1)                                 # (B, 1)

        # Vectorised polynomial evaluation:
        # mu = c0 + c1*dt + c2*dt^2 + c3*dt^3
        # mu' = c1 + 2*c2*dt + 3*c3*dt^2
        dt_pows = torch.stack([
            torch.ones_like(dt.squeeze(-1)),
            dt.squeeze(-1),
            dt.squeeze(-1) ** 2,
            dt.squeeze(-1) ** 3,
        ], dim=-1)                                                    # (B, 4)
        dt_dpows = torch.stack([
            torch.zeros_like(dt.squeeze(-1)),
            torch.ones_like(dt.squeeze(-1)),
            2 * dt.squeeze(-1),
            3 * dt.squeeze(-1) ** 2,
        ], dim=-1)                                                    # (B, 4)
        mu = (c_sel * dt_pows.unsqueeze(-1)).sum(dim=1)               # (B, D)
        mu_prime = (c_sel * dt_dpows.unsqueeze(-1)).sum(dim=1)        # (B, D)

        # Time-varying noise schedule (Eq. 10) and its derivative
        sigma = sigma_scale * 4.0 * (tkp1 - t_b) * (t_b - tk) / (dk * dk)        # (B,)
        sigma_prime = sigma_scale * 4.0 * (tkp1 + tk - 2.0 * t_b) / (dk * dk)    # (B,)
        sigma = sigma.unsqueeze(-1)                                              # (B, 1)
        sigma_prime = sigma_prime.unsqueeze(-1)                                  # (B, 1)

        # Sample x = mu + sigma * eps;  u = sigma' * eps + mu'
        eps = torch.randn(batch_size, D, device=rt.device)
        x = mu + sigma * eps
        u_target = sigma_prime * eps + mu_prime

        with torch.autocast(device_type=rt.device.type, dtype=rt.amp_dtype, enabled=rt.use_amp):
            v_pred = model(x.detach(), t_b.detach())
            loss = ((v_pred - u_target.detach()) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        if it < warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr * (it + 1) / warmup_iters
        elif it == warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr
        opt.step()
        with torch.no_grad():
            if it == warmup_iters:
                for k, v in model.state_dict().items():
                    ema_state[k].copy_(v)
            else:
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

        if it % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:30s} iter {it:4d}  loss={loss.item():.4f}")

    if use_ema:
        model.load_state_dict(ema_state)
    return model, losses
