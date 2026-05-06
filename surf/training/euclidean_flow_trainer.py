"""Multi-marginal Euclidean flow matching in a configurable Euclidean space."""

import numpy as np
import torch

from surf.runtime import get as get_runtime
from surf.models.flow_net_v2 import build_flow_net
from surf.ot.costs import compute_euclidean_cost_matrix
from surf.ot.coupling import ot_coupling


def train_multi_marginal_euclidean_flow(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    label="MultiMarginalEuclidean",
    ot_subsample=2000,
    cost_fn=None,
    use_ema=True,
    coupling_mode="joint",
    flow_net_arch="v1",
    ema_decay=0.999,
    grad_clip=1.0,
    epoch_based=False,
    n_epochs=None,
    warmup_iters=None,
):
    """
    Multi-marginal Euclidean flow matching on Euclidean state vectors.

    This mirrors the sphere trainer structurally:
      1. Precompute one OT coupling per adjacent pair of stages.
      2. Sample a random adjacent interval and OT-coupled pairs each step.
      3. Train a single time-conditioned vector field on linear interpolants
         x_t = (1 - s) * x0 + s * x1.
      4. Use the constant Euclidean target velocity (x1 - x0) / dt_interval.

    Two coupling modes:
      - "joint" (default): solve OT and SAMPLE pairs proportional to the
        transport plan T_ij. Each iter draws fresh pairs from the plan.
        This is MMFM in the spirit of Tong et al. 2023.
      - "argmax_chain": solve OT and pick mapping_i[c] = argmax(T_i[c, :])
        as a deterministic source->target map; build full chains across
        all train stages so each source cell has a fixed trajectory
        c -> mapping_0[c] -> mapping_1[mapping_0[c]] -> ... and each iter
        draws a (chain, adjacent-stage) pair from those fixed chains.
        This matches the MMFM baseline implementation in Atanackovic et
        al.'s OTP-FM repo (experiments/.../data.py:_build_ot_chains).
    """
    rt = get_runtime()

    assert len(stage_cells) == len(stage_times)
    S = len(stage_cells)
    assert S >= 2
    for i in range(S - 1):
        assert stage_times[i] < stage_times[i + 1]

    model = build_flow_net(D, arch=flow_net_arch).to(rt.device)
    print(f"  flow_net_arch={flow_net_arch}, ema_decay={ema_decay}, grad_clip={grad_clip}")
    if rt.use_amp:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"  torch.compile failed ({e}); running uncompiled")
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)

    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    if warmup_iters is None:
        warmup_iters = min(500, n_iters // 10)

    print(f"  Computing OT couplings for {S - 1} adjacent pairs...")
    stage_cells_sub = []
    rng = np.random.default_rng(0)
    for i, cells in enumerate(stage_cells):
        if len(cells) > ot_subsample:
            idx = rng.choice(len(cells), size=ot_subsample, replace=False)
            stage_cells_sub.append(cells[idx])
        else:
            stage_cells_sub.append(cells)
        print(f"    stage {i}: {len(stage_cells_sub[i])} cells (t={stage_times[i]:.2f})")

    cost_label = "euclidean W2"
    if cost_fn is None:
        cost_fn = compute_euclidean_cost_matrix
    elif hasattr(cost_fn, "__name__"):
        cost_label = cost_fn.__name__
    print(f"  OT cost: {cost_label}")

    if coupling_mode == "argmax_chain":
        # OTP-FM-style MMFM: argmax-chain coupling. Each source cell c at
        # stage 0 is mapped deterministically to stage 1 via argmax(T_0[c]),
        # then onward to stage 2 via argmax(T_1[map_0[c]]), etc., giving
        # one fixed trajectory per source cell across all train stages.
        try:
            import ot
        except ImportError:
            raise RuntimeError("argmax_chain coupling requires the 'pot' (POT) package")
        n0 = len(stage_cells_sub[0])
        chains = np.zeros((n0, S), dtype=np.int64)
        chains[:, 0] = np.arange(n0)
        adj_couplings = []
        for i in range(S - 1):
            X0, X1 = stage_cells_sub[i], stage_cells_sub[i + 1]
            cost = cost_fn(X0, X1)
            n_src, n_tgt = cost.shape
            a = np.ones(n_src) / n_src
            b = np.ones(n_tgt) / n_tgt
            T = ot.emd(a, b, cost)
            mapping = T.argmax(axis=1)        # mapping[c] = best target idx
            chains[:, i + 1] = mapping[chains[:, i]]
            adj_couplings.append((chains[:, i].copy(), chains[:, i + 1].copy()))
        print(f"  argmax_chain mode: {n0} chains across {S} stages")
    else:
        adj_couplings = []
        for i in range(S - 1):
            X0, X1 = stage_cells_sub[i], stage_cells_sub[i + 1]
            cost = cost_fn(X0, X1)
            n_pool = min(20000, len(X0) * len(X1))
            os_, ot_ = ot_coupling(cost, n_pool)
            adj_couplings.append((os_, ot_))

    # Epoch-based (Rohbeck/OTP-FM-style reshuffle): if argmax_chain
    # mode, an "epoch" is one pass through the n_chains across all S
    # stages, picking a random adjacent stage per batch from the chain.
    # If joint mode, an epoch is one pass through coupling pool of
    # min(adj_pool_lens). Either way, we cycle a permutation index.
    if epoch_based:
        if n_epochs is None:
            raise ValueError("epoch_based=True requires n_epochs")
        if coupling_mode == "argmax_chain":
            n_chains = len(adj_couplings[0][0])
        else:
            n_chains = min(len(c[0]) for c in adj_couplings)
        batches_per_epoch = max(1, n_chains // batch_size)
        n_iters = n_epochs * batches_per_epoch
        print(f"  epoch_based: {n_epochs} epochs x {batches_per_epoch} batches = {n_iters} iters")
        epoch_perms = [np.random.permutation(len(c[0])) for c in adj_couplings]
        epoch_pos = [0] * len(adj_couplings)

    losses = []
    for it in range(n_iters):
        i = np.random.randint(0, S - 1)
        X0 = stage_cells_sub[i]
        X1 = stage_cells_sub[i + 1]
        ot_src_i, ot_tgt_i = adj_couplings[i]
        t_start = stage_times[i]
        t_end = stage_times[i + 1]
        dt_interval = t_end - t_start

        if epoch_based:
            n_pool = len(ot_src_i)
            if epoch_pos[i] + batch_size > n_pool:
                epoch_perms[i] = np.random.permutation(n_pool)
                epoch_pos[i] = 0
            idx = epoch_perms[i][epoch_pos[i]:epoch_pos[i] + batch_size]
            epoch_pos[i] += batch_size
        else:
            idx = np.random.choice(len(ot_src_i), size=batch_size, replace=True)
        s_idx = ot_src_i[idx]
        g_idx = ot_tgt_i[idx]
        s_idx_t = torch.as_tensor(s_idx, device=rt.device, dtype=torch.long)
        g_idx_t = torch.as_tensor(g_idx, device=rt.device, dtype=torch.long)

        s_local = torch.rand(batch_size, 1, device=rt.device)
        t_global = t_start + dt_interval * s_local

        x0 = X0[s_idx_t]
        x1 = X1[g_idx_t]
        x_t = (1.0 - s_local) * x0 + s_local * x1
        v_target = (x1 - x0) / dt_interval

        with torch.autocast(device_type=rt.device.type, dtype=rt.amp_dtype, enabled=rt.use_amp):
            v_pred = model(x_t.detach(), t_global.squeeze(1))
            loss = ((v_pred - v_target.detach()) ** 2).mean()
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
                # Polyak-style: snap EMA to live weights at end of warmup, so
                # the iter-0 random init never contaminates the EMA average.
                # (At decay=0.999 over 3000 iters, this would otherwise be
                # ~5% of the final state.)
                for k, v in model.state_dict().items():
                    ema_state[k].copy_(v)
            else:
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

        if it % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {it:4d}  loss={loss.item():.4f}")

    if use_ema:
        model.load_state_dict(ema_state)
    return model, losses
