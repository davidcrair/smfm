"""
Endpoint mean-flow baseline on the positive-orthant sphere.

This trains a direct map from a source point and time interval to the average
tangent velocity whose one-step exponential map reaches the target marginal.
"""

import numpy as np
import torch

from surf.runtime import get as get_runtime
from surf.geometry.sphere import (
    compute_sphere_cost_matrix,
    sphere_log_map,
)
from surf.models.mean_flow_net import MeanFlowNet
from surf.ot.coupling import ot_coupling


def train_sphere_endpoint_mean_flow(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    label="MM+MeanFlow",
    ot_subsample=2000,
):
    """
    Train a one-step endpoint predictor on sphere coordinates.

    For every stage pair i<j, OT pairs are computed once. A training sample
    predicts Log_{x_i}(x_j) / (t_j - t_i) from (x_i, t_i, t_j), so evaluation
    can move with a single Exp step over arbitrary stage intervals.
    """
    rt = get_runtime()

    assert len(stage_cells) == len(stage_times)
    S = len(stage_cells)
    assert S >= 2
    for i in range(S - 1):
        assert stage_times[i] < stage_times[i + 1]

    model = MeanFlowNet(D).to(rt.device)
    if rt.use_amp:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"  torch.compile failed ({e}); running uncompiled")
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)

    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    warmup_iters = min(500, n_iters // 10)

    print(f"  Computing OT couplings for {S * (S - 1) // 2} endpoint pairs...")
    stage_cells_sub = []
    rng = np.random.default_rng(0)
    for i, cells in enumerate(stage_cells):
        if len(cells) > ot_subsample:
            idx = rng.choice(len(cells), size=ot_subsample, replace=False)
            stage_cells_sub.append(cells[idx])
        else:
            stage_cells_sub.append(cells)
        print(f"    stage {i}: {len(stage_cells_sub[i])} cells (t={stage_times[i]:.2f})")

    pair_specs = []
    for i in range(S - 1):
        for j in range(i + 1, S):
            Y0, Y1 = stage_cells_sub[i], stage_cells_sub[j]
            cost = compute_sphere_cost_matrix(Y0, Y1)
            n_pool = min(20000, len(Y0) * len(Y1))
            ot_src, ot_tgt = ot_coupling(cost, n_pool)
            pair_specs.append({
                "i": i,
                "j": j,
                "Y0": Y0,
                "Y1": Y1,
                "src": torch.as_tensor(ot_src, device=rt.device, dtype=torch.long),
                "tgt": torch.as_tensor(ot_tgt, device=rt.device, dtype=torch.long),
                "t_start": float(stage_times[i]),
                "t_end": float(stage_times[j]),
                "dt": float(stage_times[j] - stage_times[i]),
            })

    losses = []
    for it in range(n_iters):
        spec = pair_specs[np.random.randint(0, len(pair_specs))]
        pair_idx = torch.randint(len(spec["src"]), (batch_size,), device=rt.device)
        s_idx = spec["src"][pair_idx]
        t_idx = spec["tgt"][pair_idx]

        x0 = spec["Y0"][s_idx]
        x1 = spec["Y1"][t_idx]
        t_start = torch.full((batch_size,), spec["t_start"], device=rt.device)
        t_end = torch.full((batch_size,), spec["t_end"], device=rt.device)

        with torch.no_grad():
            v_target = sphere_log_map(x0, x1) / spec["dt"]

        with torch.autocast(device_type=rt.device.type, dtype=rt.amp_dtype, enabled=rt.use_amp):
            v_pred_raw = model(x0.detach(), t_start, t_end)
            v_pred = v_pred_raw - (v_pred_raw * x0).sum(dim=-1, keepdim=True) * x0
            loss = ((v_pred - v_target.detach()) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if it < warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr * (it + 1) / warmup_iters
        elif it == warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr
        opt.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

        if it % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {it:4d}  loss={loss.item():.4f}")

    model.load_state_dict(ema_state)
    return model, losses
