#!/usr/bin/env python
"""Debug the biharmonic premetric target against the SLERP target."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from surf.data.embryoid import load_embryoid_body
from surf.geometry.premetric import BiharmonicPremetric
from surf.geometry.sphere import normalize_sphere, to_compositional, to_orthant
from surf.ot.costs import compute_biharmonic_cost_matrix
from surf.ot.coupling import ot_coupling
from surf.runtime import setup


@dataclass
class IntervalDebug:
    label: str
    dt_interval: float
    slerp_norms: torch.Tensor
    premetric_norms: torch.Tensor
    start_distances: torch.Tensor
    end_distances: torch.Tensor
    start_grad_norms: torch.Tensor
    end_grad_norms: torch.Tensor
    start_drive_scales: torch.Tensor
    end_drive_scales: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_std: torch.Tensor
    decrease_fraction: torch.Tensor
    linear_mae: float


def _parse_holdout_stages(raw: str | None, n_stages: int) -> set[int]:
    if not raw:
        return set()
    held = {int(x) for x in raw.split(",") if x.strip()}
    for idx in held:
        if idx < 0 or idx >= n_stages:
            raise ValueError(f"holdout stage {idx} out of range [0, {n_stages - 1}]")
    return held


def _subsample_stage_cells(stage_cells: list[torch.Tensor], ot_subsample: int) -> list[torch.Tensor]:
    rng = np.random.default_rng(0)
    subsampled = []
    for cells in stage_cells:
        if len(cells) > ot_subsample:
            idx = rng.choice(len(cells), size=ot_subsample, replace=False)
            subsampled.append(cells[idx])
        else:
            subsampled.append(cells)
    return subsampled


def _slerp_target(y0: torch.Tensor, y1: torch.Tensor, s_local: torch.Tensor, dt_interval: float):
    cos_omega = (y0 * y1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega).clamp(min=1e-8)
    w0 = torch.sin((1 - s_local) * omega) / sin_omega
    w1 = torch.sin(s_local * omega) * 1.0 / sin_omega
    z_t = normalize_sphere(w0 * y0 + w1 * y1)
    v_local = omega * (
        -torch.cos((1 - s_local) * omega) / sin_omega * y0
        + torch.cos(s_local * omega) / sin_omega * y1
    )
    v_local = v_local - (v_local * z_t).sum(dim=-1, keepdim=True) * z_t
    return z_t, v_local / dt_interval


def _summary_stats(x: torch.Tensor) -> str:
    q = torch.quantile(x, torch.tensor([0.5, 0.95, 0.99], device=x.device))
    return (
        f"mean={x.mean().item():.4f}  std={x.std().item():.4f}  "
        f"p50={q[0].item():.4f}  p95={q[1].item():.4f}  p99={q[2].item():.4f}  "
        f"max={x.max().item():.4f}"
    )


def _distance_grad_drive(
    premetric: BiharmonicPremetric,
    x: torch.Tensor,
    target_emb: torch.Tensor,
):
    d, grad_tan = premetric.distance_and_gradient(x, target_emb)
    grad_norm = grad_tan.norm(dim=-1)
    drive_scale = d.squeeze(-1) / grad_norm.square().clamp(min=premetric.eps)
    return d.squeeze(-1), grad_norm, drive_scale


def _debug_interval(
    interval_idx: int,
    src_label: str,
    tgt_label: str,
    y0_all: torch.Tensor,
    y1_all: torch.Tensor,
    target_stage_idx: int,
    dt_interval: float,
    coupling: tuple[np.ndarray, np.ndarray],
    premetric: BiharmonicPremetric,
    batch_size: int,
    schedule_batch_size: int,
    ode_steps: int,
    grid_steps: int,
    seed: int,
) -> IntervalDebug:
    rng = np.random.default_rng(seed + interval_idx)
    torch.manual_seed(seed + interval_idx)

    ot_src_i, ot_tgt_i = coupling
    idx = rng.choice(len(ot_src_i), size=batch_size, replace=True)
    s_idx = torch.as_tensor(ot_src_i[idx], device=y0_all.device, dtype=torch.long)
    g_idx = torch.as_tensor(ot_tgt_i[idx], device=y0_all.device, dtype=torch.long)
    s_local = torch.rand(batch_size, 1, device=y0_all.device).clamp(max=1.0 - 1e-4)

    y0 = y0_all[s_idx]
    y1 = y1_all[g_idx]
    target_emb = premetric.target_embeddings(target_stage_idx, g_idx)

    _, v_slerp = _slerp_target(y0, y1, s_local, dt_interval)
    with torch.no_grad():
        z_premetric = premetric.integrate(y0, target_emb, s_local, n_steps=ode_steps)
        v_premetric = premetric.field(z_premetric, target_emb, s_local) / dt_interval
        d0, grad0, drive0 = _distance_grad_drive(premetric, y0, target_emb)
        dt_end, grad_end, drive_end = _distance_grad_drive(premetric, z_premetric, target_emb)

    sched_idx = rng.choice(len(ot_src_i), size=schedule_batch_size, replace=True)
    sched_s = torch.as_tensor(ot_src_i[sched_idx], device=y0_all.device, dtype=torch.long)
    sched_g = torch.as_tensor(ot_tgt_i[sched_idx], device=y0_all.device, dtype=torch.long)
    x0 = y0_all[sched_s]
    target_sched = premetric.target_embeddings(target_stage_idx, sched_g)
    d_init, _ = premetric.distance_and_gradient(x0, target_sched)

    grid = torch.linspace(0.0, 1.0, grid_steps + 1, device=y0_all.device).unsqueeze(1)
    ratios = []
    for tau in grid:
        tau_batch = torch.full((schedule_batch_size, 1), float(tau.item()), device=y0_all.device)
        with torch.no_grad():
            x_tau = premetric.integrate(x0, target_sched, tau_batch, n_steps=ode_steps)
            d_tau, _ = premetric.distance_and_gradient(x_tau, target_sched)
        ratios.append((d_tau / d_init.clamp(min=1e-8)).squeeze(-1))
    ratio_mat = torch.stack(ratios, dim=1)
    ideal = 1.0 - grid.squeeze(1)
    ratio_mean = ratio_mat.mean(dim=0)
    ratio_std = ratio_mat.std(dim=0)
    decrease_fraction = (ratio_mat[:, 1:] <= ratio_mat[:, :-1] + 1e-5).float().mean(dim=0)
    linear_mae = (ratio_mean - ideal).abs().mean().item()

    return IntervalDebug(
        label=f"{src_label} -> {tgt_label}",
        dt_interval=dt_interval,
        slerp_norms=v_slerp.norm(dim=-1),
        premetric_norms=v_premetric.norm(dim=-1),
        start_distances=d0,
        end_distances=dt_end,
        start_grad_norms=grad0,
        end_grad_norms=grad_end,
        start_drive_scales=drive0,
        end_drive_scales=drive_end,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        decrease_fraction=decrease_fraction,
        linear_mae=linear_mae,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="embryoid_body.h5ad")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--holdout-stages", default="1,3")
    parser.add_argument("--ot-subsample", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--schedule-batch-size", type=int, default=256)
    parser.add_argument("--grid-steps", type=int, default=8)
    parser.add_argument("--knn", type=int, default=15)
    parser.add_argument("--n-eig", type=int, default=50)
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--extension-k", type=int, default=64)
    parser.add_argument("--softmax-beta", type=float, default=10.0)
    parser.add_argument("--ode-steps", type=int, default=16)
    parser.add_argument("--time-cap", type=float, default=0.9)
    parser.add_argument("--grad-norm-floor", type=float, default=0.05)
    parser.add_argument("--max-drive-scale", type=float, default=50.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rt = setup(args.device)
    print(f"Debug device: {rt.device}")
    print(
        "Premetric config:"
        f" knn={args.knn}, n_eig={args.n_eig}, weight_power={args.weight_power},"
        f" extension_k={args.extension_k}, softmax_beta={args.softmax_beta},"
        f" ode_steps={args.ode_steps}, time_cap={args.time_cap},"
        f" grad_norm_floor={args.grad_norm_floor}, max_drive_scale={args.max_drive_scale}"
    )

    data = load_embryoid_body(args.data_path, n_hvg=args.n_hvg)
    stages = data["train"]["stages"]
    stage_times = [i / (len(stages) - 1) for i in range(len(stages))]
    held_set = _parse_holdout_stages(args.holdout_stages, len(stages))
    train_idx = [i for i in range(len(stages)) if i not in held_set]
    train_labels = [stages[i] for i in train_idx]
    train_stage_times = [stage_times[i] for i in train_idx]

    train_stage_cells = [
        normalize_sphere(to_orthant(to_compositional(data["train"]["cells"][stages[i]]))).to(rt.device)
        for i in train_idx
    ]
    stage_cells_sub = _subsample_stage_cells(train_stage_cells, args.ot_subsample)

    print(f"Training stages: {train_labels}")
    print(f"Training times:  {[f'{t:.2f}' for t in train_stage_times]}")
    print(f"Subsampled sizes: {[len(x) for x in stage_cells_sub]}")

    couplings = []
    print("\nComputing biharmonic OT couplings...")
    for i in range(len(stage_cells_sub) - 1):
        cost = compute_biharmonic_cost_matrix(
            stage_cells_sub[i],
            stage_cells_sub[i + 1],
            knn=args.knn,
            n_eig=args.n_eig,
            weight_power=args.weight_power,
        )
        coupling = ot_coupling(cost, n_samples=min(20000, cost.shape[0] * cost.shape[1]))
        couplings.append(coupling)
        print(
            f"  interval {train_labels[i]} -> {train_labels[i + 1]}:"
            f" cost_shape={cost.shape}, sampled_pairs={len(coupling[0])}"
        )

    premetric = BiharmonicPremetric(
        stage_cells_sub,
        knn_graph=args.knn,
        n_eig=args.n_eig,
        weight_power=args.weight_power,
        extension_k=args.extension_k,
        softmax_beta=args.softmax_beta,
        time_cap=args.time_cap,
        grad_norm_floor=args.grad_norm_floor,
        max_drive_scale=args.max_drive_scale,
    )

    results = []
    for i in range(len(stage_cells_sub) - 1):
        result = _debug_interval(
            interval_idx=i,
            src_label=train_labels[i],
            tgt_label=train_labels[i + 1],
            y0_all=stage_cells_sub[i],
            y1_all=stage_cells_sub[i + 1],
            target_stage_idx=i + 1,
            dt_interval=train_stage_times[i + 1] - train_stage_times[i],
            coupling=couplings[i],
            premetric=premetric,
            batch_size=args.batch_size,
            schedule_batch_size=args.schedule_batch_size,
            ode_steps=args.ode_steps,
            grid_steps=args.grid_steps,
            seed=args.seed,
        )
        results.append(result)

    for result in results:
        ratio = result.premetric_norms / result.slerp_norms.clamp(min=1e-8)
        shrink = result.end_distances / result.start_distances.clamp(min=1e-8)
        print("\n" + "=" * 88)
        print(f"INTERVAL {result.label}  (dt={result.dt_interval:.2f})")
        print("=" * 88)
        print(f"SLERP target norms:      {_summary_stats(result.slerp_norms)}")
        print(f"Premetric target norms:  {_summary_stats(result.premetric_norms)}")
        print(f"Norm ratio pre/SLERP:    {_summary_stats(ratio)}")
        print(f"Distance shrink d_t/d_0: {_summary_stats(shrink)}")
        print(f"Start premetric d:       {_summary_stats(result.start_distances)}")
        print(f"End premetric d:         {_summary_stats(result.end_distances)}")
        print(f"Start ||grad_tan||:      {_summary_stats(result.start_grad_norms)}")
        print(f"End ||grad_tan||:        {_summary_stats(result.end_grad_norms)}")
        print(f"Start d/||grad||^2:      {_summary_stats(result.start_drive_scales)}")
        print(f"End d/||grad||^2:        {_summary_stats(result.end_drive_scales)}")
        print(f"Mean schedule MAE vs (1-t): {result.linear_mae:.4f}")
        print("t      mean[d_t/d_0]   std      ideal    stepwise decrease frac")
        for j in range(len(result.ratio_mean)):
            tau = j / (len(result.ratio_mean) - 1)
            ideal = 1.0 - tau
            if j == 0:
                dec = float("nan")
                dec_str = "   n/a"
            else:
                dec = result.decrease_fraction[j - 1].item()
                dec_str = f"{dec:7.3f}"
            print(
                f"{tau:0.3f}    {result.ratio_mean[j].item():0.4f}        "
                f"{result.ratio_std[j].item():0.4f}    {ideal:0.4f}    {dec_str}"
            )


if __name__ == "__main__":
    main()
