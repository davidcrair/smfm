"""Spectral-path Fisher flow training with a Laplacian-space JVP loss."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from surf.geometry.spectral_path import SpectralKNNDecoder
from surf.models.flow_net import FlowNet
from surf.models.spectral_encoder import SpectralEncoderNet
from surf.ot.costs import compute_global_biharmonic_embedding
from surf.ot.coupling import ot_coupling
from surf.runtime import get as get_runtime


def _safe_label(label):
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


def _spectral_cost(e0, e1):
    from scipy.spatial.distance import cdist

    return cdist(e0, e1, metric="sqeuclidean").astype(np.float32, copy=False)


def _subsample_stages(stage_cells, stage_times, ot_subsample):
    rng = np.random.default_rng(0)
    stage_cells_sub = []
    print(f"  Computing spectral OT couplings for {len(stage_cells) - 1} adjacent pairs...")
    for i, cells in enumerate(stage_cells):
        if len(cells) > ot_subsample:
            idx = rng.choice(len(cells), size=ot_subsample, replace=False)
            stage_cells_sub.append(cells[idx])
        else:
            stage_cells_sub.append(cells)
        print(f"    stage {i}: {len(stage_cells_sub[i])} cells (t={stage_times[i]:.2f})")
    return stage_cells_sub


def _sample_spectral_pairs(
    *,
    stage_embeddings,
    adj_couplings_device,
    stage_times,
    batch_size,
    device,
):
    interval_idx = np.random.randint(0, len(stage_embeddings) - 1)
    ot_src_i, ot_tgt_i = adj_couplings_device[interval_idx]
    pair_idx = torch.randint(len(ot_src_i), (batch_size,), device=device)
    src_idx = ot_src_i[pair_idx]
    tgt_idx = ot_tgt_i[pair_idx]

    t_start = float(stage_times[interval_idx])
    t_end = float(stage_times[interval_idx + 1])
    dt_interval = t_end - t_start
    s_local = torch.rand(batch_size, 1, device=device)
    t_global = t_start + dt_interval * s_local

    z0 = stage_embeddings[interval_idx][src_idx]
    z1 = stage_embeddings[interval_idx + 1][tgt_idx]
    z_t = (1.0 - s_local) * z0 + s_local * z1
    dz_true = (z1 - z0) / dt_interval
    return interval_idx, z_t, dz_true, t_global


def _pretrain_encoder(
    *,
    encoder,
    decoder,
    all_cells,
    all_embeddings,
    stage_embeddings,
    adj_couplings_device,
    stage_times,
    n_iters,
    batch_size,
    lr,
    interp_fraction,
    label,
):
    rt = get_runtime()
    if n_iters <= 0:
        return []

    opt = torch.optim.Adam(encoder.parameters(), lr=lr, foreach=True)
    losses = []
    n_interp = int(round(batch_size * float(interp_fraction)))
    n_interp = min(max(n_interp, 0), batch_size)
    n_real = batch_size - n_interp
    if n_real == 0 and n_interp == 0:
        raise ValueError("encoder batch_size must be positive")

    for it in range(n_iters):
        x_parts = []
        z_parts = []

        if n_real > 0:
            real_idx = torch.randint(len(all_cells), (n_real,), device=rt.device)
            x_parts.append(all_cells[real_idx])
            z_parts.append(all_embeddings[real_idx])

        if n_interp > 0:
            _, z_t, _, _ = _sample_spectral_pairs(
                stage_embeddings=stage_embeddings,
                adj_couplings_device=adj_couplings_device,
                stage_times=stage_times,
                batch_size=n_interp,
                device=rt.device,
            )
            x_t = decoder.decode(z_t)
            x_parts.append(x_t)
            z_parts.append(z_t)

        x_batch = torch.cat(x_parts, dim=0).detach()
        z_batch = torch.cat(z_parts, dim=0).detach()

        pred = encoder(x_batch)
        loss = ((pred - z_batch) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        opt.step()

        if it % 200 == 0 or it == n_iters - 1:
            losses.append(float(loss.detach().cpu()))
            print(f"  {label:22s} encoder iter {it:4d}  loss={loss.item():.6f}")

    return losses


def _encoder_diagnostics(
    *,
    encoder,
    decoder,
    all_cells,
    all_embeddings,
    stage_embeddings,
    adj_couplings_device,
    stage_times,
    n_samples=1024,
):
    rt = get_runtime()
    n_real = min(int(n_samples), len(all_cells))
    with torch.no_grad():
        real_idx = torch.randint(len(all_cells), (n_real,), device=rt.device)
        real_pred = encoder(all_cells[real_idx])
        real_mse = ((real_pred - all_embeddings[real_idx]) ** 2).mean()

        _, z_t, _, _ = _sample_spectral_pairs(
            stage_embeddings=stage_embeddings,
            adj_couplings_device=adj_couplings_device,
            stage_times=stage_times,
            batch_size=n_real,
            device=rt.device,
        )
        x_t = decoder.decode(z_t)
        interp_pred = encoder(x_t)
        interp_mse = ((interp_pred - z_t) ** 2).mean()
        decode_diag = decoder.diagnostics(z_t, x_t)

    diagnostics = {
        "encoder_real_mse": float(real_mse.detach().cpu()),
        "encoder_interpolant_mse": float(interp_mse.detach().cpu()),
        **decode_diag,
    }
    print(
        "  Spectral encoder diagnostics: "
        f"real_mse={diagnostics['encoder_real_mse']:.6g}, "
        f"interp_mse={diagnostics['encoder_interpolant_mse']:.6g}, "
        f"decode_tau={diagnostics['decode_tau']:.6g}, "
        f"min_coord={diagnostics['min_coordinate']:.3e}"
    )
    return diagnostics


def train_spectral_path_jvp_flow(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    label="MM+SpectralPathJVP",
    ot_subsample=2000,
    spectral_knn=15,
    spectral_n_eig=50,
    spectral_family="power",
    spectral_weight_power=0.0,
    spectral_diffusion_time=1.0,
    decode_k=64,
    decode_tau="auto",
    decode_chunk_size=64,
    encoder_iters=2000,
    encoder_batch_size=None,
    encoder_lr=3e-4,
    encoder_hidden_dim=256,
    encoder_depth=4,
    encoder_interp_fraction=0.5,
    velocity_norm_weight=1e-4,
    diagnostics=True,
):
    """Train an ambient Fisher-sphere flow using only spectral-space velocity loss."""
    rt = get_runtime()

    assert len(stage_cells) == len(stage_times)
    S = len(stage_cells)
    assert S >= 2
    for i in range(S - 1):
        assert stage_times[i] < stage_times[i + 1]

    stage_cells_sub = _subsample_stages(stage_cells, stage_times, ot_subsample)
    print(
        "  Spectral path JVP: "
        f"family={spectral_family}, weight_power={spectral_weight_power:.3f}, "
        f"diffusion_time={spectral_diffusion_time:.3f}, knn={spectral_knn}, "
        f"n_eig={spectral_n_eig}, decode_k={decode_k}, decode_tau={decode_tau}, "
        f"v_norm={velocity_norm_weight:.2g}"
    )

    emb_per_stage = compute_global_biharmonic_embedding(
        stage_cells_sub,
        knn=spectral_knn,
        n_eig=spectral_n_eig,
        spectral_family=spectral_family,
        weight_power=spectral_weight_power,
        diffusion_time=spectral_diffusion_time,
    )
    stage_embeddings_np = [emb.astype("float32", copy=False) for emb in emb_per_stage]
    stage_embeddings = [
        torch.from_numpy(emb).to(rt.device) for emb in stage_embeddings_np
    ]
    all_cells = torch.cat(stage_cells_sub, dim=0)
    all_embeddings = torch.cat(stage_embeddings, dim=0)
    K = int(all_embeddings.shape[1])
    if K <= 0:
        raise ValueError("spectral embedding has no nontrivial dimensions")

    decoder = SpectralKNNDecoder(
        all_embeddings,
        all_cells,
        k=decode_k,
        tau=decode_tau,
        chunk_size=decode_chunk_size,
    )

    adj_couplings = []
    for i in range(S - 1):
        cost = _spectral_cost(stage_embeddings_np[i], stage_embeddings_np[i + 1])
        n_pool = min(20000, len(stage_cells_sub[i]) * len(stage_cells_sub[i + 1]))
        ot_src, ot_tgt = ot_coupling(cost, n_pool)
        adj_couplings.append((ot_src, ot_tgt))

    adj_couplings_device = [
        (
            torch.as_tensor(ot_src, device=rt.device, dtype=torch.long),
            torch.as_tensor(ot_tgt, device=rt.device, dtype=torch.long),
        )
        for ot_src, ot_tgt in adj_couplings
    ]

    encoder = SpectralEncoderNet(
        D,
        K,
        hidden=encoder_hidden_dim,
        depth=encoder_depth,
    ).to(rt.device)
    encoder_batch_size = int(encoder_batch_size or batch_size)
    encoder_losses = _pretrain_encoder(
        encoder=encoder,
        decoder=decoder,
        all_cells=all_cells,
        all_embeddings=all_embeddings,
        stage_embeddings=stage_embeddings,
        adj_couplings_device=adj_couplings_device,
        stage_times=stage_times,
        n_iters=int(encoder_iters),
        batch_size=encoder_batch_size,
        lr=encoder_lr,
        interp_fraction=encoder_interp_fraction,
        label=label,
    )

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    encoder_diag = {}
    if diagnostics:
        encoder_diag = _encoder_diagnostics(
            encoder=encoder,
            decoder=decoder,
            all_cells=all_cells,
            all_embeddings=all_embeddings,
            stage_embeddings=stage_embeddings,
            adj_couplings_device=adj_couplings_device,
            stage_times=stage_times,
        )
        safe = _safe_label(label)
        try:
            Path(f"spectral_jvp_diagnostics_{safe}.json").write_text(
                json.dumps(encoder_diag, indent=2)
            )
        except OSError as e:
            print(f"  [warn] could not write spectral JVP diagnostics JSON: {e}")

    model = FlowNet(D).to(rt.device)
    if rt.use_amp:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"  torch.compile failed ({e}); running uncompiled")
    opt = torch.optim.Adam(model.parameters(), lr=lr, foreach=True)
    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    warmup_iters = min(500, n_iters // 10)

    losses = []
    for it in range(n_iters):
        _, z_t, dz_true, t_global = _sample_spectral_pairs(
            stage_embeddings=stage_embeddings,
            adj_couplings_device=adj_couplings_device,
            stage_times=stage_times,
            batch_size=batch_size,
            device=rt.device,
        )
        x_t = decoder.decode(z_t).detach()
        dz_true = dz_true.detach()

        with torch.autocast(device_type=rt.device.type, dtype=rt.amp_dtype, enabled=rt.use_amp):
            v_raw = model(x_t, t_global.squeeze(1))
            v_pred = v_raw - (v_raw * x_t).sum(dim=-1, keepdim=True) * x_t
            _, dz_pred = torch.func.jvp(encoder, (x_t,), (v_pred,))
            spectral_loss = ((dz_pred - dz_true) ** 2).mean()
            norm_loss = v_pred.square().sum(dim=-1).mean()
            loss = spectral_loss + float(velocity_norm_weight) * norm_loss

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
            losses.append(float(loss.detach().cpu()))
            print(
                f"  {label:22s} iter {it:4d}  loss={loss.item():.6f}  "
                f"spectral={spectral_loss.item():.6f}  vnorm={norm_loss.item():.6f}"
            )

    model.load_state_dict(ema_state)
    model.spectral_encoder = encoder
    model.spectral_jvp_diagnostics = encoder_diag
    model.spectral_encoder_losses = encoder_losses
    return model, losses
