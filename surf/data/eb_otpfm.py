"""
Embryoid Body loader matching the OTP-FM / TrajectoryNet protocol.

Loads ``eb_velocity_v5.npz`` (KrishnaswamyLab/TrajectoryNet), takes the top
``pca_dim`` principal components, applies StandardScaler, and returns the
five timepoint marginals in the standard dict shape that ``surf/train.py``
expects (``train``/``val``/``test`` per stage).

This is the *Euclidean* counterpart of ``embryoid.py`` -- the data is already
in low-dim PCA space, so there is no log1p / HVG / compositional / sphere
encoding. Methods that consume this dataset must use the Euclidean trainer
path (``MM+Linear``, ``MM+Linear+SquaredSpectral@alpha=...``, etc.); sphere
methods don't have a coherent interpretation here and should be rejected at
startup.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/"
    "eb_velocity_v5.npz"
)


def download_eb_otpfm_data(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_eb_otpfm(
    path: str = "data/eb_velocity_v5.npz",
    pca_dim: int = 100,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    max_cells_per_stage: int | None = None,
    otpfm_split: bool = False,
    holdout_stages: list[int] | None = None,
):
    """Load EB v5 (TrajectoryNet preprocessing), top-k PCs, StandardScaler.

    Returns the same dict shape as ``load_embryoid_body``: keys
    ``train``/``val``/``test`` each hold ``stages``, ``cells`` (mapping stage
    id -> torch.Tensor), ``cell_types`` (None placeholder; the npz has no
    cell-type labels), and ``transitions``.
    """
    p = Path(path)
    if not p.exists():
        # Caller must download (auto mode harness blocks external fetches).
        raise FileNotFoundError(
            f"{p} not found. Pre-download with:\n"
            f"  python -c \"import urllib.request; "
            f"urllib.request.urlretrieve('{DATA_URL}', '{p}')\""
        )

    print(f"Loading EB v5 (TrajectoryNet preprocessing) from {p}...")
    npz = np.load(p, allow_pickle=True)
    pcs = npz["pcs"][:, :pca_dim].astype(np.float32)
    labels = npz["sample_labels"].astype(np.int64)
    print(f"  Raw: {pcs.shape[0]} cells x {pcs.shape[1]} PCs")

    scaler = StandardScaler()
    pcs = scaler.fit_transform(pcs).astype(np.float32)

    unique_times = sorted(set(labels.tolist()))
    stages = [str(t) for t in unique_times]
    transitions = [(stages[i], stages[i + 1]) for i in range(len(stages) - 1)]

    rng = np.random.default_rng(seed)
    split_data = {
        s: {"stages": stages, "cells": {}, "cell_types": {}, "transitions": transitions}
        for s in ("train", "val", "test")
    }

    held_set = set(int(t) for t in (holdout_stages or []))

    for t, stage_label in zip(unique_times, stages):
        idx = np.where(labels == t)[0]
        rng.shuffle(idx)
        if max_cells_per_stage is not None and len(idx) > max_cells_per_stage:
            idx = idx[:max_cells_per_stage]

        if otpfm_split:
            # OTP-FM protocol: held-out stages contribute zero training cells
            # and all of their cells are the eval target; training stages
            # contribute all of their cells to BOTH train and eval (OTP-FM
            # measures "Rest" against the same cells used for training).
            if int(t) in held_set:
                train_idx = np.empty(0, dtype=idx.dtype)
                val_idx = np.empty(0, dtype=idx.dtype)
                test_idx = idx
            else:
                train_idx = idx
                val_idx = np.empty(0, dtype=idx.dtype)
                test_idx = idx
        else:
            n = len(idx)
            n_test = max(1, int(n * test_frac))
            n_val = max(1, int(n * val_frac))
            test_idx = idx[:n_test]
            val_idx = idx[n_test : n_test + n_val]
            train_idx = idx[n_test + n_val :]

        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            X = pcs[split_idx]
            split_data[split_name]["cells"][stage_label] = torch.tensor(X, dtype=torch.float32)
            # No cell-type metadata in the v5 npz; placeholder so callers
            # accessing this key don't crash.
            split_data[split_name]["cell_types"][stage_label] = np.full(len(split_idx), "", dtype=object)

    print(f"\n  Split summary (val={val_frac:.0%}, test={test_frac:.0%}):")
    print(f"  {'Stage':<6} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-' * 30}")
    for stage in stages:
        nt = len(split_data["train"]["cells"][stage])
        nv = len(split_data["val"]["cells"][stage])
        ne = len(split_data["test"]["cells"][stage])
        print(f"  {stage:<6} {nt:>7} {nv:>7} {ne:>7}")
    print(f"  Stages: {stages}")
    print(f"  Transitions: {transitions}")
    print(f"  Dim: {pcs.shape[1]}")

    return split_data
