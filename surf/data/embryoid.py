"""
Embryoid body dataset loader: load, HVG-filter, and stratified train/val/test split.
"""

import numpy as np
import torch


def load_embryoid_body(path="embryoid_body.h5ad", n_hvg=2000, val_frac=0.1,
                       test_frac=0.1, seed=42, max_cells_per_stage=None,
                       otpfm_split=False, holdout_stages=None):
    """
    Load embryoid body dataset with 5 timepoints and split into train/val/test.

    The dataset has stages: 0-1, 2-3, 4-5, 6-7, 8-9.
    We treat consecutive timepoints as (source, target) pairs for flow matching:
      (0-1 -> 2-3), (2-3 -> 4-5), (4-5 -> 6-7), (6-7 -> 8-9)

    Split: 10% test, 10% val, 80% train -- stratified per timepoint so each
    split has proportional representation from every stage.

    Returns dict with keys 'train', 'val', 'test', each containing:
      - 'stages': ordered list of stage labels
      - 'cells': dict mapping stage -> torch.Tensor of shape (n_cells, n_hvg)
      - 'cell_types': dict mapping stage -> np.array of cell type labels
      - 'transitions': list of (src_stage, tgt_stage) pairs for flow matching
    """
    import anndata as ad
    import scipy.sparse

    print(f"Loading {path}...")
    adata = ad.read_h5ad(path)
    print(f"  Raw: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Subset to highly variable genes if available, otherwise select top HVG
    if "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values
        n_hvg_available = hvg_mask.sum()
        if n_hvg_available >= n_hvg:
            # Take top n_hvg by dispersion norm among HVGs
            hvg_idx = np.where(hvg_mask)[0]
            disp = adata.var["dispersions_norm"].values[hvg_idx]
            top_idx = hvg_idx[np.argsort(disp)[::-1][:n_hvg]]
            adata = adata[:, top_idx]
        else:
            adata = adata[:, hvg_mask]
        print(f"  After HVG filter: {adata.shape[1]} genes")
    else:
        print(f"  No HVG annotation found, using all {adata.shape[1]} genes")

    # Ordered stages
    stages = ["0-1", "2-3", "4-5", "6-7", "8-9"]
    transitions = [(stages[i], stages[i + 1]) for i in range(len(stages) - 1)]

    # Stratified split per timepoint
    rng = np.random.default_rng(seed)
    split_data = {s: {"stages": stages, "cells": {}, "cell_types": {}, "transitions": transitions} for s in ["train", "val", "test"]}

    # OTP-FM-style protocol: held-out stages contribute zero training cells
    # and *all* of their cells to the eval target; training stages contribute
    # *all* of their cells to BOTH train and eval (Rest column compares
    # generated samples against the same cells used to fit the flow).
    held_set = set()
    if otpfm_split and holdout_stages:
        held_set = set(int(i) for i in holdout_stages)

    for stage_idx, stage in enumerate(stages):
        mask = adata.obs["stage"] == stage
        idx = np.where(mask)[0]
        rng.shuffle(idx)

        # Subsample if requested
        if max_cells_per_stage is not None and len(idx) > max_cells_per_stage:
            idx = idx[:max_cells_per_stage]

        if otpfm_split:
            if stage_idx in held_set:
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
            X = adata.X[split_idx]
            if scipy.sparse.issparse(X):
                X = X.toarray()
            split_data[split_name]["cells"][stage] = torch.tensor(X, dtype=torch.float32)
            split_data[split_name]["cell_types"][stage] = adata.obs["cell_type"].values[split_idx]

    # Print summary
    print(f"\n  Split summary (val={val_frac:.0%}, test={test_frac:.0%}):")
    print(f"  {'Stage':<8} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*31}")
    for stage in stages:
        nt = len(split_data["train"]["cells"][stage])
        nv = len(split_data["val"]["cells"][stage])
        ne = len(split_data["test"]["cells"][stage])
        print(f"  {stage:<8} {nt:>7} {nv:>7} {ne:>7}")
    total_tr = sum(len(split_data["train"]["cells"][s]) for s in stages)
    total_va = sum(len(split_data["val"]["cells"][s]) for s in stages)
    total_te = sum(len(split_data["test"]["cells"][s]) for s in stages)
    print(f"  {'Total':<8} {total_tr:>7} {total_va:>7} {total_te:>7}")
    print(f"\n  Transitions: {transitions}")
    print(f"  Gene dim: {adata.shape[1]}")

    return split_data
