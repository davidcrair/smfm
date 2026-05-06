"""
Pancreatic endocrinogenesis dataset loader (Bastidas-Ponce 2019, via scVelo).

Stage construction follows ``adata.obs["clusters_coarse"]``: a 5-stage
developmental ordering Ductal -> Ngn3 low EP -> Ngn3 high EP -> Pre-endocrine
-> Endocrine. The four endocrine cell types (Alpha/Beta/Delta/Epsilon) collapse
into a single terminal stage, which keeps the marginal count matched to the
embryoid-body setup.

The raw scVelo bundle is integer counts; this loader applies the standard
scanpy preprocessing (normalize_total -> log1p -> HVG selection) so the
output matches the format ``train.py`` expects (log1p-transformed expression
that ``to_compositional`` will re-exponentiate via expm1).
"""

import numpy as np
import torch


# Order matters: it defines the chain of marginals (S=5 stages, 4 transitions).
PANCREAS_STAGE_ORDER = [
    "Ductal",
    "Ngn3 low EP",
    "Ngn3 high EP",
    "Pre-endocrine",
    "Endocrine",
]


def _load_or_download_anndata(path: str | None = None):
    import scvelo as scv

    if path is None or path == "":
        # scVelo's default cache location, relative to cwd
        return scv.datasets.pancreas()
    import os
    if os.path.exists(path):
        import anndata as ad
        return ad.read_h5ad(path)
    # Pass through so scVelo downloads to the requested path
    return scv.datasets.pancreas(file_path=path)


def _preprocess(adata, n_hvg: int):
    """Normalize-total -> log1p -> top-n_hvg HVG slice. Returns sliced AnnData."""
    import scanpy as sc

    # The scVelo bundle's .X is raw integer counts. Defensive-copy so we don't
    # mutate the cached object.
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # scanpy's seurat-flavor HVG selection picks exactly n_top_genes by
    # normalized dispersion, skipping all-zero genes (which have NaN
    # dispersions). Use the boolean mask directly -- argsort over the raw
    # dispersions_norm column would silently surface those NaN entries first
    # and select a column block of all-zero genes.
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_hvg, flavor="seurat", inplace=True,
    )
    adata = adata[:, adata.var["highly_variable"].values].copy()
    return adata


def _coarse_stage(adata):
    """Build the 5-stage label per cell. Endocrine merges Alpha/Beta/Delta/Epsilon."""
    fine = adata.obs["clusters"].astype(str).values
    endocrine = {"Alpha", "Beta", "Delta", "Epsilon"}
    out = np.array([
        "Endocrine" if c in endocrine else c for c in fine
    ])
    return out


def load_pancreas(path: str | None = None, n_hvg: int = 2000, val_frac: float = 0.1,
                  test_frac: float = 0.1, seed: int = 42,
                  max_cells_per_stage: int | None = None,
                  otpfm_split: bool = False,
                  holdout_stages: list[int] | None = None):
    """
    Load Bastidas-Ponce 2019 pancreatic endocrinogenesis as a 5-stage trajectory.

    Output shape mirrors ``load_embryoid_body``: a dict with keys
    ``train`` / ``val`` / ``test``. Each split contains the per-stage cells as
    log1p-transformed HVG expression tensors -- the same representation
    ``train.py`` expects for the sphere-encoded pipeline.
    """
    print(f"Loading pancreas (Bastidas-Ponce 2019) from path={path!r}...")
    adata = _load_or_download_anndata(path)
    print(f"  Raw: {adata.shape[0]} cells x {adata.shape[1]} genes")

    adata = _preprocess(adata, n_hvg=n_hvg)
    print(f"  After log1p + HVG: {adata.shape[1]} genes")

    stage_labels = _coarse_stage(adata)
    stages = PANCREAS_STAGE_ORDER
    transitions = [(stages[i], stages[i + 1]) for i in range(len(stages) - 1)]

    rng = np.random.default_rng(seed)
    split_data = {
        s: {"stages": stages, "cells": {}, "cell_types": {}, "transitions": transitions}
        for s in ("train", "val", "test")
    }

    import scipy.sparse as sp

    held_set = set()
    if otpfm_split and holdout_stages:
        held_set = set(int(i) for i in holdout_stages)

    for stage_idx, stage in enumerate(stages):
        idx = np.where(stage_labels == stage)[0]
        if len(idx) == 0:
            raise RuntimeError(
                f"No cells found for stage {stage!r}. Available labels: "
                f"{sorted(set(stage_labels))}"
            )
        rng.shuffle(idx)
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
            train_idx = idx[n_test + n_val:]

        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            X = adata.X[split_idx]
            if sp.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            split_data[split_name]["cells"][stage] = torch.tensor(X, dtype=torch.float32)
            # Preserve fine cluster identity for diagnostic plots / metrics.
            split_data[split_name]["cell_types"][stage] = (
                adata.obs["clusters"].astype(str).values[split_idx]
            )

    print(f"\n  Split summary (val={val_frac:.0%}, test={test_frac:.0%}):")
    print(f"  {'Stage':<14} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-' * 38}")
    for stage in stages:
        nt = len(split_data["train"]["cells"][stage])
        nv = len(split_data["val"]["cells"][stage])
        ne = len(split_data["test"]["cells"][stage])
        print(f"  {stage:<14} {nt:>7} {nv:>7} {ne:>7}")
    total_tr = sum(len(split_data["train"]["cells"][s]) for s in stages)
    total_va = sum(len(split_data["val"]["cells"][s]) for s in stages)
    total_te = sum(len(split_data["test"]["cells"][s]) for s in stages)
    print(f"  {'Total':<14} {total_tr:>7} {total_va:>7} {total_te:>7}")
    print(f"\n  Transitions: {transitions}")
    print(f"  Gene dim: {adata.shape[1]}")

    return split_data
