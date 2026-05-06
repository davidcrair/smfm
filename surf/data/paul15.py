"""
Paul15 (Paul et al. 2015) hematopoietic differentiation as a 5-stage trajectory.

The published dataset has 2,730 cells x 3,451 genes (raw integer counts) labeled
by 19 lineage clusters. We extract the erythroid trajectory (the cleanest linear
sub-tree) and group the published cluster labels into 5 ordered stages by
maturity:

  stage 0:  7MEP                  (megakaryocyte-erythroid progenitor)
  stage 1:  1Ery + 2Ery           (early-committed erythroid)
  stage 2:  3Ery                  (mid erythroid)
  stage 3:  4Ery + 5Ery           (late erythroid)
  stage 4:  6Ery                  (terminal erythroid)

Preprocessing mirrors the pancreas loader: scanpy `normalize_total(target_sum=1e4)`
followed by `log1p` and `highly_variable_genes` selection. Output dict matches
the embryoid/pancreas loader contract:

  data['train' | 'val' | 'test'] = {
      'stages': ordered list of 5 stage labels,
      'cells':  dict[stage] -> torch.FloatTensor (n_cells, n_hvg),
      'cell_types': dict[stage] -> np.ndarray of original cluster labels,
      'transitions': list of (src, tgt) consecutive-stage pairs,
  }
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def _default_cache_path(filename: str = "paul15.h5ad") -> Path:
    """Return an absolute, project-stable cache path that survives Hydra cwd changes."""
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent  # surf/data/paul15.py -> project_root
    return project_root / "data" / "paul15" / filename


PAUL15_STAGE_DEF: list[tuple[str, list[str]]] = [
    ("MEP",      ["7MEP"]),
    ("Ery_1_2",  ["1Ery", "2Ery"]),
    ("Ery_3",    ["3Ery"]),
    ("Ery_4_5",  ["4Ery", "5Ery"]),
    ("Ery_6",    ["6Ery"]),
]
PAUL15_STAGE_ORDER = [s for s, _ in PAUL15_STAGE_DEF]


def _coarse_stage(adata) -> np.ndarray:
    """Map paul15_clusters labels to coarse erythroid-trajectory stage labels.

    Cells whose cluster does not belong to the erythroid trajectory are tagged
    "_other" and dropped by the caller.
    """
    fine = adata.obs["paul15_clusters"].astype(str).values
    cluster_to_stage = {
        c: stage for stage, clusters in PAUL15_STAGE_DEF for c in clusters
    }
    return np.array([cluster_to_stage.get(c, "_other") for c in fine])


def _preprocess(adata, n_hvg: int):
    """Library-size normalize to 10k counts, log1p, and HVG-select."""
    import scanpy as sc

    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"].values].copy()
    return adata


def load_paul15(path: str | None = None, n_hvg: int = 2000, val_frac: float = 0.1,
                test_frac: float = 0.1, seed: int = 42,
                max_cells_per_stage: int | None = None,
                otpfm_split: bool = False,
                holdout_stages: list[int] | None = None):
    """
    Load Paul15 hematopoiesis as a 5-stage erythroid trajectory.

    Parameters mirror ``load_pancreas`` and ``load_embryoid_body`` so the same
    train.py dispatch + experiment configs apply.
    """
    # Default to a project-stable absolute cache path so split sweeps don't
    # re-download under Hydra cwd rotation.
    if path is None or str(path).lower() == "null":
        path = str(_default_cache_path())

    print(f"Loading Paul15 (Paul et al. 2015) hematopoiesis...")
    if Path(path).exists():
        import anndata as ad
        adata = ad.read_h5ad(path)
        print(f"  Loaded cached h5ad: {path}")
    else:
        import scanpy as sc
        adata = sc.datasets.paul15()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)
        print(f"  Downloaded + cached to: {path}")
    print(f"  Raw: {adata.shape[0]} cells x {adata.shape[1]} genes")

    adata = _preprocess(adata, n_hvg=n_hvg)
    print(f"  After normalize_total + log1p + HVG: {adata.shape[1]} genes")

    stage_labels = _coarse_stage(adata)
    erythroid_mask = stage_labels != "_other"
    adata = adata[erythroid_mask].copy()
    stage_labels = stage_labels[erythroid_mask]
    print(f"  Erythroid trajectory: {adata.shape[0]} cells "
          f"(dropped {(~erythroid_mask).sum()} non-erythroid cells)")

    stages = PAUL15_STAGE_ORDER
    transitions = [(stages[i], stages[i + 1]) for i in range(len(stages) - 1)]

    rng = np.random.default_rng(seed)
    split_data = {
        s: {"stages": stages, "cells": {}, "cell_types": {}, "transitions": transitions}
        for s in ("train", "val", "test")
    }

    held_set: set[int] = set()
    if otpfm_split and holdout_stages:
        held_set = set(int(i) for i in holdout_stages)

    for stage_idx, stage in enumerate(stages):
        idx = np.where(stage_labels == stage)[0]
        if len(idx) == 0:
            raise RuntimeError(
                f"No cells found for stage {stage!r}. "
                f"Available stage labels: {sorted(set(stage_labels))}"
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
            split_data[split_name]["cell_types"][stage] = (
                adata.obs["paul15_clusters"].astype(str).values[split_idx]
            )

    print(f"\n  Split summary (val={val_frac:.0%}, test={test_frac:.0%}):")
    print(f"  {'Stage':<10} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-' * 34}")
    for stage in stages:
        nt = len(split_data["train"]["cells"][stage])
        nv = len(split_data["val"]["cells"][stage])
        ne = len(split_data["test"]["cells"][stage])
        print(f"  {stage:<10} {nt:>7} {nv:>7} {ne:>7}")
    total_tr = sum(len(split_data["train"]["cells"][s]) for s in stages)
    total_va = sum(len(split_data["val"]["cells"][s]) for s in stages)
    total_te = sum(len(split_data["test"]["cells"][s]) for s in stages)
    print(f"  {'Total':<10} {total_tr:>7} {total_va:>7} {total_te:>7}")
    print(f"\n  Transitions: {transitions}")
    print(f"  Gene dim: {adata.shape[1]}")

    return split_data
