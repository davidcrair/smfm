"""
Gulf of Mexico vortex tracking dataset (Shen et al. SB-IRR data).

Nine timepoints x 111 2-D positions per timepoint, capturing the surface
trajectories of Lagrangian particles in a Gulf-of-Mexico vortex. The kNN
union graph across all 9 marginals is *disconnected* into ~6 components
at k=15 (we measured this in scripts/connectivity_diagnostic.py). This
makes GoM the natural stress test for the spectral-OT family: the
Belkin-Niyogi consistency precondition fails, and pure spectral OT is
expected to degrade. The combined Spectral+Euclidean cost is the
candidate fix.

Output dict matches the embryoid / pancreas / paul15 / bonemarrow
contract:

  data['train' | 'val' | 'test'] = {
      'stages': ordered list of stage labels,
      'cells': dict[stage] -> torch.FloatTensor (n_cells, 2),
      'cell_types': dict[stage] -> np.ndarray of stage-name strings,
      'transitions': list of (src, tgt) pairs,
  }

By default we subsample timepoints [0, 2, 4, 6, 8] -> 5 stages so the
results table matches the other 4 datasets. Pass `n_stages=9` to use
all timepoints.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _default_cache_path() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent
    return project_root / "data" / "GoMvortex_data.npy"


def load_gom(path: str | None = None,
             n_stages: int = 5,
             val_frac: float = 0.1,
             test_frac: float = 0.1,
             seed: int = 42,
             max_cells_per_stage: int | None = None,
             otpfm_split: bool = False,
             holdout_stages: list[int] | None = None,
             normalize: bool = True):
    """
    Load Gulf-of-Mexico vortex marginals as a multi-stage trajectory.

    Returns the same {train, val, test} dict structure as the single-cell
    loaders so train.py can dispatch via `data=gom`.
    """
    if path is None or str(path).lower() == "null":
        path = str(_default_cache_path())

    print(f"Loading GoM vortex marginals from {path!r}...")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"GoMvortex_data.npy not found at {path}. "
            f"Re-run scripts/connectivity_diagnostic.py once with internet "
            f"access, or copy the file from /Users/.../Downloads/data.py's cache."
        )
    raw = np.load(path, allow_pickle=True)
    n_avail = len(raw)
    marginals_raw = [arr.astype(np.float32) for arr in raw]
    print(f"  Raw: {n_avail} timepoints, {marginals_raw[0].shape} cells x dims")

    # Subsample stages.
    if n_stages == 9:
        chosen_idx = list(range(9))
    elif n_stages == 5:
        chosen_idx = [0, 2, 4, 6, 8]
    else:
        # uniform sampling
        chosen_idx = list(np.linspace(0, n_avail - 1, n_stages).astype(int))
    marginals = [marginals_raw[i] for i in chosen_idx]
    print(f"  Selected {n_stages} timepoints (indices: {chosen_idx})")

    # Optional global standardization.
    if normalize:
        from sklearn.preprocessing import StandardScaler
        all_data = np.concatenate(marginals, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_data)
        marginals = [scaler.transform(m).astype(np.float32) for m in marginals]
        print(f"  Standardized: mean={all_data.mean():.3f}, std={all_data.std():.3f}")

    stages = [f"t{idx}" for idx in chosen_idx]
    transitions = [(stages[i], stages[i + 1]) for i in range(len(stages) - 1)]

    rng = np.random.default_rng(seed)
    split_data = {
        s: {"stages": stages, "cells": {}, "cell_types": {}, "transitions": transitions}
        for s in ("train", "val", "test")
    }

    held_set: set[int] = set()
    if otpfm_split and holdout_stages:
        held_set = set(int(i) for i in holdout_stages)

    for stage_idx, (stage, cells_arr) in enumerate(zip(stages, marginals)):
        n = len(cells_arr)
        idx = np.arange(n)
        rng.shuffle(idx)
        if max_cells_per_stage is not None and n > max_cells_per_stage:
            idx = idx[:max_cells_per_stage]
            n = max_cells_per_stage

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
            n_test = max(1, int(n * test_frac))
            n_val = max(1, int(n * val_frac))
            test_idx = idx[:n_test]
            val_idx = idx[n_test : n_test + n_val]
            train_idx = idx[n_test + n_val:]

        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            X = cells_arr[split_idx]
            split_data[split_name]["cells"][stage] = torch.tensor(X, dtype=torch.float32)
            split_data[split_name]["cell_types"][stage] = np.array(
                [stage] * len(split_idx), dtype=object
            )

    print(f"\n  Split summary (val={val_frac:.0%}, test={test_frac:.0%}):")
    print(f"  {'Stage':<8} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-' * 32}")
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
    print(f"  Dim: {marginals[0].shape[1]}")

    return split_data
