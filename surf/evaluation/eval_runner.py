"""
Evaluation routines for multi-marginal Fisher Flow models.

Contains:
- eval_chained_metrics: integrate from t=0 to each stage time (chained evaluation)
- eval_per_segment_metrics: integrate each hop independently (per-segment evaluation)
- print_table: formatted results table with optional holdout column tagging
- fmt: format mean +/- std for display
"""

import numpy as np

from surf.evaluation.generation import (
    generate_euclidean_flow,
    generate_fisher_flow,
    generate_sphere_mean_flow,
)
from surf.evaluation.metrics import mmd_rbf
from surf.geometry.sphere import from_orthant, to_compositional


def _predict_to_compositional(pred_state, representation, state_space=None):
    if representation in ("sphere", "sphere_mean"):
        return from_orthant(pred_state)
    if representation == "euclidean":
        decoded = state_space.decode(pred_state) if state_space is not None else pred_state
        return to_compositional(decoded.clamp(min=0.0))
    if representation == "euclidean_raw":
        # OTP-FM-style: data already lives in the eval target's space (raw
        # PC coordinates, etc.). No decoding, no compositional conversion --
        # metrics compare raw model output against raw test cells.
        return pred_state
    raise ValueError(f"Unknown representation: {representation!r}")


def _generate_prediction(model, source, representation, n_steps, t_start, t_end,
                         score_n, a, score_net_sigma, inf_sigma):
    if representation == "sphere":
        return generate_fisher_flow(
            model, source, n_steps=n_steps, t_start=t_start, t_end=t_end,
            score_net=score_n, alpha=a, score_net_sigma=score_net_sigma,
            inf_sigma=inf_sigma,
        )
    if representation == "sphere_mean":
        return generate_sphere_mean_flow(
            model, source, n_steps=n_steps, t_start=t_start, t_end=t_end,
            score_net=score_n, alpha=a, score_net_sigma=score_net_sigma,
            inf_sigma=inf_sigma,
        )
    if representation in ("euclidean", "euclidean_raw"):
        return generate_euclidean_flow(
            model, source, n_steps=n_steps, t_start=t_start, t_end=t_end,
            score_net=score_n, alpha=a, score_net_sigma=score_net_sigma,
            inf_sigma=inf_sigma,
        )
    raise ValueError(f"Unknown representation: {representation!r}")


def eval_chained_metrics(model, test_stage_inputs, test_stage_comp, stage_times, S,
                         score_net_sigma, inf_sigma, metric_fns, score_n=None, a=0.0,
                         representation="sphere", state_space=None,
                         final_only_metrics=None):
    """Chained evaluation: integrate once per target time, score with many metrics.

    Integrates the flow model from the test source cells (stage 0) forward
    to each subsequent stage time and computes one or more metrics in
    compositional space against held-out test cells at that stage.

    Parameters
    ----------
    model : FlowNet
        Trained velocity field model.
    test_stage_sphere : list of tensors
        Per-stage test cells on the sphere.
    test_stage_comp : list of tensors
        Per-stage test cells in compositional space (for MMD comparison).
    stage_times : list of float
        Time coordinates for each stage.
    S : int
        Number of stages.
    score_net_sigma : float
        Sigma for score network evaluation.
    inf_sigma : float
        Noise scale for stochastic integration (0 = deterministic).
    score_n : optional score network
        Score net for inference-time correction.
    a : float
        Score regularization strength at inference.

    Returns dict[str, np.array] mapping metric name -> per-stage values."""
    source = test_stage_inputs[0]
    final_only = set(final_only_metrics or [])
    last_idx = S - 1
    rows = {name: [] for name in metric_fns}
    for i in range(1, S):
        t_target = stage_times[i]
        n_steps = max(10, int(50 * t_target))
        pred_state = _generate_prediction(
            model, source, representation, n_steps, 0.0, t_target,
            score_n, a, score_net_sigma, inf_sigma,
        )
        pred_comp = _predict_to_compositional(pred_state, representation, state_space=state_space)
        for name, fn in metric_fns.items():
            if name in final_only and i != last_idx:
                rows[name].append(np.nan)
            else:
                rows[name].append(fn(pred_comp, test_stage_comp[i]))
    return {name: np.array(vals) for name, vals in rows.items()}


def eval_per_segment_metrics(model, test_stage_inputs, test_stage_comp, stage_times, S,
                             score_net_sigma, inf_sigma, metric_fns, score_n=None, a=0.0,
                             representation="sphere", state_space=None,
                             final_only_metrics=None):
    """Per-segment evaluation: integrate once per hop, score with many metrics.

    For each adjacent stage pair, integrates from test cells at stage i to
    stage i+1 and computes one or more metrics against held-out test cells
    at stage i+1.

    Parameters
    ----------
    model : FlowNet
        Trained velocity field model.
    test_stage_sphere : list of tensors
        Per-stage test cells on the sphere.
    test_stage_comp : list of tensors
        Per-stage test cells in compositional space (for MMD comparison).
    stage_times : list of float
        Time coordinates for each stage.
    S : int
        Number of stages.
    score_net_sigma : float
        Sigma for score network evaluation.
    inf_sigma : float
        Noise scale for stochastic integration (0 = deterministic).
    score_n : optional score network
        Score net for inference-time correction.
    a : float
        Score regularization strength at inference.

    Returns dict[str, np.array] mapping metric name -> per-hop values."""
    final_only = set(final_only_metrics or [])
    last_idx = S - 2  # last hop index in range(S - 1)
    rows = {name: [] for name in metric_fns}
    for i in range(S - 1):
        src_state = test_stage_inputs[i]
        pred_state = _generate_prediction(
            model, src_state, representation, 50, stage_times[i], stage_times[i + 1],
            score_n, a, score_net_sigma, inf_sigma,
        )
        pred_comp = _predict_to_compositional(pred_state, representation, state_space=state_space)
        for name, fn in metric_fns.items():
            if name in final_only and i != last_idx:
                rows[name].append(np.nan)
            else:
                rows[name].append(fn(pred_comp, test_stage_comp[i + 1]))
    return {name: np.array(vals) for name, vals in rows.items()}


def eval_chained(model, test_stage_inputs, test_stage_comp, stage_times, S,
                 score_net_sigma, inf_sigma, score_n=None, a=0.0,
                 representation="sphere", state_space=None):
    """Backward-compatible single-metric chained evaluation wrapper."""
    return eval_chained_metrics(
        model, test_stage_inputs, test_stage_comp, stage_times, S,
        score_net_sigma, inf_sigma, {"MMD^2": mmd_rbf}, score_n=score_n, a=a,
        representation=representation, state_space=state_space,
    )["MMD^2"]


def eval_per_segment(model, test_stage_inputs, test_stage_comp, stage_times, S,
                     score_net_sigma, inf_sigma, score_n=None, a=0.0,
                     representation="sphere", state_space=None):
    """Backward-compatible single-metric per-segment evaluation wrapper."""
    return eval_per_segment_metrics(
        model, test_stage_inputs, test_stage_comp, stage_times, S,
        score_net_sigma, inf_sigma, {"MMD^2": mmd_rbf}, score_n=score_n, a=a,
        representation=representation, state_space=state_space,
    )["MMD^2"]


def fmt(mean, std, n_seeds):
    """Format mean +/- std or just mean for single seed; '-' for missing entries."""
    if np.isnan(mean):
        return "-"
    return f"{mean:.4f}\u00b1{std:.4f}" if n_seeds > 1 else f"{mean:.4f}"


def print_table(title, method_rows, infer_rows, n_seeds, col_names,
                holdout_cols, train_cols, held_set):
    """Print formatted results table with optional holdout column tagging.

    Parameters
    ----------
    title : str
        Table title.
    method_rows : dict
        Maps method name -> list of (n_cols,) arrays, one per seed.
    infer_rows : dict
        Maps alpha value -> list of (n_cols,) arrays for inference-time sweep.
    n_seeds : int
        Number of seeds (controls formatting).
    col_names : list of str
        Column header names.
    holdout_cols : list of int
        Column indices corresponding to held-out stages.
    train_cols : list of int
        Column indices corresponding to training stages.
    held_set : set of int
        Set of held-out stage indices (empty if no holdout).
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    col_w = 16 if n_seeds > 1 else 12

    # Tag each column with TRAIN/HOLD when any holdout is active
    if held_set and holdout_cols and train_cols:
        tagged_names = []
        for j, cn in enumerate(col_names):
            tag = "TR" if j in train_cols else "HO"
            tagged_names.append(f"{cn}[{tag}]")
        header_cols = tagged_names
        show_split = True
    else:
        header_cols = col_names
        show_split = False

    # Size the method-name column to fit the longest name in this table so
    # names like 'MM+SLERP+SquaredSpectral@alpha=0.5' (>32 chars) don't push
    # subsequent columns right.
    method_names = list(method_rows.keys()) + [
        f"MM+SLERP+infer_alpha={a}" for a in infer_rows
        if infer_rows[a] and a != 0.0
    ]
    name_w = max([len("Method")] + [len(m) for m in method_names]) + 2

    extra = ["hold_mean", "train_mean"] if show_split else []
    header = "  " + f"{'Method':<{name_w}}" + "  ".join(
        f"{s:>{col_w}}" for s in header_cols
    ) + f"  {'mean':>{col_w}}" + "".join(f"  {e:>{col_w}}" for e in extra)
    print(header)
    print("  " + "-" * (name_w + (col_w + 2) * (len(header_cols) + 1 + len(extra))))

    def row_str(rows):
        arr = np.stack(rows)  # (n_seeds, n_cols); may contain NaNs
        def finite_mean_std(values):
            values = np.asarray(values)
            if not np.isfinite(values).any():
                return np.nan, np.nan
            return np.nanmean(values), np.nanstd(values)

        mean = np.array([
            np.nanmean(col) if np.isfinite(col).any() else np.nan
            for col in arr.T
        ])
        std = np.array([
            np.nanstd(col) if np.isfinite(col).any() else np.nan
            for col in arr.T
        ])
        total = np.array([
            np.nanmean(row) if np.isfinite(row).any() else np.nan
            for row in arr
        ])
        cols = "  ".join(f"{fmt(mean[i], std[i], n_seeds):>{col_w}}" for i in range(len(mean)))
        total_mean, total_std = finite_mean_std(total)
        total_str = f"{fmt(total_mean, total_std, n_seeds):>{col_w}}"
        s = cols + "  " + total_str
        if show_split:
            hold_arr = arr[:, holdout_cols]
            train_arr = arr[:, train_cols]
            hold_per_seed = np.array([
                np.nanmean(row) if np.isfinite(row).any() else np.nan
                for row in hold_arr
            ])
            train_per_seed = np.array([
                np.nanmean(row) if np.isfinite(row).any() else np.nan
                for row in train_arr
            ])
            hold_mean, hold_std = finite_mean_std(hold_per_seed)
            train_mean, train_std = finite_mean_std(train_per_seed)
            s += "  " + f"{fmt(hold_mean, hold_std, n_seeds):>{col_w}}"
            s += "  " + f"{fmt(train_mean, train_std, n_seeds):>{col_w}}"
        return s

    for mname in method_rows:
        print(f"  {mname:<{name_w}}{row_str(method_rows[mname])}")
    for a_val in sorted(infer_rows):
        if not infer_rows[a_val] or a_val == 0.0:
            continue
        label = f"MM+SLERP+infer_alpha={a_val}"
        print(f"  {label:<{name_w}}{row_str(infer_rows[a_val])}")
