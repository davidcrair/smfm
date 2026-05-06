"""
Hydra entry point for multi-marginal Fisher Flow Matching.

Replaces the ``main_mm()`` function from the monolithic ``main.py``.
Run with::

    python -m surf.train                         # defaults
    python -m surf.train +experiment=full_5stage  # preset
    python -m surf.train methods='[MM+SLERP,MM+SI]' training.n_iters=5000
"""
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from surf.runtime import setup, get as get_runtime, log_provenance, _subsample_tensor
from surf.data.embryoid import load_embryoid_body
from surf.data.bonemarrow import load_bonemarrow
from surf.data.gom import load_gom
from surf.data.pancreas import load_pancreas
from surf.data.paul15 import load_paul15
from surf.data.eb_otpfm import load_eb_otpfm
from surf.geometry.sphere import (
    to_orthant, from_orthant, normalize_sphere, to_compositional,
)
from surf.training.score_trainer import (
    train_riemannian_score, train_timed_riemannian_score,
    train_forward_score_nets,
)
from surf.training.flow_trainer import train_multi_marginal_flow
from surf.training.euclidean_flow_trainer import train_multi_marginal_euclidean_flow
from surf.training.rohbeck_mmfm_trainer import train_rohbeck_mmfm
from surf.training.torchcfm_trainer import train_multi_marginal_torchcfm
from surf.training.mean_flow_trainer import train_sphere_endpoint_mean_flow
from surf.training.spectral_jvp_trainer import train_spectral_path_jvp_flow
from surf.training.method_registry import (
    MM_METHOD_NAMES, resolve_method_kwargs,
    needs_learned_score, needs_timed_score, needs_forward_score,
    needs_kde, needs_global_biharmonic, method_representation,
)
from surf.space import build_space
from surf.models.kde_score import estimate_rbf_sigma, _KDEScoreWrapper
from surf.ot.costs import (
    compute_global_biharmonic_embedding, make_global_biharmonic_cost_fn,
)
from surf.evaluation.generation import generate_fisher_flow
from surf.evaluation.metrics import mmd_rbf, mmd_otpfm, fgd, swd
from surf.evaluation.eval_runner import (
    eval_chained_metrics, eval_per_segment_metrics, print_table,
)
from surf.visualization.phate_plots import (
    visualize_endpoints_phate, visualize_endpoints_pca, visualize_pca_grid,
    visualize_trajectories_phate, visualize_nn_distance_histograms,
    visualize_pca_vector_field,
)


class _TeeStream:
    """Write to both the original stream and a log file."""
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file
    def write(self, text):
        self.original.write(text)
        self.log_file.write(text)
        self.log_file.flush()
    def flush(self):
        self.original.flush()
        self.log_file.flush()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Multi-marginal Fisher Flow Matching with OTP-FM eval protocol."""

    # ── 1. Runtime setup ────────────────────────────────────────────────
    import sys
    log_fh = open("run.log", "w")
    sys.stdout = _TeeStream(sys.__stdout__, log_fh)
    sys.stderr = _TeeStream(sys.__stderr__, log_fh)

    rt = setup()
    log_provenance()
    print(OmegaConf.to_yaml(cfg))
    print(f"  Hydra output dir: {os.getcwd()}")

    # ── 2. Load data ────────────────────────────────────────────────────
    data_name = str(cfg.data.get("name", "embryoid"))
    if data_name == "embryoid":
        # Optionally apply the OTP-FM-style per-stage split (full-marginal
        # eval for held-out stages; all cells in train AND test for training
        # stages). Holdout list is derived from the same flags train.py
        # consults below for held_set.
        _otpfm_split = bool(cfg.data.get("otpfm_split", False))
        _embryoid_holdout: list[int] | None = None
        if _otpfm_split:
            if cfg.data.get("otpfm_holdout", False):
                _embryoid_holdout = [1, 3]   # 5 stages -> hold out indices 1, 3
            elif cfg.data.holdout_stages is not None:
                _embryoid_holdout = [
                    int(x) for x in str(cfg.data.holdout_stages).split(",") if x.strip()
                ]
            else:
                _embryoid_holdout = []
        _mcps = cfg.data.get("max_cells_per_stage", None)
        data = load_embryoid_body(
            cfg.data.path,
            n_hvg=cfg.data.n_hvg,
            seed=int(cfg.data.get("seed", 42)),
            max_cells_per_stage=int(_mcps) if _mcps is not None else None,
            otpfm_split=_otpfm_split,
            holdout_stages=_embryoid_holdout,
        )
    elif data_name == "pancreas":
        _otpfm_split = bool(cfg.data.get("otpfm_split", False))
        _pancreas_holdout: list[int] | None = None
        if _otpfm_split:
            if cfg.data.get("otpfm_holdout", False):
                _pancreas_holdout = [1, 3]   # 5 stages -> hold out indices 1, 3
            elif cfg.data.holdout_stages is not None:
                _pancreas_holdout = [
                    int(x) for x in str(cfg.data.holdout_stages).split(",") if x.strip()
                ]
            else:
                _pancreas_holdout = []
        _mcps = cfg.data.get("max_cells_per_stage", None)
        data = load_pancreas(
            path=cfg.data.path,
            n_hvg=cfg.data.n_hvg,
            seed=int(cfg.data.get("seed", 42)),
            max_cells_per_stage=int(_mcps) if _mcps is not None else None,
            otpfm_split=_otpfm_split,
            holdout_stages=_pancreas_holdout,
        )
    elif data_name == "paul15":
        _otpfm_split = bool(cfg.data.get("otpfm_split", False))
        _paul15_holdout: list[int] | None = None
        if _otpfm_split:
            if cfg.data.get("otpfm_holdout", False):
                _paul15_holdout = [1, 3]   # 5 stages -> hold out indices 1, 3
            elif cfg.data.holdout_stages is not None:
                _paul15_holdout = [
                    int(x) for x in str(cfg.data.holdout_stages).split(",") if x.strip()
                ]
            else:
                _paul15_holdout = []
        _mcps = cfg.data.get("max_cells_per_stage", None)
        _path = cfg.data.get("path", None)
        if _path is not None and str(_path).lower() == "null":
            _path = None
        data = load_paul15(
            path=_path,
            n_hvg=cfg.data.n_hvg,
            seed=int(cfg.data.get("seed", 42)),
            max_cells_per_stage=int(_mcps) if _mcps is not None else None,
            otpfm_split=_otpfm_split,
            holdout_stages=_paul15_holdout,
        )
    elif data_name == "gom":
        _otpfm_split = bool(cfg.data.get("otpfm_split", False))
        _gom_holdout: list[int] | None = None
        if _otpfm_split:
            if cfg.data.get("otpfm_holdout", False):
                # 5-stage default holds out indices 1, 3; for 9-stage use 1,3,5,7
                _ns = int(cfg.data.get("n_stages", 5))
                _gom_holdout = [i for i in range(_ns) if i % 2 == 1]
            elif cfg.data.holdout_stages is not None:
                _gom_holdout = [
                    int(x) for x in str(cfg.data.holdout_stages).split(",") if x.strip()
                ]
            else:
                _gom_holdout = []
        _mcps = cfg.data.get("max_cells_per_stage", None)
        _path = cfg.data.get("path", None)
        if _path is not None and str(_path).lower() == "null":
            _path = None
        data = load_gom(
            path=_path,
            n_stages=int(cfg.data.get("n_stages", 5)),
            seed=int(cfg.data.get("seed", 42)),
            max_cells_per_stage=int(_mcps) if _mcps is not None else None,
            otpfm_split=_otpfm_split,
            holdout_stages=_gom_holdout,
        )
    elif data_name == "bonemarrow":
        _otpfm_split = bool(cfg.data.get("otpfm_split", False))
        _bm_holdout: list[int] | None = None
        if _otpfm_split:
            if cfg.data.get("otpfm_holdout", False):
                _bm_holdout = [1, 3]   # 5 stages -> hold out indices 1, 3
            elif cfg.data.holdout_stages is not None:
                _bm_holdout = [
                    int(x) for x in str(cfg.data.holdout_stages).split(",") if x.strip()
                ]
            else:
                _bm_holdout = []
        _mcps = cfg.data.get("max_cells_per_stage", None)
        _path = cfg.data.get("path", None)
        if _path is not None and str(_path).lower() == "null":
            _path = None
        data = load_bonemarrow(
            path=_path,
            n_hvg=cfg.data.n_hvg,
            seed=int(cfg.data.get("seed", 42)),
            max_cells_per_stage=int(_mcps) if _mcps is not None else None,
            otpfm_split=_otpfm_split,
            holdout_stages=_bm_holdout,
        )
    elif data_name == "eb_otpfm":
        # If the dataset uses OTP-FM-style splits, derive the per-stage
        # holdout list from the same flags train.py later inspects, so the
        # loader can put zero training cells in held-out stages.
        _otpfm_split = bool(cfg.data.get("otpfm_split", False))
        _holdout_for_loader: list[int] | None = None
        if _otpfm_split:
            _stages = list(range(5))  # eb_velocity_v5 has exactly 5 timepoints
            if cfg.data.get("otpfm_holdout", False):
                _holdout_for_loader = [i for i in _stages if i % 2 == 1]
            elif cfg.data.holdout_stages is not None:
                _holdout_for_loader = [
                    int(x) for x in str(cfg.data.holdout_stages).split(",") if x.strip()
                ]
            else:
                _holdout_for_loader = []
        data = load_eb_otpfm(
            path=cfg.data.path,
            pca_dim=int(cfg.data.get("pca_dim", 100)),
            seed=int(cfg.data.get("seed", 42)),
            otpfm_split=_otpfm_split,
            holdout_stages=_holdout_for_loader,
        )
    else:
        raise SystemExit(
            f"Unknown data.name={data_name!r}; expected 'embryoid', 'pancreas', 'paul15', 'bonemarrow', 'gom', or 'eb_otpfm'."
        )

    data_representation = str(cfg.data.get("representation", "sphere"))
    if data_representation not in ("sphere", "euclidean_raw"):
        raise SystemExit(
            f"Unknown data.representation={data_representation!r}; expected 'sphere' or 'euclidean_raw'."
        )
    stages = data["train"]["stages"]
    S = len(stages)
    stage_times = [i / (S - 1) for i in range(S)]
    D = data["train"]["cells"][stages[0]].shape[1]

    print(f"\n  Stages: {stages}")
    print(f"  Times:  {[f'{t:.2f}' for t in stage_times]}")
    print(f"  Gene dim: {D}")

    # ── 3. Build training/eval representations ─────────────────────────
    train_stage_log1p = [data["train"]["cells"][s].to(rt.device) for s in stages]
    test_stage_log1p = [data["test"]["cells"][s] for s in stages]
    if data_representation == "sphere":
        train_stage_cells = [
            normalize_sphere(to_orthant(to_compositional(
                data["train"]["cells"][s]
            ))).to(rt.device)
            for s in stages
        ]
        test_stage_comp = [to_compositional(data["test"]["cells"][s]) for s in stages]
        test_stage_sphere = [
            normalize_sphere(to_orthant(tc)).to(rt.device) for tc in test_stage_comp
        ]
    else:
        # euclidean_raw: data already lives in the eval target's space
        # (e.g., 100 PCs from TrajectoryNet). No sphere encoding; eval target
        # is the raw test cells themselves; sphere methods are rejected
        # below since they require compositional input.
        train_stage_cells = train_stage_log1p
        test_stage_comp = test_stage_log1p
        test_stage_sphere = test_stage_log1p

    # ── 4. Resolve holdout pattern ──────────────────────────────────────
    holdout_stages = cfg.data.holdout_stages
    if holdout_stages is not None:
        try:
            held_list = [int(x) for x in str(holdout_stages).split(",") if x.strip() != ""]
        except ValueError:
            raise SystemExit(
                f"data.holdout_stages must be comma-separated ints; got {holdout_stages!r}"
            )
        for i in held_list:
            if not (0 <= i < S):
                raise SystemExit(
                    f"data.holdout_stages index {i} out of range [0, {S - 1}]"
                )
        if 0 in held_list or (S - 1) in held_list:
            raise SystemExit(
                "Cannot hold out the first or last stage (no valid training interval)."
            )
        held_set = set(held_list)
    elif cfg.data.get("otpfm_holdout", False):
        if S < 3:
            raise SystemExit(f"otpfm_holdout needs S>=3 stages; got S={S}")
        held_set = {i for i in range(S) if i % 2 == 1}
    else:
        held_set = set()

    train_idx = [i for i in range(S) if i not in held_set]
    if held_set:
        print(
            f"\n  [holdout] training on stages {[stages[i] for i in train_idx]}"
            f" (times {[f'{stage_times[i]:.2f}' for i in train_idx]})"
        )
        print(
            f"  [holdout] held-out stages:  {[stages[i] for i in sorted(held_set)]}"
            f" (times {[f'{stage_times[i]:.2f}' for i in sorted(held_set)]})"
        )

    train_stage_log1p_sub = [train_stage_log1p[i] for i in train_idx]
    train_stage_cells_sub = [train_stage_cells[i] for i in train_idx]
    train_stage_times_sub = [stage_times[i] for i in train_idx]
    train_idx_set = set(train_idx)

    from surf.training.method_registry import EUCLIDEAN_METHOD_NAMES as euclidean_methods
    use_euclidean = any(m.split("@", 1)[0] in euclidean_methods for m in cfg.methods)
    euclidean_space = None
    train_stage_euclidean_sub = None
    test_stage_euclidean = None
    D_euclidean = None
    if data_representation == "euclidean_raw":
        offending = [m for m in cfg.methods
                     if m.split("@", 1)[0] not in euclidean_methods
                     and m.split("@", 1)[0] not in {"MM+MeanFlow"}]
        if offending:
            raise SystemExit(
                "data.representation='euclidean_raw' only supports Euclidean trainer "
                f"methods ({sorted(euclidean_methods)}); got {offending}."
            )
        # Force a passthrough Euclidean space -- the data is already in the
        # latent the trainer should operate on.
        from surf.space import IdentitySpace
        euclidean_space = IdentitySpace(input_dim=D)
        train_stage_euclidean_sub = [x.to(rt.device) for x in train_stage_log1p_sub]
        test_stage_euclidean = [x.to(rt.device) for x in test_stage_log1p]
        D_euclidean = D
        print(f"  Euclidean training space: {euclidean_space.summary()} (raw)")
    elif use_euclidean:
        euclidean_space = build_space(cfg.space, train_stage_log1p_sub)
        train_stage_euclidean_sub = [euclidean_space.encode(x).to(rt.device) for x in train_stage_log1p_sub]
        test_stage_euclidean = [euclidean_space.encode(x).to(rt.device) for x in test_stage_log1p]
        D_euclidean = euclidean_space.output_dim
        print(f"  Euclidean training space: {euclidean_space.summary()}")

    # ── 5. Column labels for result tables ──────────────────────────────
    chained_col_labels = [f"t={stage_times[i]:.2f}" for i in range(1, S)]
    perseg_col_labels = [
        f"t={stage_times[i]:.2f}\u2192{stage_times[i+1]:.2f}" for i in range(S - 1)
    ]
    if held_set:
        chained_holdout_cols = [
            j for j, i in enumerate(range(1, S)) if i not in train_idx_set
        ]
        chained_train_cols = [
            j for j, i in enumerate(range(1, S)) if i in train_idx_set
        ]
        perseg_holdout_cols = [
            j for j, i in enumerate(range(S - 1))
            if not (i in train_idx_set and (i + 1) in train_idx_set)
        ]
        perseg_train_cols = [
            j for j, i in enumerate(range(S - 1))
            if i in train_idx_set and (i + 1) in train_idx_set
        ]
    else:
        chained_holdout_cols = chained_train_cols = []
        perseg_holdout_cols = perseg_train_cols = []

    # ── 6. Method list ──────────────────────────────────────────────────
    method_list = list(cfg.methods)
    unknown = [m for m in method_list if m.split("@", 1)[0] not in MM_METHOD_NAMES]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Choices: {MM_METHOD_NAMES}")
    print(f"  Methods: {method_list}")
    if any(m in method_list for m in euclidean_methods) and cfg.space.name == "log1p" and cfg.data.n_hvg != 1500:
        print(
            f"  [warn] Euclidean log1p-space baselines were planned as 1500-HVG runs, "
            f"but data.n_hvg={cfg.data.n_hvg}."
        )

    n_seeds = cfg.eval.n_seeds
    score_net_sigma = cfg.training.score.net_sigma
    inf_sigma = cfg.eval.inf_sigma
    print(f"  n_seeds: {n_seeds}")

    # ── 7. Precompute global biharmonic if needed ───────────────────────
    global_biharmonic_cost_fn = None
    global_biharm_embeddings = None
    if needs_global_biharmonic(method_list):
        print("  Computing global biharmonic embedding across all training stages...")
        global_biharm_embeddings = compute_global_biharmonic_embedding(
            train_stage_cells_sub
        )
        if "MM+SLERP+GlobalBiharmonic" in method_list:
            global_biharmonic_cost_fn = make_global_biharmonic_cost_fn(
                train_stage_cells_sub
            )

    mmd_protocol = str(cfg.eval.get("mmd_protocol", "median"))
    if mmd_protocol not in ("median", "otpfm", "both"):
        raise SystemExit(
            f"Unknown eval.mmd_protocol={mmd_protocol!r}; "
            "expected 'median', 'otpfm', or 'both'."
        )
    metric_fns = {}
    if mmd_protocol in ("median", "both"):
        metric_fns["MMD^2"] = mmd_rbf
    if mmd_protocol in ("otpfm", "both"):
        # OTP-FM-style multi-scale MMD^2: 5 RBF kernels at bandwidths
        # `b * 2^i, i in [0..4]` with `b = mean_sqdist / 4`. Apples-to-apples
        # comparable to OTP-FM Table 2 / MMFM published numbers.
        metric_fns["MMD^2_otpfm"] = mmd_otpfm
    # FGD uses scipy.linalg.sqrtm on a D x D matrix which is CPU-only and
    # O(D^3) with a large constant; ~30-60s/call at D=2000. Disable via
    # `eval.compute_fgd=false` for fast sweeps where MMD/SWD are sufficient.
    if cfg.eval.get("compute_fgd", True):
        metric_fns["FGD"] = fgd
    metric_fns["SWD"] = lambda X, Y: swd(X, Y, n_projections=cfg.eval.swd_projections)
    if cfg.eval.get("compute_w2", False):
        from surf.evaluation.metrics import w2 as _w2
        metric_fns["W_2"] = _w2

    # Optionally fit a PCA-K projector on training cells (union over all
    # training stages) and add PCA-projected variants of MMD^2 / MMD^2_otpfm
    # / SWD. Lets ambient log1p HVG runs report OTP-FM-comparable PCA-50
    # numbers without retraining.
    if cfg.eval.get("compute_pca_metrics", False):
        from sklearn.decomposition import PCA as _PCA
        pca_dim = int(cfg.eval.get("pca_metric_dim", 50))
        train_cells_for_pca = np.concatenate(
            [c.detach().cpu().numpy() for c in train_stage_log1p_sub], axis=0
        )
        max_rank = min(train_cells_for_pca.shape)
        if pca_dim > max_rank:
            print(f"  [warn] eval.pca_metric_dim={pca_dim} > rank {max_rank}; clipping")
            pca_dim = max_rank
        _eval_pca = _PCA(n_components=pca_dim, random_state=42, svd_solver="randomized")
        _eval_pca.fit(train_cells_for_pca)
        var_kept = float(_eval_pca.explained_variance_ratio_.sum())
        print(f"  Eval-time PCA-{pca_dim} fit on {train_cells_for_pca.shape[0]} "
              f"training cells (var explained = {var_kept:.3f})")

        def _project(X):
            arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)
            return torch.from_numpy(
                _eval_pca.transform(arr).astype(np.float32, copy=False)
            )

        suffix = f"_pca{pca_dim}"
        if mmd_protocol in ("median", "both"):
            metric_fns[f"MMD^2{suffix}"] = lambda X, Y: mmd_rbf(_project(X), _project(Y))
        if mmd_protocol in ("otpfm", "both"):
            metric_fns[f"MMD^2_otpfm{suffix}"] = lambda X, Y: mmd_otpfm(_project(X), _project(Y))
        metric_fns[f"SWD{suffix}"] = lambda X, Y: swd(
            _project(X), _project(Y), n_projections=cfg.eval.swd_projections
        )
    # Optional list of metrics whose value is computed only at the final
    # timepoint/hop. Other slots are filled with NaN, which downstream
    # nanmean printing skips. Defaults to ["FGD"] to amortize the
    # expensive scipy.linalg.sqrtm call.
    final_only_metrics = list(cfg.eval.get("final_only_metrics", []) or [])

    # ── 8. Per-seed accumulators ────────────────────────────────────────
    chained_by_metric = {metric_name: {} for metric_name in metric_fns}
    perseg_by_metric = {metric_name: {} for metric_name in metric_fns}
    infer_chained_by_metric = {
        metric_name: {a: [] for a in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)}
        for metric_name in metric_fns
    }
    infer_perseg_by_metric = {
        metric_name: {a: [] for a in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)}
        for metric_name in metric_fns
    }

    # ── 9. Per-seed train + eval loop ───────────────────────────────────
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        print("\n" + "#" * 60)
        print(f"# SEED {seed_idx + 1}/{n_seeds}  (base={seed})")
        print("#" * 60)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Train global Riemannian score network (all stages)
        score_net = None
        if needs_learned_score(method_list):
            print("\n" + "=" * 60)
            print("TRAINING RIEMANNIAN SCORE NETWORK (DSM, global)")
            print("=" * 60)
            all_training = torch.cat(train_stage_cells_sub, dim=0)
            print(
                f"  Training on {len(all_training)} cells"
                f" (union of {len(train_stage_cells_sub)} stages)"
            )
            score_net, _ = train_riemannian_score(
                all_training, D,
                n_iters=cfg.training.score.iters,
                batch_size=cfg.training.batch_size,
            )

        # Train per-interval forward-directed score nets
        forward_score_nets = None
        if needs_forward_score(method_list):
            print("\n" + "=" * 60)
            print("TRAINING FORWARD-DIRECTED SCORE NETS (one per interval)")
            print("=" * 60)
            forward_score_nets = train_forward_score_nets(
                train_stage_cells_sub, D,
                n_iters=cfg.training.score.iters,
                batch_size=cfg.training.batch_size,
            )

        # Train time-conditioned score network
        timed_score_net = None
        if needs_timed_score(method_list):
            print("\n" + "=" * 60)
            print("TRAINING TIME-CONDITIONED SCORE NETWORK (DSM)")
            print("=" * 60)
            print(
                f"  Training on {sum(len(c) for c in train_stage_cells_sub)} cells"
                f" across {len(train_stage_cells_sub)} stages (per-stage conditional)"
            )
            timed_score_net, _ = train_timed_riemannian_score(
                train_stage_cells_sub, train_stage_times_sub, D,
                n_iters=cfg.training.score.iters,
                batch_size=cfg.training.batch_size,
            )

        # Precompute KDE score cloud
        kde_wrapper = None
        if needs_kde(method_list):
            print("\n  Building KDE score cloud (for MM+Score_kde baseline)...")
            all_training = torch.cat(train_stage_cells_sub, dim=0)
            kde_cells, _ = _subsample_tensor(all_training, 2000, rng)
            kde_sigma = estimate_rbf_sigma(all_training, k=50)
            print(f"    {len(kde_cells)} cells, sigma={kde_sigma:.4f}")
            kde_wrapper = _KDEScoreWrapper(kde_cells, kde_sigma)

        # Train each multi-marginal method
        models = {}
        model_representations = {}
        model_spaces = {}
        for name in method_list:
            print("\n" + "=" * 60)
            print(f"TRAINING {name}  (seed {seed_idx + 1}/{n_seeds})")
            print("=" * 60)
            representation = method_representation(name)
            kwargs = resolve_method_kwargs(
                name,
                score_net=score_net,
                timed_score_net=timed_score_net,
                forward_score_nets=forward_score_nets,
                kde_wrapper=kde_wrapper,
                global_biharmonic_cost_fn=global_biharmonic_cost_fn,
                global_biharm_embeddings=global_biharm_embeddings,
                stage_cells_for_global=(
                    train_stage_euclidean_sub
                    if data_representation == "euclidean_raw" or representation == "euclidean"
                    else train_stage_cells_sub
                ),
                score_alpha=cfg.training.score.alpha,
                score_net_sigma=score_net_sigma,
                si_sigma=cfg.training.si_sigma,
                biharm_beta=cfg.training.biharm_beta,
                premetric_extension_k=cfg.training.premetric.extension_k,
                premetric_softmax_beta=cfg.training.premetric.softmax_beta,
                premetric_ode_steps=cfg.training.premetric.ode_steps,
                premetric_trajectory_mode=cfg.training.premetric.trajectory_mode,
                premetric_decode_k=cfg.training.premetric.decode_k,
                premetric_decode_beta=cfg.training.premetric.decode_beta,
                premetric_decode_chunk_size=cfg.training.premetric.decode_chunk_size,
                premetric_velocity_fd_eps=cfg.training.premetric.velocity_fd_eps,
                premetric_knn=cfg.training.premetric.knn,
                premetric_n_eig=cfg.training.premetric.n_eig,
                premetric_spectral_family=cfg.training.premetric.spectral_family,
                premetric_weight_power=cfg.training.premetric.weight_power,
                premetric_diffusion_time=cfg.training.premetric.diffusion_time,
                premetric_time_cap=cfg.training.premetric.time_cap,
                premetric_grad_norm_floor=cfg.training.premetric.grad_norm_floor,
                premetric_max_drive_scale=cfg.training.premetric.max_drive_scale,
                graph_smooth_strength=cfg.training.graph_smooth.strength,
                graph_smooth_knn=cfg.training.graph_smooth.knn,
                graph_smooth_batch_edges=cfg.training.graph_smooth.batch_edges,
                graph_smooth_sigma_scale=cfg.training.graph_smooth.sigma_scale,
                phate_n_components=cfg.training.phate.n_components,
                phate_knn=cfg.training.phate.knn,
                phate_decay=cfg.training.phate.decay,
                phate_n_landmark=cfg.training.phate.n_landmark,
                phate_graph_metric=cfg.training.phate.graph_metric,
            )
            if representation == "euclidean":
                framework = kwargs.get("framework", "tong")
                _net_kwargs = dict(
                    flow_net_arch=cfg.training.get("flow_net_arch", "v1"),
                    ema_decay=cfg.training.get("ema_decay", 0.999),
                    grad_clip=cfg.training.get("grad_clip", 1.0),
                    epoch_based=cfg.training.get("epoch_based", False),
                    n_epochs=cfg.training.get("n_epochs", None),
                    warmup_iters=cfg.training.get("warmup_iters", None),
                )
                if framework == "rohbeck":
                    models[name], _ = train_rohbeck_mmfm(
                        train_stage_euclidean_sub, train_stage_times_sub, D_euclidean,
                        n_iters=cfg.training.n_iters,
                        batch_size=cfg.training.batch_size,
                        lr=cfg.training.lr,
                        ot_subsample=cfg.training.ot_subsample,
                        label=name,
                        cost_fn=kwargs.get("cost_fn"),
                        sigma_scale=cfg.training.get("rohbeck_sigma_scale", 1.0),
                        use_ema=cfg.training.get("use_ema", True),
                        **_net_kwargs,
                    )
                elif framework == "torchcfm":
                    models[name], _ = train_multi_marginal_torchcfm(
                        train_stage_euclidean_sub, train_stage_times_sub, D_euclidean,
                        n_iters=cfg.training.n_iters,
                        batch_size=cfg.training.batch_size,
                        lr=cfg.training.lr,
                        ot_subsample=cfg.training.ot_subsample,
                        label=name,
                        cost_fn=kwargs.get("cost_fn"),
                        coupling_mode=kwargs.get("coupling_mode", "argmax_chain"),
                        matcher_type=kwargs.get("torchcfm_matcher", "otcfm"),
                        sigma=kwargs.get("torchcfm_sigma", 0.0),
                        use_ema=cfg.training.get("use_ema", True),
                        **_net_kwargs,
                    )
                else:
                    models[name], _ = train_multi_marginal_euclidean_flow(
                        train_stage_euclidean_sub, train_stage_times_sub, D_euclidean,
                        n_iters=cfg.training.n_iters,
                        batch_size=cfg.training.batch_size,
                        lr=cfg.training.lr,
                        ot_subsample=cfg.training.ot_subsample,
                        label=name,
                        cost_fn=kwargs.get("cost_fn"),
                        coupling_mode=kwargs.get("coupling_mode", "joint"),
                        use_ema=cfg.training.get("use_ema", True),
                        **_net_kwargs,
                    )
            elif representation == "sphere_mean":
                models[name], _ = train_sphere_endpoint_mean_flow(
                    train_stage_cells_sub, train_stage_times_sub, D,
                    n_iters=cfg.training.n_iters,
                    batch_size=cfg.training.batch_size,
                    lr=cfg.training.lr,
                    ot_subsample=cfg.training.ot_subsample,
                    label=name,
                )
            elif name.split("@", 1)[0] == "MM+SpectralPathJVP":
                models[name], _ = train_spectral_path_jvp_flow(
                    train_stage_cells_sub, train_stage_times_sub, D,
                    n_iters=cfg.training.n_iters,
                    batch_size=cfg.training.batch_size,
                    lr=cfg.training.lr,
                    ot_subsample=cfg.training.ot_subsample,
                    label=name,
                    spectral_knn=kwargs["spectral_knn"],
                    spectral_n_eig=kwargs["spectral_n_eig"],
                    spectral_family=kwargs["spectral_family"],
                    spectral_weight_power=kwargs["spectral_weight_power"],
                    spectral_diffusion_time=kwargs["spectral_diffusion_time"],
                    decode_k=cfg.training.spectral_jvp.decode_k,
                    decode_tau=cfg.training.spectral_jvp.decode_tau,
                    decode_chunk_size=cfg.training.spectral_jvp.decode_chunk_size,
                    encoder_iters=cfg.training.spectral_jvp.encoder_iters,
                    encoder_batch_size=cfg.training.spectral_jvp.encoder_batch_size,
                    encoder_lr=cfg.training.spectral_jvp.encoder_lr,
                    encoder_hidden_dim=cfg.training.spectral_jvp.encoder_hidden_dim,
                    encoder_depth=cfg.training.spectral_jvp.encoder_depth,
                    encoder_interp_fraction=cfg.training.spectral_jvp.interp_fraction,
                    velocity_norm_weight=cfg.training.spectral_jvp.velocity_norm_weight,
                    diagnostics=cfg.training.spectral_jvp.diagnostics,
                )
            else:
                models[name], _ = train_multi_marginal_flow(
                    train_stage_cells_sub, train_stage_times_sub, D,
                    n_iters=cfg.training.n_iters,
                    batch_size=cfg.training.batch_size,
                    lr=cfg.training.lr,
                    ot_subsample=cfg.training.ot_subsample,
                    label=name,
                    premetric_diagnostics=cfg.training.premetric.diagnostics.enabled,
                    premetric_diagnostic_samples=cfg.training.premetric.diagnostics.samples,
                    premetric_diagnostic_t_values=list(cfg.training.premetric.diagnostics.t_values),
                    use_ema=cfg.training.get("use_ema", True),
                    **kwargs,
                )
            # When the dataset declares raw_euclidean (data already in the
            # eval target's coordinate system), tag the model representation
            # accordingly so the eval pipeline skips the compositional
            # conversion that the default 'euclidean' branch performs.
            effective_repr = (
                "euclidean_raw"
                if data_representation == "euclidean_raw" and representation == "euclidean"
                else representation
            )
            model_representations[name] = effective_repr
            model_spaces[name] = euclidean_space if representation == "euclidean" else None

        # Stage-width reference (first seed only)
        if seed_idx == 0:
            print(
                "\n  Stage-width reference"
                " (train vs test MMD\u00b2 per stage \u2014 generative floor):"
            )
            for i, s in enumerate(stages):
                # For euclidean_raw datasets the eval target is already the
                # raw cells; for sphere datasets we re-compute compositional
                # vectors here (matches the training/eval space).
                if data_representation == "euclidean_raw":
                    train_comp_s = data["train"]["cells"][s]
                else:
                    train_comp_s = to_compositional(data["train"]["cells"][s])
                test_comp_s = test_stage_comp[i]
                idx = rng.choice(
                    len(train_comp_s),
                    size=min(len(train_comp_s), len(test_comp_s)),
                    replace=False,
                )
                floor = mmd_rbf(train_comp_s[idx], test_comp_s)
                print(f"    {s} (t={stage_times[i]:.2f}): {floor:.4f}")
            print(
                "\n  Do-nothing baseline"
                " (stage-0 test cells vs target stage test cells):"
            )
            source_test = test_stage_comp[0]
            for i in range(1, S):
                mmd = mmd_rbf(source_test, test_stage_comp[i])
                print(f"    {stages[i]} (t={stage_times[i]:.2f}): {mmd:.4f}")

        # Trajectory visualization (first seed only)
        if seed_idx == 0 and cfg.eval.visualize:
            sphere_models = {
                name: model for name, model in models.items()
                if model_representations[name] == "sphere"
            }
            if sphere_models:
                visualize_pca_grid(
                    sphere_models, test_stage_log1p, test_stage_sphere,
                    stage_times, stages,
                )
                visualize_endpoints_pca(
                    sphere_models, test_stage_log1p, test_stage_sphere,
                    stage_times, stages,
                )
                visualize_pca_vector_field(
                    sphere_models, test_stage_log1p, test_stage_sphere,
                    stage_times, stages,
                )
                visualize_endpoints_phate(
                    sphere_models, test_stage_log1p, test_stage_sphere,
                    stage_times, stages,
                )
                visualize_nn_distance_histograms(
                    sphere_models, test_stage_log1p, test_stage_sphere,
                    stage_times, stages,
                )
            else:
                print("  Skipping sphere-only visualizations (no sphere-based methods selected).")
            # Trajectory PHATE handles both sphere and Euclidean models directly.
            visualize_trajectories_phate(
                models, test_stage_log1p, test_stage_sphere,
                stage_times, stages,
                model_representations=model_representations,
                model_spaces=model_spaces,
            )

        # Per-seed eval
        for mname, mmodel in models.items():
            representation = model_representations[mname]
            test_stage_inputs = (
                test_stage_sphere
                if representation in ("sphere", "sphere_mean")
                else test_stage_euclidean
            )
            chained_metrics = eval_chained_metrics(
                mmodel, test_stage_inputs, test_stage_comp,
                stage_times, S, score_net_sigma, inf_sigma, metric_fns,
                representation=representation,
                state_space=model_spaces[mname],
                final_only_metrics=final_only_metrics,
            )
            perseg_metrics = eval_per_segment_metrics(
                mmodel, test_stage_inputs, test_stage_comp,
                stage_times, S, score_net_sigma, inf_sigma, metric_fns,
                representation=representation,
                state_space=model_spaces[mname],
                final_only_metrics=final_only_metrics,
            )
            for metric_name in metric_fns:
                chained_by_metric[metric_name].setdefault(mname, []).append(
                    chained_metrics[metric_name]
                )
                perseg_by_metric[metric_name].setdefault(mname, []).append(
                    perseg_metrics[metric_name]
                )

        # Inference-time score sweep on MM+SLERP (only if score_net available)
        slerp_model = models.get("MM+SLERP")
        if slerp_model is not None and score_net is not None:
            for a in [0.005, 0.01, 0.02, 0.05, 0.1]:
                chained_metrics = eval_chained_metrics(
                    slerp_model, test_stage_sphere, test_stage_comp,
                    stage_times, S, score_net_sigma, inf_sigma, metric_fns,
                    score_n=score_net, a=a,
                    representation="sphere",
                    final_only_metrics=final_only_metrics,
                )
                perseg_metrics = eval_per_segment_metrics(
                    slerp_model, test_stage_sphere, test_stage_comp,
                    stage_times, S, score_net_sigma, inf_sigma, metric_fns,
                    score_n=score_net, a=a,
                    representation="sphere",
                    final_only_metrics=final_only_metrics,
                )
                for metric_name in metric_fns:
                    infer_chained_by_metric[metric_name][a].append(chained_metrics[metric_name])
                    infer_perseg_by_metric[metric_name][a].append(perseg_metrics[metric_name])

    # ── 10. Print tables ────────────────────────────────────────────────
    for metric_name in metric_fns:
        print_table(
            f"CHAINED EVAL [{metric_name}] ({n_seeds} seed{'s' if n_seeds > 1 else ''})",
            chained_by_metric[metric_name], infer_chained_by_metric[metric_name], n_seeds,
            chained_col_labels, chained_holdout_cols, chained_train_cols, held_set,
        )
        print_table(
            f"PER-SEGMENT EVAL [{metric_name}] ({n_seeds} seed{'s' if n_seeds > 1 else ''})",
            perseg_by_metric[metric_name], infer_perseg_by_metric[metric_name], n_seeds,
            perseg_col_labels, perseg_holdout_cols, perseg_train_cols, held_set,
        )

    print("\n  Metric conventions:")
    print("    MMD^2: RBF-kernel MMD in compositional space (lower is better)")
    print("    FGD: Fr\u00e9chet Gaussian Distance in compositional space (lower is better)")
    print(f"    SWD: Sliced Wasserstein Distance with {cfg.eval.swd_projections} projections in compositional space (lower is better)")
    if n_seeds > 1:
        print(f"  Format: mean\u00b1std over {n_seeds} seeds")


if __name__ == "__main__":
    main()
