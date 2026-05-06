"""
Method registry for multi-marginal Fisher Flow experiments.

Maps method names to the kwargs expected by train_multi_marginal_flow,
and provides need-detection helpers so the orchestrator can skip
expensive precomputations (score nets, biharmonic embeddings) when no
selected method requires them.
"""

from surf.ot.costs import (
    compute_biharmonic_cost_matrix,
    compute_euclidean_cost_matrix,
    compute_random_cost_matrix,
    make_biharmonic_cost_fn,
    make_spectral_cost_fn,
    make_phate_cost_fn,
)

MM_METHOD_NAMES = [
    "MM+MeanFlow",
    "MM+Linear",
    "MM+MMFM-OTPFM",
    "MM+MMFM-Rohbeck",
    "MM+MMFM-Rohbeck+SquaredSpectral",
    "MM+Linear+Biharmonic",
    "MM+Linear+SquaredSpectral",
    "MM+Linear+GlobalSpectral",
    "MM+MMFM-Rohbeck+GlobalSpectral",
    "MM+Linear+Random",
    # torchcfm-backed adapters: per-pair (t, x_t, u_t) sampling is
    # delegated to Tong et al.'s torchcfm matcher classes. The cost
    # matrix (Euclidean / spectral / ...) is still computed by us.
    "MM+MMFM-TorchCFM",
    "MM+MMFM-TorchCFM-ICFM",
    "MM+MMFM-TorchCFM-SBCFM",
    "MM+MMFM-TorchCFM-TCFM",
    "MM+MMFM-TorchCFM-VPCFM",
    "MM+MMFM-TorchCFM+SquaredSpectral",
    "MM+MMFM-TorchCFM+GlobalSpectral",
    "MM+SLERP+Random",
    "MM+SLERP", "MM+SLERP+GraphSmooth",
    "MM+Score_kde", "MM+Score_learned", "MM+Score_forward",
    "MM+SLERP+PHATE", "MM+SLERP+PHATE-Euclidean",
    "MM+SLERP+PHATE-SphereArc",
    "MM+SLERP+SquaredSpectral",
    "MM+SpectralPathJVP",
    "MM+SLERP+Biharmonic", "MM+SLERP+GlobalBiharmonic",
    "MM+PremetricBiharmonic-SphereOT",
    "MM+PremetricBiharmonic-SpectralOT",
    "MM+Score_learned+PHATE", "MM+Score_learned+Biharmonic",
    "MM+Score_timed", "MM+Score_timed+Biharmonic",
    "MM+SI", "MM+SI+Biharmonic",
    "MM+BiharmonicVel", "MM+BiharmonicWaypoint",
    "MM+PremetricBiharmonic",
]

EUCLIDEAN_METHOD_NAMES = {
    "MM+Linear", "MM+MMFM-OTPFM",
    "MM+MMFM-Rohbeck", "MM+MMFM-Rohbeck+SquaredSpectral",
    "MM+MMFM-Rohbeck+GlobalSpectral",
    "MM+Linear+Biharmonic", "MM+Linear+SquaredSpectral",
    "MM+Linear+GlobalSpectral",
    "MM+Linear+Random",
    "MM+MMFM-TorchCFM",
    "MM+MMFM-TorchCFM-ICFM",
    "MM+MMFM-TorchCFM-SBCFM",
    "MM+MMFM-TorchCFM-TCFM",
    "MM+MMFM-TorchCFM-VPCFM",
    "MM+MMFM-TorchCFM+SquaredSpectral",
    "MM+MMFM-TorchCFM+GlobalSpectral",
}
SPHERE_MEAN_METHOD_NAMES = {"MM+MeanFlow"}


# Parametric method overrides: append "@key=val,key=val" to a base method name
# (e.g. "MM+SLERP+SquaredSpectral@alpha=1.5") to sweep its spectral parameters
# inside one run without re-loading data per Hydra multirun cell.
_PARAM_KEY_TO_KW = {
    # alpha is the canonical spectral exponent in w(lambda)=lambda^(-alpha);
    # the code's weight_power maps via w_effective = lambda^(-2*weight_power).
    "alpha": ("premetric_weight_power", lambda v: float(v) / 2.0),
    "weight_power": ("premetric_weight_power", float),
    "tau": ("premetric_diffusion_time", float),
    "diffusion_time": ("premetric_diffusion_time", float),
    "family": ("premetric_spectral_family", str),
    "n_eig": ("premetric_n_eig", int),
    "knn": ("premetric_knn", int),
    # Graph augmentation: low-weight Gaussian-Euclidean edges added to the
    # kNN graph so the union stays connected when the kNN structure
    # fragments. aug=0.0 (default) preserves the original kNN-only behavior.
    "aug": ("premetric_augmentation_eps", float),
    "aug_sigma": ("premetric_augmentation_sigma_scale", float),
    # Convex blend with Euclidean cost: blend=0 -> pure spectral,
    # blend=1 -> pure Euclidean, in [0,1] in between (mean-normalized).
    # Used to test on disconnected-graph datasets like GoM.
    "blend": ("euclidean_blend", float),
    # torchcfm matcher selector + sigma override (only used by the
    # MM+MMFM-TorchCFM* methods, ignored otherwise).
    "matcher": ("torchcfm_matcher", str),
    "sigma": ("torchcfm_sigma", float),
}


def _split_method_name(name):
    """Return (base_name, overrides_dict) for a possibly-parametric method name."""
    if "@" not in name:
        return name, {}
    base, suffix = name.split("@", 1)
    overrides = {}
    for piece in suffix.split(","):
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Bad method param {piece!r} in {name!r}; expected key=val")
        key, val = piece.split("=", 1)
        if key not in _PARAM_KEY_TO_KW:
            raise ValueError(
                f"Unknown method param {key!r} in {name!r}. "
                f"Known keys: {sorted(_PARAM_KEY_TO_KW)}"
            )
        kw, caster = _PARAM_KEY_TO_KW[key]
        overrides[kw] = caster(val)
    return base, overrides


def method_representation(name):
    """Return the representation family used by *name*."""
    base, _ = _split_method_name(name)
    if base in EUCLIDEAN_METHOD_NAMES:
        return "euclidean"
    if base in SPHERE_MEAN_METHOD_NAMES:
        return "sphere_mean"
    return "sphere"


def _base_names(method_list):
    return {m.split("@", 1)[0] for m in method_list}


def needs_learned_score(method_list):
    """True if any method in *method_list* requires a global learned score net."""
    bases = _base_names(method_list)
    return bool(bases & {"MM+Score_learned", "MM+Score_learned+PHATE", "MM+Score_learned+Biharmonic"})


def needs_timed_score(method_list):
    """True if any method requires a time-conditioned score net."""
    return bool(_base_names(method_list) & {"MM+Score_timed", "MM+Score_timed+Biharmonic"})


def needs_forward_score(method_list):
    """True if any method requires per-interval forward score nets."""
    return "MM+Score_forward" in _base_names(method_list)


def needs_kde(method_list):
    """True if any method requires a KDE score wrapper."""
    return "MM+Score_kde" in _base_names(method_list)


def needs_global_biharmonic(method_list):
    """True if any method requires global biharmonic embeddings."""
    return bool(_base_names(method_list) & {"MM+SLERP+GlobalBiharmonic", "MM+BiharmonicVel", "MM+BiharmonicWaypoint"})


def resolve_method_kwargs(name, *, score_net=None, timed_score_net=None,
                          forward_score_nets=None, kde_wrapper=None,
                          global_biharmonic_cost_fn=None,
                          global_biharm_embeddings=None,
                          stage_cells_for_global=None,
                          score_alpha=0.005, score_net_sigma=0.7,
                          si_sigma=0.05, biharm_beta=0.3,
                          premetric_extension_k=64,
                          premetric_softmax_beta=10.0,
                          premetric_ode_steps=16,
                          premetric_trajectory_mode="spectral_decode",
                          premetric_decode_k=64,
                          premetric_decode_beta=10.0,
                          premetric_decode_chunk_size=64,
                          premetric_velocity_fd_eps=0.02,
                          premetric_knn=15,
                          premetric_n_eig=50,
                          premetric_spectral_family="power",
                          premetric_weight_power=0.5,
                          premetric_diffusion_time=1.0,
                          premetric_augmentation_eps=0.0,
                          premetric_augmentation_sigma_scale=2.0,
                          premetric_time_cap=0.9,
                          premetric_grad_norm_floor=0.05,
                          premetric_max_drive_scale=50.0,
                          graph_smooth_strength=0.01,
                          graph_smooth_knn=15,
                          graph_smooth_batch_edges=512,
                          graph_smooth_sigma_scale=1.0,
                          phate_n_components=10,
                          phate_knn=15,
                          phate_decay=40,
                          phate_n_landmark=2000,
                          phate_graph_metric="euclidean"):
    """Return the keyword arguments for ``train_multi_marginal_flow`` given a method *name*.

    Parameters
    ----------
    name : str
        One of :data:`MM_METHOD_NAMES`.
    score_net : optional
        Global learned Riemannian score network (or None).
    timed_score_net : optional
        Time-conditioned score network (or None).
    forward_score_nets : optional
        List of per-interval forward score nets (or None).
    kde_wrapper : optional
        ``_KDEScoreWrapper`` instance (or None).
    global_biharmonic_cost_fn : optional
        Precomputed global biharmonic cost function (or None).
    global_biharm_embeddings : optional
        Precomputed global biharmonic embeddings (or None).
    score_alpha : float
        Score regularization strength (``alpha`` kwarg).
    score_net_sigma : float
        Sigma used when evaluating the score network.
    si_sigma : float
        Stochastic interpolant noise scale.
    biharm_beta : float
        Biharmonic velocity blending coefficient.

    Returns
    -------
    dict
        Keyword arguments to unpack into ``train_multi_marginal_flow``.
    """
    name, _overrides = _split_method_name(name)
    premetric_weight_power = _overrides.get("premetric_weight_power", premetric_weight_power)
    premetric_diffusion_time = _overrides.get("premetric_diffusion_time", premetric_diffusion_time)
    premetric_spectral_family = _overrides.get("premetric_spectral_family", premetric_spectral_family)
    premetric_n_eig = _overrides.get("premetric_n_eig", premetric_n_eig)
    premetric_knn = _overrides.get("premetric_knn", premetric_knn)
    premetric_augmentation_eps = _overrides.get(
        "premetric_augmentation_eps", premetric_augmentation_eps,
    )
    premetric_augmentation_sigma_scale = _overrides.get(
        "premetric_augmentation_sigma_scale", premetric_augmentation_sigma_scale,
    )
    # Convex blend with Euclidean cost (default 0 -> pure spectral, no change).
    euclidean_blend = _overrides.get("euclidean_blend", 0.0)
    # torchcfm overrides
    torchcfm_matcher = _overrides.get("torchcfm_matcher", None)
    torchcfm_sigma = _overrides.get("torchcfm_sigma", None)

    if name == "MM+MeanFlow":
        return {}

    if name == "MM+Linear":
        return {"cost_fn": compute_euclidean_cost_matrix}

    if name == "MM+MMFM-OTPFM":
        # MMFM baseline matching the OTP-FM repo convention: argmax-chain
        # OT alignment across all training stages (same algorithm as
        # MM+Linear, but the coupling is a deterministic per-source-cell
        # trajectory rather than fresh joint sampling each iter). See
        # surf/training/euclidean_flow_trainer.py for the chain build.
        return {
            "cost_fn": compute_euclidean_cost_matrix,
            "coupling_mode": "argmax_chain",
        }

    if name == "MM+MMFM-Rohbeck":
        # Rohbeck et al. ICLR 2025 MMFM framework: MMOT chains across
        # all observed marginals + natural-cubic-spline interpolant
        # μ_t(z) + time-dependent variance σ_t(z). OT cost defaults to
        # squared-Euclidean. This is the "true" MMFM baseline that
        # OTP-FM Table 2 reports against.
        return {
            "cost_fn": compute_euclidean_cost_matrix,
            "framework": "rohbeck",
        }

    # ------------------------------------------------------------------
    # torchcfm-backed adapters. The matcher class is selected via the
    # method-name suffix (or @matcher=... override); coupling_mode is
    # always "argmax_chain" so the comparison to MM+MMFM-OTPFM and
    # MM+MMFM-Rohbeck is apples-to-apples on the OT scaffold.
    # ------------------------------------------------------------------
    _torchcfm_default_matchers = {
        "MM+MMFM-TorchCFM":      "otcfm",
        "MM+MMFM-TorchCFM-ICFM": "icfm",
        "MM+MMFM-TorchCFM-SBCFM": "sbcfm",
        "MM+MMFM-TorchCFM-TCFM": "tcfm",
        "MM+MMFM-TorchCFM-VPCFM": "vpcfm",
        "MM+MMFM-TorchCFM+SquaredSpectral": "otcfm",
        "MM+MMFM-TorchCFM+GlobalSpectral":  "otcfm",
    }
    if name in _torchcfm_default_matchers:
        chosen_matcher = torchcfm_matcher or _torchcfm_default_matchers[name]
        # Default sigma matches the matcher's torchcfm spec: 0 for sharp
        # paths (I-CFM/OT-CFM/T-CFM/VP-CFM); 0.1 for SB-CFM (must be > 0).
        if torchcfm_sigma is None:
            chosen_sigma = 0.1 if chosen_matcher.lower().startswith("sb") else 0.0
        else:
            chosen_sigma = torchcfm_sigma

        if name == "MM+MMFM-TorchCFM+SquaredSpectral":
            if euclidean_blend > 0.0:
                from surf.ot.costs import make_spectral_plus_euclidean_cost_fn
                cost = make_spectral_plus_euclidean_cost_fn(
                    blend=euclidean_blend,
                    knn=premetric_knn,
                    n_eig=premetric_n_eig,
                    spectral_family=premetric_spectral_family,
                    weight_power=premetric_weight_power,
                    diffusion_time=premetric_diffusion_time,
                    augmentation_eps=premetric_augmentation_eps,
                    augmentation_sigma_scale=premetric_augmentation_sigma_scale,
                )
            else:
                cost = make_spectral_cost_fn(
                    knn=premetric_knn,
                    n_eig=premetric_n_eig,
                    spectral_family=premetric_spectral_family,
                    weight_power=premetric_weight_power,
                    diffusion_time=premetric_diffusion_time,
                    augmentation_eps=premetric_augmentation_eps,
                    augmentation_sigma_scale=premetric_augmentation_sigma_scale,
                )
        elif name == "MM+MMFM-TorchCFM+GlobalSpectral":
            if stage_cells_for_global is None:
                raise SystemExit(
                    "MM+MMFM-TorchCFM+GlobalSpectral requires stage_cells_for_global."
                )
            from surf.ot.costs import make_global_spectral_cost_fn
            cost = make_global_spectral_cost_fn(
                stage_cells_for_global,
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
                blend=euclidean_blend,
            )
        else:
            cost = compute_euclidean_cost_matrix
        return {
            "cost_fn": cost,
            "framework": "torchcfm",
            "coupling_mode": "argmax_chain",
            "torchcfm_matcher": chosen_matcher,
            "torchcfm_sigma": chosen_sigma,
        }

    if name == "MM+Linear+GlobalSpectral":
        # Spectral OT with the GLOBAL Laplacian eigenbasis: one kNN graph
        # built on the union of ALL training stages, eigendecomposed
        # ONCE, every adjacent-pair OT cost looks up rows from the same
        # embedding. Theoretically the right cost for the
        # Belkin-Niyogi-style consistency precondition.
        if stage_cells_for_global is None:
            raise SystemExit(
                "MM+Linear+GlobalSpectral requires stage_cells_for_global; "
                "ensure resolve_method_kwargs is called with this kwarg."
            )
        from surf.ot.costs import make_global_spectral_cost_fn
        return {"cost_fn": make_global_spectral_cost_fn(
            stage_cells_for_global,
            knn=premetric_knn,
            n_eig=premetric_n_eig,
            spectral_family=premetric_spectral_family,
            weight_power=premetric_weight_power,
            diffusion_time=premetric_diffusion_time,
            blend=euclidean_blend,
        )}

    if name == "MM+MMFM-Rohbeck+GlobalSpectral":
        # Global spectral cost stacked on Rohbeck's MMFM scaffold.
        if stage_cells_for_global is None:
            raise SystemExit(
                "MM+MMFM-Rohbeck+GlobalSpectral requires stage_cells_for_global."
            )
        from surf.ot.costs import make_global_spectral_cost_fn
        return {
            "cost_fn": make_global_spectral_cost_fn(
                stage_cells_for_global,
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
                blend=euclidean_blend,
            ),
            "framework": "rohbeck",
        }

    if name == "MM+MMFM-Rohbeck+SquaredSpectral":
        # Spectral OT cost stacked on top of Rohbeck's MMFM scaffold. The
        # cost matrix is the only thing that changes vs MM+MMFM-Rohbeck;
        # spline + variance + chain construction are all from Rohbeck.
        # Use @alpha,blend,knn,n_eig parametric overrides as with the
        # plain MM+Linear+SquaredSpectral entry.
        if euclidean_blend > 0.0:
            from surf.ot.costs import make_spectral_plus_euclidean_cost_fn
            cost = make_spectral_plus_euclidean_cost_fn(
                blend=euclidean_blend,
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
                augmentation_eps=premetric_augmentation_eps,
                augmentation_sigma_scale=premetric_augmentation_sigma_scale,
            )
        else:
            cost = make_spectral_cost_fn(
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
                augmentation_eps=premetric_augmentation_eps,
                augmentation_sigma_scale=premetric_augmentation_sigma_scale,
            )
        return {"cost_fn": cost, "framework": "rohbeck"}

    if name == "MM+Linear+Biharmonic":
        return {"cost_fn": compute_biharmonic_cost_matrix}

    if name == "MM+Linear+Random":
        return {"cost_fn": compute_random_cost_matrix}

    if name == "MM+Linear+SquaredSpectral":
        if euclidean_blend > 0.0:
            from surf.ot.costs import make_spectral_plus_euclidean_cost_fn
            return {"cost_fn": make_spectral_plus_euclidean_cost_fn(
                blend=euclidean_blend,
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
                augmentation_eps=premetric_augmentation_eps,
                augmentation_sigma_scale=premetric_augmentation_sigma_scale,
            )}
        return {"cost_fn": make_spectral_cost_fn(
            knn=premetric_knn,
            n_eig=premetric_n_eig,
            spectral_family=premetric_spectral_family,
            weight_power=premetric_weight_power,
            diffusion_time=premetric_diffusion_time,
            augmentation_eps=premetric_augmentation_eps,
            augmentation_sigma_scale=premetric_augmentation_sigma_scale,
        )}

    if name == "MM+SLERP+Random":
        return {"cost_fn": compute_random_cost_matrix}

    if name == "MM+SLERP":
        return {}

    if name == "MM+SLERP+GraphSmooth":
        return {
            "graph_smooth_lambda": graph_smooth_strength,
            "graph_smooth_knn": graph_smooth_knn,
            "graph_smooth_batch_edges": graph_smooth_batch_edges,
            "graph_smooth_sigma_scale": graph_smooth_sigma_scale,
        }

    if name == "MM+Score_kde":
        return {
            "score_net": kde_wrapper,
            "alpha": score_alpha,
            "score_net_sigma": kde_wrapper.sigma if kde_wrapper is not None else score_net_sigma,
        }

    if name == "MM+Score_learned":
        return {
            "score_net": score_net,
            "alpha": score_alpha,
            "score_net_sigma": score_net_sigma,
        }

    if name == "MM+Score_forward":
        return {
            "score_nets_per_interval": forward_score_nets,
            "alpha": score_alpha,
            "score_net_sigma": score_net_sigma,
        }

    if name == "MM+SLERP+PHATE":
        return {"cost_fn": make_phate_cost_fn(
            n_components=phate_n_components,
            knn=phate_knn,
            decay=phate_decay,
            n_landmark=phate_n_landmark,
            graph_metric=phate_graph_metric,
        )}

    if name == "MM+SLERP+PHATE-Euclidean":
        return {"cost_fn": make_phate_cost_fn(
            n_components=phate_n_components,
            knn=phate_knn,
            decay=phate_decay,
            n_landmark=phate_n_landmark,
            graph_metric="euclidean",
        )}

    if name == "MM+SLERP+PHATE-SphereArc":
        return {"cost_fn": make_phate_cost_fn(
            n_components=phate_n_components,
            knn=phate_knn,
            decay=phate_decay,
            n_landmark=phate_n_landmark,
            graph_metric="sphere_arc",
        )}

    if name == "MM+SLERP+SquaredSpectral":
        return {"cost_fn": make_spectral_cost_fn(
            knn=premetric_knn,
            n_eig=premetric_n_eig,
            spectral_family=premetric_spectral_family,
            weight_power=premetric_weight_power,
            diffusion_time=premetric_diffusion_time,
            augmentation_eps=premetric_augmentation_eps,
            augmentation_sigma_scale=premetric_augmentation_sigma_scale,
        )}

    if name == "MM+SpectralPathJVP":
        return {
            "spectral_knn": premetric_knn,
            "spectral_n_eig": premetric_n_eig,
            "spectral_family": premetric_spectral_family,
            "spectral_weight_power": premetric_weight_power,
            "spectral_diffusion_time": premetric_diffusion_time,
        }

    if name == "MM+SLERP+Biharmonic":
        return {"cost_fn": make_biharmonic_cost_fn(
            knn=premetric_knn,
            n_eig=premetric_n_eig,
        )}

    if name == "MM+SLERP+GlobalBiharmonic":
        return {"cost_fn": global_biharmonic_cost_fn}

    if name == "MM+PremetricBiharmonic-SphereOT":
        return {
            "premetric_type": "biharmonic",
            "premetric_ot_cost": False,
            "premetric_extension_k": premetric_extension_k,
            "premetric_softmax_beta": premetric_softmax_beta,
            "premetric_ode_steps": premetric_ode_steps,
            "premetric_trajectory_mode": premetric_trajectory_mode,
            "premetric_decode_k": premetric_decode_k,
            "premetric_decode_beta": premetric_decode_beta,
            "premetric_decode_chunk_size": premetric_decode_chunk_size,
            "premetric_velocity_fd_eps": premetric_velocity_fd_eps,
            "premetric_knn": premetric_knn,
            "premetric_n_eig": premetric_n_eig,
            "premetric_spectral_family": premetric_spectral_family,
            "premetric_weight_power": premetric_weight_power,
            "premetric_diffusion_time": premetric_diffusion_time,
            "premetric_time_cap": premetric_time_cap,
            "premetric_grad_norm_floor": premetric_grad_norm_floor,
            "premetric_max_drive_scale": premetric_max_drive_scale,
        }

    if name in ("MM+PremetricBiharmonic-SpectralOT", "MM+PremetricBiharmonic"):
        return {
            "premetric_type": "biharmonic",
            "premetric_ot_cost": True,
            "premetric_extension_k": premetric_extension_k,
            "premetric_softmax_beta": premetric_softmax_beta,
            "premetric_ode_steps": premetric_ode_steps,
            "premetric_trajectory_mode": premetric_trajectory_mode,
            "premetric_decode_k": premetric_decode_k,
            "premetric_decode_beta": premetric_decode_beta,
            "premetric_decode_chunk_size": premetric_decode_chunk_size,
            "premetric_velocity_fd_eps": premetric_velocity_fd_eps,
            "premetric_knn": premetric_knn,
            "premetric_n_eig": premetric_n_eig,
            "premetric_spectral_family": premetric_spectral_family,
            "premetric_weight_power": premetric_weight_power,
            "premetric_diffusion_time": premetric_diffusion_time,
            "premetric_time_cap": premetric_time_cap,
            "premetric_grad_norm_floor": premetric_grad_norm_floor,
            "premetric_max_drive_scale": premetric_max_drive_scale,
        }

    if name == "MM+Score_learned+PHATE":
        return {
            "score_net": score_net,
            "alpha": score_alpha,
            "score_net_sigma": score_net_sigma,
            "cost_fn": make_phate_cost_fn(
                n_components=phate_n_components,
                knn=phate_knn,
                decay=phate_decay,
                n_landmark=phate_n_landmark,
                graph_metric=phate_graph_metric,
            ),
        }

    if name == "MM+Score_learned+Biharmonic":
        return {
            "score_net": score_net,
            "alpha": score_alpha,
            "score_net_sigma": score_net_sigma,
            "cost_fn": make_spectral_cost_fn(
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
            ),
        }

    if name == "MM+Score_timed":
        return {
            "score_net": timed_score_net,
            "alpha": score_alpha,
            "score_net_sigma": score_net_sigma,
        }

    if name == "MM+Score_timed+Biharmonic":
        return {
            "score_net": timed_score_net,
            "alpha": score_alpha,
            "score_net_sigma": score_net_sigma,
            "cost_fn": make_spectral_cost_fn(
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
            ),
        }

    if name == "MM+SI":
        return {"si_sigma": si_sigma}

    if name == "MM+SI+Biharmonic":
        return {
            "si_sigma": si_sigma,
            "cost_fn": make_spectral_cost_fn(
                knn=premetric_knn,
                n_eig=premetric_n_eig,
                spectral_family=premetric_spectral_family,
                weight_power=premetric_weight_power,
                diffusion_time=premetric_diffusion_time,
            ),
        }

    if name == "MM+BiharmonicVel":
        return {
            "biharm_beta": biharm_beta,
            "global_biharm_embeddings": global_biharm_embeddings,
        }

    if name == "MM+BiharmonicWaypoint":
        return {
            "biharm_waypoints": True,
            "global_biharm_embeddings": global_biharm_embeddings,
        }

    raise ValueError(f"Unknown method: {name!r}. Choices: {MM_METHOD_NAMES}")
