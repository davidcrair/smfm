"""Modal entrypoint for SMFM training sweeps.

Auth: uses the Modal secret `smfm-git-token` (key `GITHUB_TOKEN`) to clone
this private repo at image-build time and to `git pull` fresh commits on
each run.

One-time setup:
    pip install modal
    modal token new
    modal secret create smfm-git-token GITHUB_TOKEN=<your_PAT>

Run a single sweep from your laptop:
    modal run modal_smfm.py::run_bm_sweep
    modal run modal_smfm.py::run_pe_sweep
    modal run modal_smfm.py::run_paul15_sweep
    modal run modal_smfm.py::run_eb_sweep

After it finishes, pull logs back:
    modal volume get smfm-results /logs ./modal_outputs
    cp modal_outputs/*.log logs/
    .venv/bin/python scripts/build_bonemarrow_table.py
"""
from __future__ import annotations

import os as _os

import modal


app = modal.App("smfm")

# Modal secret containing GITHUB_TOKEN (a fine-grained PAT with read access
# to davidcrair/surf_project). Used at both image-build time and at runtime
# (for git pull) so each sweep gets the latest commits without rebuilding.
GIT_SECRET = modal.Secret.from_name("smfm-git-token")

REPO_URL = "https://x-access-token:$GITHUB_TOKEN@github.com/davidcrair/surf_project.git"

# Image: Python + system deps + repo clone + pip deps. The clone uses the
# secret so private repos work. Image build is cached after first run; new
# commits are fetched at runtime via `git pull`.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")
    .pip_install(
        "torch==2.3.1",
        "numpy",
        "scipy",
        "scikit-learn",
        "POT",
        "hydra-core",
        "omegaconf",
        "scanpy",
        "anndata",
        "phate",
        "torchcfm",
        "matplotlib",
        "scienceplots",
    )
    .run_commands(
        f"git clone {REPO_URL} /workspace",
        "cd /workspace && pip install -e .",
        # Pre-download EB-OTPFM dataset (TrajectoryNet eb_velocity_v5.npz, ~150MB).
        # Baked into the image so all subsequent runs skip the download.
        "mkdir -p /workspace/data",
        "curl -sL "
        "https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/eb_velocity_v5.npz "
        "-o /workspace/data/eb_velocity_v5.npz",
        secrets=[GIT_SECRET],
    )
)

# Persistent volume so logs survive across runs and can be pulled to laptop.
results = modal.Volume.from_name("smfm-results", create_if_missing=True)

# Volume for non-public dataset files that aren't in the git repo (e.g.,
# embryoid_body.h5ad, ~430 MB). Upload from laptop once with:
#   modal volume put smfm-data ./embryoid_body.h5ad /
# Then files appear at /data/embryoid_body.h5ad in the container.
data_vol = modal.Volume.from_name("smfm-data", create_if_missing=True)


def _run_script(script_name: str, env_vars: dict | None = None) -> None:
    """Pull latest commits, run the bash script, sync logs to the volume."""
    import subprocess

    env_prefix = " ".join(f"{k}={v}" for k, v in (env_vars or {}).items())
    # Fetch latest commits (the remote URL baked into .git/config already
    # embeds the PAT from the image-build clone, so no extra auth needed).
    subprocess.run("cd /workspace && git pull --quiet", shell=True, check=True)
    # Symlink data files from /data volume into /workspace where the loaders
    # expect them. No-op if /data is empty or files already linked.
    subprocess.run(
        "for f in /data/*.h5ad /data/*.npz /data/*.npy; do "
        "  [ -f \"$f\" ] && ln -sf \"$f\" /workspace/$(basename \"$f\"); "
        "done 2>/dev/null || true",
        shell=True,
        check=False,
    )
    subprocess.run(
        f"cd /workspace && {env_prefix} bash scripts/{script_name}",
        shell=True,
        check=True,
    )
    subprocess.run(
        "mkdir -p /results/logs && cp /workspace/logs/*.log /results/logs/ 2>/dev/null || true",
        shell=True,
        check=True,
    )
    results.commit()


# Hardware spec: L4 (24GB Ada) is the cost/perf sweet spot for our 4x256 MLP +
# CPU-heavy spectral OT setup. ~$0.80/hr -- about 1/3 the cost of A100-40GB.
# Override via env var:
#   MODAL_GPU=T4 modal run modal_smfm.py::run_bm_sweep   # cheapest (~$0.59/hr)
#   MODAL_GPU=A100-40GB modal run ...                    # fastest, overkill
GPU_KW = dict(
    gpu=_os.environ.get("MODAL_GPU", "L4"),
    # Bumped to 24 vCPU so PARALLEL_SPLITS=5 with THREADS_PER_PROC=4 (=20 used)
    # leaves 4-core headroom for the OS / Python overhead. Definitively
    # prevents the load-average>>cpu thrashing we saw earlier.
    cpu=float(_os.environ.get("MODAL_CPU", "24")),
    memory=32 * 1024,
)

FN_KW = dict(
    image=image,
    secrets=[GIT_SECRET],            # for runtime git pull
    volumes={"/results": results, "/data": data_vol},
    **GPU_KW,
)


@app.function(timeout=3 * 3600, **FN_KW)
def run_pe_sweep():
    _run_script(
        "run_power_law_kfold_pancreas_linear_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_bm_sweep():
    _run_script(
        "run_power_law_kfold_bonemarrow_linear_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=4 * 3600, **FN_KW)
def run_bm_pca50_sweep():
    """Full 7-method spectral-OT sweep on bonemarrow in 50-dim PCA latent space.

    Mirrors run_bm_sweep but with `space=pca50`. PCA fit per split on the
    training-cell union, predictions decoded back to gene space and evaluated
    against held-out test cells, identical eval to the sphere version.
    """
    _run_script(
        "run_power_law_kfold_bonemarrow_linear_pca50_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=2 * 3600, **FN_KW)
def run_bm_split42_only():
    """Re-run only data.seed=42 for bonemarrow (it timed out in the main run).

    Uses the SPLITS_OVERRIDE env hook in the bash script to skip splits 43-46.
    The resulting log will be copied into the same shared volume; merge it
    locally with `modal volume get` and re-run build_bonemarrow_table.py.
    """
    _run_script(
        "run_power_law_kfold_bonemarrow_linear_b512.sh",
        {"SPLITS_OVERRIDE": "42", "PARALLEL_SPLITS": "1", "THREADS_PER_PROC": "8", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_paul15_sweep():
    _run_script(
        "run_power_law_kfold_paul15_linear_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_bm_log1p_eucknn_sweep():
    """Bonemarrow log1p ambient + **Euclidean kNN** for the spectral Laplacian.
    Tests whether the BM -44% sphere headline recovers in log1p once we stop
    discarding the magnitude axis via cosine kNN. Set SMFM_KNN_METRIC=euclidean.
    """
    _run_script(
        "run_power_law_kfold_bonemarrow_linear_log1p_eucknn_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_bm_log1p_sweep():
    """Bonemarrow in raw log1p HVG ambient (no sphere encoding) with dual-space
    eval (2000-dim ambient + 50-dim PCA latent) and use_ema=false.

    Sanity check: does the strong sphere-encoded BM headline (-44% chained
    MMD^2) hold when we strip the sphere normalization?
    """
    _run_script(
        "run_power_law_kfold_bonemarrow_linear_log1p_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_pe_log1p_eucknn_sweep():
    """PE log1p ambient + Euclidean kNN. Methodologically consistent with the
    GoM EucKNN run; checks whether the existing PE log1p -21% (cosine kNN)
    holds, improves, or degrades under the more principled metric.
    """
    _run_script(
        "run_power_law_kfold_pancreas_linear_log1p_eucknn_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_pe_log1p_sweep():
    """Pancreas in raw log1p HVG ambient (no sphere encoding) with dual-space
    eval (2000-dim ambient + 50-dim PCA latent). Mirrors run_paul15_log1p_sweep.
    """
    _run_script(
        "run_power_law_kfold_pancreas_linear_log1p_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_paul15_log1p_eucknn_sweep():
    """Paul15 log1p ambient + Euclidean kNN. Methodologically consistent
    with the GoM EucKNN result; tests whether Paul15's prior log1p mixed/
    null result improves under the more principled metric.
    """
    _run_script(
        "run_power_law_kfold_paul15_linear_log1p_eucknn_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=3 * 3600, **FN_KW)
def run_paul15_log1p_sweep():
    """Paul15 sanity check in raw log1p HVG ambient (no sphere encoding).

    Reports MMD^2 / MMD^2_otpfm / SWD in both 2000-dim ambient and 50-dim
    PCA latent (eval-time PCA fit on the training cells, see
    `eval.compute_pca_metrics=true` in the experiment config).
    """
    _run_script(
        "run_power_law_kfold_paul15_linear_log1p_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=2 * 3600, **FN_KW)
def run_gom_otpfm9_alpha_sweep_eucknn():
    """GoM-9 OTP-FM alpha-sweep with Euclidean kNN (correct metric for 2D
    spatial data). Mirrors run_gom_otpfm9_alpha_sweep but exports
    SMFM_KNN_METRIC=euclidean inside the bash script.
    """
    _run_script(
        "run_gom_otpfm9_alpha_sweep_eucknn.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=4 * 3600, **FN_KW)
def run_gom_otpfm9_alpha_blend_grid_eucknn():
    """GoM-9 OTP-FM alpha x blend grid (5 alpha x 4 blend = 20 spectral
    cells + MM+Linear + Random OT) with Euclidean kNN.

    5 init seeds x 5 data splits, exact W_2 as primary metric. Use this
    over the alpha-only sweep when you want to characterize the joint
    (alpha, beta) landscape (e.g. for the heatmap figure or to confirm
    the best (alpha=0.5, beta=0) cell against full grid neighbors).

    Wall-clock: roughly 4-5x the alpha-only sweep at the same parallelism.
    """
    _run_script(
        "run_gom_alpha_blend_grid_eucknn.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=2 * 3600, **FN_KW)
def run_gom_otpfm9_alpha_sweep():
    """GoM 9-stage OTP-FM hold-out alpha sweep (beta=0, pure spectral),
    5 init seeds x 5 data splits, exact W_2 as primary metric.

    Methods: MM+Linear, MM+Linear+Random, MM+Linear+SquaredSpectral@alpha=
    {0, 0.5, 1, 1.5, 2}. Each split log will contain CHAINED+PER-SEGMENT
    eval blocks for MMD^2_otpfm and W_2; the W_2 block is the primary
    headline (~111 cells/marginal makes exact W_2 fast).
    """
    _run_script(
        "run_gom_otpfm9_alpha_sweep.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=6 * 3600, **FN_KW)
def run_eb_sweep():
    _run_script(
        "run_power_law_kfold_eb_linear_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=4 * 3600, **FN_KW)
def run_eb_otpfm_eucknn_sweep():
    """EB OTP-FM hold-out alpha sweep in PCA-100 latent + Euclidean kNN,
    5 init seeds x 5 data splits. Matches the OTP-FM/MMFM published
    preprocessing convention (TrajectoryNet eb_velocity_v5.npz, top-100
    PCs, StandardScaler) so numbers are directly comparable.
    """
    _run_script(
        "run_power_law_kfold_eb_otpfm_eucknn_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=2 * 3600, **FN_KW)
def run_eb_otpfm_eucknn_2k_sweep():
    """Same as run_eb_otpfm_eucknn_sweep but with max_cells_per_stage=2000
    for ~5x faster wallclock (~$0.50 instead of ~$2.50). Use as a quick
    iteration / sanity check; the full sweep is the headline result.
    """
    _run_script(
        "run_power_law_kfold_eb_otpfm_eucknn_b512.sh",
        {
            "PARALLEL_SPLITS": "5",
            "THREADS_PER_PROC": "4",
            "N_SEEDS": "5",
            # `+` prefix to ADD the key (eb_otpfm.yaml is in struct mode and
            # doesn't define max_cells_per_stage by default).
            "EXTRA_OVERRIDES": "+data.max_cells_per_stage=2000",
            "LOG_SUFFIX": "_2k",
        },
    )


@app.function(timeout=2 * 3600, **FN_KW)
def run_eb_log1p_eucknn_2k_sweep():
    """EB 80/10/10 split, log1p HVG ambient + Euclidean kNN, subsampled to
    2000 cells per marginal. Mirrors BM/PE log1p+EucKNN; the cell cap
    keeps wallclock manageable on the largest scRNA-seq dataset.
    """
    _run_script(
        "run_power_law_kfold_eb_linear_log1p_eucknn_2k_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.function(timeout=6 * 3600, **FN_KW)
def run_eb_log1p_sweep():
    """EB 80/10/10 in raw log1p HVG ambient (no sphere encoding) with
    dual-space eval (2000-dim ambient + 50-dim PCA latent). Largest dataset
    in the suite (~25k cells); 4-6 hr wall on L4 with 5-way parallel splits.
    """
    _run_script(
        "run_power_law_kfold_eb_linear_log1p_b512.sh",
        {"PARALLEL_SPLITS": "5", "THREADS_PER_PROC": "4", "N_SEEDS": "5"},
    )


@app.local_entrypoint()
def main():
    """Default entrypoint — runs BM. Other sweeps via `modal run modal_smfm.py::<fn>`."""
    run_bm_sweep.remote()
