r"""
Build an 8-Gaussians -> two-moons SMFM schematic and toy performance sweep.

The figure is meant to be a compact pedagogical check for the cost-matrix
claim in the report:

  - random pairing is a no-coupling baseline;
  - Euclidean OT is the standard OT-CFM / linear-FM pairing;
  - spectral OT swaps only the ground cost, with an alpha x beta sweep.

Here alpha is the report exponent in

    C_spec_alpha(i,j) = sum_k (u_k(i)-u_k(j))^2 / lambda_k^alpha,

and beta is the Euclidean blend used elsewhere in the codebase:

    C_alpha,beta = (1-beta) C_spec_alpha + beta C_eucl,

with both costs mean-normalized before blending.

Outputs:
  outputs/smfm_schematic.{png,pdf,svg}
  outputs/smfm_schematic_metrics.csv
  outputs/smfm_schematic_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.nn as nn
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

from surf.evaluation.metrics import mmd_rbf, swd


PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS_DEFAULT = [0.0, 0.5, 1.0, 1.5, 2.0]
BETAS_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class TrainResult:
    repeat: int
    method: str
    alpha: float | None
    beta: float | None
    mmd2: float
    swd: float
    final_loss: float
    generated: np.ndarray
    plan: np.ndarray | None


@dataclass(frozen=True)
class MetricSummary:
    method: str
    alpha: float | None
    beta: float | None
    n: int
    mmd2_mean: float
    mmd2_std: float
    swd_mean: float
    swd_std: float
    final_loss_mean: float
    final_loss_std: float


class ToyVelocityNet(nn.Module):
    """Small time-conditioned velocity field for 2-D CFM."""

    def __init__(self, hidden: int = 128, depth: int = 4, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        in_dim = 2 + 1 + 2 * n_freqs
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t = t.view(-1, 1)
        freqs = torch.arange(1, self.n_freqs + 1, device=x.device, dtype=x.dtype)
        angles = 2.0 * np.pi * t * freqs.view(1, -1)
        t_feat = torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=1)
        return self.net(torch.cat([x, t_feat], dim=1))


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def apply_paper_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use(["seaborn-v0_8-paper"])
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def make_8gaussians_to_moons(
    n: int,
    seed: int,
    source_noise: float = 0.08,
    moon_noise: float = 0.07,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source 8-Gaussian mixture and target two-moons samples."""

    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    centers = 1.35 * np.column_stack([np.cos(angles), np.sin(angles)])
    comp = rng.integers(0, 8, size=n)
    x0 = centers[comp] + rng.normal(0.0, source_noise, size=(n, 2))

    x1, _ = make_moons(n_samples=n, noise=moon_noise, random_state=seed + 17)
    x1 = x1.astype(np.float64)
    x1[:, 0] -= 0.5
    x1[:, 1] -= 0.25
    x1 *= 1.15
    return x0.astype(np.float64), x1.astype(np.float64)


def build_union_graph(points: np.ndarray, knn: int) -> csr_matrix:
    """Symmetric Euclidean kNN graph with Gaussian weights."""

    nnbrs = min(knn + 1, len(points))
    nn_model = NearestNeighbors(n_neighbors=nnbrs, metric="euclidean")
    nn_model.fit(points)
    dists, inds = nn_model.kneighbors(points)
    positive = dists[:, 1:].reshape(-1)
    positive = positive[positive > 0]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    sigma = max(sigma, 1e-8)
    weights = np.exp(-(dists**2) / (2.0 * sigma**2))

    n = points.shape[0]
    rows = np.repeat(np.arange(n), nnbrs)
    cols = inds.reshape(-1)
    data = weights.reshape(-1)
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W.maximum(W.T)


def build_connected_union_graph(
    points: np.ndarray,
    knn: int,
    max_knn: int,
) -> tuple[csr_matrix, int, int]:
    """Increase k until the union graph is connected, if possible."""

    n = len(points)
    k = min(knn, max(1, n - 2))
    max_k = min(max_knn, max(1, n - 2))
    last_W = build_union_graph(points, k)
    last_cc, _ = connected_components(last_W, directed=False)
    while last_cc > 1 and k < max_k:
        k = min(max_k, max(k + 1, int(np.ceil(1.25 * k))))
        last_W = build_union_graph(points, k)
        last_cc, _ = connected_components(last_W, directed=False)
    return last_W, k, int(last_cc)


def laplacian_eigs(W: csr_matrix, n_eig: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric normalized graph Laplacian eigenpairs after dropping one zero."""

    n = W.shape[0]
    degree = np.asarray(W.sum(axis=1)).reshape(-1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    D_inv_sqrt = diags(d_inv_sqrt)
    L = diags(np.ones(n)) - D_inv_sqrt @ W @ D_inv_sqrt
    k_eig = min(n_eig + 1, n - 2)
    eigvals, eigvecs = eigsh(L, k=k_eig, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order][1:]
    eigvecs = eigvecs[:, order][:, 1:]
    return eigvals.astype(np.float64), eigvecs.astype(np.float64)


def spectral_cost_from_eigs(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    n0: int,
    alpha: float,
) -> np.ndarray:
    """Power-family spectral cost using the report exponent alpha."""

    eigvals = np.maximum(eigvals, 1e-8)
    weights = 1.0 / eigvals ** (0.5 * alpha)
    emb = eigvecs * weights[None, :]
    return cdist(emb[:n0], emb[n0:], metric="sqeuclidean").astype(np.float64)


def euclidean_cost(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    return cdist(X0, X1, metric="sqeuclidean").astype(np.float64)


def estimate_fixed_mmd_sigma(*clouds: np.ndarray) -> float:
    """One MMD bandwidth shared by all methods in this run."""

    points = torch.as_tensor(np.vstack(clouds), dtype=torch.float32)
    dists = torch.pdist(points)
    positive = dists[dists > 0]
    if len(positive) == 0:
        return 1.0
    return max(float(positive.median().item()) / np.sqrt(2.0), 1e-3)


def normalize_cost(C: np.ndarray) -> np.ndarray:
    scale = max(float(np.mean(C)), 1e-12)
    return C / scale


def solve_plan(C: np.ndarray) -> np.ndarray:
    n0, n1 = C.shape
    mu = np.full(n0, 1.0 / n0)
    nu = np.full(n1, 1.0 / n1)
    return ot.emd(mu, nu, C.astype(np.float64, copy=False))


def plan_cdf(plan: np.ndarray) -> np.ndarray:
    probs = plan.reshape(-1).astype(np.float64, copy=True)
    probs /= max(probs.sum(), 1e-12)
    cdf_vals = np.cumsum(probs)
    cdf_vals[-1] = 1.0
    return cdf_vals


def sample_pair_indices(
    rng: np.random.Generator,
    batch_size: int,
    n0: int,
    n1: int,
    cdf_vals: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if cdf_vals is None:
        return rng.integers(0, n0, size=batch_size), rng.integers(0, n1, size=batch_size)
    flat = np.searchsorted(cdf_vals, rng.random(batch_size), side="right")
    flat = np.minimum(flat, n0 * n1 - 1)
    return flat // n1, flat % n1


def train_and_eval(
    *,
    repeat: int,
    method: str,
    alpha: float | None,
    beta: float | None,
    X0_train: np.ndarray,
    X1_train: np.ndarray,
    X0_test: np.ndarray,
    X1_test: np.ndarray,
    plan: np.ndarray | None,
    seed: int,
    mmd_sigma: float,
    train_iters: int,
    batch_size: int,
    lr: float,
    hidden: int,
    depth: int,
    n_steps: int,
    device: torch.device,
) -> TrainResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ToyVelocityNet(hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    X0_t = torch.as_tensor(X0_train, dtype=torch.float32, device=device)
    X1_t = torch.as_tensor(X1_train, dtype=torch.float32, device=device)
    cdf_vals = None if plan is None else plan_cdf(plan)
    n0, n1 = len(X0_train), len(X1_train)

    final_loss = float("nan")
    for _ in range(train_iters):
        i_np, j_np = sample_pair_indices(rng, batch_size, n0, n1, cdf_vals)
        i = torch.as_tensor(i_np, dtype=torch.long, device=device)
        j = torch.as_tensor(j_np, dtype=torch.long, device=device)
        x0 = X0_t[i]
        x1 = X1_t[j]
        t = torch.rand(batch_size, dtype=torch.float32, device=device)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        target_v = x1 - x0
        pred_v = model(xt, t)
        loss = (pred_v - target_v).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        final_loss = float(loss.detach().cpu())

    with torch.no_grad():
        x = torch.as_tensor(X0_test, dtype=torch.float32, device=device)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = torch.full((len(x),), step * dt, dtype=torch.float32, device=device)
            x = x + dt * model(x, t_val)
        generated = x.detach().cpu().numpy()

    mmd2_val = float(mmd_rbf(generated, X1_test, sigma=mmd_sigma))
    # POT's sliced-Wasserstein helper samples projection directions from
    # NumPy's global RNG, so pin it for a deterministic auxiliary metric.
    np_state = np.random.get_state()
    np.random.seed(seed + 12345)
    try:
        swd_val = float(swd(generated, X1_test, n_projections=100))
    finally:
        np.random.set_state(np_state)
    return TrainResult(
        repeat=repeat,
        method=method,
        alpha=alpha,
        beta=beta,
        mmd2=mmd2_val,
        swd=swd_val,
        final_loss=final_loss,
        generated=generated,
        plan=plan,
    )


def edge_segments_from_graph(W: csr_matrix, points: np.ndarray) -> np.ndarray:
    coo = W.tocoo()
    mask = coo.row < coo.col
    rows, cols = coo.row[mask], coo.col[mask]
    if len(rows) == 0:
        return np.empty((0, 2, 2))
    return np.stack([points[rows], points[cols]], axis=1)


def transport_segments(
    plan: np.ndarray | None,
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    rng: np.random.Generator,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if plan is None:
        i = rng.integers(0, len(X0), size=top_k)
        j = rng.integers(0, len(X1), size=top_k)
        widths = np.full(top_k, 0.45)
        return np.stack([X0[i], X1[j]], axis=1), widths

    flat = plan.reshape(-1)
    pos = np.flatnonzero(flat > 0)
    if len(pos) == 0:
        return np.empty((0, 2, 2)), np.empty(0)
    order = pos[np.argsort(flat[pos])[::-1]]
    if len(order) > top_k:
        order = order[:top_k]
    n1 = X1.shape[0]
    i = order // n1
    j = order % n1
    mass = flat[order]
    widths = mass / max(float(mass.max()), 1e-12)
    return np.stack([X0[i], X1[j]], axis=1), widths


def draw_pairing_panel(
    ax: plt.Axes,
    X0: np.ndarray,
    X1: np.ndarray,
    plan: np.ndarray | None,
    title: str,
    rng: np.random.Generator,
    top_k: int = 140,
) -> None:
    segs, widths = transport_segments(plan, X0, X1, rng=rng, top_k=top_k)
    if len(segs):
        lc = LineCollection(
            segs,
            colors="#3f3f3f",
            linewidths=1.1 * widths,
            alpha=0.42,
            zorder=1,
        )
        ax.add_collection(lc)
    ax.scatter(X0[:, 0], X0[:, 1], c="#2b6cb0", s=16, edgecolors="none", rasterized=True, zorder=3)
    ax.scatter(X1[:, 0], X1[:, 1], c="#c53030", s=16, edgecolors="none", rasterized=True, zorder=3)
    ax.set_title(title)


def result_key(result: TrainResult) -> tuple[str, float | None, float | None]:
    return result.method, result.alpha, result.beta


def summarize_results(results: list[TrainResult]) -> dict[tuple[str, float | None, float | None], MetricSummary]:
    grouped: dict[tuple[str, float | None, float | None], list[TrainResult]] = {}
    for r in results:
        grouped.setdefault(result_key(r), []).append(r)

    summaries = {}
    for (method, alpha, beta), rows in grouped.items():
        ddof = 1 if len(rows) > 1 else 0
        mmd2_vals = np.array([r.mmd2 for r in rows], dtype=np.float64)
        swd_vals = np.array([r.swd for r in rows], dtype=np.float64)
        loss_vals = np.array([r.final_loss for r in rows], dtype=np.float64)
        summaries[(method, alpha, beta)] = MetricSummary(
            method=method,
            alpha=alpha,
            beta=beta,
            n=len(rows),
            mmd2_mean=float(mmd2_vals.mean()),
            mmd2_std=float(mmd2_vals.std(ddof=ddof)),
            swd_mean=float(swd_vals.mean()),
            swd_std=float(swd_vals.std(ddof=ddof)),
            final_loss_mean=float(loss_vals.mean()),
            final_loss_std=float(loss_vals.std(ddof=ddof)),
        )
    return summaries


def write_metrics_csv(path: Path, results: list[TrainResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["repeat", "method", "alpha", "beta", "mmd2", "swd", "final_loss"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "repeat": r.repeat,
                    "method": r.method,
                    "alpha": "" if r.alpha is None else f"{r.alpha:g}",
                    "beta": "" if r.beta is None else f"{r.beta:g}",
                    "mmd2": f"{r.mmd2:.8f}",
                    "swd": f"{r.swd:.8f}",
                    "final_loss": f"{r.final_loss:.8f}",
                }
            )


def write_summary_csv(path: Path, summaries: list[MetricSummary]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "alpha",
                "beta",
                "n",
                "mmd2_mean",
                "mmd2_std",
                "swd_mean",
                "swd_std",
                "final_loss_mean",
                "final_loss_std",
            ],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "method": s.method,
                    "alpha": "" if s.alpha is None else f"{s.alpha:g}",
                    "beta": "" if s.beta is None else f"{s.beta:g}",
                    "n": s.n,
                    "mmd2_mean": f"{s.mmd2_mean:.8f}",
                    "mmd2_std": f"{s.mmd2_std:.8f}",
                    "swd_mean": f"{s.swd_mean:.8f}",
                    "swd_std": f"{s.swd_std:.8f}",
                    "final_loss_mean": f"{s.final_loss_mean:.8f}",
                    "final_loss_std": f"{s.final_loss_std:.8f}",
                }
            )


def fmt_mean_std(summary: MetricSummary) -> str:
    return f"{summary.mmd2_mean:.4f}$\\pm${summary.mmd2_std:.4f}"


def make_figure(
    *,
    X0: np.ndarray,
    X1: np.ndarray,
    generated_by_method: dict[str, np.ndarray],
    W: csr_matrix,
    u1: np.ndarray,
    random_result: TrainResult,
    euclidean_result: TrainResult,
    best_spectral: TrainResult,
    random_summary: MetricSummary,
    euclidean_summary: MetricSummary,
    best_summary: MetricSummary,
    spectral_grid: np.ndarray,
    spectral_grid_std: np.ndarray,
    alphas: list[float],
    betas: list[float],
    output_base: Path,
) -> None:
    apply_paper_style()
    fig = plt.figure(figsize=(14.5, 7.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.08], width_ratios=[1, 1, 1, 1])
    axes = {
        "data": fig.add_subplot(gs[0, 0]),
        "graph": fig.add_subplot(gs[0, 1]),
        "random": fig.add_subplot(gs[0, 2]),
        "eucl": fig.add_subplot(gs[0, 3]),
        "spec": fig.add_subplot(gs[1, 0]),
        "heatmap": fig.add_subplot(gs[1, 1:3]),
        "bar": fig.add_subplot(gs[1, 3]),
    }

    rng = np.random.default_rng(123)
    axes["data"].scatter(X0[:, 0], X0[:, 1], c="#2b6cb0", s=18, edgecolors="none", label="8 Gaussians", rasterized=True)
    axes["data"].scatter(X1[:, 0], X1[:, 1], c="#c53030", s=18, edgecolors="none", label="two moons", rasterized=True)
    axes["data"].set_title("(a) Source and target")
    axes["data"].legend(loc="upper right", frameon=True, framealpha=0.92)

    segs = edge_segments_from_graph(W, np.vstack([X0, X1]))
    if len(segs):
        axes["graph"].add_collection(LineCollection(segs, colors="black", linewidths=0.22, alpha=0.22, zorder=1))
    sc = axes["graph"].scatter(
        np.vstack([X0, X1])[:, 0],
        np.vstack([X0, X1])[:, 1],
        c=u1,
        cmap="coolwarm",
        s=17,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    axes["graph"].set_title(r"(b) Union $k$NN graph")
    cbar = fig.colorbar(sc, ax=axes["graph"], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r"$u_1$", fontsize=9)

    draw_pairing_panel(
        axes["random"],
        X0,
        X1,
        None,
        f"(c) Random pairing\nMMD$^2$={fmt_mean_std(random_summary)}",
        rng,
    )
    draw_pairing_panel(
        axes["eucl"],
        X0,
        X1,
        euclidean_result.plan,
        f"(d) Euclidean OT\nMMD$^2$={fmt_mean_std(euclidean_summary)}",
        rng,
    )
    draw_pairing_panel(
        axes["spec"],
        X0,
        X1,
        best_spectral.plan,
        (
            "(e) Best spectral OT\n"
            rf"$\alpha={best_spectral.alpha:g},\ \beta={best_spectral.beta:g}$"
            f", MMD$^2$={fmt_mean_std(best_summary)}"
        ),
        rng,
    )

    im = axes["heatmap"].imshow(spectral_grid, cmap="viridis_r", aspect="auto")
    axes["heatmap"].set_title(
        rf"(f) Spectral OT sweep, mean MMD$^2$ over {best_summary.n} seeds"
    )
    axes["heatmap"].set_xlabel(r"Euclidean blend $\beta$")
    axes["heatmap"].set_ylabel(r"Spectral exponent $\alpha$")
    axes["heatmap"].set_xticks(range(len(betas)))
    axes["heatmap"].set_xticklabels([f"{b:g}" for b in betas])
    axes["heatmap"].set_yticks(range(len(alphas)))
    axes["heatmap"].set_yticklabels([f"{a:g}" for a in alphas])
    for i in range(len(alphas)):
        for j in range(len(betas)):
            val = spectral_grid[i, j]
            color = "white" if val > np.nanmean(spectral_grid) else "black"
            axes["heatmap"].text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=axes["heatmap"], fraction=0.032, pad=0.02)
    cbar.set_label(r"MMD$^2$", fontsize=9)

    names = ["Random", "Eucl. OT", "Best spec."]
    values = [random_summary.mmd2_mean, euclidean_summary.mmd2_mean, best_summary.mmd2_mean]
    yerr = [random_summary.mmd2_std, euclidean_summary.mmd2_std, best_summary.mmd2_std]
    colors = ["#718096", "#dd6b20", "#2f855a"]
    axes["bar"].bar(names, values, yerr=yerr, color=colors, width=0.68, capsize=3)
    axes["bar"].set_title("(g) Endpoint performance")
    axes["bar"].set_ylabel(r"MMD$^2$ to two moons")
    axes["bar"].tick_params(axis="x", rotation=25)
    axes["bar"].grid(axis="y", alpha=0.22)

    # Overlay generated endpoints in the bar panel as a tiny qualitative inset.
    inset = axes["bar"].inset_axes([0.52, 0.48, 0.45, 0.45])
    inset.scatter(X1[:, 0], X1[:, 1], c="#c53030", s=6, alpha=0.35, edgecolors="none", rasterized=True)
    inset.scatter(generated_by_method["best_spectral"][:, 0], generated_by_method["best_spectral"][:, 1],
                  c="#2f855a", s=6, alpha=0.45, edgecolors="none", rasterized=True)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("best", fontsize=8)

    all_points = np.vstack([X0, X1, generated_by_method["best_spectral"]])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    x_pad = 0.06 * (x_max - x_min)
    y_pad = 0.06 * (y_max - y_min)
    for key in ["data", "graph", "random", "eucl", "spec"]:
        ax = axes[key]
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    inset.set_xlim(x_min - x_pad, x_max + x_pad)
    inset.set_ylim(y_min - y_pad, y_max + y_pad)
    inset.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(f"{output_base}.png")
    fig.savefig(f"{output_base}.pdf")
    fig.savefig(f"{output_base}.svg")
    plt.close(fig)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--knn", type=int, default=24)
    parser.add_argument("--max-knn", type=int, default=80)
    parser.add_argument("--allow-disconnected", action="store_true")
    parser.add_argument("--n-eig", type=int, default=40)
    parser.add_argument("--alphas", default=",".join(f"{a:g}" for a in ALPHAS_DEFAULT))
    parser.add_argument("--betas", default=",".join(f"{b:g}" for b in BETAS_DEFAULT))
    parser.add_argument("--train-iters", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=80)
    parser.add_argument("--mmd-sigma", type=float, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    if 1.0 not in betas:
        betas.append(1.0)
    betas = sorted(betas)
    device = choose_device(args.device)

    print("Generating 8-Gaussians -> two-moons toy data...")
    X0_train, X1_train = make_8gaussians_to_moons(args.n_train, args.seed)
    X0_test, X1_test = make_8gaussians_to_moons(args.n_test, args.seed + 1000)
    C_eucl = euclidean_cost(X0_train, X1_train)
    mmd_sigma = args.mmd_sigma
    if mmd_sigma is None:
        mmd_sigma = estimate_fixed_mmd_sigma(X0_test, X1_test)
    print(f"Fixed MMD bandwidth sigma={mmd_sigma:.6f}")

    print(f"Building union kNN graph (initial k={args.knn}) and spectral basis...")
    combined = np.vstack([X0_train, X1_train])
    if args.allow_disconnected:
        W = build_union_graph(combined, knn=args.knn)
        used_knn = args.knn
        n_cc, _ = connected_components(W, directed=False)
        n_cc = int(n_cc)
    else:
        W, used_knn, n_cc = build_connected_union_graph(
            combined,
            knn=args.knn,
            max_knn=args.max_knn,
        )
    eigvals, eigvecs = laplacian_eigs(W, n_eig=args.n_eig)
    print(f"  used k={used_knn}; connected components: {n_cc}")
    if n_cc > 1:
        print("  WARNING: spectral eigenvectors include component-indicator modes.")
    print(f"  first eigenvalues after trivial zero: {np.array2string(eigvals[:5], precision=4)}")

    print("Solving Euclidean OT baseline...")
    pi_eucl = solve_plan(C_eucl)

    print("Solving spectral OT plans for the alpha x beta grid...")
    spectral_plans: dict[tuple[float, float], np.ndarray] = {}
    C_eucl_n = normalize_cost(C_eucl)
    for alpha in alphas:
        C_spec = spectral_cost_from_eigs(eigvals, eigvecs, n0=len(X0_train), alpha=alpha)
        C_spec_n = normalize_cost(C_spec)
        for beta in betas:
            if beta == 1.0:
                continue
            C = (1.0 - beta) * C_spec_n + beta * C_eucl_n
            spectral_plans[(alpha, beta)] = solve_plan(C)

    common_kwargs = dict(
        X0_train=X0_train,
        X1_train=X1_train,
        X0_test=X0_test,
        X1_test=X1_test,
        train_iters=args.train_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        depth=args.depth,
        n_steps=args.ode_steps,
        mmd_sigma=mmd_sigma,
        device=device,
    )

    results: list[TrainResult] = []
    representatives: dict[tuple[str, float | None, float | None], TrainResult] = {}
    print(f"Training {args.n_seeds} matched model seeds on {device}...")
    for repeat in range(args.n_seeds):
        model_seed = args.seed + 10_000 + repeat
        random_result = train_and_eval(
            repeat=repeat,
            method="random",
            alpha=None,
            beta=None,
            plan=None,
            seed=model_seed,
            **common_kwargs,
        )
        euclidean_result = train_and_eval(
            repeat=repeat,
            method="euclidean_ot",
            alpha=None,
            beta=1.0,
            plan=pi_eucl,
            seed=model_seed,
            **common_kwargs,
        )
        results.extend([random_result, euclidean_result])
        representatives.setdefault(result_key(random_result), random_result)
        representatives.setdefault(result_key(euclidean_result), euclidean_result)

        for alpha in alphas:
            for beta in betas:
                if beta == 1.0:
                    continue
                r = train_and_eval(
                    repeat=repeat,
                    method="spectral_ot",
                    alpha=alpha,
                    beta=beta,
                    plan=spectral_plans[(alpha, beta)],
                    seed=model_seed,
                    **common_kwargs,
                )
                results.append(r)
                representatives.setdefault(result_key(r), r)

        summaries_so_far = summarize_results(results)
        random_so_far = summaries_so_far[("random", None, None)]
        euclidean_so_far = summaries_so_far[("euclidean_ot", None, 1.0)]
        spectral_so_far = [
            s for k, s in summaries_so_far.items()
            if k[0] == "spectral_ot" and k[2] != 1.0
        ]
        best_so_far = min(spectral_so_far, key=lambda s: s.mmd2_mean)
        print(
            f"  seed {repeat + 1:02d}/{args.n_seeds}: "
            f"random={random_result.mmd2:.4f}, eucl={euclidean_result.mmd2:.4f}; "
            f"running best spectral alpha={best_so_far.alpha:g}, beta={best_so_far.beta:g}, "
            f"mean MMD^2={best_so_far.mmd2_mean:.4f}",
            flush=True,
        )

    summaries = summarize_results(results)
    random_summary = summaries[("random", None, None)]
    euclidean_summary = summaries[("euclidean_ot", None, 1.0)]
    spectral_summaries = {
        (s.alpha, s.beta): s
        for k, s in summaries.items()
        if k[0] == "spectral_ot" and s.beta != 1.0
    }
    best_key, best_summary = min(
        spectral_summaries.items(),
        key=lambda item: item[1].mmd2_mean,
    )
    best_spectral = representatives[("spectral_ot", best_key[0], best_key[1])]

    spectral_grid = np.full((len(alphas), len(betas)), np.nan)
    spectral_grid_std = np.full((len(alphas), len(betas)), np.nan)
    for ai, alpha in enumerate(alphas):
        for bi, beta in enumerate(betas):
            if beta == 1.0:
                spectral_grid[ai, bi] = euclidean_summary.mmd2_mean
                spectral_grid_std[ai, bi] = euclidean_summary.mmd2_std
                continue
            s = spectral_summaries[(alpha, beta)]
            spectral_grid[ai, bi] = s.mmd2_mean
            spectral_grid_std[ai, bi] = s.mmd2_std

    print(
        "\nBest spectral cell: "
        f"alpha={best_key[0]:g}, beta={best_key[1]:g}, "
        f"MMD^2={best_summary.mmd2_mean:.4f} ± {best_summary.mmd2_std:.4f}, "
        f"SWD={best_summary.swd_mean:.4f} ± {best_summary.swd_std:.4f}"
    )
    print(
        "Baselines: "
        f"random MMD^2={random_summary.mmd2_mean:.4f} ± {random_summary.mmd2_std:.4f}; "
        f"Euclidean OT MMD^2={euclidean_summary.mmd2_mean:.4f} ± {euclidean_summary.mmd2_std:.4f}"
    )

    csv_path = OUT_DIR / "smfm_schematic_metrics.csv"
    summary_path = OUT_DIR / "smfm_schematic_summary.csv"
    write_metrics_csv(csv_path, results)
    summary_rows = sorted(
        summaries.values(),
        key=lambda s: (
            {"random": 0, "euclidean_ot": 1, "spectral_ot": 2}.get(s.method, 99),
            -1.0 if s.alpha is None else s.alpha,
            -1.0 if s.beta is None else s.beta,
        ),
    )
    write_summary_csv(summary_path, summary_rows)
    output_base = OUT_DIR / "smfm_schematic"
    make_figure(
        X0=X0_train,
        X1=X1_train,
        generated_by_method={"best_spectral": best_spectral.generated},
        W=W,
        u1=eigvecs[:, 0],
        random_result=representatives[("random", None, None)],
        euclidean_result=representatives[("euclidean_ot", None, 1.0)],
        best_spectral=best_spectral,
        random_summary=random_summary,
        euclidean_summary=euclidean_summary,
        best_summary=best_summary,
        spectral_grid=spectral_grid,
        spectral_grid_std=spectral_grid_std,
        alphas=alphas,
        betas=betas,
        output_base=output_base,
    )
    print(f"Wrote {output_base}.png, {output_base}.pdf, {output_base}.svg")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
