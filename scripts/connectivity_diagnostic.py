"""
Connectivity diagnostic for spectral OT viability across all four datasets:
EB, pancreas, Gulf of Mexico vortex, Beijing PM2.5.

For each dataset we form the union of all marginals in the same representation
the spectral OT cost would see, build the kNN graph at varying k, and count
connected components. Fewer components means the union manifold is more
connected, which is the precondition for the bottom-Laplacian eigenvectors
to encode global manifold structure rather than per-cluster indicators.

Output: one panel (kNN k on x-axis, log #components on y-axis), one curve per
dataset. Saved as PNG and PDF under surf_latex/final_report/figures/.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


PROJ = Path("/Users/davidcrair/Documents/personal/cpsc_5860/final_project")
DOWNLOADS = Path("/Users/davidcrair/Downloads")
K_VALUES = [3, 5, 8, 10, 15, 20, 30, 50, 75, 100]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_eb_marginals():
    """EB train cells per stage, sphere-encoded (matches the spectral cost path)."""
    from surf.data.embryoid import load_embryoid_body
    from surf.geometry.sphere import to_orthant, to_compositional, normalize_sphere

    data = load_embryoid_body(str(PROJ / "embryoid_body.h5ad"), n_hvg=2000, seed=42)
    stages = data["train"]["stages"]
    out = []
    for s in stages:
        x = data["train"]["cells"][s]
        out.append(normalize_sphere(to_orthant(to_compositional(x))).numpy())
    return out, "cosine"


def load_pancreas_marginals():
    from surf.data.pancreas import load_pancreas
    from surf.geometry.sphere import to_orthant, to_compositional, normalize_sphere

    data = load_pancreas(
        path=str(PROJ / "data/Pancreas/endocrinogenesis_day15.h5ad"),
        n_hvg=2000, seed=42,
    )
    stages = data["train"]["stages"]
    out = []
    for s in stages:
        x = data["train"]["cells"][s]
        out.append(normalize_sphere(to_orthant(to_compositional(x))).numpy())
    return out, "cosine"


def load_gom_marginals():
    """GoM 2D vortex positions, normalized."""
    gom = _load_module("gom_data", DOWNLOADS / "data.py")
    bundle = gom.load_gom_data(data_dir=PROJ / "data", normalize=True, ot_coupling=False)
    return bundle["marginals_list"], "euclidean"


def load_beijing_marginals():
    beij = _load_module("beijing_data", DOWNLOADS / "data-2.py")
    bundle = beij.load_beijing_data(
        data_dir=PROJ / "data" / "beijing",
        station="Dingling", normalize=True, ot_coupling=False,
    )
    return bundle["marginals_list"], "euclidean"


def component_count(marginals, k: int, metric: str) -> int:
    """Build kNN union graph (symmetrized) and return number of connected components."""
    combined = np.vstack(marginals)
    N = len(combined)
    if k + 1 >= N:
        return -1
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(combined)
    _, inds = nn.kneighbors(combined)
    rows = np.repeat(np.arange(N), k + 1)
    cols = inds.reshape(-1)
    data = np.ones_like(rows, dtype=np.float64)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W.maximum(W.T)
    n_comp, _ = connected_components(W, directed=False)
    return n_comp


def main():
    datasets = [
        ("Embryoid Body",     load_eb_marginals),
        ("Pancreas",          load_pancreas_marginals),
        ("GoM vortex",        load_gom_marginals),
        ("Beijing PM2.5",     load_beijing_marginals),
    ]

    results = {}
    for name, loader in datasets:
        print(f"Loading {name}...")
        marginals, metric = loader()
        sizes = [len(m) for m in marginals]
        n_total = sum(sizes)
        print(f"  {name}: {len(marginals)} marginals, sizes={sizes}, total={n_total}, metric={metric}")
        comps = []
        for k in K_VALUES:
            c = component_count(marginals, k, metric)
            comps.append(c)
            print(f"    k={k:>3}  components={c}")
        results[name] = {"k": K_VALUES, "comps": comps, "metric": metric, "n": n_total}

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0))
    # Open markers / dashed for the two scRNA-seq curves so they're both
    # visible when they pin at 1 component.
    style = {
        "Embryoid Body":  dict(color="#1f77b4", marker="o", linestyle="-",
                               markerfacecolor="none", markeredgewidth=1.6,
                               label="Embryoid Body (cos, sphere-enc)"),
        "Pancreas":       dict(color="#2ca02c", marker="s", linestyle=(0, (3, 2)),
                               markerfacecolor="none", markeredgewidth=1.6,
                               label="Pancreas (cos, sphere-enc)"),
        "GoM vortex":     dict(color="#d62728", marker="^", linestyle="-",
                               label="GoM vortex (Euclidean, 2D)"),
        "Beijing PM2.5":  dict(color="#ff7f0e", marker="D", linestyle="-",
                               label="Beijing PM2.5 (Euclidean, 1D)"),
    }
    plot_order = ["Beijing PM2.5", "GoM vortex", "Embryoid Body", "Pancreas"]
    for name in plot_order:
        vals = results[name]
        comps = np.array(vals["comps"], dtype=float)
        comps[comps < 0] = np.nan
        ax.plot(vals["k"], comps, markersize=6, linewidth=1.6, **style[name])
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.0, label="connected (1 component)")
    ax.set_yscale("log")
    ax.set_xlabel("kNN neighbors $k$")
    ax.set_ylabel("# connected components in union graph (log)")
    ax.set_title("Spectral OT precondition: union-graph connectivity vs $k$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_dir = PROJ / "surf_latex" / "final_report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "connectivity_diagnostic.png"
    pdf = out_dir / "connectivity_diagnostic.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")

    # Also dump CSV for the appendix
    csv_path = out_dir / "connectivity_diagnostic.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,metric,n_marginals,n_total," +
                ",".join(f"k={k}" for k in K_VALUES) + "\n")
        for name, vals in results.items():
            f.write(f"{name},{vals['metric']},{len(vals['comps'])},{vals['n']}," +
                    ",".join(str(c) for c in vals["comps"]) + "\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
